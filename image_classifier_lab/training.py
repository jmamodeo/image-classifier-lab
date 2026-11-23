"""The training loop: mean loss, held-out accuracy, and a final confusion matrix.

Loss recorded for an epoch is the total cross-entropy divided by the number of
images actually scored. That stays a per-image mean when batch sizes differ,
and it does not grow just because the dataset was split into more batches.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights

from image_classifier_lab.data import SplitData, build_datasets
from image_classifier_lab.errors import TrainingError
from image_classifier_lab.model import (
    AUTO,
    LearningMode,
    _AutoWeights,
    backbone_is_frozen,
    count_parameters,
    create_model,
    uses_batchnorm,
)

MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 64
LEARNING_RATES: tuple[float, ...] = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03)
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 5
MAX_EPOCHS = 30
DEFAULT_VAL_PERCENT = 20
DEFAULT_SEED = 0
DEFAULT_SHUFFLE = True
DEFAULT_AUGMENT = True


@dataclass(frozen=True)
class EpochMetrics:
    """Mean loss and accuracy for one completed epoch, on both splits."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    train_seconds: float = 0.0
    val_seconds: float = 0.0
    train_images: int = 0
    train_steps: int = 0
    peak_memory_bytes: int | None = None


@dataclass(frozen=True)
class TrainResult:
    """What a finished run leaves behind for the charts and the summary."""

    history: tuple[EpochMetrics, ...]
    confusion: tuple[tuple[int, ...], ...]
    per_class_accuracy: tuple[float, ...]
    trainable_parameters: int
    total_parameters: int


@dataclass(frozen=True)
class BatchPlan:
    """How the training loader should batch, plus a note for the screen."""

    batch_size: int
    drop_last: bool
    note: str | None


TRAINING_DEVICE_LABELS = {"cpu": "CPU", "cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (Metal/MPS)"}


def available_devices(
    *,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Device labels and torch device strings this installation can use."""
    devices = ["cpu"]
    if (cuda_available or torch.cuda.is_available)():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    has_mps = (
        mps_available() if mps_available is not None else mps is not None and mps.is_available()
    )
    if has_mps:
        devices.append("mps")
    return tuple((TRAINING_DEVICE_LABELS[device], device) for device in devices)


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch before the network is created.

    The split, the augmentation generator, and the shuffle generator each
    carry their own seed as well. Call this immediately before
    :func:`image_classifier_lab.model.create_model` so the random initial weights repeat.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TrainingError("Seed must be a non-negative integer.")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_batching(num_images: int, batch_size: int, *, batchnorm: bool) -> BatchPlan:
    """Choose a batch size the network can actually train with.

    ResNet-18's BatchNorm cannot learn from a batch of one image: the batch
    variance is zero and the loss becomes NaN. When the last batch would be a
    single image, that image is left out of the epoch. Models without
    BatchNorm, including the tiny ones in the tests, keep every image.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise TrainingError("Batch size must be a positive integer.")
    if num_images < 1:
        raise TrainingError("There are no training images.")
    note: str | None = None
    drop_last = False
    if batchnorm:
        if batch_size < MIN_BATCH_SIZE or num_images < MIN_BATCH_SIZE:
            raise TrainingError(
                "Batch size must be at least 2, and the training set needs at least 2 images. "
                "ResNet-18 uses BatchNorm, which cannot learn from a single image."
            )
        if num_images % batch_size == 1:
            drop_last = True
            note = (
                "One training image is left out of each epoch so BatchNorm never sees a "
                "batch of one."
            )
    if batch_size > num_images and note is None:
        note = (
            f"Batch size ({batch_size}) is larger than the {num_images} training images, "
            "so each epoch is a single update."
        )
    return BatchPlan(batch_size=batch_size, drop_last=drop_last, note=note)


def per_class_accuracy(confusion: Sequence[Sequence[int]]) -> tuple[float, ...]:
    """Fraction correct within each true class. Empty rows are NaN."""
    scores: list[float] = []
    for index, row in enumerate(confusion):
        total = sum(row)
        scores.append(row[index] / total if total else float("nan"))
    return tuple(scores)


def build_confusion(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
) -> tuple[tuple[int, ...], ...]:
    """Rows are the true class, columns are the predicted class."""
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        matrix[truth][prediction] += 1
    return tuple(tuple(row) for row in matrix)


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Replace BatchNorm running stats with a pass over ``loader``.

    The default momentum barely moves those averages on a few dozen images, so
    evaluation then uses a different normalization from the one training just
    used, and the validation loss explodes. A cumulative average over one pass
    leaves the stored stats on the same scale as the training batches. Momentum
    is restored afterward. Weights are not changed.
    """
    saved: list[tuple[nn.Module, float | None]] = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            saved.append((module, module.momentum))
            module.reset_running_stats()
            module.momentum = None
    if not saved:
        return
    model.train()
    for images, _labels in loader:
        model(images.to(device))
    for module, momentum in saved:
        module.momentum = momentum


@dataclass(frozen=True)
class _PassResult:
    """One train or validation pass, including the wall-clock time around it."""

    loss: float
    accuracy: float
    truths: tuple[int, ...]
    predictions: tuple[int, ...]
    seconds: float
    images: int
    steps: int


def _peak_allocated(device: torch.device) -> int | None:
    """CUDA allocator peak since the last reset, or ``None`` when this is not CUDA.

    PyTorch reports that peak for CUDA. CPU and MPS runs leave the field empty
    rather than substituting a guess.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _freeze_batchnorm(model: nn.Module) -> None:
    """Keep every BatchNorm layer in eval while the rest of the net trains.

    In transfer mode the backbone is frozen, but ``model.train()`` would still
    update BatchNorm running statistics. Leaving those layers in eval keeps
    the ImageNet statistics in place, which is what "frozen backbone" means.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def _run_pass(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    update: bool,
    freeze_bn: bool,
    num_classes: int,
) -> _PassResult:
    started = time.perf_counter()
    if update:
        model.train()
        if freeze_bn:
            _freeze_batchnorm(model)
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    seen = 0
    steps = 0
    truths: list[int] = []
    predictions: list[int] = []
    context = torch.enable_grad() if update else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long)
            if labels.numel() == 0:
                continue
            if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
                raise TrainingError("A label is outside the range of classes.")
            if update:
                if optimizer is None:
                    raise TrainingError("Training requires an optimizer.")
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not math.isfinite(float(loss.item())):
                raise TrainingError(
                    "The loss became infinite or NaN. Lower the learning rate and train again."
                )
            predicted = logits.argmax(dim=1)
            if update and optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_count = int(labels.shape[0])
            total_loss += float(loss.item())
            correct += int((predicted == labels).sum().item())
            seen += batch_count
            steps += 1
            if not update:
                truths.extend(int(value) for value in labels.detach().cpu().tolist())
                predictions.extend(int(value) for value in predicted.detach().cpu().tolist())
    if seen == 0:
        raise TrainingError(
            "There were no images to score. Add more images or lower the batch size."
        )
    return _PassResult(
        loss=total_loss / seen,
        accuracy=correct / seen,
        truths=tuple(truths),
        predictions=tuple(predictions),
        seconds=time.perf_counter() - started,
        images=seen,
        steps=steps,
    )


def fit(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device | str,
    num_classes: int,
    on_epoch: Callable[[tuple[EpochMetrics, ...]], None] | None = None,
) -> TrainResult:
    """Train ``model`` and score the validation set after every epoch.

    The optimizer is Adam over whatever currently requires gradients, so the
    same loop serves a frozen backbone and a fully trainable network. Shuffle
    uses its own generator seeded with ``seed`` and does not consume the
    global RNG that initialized the weights.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise TrainingError("Epochs must be a positive integer.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
    ):
        raise TrainingError("Learning rate must be a positive finite number.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise TrainingError("num_classes must be an integer of at least 2.")
    if len(val_dataset) < 1:
        raise TrainingError("The validation set is empty.")

    device_t = torch.device(device)
    model.to(device_t)
    batching = resolve_batching(
        len(train_dataset),
        batch_size,
        batchnorm=uses_batchnorm(model),
    )
    trainable, total = count_parameters(model)
    if trainable == 0:
        raise TrainingError("This model has no trainable parameters.")

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batching.batch_size,
        shuffle=shuffle,
        drop_last=batching.drop_last,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batching.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(learning_rate),
    )
    freeze_bn = backbone_is_frozen(model) and uses_batchnorm(model)
    history: list[EpochMetrics] = []
    y_true: list[int] = []
    y_pred: list[int] = []

    for epoch in range(1, epochs + 1):
        if device_t.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_t)
        trained = _run_pass(
            model,
            train_loader,
            criterion,
            optimizer,
            device_t,
            update=True,
            freeze_bn=freeze_bn,
            num_classes=num_classes,
        )
        if uses_batchnorm(model) and not freeze_bn:
            recalibrate_batchnorm(model, train_loader, device_t)
        # Snapshot before validation so a CUDA peak is the training side of the
        # epoch (the update, plus novel-mode BatchNorm refresh), not the holdout.
        peak_memory = _peak_allocated(device_t)
        validated = _run_pass(
            model,
            val_loader,
            criterion,
            None,
            device_t,
            update=False,
            freeze_bn=False,
            num_classes=num_classes,
        )
        y_true = list(validated.truths)
        y_pred = list(validated.predictions)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=trained.loss,
                train_accuracy=trained.accuracy,
                val_loss=validated.loss,
                val_accuracy=validated.accuracy,
                train_seconds=trained.seconds,
                val_seconds=validated.seconds,
                train_images=trained.images,
                train_steps=trained.steps,
                peak_memory_bytes=peak_memory,
            )
        )
        if on_epoch is not None:
            on_epoch(tuple(history))

    model.eval()
    confusion = build_confusion(y_true, y_pred, num_classes)
    return TrainResult(
        history=tuple(history),
        confusion=confusion,
        per_class_accuracy=per_class_accuracy(confusion),
        trainable_parameters=trainable,
        total_parameters=total,
    )


def train_classifier(
    split: SplitData,
    *,
    mode: LearningMode,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    shuffle: bool,
    augment: bool,
    device: str,
    weights: ResNet18_Weights | None | _AutoWeights = AUTO,
    on_epoch: Callable[[tuple[EpochMetrics, ...]], None] | None = None,
) -> tuple[nn.Module, TrainResult]:
    """Seed, build, and train. This is the path the app uses.

    ``set_seed`` runs before the network is constructed so novel-mode
    initialization and the new final layer repeat. Augmentation and shuffle
    then use their own generators, also seeded with ``seed``.
    """
    set_seed(seed)
    model = create_model(len(split.class_names), mode, weights=weights)
    train_dataset, val_dataset = build_datasets(split, augment=augment, seed=seed)
    result = fit(
        model,
        train_dataset,
        val_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        device=device,
        num_classes=len(split.class_names),
        on_epoch=on_epoch,
    )
    return model, result


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    image_batch: torch.Tensor,
    device: torch.device | str,
) -> list[float]:
    """Softmax scores for one image. The list sums to 1 and follows class order."""
    if image_batch.ndim != 4 or image_batch.shape[0] != 1:
        raise ValueError("expected a batch of one image.")
    device_t = torch.device(device)
    model.eval()
    model.to(device_t)
    logits = model(image_batch.to(device_t))
    probabilities = torch.softmax(logits, dim=1).squeeze(0)
    return [float(value) for value in probabilities.detach().cpu().tolist()]
