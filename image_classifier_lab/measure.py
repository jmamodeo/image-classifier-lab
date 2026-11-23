"""Multiply-adds and memory parts, counted from a real forward pass.

The numbers come from the module that will be trained: kernel sizes, channel
counts, and parameter tensors. They are not copied from a benchmark table.

A multiply-add is one multiplication of a weight by an input. FLOPs, in this
module, means twice that, because each multiply-add is one multiplication and
one addition. Bias additions, BatchNorm, ReLU, and pooling are not included
in the multiply-add count; they are small next to the convolutions, and the
notes say they were left out.

Memory is split into parts that can be counted exactly (parameters, gradients,
Adam moments, buffers) and a parts list of forward-output tensors. That list
sums tensors that are not all stored at the same time, so it is not a
peak-memory reading.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
from torch import nn

from image_classifier_lab.data import IMAGE_SIZE
from image_classifier_lab.model import LearningMode, architecture_parameter_counts, create_model
from image_classifier_lab.training import EpochMetrics, TrainResult

FLOAT32_BYTES = 4

MEASURE_NOTES: tuple[str, ...] = (
    "Multiply-adds count Conv2d and Linear only. Each output element of a convolution "
    "is a dot product of kernel height × kernel width × input channels per group. A "
    "linear layer contributes inputs × outputs for every image in the batch. Bias "
    "additions, BatchNorm, ReLU, and pooling are not in this count.",
    "FLOPs here means twice the multiply-adds: one multiplication and one addition. "
    "Some write-ups call a multiply-add itself one FLOP. This page uses the first "
    "convention and shows both numbers.",
    "The forward-output total adds the tensor each leaf layer returned on one forward "
    "pass. Those tensors are not all stored at the same instant, so the total is a "
    "parts list, not a peak-memory reading. The backward pass keeps extra tensors. "
    "Transfer mode needs fewer of them, because the frozen backbone receives no "
    "gradients.",
    "Parameter, gradient, and Adam bytes do not grow with the batch. Forward outputs "
    "do, one copy per image. Adam's two moment buffers are counted as float32, two "
    "per trainable parameter, which is what PyTorch's Adam stores. BatchNorm running "
    "statistics are in the buffer bytes, not in the Adam bytes.",
    "The counted multiply-adds are one forward pass. A training step also runs a "
    "backward pass, which does more arithmetic than the forward pass. Wall-clock "
    "seconds elsewhere on the page are measured; this count is not a timer.",
)

_GROUP_TITLES: dict[str, str] = {
    "conv1": "First convolution",
    "bn1": "First batch norm",
    "relu": "ReLU",
    "maxpool": "Max pool",
    "layer1": "Stage 1",
    "layer2": "Stage 2",
    "layer3": "Stage 3",
    "layer4": "Stage 4",
    "avgpool": "Average pool",
    "fc": "Final linear layer",
}


@dataclass(frozen=True)
class Analysis:
    """Costs for one forward pass of one model, at one batch size."""

    batch: int
    macs: int
    flops: int
    macs_conv: int
    macs_linear: int
    macs_by_group: tuple[tuple[str, int], ...]
    parameter_bytes: int
    trainable_parameters: int
    total_parameters: int
    gradient_bytes: int
    adam_bytes: int
    buffer_bytes: int
    output_bytes_by_kind: tuple[tuple[str, int], ...]
    output_bytes_total: int
    group_parameters: tuple[tuple[str, int], ...]
    group_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    stem_kernel: tuple[int, int] | None
    stem_stride: tuple[int, int] | None
    stem_padding: tuple[int, int] | None
    stem_in_channels: int | None
    stem_out_channels: int | None
    final_in_features: int | None
    final_out_features: int | None


@dataclass(frozen=True)
class TimingSummary:
    """Wall-clock summary of the training passes in one finished run."""

    mean_train_seconds: float
    mean_val_seconds: float
    images_per_second: float | None
    seconds_per_step: float | None
    train_images_per_epoch: int
    steps_per_epoch: int
    peak_memory_bytes: int | None


@dataclass(frozen=True)
class RunRecord:
    """One finished run, kept so a later exercise can compare settings."""

    mode: str
    learning_rate: float
    batch_size: int
    epochs: int
    augment: bool
    device_label: str
    seed: int
    final_train_loss: float
    final_val_loss: float
    final_val_accuracy: float
    mean_train_seconds: float
    mean_val_seconds: float
    images_per_second: float | None
    trainable: int
    total: int
    peak_memory_bytes: int | None


def group_label(name: str, shape: tuple[int, ...] | None = None) -> str:
    """A short label, with the measured output shape when we have one."""
    title = _GROUP_TITLES.get(name, name)
    if not shape:
        return title
    if len(shape) >= 3:
        channels, height, width = shape[-3], shape[-2], shape[-1]
        return f"{title} · {height}×{width} · {channels} channels"
    if len(shape) == 1:
        return f"{title} · {shape[0]} outputs"
    return title


def format_count(value: int) -> str:
    """Thousands separators, and a short form once the number reaches millions."""
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} billion ({value:,})"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} million ({value:,})"
    return f"{value:,}"


def format_bytes(value: int) -> str:
    """Binary units plus the exact byte count."""
    exact = f"{value:,} bytes"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB ({exact})"
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB ({exact})"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB ({exact})"
    return exact


def adam_bytes(trainable_parameters: int) -> int:
    """Two float32 moment buffers per trainable parameter, as in PyTorch's Adam."""
    if (
        isinstance(trainable_parameters, bool)
        or not isinstance(trainable_parameters, int)
        or trainable_parameters < 0
    ):
        raise ValueError("trainable_parameters must be a non-negative integer.")
    return 2 * trainable_parameters * FLOAT32_BYTES


def adam_step_defaults() -> tuple[float, float, float]:
    """``(beta1, beta2, weight_decay)`` from a stock :class:`torch.optim.Adam`."""
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter])
    beta1, beta2 = optimizer.defaults["betas"]
    return float(beta1), float(beta2), float(optimizer.defaults["weight_decay"])


def _pair(value: int | tuple[int, ...]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def analyze_forward(model: nn.Module, example: torch.Tensor) -> Analysis:
    """Count one forward pass of ``example`` through ``model``.

    ``example`` is NCHW. The model is left in the training mode it started in,
    and its weights are not changed. Hooks are removed before this returns.
    """
    if example.ndim != 4 or int(example.shape[0]) < 1:
        raise ValueError("example must be a non-empty NCHW batch.")
    batch = int(example.shape[0])
    macs_conv = 0
    macs_linear = 0
    macs_by_group: dict[str, int] = {}
    output_by_kind: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _group(name: str) -> str:
        return name.split(".", 1)[0]

    for name, module in model.named_modules():
        if name == "":
            continue
        if "." not in name:

            def _shape_hook(
                _module: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                key: str = name,
            ) -> None:
                if torch.is_tensor(output) and output.ndim >= 1:
                    shapes[key] = tuple(int(value) for value in output.shape[1:])

            hooks.append(module.register_forward_hook(_shape_hook))

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            def _mac_hook(
                layer: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                key: str = _group(name),
            ) -> None:
                nonlocal macs_conv, macs_linear
                if not torch.is_tensor(output):
                    return
                if isinstance(layer, nn.Conv2d):
                    height, width = _pair(layer.kernel_size)
                    counted = (
                        int(output.numel()) * height * width * (layer.in_channels // layer.groups)
                    )
                    macs_conv += counted
                elif isinstance(layer, nn.Linear):
                    counted = int(output.numel()) * int(layer.in_features)
                    macs_linear += counted
                else:
                    return
                macs_by_group[key] = macs_by_group.get(key, 0) + counted

            hooks.append(module.register_forward_hook(_mac_hook))

        if not any(module.children()):

            def _output_hook(
                layer: nn.Module,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                kind: str = type(module).__name__,
            ) -> None:
                if torch.is_tensor(output):
                    size = int(output.numel()) * int(output.element_size())
                    output_by_kind[kind] = output_by_kind.get(kind, 0) + size

            hooks.append(module.register_forward_hook(_output_hook))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(example)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    parameters = list(model.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    gradient_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
        if parameter.requires_grad
    )
    buffers = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    grouped: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        key = _group(name)
        grouped[key] = grouped.get(key, 0) + parameter.numel()

    stem = next((module for module in model.modules() if isinstance(module, nn.Conv2d)), None)
    final = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            final = module

    output_total = sum(output_by_kind.values())
    macs = macs_conv + macs_linear
    return Analysis(
        batch=batch,
        macs=macs,
        flops=2 * macs,
        macs_conv=macs_conv,
        macs_linear=macs_linear,
        macs_by_group=tuple(macs_by_group.items()),
        parameter_bytes=parameter_bytes,
        trainable_parameters=trainable,
        total_parameters=total,
        gradient_bytes=gradient_bytes,
        adam_bytes=adam_bytes(trainable),
        buffer_bytes=buffers,
        output_bytes_by_kind=tuple(output_by_kind.items()),
        output_bytes_total=output_total,
        group_parameters=tuple(grouped.items()),
        group_shapes=tuple(shapes.items()),
        stem_kernel=_pair(stem.kernel_size) if stem is not None else None,
        stem_stride=_pair(stem.stride) if stem is not None else None,
        stem_padding=_pair(stem.padding) if stem is not None else None,
        stem_in_channels=int(stem.in_channels) if stem is not None else None,
        stem_out_channels=int(stem.out_channels) if stem is not None else None,
        final_in_features=int(final.in_features) if final is not None else None,
        final_out_features=int(final.out_features) if final is not None else None,
    )


@functools.lru_cache(maxsize=8)
def resnet18_structure(num_classes: int) -> Analysis:
    """Forward cost of this app's ResNet-18 for one image, without downloading weights.

    The global RNG is restored. Repeated calls are cached. Trainable counts on
    the returned object are for a fully trainable network; use
    :func:`analysis_for_mode` when the sidebar is in transfer mode.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")
    state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        model = create_model(num_classes, LearningMode.NOVEL, weights=None)
        example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        return analyze_forward(model, example)
    finally:
        torch.set_rng_state(state)


def analysis_for_mode(num_classes: int, mode: LearningMode, batch: int) -> Analysis:
    """Structure of the ResNet, with trainable counts and batch size filled in.

    Multiply-adds and forward outputs scale with ``batch``. Parameter bytes do
    not. Gradient and Adam bytes follow the trainable count of ``mode``.
    """
    if not isinstance(mode, LearningMode):
        raise ValueError("mode must be a LearningMode.")
    scaled = with_batch(resnet18_structure(num_classes), batch)
    trainable, total = architecture_parameter_counts(num_classes, mode.value)
    if total != scaled.total_parameters:
        raise RuntimeError("parameter counts from the two builders disagree.")
    element = scaled.parameter_bytes // total if total else FLOAT32_BYTES
    return replace(
        scaled,
        trainable_parameters=trainable,
        gradient_bytes=trainable * element,
        adam_bytes=adam_bytes(trainable),
    )


def with_batch(analysis: Analysis, batch: int) -> Analysis:
    """Scale a single-image analysis up to ``batch`` images.

    Output shapes of these layers grow in exact proportion to the batch, and
    so do the multiply-adds. Parameters do not.
    """
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer.")
    if analysis.batch != 1:
        raise ValueError("with_batch expects an analysis of a single image.")
    if batch == 1:
        return analysis
    return replace(
        analysis,
        batch=batch,
        macs=analysis.macs * batch,
        flops=analysis.flops * batch,
        macs_conv=analysis.macs_conv * batch,
        macs_linear=analysis.macs_linear * batch,
        macs_by_group=tuple((name, count * batch) for name, count in analysis.macs_by_group),
        output_bytes_by_kind=tuple(
            (name, count * batch) for name, count in analysis.output_bytes_by_kind
        ),
        output_bytes_total=analysis.output_bytes_total * batch,
    )


def summarize_timing(history: Sequence[EpochMetrics]) -> TimingSummary:
    """Images per second from the training passes only, not the validation passes."""
    if not history:
        raise ValueError("history is empty.")
    train_seconds = sum(item.train_seconds for item in history)
    val_seconds = sum(item.val_seconds for item in history)
    images = sum(item.train_images for item in history)
    steps = sum(item.train_steps for item in history)
    peaks = [item.peak_memory_bytes for item in history if item.peak_memory_bytes]
    images_per_second = images / train_seconds if train_seconds > 0 and images > 0 else None
    seconds_per_step = train_seconds / steps if train_seconds > 0 and steps > 0 else None
    return TimingSummary(
        mean_train_seconds=train_seconds / len(history),
        mean_val_seconds=val_seconds / len(history),
        images_per_second=images_per_second,
        seconds_per_step=seconds_per_step,
        train_images_per_epoch=history[-1].train_images,
        steps_per_epoch=history[-1].train_steps,
        peak_memory_bytes=max(peaks) if peaks else None,
    )


def make_run_record(
    result: TrainResult,
    *,
    mode: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    augment: bool,
    device_label: str,
    seed: int,
) -> RunRecord:
    """Fold a finished run into one row of the recent-runs table."""
    timing = summarize_timing(result.history)
    last = result.history[-1]
    return RunRecord(
        mode=mode,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        augment=augment,
        device_label=device_label,
        seed=seed,
        final_train_loss=last.train_loss,
        final_val_loss=last.val_loss,
        final_val_accuracy=last.val_accuracy,
        mean_train_seconds=timing.mean_train_seconds,
        mean_val_seconds=timing.mean_val_seconds,
        images_per_second=timing.images_per_second,
        trainable=result.trainable_parameters,
        total=result.total_parameters,
        peak_memory_bytes=timing.peak_memory_bytes,
    )
