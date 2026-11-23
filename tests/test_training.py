"""Training metrics, which weights move, and prediction."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset

from image_classifier_lab.errors import TrainingError
from image_classifier_lab.model import LearningMode, create_model
from image_classifier_lab.training import (
    BatchPlan,
    available_devices,
    fit,
    per_class_accuracy,
    predict_probabilities,
    resolve_batching,
    set_seed,
)


class ConstantLogits(nn.Module):
    """A classifier whose outputs ignore the pixels, so the loss is known exactly.

    ``anchor - anchor`` keeps a gradient path (it is zero, not missing) without
    letting the optimizer change what the forward pass returns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        base = images.new_tensor([0.4, -0.2])
        shift = self.anchor - self.anchor
        return base.view(1, 2).expand(images.shape[0], 2) + shift


def test_epoch_loss_is_the_mean_over_images_not_the_sum_of_batches() -> None:
    images = torch.zeros(3, 3, 8, 8)
    labels = torch.tensor([0, 0, 1])
    dataset = TensorDataset(images, labels)
    result = fit(
        ConstantLogits(),
        dataset,
        dataset,
        epochs=1,
        learning_rate=0.01,
        batch_size=2,
        seed=0,
        shuffle=False,
        device="cpu",
        num_classes=2,
    )
    logits = torch.tensor([[0.4, -0.2]])
    loss_zero = F.cross_entropy(logits, torch.tensor([0])).item()
    loss_one = F.cross_entropy(logits, torch.tensor([1])).item()
    expected = (loss_zero + loss_zero + loss_one) / 3
    metrics = result.history[0]
    assert metrics.train_loss == pytest.approx(expected)
    assert metrics.val_loss == pytest.approx(expected)
    assert metrics.train_accuracy == pytest.approx(2 / 3)
    assert metrics.epoch == 1
    assert metrics.train_images == 3
    assert metrics.train_steps == 2
    assert metrics.train_seconds > 0
    assert metrics.val_seconds > 0
    assert metrics.peak_memory_bytes is None
    assert result.confusion == ((2, 0), (1, 0))
    assert result.per_class_accuracy == pytest.approx((1.0, 0.0))
    assert sum(sum(row) for row in result.confusion) == 3


def test_transfer_step_updates_only_the_final_layer_and_keeps_every_validation_image() -> None:
    torch.manual_seed(0)
    model = create_model(2, LearningMode.TRANSFER, weights=None)
    before_fc = model.fc.weight.detach().clone()
    before_conv = model.conv1.weight.detach().clone()
    before_bn = model.bn1.running_mean.detach().clone()
    train = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    # Three validation images and a batch size of 2: the leftover image must still be scored.
    validation = TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 0]))
    result = fit(
        model,
        train,
        validation,
        epochs=1,
        learning_rate=1e-2,
        batch_size=2,
        seed=0,
        shuffle=False,
        device="cpu",
        num_classes=2,
    )
    assert not torch.equal(model.fc.weight, before_fc)
    assert torch.equal(model.conv1.weight, before_conv)
    assert torch.equal(model.bn1.running_mean, before_bn)
    assert sum(sum(row) for row in result.confusion) == 3


def test_novel_batchnorm_stats_are_refreshed_from_the_training_images() -> None:
    """A constant input would otherwise leave the running mean near zero."""

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.fc = nn.Linear(1, 2)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            pooled = self.bn(images).mean(dim=(2, 3))
            return self.fc(pooled)

    images = torch.full((4, 1, 8, 8), 5.0)
    labels = torch.tensor([0, 1, 0, 1])
    dataset = TensorDataset(images, labels)
    model = Net()
    result = fit(
        model,
        dataset,
        dataset,
        epochs=1,
        learning_rate=1e-3,
        batch_size=4,
        seed=0,
        shuffle=False,
        device="cpu",
        num_classes=2,
    )
    assert torch.allclose(model.bn.running_mean, torch.tensor([5.0]), atol=1e-4)
    assert math.isfinite(result.history[0].val_loss)


def test_novel_step_updates_the_backbone_and_batchnorm() -> None:
    torch.manual_seed(0)
    model = create_model(2, LearningMode.NOVEL, weights=None)
    before_conv = model.conv1.weight.detach().clone()
    before_bn = model.bn1.running_mean.detach().clone()
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    fit(
        model,
        dataset,
        dataset,
        epochs=1,
        learning_rate=1e-2,
        batch_size=2,
        seed=0,
        shuffle=False,
        device="cpu",
        num_classes=2,
    )
    assert not torch.equal(model.conv1.weight, before_conv)
    assert not torch.equal(model.bn1.running_mean, before_bn)


def test_same_seed_rebuilds_the_same_first_epoch() -> None:
    def once(seed: int) -> float:
        set_seed(seed)
        model = create_model(2, LearningMode.NOVEL, weights=None)
        images = torch.linspace(-1, 1, 4 * 3 * 32 * 32).reshape(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        dataset = TensorDataset(images, labels)
        result = fit(
            model,
            dataset,
            dataset,
            epochs=1,
            learning_rate=1e-3,
            batch_size=2,
            seed=seed,
            shuffle=True,
            device="cpu",
            num_classes=2,
        )
        return result.history[0].train_loss

    assert once(11) == once(11)


def test_seed_fixes_novel_initialization() -> None:
    set_seed(1)
    first = create_model(2, LearningMode.NOVEL, weights=None).fc.weight.detach().clone()
    set_seed(1)
    second = create_model(2, LearningMode.NOVEL, weights=None).fc.weight.detach().clone()
    set_seed(2)
    third = create_model(2, LearningMode.NOVEL, weights=None).fc.weight.detach().clone()
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_prediction_is_a_probability_distribution() -> None:
    model = create_model(3, LearningMode.NOVEL, weights=None)
    probabilities = predict_probabilities(model, torch.rand(1, 3, 32, 32), "cpu")
    assert len(probabilities) == 3
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-5)
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_batching_rules_for_batchnorm() -> None:
    dropped = resolve_batching(5, 4, batchnorm=True)
    assert dropped == BatchPlan(
        batch_size=4,
        drop_last=True,
        note=dropped.note,
    )
    assert dropped.note is not None
    assert "batch of one" in dropped.note
    plain = resolve_batching(5, 4, batchnorm=False)
    assert plain.drop_last is False
    large = resolve_batching(4, 8, batchnorm=True)
    assert large.drop_last is False
    assert large.note is not None
    assert "single update" in large.note
    with pytest.raises(TrainingError, match="at least 2") as caught:
        resolve_batching(4, 1, batchnorm=True)
    assert "BatchNorm" in str(caught.value)


def test_nan_loss_stops_the_run() -> None:
    class Boom(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(1, 2)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.fc(images.new_ones(images.shape[0], 1)) * float("nan")

    dataset = TensorDataset(torch.zeros(2, 1, 1, 1), torch.tensor([0, 1]))
    with pytest.raises(TrainingError, match="NaN"):
        fit(
            Boom(),
            dataset,
            dataset,
            epochs=1,
            learning_rate=0.1,
            batch_size=2,
            seed=0,
            shuffle=False,
            device="cpu",
            num_classes=2,
        )


def test_cpu_is_always_offered() -> None:
    devices = available_devices()
    assert devices[0] == ("CPU", "cpu")
    assert all(len(item) == 2 for item in devices)


def test_per_class_accuracy_handles_an_empty_row() -> None:
    accuracy = per_class_accuracy(((1, 1), (0, 0)))
    assert accuracy[0] == pytest.approx(0.5)
    assert math.isnan(accuracy[1])
