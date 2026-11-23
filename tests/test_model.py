"""ResNet-18 construction: what is trainable, and what would be downloaded."""

from __future__ import annotations

import pytest
import torch
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18 as real_resnet18

from image_classifier_lab.model import (
    LearningMode,
    architecture_parameter_counts,
    backbone_is_frozen,
    count_parameters,
    create_model,
    resolve_weights,
)


def test_weight_defaults_match_the_mode() -> None:
    assert resolve_weights(LearningMode.TRANSFER) is ResNet18_Weights.IMAGENET1K_V1
    assert resolve_weights(LearningMode.NOVEL) is None
    assert resolve_weights(LearningMode.TRANSFER, weights=None) is None


def test_constructor_receives_those_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def spy(*, weights: object = None, **kwargs: object) -> torch.nn.Module:
        captured["weights"] = weights
        return real_resnet18(weights=None)

    monkeypatch.setattr("image_classifier_lab.model.resnet18", spy)
    transfer = create_model(2, LearningMode.TRANSFER)
    assert captured["weights"] is ResNet18_Weights.IMAGENET1K_V1
    assert transfer.fc.out_features == 2
    novel = create_model(4, LearningMode.NOVEL)
    assert captured["weights"] is None
    assert novel.fc.out_features == 4


def test_trainable_counts_differ_by_mode() -> None:
    transfer = create_model(5, LearningMode.TRANSFER, weights=None)
    novel = create_model(5, LearningMode.NOVEL, weights=None)
    transfer_trainable, transfer_total = count_parameters(transfer)
    novel_trainable, novel_total = count_parameters(novel)
    final_layer = sum(parameter.numel() for parameter in transfer.fc.parameters())

    assert transfer.fc.in_features == 512
    assert transfer_trainable == final_layer
    assert final_layer == 512 * 5 + 5
    assert novel_trainable == novel_total == transfer_total
    assert transfer_trainable < novel_trainable
    assert backbone_is_frozen(transfer)
    assert not backbone_is_frozen(novel)
    assert all(parameter.requires_grad for parameter in novel.parameters())
    frozen = [
        parameter for name, parameter in transfer.named_parameters() if not name.startswith("fc.")
    ]
    assert frozen
    assert all(not parameter.requires_grad for parameter in frozen)


def test_parameter_count_helper_does_not_touch_the_rng() -> None:
    architecture_parameter_counts.cache_clear()
    torch.manual_seed(0)
    expected = torch.rand(4)
    torch.manual_seed(0)
    novel = architecture_parameter_counts(3, LearningMode.NOVEL.value)
    transfer = architecture_parameter_counts(3, LearningMode.TRANSFER.value)
    assert novel[0] == novel[1]
    assert transfer[0] < transfer[1]
    assert novel[1] == transfer[1]
    assert architecture_parameter_counts(3, LearningMode.NOVEL.value) == novel
    assert torch.equal(torch.rand(4), expected)


def test_create_model_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        create_model(1, LearningMode.NOVEL, weights=None)
