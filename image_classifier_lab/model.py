"""ResNet-18 construction for transfer learning and training from scratch.

Transfer mode loads ImageNet weights, freezes the backbone, and replaces the
final layer with a new trainable classifier. Novel mode keeps the same
architecture, leaves every parameter trainable, and starts from the library's
random initialization. Tests pass ``weights=None`` so neither path downloads
anything.
"""

from __future__ import annotations

import functools
from enum import Enum

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class LearningMode(Enum):
    """Which weights exist at the start of training, and which ones may move."""

    TRANSFER = "transfer"
    NOVEL = "novel"

    @property
    def title(self) -> str:
        if self is LearningMode.TRANSFER:
            return "Transfer"
        return "Novel"


class _AutoWeights:
    """Sentinel: pick the weights that match the learning mode."""


AUTO = _AutoWeights()


def resolve_weights(
    mode: LearningMode,
    weights: ResNet18_Weights | None | _AutoWeights = AUTO,
) -> ResNet18_Weights | None:
    """Return the torchvision weight enum ``resnet18`` should load.

    ``AUTO`` selects ImageNet-1K v1 for transfer mode (the same checkpoint the
    old ``pretrained=True`` flag loaded) and ``None`` for novel mode. Passing
    ``None`` explicitly skips the download in either mode, which is what the
    tests do. The freeze-or-train decision is separate and lives in
    :func:`create_model`.
    """
    if not isinstance(mode, LearningMode):
        raise ValueError("mode must be a LearningMode.")
    if weights is AUTO:
        if mode is LearningMode.TRANSFER:
            return ResNet18_Weights.IMAGENET1K_V1
        return None
    if weights is None or isinstance(weights, ResNet18_Weights):
        return weights
    raise TypeError("weights must be ResNet18_Weights, None, or AUTO.")


def create_model(
    num_classes: int,
    mode: LearningMode,
    weights: ResNet18_Weights | None | _AutoWeights = AUTO,
) -> nn.Module:
    """Build a ResNet-18 whose last layer has ``num_classes`` outputs.

    In transfer mode the backbone is frozen before the final layer is
    replaced, so the new ``fc`` is the only trainable module. In novel mode
    every parameter, including the new final layer, stays trainable. The new
    final layer is randomly initialized in both modes; transfer mode's head
    start is random too. Its advantage is the frozen feature extractor in
    front of that layer.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least 2.")
    resolved = resolve_weights(mode, weights)
    network = resnet18(weights=resolved, progress=False)
    in_features = int(network.fc.in_features)
    if mode is LearningMode.TRANSFER:
        # Freeze first, then replace the classifier. The replacement is created trainable.
        for parameter in network.parameters():
            parameter.requires_grad = False
    network.fc = nn.Linear(in_features, num_classes)
    if mode is LearningMode.NOVEL:
        for parameter in network.parameters():
            parameter.requires_grad = True
    return network


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def backbone_is_frozen(model: nn.Module) -> bool:
    """Whether every parameter outside the final ``fc`` layer is frozen."""
    for name, parameter in model.named_parameters():
        if not name.startswith("fc.") and parameter.requires_grad:
            return False
    return True


def uses_batchnorm(model: nn.Module) -> bool:
    """Whether training must keep batches at size 2 or more."""
    batchnorm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    return any(isinstance(module, batchnorm_types) for module in model.modules())


@functools.lru_cache(maxsize=32)
def architecture_parameter_counts(num_classes: int, mode_value: str) -> tuple[int, int]:
    """Trainable and total counts for a mode, without downloading weights.

    Building the network draws random numbers for the unused initialization.
    The global RNG is restored so asking for these counts cannot change a
    later training run. Repeated calls are cached.
    """
    state = torch.get_rng_state()
    try:
        model = create_model(num_classes, LearningMode(mode_value), weights=None)
        return count_parameters(model)
    finally:
        torch.set_rng_state(state)
