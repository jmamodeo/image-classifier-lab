"""Figures for the lessons.

Charts that claim a measurement take their numbers from the caller. Sketches
say so in the title. Filter pictures are scaled per filter so a small weight
and a large one are both visible; the caption in the lesson says that.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from torch import nn

from image_classifier_lab.demos import (
    VERTICAL_EDGE,
    convolve_at,
    convolve_valid,
    cross_entropy,
    descent_path,
    neuron,
    softmax,
    toy_loss,
)
from image_classifier_lab.model import LearningMode, create_model
from image_classifier_lab.plots import QUIET_COLOR, TRAIN_COLOR, VAL_COLOR

_EDGE = "#1F4E79"
_INK = "#17202A"
_PALE = "#E7EEF5"


def _new(*size: float) -> tuple[Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=size, dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    return figure, axis


def _quiet(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis="y", color="#E1E4E8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9)


def _panel() -> tuple[Figure, plt.Axes]:
    figure, axis = _new(9.2, 3.4)
    _quiet(axis)
    return figure, axis


def plot_channel_planes(image: Image.Image) -> Figure:
    """Red, green, and blue planes of one RGB image, values 0–255."""
    rgb = image.convert("RGB")
    channels = rgb.split()
    figure, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    titles = ("Red", "Green", "Blue")
    cmaps = ("Reds", "Greens", "Blues")
    for axis, channel, title, cmap in zip(axes, channels, titles, cmaps, strict=True):
        shown = axis.imshow(channel, cmap=cmap, vmin=0, vmax=255)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    return figure


def plot_number_grid(crop: Sequence[Sequence[tuple[int, int, int]]]) -> Figure:
    """Red-channel integers for a small crop, drawn in the cells."""
    if not crop or not crop[0]:
        raise ValueError("crop is empty.")
    rows = len(crop)
    cols = len(crop[0])
    red = [[pixel[0] for pixel in row] for row in crop]
    figure, axis = _new(5.2, 4.6)
    image = axis.imshow(red, cmap="Reds", vmin=0, vmax=255)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for row in range(rows):
        for col in range(cols):
            value = red[row][col]
            ink = "white" if value > 160 else _INK
            axis.text(col, row, str(value), ha="center", va="center", color=ink, fontsize=8)
    axis.set_xticks(range(cols))
    axis.set_yticks(range(rows))
    axis.set_title("Red channel, 0–255")
    axis.tick_params(labelsize=8)
    return figure


def plot_normalization(
    raw: tuple[int, int, int],
    unit: Sequence[float],
    normalized: Sequence[float],
) -> tuple[Figure, Figure]:
    """Two charts: 0–1 scaling, then ImageNet normalization, for one pixel."""
    names = ["Red", "Green", "Blue"]
    colors = ["#B23A48", "#2E7D4F", _EDGE]
    scaled, axis = _panel()
    axis.bar(names, list(unit), color=colors)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Value")
    axis.set_title(f"Divide by 255  ·  pixel ({raw[0]}, {raw[1]}, {raw[2]})")
    shifted, other = _panel()
    _quiet(other)
    other.axhline(0, color="#8AA0B4", linewidth=0.8)
    other.bar(names, list(normalized), color=colors)
    other.set_ylabel("Normalized value")
    other.set_title("Subtract the ImageNet mean, divide by its standard deviation")
    return scaled, shifted


def plot_batch_layout(image_size: int, channels: int = 3) -> Figure:
    """One image as stacked channels, and the batch shape this app actually uses."""
    figure, axis = _new(9.2, 3.6)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    swatches = ("#C46B2C", "#2E7D4F", _EDGE)
    names = ("Red", "Green", "Blue")
    for index in range(channels):
        origin = 0.6 + index * 0.38
        axis.add_patch(
            FancyBboxPatch(
                (origin, 0.9 + index * 0.28),
                3.1,
                3.3,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=swatches[index],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.92,
            )
        )
        axis.text(
            origin + 1.55,
            2.5 + index * 0.28,
            names[index],
            ha="center",
            va="center",
            color="white",
            fontsize=11,
        )
    axis.text(
        6.3,
        4.5,
        "What the eye sees",
        color=_INK,
        fontsize=11,
    )
    axis.text(
        6.3,
        3.7,
        f"height × width × {channels} colors",
        color=_INK,
        fontsize=11,
    )
    axis.text(6.3, 2.3, "What this app feeds the network", color=_INK, fontsize=11)
    axis.text(
        6.3,
        1.5,
        f"one image:  {channels} × {image_size} × {image_size}",
        color=_INK,
        fontsize=11,
    )
    axis.text(
        6.3,
        0.7,
        f"a batch:  N × {channels} × {image_size} × {image_size}",
        color=_INK,
        fontsize=11,
    )
    return figure


def plot_neuron(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
) -> Figure:
    """Each input's contribution, and the total after the bias."""
    parts, total = neuron(inputs, weights, bias)
    figure, axis = _panel()
    labels = [f"x{index + 1}·w{index + 1}" for index in range(len(parts))] + ["bias", "output"]
    values = [*parts, bias, total]
    colors = [TRAIN_COLOR] * len(parts) + [QUIET_COLOR, VAL_COLOR]
    axis.bar(labels, values, color=colors)
    axis.axhline(0, color="#8AA0B4", linewidth=0.8)
    axis.set_ylabel("Contribution")
    axis.set_title(f"Weighted sum + bias  =  {total:.2f}")
    return figure


def plot_softmax(scores: Sequence[float]) -> Figure:
    """Raw scores beside the shares they become."""
    shares = softmax(scores)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.3), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    labels = [f"Class {index + 1}" for index in range(len(scores))]
    for axis in axes:
        _quiet(axis)
    axes[0].bar(labels, list(scores), color=QUIET_COLOR)
    axes[0].axhline(0, color="#8AA0B4", linewidth=0.8)
    axes[0].set_title("Scores (can be negative)")
    axes[0].set_ylabel("Score")
    winner = max(range(len(shares)), key=lambda index: shares[index])
    colors = [TRAIN_COLOR if index == winner else QUIET_COLOR for index in range(len(shares))]
    axes[1].bar(labels, [share * 100 for share in shares], color=colors)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Softmax shares (add to 100%)")
    axes[1].set_ylabel("Share (%)")
    return figure


def plot_cross_entropy() -> Figure:
    """Negative log of the correct class's share, for a single image."""
    shares = [index / 200 for index in range(2, 201)]
    losses = [cross_entropy(share) for share in shares]
    figure, axis = _panel()
    axis.plot(shares, losses, color=TRAIN_COLOR, linewidth=2)
    for mark in (0.1, 0.5, 0.9):
        axis.scatter([mark], [cross_entropy(mark)], color=VAL_COLOR, zorder=3)
        axis.annotate(
            f"p={mark:.1f}\nloss={cross_entropy(mark):.2f}",
            (mark, cross_entropy(mark)),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color=_INK,
        )
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share given to the correct class")
    axis.set_ylabel("Loss  −log(share)")
    axis.set_title("One image. A sure miss costs more than a near miss.")
    return figure


def plot_descent(learning_rate: float, steps: int = 8) -> Figure:
    """Plain gradient descent on (w − 2)². Not Adam, and not the ResNet."""
    path = descent_path(-1.0, learning_rate, steps)
    grid = [index / 20 for index in range(-20, 121)]
    figure, axis = _panel()
    axis.plot(grid, [toy_loss(point) for point in grid], color=QUIET_COLOR, linewidth=2)
    axis.plot(path, [toy_loss(point) for point in path], color=TRAIN_COLOR, marker="o")
    axis.scatter([path[0]], [toy_loss(path[0])], color=VAL_COLOR, zorder=3)
    axis.set_xlabel("The one weight")
    axis.set_ylabel("Toy loss")
    axis.set_title(
        f"Step size {learning_rate:.2f} on (weight − 2)². A sketch of step size, not Adam."
    )
    return figure


def plot_convolution(
    image: Sequence[Sequence[float]],
    top: int,
    left: int,
    kernel: Sequence[Sequence[float]] = VERTICAL_EDGE,
) -> Figure:
    """A hand-built filter on a real luminance grid, and the window it is on."""
    response = convolve_valid(image, kernel)
    height = len(kernel)
    width = len(kernel[0])
    figure, axes = plt.subplots(1, 3, figsize=(9.4, 3.5), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].add_patch(
        Rectangle(
            (left - 0.5, top - 0.5),
            width,
            height,
            fill=False,
            edgecolor="#C46B2C",
            linewidth=1.5,
        )
    )
    axes[0].set_title("Luminance")
    axes[1].imshow(response, cmap="coolwarm")
    axes[1].scatter([left], [top], s=30, color=_INK)
    axes[1].set_title("Filter response")
    patch = [[image[top + row][left + col] for col in range(width)] for row in range(height)]
    text = "Window\n" + "\n".join("  ".join(f"{value:5.0f}" for value in row) for row in patch)
    text += "\n\nFilter\n" + "\n".join(
        "  ".join(f"{value:5.0f}" for value in row) for row in kernel
    )
    dot = convolve_at(image, kernel, top, left)
    text += f"\n\nDot product\n{dot:.0f}"
    axes[2].axis("off")
    axes[2].text(0.0, 0.95, text, va="top", family="monospace", fontsize=9, color=_INK)
    for axis in axes[:2]:
        axis.set_xticks([])
        axis.set_yticks([])
    return figure


def plot_residual() -> Figure:
    """A residual block: two convolutions, then add the input back."""
    figure, axis = _new(9.2, 4.2)
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")
    boxes = (
        (0.5, 2.6, "Input"),
        (3.0, 2.6, "3×3 conv\nbatch norm\nReLU"),
        (5.7, 2.6, "3×3 conv\nbatch norm"),
        (8.4, 2.6, "Add"),
        (10.4, 2.6, "ReLU"),
    )
    for x, y, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                1.8,
                1.7,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=_PALE,
                edgecolor=_EDGE,
                linewidth=1.2,
            )
        )
        axis.text(x + 0.9, y + 0.85, label, ha="center", va="center", fontsize=8, color=_INK)
    for start, end in ((2.3, 3.0), (4.8, 5.7), (7.5, 8.4), (10.2, 10.4)):
        axis.annotate(
            "",
            xy=(end, 3.45),
            xytext=(start, 3.45),
            arrowprops={"arrowstyle": "->", "color": _EDGE, "lw": 1.2},
        )
    axis.annotate(
        "",
        xy=(8.7, 4.3),
        xytext=(1.4, 4.3),
        arrowprops={
            "arrowstyle": "->",
            "color": VAL_COLOR,
            "lw": 1.4,
            "connectionstyle": "arc3,rad=-0.35",
        },
    )
    axis.text(
        4.6,
        6.15,
        "Add the input. If the shape changed, a 1×1 convolution reshapes it first.",
        ha="center",
        color=VAL_COLOR,
        fontsize=9,
    )
    axis.text(
        6.0,
        0.8,
        "The block learns a change to the features. The added path still carries the input.",
        ha="center",
        color=_INK,
        fontsize=9,
    )
    return figure


def plot_epoch_strip(images: int, batch_size: int, *, drop_last: bool) -> Figure:
    """One epoch as groups of images, using the counts the caller computed."""
    if images < 1 or batch_size < 1:
        raise ValueError("images and batch_size must be positive.")
    figure, axis = _new(9.2, 2.6)
    axis.set_xlim(-0.2, images + 0.4)
    axis.set_ylim(0, 3)
    axis.axis("off")
    kept = images - 1 if drop_last and images % batch_size == 1 else images
    for index in range(images):
        dropped = index >= kept
        group = index // batch_size
        color = "#D7DDE3" if dropped else (TRAIN_COLOR if group % 2 == 0 else "#4E86B8")
        axis.add_patch(Rectangle((index * 1.0, 1.1), 0.82, 0.9, facecolor=color, edgecolor="white"))
    axis.text(
        0,
        2.4,
        f"{images} training images  ·  batch size {batch_size}",
        color=_INK,
        fontsize=11,
    )
    note = (
        "One image set aside so a batch is never a single image."
        if drop_last
        else ("Each shaded group is one weight update.")
    )
    axis.text(0, 0.35, note, color=_INK, fontsize=9)
    return figure


def plot_overfit_sketch() -> Figure:
    """The shape to look for. The numbers are not from a run."""
    epochs = list(range(1, 9))
    train = [1.0, 0.72, 0.5, 0.34, 0.24, 0.16, 0.11, 0.07]
    validation = [1.05, 0.84, 0.66, 0.58, 0.6, 0.72, 0.88, 1.02]
    figure, axis = _panel()
    axis.plot(epochs, train, color=TRAIN_COLOR, marker="o", label="Train")
    axis.plot(epochs, validation, color=VAL_COLOR, marker="o", label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_yticks([])
    axis.set_ylabel("Loss")
    axis.set_title("Sketch of overfitting. Not a measurement.")
    axis.legend(frameon=False)
    return figure


def plot_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    xlabel: str,
    title: str,
    color: str = TRAIN_COLOR,
) -> Figure:
    """Horizontal bars. ``labels`` and ``values`` must be the same length."""
    if len(labels) != len(values) or not labels:
        raise ValueError("labels and values must be the same non-empty length.")
    height = max(2.8, 0.42 * len(labels) + 1.2)
    figure, axis = plt.subplots(figsize=(9.2, height), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    _quiet(axis)
    positions = list(range(len(labels)))
    axis.barh(positions, list(values), color=color)
    axis.set_yticks(positions, labels=list(labels))
    axis.invert_yaxis()
    axis.set_xlabel(xlabel)
    axis.set_title(title)
    axis.grid(True, axis="x", color="#E1E4E8", linewidth=0.8)
    axis.grid(False, axis="y")
    return figure


def plot_memory_parts(
    labels: Sequence[str],
    values: Sequence[int],
    *,
    parts_list: Sequence[bool],
) -> Figure:
    """Byte counts. Entries marked as a parts list are drawn in the other color."""
    if not (len(labels) == len(values) == len(parts_list)) or not labels:
        raise ValueError("labels, values, and parts_list must match and be non-empty.")
    colors = [VAL_COLOR if flag else TRAIN_COLOR for flag in parts_list]
    height = max(2.8, 0.46 * len(labels) + 1.3)
    figure, axis = plt.subplots(figsize=(9.2, height), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    _quiet(axis)
    positions = list(range(len(labels)))
    axis.barh(positions, [value / (1024**2) for value in values], color=colors)
    axis.set_yticks(positions, labels=list(labels))
    axis.invert_yaxis()
    axis.set_xlabel("MiB")
    axis.set_title("Memory parts for this model and this batch")
    axis.grid(True, axis="x", color="#E1E4E8", linewidth=0.8)
    axis.grid(False, axis="y")
    return figure


def varying_indexes(maps: torch.Tensor, count: int) -> torch.Tensor:
    """Indexes of the ``count`` slices with the largest spatial variance.

    ``maps`` is (count_all, height, width) or (count_all, channels, height, width).
    Variance is over every axis except the first.
    """
    if maps.ndim < 2:
        raise ValueError("maps needs a slice axis and a spatial axis.")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    kept = min(count, int(maps.shape[0]))
    flat = maps.detach().float().reshape(maps.shape[0], -1)
    score = flat.var(dim=1)
    return torch.topk(score, k=kept).indices


def display_filters(weight: torch.Tensor, count: int = 8) -> torch.Tensor:
    """``count`` filters scaled independently to 0–1, shape (count, 3, k, k).

    The scaling is for the picture only. It is not a normalization the network uses.
    """
    if weight.ndim != 4 or int(weight.shape[1]) != 3:
        raise ValueError("expected conv weights of shape (filters, 3, k, k).")
    indexes = varying_indexes(weight, count)
    chosen = weight.detach().float().cpu()[indexes]
    tiles: list[torch.Tensor] = []
    for tile in chosen:
        low = float(tile.min())
        high = float(tile.max())
        if high - low < 1e-8:
            tiles.append(torch.zeros_like(tile))
        else:
            tiles.append((tile - low) / (high - low))
    return torch.stack(tiles)


def plot_filter_tiles(tiles: torch.Tensor) -> Figure:
    """RGB filter tiles. ``tiles`` is (n, 3, k, k) in 0–1."""
    if tiles.ndim != 4 or int(tiles.shape[1]) != 3:
        raise ValueError("tiles must have shape (n, 3, k, k).")
    count = int(tiles.shape[0])
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(9.2, 1.6 * rows + 0.4),
        dpi=120,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    flat = list(axes.flat) if count > 1 else [axes]
    for axis, tile in zip(flat, tiles, strict=False):
        picture = tile.permute(1, 2, 0).numpy()
        axis.imshow(picture, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat[count:]:
        axis.axis("off")
    return figure


@functools.lru_cache(maxsize=1)
def novel_conv1_weight() -> torch.Tensor:
    """First-layer filters from a seeded novel-mode ResNet, before any training.

    The global RNG is restored. The tensor is cached and must not be modified.
    """
    state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        model = create_model(2, LearningMode.NOVEL, weights=None)
        return model.conv1.weight.detach().cpu().clone()
    finally:
        torch.set_rng_state(state)


def conv1_feature_maps(weight: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Feature maps from a 7×7 stride-2 convolution using ``weight``.

    ``weight`` is (filters, 3, kernel, kernel). ``batch`` is NCHW. The model's
    own module is not moved; a temporary convolution runs on CPU.
    """
    if weight.ndim != 4:
        raise ValueError("weight must be (filters, channels, kernel, kernel).")
    if batch.ndim != 4:
        raise ValueError("batch must be NCHW.")
    filters, channels, kernel, _width = (int(value) for value in weight.shape)
    # Building the temporary layer draws random weights. Restore the RNG so a
    # lesson cannot change the run the learner is about to start.
    state = torch.get_rng_state()
    try:
        layer = nn.Conv2d(
            channels,
            filters,
            kernel_size=kernel,
            stride=2,
            padding=kernel // 2,
            bias=False,
        )
        source = weight.detach().float().cpu()
        image = batch.detach().float().cpu()
        with torch.no_grad():
            layer.weight.copy_(source)
            maps = layer(image)
        return maps[0].clone()
    finally:
        torch.set_rng_state(state)


def plot_feature_maps(maps: torch.Tensor) -> Figure:
    """Grayscale feature maps, each scaled on its own. ``maps`` is (n, h, w)."""
    if maps.ndim != 3:
        raise ValueError("maps must have shape (n, h, w).")
    count = int(maps.shape[0])
    columns = min(8, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(9.2, 1.7 * rows + 0.3),
        dpi=120,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    flat = list(axes.flat) if count > 1 else [axes]
    chosen = maps.detach().float().cpu()
    for axis, feature in zip(flat, chosen, strict=False):
        axis.imshow(feature.numpy(), cmap="magma")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat[count:]:
        axis.axis("off")
    return figure
