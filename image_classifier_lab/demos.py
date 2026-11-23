"""Small calculations the lessons show directly.

A hand-built filter, a softmax, and a one-number gradient step live here so a
lesson can show the arithmetic without pretending it is the ResNet. Nothing
in this module reads a trained model or a finished run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image

from image_classifier_lab.data import IMAGENET_MEAN, IMAGENET_STD

# Sobel kernel: responds to a vertical edge. Hand-built, not learned.
VERTICAL_EDGE: tuple[tuple[float, float, float], ...] = (
    (-1.0, 0.0, 1.0),
    (-2.0, 0.0, 2.0),
    (-1.0, 0.0, 1.0),
)

NEURON_INPUTS: tuple[float, ...] = (1.0, 0.0, 0.5)
NEURON_WEIGHTS: tuple[float, ...] = (0.4, -1.0, 0.6)
NEURON_BIAS = 0.1

# A made-up grid for practicing how to read a confusion matrix. Not a run.
TOY_NAMES: tuple[str, ...] = ("circle", "square", "triangle")
TOY_CONFUSION: tuple[tuple[int, ...], ...] = (
    (2, 0, 0),
    (1, 1, 0),
    (0, 0, 2),
)

TOY_LOSS_TARGET = 2.0


def pixel_pipeline(
    rgb: tuple[int, int, int],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return the 0–1 values and the ImageNet-normalized values for one pixel.

    Matches :func:`image_classifier_lab.data.to_normalized_tensor`: divide by 255, subtract
    the ImageNet mean, divide by the ImageNet standard deviation, per channel.
    """
    if len(rgb) != 3:
        raise ValueError("rgb must have three channels.")
    for value in rgb:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("each channel must be an integer from 0 to 255.")
    unit = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
    normalized = (
        (unit[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
        (unit[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
        (unit[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
    )
    return unit, normalized


def interesting_crop(
    image: Image.Image,
    size: int = 8,
) -> tuple[tuple[tuple[tuple[int, int, int], ...], ...], tuple[int, int]]:
    """An ``size``×``size`` RGB crop whose red channel varies the most.

    Non-overlapping windows, so a flat background loses to an edge. The crop
    is what the number grid draws, and the second value is its top-left corner
    in row, column order. Values are the picture's 0–255 channels, before
    normalization.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least 2.")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < size or height < size:
        raise ValueError("image is smaller than the crop.")
    pixels = rgb.load()
    best: tuple[tuple[tuple[int, int, int], ...], ...] | None = None
    best_origin = (0, 0)
    best_score = -1.0
    for top in range(0, height - size + 1, size):
        for left in range(0, width - size + 1, size):
            rows: list[tuple[tuple[int, int, int], ...]] = []
            reds: list[int] = []
            for row in range(size):
                cols: list[tuple[int, int, int]] = []
                for col in range(size):
                    value = pixels[left + col, top + row]
                    if not isinstance(value, tuple):
                        raise ValueError("expected an RGB pixel.")
                    channel = (int(value[0]), int(value[1]), int(value[2]))
                    cols.append(channel)
                    reds.append(channel[0])
                rows.append(tuple(cols))
            mean = sum(reds) / len(reds)
            score = sum((value - mean) ** 2 for value in reds) / len(reds)
            if score > best_score:
                best_score = score
                best = tuple(rows)
                best_origin = (top, left)
    if best is None:
        raise ValueError("no crop could be taken.")
    return best, best_origin


def grayscale(image: Image.Image, size: int) -> tuple[tuple[float, ...], ...]:
    """Downsample to a square luminance grid in the 0–255 range."""
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise ValueError("size must be an integer of at least 3.")
    gray = image.convert("L").resize((size, size))
    reader = getattr(gray, "get_flattened_data", gray.getdata)
    raw = list(reader())
    return tuple(tuple(float(raw[row * size + col]) for col in range(size)) for row in range(size))


def convolve_at(
    image: Sequence[Sequence[float]],
    kernel: Sequence[Sequence[float]],
    top: int,
    left: int,
) -> float:
    """Dot product of ``kernel`` with the window whose top-left is ``(top, left)``."""
    if not kernel or not kernel[0]:
        raise ValueError("kernel is empty.")
    height = len(kernel)
    width = len(kernel[0])
    if any(len(row) != width for row in kernel):
        raise ValueError("kernel must be rectangular.")
    total = 0.0
    for row in range(height):
        for col in range(width):
            total += float(image[top + row][left + col]) * float(kernel[row][col])
    return total


def convolve_valid(
    image: Sequence[Sequence[float]],
    kernel: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Slide ``kernel`` over ``image`` with stride 1 and no padding."""
    if not image or not image[0]:
        raise ValueError("image is empty.")
    out_h = len(image) - len(kernel) + 1
    out_w = len(image[0]) - len(kernel[0]) + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("kernel is larger than the image.")
    rows = []
    for top in range(out_h):
        rows.append(tuple(convolve_at(image, kernel, top, left) for left in range(out_w)))
    return tuple(rows)


def softmax(scores: Sequence[float]) -> tuple[float, ...]:
    """Turn scores into positive shares that add to 1.

    Subtracting the largest score first keeps the exponential from overflowing.
    The shares are identical to ``exp(score) / sum(exp(scores))``.
    """
    if not scores:
        raise ValueError("scores is empty.")
    values = [float(score) for score in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scores must be finite.")
    peak = max(values)
    shifted = [math.exp(value - peak) for value in values]
    total = sum(shifted)
    return tuple(value / total for value in shifted)


def cross_entropy(probability: float) -> float:
    """Negative natural log of the correct class's share. Zero when the share is 1."""
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("probability must be a number in (0, 1].")
    share = float(probability)
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError("probability must be in (0, 1].")
    return -math.log(share)


def neuron(
    inputs: Sequence[float],
    weights: Sequence[float],
    bias: float,
) -> tuple[tuple[float, ...], float]:
    """Weighted sum plus a bias. Also returns each input's contribution."""
    if len(inputs) != len(weights) or not inputs:
        raise ValueError("inputs and weights must be the same non-empty length.")
    parts = tuple(
        float(value) * float(weight) for value, weight in zip(inputs, weights, strict=True)
    )
    return parts, sum(parts) + float(bias)


def toy_loss(weight: float, target: float = TOY_LOSS_TARGET) -> float:
    """The bowl the step-size diagram draws: ``(weight - target)²``."""
    return (float(weight) - float(target)) ** 2


def descent_path(
    start: float,
    learning_rate: float,
    steps: int,
    target: float = TOY_LOSS_TARGET,
) -> tuple[float, ...]:
    """Plain gradient descent on :func:`toy_loss`, including the starting point.

    The gradient of ``(w - target)²`` is ``2 (w - target)``. This is not Adam
    and it is not the ResNet; it is a one-number picture of step size.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 40:
        raise ValueError("steps must be an integer from 1 to 40.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or float(learning_rate) <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number.")
    weight = float(start)
    points = [weight]
    for _ in range(steps):
        gradient = 2.0 * (weight - float(target))
        weight = weight - float(learning_rate) * gradient
        points.append(weight)
    return tuple(points)


def confusion_reading(
    confusion: Sequence[Sequence[int]],
    names: Sequence[str],
) -> tuple[str, ...]:
    """One sentence per class, then the off-diagonal mistakes, then the total."""
    if len(confusion) != len(names) or not names:
        raise ValueError("names must match the confusion matrix.")
    lines: list[str] = []
    correct = 0
    total = 0
    mistakes: list[str] = []
    for index, (name, row) in enumerate(zip(names, confusion, strict=True)):
        if len(row) != len(names):
            raise ValueError("each row must have one entry per class.")
        row_total = sum(row)
        hit = row[index]
        correct += hit
        total += row_total
        if row_total == 0:
            lines.append(f"{name}: no images in this row.")
            continue
        lines.append(f"{name}: {hit} of {row_total} correct ({hit / row_total:.0%}).")
        for other_index, count in enumerate(row):
            if other_index != index and count:
                called = names[other_index]
                noun = "image" if count == 1 else "images"
                mistakes.append(f"{count} {name} {noun} called {called}")
    if total == 0:
        lines.append("The grid is empty.")
        return tuple(lines)
    if mistakes:
        lines.append("Off the diagonal: " + "; ".join(mistakes) + ".")
    else:
        lines.append("Nothing off the diagonal: every image was labeled correctly.")
    lines.append(f"Overall: {correct} of {total} ({correct / total:.0%}).")
    return tuple(lines)
