"""A tiny synthetic dataset so the app can be tried without any photos.

Circles, squares, and triangles on a light background. Color and shape both
differ, on purpose: a short run should move the curves, and a person can see
at a glance which class is which. These are drawings, not a benchmark.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from image_classifier_lab.data import IMAGE_SIZE, LabeledImages

EXAMPLE_IMAGES_PER_CLASS = 8
_EXAMPLE_SEED = 0


def make_shape_dataset(
    images_per_class: int = EXAMPLE_IMAGES_PER_CLASS,
    image_size: int = IMAGE_SIZE,
    seed: int = _EXAMPLE_SEED,
) -> list[LabeledImages]:
    """Build the three shape classes. The same seed redraws the same pictures."""
    if isinstance(images_per_class, bool) or not isinstance(images_per_class, int):
        raise ValueError("images_per_class must be an integer.")
    if images_per_class < 2:
        raise ValueError("images_per_class must be at least 2.")
    if image_size < 32:
        raise ValueError("image_size must be at least 32.")
    rng = random.Random(seed)
    drawers = (
        ("circle", _draw_circle, (176, 48, 42)),
        ("square", _draw_square, (32, 86, 150)),
        ("triangle", _draw_triangle, (36, 122, 70)),
    )
    groups: list[LabeledImages] = []
    for label, drawer, color in drawers:
        images = tuple(drawer(rng, image_size, color) for _ in range(images_per_class))
        groups.append(LabeledImages(label, images))
    return groups


def _canvas(rng: random.Random, size: int) -> Image.Image:
    shade = rng.randint(236, 248)
    return Image.new("RGB", (size, size), (shade, shade, shade))


def _center(rng: random.Random, size: int) -> tuple[int, int]:
    jitter = int(size * 0.08)
    mid = size // 2
    return (
        rng.randint(mid - jitter, mid + jitter),
        rng.randint(mid - jitter, mid + jitter),
    )


def _radius(rng: random.Random, size: int) -> int:
    return rng.randint(int(size * 0.22), int(size * 0.34))


def _draw_circle(
    rng: random.Random,
    size: int,
    color: tuple[int, int, int],
) -> Image.Image:
    image = _canvas(rng, size)
    draw = ImageDraw.Draw(image)
    cx, cy = _center(rng, size)
    radius = _radius(rng, size)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    return image


def _draw_square(
    rng: random.Random,
    size: int,
    color: tuple[int, int, int],
) -> Image.Image:
    image = _canvas(rng, size)
    draw = ImageDraw.Draw(image)
    cx, cy = _center(rng, size)
    half = _radius(rng, size)
    draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=color)
    return image


def _draw_triangle(
    rng: random.Random,
    size: int,
    color: tuple[int, int, int],
) -> Image.Image:
    image = _canvas(rng, size)
    draw = ImageDraw.Draw(image)
    cx, cy = _center(rng, size)
    radius = _radius(rng, size)
    points = [
        (cx, cy - radius),
        (cx - int(radius * 0.92), cy + int(radius * 0.78)),
        (cx + int(radius * 0.92), cy + int(radius * 0.78)),
    ]
    draw.polygon(points, fill=color)
    return image
