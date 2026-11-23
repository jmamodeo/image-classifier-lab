"""Built-in datasets so the app can be tried without any photos.

The shapes are synthetic: circles, squares, and triangles on a light
background. Color and shape both differ, on purpose: a short run should move
the curves, and a person can see at a glance which class is which. These are
drawings, not a benchmark. The built-in images are a small labeled subset of
the public CIFAR-10 dataset, downloaded once and cached.
"""

from __future__ import annotations

import os
import random
import shutil
import ssl
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from image_classifier_lab.data import IMAGE_SIZE, LabeledImages

EXAMPLE_IMAGES_PER_CLASS = 8
_EXAMPLE_SEED = 0
PUBLIC_IMAGES_PER_CLASS = 24
PUBLIC_LABELS = ("airplane", "automobile", "bird")
_DOWNLOAD_TIMEOUT_SECONDS = 60


def make_cifar_dataset(
    images_per_class: int = PUBLIC_IMAGES_PER_CLASS,
    seed: int = _EXAMPLE_SEED,
    root: str | None = None,
) -> list[LabeledImages]:
    """Load a small, deterministically selected subset of labeled CIFAR-10 images."""
    if isinstance(images_per_class, bool) or not isinstance(images_per_class, int):
        raise ValueError("images_per_class must be an integer.")
    if images_per_class < 2:
        raise ValueError("images_per_class must be at least 2.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    cache_root = Path(root) if root else Path.home() / ".cache" / "imagepro"
    dataset = _open_cifar(cache_root)
    label_indexes = {label: dataset.class_to_idx[label] for label in PUBLIC_LABELS}
    rng = random.Random(seed)
    groups: list[LabeledImages] = []
    for label in PUBLIC_LABELS:
        indexes = [
            index
            for index, sample_label in enumerate(dataset.targets)
            if sample_label == label_indexes[label]
        ]
        rng.shuffle(indexes)
        selected = indexes[:images_per_class]
        if len(selected) < images_per_class:
            raise RuntimeError(f"CIFAR-10 did not provide {images_per_class} images for {label}.")
        images = tuple(dataset[index][0].convert("RGB") for index in selected)
        groups.append(LabeledImages(label, images))
    return groups


def _open_cifar(root: Path) -> Any:
    """The CIFAR-10 training split, downloading and extracting it on first use."""
    from torchvision.datasets import CIFAR10
    from torchvision.datasets.utils import check_integrity, extract_archive

    try:
        return CIFAR10(root=str(root), train=True, download=False)
    except RuntimeError:
        pass
    archive = root / CIFAR10.filename
    if not check_integrity(str(archive), CIFAR10.tgz_md5):
        _download(CIFAR10.url, archive)
        if not check_integrity(str(archive), CIFAR10.tgz_md5):
            archive.unlink(missing_ok=True)
            raise RuntimeError("The CIFAR-10 download was incomplete. Try again.")
    extract_archive(str(archive), str(root))
    return CIFAR10(root=str(root), train=True, download=False)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "imagepro"})
        with (
            os.fdopen(handle, "wb") as file,
            urllib.request.urlopen(
                request, context=https_context(), timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
        ):
            shutil.copyfileobj(response, file, length=1 << 20)
        os.replace(partial, destination)
    finally:
        Path(partial).unlink(missing_ok=True)


def https_context() -> ssl.SSLContext:
    """The system's trusted certificates plus certifi's bundle.

    The python.org build of Python for macOS ships without trusted roots until
    its Install Certificates script is run, so the default context alone fails.
    """
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return context
    context.load_verify_locations(cafile=certifi.where())
    return context


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
