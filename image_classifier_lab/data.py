"""Loading, splitting, and augmenting the images a run trains on.

Augmentation is applied inside :meth:`ImageListDataset.__getitem__`, so each
epoch draws new flips and rotations. Validation images never take that path.
The split is stratified by class and keeps at least one training image and one
validation image per class.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from image_classifier_lab.errors import DatasetError

IMAGE_SIZE = 224
FLIP_PROBABILITY = 0.5
MAX_ROTATION_DEGREES = 15.0
MIN_CLASSES = 2
MAX_CLASSES = 8
MIN_IMAGES_PER_CLASS = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class LabeledImages:
    """One class name and the images that belong to it, in the order given."""

    label: str
    images: tuple[Image.Image, ...]


@dataclass(frozen=True)
class SplitData:
    """A stratified holdout. Labels are indexes into ``class_names``."""

    train_images: tuple[Image.Image, ...]
    train_labels: tuple[int, ...]
    val_images: tuple[Image.Image, ...]
    val_labels: tuple[int, ...]
    class_names: tuple[str, ...]


def open_rgb(data: bytes) -> Image.Image:
    """Decode image bytes into a detached RGB image.

    EXIF orientation is applied, so a phone photo is not silently sideways.
    Transparent pixels are composited onto white; otherwise a PNG sticker
    becomes a black rectangle the classifier can memorize.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            oriented = ImageOps.exif_transpose(image)
            source = image if oriented is None else oriented
            if source.mode == "RGBA":
                background = Image.new("RGB", source.size, (255, 255, 255))
                background.paste(source, mask=source.getchannel("A"))
                rgb = background
            else:
                rgb = source.convert("RGB")
            return rgb.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DatasetError("The file is not an image this app can read.") from exc


def open_uploaded_file(data: bytes, name: str) -> Image.Image:
    """Decode one upload. The error names the file when decoding fails."""
    safe_name = Path(name).name.replace("\n", " ").strip() or "upload"
    try:
        return open_rgb(data)
    except DatasetError as exc:
        raise DatasetError(f"Could not read '{safe_name}' as an image.") from exc


def validate_groups(groups: Sequence[LabeledImages]) -> None:
    """Raise :class:`DatasetError` when the classes cannot be trained."""
    if len(groups) < MIN_CLASSES:
        raise DatasetError(f"Add at least {MIN_CLASSES} classes.")
    if len(groups) > MAX_CLASSES:
        raise DatasetError(f"Use at most {MAX_CLASSES} classes.")
    labels = [group.label.strip() for group in groups]
    if any(label == "" for label in labels):
        raise DatasetError("Every class needs a label.")
    folded = [label.casefold() for label in labels]
    if len(set(folded)) != len(folded):
        raise DatasetError(
            "Those labels match once capitalization is ignored. Give each class its own name."
        )
    for label, group in zip(labels, groups, strict=True):
        count = len(group.images)
        if count < MIN_IMAGES_PER_CLASS:
            noun = "image" if count == 1 else "images"
            raise DatasetError(
                f"'{label}' has {count} {noun}. Each class needs at least "
                f"{MIN_IMAGES_PER_CLASS} so one image can stay out of training for validation."
            )


def validation_count(num_images: int, val_fraction: float) -> int:
    """How many images of one class to hold out.

    The fraction is rounded to the nearest image, then clamped so the class
    keeps at least one training image and one validation image.
    """
    if num_images < MIN_IMAGES_PER_CLASS:
        raise DatasetError(
            f"Each class needs at least {MIN_IMAGES_PER_CLASS} images to make a holdout."
        )
    if isinstance(val_fraction, bool) or not isinstance(val_fraction, (int, float)):
        raise DatasetError("Validation fraction must be a number between 0 and 1.")
    fraction = float(val_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise DatasetError("Validation fraction must be between 0 and 1, not including the ends.")
    count = int(round(fraction * num_images))
    return min(max(count, 1), num_images - 1)


def split_dataset(groups: Sequence[LabeledImages], val_fraction: float, seed: int) -> SplitData:
    """Stratify each class into train and validation.

    One :class:`random.Random` seeded with ``seed`` shuffles every class, so
    the holdout does not depend on Python's global RNG or on PyTorch's. Class
    index 0 is the first group, which is the order the person added them.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetError("Seed must be a non-negative integer.")
    validate_groups(groups)
    rng = random.Random(seed)
    class_names = tuple(group.label.strip() for group in groups)
    train_images: list[Image.Image] = []
    train_labels: list[int] = []
    val_images: list[Image.Image] = []
    val_labels: list[int] = []
    for class_index, group in enumerate(groups):
        order = list(range(len(group.images)))
        rng.shuffle(order)
        held_out = validation_count(len(group.images), val_fraction)
        val_ids = order[:held_out]
        train_ids = order[held_out:]
        for image_index in train_ids:
            train_images.append(group.images[image_index])
            train_labels.append(class_index)
        for image_index in val_ids:
            val_images.append(group.images[image_index])
            val_labels.append(class_index)
    return SplitData(
        train_images=tuple(train_images),
        train_labels=tuple(train_labels),
        val_images=tuple(val_images),
        val_labels=tuple(val_labels),
        class_names=class_names,
    )


def describe_split(split: SplitData) -> str:
    """One line a person can read: counts per class, train then validation."""
    parts: list[str] = []
    for index, name in enumerate(split.class_names):
        train_count = sum(label == index for label in split.train_labels)
        val_count = sum(label == index for label in split.val_labels)
        parts.append(f"{name}: {train_count} train / {val_count} validation")
    return " · ".join(parts)


def fingerprint_groups(groups: Sequence[LabeledImages]) -> tuple[str, ...]:
    """Stable identity of the labeled images, based on the resized pixels.

    Used to tell a stored training run from the images currently on screen.
    Hashing the resized image keeps a multi-megapixel upload from being hashed
    at full resolution on every rerun.
    """
    hashes: list[str] = []
    for group in groups:
        digest = hashlib.sha256()
        for image in group.images:
            digest.update(prepare_image(image).tobytes())
        hashes.append(f"{group.label.strip()}:{len(group.images)}:{digest.hexdigest()}")
    return tuple(hashes)


def prepare_image(image: Image.Image) -> Image.Image:
    """Return a square RGB image at the size ResNet-18 is trained on."""
    rgb = image.convert("RGB")
    resized = F.resize(
        rgb,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return resized


def apply_augmentation(
    image: Image.Image,
    generator: torch.Generator,
    *,
    flip_probability: float = FLIP_PROBABILITY,
    max_degrees: float = MAX_ROTATION_DEGREES,
) -> Image.Image:
    """Return a new image with a random horizontal flip and a small rotation.

    ``generator`` is the only source of randomness. A seeded generator repeats
    the same views, and this function does not advance PyTorch's global RNG.
    Exposed corners keep the corner pixel's color so a rotated photo does not
    grow a black triangle the model can latch onto.
    """
    image = image.copy()
    if torch.rand((), generator=generator).item() < flip_probability:
        image = F.hflip(image)
    if max_degrees > 0:
        angle = (torch.rand((), generator=generator).item() * 2.0 - 1.0) * max_degrees
        if abs(angle) > 1e-3:
            corner = image.getpixel((0, 0))
            fill = list(corner) if isinstance(corner, tuple) else [int(corner)]
            image = F.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=False,
                fill=fill,
            )
    return image


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    """Scale to ``[0, 1]`` and apply ImageNet mean and standard deviation."""
    tensor = F.to_tensor(image)
    return F.normalize(tensor, list(IMAGENET_MEAN), list(IMAGENET_STD))


def image_to_model_input(image: Image.Image) -> torch.Tensor:
    """Resize and normalize one image into a batch of shape ``(1, 3, 224, 224)``."""
    return to_normalized_tensor(prepare_image(image)).unsqueeze(0)


def augmentation_demo_index(images: Sequence[Image.Image]) -> int:
    """Index of the image a flip or a small rotation changes the most.

    A centered circle looks the same after those transforms, so it is a poor
    picture of augmentation. Callers use this to pick the preview.
    """
    if not images:
        raise ValueError("images is empty.")
    best_index = 0
    best_score = -1.0
    for index, image in enumerate(images):
        score = _augmentation_visibility(prepare_image(image))
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _augmentation_visibility(image: Image.Image) -> float:
    small = image.resize((24, 24))
    corner = small.getpixel((0, 0))
    fill = list(corner) if isinstance(corner, tuple) else [int(corner)]
    rotated = F.rotate(
        small,
        12,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=fill,
    )
    return _pixel_delta(small, rotated) + _pixel_delta(small, F.hflip(small))


def _pixel_delta(left: Image.Image, right: Image.Image) -> float:
    left_bytes = left.tobytes()
    right_bytes = right.tobytes()
    return float(sum(abs(a - b) for a, b in zip(left_bytes, right_bytes, strict=True)))


def preview_augmentations(image: Image.Image, count: int, seed: int) -> list[Image.Image]:
    """Draw ``count`` augmented views without touching the global RNG.

    The draws match the training augmentation (flip, then a small rotation)
    but use their own generator, so looking at a preview does not change the
    sequence a later training run will see.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be an integer of at least 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    generator = torch.Generator().manual_seed(seed)
    prepared = prepare_image(image)
    return [apply_augmentation(prepared, generator) for _ in range(count)]


class ImageListDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images stored as pictures. Tensors are built when an item is read.

    That is the whole fix for augmentation that used to be baked in at upload
    time: a training dataset with ``augment=True`` calls
    :func:`apply_augmentation` on every ``__getitem__``. A new epoch reads the
    dataset again, so it gets a new draw. Validation datasets pass
    ``augment=False`` and return the same tensor every time.
    """

    def __init__(
        self,
        images: Sequence[Image.Image],
        labels: Sequence[int],
        *,
        augment: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must be the same length.")
        if augment and generator is None:
            raise ValueError("An augmenting dataset needs a torch.Generator.")
        self.images = tuple(prepare_image(image) for image in images)
        self.labels = tuple(int(label) for label in labels)
        self.augment = augment
        self.generator = generator

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image: Image.Image = self.images[index]
        if self.augment:
            if self.generator is None:
                raise ValueError("An augmenting dataset needs a torch.Generator.")
            image = apply_augmentation(image, self.generator)
        return to_normalized_tensor(image), self.labels[index]


def build_datasets(
    split: SplitData,
    *,
    augment: bool,
    seed: int,
) -> tuple[ImageListDataset, ImageListDataset]:
    """Training dataset, optionally augmented, and a plain validation dataset.

    The training generator is seeded with ``seed`` and then left to run, so
    epoch 2 continues the sequence instead of repeating epoch 1.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    generator = torch.Generator().manual_seed(seed) if augment else None
    train_dataset = ImageListDataset(
        split.train_images,
        split.train_labels,
        augment=augment,
        generator=generator,
    )
    val_dataset = ImageListDataset(split.val_images, split.val_labels, augment=False)
    return train_dataset, val_dataset
