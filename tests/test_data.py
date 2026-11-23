"""Splits, image decoding, and per-read augmentation."""

from __future__ import annotations

from io import BytesIO

import pytest
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F

from image_classifier_lab.data import (
    ImageListDataset,
    LabeledImages,
    apply_augmentation,
    augmentation_demo_index,
    build_datasets,
    describe_split,
    fingerprint_groups,
    image_to_model_input,
    open_uploaded_file,
    preview_augmentations,
    split_dataset,
    validation_count,
)
from image_classifier_lab.errors import DatasetError


def _swatch(red: int) -> Image.Image:
    return Image.new("RGB", (8, 8), (red, 40, 80))


def _asymmetric() -> Image.Image:
    image = Image.new("RGB", (24, 24), (240, 240, 240))
    for offset in range(10):
        image.putpixel((offset, offset), (0, 0, 0))
    return image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_validation_count_rounds_then_keeps_both_sides() -> None:
    assert validation_count(10, 0.2) == 2
    assert validation_count(2, 0.2) == 1
    # 2.5 rounds to the nearest even integer, then the clamp leaves a training image.
    assert validation_count(5, 0.5) == 2
    assert validation_count(3, 0.9) == 2


def test_split_is_stratified_and_repeats_for_a_seed() -> None:
    reds = tuple(_swatch(index) for index in range(10))
    blues = tuple(_swatch(100 + index) for index in range(5))
    groups = [LabeledImages("red", reds), LabeledImages(" blue ", blues)]
    torch.manual_seed(0)
    expected_random = torch.rand(2)
    torch.manual_seed(0)
    first = split_dataset(groups, 0.2, seed=7)
    assert torch.equal(torch.rand(2), expected_random)
    second = split_dataset(groups, 0.2, seed=7)

    assert first.class_names == ("red", "blue")
    assert first.train_images == second.train_images
    assert first.val_images == second.val_images
    assert first.train_labels.count(0) == 8
    assert first.val_labels.count(0) == 2
    assert first.train_labels.count(1) == 4
    assert first.val_labels.count(1) == 1
    assert set(map(id, first.train_images)).isdisjoint(map(id, first.val_images))
    assert describe_split(first) == "red: 8 train / 2 validation · blue: 4 train / 1 validation"


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        ([LabeledImages("only", (_swatch(1), _swatch(2)))], "at least 2 classes"),
        (
            [
                LabeledImages("", (_swatch(1), _swatch(2))),
                LabeledImages("b", (_swatch(3), _swatch(4))),
            ],
            "label",
        ),
        (
            [
                LabeledImages("Cat", (_swatch(1), _swatch(2))),
                LabeledImages("cat", (_swatch(3), _swatch(4))),
            ],
            "capitalization",
        ),
        (
            [LabeledImages("a", (_swatch(1),)), LabeledImages("b", (_swatch(2), _swatch(3)))],
            "at least 2",
        ),
    ],
)
def test_split_rejects_unusable_classes(groups: list[LabeledImages], match: str) -> None:
    with pytest.raises(DatasetError, match=match):
        split_dataset(groups, 0.2, seed=0)


def test_open_uploaded_file_flattens_alpha_and_names_corrupt_files() -> None:
    transparent = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    opened = open_uploaded_file(_png_bytes(transparent), "sticker.png")
    assert opened.mode == "RGB"
    assert opened.getpixel((0, 0)) == (255, 255, 255)

    plain = open_uploaded_file(_png_bytes(_swatch(9)), "folder/photo.jpg")
    assert plain.getpixel((0, 0)) == (9, 40, 80)

    with pytest.raises(DatasetError, match="notes.txt"):
        open_uploaded_file(b"not an image", "notes.txt")


def test_forced_flip_mirrors_and_leaves_the_original_alone() -> None:
    image = _asymmetric()
    before = image.tobytes()
    generator = torch.Generator().manual_seed(5)
    flipped = apply_augmentation(image, generator, flip_probability=1, max_degrees=0)
    assert flipped.tobytes() == F.hflip(image).tobytes()
    assert image.tobytes() == before


def test_dataset_augments_on_every_read_and_validation_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def fake(image: Image.Image, generator: torch.Generator, **kwargs: object) -> Image.Image:
        calls["n"] += 1
        return image

    monkeypatch.setattr("image_classifier_lab.data.apply_augmentation", fake)
    image = _swatch(1)
    training = ImageListDataset(
        [image], [0], augment=True, generator=torch.Generator().manual_seed(0)
    )
    training[0]
    training[0]
    assert calls["n"] == 2

    def forbidden(*args: object, **kwargs: object) -> Image.Image:
        raise AssertionError("validation must not augment")

    monkeypatch.setattr("image_classifier_lab.data.apply_augmentation", forbidden)
    validation = ImageListDataset([image], [3], augment=False)
    first, label = validation[0]
    second, _label = validation[0]
    assert label == 3
    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def test_real_training_draws_are_not_identical() -> None:
    dataset = ImageListDataset(
        [_asymmetric()],
        [0],
        augment=True,
        generator=torch.Generator().manual_seed(0),
    )
    draws = [dataset[0][0].clone() for _ in range(8)]
    assert any(not torch.equal(draws[0], other) for other in draws[1:])


def test_augmentation_preview_skips_a_circle_when_a_square_is_available() -> None:
    circle = Image.new("RGB", (64, 64), (240, 240, 240))
    ImageDraw.Draw(circle).ellipse((8, 8, 56, 56), fill=(180, 40, 40))
    square = Image.new("RGB", (64, 64), (240, 240, 240))
    ImageDraw.Draw(square).rectangle((14, 14, 50, 50), fill=(30, 80, 150))
    assert augmentation_demo_index((circle, square)) == 1


def test_preview_is_seeded_and_does_not_touch_the_global_rng() -> None:
    image = _asymmetric()
    assert (
        preview_augmentations(image, 3, seed=4)[0].tobytes()
        == preview_augmentations(image, 3, seed=4)[0].tobytes()
    )
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    preview_augmentations(image, 4, seed=99)
    assert torch.equal(torch.rand(3), expected)


def test_model_input_uses_imagenet_normalization() -> None:
    from image_classifier_lab.data import IMAGENET_MEAN, IMAGENET_STD

    tensor = image_to_model_input(Image.new("RGB", (12, 12), (128, 128, 128)))
    assert tensor.shape == (1, 3, 224, 224)
    pixel = 128 / 255
    expected = [(pixel - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)]
    assert torch.allclose(tensor[0, :, 112, 112], torch.tensor(expected), atol=1e-5)


def test_fingerprint_tracks_pixels_and_labels() -> None:
    groups = [
        LabeledImages("a", (_swatch(1), _swatch(2))),
        LabeledImages("b", (_swatch(3), _swatch(4))),
    ]
    changed = [
        LabeledImages("a", (_swatch(1), _swatch(5))),
        LabeledImages("b", (_swatch(3), _swatch(4))),
    ]
    assert fingerprint_groups(groups) == fingerprint_groups(groups)
    assert fingerprint_groups(groups) != fingerprint_groups(changed)


def test_build_datasets_reserves_augmentation_for_training() -> None:
    groups = [
        LabeledImages("a", (_swatch(1), _swatch(2))),
        LabeledImages("b", (_swatch(3), _swatch(4))),
    ]
    split = split_dataset(groups, 0.5, seed=1)
    train_dataset, val_dataset = build_datasets(split, augment=True, seed=1)
    assert train_dataset.augment is True
    assert val_dataset.augment is False
    assert len(train_dataset) == 2
    assert len(val_dataset) == 2
    with pytest.raises(ValueError, match="Generator"):
        ImageListDataset([_swatch(1)], [0], augment=True)
