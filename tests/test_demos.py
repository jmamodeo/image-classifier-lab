"""Teaching arithmetic: pixels, a filter, softmax, and a one-number step."""

from __future__ import annotations

import math

import pytest
from PIL import Image

from image_classifier_lab.data import to_normalized_tensor
from image_classifier_lab.demos import (
    NEURON_BIAS,
    NEURON_INPUTS,
    NEURON_WEIGHTS,
    TOY_CONFUSION,
    TOY_NAMES,
    VERTICAL_EDGE,
    confusion_reading,
    convolve_at,
    convolve_valid,
    cross_entropy,
    descent_path,
    grayscale,
    interesting_crop,
    neuron,
    pixel_pipeline,
    softmax,
    toy_loss,
)
from image_classifier_lab.examples import make_shape_dataset


def test_pixel_pipeline_matches_the_tensor_the_model_sees() -> None:
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    _unit, normalized = pixel_pipeline((10, 20, 30))
    tensor = to_normalized_tensor(image)
    assert tensor.shape == (3, 32, 32)
    assert tensor[:, 0, 0].tolist() == pytest.approx(normalized)


def test_interesting_crop_finds_a_patch_that_varies() -> None:
    image = make_shape_dataset(images_per_class=2, image_size=64)[0].images[0]
    crop, origin = interesting_crop(image, 8)
    assert len(crop) == 8
    assert len(crop[0]) == 8
    assert origin[0] % 8 == 0
    reds = [pixel[0] for row in crop for pixel in row]
    assert max(reds) != min(reds)
    assert all(0 <= channel <= 255 for row in crop for pixel in row for channel in pixel)


def test_convolution_dot_product_is_the_sliding_window() -> None:
    image = (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (7.0, 8.0, 9.0),
    )
    kernel = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert convolve_at(image, kernel, 0, 0) == 1.0
    ones = tuple(tuple(1.0 for _ in range(3)) for _ in range(3))
    assert convolve_valid(image, ones) == ((45.0,),)
    gray = grayscale(Image.new("L", (12, 12), 40), 8)
    response = convolve_valid(gray, VERTICAL_EDGE)
    assert len(response) == 6
    assert len(response[0]) == 6


def test_softmax_shares_add_to_one_and_keep_order() -> None:
    even = softmax((0.0, 0.0))
    assert even == pytest.approx((0.5, 0.5))
    tilted = softmax((2.0, 0.0, -1.0))
    assert sum(tilted) == pytest.approx(1.0)
    assert tilted[0] > tilted[1] > tilted[2]
    assert all(share > 0 for share in tilted)


def test_cross_entropy_is_zero_only_for_a_certain_share() -> None:
    assert cross_entropy(1.0) == 0.0
    assert cross_entropy(math.exp(-1.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cross_entropy(0.0)


def test_neuron_and_the_toy_step() -> None:
    parts, total = neuron(NEURON_INPUTS, NEURON_WEIGHTS, NEURON_BIAS)
    assert parts == pytest.approx((0.4, 0.0, 0.3))
    assert total == pytest.approx(0.8)
    path = descent_path(-1.0, 0.5, 1)
    assert path[-1] == pytest.approx(2.0)
    assert toy_loss(path[-1]) == pytest.approx(0.0)
    oscillating = descent_path(-1.0, 1.0, 2)
    assert oscillating[1] == pytest.approx(5.0)
    assert oscillating[2] == pytest.approx(-1.0)


def test_toy_confusion_reading_names_the_off_diagonal() -> None:
    lines = " ".join(confusion_reading(TOY_CONFUSION, TOY_NAMES))
    assert "circle: 2 of 2" in lines
    assert "square: 1 of 2" in lines
    assert "1 square image called circle" in lines
    assert "5 of 6" in lines


def test_grayscale_rejects_a_tiny_grid() -> None:
    with pytest.raises(ValueError):
        grayscale(Image.new("L", (8, 8)), 2)
    with pytest.raises(ValueError):
        softmax(())
