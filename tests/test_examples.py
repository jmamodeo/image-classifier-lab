"""The built-in shapes are deterministic and legal training data."""

from __future__ import annotations

from image_classifier_lab.data import split_dataset
from image_classifier_lab.examples import EXAMPLE_IMAGES_PER_CLASS, make_shape_dataset


def test_shapes_are_stable_and_visibly_different() -> None:
    first = make_shape_dataset()
    second = make_shape_dataset()
    assert [group.label for group in first] == ["circle", "square", "triangle"]
    assert all(len(group.images) == EXAMPLE_IMAGES_PER_CLASS for group in first)
    assert first[0].images[0].size == second[0].images[0].size
    assert first[0].images[0].tobytes() == second[0].images[0].tobytes()
    assert first[0].images[0].tobytes() != first[0].images[1].tobytes()
    assert first[0].images[0].tobytes() != first[1].images[0].tobytes()


def test_shapes_can_be_split() -> None:
    split = split_dataset(make_shape_dataset(), 0.25, seed=0)
    assert split.class_names == ("circle", "square", "triangle")
    assert len(split.train_images) == 18
    assert len(split.val_images) == 6
