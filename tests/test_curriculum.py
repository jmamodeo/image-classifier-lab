"""The lesson sequence: order, exercises, and the diagrams they name."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch

from image_classifier_lab.curriculum import (
    COMPARISON_BATCH,
    COMPARISON_EPOCHS,
    LAB_PAGE,
    SECTIONS,
    keep_sentence,
    visual_names,
)
from image_classifier_lab.data import IMAGE_SIZE
from image_classifier_lab.demos import VERTICAL_EDGE
from image_classifier_lab.diagrams import (
    conv1_feature_maps,
    display_filters,
    novel_conv1_weight,
    plot_bars,
    plot_batch_layout,
    plot_convolution,
    plot_cross_entropy,
    plot_descent,
    plot_epoch_strip,
    plot_feature_maps,
    plot_filter_tiles,
    plot_memory_parts,
    plot_neuron,
    plot_overfit_sketch,
    plot_residual,
    plot_softmax,
)
from image_classifier_lab.examples import make_shape_dataset
from image_classifier_lab.lesson_ui import VISUAL_RENDERERS, renderers_cover_the_lessons
from image_classifier_lab.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MIN_BATCH_SIZE,
)


def _close(figure: object) -> None:
    plt.close(figure)  # type: ignore[arg-type]


def test_sections_form_a_curriculum() -> None:
    assert [section.number for section in SECTIONS] == list(range(1, 9))
    assert len({section.key for section in SECTIONS}) == 8
    keys = [section.key for section in SECTIONS]
    assert keys == [
        "images",
        "neuron",
        "learning",
        "convolutions",
        "dynamics",
        "transfer",
        "evaluation",
        "hardware",
    ]
    assert LAB_PAGE not in keys
    for section in SECTIONS:
        assert section.objectives
        assert section.blocks
        for _, exercise in section.exercise_items():
            assert exercise.instructions
            assert exercise.look_for
            assert exercise.question
            assert len(exercise.answer) > 40
            assert "default" in keep_sentence(exercise.changed).lower()
    assert renderers_cover_the_lessons()
    assert visual_names() == set(VISUAL_RENDERERS)


def test_exercises_name_one_or_two_settings_and_keep_the_rest() -> None:
    by_key = {section.key: section.exercise for section in SECTIONS}
    assert by_key["images"].changed == ()
    assert by_key["neuron"].changed == ("mode",)
    assert by_key["learning"].changed == ("learning_rate",)
    assert by_key["convolutions"].changed == ("mode",)
    assert by_key["dynamics"].changed == ("epochs", "augment")
    assert by_key["transfer"].changed == ("mode",)
    assert by_key["evaluation"].changed == ()
    assert by_key["hardware"].changed == ("batch_size", "device")
    for exercise in by_key.values():
        assert len(exercise.changed) <= 2
    learning_keep = keep_sentence(("learning_rate",))
    assert "learning rate" not in learning_keep
    assert "batch size" in learning_keep
    assert "Transfer" in learning_keep
    hardware_keep = keep_sentence(("batch_size", "device"))
    assert "batch size" not in hardware_keep
    assert "Training device" not in hardware_keep
    assert "learning rate" in hardware_keep
    whole = " ".join(exercise.instructions for exercise in by_key.values())
    assert "Built-in shapes" in whole or "built-in shapes" in whole
    assert str(COMPARISON_EPOCHS) in by_key["dynamics"].instructions
    assert str(COMPARISON_BATCH) in by_key["hardware"].instructions
    assert str(MIN_BATCH_SIZE) in by_key["hardware"].instructions
    assert MIN_BATCH_SIZE <= COMPARISON_BATCH <= MAX_BATCH_SIZE
    assert DEFAULT_EPOCHS < COMPARISON_EPOCHS <= MAX_EPOCHS
    assert DEFAULT_BATCH_SIZE != COMPARISON_BATCH
    image_section = next(section for section in SECTIONS if section.key == "images")
    images = " ".join(paragraph for block in image_section.blocks for paragraph in block.paragraphs)
    assert str(IMAGE_SIZE) in images
    assert f"{3 * IMAGE_SIZE * IMAGE_SIZE:,}" in images


def test_hardware_exercises_match_the_panel_and_measurements() -> None:
    hardware = SECTIONS[-1]
    training, neural_engine, precision, all_units = hardware.exercise_items()
    assert "Train s/epoch" in training[1].look_for
    assert all(
        device in neural_engine[1].instructions for device in ("CPU", "GPU", "Neural Engine")
    )
    assert precision[1].changed == ("precision",)
    assert "top-1 agreement" in precision[1].answer
    assert "median latency" in precision[1].look_for
    assert "not proof" in all_units[1].answer
    text = " ".join(paragraph for block in hardware.blocks for paragraph in block.paragraphs)
    assert all(word in text for word in ("CPU", "GPU", "Neural Engine", "Training", "Inference"))


def test_lesson_figures_draw_without_a_trained_model() -> None:
    image = make_shape_dataset(images_per_class=2, image_size=32)[0].images[0]
    gray = tuple(tuple(float(value) for value in row) for row in [[0] * 6] * 6)
    figures = [
        plot_batch_layout(IMAGE_SIZE),
        plot_neuron((1.0, 0.0), (0.5, -0.25), 0.1),
        plot_softmax((1.0, 0.0, -1.0)),
        plot_cross_entropy(),
        plot_descent(0.25, steps=4),
        plot_convolution(gray, 0, 0, VERTICAL_EDGE),
        plot_residual(),
        plot_epoch_strip(6, 4, drop_last=False),
        plot_overfit_sketch(),
        plot_bars(("A", "B"), (1, 4), xlabel="Count", title="Parts"),
        plot_memory_parts(("Weights", "Outputs"), (400, 800), parts_list=(False, True)),
    ]
    assert image.size[0] == 32
    for figure in figures:
        assert figure.axes
        _close(figure)
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    weight = novel_conv1_weight()
    tiles = display_filters(weight, 4)
    maps = conv1_feature_maps(weight, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert tiles.shape[0] == 4
    assert float(tiles.min()) >= 0.0
    assert float(tiles.max()) <= 1.0
    assert maps.shape == (64, 112, 112)
    filters = plot_filter_tiles(tiles)
    features = plot_feature_maps(maps[:4])
    assert filters.axes
    assert features.axes
    _close(filters)
    _close(features)
    assert torch.equal(torch.rand(3), expected)
