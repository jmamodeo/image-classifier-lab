"""The lesson sequence.

Each section has objectives, a few blocks of explanation, and exercises. The
text lives in ``image_classifier_lab/content/lessons``, one TOML file per section. Numbers
that the training code also uses are written there as placeholders and filled
in from those constants. Diagrams are drawn by the app from the block's
``visual`` name; this module does not import Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from image_classifier_lab.content_files import fill, read_toml_dir
from image_classifier_lab.data import (
    FLIP_PROBABILITY,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MAX_ROTATION_DEGREES,
)
from image_classifier_lab.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_VAL_PERCENT,
    LEARNING_RATES,
    MIN_BATCH_SIZE,
)

# Second settings used by exercises. Both sit inside the sidebar's range.
COMPARISON_EPOCHS = 15
COMPARISON_BATCH = 8
LAB_PAGE = "lab"

_INPUTS = 3 * IMAGE_SIZE * IMAGE_SIZE


@dataclass(frozen=True)
class Block:
    """One chunk of a lesson, optionally followed by a named diagram."""

    heading: str
    paragraphs: tuple[str, ...]
    visual: str = ""
    formula: str = ""


@dataclass(frozen=True)
class Exercise:
    """What to change in the training lab, and the check at the end.

    ``changed`` names sidebar settings. Everything else stays at the default
    described by :func:`keep_sentence`.
    """

    changed: tuple[str, ...]
    instructions: str
    look_for: str
    question: str
    answer: str
    key: str = ""


@dataclass(frozen=True)
class Section:
    """One lesson in order."""

    key: str
    number: int
    short: str
    title: str
    objectives: tuple[str, ...]
    blocks: tuple[Block, ...]
    exercise: Exercise
    extra_exercises: tuple[Exercise, ...] = ()

    def exercise_items(self) -> tuple[tuple[str, Exercise], ...]:
        """Return the exercises with their durable checkbox IDs."""
        return (
            (self.key, self.exercise),
            *((f"{self.key}-{item.key}", item) for item in self.extra_exercises),
        )


def default_phrases() -> tuple[tuple[str, str], ...]:
    """Sidebar settings in menu order, phrased as the value the widgets start on."""
    rate = f"{DEFAULT_LEARNING_RATE:.4g}"
    return (
        ("device", "Training device on CPU"),
        ("inference_backends", "all available inference backends checked"),
        ("precision", "export precision on float32"),
        ("mode", "Learning mode on Transfer"),
        ("learning_rate", f"learning rate at {rate}"),
        ("batch_size", f"batch size at {DEFAULT_BATCH_SIZE}"),
        ("epochs", f"epochs at {DEFAULT_EPOCHS}"),
        ("val_percent", f"validation holdout at {DEFAULT_VAL_PERCENT}%"),
        ("shuffle", "shuffle on"),
        ("augment", "augmentation on"),
        ("seed", f"seed at {DEFAULT_SEED}"),
    )


def keep_sentence(changed: Sequence[str]) -> str:
    """Tell the learner which settings to leave alone."""
    known = {key for key, _phrase in default_phrases()}
    unknown = [key for key in changed if key not in known]
    if unknown:
        raise ValueError(f"unknown setting: {', '.join(unknown)}")
    kept = [phrase for key, phrase in default_phrases() if key not in changed]
    if not kept:
        return "Every sidebar setting is part of this exercise."
    joined = "; ".join(kept)
    return f"Leave these at their defaults: {joined}."


def _mean_std() -> tuple[str, str]:
    mean = ", ".join(f"{value:.3f}" for value in IMAGENET_MEAN)
    std = ", ".join(f"{value:.3f}" for value in IMAGENET_STD)
    return mean, std


def placeholder_values() -> dict[str, object]:
    """Values for the ``{name}`` placeholders in the lesson files."""
    mean, std = _mean_std()
    return {
        "image_size": IMAGE_SIZE,
        "input_values": f"{_INPUTS:,}",
        "imagenet_mean": mean,
        "imagenet_std": std,
        "flip_percent": f"{FLIP_PROBABILITY:.0%}",
        "rotation_degrees": f"{MAX_ROTATION_DEGREES:g}",
        "lowest_learning_rate": f"{LEARNING_RATES[0]:.4g}",
        "default_learning_rate": f"{DEFAULT_LEARNING_RATE:.4g}",
        "highest_learning_rate": f"{LEARNING_RATES[-1]:.4g}",
        "default_epochs": DEFAULT_EPOCHS,
        "default_seed": DEFAULT_SEED,
        "min_batch_size": MIN_BATCH_SIZE,
        "comparison_epochs": COMPARISON_EPOCHS,
        "comparison_batch": COMPARISON_BATCH,
    }


def _exercise(record: dict[str, Any], values: dict[str, object]) -> Exercise:
    return Exercise(
        changed=tuple(record["changed"]),
        instructions=fill(record["instructions"], values),
        look_for=fill(record["look_for"], values),
        question=fill(record["question"], values),
        answer=fill(record["answer"], values),
        key=record.get("key", ""),
    )


def _block(record: dict[str, Any], values: dict[str, object]) -> Block:
    return Block(
        heading=fill(record["heading"], values),
        paragraphs=tuple(fill(text, values) for text in record["paragraphs"]),
        visual=record.get("visual", ""),
        formula=record.get("formula", ""),
    )


def _section(record: dict[str, Any], values: dict[str, object]) -> Section:
    return Section(
        key=record["key"],
        number=record["number"],
        short=fill(record["short"], values),
        title=fill(record["title"], values),
        objectives=tuple(fill(text, values) for text in record["objectives"]),
        blocks=tuple(_block(block, values) for block in record["blocks"]),
        exercise=_exercise(record["exercise"], values),
        extra_exercises=tuple(
            _exercise(item, values) for item in record.get("extra_exercises", ())
        ),
    )


def _build() -> tuple[Section, ...]:
    values = placeholder_values()
    return tuple(_section(record, values) for record in read_toml_dir("lessons"))


SECTIONS: tuple[Section, ...] = _build()
SECTION_BY_KEY: dict[str, Section] = {section.key: section for section in SECTIONS}


def visual_names() -> frozenset[str]:
    """Diagram names the lessons ask for."""
    return frozenset(
        block.visual for section in SECTIONS for block in section.blocks if block.visual
    )
