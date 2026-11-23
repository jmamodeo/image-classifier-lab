"""Short explanations shown next to the controls.

Each concept has a one-sentence summary for the control's caption and a longer
body for the panel. The text lives in ``image_classifier_lab/content/concepts.toml``, with
one paragraph per list item in ``body``. Numbers that the code also uses are
written there as placeholders and filled in from those constants so the prose
cannot drift from the implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from image_classifier_lab.content_files import fill, read_toml
from image_classifier_lab.data import FLIP_PROBABILITY, MAX_ROTATION_DEGREES, MIN_IMAGES_PER_CLASS
from image_classifier_lab.training import MIN_BATCH_SIZE


@dataclass(frozen=True)
class Concept:
    """One idea the app is trying to make visible."""

    title: str
    summary: str
    body: str


def placeholder_values() -> dict[str, object]:
    """Values for the ``{name}`` placeholders in ``concepts.toml``."""
    return {
        "flip_percent": f"{FLIP_PROBABILITY:.0%}",
        "rotation_degrees": f"{MAX_ROTATION_DEGREES:g}",
        "min_batch_size": MIN_BATCH_SIZE,
        "min_images_per_class": MIN_IMAGES_PER_CLASS,
    }


def _build() -> dict[str, Concept]:
    values = placeholder_values()
    return {
        key: Concept(
            title=fill(record["title"], values),
            summary=fill(record["summary"], values),
            body="\n\n".join(fill(text, values) for text in record["body"]),
        )
        for key, record in read_toml("concepts.toml").items()
    }


CONCEPTS: dict[str, Concept] = _build()
CONCEPT_ORDER: tuple[str, ...] = (
    "transfer_learning",
    "learning_rate",
    "batch_size",
    "epochs",
    "validation",
    "overfitting",
    "augmentation",
    "confidence",
    "seed",
    "shuffle",
)
