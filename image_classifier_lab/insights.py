"""Plain-language notes about a finished run.

The rules are deliberately coarse. A holdout of a few images is noisy, so the
text describes the shape of the curves and leaves the chart to carry the
exact values.
"""

from __future__ import annotations

from collections.abc import Sequence

from image_classifier_lab.model import LearningMode
from image_classifier_lab.training import EpochMetrics

_LOSS_MOVE = 0.02
_ACCURACY_GAP = 0.25


def summarize_training(
    history: Sequence[EpochMetrics],
    *,
    mode: LearningMode,
    trainable: int,
    total: int,
    train_size: int,
    val_size: int,
) -> tuple[str, ...]:
    """A few sentences on what this run's curves are doing."""
    if not history:
        raise ValueError("history is empty.")
    if mode is LearningMode.TRANSFER:
        scope = (
            "Transfer mode updated the new final layer and left the pretrained backbone "
            "where it was."
        )
    else:
        scope = "Novel mode updated every layer, starting from random weights."
    paragraphs = [
        (
            f"{scope} {trainable:,} of {total:,} parameters received gradients. "
            f"The optimizer learned from {train_size} images. {val_size} images were held "
            "out and only used to compute validation loss and accuracy."
        )
    ]
    last = history[-1]
    if len(history) == 1:
        paragraphs.append(
            "This was a single pass over the training images. Add epochs to see whether "
            "the curves keep falling."
        )
    else:
        paragraphs.append(_compare_endpoints(history))
    if last.train_accuracy - last.val_accuracy >= _ACCURACY_GAP and train_size >= 4:
        paragraphs.append(
            f"Training accuracy finished at {last.train_accuracy:.0%} and validation "
            f"accuracy finished at {last.val_accuracy:.0%}. The training images were easier "
            "for the model than the held-out ones."
        )
    paragraphs.append(
        f"The run ended at training loss {last.train_loss:.3f} and validation loss "
        f"{last.val_loss:.3f}."
    )
    return tuple(paragraphs)


def _compare_endpoints(history: Sequence[EpochMetrics]) -> str:
    first = history[0]
    last = history[-1]
    train_delta = last.train_loss - first.train_loss
    val_delta = last.val_loss - first.val_loss
    train_down = train_delta < -_LOSS_MOVE
    val_down = val_delta < -_LOSS_MOVE
    val_up = val_delta > _LOSS_MOVE
    if train_down and val_up:
        return (
            "Training loss fell while validation loss rose. The model is overfitting: it is "
            "getting better at the training images and worse at images held out from training. "
            "Fewer epochs, more images, or transfer mode, which trains fewer weights, are the "
            "usual ways to push back."
        )
    if train_down and val_down:
        return (
            "Training loss and validation loss both fell, so the weight updates helped on "
            "images the optimizer never used for training."
        )
    if train_down:
        return (
            "Training loss fell, and validation loss stayed about the same. A small holdout "
            "is noisy, so a flat validation curve is a weak signal rather than a verdict."
        )
    return (
        "Training loss did not fall from the first epoch to the last. A different learning "
        "rate, more epochs, or classes that are easier to tell apart are the usual things "
        "to try."
    )
