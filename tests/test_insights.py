"""The sentences under the curves describe the shape of the run."""

from __future__ import annotations

from image_classifier_lab.insights import summarize_training
from image_classifier_lab.model import LearningMode
from image_classifier_lab.training import EpochMetrics


def _history(*rows: tuple[float, float, float, float]) -> tuple[EpochMetrics, ...]:
    return tuple(
        EpochMetrics(index, train_loss, train_acc, val_loss, val_acc)
        for index, (train_loss, train_acc, val_loss, val_acc) in enumerate(rows, start=1)
    )


def _text(history: tuple[EpochMetrics, ...], mode: LearningMode = LearningMode.NOVEL) -> str:
    return " ".join(
        summarize_training(
            history,
            mode=mode,
            trainable=1_026,
            total=11_177_538,
            train_size=8,
            val_size=2,
        )
    )


def test_overfitting_is_named_when_validation_rises() -> None:
    text = _text(_history((1.2, 0.4, 0.7, 0.5), (0.3, 1.0, 1.1, 0.25)))
    assert "overfitting" in text
    assert "1,026" in text
    assert "11,177,538" in text
    assert "100%" in text
    assert "25%" in text


def test_both_losses_falling_is_described_as_learning() -> None:
    text = _text(_history((1.0, 0.4, 0.9, 0.4), (0.4, 0.8, 0.5, 0.7)))
    assert "both fell" in text
    assert "overfitting" not in text


def test_flat_validation_is_not_called_a_win() -> None:
    text = _text(_history((1.0, 0.5, 0.70, 0.5), (0.4, 0.75, 0.69, 0.5)))
    assert "about the same" in text


def test_failure_to_improve_training_loss_is_stated() -> None:
    text = _text(_history((0.7, 0.5, 0.7, 0.5), (0.72, 0.5, 0.71, 0.5)), LearningMode.TRANSFER)
    assert "did not fall" in text
    assert "Transfer mode" in text


def test_a_single_epoch_asks_for_another_pass() -> None:
    text = _text(_history((0.6, 0.5, 0.7, 0.4)))
    assert "single pass" in text
    assert "0.600" in text
    assert "0.700" in text
