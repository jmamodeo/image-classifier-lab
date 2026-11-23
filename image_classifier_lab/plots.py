"""Charts for the training curves, the confusion matrix, and a prediction."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from image_classifier_lab.training import EpochMetrics

TRAIN_COLOR = "#1F4E79"
VAL_COLOR = "#C46B2C"
QUIET_COLOR = "#B7C3D0"


def _quiet_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis="y", color="#E1E4E8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=9)


def plot_history(history: Sequence[EpochMetrics]) -> Figure:
    """Train and validation loss, and train and validation accuracy, side by side."""
    if not history:
        raise ValueError("history is empty.")
    epochs = [item.epoch for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    loss_axis, accuracy_axis = axes
    _quiet_axis(loss_axis)
    _quiet_axis(accuracy_axis)
    loss_axis.plot(
        epochs,
        [item.train_loss for item in history],
        color=TRAIN_COLOR,
        marker="o",
        label="Train",
    )
    loss_axis.plot(
        epochs,
        [item.val_loss for item in history],
        color=VAL_COLOR,
        marker="o",
        label="Validation",
    )
    loss_axis.set_title("Mean loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy")
    loss_axis.legend(frameon=False)
    accuracy_axis.plot(
        epochs,
        [item.train_accuracy for item in history],
        color=TRAIN_COLOR,
        marker="o",
        label="Train",
    )
    accuracy_axis.plot(
        epochs,
        [item.val_accuracy for item in history],
        color=VAL_COLOR,
        marker="o",
        label="Validation",
    )
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Fraction correct")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.legend(frameon=False)
    if len(epochs) <= 12:
        loss_axis.set_xticks(epochs)
        accuracy_axis.set_xticks(epochs)
    return figure


def plot_confusion(confusion: Sequence[Sequence[int]], class_names: Sequence[str]) -> Figure:
    """Annotated validation confusion matrix. Rows are true, columns are predicted."""
    if len(confusion) != len(class_names):
        raise ValueError("class_names must match the confusion matrix.")
    matrix = [list(row) for row in confusion]
    figure, axis = plt.subplots(figsize=(4.8, 4.2), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(class_names)), labels=list(class_names), rotation=30, ha="right")
    axis.set_yticks(range(len(class_names)), labels=list(class_names))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Validation")
    peak = max((max(row) for row in matrix), default=0)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ink = "white" if peak and value > 0 and value >= peak * 0.6 else "#1A1A1A"
            axis.text(col_index, row_index, str(value), ha="center", va="center", color=ink)
    return figure


def plot_probabilities(class_names: Sequence[str], probabilities: Sequence[float]) -> Figure:
    """Horizontal bars of softmax share. The leading class is drawn in the train color."""
    if len(class_names) != len(probabilities) or not probabilities:
        raise ValueError("class_names and probabilities must be the same non-empty length.")
    figure, axis = plt.subplots(figsize=(5.4, 2.8), dpi=120, constrained_layout=True)
    figure.patch.set_facecolor("white")
    winner = max(range(len(probabilities)), key=lambda index: probabilities[index])
    colors = [
        TRAIN_COLOR if index == winner else QUIET_COLOR for index in range(len(probabilities))
    ]
    positions = list(range(len(class_names)))
    axis.barh(positions, [value * 100 for value in probabilities], color=colors)
    axis.set_yticks(positions, labels=list(class_names))
    axis.set_xlim(0, 100)
    axis.set_xlabel("Softmax share (%)")
    axis.invert_yaxis()
    _quiet_axis(axis)
    axis.grid(False)
    return figure
