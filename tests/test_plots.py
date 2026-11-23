"""Charts exist and carry both splits. No pixel snapshots."""

from __future__ import annotations

import matplotlib.pyplot as plt

from image_classifier_lab.plots import plot_confusion, plot_history, plot_probabilities
from image_classifier_lab.training import EpochMetrics


def test_history_plot_overlays_train_and_validation() -> None:
    history = (
        EpochMetrics(1, 1.0, 0.4, 1.1, 0.3),
        EpochMetrics(2, 0.5, 0.8, 0.7, 0.6),
    )
    figure = plot_history(history)
    labels = [line.get_label() for axis in figure.axes for line in axis.get_lines()]
    assert labels.count("Train") == 2
    assert labels.count("Validation") == 2
    plt.close(figure)


def test_confusion_and_probability_plots_accept_class_names() -> None:
    confusion = plot_confusion(((2, 1), (0, 3)), ["circle", "square"])
    assert confusion.axes[0].get_xlabel() == "Predicted"
    plt.close(confusion)
    bars = plot_probabilities(["circle", "square"], [0.2, 0.8])
    assert bars.axes[0].patches
    plt.close(bars)
