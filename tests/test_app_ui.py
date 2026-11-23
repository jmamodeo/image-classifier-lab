"""The lab page, driven through Streamlit's AppTest the way a browser would."""

from __future__ import annotations

import io
import urllib.error
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from image_classifier_lab import examples, ui
from image_classifier_lab.data import LabeledImages
from image_classifier_lab.errors import TrainingError
from image_classifier_lab.model import LearningMode

APP_PATH = str(Path(__file__).resolve().parent.parent / "main.py")
TIMEOUT = 180
IMAGES_PER_CLASS = 4


def new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


def by_label(widgets: Sequence[Any], label: str) -> Any:
    return next(widget for widget in widgets if widget.label == label)


def train_button(app: AppTest) -> Any:
    return by_label(app.button, "Train model")


def png(color: tuple[int, int, int] = (200, 40, 40), size: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def images(app: AppTest) -> list[Any]:
    """Every image element on the page. Matplotlib figures arrive as images."""
    return [element for element in app.main if type(element).__name__ == "Image"]


def texts(elements: Sequence[Any]) -> list[str]:
    return [str(element.value) for element in elements]


def configure_tiny_run(app: AppTest, mode: LearningMode = LearningMode.NOVEL) -> AppTest:
    """Smallest run the controls allow: one epoch, batch of two, CPU, seed 0."""
    app.radio(key="source").set_value("example")
    by_label(app.sidebar.radio, "Training device").set_value("cpu")
    by_label(app.sidebar.selectbox, "Learning mode").set_value(mode)
    by_label(app.sidebar.number_input, "Batch size").set_value(2)
    by_label(app.sidebar.slider, "Epochs").set_value(1)
    by_label(app.sidebar.slider, "Validation holdout (%)").set_value(50)
    by_label(app.sidebar.number_input, "Seed").set_value(0)
    return app


@pytest.fixture
def small_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    original = examples.make_shape_dataset
    monkeypatch.setattr(ui, "make_shape_dataset", lambda: original(IMAGES_PER_CLASS))


@pytest.fixture(scope="module")
def trained_app() -> Iterator[AppTest]:
    """One real tiny training run, shared by the tests that read its results."""
    patch = pytest.MonkeyPatch()
    original = examples.make_shape_dataset
    patch.setattr(ui, "make_shape_dataset", lambda: original(IMAGES_PER_CLASS))
    try:
        app = new_app().run()
        configure_tiny_run(app).run()
        train_button(app).click().run()
        yield app
    finally:
        patch.undo()


def test_app_loads_on_the_upload_form() -> None:
    app = new_app().run()
    assert not app.exception
    assert app.title[0].value == ui.APP_TITLE
    assert app.radio(key="source").value == "upload"
    assert train_button(app).disabled
    assert any("Add a label" in message for message in texts(app.info))


def test_training_renders_the_results(trained_app: AppTest) -> None:
    app = trained_app
    assert not app.exception
    assert "How the run went" in texts(app.subheader)
    assert not app.error
    assert not app.warning

    names = ["circle", "square", "triangle"]
    class_table = next(t.value for t in app.table if "Accuracy" in t.value.columns)
    assert list(class_table["Class"]) == names
    assert class_table["Validation images"].sum() > 0

    run_table = next(t.value for t in app.table if "Val acc." in t.value.columns)
    assert list(run_table["Mode"]) == ["Novel"]
    assert list(run_table["Epochs"]) == [1]
    assert list(run_table["Batch"]) == [2]
    assert any(t.value.columns[0] == "Mode" and "Images/s" in t.value.columns for t in app.table)

    captions = texts(app.caption)
    assert any(text.startswith("Novel ·") for text in captions)
    assert any(text.startswith("Measured on CPU") for text in captions)
    assert any(text.startswith("Loss is the mean cross-entropy") for text in captions)
    assert "**Recent runs**" in texts(app.markdown)
    assert "**Validation confusion matrix**" in texts(app.markdown)


def test_training_adds_loss_and_confusion_figures(
    trained_app: AppTest,
    small_examples: None,
) -> None:
    fresh = new_app().run()
    configure_tiny_run(fresh).run()
    assert len(images(trained_app)) - len(images(fresh)) == 2


def test_prediction_names_a_class_and_confidence(trained_app: AppTest) -> None:
    uploader = trained_app.file_uploader(key="prediction_file")
    uploader.set_value(("probe.png", png(), "image/png")).run()
    assert not trained_app.exception
    assert not trained_app.error
    names = ("circle", "square", "triangle")
    assert any(text.strip("*") in names for text in texts(trained_app.markdown))
    assert any(text.endswith("% confidence") for text in texts(trained_app.markdown))
    uploader = trained_app.file_uploader(key="prediction_file")
    uploader.set_value(None).run()
    assert not any(text.endswith("% confidence") for text in texts(trained_app.markdown))


def test_prediction_reports_an_unreadable_upload(trained_app: AppTest) -> None:
    uploader = trained_app.file_uploader(key="prediction_file")
    uploader.set_value(("broken.png", b"not an image", "image/png")).run()
    assert not trained_app.exception
    assert any("broken.png" in message for message in texts(trained_app.error))
    trained_app.file_uploader(key="prediction_file").set_value(None).run()


def test_prediction_reports_a_model_failure(
    trained_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> list[float]:
        raise RuntimeError("boom")

    monkeypatch.setattr(ui, "predict_probabilities", fail)
    trained_app.file_uploader(key="prediction_file").set_value(("a.png", png(), "image/png")).run()
    assert any("Could not run the model on cpu" in message for message in texts(trained_app.error))
    trained_app.file_uploader(key="prediction_file").set_value(None).run()


def test_prediction_reports_a_class_mismatch(
    trained_app: AppTest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ui, "predict_probabilities", lambda *_args, **_kwargs: [0.5, 0.5])
    trained_app.file_uploader(key="prediction_file").set_value(("a.png", png(), "image/png")).run()
    assert any("does not match these classes" in message for message in texts(trained_app.error))
    trained_app.file_uploader(key="prediction_file").set_value(None).run()


def test_changed_settings_warn_that_the_results_are_stale(trained_app: AppTest) -> None:
    seed = by_label(trained_app.sidebar.number_input, "Seed")
    seed.set_value(7).run()
    assert any("Showing the previous run" in message for message in texts(trained_app.warning))
    by_label(trained_app.sidebar.number_input, "Seed").set_value(0).run()
    assert not trained_app.warning


def test_upload_form_previews_images_and_enables_training() -> None:
    app = new_app().run()
    app.text_input(key="label_1").set_value("red")
    app.text_input(key="label_2").set_value("blue")
    app.file_uploader(key="files_1").set_value(
        [("r1.png", png((200, 30, 30)), "image/png"), ("r2.png", png((210, 20, 20)), "image/png")]
    )
    app.file_uploader(key="files_2").set_value([("b1.png", png((30, 30, 200)), "image/png")])
    app.run()
    assert not app.exception
    assert "1 image" in texts(app.caption)
    assert "2 images" in texts(app.caption)
    assert any("Each class needs at least 2" in message for message in texts(app.error))
    assert train_button(app).disabled

    app.file_uploader(key="files_2").set_value(
        [("b1.png", png((30, 30, 200)), "image/png"), ("b2.png", png((20, 20, 210)), "image/png")]
    ).run()
    assert not app.error
    assert not train_button(app).disabled
    assert "**Augmentation and the holdout**" in texts(app.markdown)


def test_upload_form_without_augmentation_shows_the_plain_preview() -> None:
    app = new_app().run()
    by_label(app.sidebar.checkbox, "Augment training images").set_value(False)
    app.text_input(key="label_1").set_value("red")
    app.text_input(key="label_2").set_value("blue")
    pair = [("1.png", png(), "image/png"), ("2.png", png((10, 10, 220)), "image/png")]
    app.file_uploader(key="files_1").set_value(pair)
    app.file_uploader(key="files_2").set_value(pair).run()
    assert not app.exception
    assert any(text.startswith("Augmentation is off") for text in texts(app.caption))


def test_upload_form_rejects_an_unreadable_file() -> None:
    app = new_app().run()
    app.file_uploader(key="files_1").set_value([("bad.png", b"nope", "image/png")]).run()
    assert not app.exception
    assert any("bad.png" in message for message in texts(app.error))
    assert train_button(app).disabled


def test_upload_form_adds_and_removes_classes() -> None:
    app = new_app().run()
    assert not any(button.label == "Remove" for button in app.button)
    by_label(app.button, "Add class").click().run()
    assert not app.exception
    assert sum(button.label == "Remove" for button in app.button) == 3
    by_label(app.button, "Remove").click().run()
    assert not any(button.label == "Remove" for button in app.button)


def test_upload_form_stops_adding_at_the_class_limit() -> None:
    app = new_app().run()
    app.session_state["class_ids"] = list(range(1, 8))
    app.session_state["next_class_id"] = 8
    app.run()
    by_label(app.button, "Add class").click().run()
    assert by_label(app.button, "Add class").disabled
    assert len(app.text_input) == 8


def test_a_single_class_cannot_be_split() -> None:
    app = new_app().run()
    app.text_input(key="label_1").set_value("only")
    app.file_uploader(key="files_1").set_value(
        [("1.png", png(), "image/png"), ("2.png", png(), "image/png")]
    ).run()
    assert not app.exception
    assert app.error
    assert train_button(app).disabled


def test_batching_problem_is_shown_and_blocks_training(
    small_examples: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise TrainingError("Batch size must be at least 2.")

    monkeypatch.setattr(ui, "resolve_batching", refuse)
    app = new_app().run()
    configure_tiny_run(app).run()
    assert not app.exception
    assert "Batch size must be at least 2." in texts(app.error)
    assert train_button(app).disabled


def _train_with(
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[[], BaseException],
    mode: LearningMode,
) -> AppTest:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise failure()

    monkeypatch.setattr(ui, "train_classifier", fail)
    app = new_app().run()
    configure_tiny_run(app, mode).run()
    train_button(app).click().run()
    return app


def test_training_error_from_the_library_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _train_with(monkeypatch, lambda: TrainingError("No usable batches."), LearningMode.NOVEL)
    assert not app.exception
    assert texts(app.error) == ["No usable batches."]
    assert "How the run went" not in texts(app.subheader)


def test_transfer_download_failure_points_to_novel_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _train_with(
        monkeypatch,
        lambda: urllib.error.URLError("unreachable"),
        LearningMode.TRANSFER,
    )
    assert not app.exception
    assert texts(app.error) == [ui._DOWNLOAD_MESSAGE]


def test_unrelated_runtime_errors_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _train_with(monkeypatch, lambda: RuntimeError("shape mismatch"), LearningMode.TRANSFER)
    assert [str(item.value) for item in app.exception] == ["shape mismatch"]


def test_novel_mode_never_reports_a_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _train_with(monkeypatch, lambda: ConnectionError("down"), LearningMode.NOVEL)
    assert [str(item.value) for item in app.exception] == ["down"]


def test_is_download_error_reads_the_message() -> None:
    assert ui._is_download_error(urllib.error.URLError("x"))
    assert ui._is_download_error(RuntimeError("HTTP Error 503"))
    assert ui._is_download_error(OSError("certificate verify failed"))
    assert not ui._is_download_error(RuntimeError("size mismatch"))


def test_unstarted_form_is_detected() -> None:
    assert ui._is_unstarted([LabeledImages("", ()), LabeledImages("", ())])
    assert not ui._is_unstarted([LabeledImages("a", ()), LabeledImages("", ())])
