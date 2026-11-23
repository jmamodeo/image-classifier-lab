"""The lesson pages: navigation, exercises, interactive demos, and the run log."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from image_classifier_lab import lesson_ui
from image_classifier_lab.curriculum import LAB_PAGE, SECTIONS
from image_classifier_lab.lesson_ui import DONE_KEY, PAGE_KEY, RUN_LOG_KEY, LabSnapshot
from image_classifier_lab.measure import RunRecord
from image_classifier_lab.model import LearningMode, create_model
from image_classifier_lab.training import EpochMetrics, TrainResult

TIMEOUT = 120
CLASS_NAMES = ("circle", "square", "triangle")


class FakeRun:
    """A finished run with a real, untrained model. No training happens."""

    def __init__(self, mode: LearningMode, signature: tuple[object, ...]) -> None:
        self.model = create_model(3, mode, weights=None)
        self.class_names = CLASS_NAMES
        self.result = TrainResult(
            history=(EpochMetrics(1, 1.2, 0.3, 1.1, 0.4),),
            confusion=((2, 0, 0), (0, 1, 1), (1, 0, 1)),
            per_class_accuracy=(1.0, 0.5, 0.5),
            trainable_parameters=10,
            total_parameters=20,
        )
        self.device = "cpu"
        self.signature = signature
        self.mode = mode
        self.details = "details"


def snapshot(
    trained: FakeRun | None = None,
    *,
    signature: tuple[object, ...] | None = (1,),
    mode: LearningMode = LearningMode.NOVEL,
    batch_size: int = 4,
) -> LabSnapshot:
    return LabSnapshot(
        class_count=3,
        source="example",
        mode=mode,
        batch_size=batch_size,
        val_percent=20,
        seed=0,
        device_label="CPU",
        device="cpu",
        signature=signature,
        trained=trained,
        inference_backends=("cpu",),
        precision="float32",
    )


def record(mode: str = "Novel", peak: int | None = None) -> RunRecord:
    return RunRecord(
        mode=mode,
        learning_rate=0.001,
        batch_size=4,
        epochs=2,
        augment=True,
        device_label="CPU",
        seed=0,
        final_train_loss=0.5,
        final_val_loss=0.6,
        final_val_accuracy=0.75,
        mean_train_seconds=0.25,
        mean_val_seconds=0.0,
        images_per_second=None if peak is None else 40.0,
        trainable=10,
        total=20,
        peak_memory_bytes=peak,
    )


def curriculum_script(lab):  # noqa: ANN001
    from image_classifier_lab.lesson_ui import render_curriculum

    render_curriculum(lab)


def run_log_script() -> None:
    from image_classifier_lab.lesson_ui import render_run_log

    render_run_log()


def open_lessons(lab: LabSnapshot | None = None, page: str | None = None) -> AppTest:
    app = AppTest.from_function(
        curriculum_script, args=(lab or snapshot(),), default_timeout=TIMEOUT
    )
    if page is not None:
        app.session_state[PAGE_KEY] = page
    return app.run()


def captions(app: AppTest) -> list[str]:
    return [str(item.value) for item in app.caption]


@pytest.mark.parametrize("section", SECTIONS, ids=[section.key for section in SECTIONS])
def test_every_lesson_renders(section: object) -> None:
    key = section.key  # type: ignore[attr-defined]
    app = open_lessons(page=key)
    assert not app.exception
    assert app.subheader[0].value.endswith(section.title)  # type: ignore[attr-defined]
    assert app.checkbox(key=f"done-{key}").value is False


@pytest.mark.parametrize("mode", [LearningMode.NOVEL, LearningMode.TRANSFER])
def test_lessons_that_read_a_run_describe_its_filters_and_matrix(mode: LearningMode) -> None:
    lab = snapshot(FakeRun(mode, (1,)), mode=mode)
    filters = open_lessons(lab, "convolutions")
    assert not filters.exception
    assert any(f"latest {mode.title} run" in text for text in captions(filters))
    assert not any("sidebar has changed" in text for text in captions(filters))

    evaluation = open_lessons(lab, "evaluation")
    assert not evaluation.exception
    assert "**Your latest run**" in [item.value for item in evaluation.markdown]
    assert not evaluation.warning


def test_stale_run_is_flagged_in_the_lessons() -> None:
    lab = snapshot(FakeRun(LearningMode.NOVEL, (1,)), signature=(2,))
    assert any(
        "sidebar has changed" in text for text in captions(open_lessons(lab, "convolutions"))
    )
    assert open_lessons(lab, "evaluation").warning


def test_evaluation_lesson_asks_for_a_run_when_there_is_none() -> None:
    app = open_lessons(page="evaluation")
    assert any("Train in the lab" in item.value for item in app.info)


def test_a_model_without_conv1_falls_back_to_the_random_filters() -> None:
    lab = snapshot(FakeRun(LearningMode.NOVEL, (1,)))
    del lab.trained.model.conv1  # type: ignore[union-attr]
    app = open_lessons(lab, "convolutions")
    assert not app.exception
    assert any(text.startswith("No trained model yet") for text in captions(app))


def test_batch_size_that_cannot_run_is_reported_in_the_epoch_lesson() -> None:
    app = open_lessons(snapshot(batch_size=1), "dynamics")
    assert not app.exception
    assert app.error


def test_next_previous_and_lab_buttons_move_between_pages() -> None:
    app = open_lessons()
    assert app.session_state[PAGE_KEY] == SECTIONS[0].key
    assert app.button(key="nav-prev").disabled

    app.button(key="nav-next").click().run()
    assert app.session_state[PAGE_KEY] == SECTIONS[1].key
    app.button(key="nav-prev").click().run()
    assert app.session_state[PAGE_KEY] == SECTIONS[0].key

    app.button(key=f"nav-{SECTIONS[-1].key}").click().run()
    assert app.session_state[PAGE_KEY] == SECTIONS[-1].key
    assert app.button(key="nav-next").disabled

    app.button(key="nav-lab").click().run()
    assert app.session_state[PAGE_KEY] == LAB_PAGE
    assert not app.subheader
    assert any(SECTIONS[-1].title in text for text in captions(app))

    app.button(key="nav-back").click().run()
    assert app.session_state[PAGE_KEY] == SECTIONS[-1].key


def test_lab_page_before_any_lesson_has_no_back_button() -> None:
    app = open_lessons(page=LAB_PAGE)
    assert not app.exception
    assert "Training lab. The lesson buttons are above." in captions(app)
    assert not [button for button in app.button if button.key == "nav-back"]


def test_an_unknown_page_opens_the_first_lesson() -> None:
    app = open_lessons(page="missing")
    assert app.subheader[0].value.startswith(f"{SECTIONS[0].number} of")


def test_progress_counts_opened_sections() -> None:
    app = open_lessons()
    assert app.progress[0].value == int(100 / len(SECTIONS))
    app.button(key="nav-next").click().run()
    assert app.progress[0].value == int(200 / len(SECTIONS))


def test_exercise_checkbox_persists_across_lessons() -> None:
    app = open_lessons()
    first, second = SECTIONS[0].key, SECTIONS[1].key
    total = sum(len(section.exercise_items()) for section in SECTIONS)
    assert f"0 of {total} exercises marked done." in captions(app)

    app.checkbox(key=f"done-{first}").check().run()
    app.button(key="nav-next").click().run()
    assert app.session_state[PAGE_KEY] == second
    assert app.session_state[DONE_KEY] == {first}
    assert f"1 of {total} exercises marked done." in captions(app)
    assert any(button.label.startswith("✓ ") for button in app.button)

    app.checkbox(key=f"done-{second}").check().run()
    app.button(key="nav-prev").click().run()
    assert app.checkbox(key=f"done-{first}").value is True
    assert app.session_state[DONE_KEY] == {first, second}

    app.checkbox(key=f"done-{first}").uncheck().run()
    app.button(key="nav-next").click().run()
    assert app.session_state[DONE_KEY] == {second}
    assert app.checkbox(key=f"done-{second}").value is True


def test_exercise_answer_is_inside_an_expander() -> None:
    app = open_lessons()
    assert [item.label for item in app.expander] == ["Show the check"]


def test_pixel_slider_changes_the_reported_pixel() -> None:
    app = open_lessons(page="images")
    before = [text for text in captions(app) if text.startswith("This pixel is row")]
    app.slider(key="norm-pixel").set_value(63).run()
    after = [text for text in captions(app) if text.startswith("This pixel is row")]
    assert not app.exception
    assert before and after and before != after


def test_neuron_bias_changes_the_output() -> None:
    app = open_lessons(page="neuron")
    first = [text for text in captions(app) if "Products:" in text]
    app.slider(key="neuron-bias").set_value(1.0).run()
    second = [text for text in captions(app) if "Products:" in text]
    assert not app.exception
    assert first != second


def test_softmax_shares_sum_to_one_after_moving_a_score() -> None:
    app = open_lessons(page="neuron")
    app.slider(key="softmax-0").set_value(-3.0).run()
    assert not app.exception
    assert any(text.endswith("Sum 1.000.") for text in captions(app))


def test_toy_step_and_convolution_sliders_rerender() -> None:
    descent = open_lessons(page="learning")
    descent.select_slider(key="toy-step").set_value(0.5).run()
    assert not descent.exception

    conv = open_lessons(page="convolutions")
    conv.slider(key="conv-row").set_value(0)
    conv.slider(key="conv-col").set_value(20).run()
    assert not conv.exception


def test_hardware_lessons_list_devices_and_runs() -> None:
    app = AppTest.from_function(curriculum_script, args=(snapshot(),), default_timeout=TIMEOUT)
    app.session_state[PAGE_KEY] = "hardware"
    app.session_state[RUN_LOG_KEY] = (record(), record("Transfer", peak=2048))
    app.run()
    assert not app.exception
    assert any(text.startswith("CUDA peak is the allocator") for text in captions(app))
    assert any("CUDA peak" in table.value.columns for table in app.table)


def test_run_log_is_empty_before_the_first_run() -> None:
    app = AppTest.from_function(run_log_script, default_timeout=TIMEOUT).run()
    assert not app.exception
    assert captions(app)[0].startswith("No runs yet")
    assert not app.table


def test_run_log_shows_settings_results_and_timing() -> None:
    app = AppTest.from_function(run_log_script, default_timeout=TIMEOUT)
    app.session_state[RUN_LOG_KEY] = (record(), record("Transfer"), "not a record")
    app.run()
    assert not app.exception
    runs, timing = (table.value for table in app.table)
    assert list(runs["Mode"]) == ["Novel", "Transfer"]
    assert list(runs["Val acc."]) == ["75%", "75%"]
    assert list(runs["Augment"]) == ["on", "on"]
    assert list(timing["Train s/epoch"]) == ["0.25", "0.25"]
    assert list(timing["Images/s"]) == ["—", "—"]
    assert "CUDA peak" not in timing.columns


def test_run_log_shows_the_cuda_peak_when_any_run_has_one() -> None:
    app = AppTest.from_function(run_log_script, default_timeout=TIMEOUT)
    app.session_state[RUN_LOG_KEY] = (record(), record(peak=3 * 1024 * 1024))
    app.run()
    timing = app.table[1].value
    assert list(timing["CUDA peak"])[0] == "—"
    assert list(timing["CUDA peak"])[1] != "—"
    assert list(timing["Images/s"]) == ["—", "40.0"]


def test_run_log_keeps_only_the_latest_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[str, object] = {}
    monkeypatch.setattr(lesson_ui.st, "session_state", store)
    for _ in range(lesson_ui._RUN_LIMIT + 3):
        lesson_ui.append_run(record())
    assert len(lesson_ui.run_log()) == lesson_ui._RUN_LIMIT


def test_seconds_and_rate_formatting() -> None:
    assert lesson_ui._seconds(0) == "—"
    assert lesson_ui._seconds(1.234) == "1.23"
    assert lesson_ui._seconds(12.34) == "12.3"
    assert lesson_ui._rate(None) == "—"
    assert lesson_ui._rate(3.14159) == "3.1"


def test_groups_skip_empty_parts() -> None:
    labels, values = lesson_ui._groups([("conv1", 5), ("fc", 0)], {})
    assert len(labels) == 1
    assert values == [5]
