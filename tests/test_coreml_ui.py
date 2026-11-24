"""The export controls for a finished run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from torch import nn

from image_classifier_lab.coreml import Export, Timing, Verification
from image_classifier_lab.examples import make_shape_dataset
from image_classifier_lab.hardware import detect_hardware
from image_classifier_lab.lesson_ui import DONE_KEY, RUN_LOG_KEY
from image_classifier_lab.measure import make_run_record
from image_classifier_lab.model import LearningMode, count_parameters, create_model
from image_classifier_lab.training import EpochMetrics, TrainResult
from image_classifier_lab.ui import Artifact

ROOT = Path(__file__).resolve().parents[1]


def _artifact(model: nn.Module) -> Artifact:
    trainable, total = count_parameters(model)
    return Artifact(
        model=model,
        class_names=("circle", "square", "triangle"),
        result=TrainResult(
            history=(EpochMetrics(1, 1.0, 0.5, 1.0, 0.5),),
            confusion=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            per_class_accuracy=(1.0, 1.0, 1.0),
            trainable_parameters=trainable,
            total_parameters=total,
        ),
        device="cpu",
        signature=("test-run",),
        summary=(),
        mode=LearningMode.NOVEL,
        train_size=3,
        val_size=3,
        details="Test run",
        val_images=tuple(group.images[0] for group in make_shape_dataset(images_per_class=2)),
    )


@pytest.mark.skipif(sys.platform == "darwin", reason="This checks the non-macOS message")
def test_export_from_finished_run() -> None:
    app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60)
    app.session_state["artifact"] = _artifact(create_model(3, LearningMode.NOVEL, weights=None))
    app.run()
    assert not app.exception
    next(button for button in app.button if button.label == "Convert to Core ML").click().run()
    assert not app.exception
    assert not app.error
    assert any("Package size (float32)" in item.value for item in app.markdown)
    assert any("Core ML runs on macOS only" in item.value for item in app.info)
    assert len(app.get("download_button")) == 1
    exported = next(iter(app.session_state["coreml_exports"].values()))
    app.run()
    assert not app.exception
    assert next(iter(app.session_state["coreml_exports"].values())) is exported
    app.run()
    assert not app.exception
    assert any("Measured float32 package" in item.value for item in app.markdown)
    assert not app.radio(key="coreml_precision").disabled


def _hardware_app(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    *,
    cuda: bool = False,
    mps: bool = False,
    trained: bool = False,
    coreml_reason: str | None = None,
) -> AppTest:
    options = detect_hardware(
        system=system,
        machine=machine,
        macos_version="14.0",
        cuda_available=lambda: cuda,
        mps_available=lambda: mps,
        coreml_runtime_reason=lambda: coreml_reason,
    )
    monkeypatch.setattr("image_classifier_lab.ui.detect_hardware", lambda: options)
    app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60)
    if trained:
        app.session_state["artifact"] = _artifact(nn.Identity())
    return app.run()


def test_plain_linux_panel_shows_disabled_hardware_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "linux", "x86_64")
    assert not app.exception
    assert "Hardware" in [item.value for item in app.sidebar.subheader]
    training = next(item for item in app.sidebar.radio if item.label == "Training")
    assert training.options == ["CPU"]
    assert not training.help
    unavailable_training_gpu = app.sidebar.radio(key="training-gpu-unavailable")
    assert unavailable_training_gpu.disabled and unavailable_training_gpu.value is None
    assert unavailable_training_gpu.options == ["GPU"]
    gpu = app.sidebar.checkbox(key="inference-gpu")
    assert gpu.disabled and not gpu.value
    cpu = app.sidebar.checkbox(key="inference-cpu")
    assert not cpu.disabled and cpu.value
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert neural_engine.disabled and not neural_engine.value
    assert not neural_engine.help
    assert [
        item.label
        for item in app.sidebar.checkbox
        if item.key in {"inference-cpu", "inference-CPU_AND_NE", "inference-gpu"}
    ] == ["CPU", "Neural Engine", "GPU"]
    captions = {item.value for item in app.sidebar.caption}
    assert not any("Inference benchmarks ·" in caption for caption in captions)
    assert not app.sidebar.get("popover")
    assert not {
        "inference-CPU_ONLY",
        "inference-CPU_AND_GPU",
        "inference-ALL",
    } & {item.key for item in app.sidebar.checkbox}
    assert app.sidebar.checkbox(key="inference-cpu").label == "CPU"
    assert app.sidebar.checkbox(key="inference-gpu").label == "GPU"
    precision = app.sidebar.radio(key="coreml_precision")
    assert precision.label == "Precision"
    assert precision.options == ["float32", "float16"] and precision.value == "float32"
    assert not precision.help
    assert not precision.proto.horizontal
    markdown = [item.value for item in app.sidebar.markdown]
    assert not any("hardware-unavailable" in text or "hardware-help" in text for text in markdown)


def test_neural_engine_explains_unavailable_coreml(monkeypatch: pytest.MonkeyPatch) -> None:
    reason = "Core ML export requires coremltools. Install the app dependencies."
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True, coreml_reason=reason)
    assert not app.exception
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert neural_engine.disabled and not neural_engine.value
    assert app.sidebar.checkbox(key="inference-cpu").value


def test_intel_mac_shows_disabled_neural_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "darwin", "x86_64", mps=True)
    assert not app.exception
    assert app.sidebar.checkbox(key="inference-cpu").value
    assert app.sidebar.checkbox(key="inference-mps").value
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert neural_engine.disabled


def test_gpu_becoming_unavailable_disables_both_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "linux", "x86_64", cuda=True)
    training = next(item for item in app.sidebar.radio if item.label == "Training")
    training.set_value("cuda").run()
    assert app.sidebar.checkbox(key="inference-cuda").value

    unavailable = detect_hardware(
        system="linux",
        machine="x86_64",
        cuda_available=lambda: False,
        mps_available=lambda: False,
    )
    monkeypatch.setattr("image_classifier_lab.ui.detect_hardware", lambda: unavailable)
    app.run()

    assert not app.exception
    assert next(item for item in app.sidebar.radio if item.label == "Training").value == "cpu"
    assert app.sidebar.radio(key="training-gpu-unavailable").disabled
    assert app.sidebar.checkbox(key="inference-gpu").disabled


def test_selected_backends_and_precision_drive_lesson_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True, trained=True)
    calls = []

    def convert(model, names, images, precision):
        calls.append(("convert", precision))
        verification = Verification(0 if precision == "float32" else 0.001, 1.0, len(images))
        return Export(b"zip", 128 if precision == "float32" else 64, precision, verification)

    def measure(export, model, image, *, backends):
        calls.append(("measure", export.precision, backends))
        return tuple(
            Timing(f"PyTorch {unit}", 1.0)
            if unit == "cpu"
            else Timing("Core ML CPU_AND_NE", None, "Runtime failure")
            for unit in backends
        )

    monkeypatch.setattr("image_classifier_lab.ui.export_model", convert)
    monkeypatch.setattr("image_classifier_lab.lesson_ui.benchmark_export", measure)
    assert not app.exception
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert not neural_engine.disabled and neural_engine.value
    training = next(item for item in app.sidebar.radio if item.label == "Training")
    assert training.options == ["CPU", "GPU"]
    assert app.sidebar.checkbox(key="inference-mps").label == "GPU"
    assert not app.sidebar.checkbox(key="inference-mps").disabled
    assert not any(item.key == "training-gpu-unavailable" for item in app.sidebar.radio)
    training.set_value("mps")
    app.sidebar.checkbox(key="inference-mps").uncheck()
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    assert any("Training device in the sidebar: GPU" in item.value for item in app.markdown)
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert not app.exception
    assert calls == [("convert", "float32"), ("measure", "float32", ("cpu", "CPU_AND_NE"))]
    table = next(item.value for item in app.table if "Median ms" in item.value)
    assert table["Device"].tolist() == ["PyTorch cpu", "Core ML CPU_AND_NE"]
    assert table["Error"].tolist() == ["", "Runtime failure"]
    app.sidebar.checkbox(key="inference-CPU_AND_NE").uncheck().run()
    assert not any("Median ms" in item.value for item in app.table)
    app.sidebar.checkbox(key="inference-CPU_AND_NE").check()
    app.sidebar.radio(key="coreml_precision").set_value("float16").run()
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert not app.exception
    assert calls[-2:] == [("convert", "float16"), ("measure", "float16", ("cpu", "CPU_AND_NE"))]
    table = next(item.value for item in app.table if "Median ms" in item.value)
    assert table["Device"].tolist() == ["PyTorch cpu", "Core ML CPU_AND_NE"]
    assert any("Measured float16 package: 64 bytes" in item.value for item in app.markdown)
    assert any(
        "top-1 agreement 100.0%; max absolute probability difference 0.001" in item.value
        for item in app.markdown
    )


def test_selected_neural_engine_does_not_block_pytorch_timing_without_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True, trained=True)
    calls = []

    def measure(export, model, image, *, backends):
        calls.append((export, backends))
        return tuple(Timing(f"PyTorch {backend}", 1.0) for backend in backends)

    monkeypatch.setattr("image_classifier_lab.lesson_ui.benchmark_export", measure)
    assert any("include the Neural Engine benchmark" in item.value for item in app.info)
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert calls == [(None, ("cpu", "mps"))]
    table = next(item.value for item in app.table if "Median ms" in item.value)
    assert table["Device"].tolist() == ["PyTorch cpu", "PyTorch mps"]

    app.sidebar.checkbox(key="inference-cpu").uncheck()
    app.sidebar.checkbox(key="inference-mps").uncheck().run()
    assert not any(item.label == "Measure on-device inference" for item in app.button)
    assert any("Select an inference backend" in item.value for item in app.info)
    assert app.sidebar.checkbox(key="inference-CPU_AND_NE").disabled


def test_cpu_selection_controls_neural_engine_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True, trained=True)
    calls = []

    def measure(export, model, image, *, backends):
        calls.append((export, backends))
        return (Timing("PyTorch GPU", 1.0),)

    monkeypatch.setattr("image_classifier_lab.lesson_ui.benchmark_export", measure)
    assert app.sidebar.checkbox(key="inference-CPU_AND_NE").value
    app.sidebar.checkbox(key="inference-cpu").uncheck().run()
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert neural_engine.disabled
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert calls == [(None, ("mps",))]

    app.sidebar.checkbox(key="inference-cpu").check().run()
    neural_engine = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert not neural_engine.disabled and neural_engine.value
    assert any("include the Neural Engine benchmark" in item.value for item in app.info)


def test_pytorch_timing_without_export_and_empty_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "linux", "x86_64", trained=True)

    def measure(export, model, image, *, backends):
        assert export is None and backends == ("cpu",)
        raise RuntimeError("Timer failed")

    monkeypatch.setattr("image_classifier_lab.lesson_ui.benchmark_export", measure)
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert any("Timer failed" in item.value for item in app.error)
    app.sidebar.checkbox(key="inference-cpu").uncheck().run()
    assert any("Select an inference backend" in item.value for item in app.info)
    assert not any(item.label == "Measure on-device inference" for item in app.button)


def test_conversion_and_verification_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True, trained=True)

    def convert(model, names, images, precision):
        raise RuntimeError("Conversion failed")

    monkeypatch.setattr("image_classifier_lab.ui.export_model", convert)
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    assert any("Conversion failed" in item.value for item in app.error)
    assert not app.get("download_button")
    monkeypatch.setattr(
        "image_classifier_lab.ui.export_model",
        lambda *args: Export(b"zip", 128, "float32", None, "Runtime failed"),
    )
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    assert len(app.get("download_button")) == 1
    assert any(e.value == "Could not verify the Core ML model: Runtime failed" for e in app.error)


def test_hardware_exercises_persist_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "darwin", "arm64", mps=True)
    ne = app.sidebar.checkbox(key="inference-CPU_AND_NE")
    assert not ne.disabled and ne.value
    app.session_state[RUN_LOG_KEY] = (
        make_run_record(
            _artifact(nn.Identity()).result,
            mode="Novel",
            learning_rate=0.001,
            batch_size=4,
            epochs=1,
            augment=True,
            device_label="CPU",
            seed=0,
        ),
    )
    app.run()
    assert {"Batch", "Device", "Train s/epoch"} <= set(app.table[-1].value.columns)
    for key in ("hardware-neural-engine", "hardware-all"):
        app.checkbox(key=f"done-{key}").check()
    app.run()
    app.slider(key="norm-pixel").set_value(63).run()
    assert app.checkbox(key="done-hardware-neural-engine").value
    assert app.checkbox(key="done-hardware-all").value
    assert not app.checkbox(key="done-hardware").value
    assert not app.checkbox(key="done-hardware-precision").value
    app.checkbox(key="done-hardware-all").uncheck().run()
    assert app.session_state[DONE_KEY] == {"hardware-neural-engine"}
    assert any(item.value == "1 of 11 exercises marked done." for item in app.caption)
