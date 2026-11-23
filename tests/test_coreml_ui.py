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
    app.button(key="nav-hardware").click().run()
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
) -> AppTest:
    options = detect_hardware(
        system=system,
        machine=machine,
        macos_version="14.0",
        cuda_available=lambda: cuda,
        mps_available=lambda: mps,
        coreml_runtime_reason=lambda: None,
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
    assert next(item for item in app.sidebar.radio if item.label == "Training device").options == [
        "CPU"
    ]
    for key in (
        "training-cuda",
        "training-mps",
        "inference-gpu",
        "inference-CPU_ONLY",
        "inference-CPU_AND_GPU",
        "inference-CPU_AND_NE",
        "inference-ALL",
    ):
        checkbox = app.sidebar.checkbox(key=key)
        assert checkbox.disabled and not checkbox.value
    cpu = app.sidebar.checkbox(key="inference-cpu")
    assert not cpu.disabled and cpu.value
    captions = {item.value for item in app.sidebar.caption}
    assert "No CUDA GPU found." in captions
    assert "Metal (MPS) needs a Mac with a usable Metal GPU." in captions
    assert "No CUDA or Metal GPU for PyTorch GPU." in captions
    assert "Core ML runs on macOS only." in captions
    assert "Neural Engine needs Apple silicon." in captions
    assert "Neural Engine: cannot train. Core ML runs it for inference only." in captions
    assert app.sidebar.radio(key="coreml_precision").options == ["float32", "float16"]
    assert app.sidebar.radio(key="coreml_precision").value == "float32"


def test_selected_backends_and_precision_drive_lesson_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _hardware_app(monkeypatch, "darwin", "x86_64", mps=True, trained=True)
    calls = []

    def convert(model, names, images, precision):
        calls.append(("convert", precision))
        verification = Verification(0 if precision == "float32" else 0.001, 1.0, len(images))
        return Export(b"zip", 128 if precision == "float32" else 64, precision, verification)

    def measure(export, model, image, *, backends):
        calls.append(("measure", export.precision, backends))
        return tuple(
            Timing(f"Core ML {unit}", 1.0)
            if unit != "ALL"
            else Timing("Core ML ALL", None, "Runtime failure")
            for unit in backends
        )

    monkeypatch.setattr("image_classifier_lab.ui.export_model", convert)
    monkeypatch.setattr("image_classifier_lab.lesson_ui.benchmark_export", measure)
    app.button(key="nav-hardware").click().run()
    assert not app.exception
    assert not any(item.label == "Measure on-device inference" for item in app.button)
    assert app.sidebar.checkbox(key="inference-CPU_AND_NE").disabled
    assert "Neural Engine needs Apple silicon." in {item.value for item in app.sidebar.caption}
    training = next(item for item in app.sidebar.radio if item.label == "Training device")
    assert training.options == ["CPU", "Apple GPU (Metal/MPS)"]
    training.set_value("mps")
    for key in ("inference-cpu", "inference-mps", "inference-CPU_AND_GPU"):
        assert app.sidebar.checkbox(key=key).value
        app.sidebar.checkbox(key=key).uncheck()
    for key in ("inference-CPU_ONLY", "inference-ALL"):
        assert app.sidebar.checkbox(key=key).value
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    assert any("Training device in the sidebar: Apple GPU" in item.value for item in app.markdown)
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert not app.exception
    assert calls == [("convert", "float32"), ("measure", "float32", ("CPU_ONLY", "ALL"))]
    table = next(item.value for item in app.table if "Median ms" in item.value)
    assert table["Device"].tolist() == ["Core ML CPU_ONLY", "Core ML ALL"]
    assert table["Error"].tolist() == ["", "Runtime failure"]
    app.sidebar.checkbox(key="inference-ALL").uncheck().run()
    assert not any("Median ms" in item.value for item in app.table)
    app.sidebar.radio(key="coreml_precision").set_value("float16").run()
    next(item for item in app.button if item.label == "Convert to Core ML").click().run()
    next(item for item in app.button if item.label == "Measure on-device inference").click().run()
    assert not app.exception
    assert calls[-2:] == [("convert", "float16"), ("measure", "float16", ("CPU_ONLY",))]
    table = next(item.value for item in app.table if "Median ms" in item.value)
    assert table["Device"].tolist() == ["Core ML CPU_ONLY"]
    assert any("Measured float16 package: 64 bytes" in item.value for item in app.markdown)
    assert any(
        "top-1 agreement 100.0%; max absolute probability difference 0.001" in item.value
        for item in app.markdown
    )


def test_pytorch_timing_without_export_and_empty_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _hardware_app(monkeypatch, "linux", "x86_64", trained=True)
    app.button(key="nav-hardware").click().run()

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
    app.button(key="nav-hardware").click().run()
    assert {"Batch", "Device", "Train s/epoch"} <= set(app.table[-1].value.columns)
    for key in ("hardware-neural-engine", "hardware-all"):
        app.checkbox(key=f"done-{key}").check()
    app.run()
    app.button(key="nav-lab").click().run()
    app.button(key="nav-back").click().run()
    assert app.checkbox(key="done-hardware-neural-engine").value
    assert app.checkbox(key="done-hardware-all").value
    assert not app.checkbox(key="done-hardware").value
    assert not app.checkbox(key="done-hardware-precision").value
    app.checkbox(key="done-hardware-all").uncheck().run()
    assert app.session_state[DONE_KEY] == {"hardware-neural-engine"}
    assert any(item.value == "1 of 11 exercises marked done." for item in app.caption)
