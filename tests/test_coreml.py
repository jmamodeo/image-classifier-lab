"""Core ML packages and inference measurements."""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import coremltools as ct
import pytest
import torch
from PIL import Image
from torch import nn

from image_classifier_lab.coreml import (
    COMPUTE_UNITS,
    PACKAGE_NAME,
    Export,
    _ImageClassifier,
    benchmark_export,
    benchmark_model,
    convert_model,
    coreml_prediction_available,
    coreml_prediction_reason,
    export_model,
    package_size,
    verify_model,
)
from image_classifier_lab.data import (
    IMAGE_SIZE,
    image_to_model_input,
    prepare_image,
    to_normalized_tensor,
)
from image_classifier_lab.hardware import detect_hardware
from image_classifier_lab.model import LearningMode, create_model


def test_conversion_packages_labels_image_and_precisions(tmp_path: Path) -> None:
    model = create_model(3, LearningMode.NOVEL, weights=None)
    model.train()
    before = model.conv1.weight.detach().clone()
    labels = ("circle", "square", "triangle")
    sizes = {}
    for precision in ("float32", "float16"):
        exported = export_model(model, labels, (), precision)
        sizes[precision] = exported.size_bytes
        assert exported.size_bytes > 0
        with zipfile.ZipFile(BytesIO(exported.zip_bytes)) as archive:
            assert archive.testzip() is None
            assert any(name.startswith(f"{PACKAGE_NAME}/Data/") for name in archive.namelist())
            archive.extractall(tmp_path / precision)
        path = tmp_path / precision / PACKAGE_NAME
        assert package_size(path) == exported.size_bytes
        spec = ct.utils.load_spec(str(path))
        assert spec.WhichOneof("Type") == "mlProgram"
        image = spec.description.input[0]
        assert image.name == "image"
        assert image.type.imageType.width == image.type.imageType.height == IMAGE_SIZE
        assert image.type.imageType.colorSpace == image.type.imageType.RGB
        assert spec.description.predictedProbabilitiesName
        function = spec.mlProgram.functions["main"]
        block = next(iter(function.block_specializations.values()))
        classifier = next(op for op in block.operations if op.type == "classify")
        values = classifier.inputs["classes"].arguments[0].value.immediateValue.list.values
        assert tuple(value.immediateValue.tensor.strings.values[0] for value in values) == labels
    assert sizes["float16"] < sizes["float32"]
    assert model.training
    assert torch.equal(model.conv1.weight, before)


def test_benchmark_warmup_median_and_failure() -> None:
    calls: dict[str, int] = {}
    clock_values = iter((0.0, 0.003, 1.0, 1.001, 2.0, 2.002) * 3)

    def factory(_path: Path, unit: str):
        if unit == "CPU_AND_GPU":
            raise RuntimeError("device unavailable")

        def predict(image: Image.Image) -> None:
            assert image.mode == "RGB" and image.size == (IMAGE_SIZE, IMAGE_SIZE)
            calls[unit] = calls.get(unit, 0) + 1
            if unit == "CPU_AND_NE":
                raise RuntimeError("prediction failed")

        return predict

    results = benchmark_model(
        Path("unused.mlpackage"),
        nn.Identity(),
        Image.new("L", (8, 12)),
        warmup=2,
        runs=3,
        clock=lambda: next(clock_values),
        predictor_factory=factory,
        compute_units=COMPUTE_UNITS,
        devices=(("CPU", "cpu"),),
    )
    assert [item.name for item in results] == [
        *(f"Core ML {unit}" for unit in COMPUTE_UNITS),
        "PyTorch CPU",
    ]
    assert results[0].median_ms == pytest.approx(2)
    assert results[1].median_ms is None
    assert results[1].error == "device unavailable"
    assert results[2].median_ms is None
    assert results[2].error == "prediction failed"
    assert results[-1].median_ms == pytest.approx(2)
    assert calls == {"CPU_ONLY": 5, "CPU_AND_NE": 1, "ALL": 5}


def test_prediction_unavailable_off_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert not coreml_prediction_available()
    reason = coreml_prediction_reason()
    assert reason is not None
    assert reason == "Core ML runs on macOS only."


@pytest.mark.parametrize("proxy", [None, object()])
def test_coreml_runtime_probe(monkeypatch: pytest.MonkeyPatch, proxy: object) -> None:
    monkeypatch.setattr(ct.models.model, "_MLModelProxy", proxy)
    reason = coreml_prediction_reason(system="darwin", macos_version="14.0")
    if proxy is None:
        assert reason is not None and "runtime could not be loaded" in reason
    else:
        assert reason is None


def test_missing_coremltools_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "coremltools", None)
    reason = coreml_prediction_reason(system="darwin", macos_version="14.0")
    assert reason is not None and "Install the app dependencies" in reason


@pytest.mark.parametrize(
    ("labels", "package", "precision", "message"),
    [
        (("a", "b"), "model.mlpackage", "bad", "Precision"),
        (("a", "a"), "model.mlpackage", "float32", "distinct class labels"),
        (("a", "b"), "model.mlmodel", "float32", "must end in .mlpackage"),
    ],
)
def test_conversion_rejects_invalid_package_inputs(labels, package, precision, message) -> None:
    with pytest.raises(ValueError, match=message):
        convert_model(nn.Identity(), labels, Path(package), precision)


def test_export_preserves_package_when_verification_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def convert(model, names, path, precision):
        path.mkdir()
        (path / "weights").write_bytes(b"weights")

    def verify(path, model, names, images):
        raise RuntimeError("Could not compare the validation images")

    monkeypatch.setattr("image_classifier_lab.coreml.convert_model", convert)
    monkeypatch.setattr("image_classifier_lab.coreml.coreml_prediction_reason", lambda: None)
    monkeypatch.setattr("image_classifier_lab.coreml.verify_model", verify)
    exported = export_model(nn.Identity(), ("a", "b"), (), "float16")
    assert exported.verification is None
    assert exported.verification_error == "Could not compare the validation images"
    with zipfile.ZipFile(BytesIO(exported.zip_bytes)) as archive:
        assert archive.read(f"{PACKAGE_NAME}/weights") == b"weights"


def test_export_wrapper_matches_validation_normalization() -> None:
    image = Image.new("RGB", (37, 19))
    for y in range(image.height):
        for x in range(image.width):
            image.putpixel((x, y), (x * 6, y * 10, (x + y) * 4))

    class Model(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.mean(dim=(2, 3))

    prepared = prepare_image(image)
    pixels = torch.frombuffer(bytearray(prepared.tobytes()), dtype=torch.uint8).float()
    pixels = pixels.reshape(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).unsqueeze(0) / 255
    expected = torch.softmax(Model()(image_to_model_input(image)), dim=1)
    assert torch.allclose(_ImageClassifier(Model())(pixels), expected, atol=1e-7)


def test_verification_uses_identical_pixels_and_label_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = Image.new("RGB", (37, 19), (70, 100, 200))
    image.putpixel((3, 5), (10, 20, 30))
    expected_image = prepare_image(image)

    class Model(nn.Module):
        def forward(self, batch: torch.Tensor) -> torch.Tensor:
            return batch.mean(dim=(2, 3))[:, :2]

    class Predictor:
        def get_spec(self):
            return SimpleNamespace(
                description=SimpleNamespace(predictedProbabilitiesName="probabilities")
            )

        def predict(self, inputs):
            actual_image = inputs["image"]
            assert actual_image.tobytes() == expected_image.tobytes()
            normalized = to_normalized_tensor(actual_image).unsqueeze(0)
            assert torch.equal(normalized, image_to_model_input(image))
            values = torch.softmax(Model()(normalized), dim=1)[0].tolist()
            return {"probabilities": {"second": values[1], "first": values[0]}}

    fake = SimpleNamespace(
        models=SimpleNamespace(MLModel=lambda *args, **kwargs: Predictor()),
        ComputeUnit=SimpleNamespace(CPU_ONLY="cpu"),
    )
    monkeypatch.setattr("image_classifier_lab.coreml.coreml_prediction_reason", lambda: None)
    monkeypatch.setattr("image_classifier_lab.coreml._coremltools", lambda: fake)
    model = Model().train()
    result = verify_model(Path("unused.mlpackage"), model, ("first", "second"), (image,))
    assert result.max_abs_difference == 0
    assert result.top1_agreement == 1
    assert model.training


@pytest.mark.skipif(sys.platform != "darwin", reason="Core ML prediction requires macOS")
def test_macos_verification_and_benchmark(tmp_path: Path) -> None:
    model = create_model(2, LearningMode.NOVEL, weights=None)
    labels = ("circle", "square")
    path = tmp_path / PACKAGE_NAME
    convert_model(model, labels, path, "float32")
    images = [Image.new("RGB", (37, 19), color) for color in ("red", "blue")]
    verification = verify_model(path, model, labels, images)
    assert verification.image_count == 2
    assert verification.top1_agreement == 1.0
    assert verification.max_abs_difference < 1e-2
    units = tuple(
        option.key
        for option in detect_hardware().inference
        if option.available and option.key in COMPUTE_UNITS
    )
    timings = benchmark_model(
        path, model, images[0], compute_units=units, devices=(("CPU", "cpu"),), warmup=1, runs=2
    )
    assert timings[0].median_ms is not None
    assert any(item.name == "PyTorch CPU" and item.median_ms is not None for item in timings)


def test_benchmark_export_runs_only_selected_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(f"{PACKAGE_NAME}/marker", "package")
    exported = Export(archive.getvalue(), 0, "float16", None)
    units = []

    def load_model(path, *, compute_units):
        assert (Path(path) / "marker").read_text() == "package"
        units.append(compute_units)
        return SimpleNamespace(predict=lambda _: None)

    monkeypatch.setattr("image_classifier_lab.coreml.coreml_prediction_reason", lambda: None)
    monkeypatch.setattr(ct.models, "MLModel", load_model)
    timings = benchmark_export(
        exported,
        nn.Identity(),
        Image.new("RGB", (8, 8)),
        backends=("ALL",),
        warmup=0,
        runs=1,
    )
    assert units == [ct.ComputeUnit.ALL]
    assert [row.name for row in timings] == ["Core ML ALL"]
    assert timings[0].median_ms is not None


def test_benchmark_pytorch_without_an_export() -> None:
    model = nn.Identity()
    image = Image.new("RGB", (8, 8))
    timings = benchmark_export(None, model, image, backends=("cpu",), warmup=0, runs=1)
    assert [row.name for row in timings] == ["PyTorch CPU"]
    assert timings[0].median_ms is not None
    with pytest.raises(ValueError, match="need an exported package"):
        benchmark_export(None, model, image, backends=("CPU_ONLY",))
    with pytest.raises(ValueError, match="must be positive"):
        benchmark_export(None, model, image, backends=("cpu",), runs=0)


def test_benchmark_rejects_archive_entries_outside_package() -> None:
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("../outside", "")
    with pytest.raises(ValueError, match="invalid path"):
        benchmark_export(
            Export(archive.getvalue(), 0, "float32", None),
            nn.Identity(),
            Image.new("RGB", (8, 8)),
            backends=("CPU_ONLY",),
        )


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_pytorch_gpu_benchmark_synchronizes_selected_device(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    events = []

    class Model(nn.Identity):
        def to(self, *, device, dtype):
            events.append((device, dtype))
            return self

        def forward(self, batch):
            events.append(batch)
            return batch

    monkeypatch.setattr(f"torch.{device}.synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        "image_classifier_lab.coreml.image_to_model_input",
        lambda image: SimpleNamespace(to=lambda device: f"batch on {device}"),
    )
    result = benchmark_export(
        None, Model(), Image.new("RGB", (8, 8)), backends=(device,), warmup=1, runs=1
    )
    assert result[0].median_ms is not None
    assert events == [
        (device, torch.float32),
        f"batch on {device}",
        "sync",
        "sync",
        f"batch on {device}",
        "sync",
    ]
