"""Core ML export, verification, and batch-one inference measurements."""

from __future__ import annotations

import copy
import platform
import statistics
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import nn

from image_classifier_lab.data import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    image_to_model_input,
    prepare_image,
)

Precision = Literal["float32", "float16"]
COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")
PYTORCH_BACKENDS = {"cpu": "CPU", "cuda": "GPU (CUDA)", "mps": "GPU (MPS)"}
WARMUP_RUNS = 3
MEASURE_RUNS = 10
PACKAGE_NAME = "image-classifier-lab.mlpackage"


@dataclass(frozen=True)
class Verification:
    """Comparison of the class probabilities on the validation images."""

    max_abs_difference: float
    top1_agreement: float
    image_count: int


@dataclass(frozen=True)
class Export:
    """A downloadable package and its uncompressed on-disk size."""

    zip_bytes: bytes
    size_bytes: int
    precision: Precision
    verification: Verification | None
    verification_error: str | None = None


@dataclass(frozen=True)
class Timing:
    """Median batch-one latency, or the reason a device was unavailable."""

    name: str
    median_ms: float | None
    error: str | None = None


def _coremltools():
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError(
            "Core ML export requires coremltools. Install the app dependencies."
        ) from exc
    return ct


def _coreml_runtime_reason() -> str | None:
    try:
        _coremltools()
        from coremltools.models.model import _MLModelProxy
    except (ImportError, RuntimeError) as exc:
        return f"Core ML predictions are unavailable: {exc}"
    if _MLModelProxy is None:
        return "Core ML predictions are unavailable: the Core ML runtime could not be loaded."
    return None


def coreml_prediction_reason(
    *,
    system: str | None = None,
    macos_version: str | None = None,
    runtime_reason: Callable[[], str | None] | None = None,
) -> str | None:
    """Why package prediction is unavailable, or ``None`` when it can run."""
    if (sys.platform if system is None else system) != "darwin":
        return "Core ML runs on macOS only."
    version = platform.mac_ver()[0] if macos_version is None else macos_version
    major = version.partition(".")[0]
    if not major.isdecimal() or int(major) < 12:
        return "Core ML predictions need macOS 12 or later."
    return (runtime_reason or _coreml_runtime_reason)()


def coreml_prediction_available() -> bool:
    """Whether this machine can run predictions for the exported packages."""
    return coreml_prediction_reason() is None


class _ImageClassifier(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model((image - self.mean) / self.std), dim=1)


def package_size(path: Path) -> int:
    """Total bytes of all files in a saved .mlpackage, including weights."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _zip_package(path: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(path.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(path.parent))
    return buffer.getvalue()


def convert_model(
    model: nn.Module,
    class_names: Sequence[str],
    path: Path,
    precision: Precision,
) -> None:
    """Save a classifier package without changing the model being trained."""
    if precision not in ("float32", "float16"):
        raise ValueError("Precision must be float32 or float16.")
    if len(class_names) < 2 or len(set(class_names)) != len(class_names):
        raise ValueError("Provide at least two distinct class labels.")
    if path.suffix != ".mlpackage":
        raise ValueError("The package path must end in .mlpackage.")
    ct = _coremltools()
    wrapped = _ImageClassifier(copy.deepcopy(model).cpu().float().eval()).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    converted = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                color_layout=ct.colorlayout.RGB,
                scale=1 / 255,
            )
        ],
        outputs=[ct.TensorType(name="class_probs")],
        classifier_config=ct.ClassifierConfig(
            list(class_names), predicted_probabilities_output="class_probs"
        ),
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32,
    )
    converted.save(str(path))


def verify_model(
    path: Path, model: nn.Module, class_names: Sequence[str], images: Sequence[Image.Image]
) -> Verification:
    """Compare Core ML and float32 PyTorch on the same resized validation images."""
    reason = coreml_prediction_reason()
    if reason:
        raise RuntimeError(reason)
    if not images:
        raise ValueError("Verification needs at least one validation image.")
    ct = _coremltools()
    predictor = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
    probability_name = predictor.get_spec().description.predictedProbabilitiesName
    cpu_model = copy.deepcopy(model).cpu().float().eval()
    maximum = 0.0
    matches = 0
    with torch.no_grad():
        for image in images:
            expected = torch.softmax(cpu_model(image_to_model_input(image)), dim=1)[0].tolist()
            probabilities = predictor.predict({"image": prepare_image(image)})[probability_name]
            actual = [float(probabilities[name]) for name in class_names]
            maximum = max(maximum, *(abs(a - b) for a, b in zip(expected, actual, strict=True)))
            matches += int(
                max(range(len(expected)), key=expected.__getitem__)
                == max(range(len(actual)), key=actual.__getitem__)
            )
    return Verification(maximum, matches / len(images), len(images))


def export_model(
    model: nn.Module,
    class_names: Sequence[str],
    images: Sequence[Image.Image],
    precision: Precision,
) -> Export:
    """Convert, package, and, when possible, verify a trained model."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / PACKAGE_NAME
        convert_model(model, class_names, path, precision)
        verification = None
        error = None
        if coreml_prediction_available():
            try:
                verification = verify_model(path, model, class_names, images)
            except Exception as exc:
                error = str(exc)
        return Export(_zip_package(path), package_size(path), precision, verification, error)


def _median_latency(
    predict: Callable[[], object],
    synchronize: Callable[[], None],
    warmup: int,
    runs: int,
    clock: Callable[[], float],
) -> float:
    for _ in range(warmup):
        predict()
    synchronize()
    samples = []
    for _ in range(runs):
        synchronize()
        start = clock()
        predict()
        synchronize()
        samples.append((clock() - start) * 1000)
    return statistics.median(samples)


def benchmark_model(
    path: Path | None,
    model: nn.Module,
    image: Image.Image,
    *,
    compute_units: Sequence[str],
    devices: Sequence[tuple[str, str]],
    warmup: int = WARMUP_RUNS,
    runs: int = MEASURE_RUNS,
    clock: Callable[[], float] = time.perf_counter,
    predictor_factory: Callable[[Path, str], Callable[[Image.Image], object]] | None = None,
) -> tuple[Timing, ...]:
    """Measure Core ML compute units and PyTorch devices on the same image."""
    if warmup < 0 or runs < 1:
        raise ValueError("Warmup must be non-negative and runs must be positive.")
    results: list[Timing] = []
    if compute_units:
        if path is None:
            raise ValueError("Core ML timings need an exported package.")
        if predictor_factory is None:
            reason = coreml_prediction_reason()
            if reason:
                raise RuntimeError(reason)
            ct = _coremltools()

            def predictor_factory(path: Path, unit: str) -> Callable[[Image.Image], object]:
                predictor = ct.models.MLModel(
                    str(path), compute_units=getattr(ct.ComputeUnit, unit)
                )
                return lambda image: predictor.predict({"image": image})

        prepared = prepare_image(image)
        for unit in compute_units:
            try:
                predict = predictor_factory(path, unit)
                latency = _median_latency(
                    lambda predict=predict: predict(prepared), lambda: None, warmup, runs, clock
                )
                results.append(Timing(f"Core ML {unit}", latency))
            except Exception as exc:
                results.append(Timing(f"Core ML {unit}", None, str(exc)))

    if not devices:
        return tuple(results)
    batch = image_to_model_input(image)
    for label, device in devices:
        try:
            device_model = copy.deepcopy(model).to(device=device, dtype=torch.float32).eval()
            device_batch = batch.to(device)

            def synchronize(device: str = device) -> None:
                if device == "cuda":
                    torch.cuda.synchronize()
                elif device == "mps":
                    torch.mps.synchronize()

            with torch.no_grad():
                latency = _median_latency(
                    lambda model=device_model, batch=device_batch: model(batch),
                    synchronize,
                    warmup,
                    runs,
                    clock,
                )
            results.append(Timing(f"PyTorch {label}", latency))
        except Exception as exc:
            results.append(Timing(f"PyTorch {label}", None, str(exc)))
    return tuple(results)


def benchmark_export(
    export: Export | None,
    model: nn.Module,
    image: Image.Image,
    *,
    backends: Sequence[str],
    warmup: int = WARMUP_RUNS,
    runs: int = MEASURE_RUNS,
) -> tuple[Timing, ...]:
    """Time Core ML unit keys and float32 PyTorch device keys (cpu, cuda, mps).

    PyTorch-only selections do not need an export.
    """
    units = tuple(key for key in backends if key in COMPUTE_UNITS)
    devices = tuple((PYTORCH_BACKENDS[key], key) for key in backends if key not in COMPUTE_UNITS)
    if not units:
        return benchmark_model(
            None, model, image, compute_units=units, devices=devices, warmup=warmup, runs=runs
        )
    if export is None:
        raise ValueError("Core ML timings need an exported package.")
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(BytesIO(export.zip_bytes)) as archive:
            root = Path(directory).resolve()
            for name in archive.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise ValueError("The package archive contains an invalid path.")
            archive.extractall(directory)
        return benchmark_model(
            Path(directory) / PACKAGE_NAME,
            model,
            image,
            compute_units=units,
            devices=devices,
            warmup=warmup,
            runs=runs,
        )
