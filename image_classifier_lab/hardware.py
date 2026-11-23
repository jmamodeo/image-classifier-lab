"""Availability of the training and inference options shown in the lab."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass

from image_classifier_lab.coreml import COMPUTE_UNITS, coreml_prediction_reason
from image_classifier_lab.training import TRAINING_DEVICE_LABELS, available_devices

NEURAL_ENGINE_REASON = "Neural Engine needs Apple silicon."


@dataclass(frozen=True)
class HardwareOption:
    key: str
    label: str
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class Hardware:
    training: tuple[HardwareOption, ...]
    inference: tuple[HardwareOption, ...]
    apple_silicon: bool


def detect_hardware(
    *,
    cuda_available: Callable[[], bool] | None = None,
    mps_available: Callable[[], bool] | None = None,
    system: str | None = None,
    machine: str | None = None,
    macos_version: str | None = None,
    coreml_runtime_reason: Callable[[], str | None] | None = None,
) -> Hardware:
    """Describe every option using runtime probes, without changing application state."""
    system = sys.platform if system is None else system
    machine = platform.machine() if machine is None else machine
    apple_silicon = system == "darwin" and machine == "arm64"
    detected = {
        device
        for _, device in available_devices(
            cuda_available=cuda_available, mps_available=mps_available
        )
    }
    training_reasons = {
        "cuda": "No CUDA GPU found.",
        "mps": "Metal (MPS) needs a Mac with a usable Metal GPU.",
    }
    training = tuple(
        HardwareOption(key, label, None if key in detected else training_reasons[key])
        for key, label in TRAINING_DEVICE_LABELS.items()
    )
    gpu = next((key for key in ("cuda", "mps") if key in detected), None)
    if system == "darwin" and macos_version is None:
        macos_version = platform.mac_ver()[0]
    coreml_reason = coreml_prediction_reason(
        system=system, macos_version=macos_version, runtime_reason=coreml_runtime_reason
    )
    ne_reason = coreml_reason
    if system == "darwin" and not apple_silicon:
        ne_reason = NEURAL_ENGINE_REASON
    elif (
        coreml_reason is None
        and macos_version is not None
        and int(macos_version.partition(".")[0]) < 13
    ):
        ne_reason = "Core ML CPU_AND_NE needs macOS 13 or later."
    inference = (
        HardwareOption("cpu", "PyTorch CPU"),
        HardwareOption(
            gpu or "gpu",
            "PyTorch GPU (CUDA or MPS)",
            None if gpu else "No CUDA or Metal GPU for PyTorch GPU.",
        ),
        *(
            HardwareOption(
                unit,
                f"Core ML {unit}",
                ne_reason if unit == "CPU_AND_NE" else coreml_reason,
            )
            for unit in COMPUTE_UNITS
        ),
    )
    return Hardware(training, inference, apple_silicon)
