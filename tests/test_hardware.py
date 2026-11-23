"""Hardware detection without relying on the machine running pytest."""

import pytest

from image_classifier_lab.hardware import NEURAL_ENGINE_REASON, detect_hardware


@pytest.mark.parametrize(
    ("system", "machine", "cuda", "mps", "training", "inference", "ne_reason"),
    [
        (
            "darwin",
            "x86_64",
            False,
            True,
            ("cpu", "mps"),
            ("cpu", "mps", "CPU_ONLY", "CPU_AND_GPU", "ALL"),
            NEURAL_ENGINE_REASON,
        ),
        (
            "darwin",
            "arm64",
            False,
            True,
            ("cpu", "mps"),
            ("cpu", "mps", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"),
            None,
        ),
        (
            "linux",
            "x86_64",
            True,
            False,
            ("cpu", "cuda"),
            ("cpu", "cuda"),
            "Core ML runs on macOS only.",
        ),
        (
            "linux",
            "arm64",
            False,
            False,
            ("cpu",),
            ("cpu",),
            "Core ML runs on macOS only.",
        ),
    ],
)
def test_detection(
    system: str,
    machine: str,
    cuda: bool,
    mps: bool,
    training: tuple[str, ...],
    inference: tuple[str, ...],
    ne_reason: str | None,
) -> None:
    def runtime_probe() -> None:
        assert system == "darwin"

    hardware = detect_hardware(
        system=system,
        machine=machine,
        cuda_available=lambda: cuda,
        mps_available=lambda: mps,
        macos_version="14.0",
        coreml_runtime_reason=runtime_probe,
    )
    assert tuple(option.key for option in hardware.training if option.available) == training
    assert tuple(option.key for option in hardware.inference if option.available) == inference
    assert len(hardware.training) == 3
    assert len(hardware.inference) == 6
    assert hardware.apple_silicon == (system == "darwin" and machine == "arm64")
    assert next(option.reason for option in hardware.inference if option.key == "CPU_AND_NE") == (
        ne_reason
    )


@pytest.mark.parametrize(
    ("version", "runtime", "cpu_reason", "ne_reason"),
    [
        ("12.6", None, None, "Core ML CPU_AND_NE needs macOS 13 or later."),
        ("11.7", None, *("Core ML predictions need macOS 12 or later.",) * 2),
        ("", None, *("Core ML predictions need macOS 12 or later.",) * 2),
        ("14.0", "Runtime not loaded.", "Runtime not loaded.", "Runtime not loaded."),
    ],
)
def test_coreml_os_and_runtime_requirements(
    version: str, runtime: str | None, cpu_reason: str | None, ne_reason: str | None
) -> None:
    hardware = detect_hardware(
        system="darwin",
        machine="arm64",
        macos_version=version,
        coreml_runtime_reason=lambda: runtime,
        cuda_available=lambda: False,
        mps_available=lambda: False,
    )
    reasons = {option.key: option.reason for option in hardware.inference}
    assert reasons["CPU_ONLY"] == reasons["CPU_AND_GPU"] == reasons["ALL"] == cpu_reason
    assert reasons["CPU_AND_NE"] == ne_reason


def test_default_macos_probes_use_reported_architecture_and_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    monkeypatch.setattr("platform.mac_ver", lambda: ("12.6", ("", "", ""), "arm64"))
    hardware = detect_hardware(
        system="darwin",
        cuda_available=lambda: False,
        mps_available=lambda: True,
        coreml_runtime_reason=lambda: None,
    )
    assert hardware.apple_silicon
    available = {option.key for option in hardware.inference if option.available}
    assert {"CPU_ONLY", "ALL"} <= available
    assert "CPU_AND_NE" not in available
