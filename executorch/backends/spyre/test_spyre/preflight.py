"""Shared physical-device preflight for Spyre and Spyre2 Spyre tests."""

import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _normalized_version(value: str) -> str:
    """Compare wheel and manifest spellings such as rc3 and -rc.3."""
    return re.sub(r"[-_.+]", "", value.split("+", 1)[0]).lower()


def _check_compatibility_manifest() -> None:
    manifest = json.loads(
        (_PROJECT_ROOT / "compatibility-manifest.json").read_text(encoding="utf-8")
    )
    installed = {
        "torch": importlib.metadata.version("torch"),
        "executorch": importlib.metadata.version("executorch"),
        "torch_spyre": importlib.metadata.version("torch-spyre"),
    }
    mismatches = [
        f"{name}: installed {installed[name]}, expected {manifest[name]}"
        for name in installed
        if _normalized_version(installed[name]) != _normalized_version(manifest[name])
    ]
    expected_python = str(manifest["python"])
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        mismatches.append(
            f"python: installed {actual_python}, expected {expected_python}"
        )
    if mismatches:
        raise pytest.UsageError(
            "Spyre compatibility-manifest mismatch: " + "; ".join(mismatches)
        )


@pytest.fixture(scope="session", autouse=True)
def require_physical_spyre():
    if os.environ.get("FLEX_DEVICE") != "PF":
        raise pytest.UsageError(
            "Spyre regression tests require FLEX_DEVICE=PF; MOCK is not accepted"
        )

    try:
        _check_compatibility_manifest()
    except importlib.metadata.PackageNotFoundError as exc:
        raise pytest.UsageError(
            "Spyre regression tests require torch and torch_spyre"
        ) from exc

    # Device ownership is process-scoped. Probe in a child that exits promptly,
    # otherwise the pytest parent can retain the single device and prevent the
    # coverage runner from starting its own runtime later in the suite.
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import torch, torch_spyre; "
                    "x = torch.zeros(1, dtype=torch.float16, device='spyre:0'); "
                    "print(x.device)"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise pytest.UsageError("spyre:0 probe timed out after 60 seconds") from exc
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        tail = detail[-1] if detail else "device probe failed without output"
        raise pytest.UsageError(f"spyre:0 is not usable: {tail}")
