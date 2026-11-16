"""Lightweight dtype resolution for Spyre export/runner scripts.

Replicates the behaviour of hf_adapters.auto_spyre_model.dtype_for_model_path()
without importing auto_spyre_model so that this module has no dependency on the
full hf-adapters stack beyond what the callers already import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Union

import torch
from transformers import AutoConfig


@dataclass(frozen=True)
class _ModelDTypePolicy:
    dtype: torch.dtype | None = None
    cpu_dtype: torch.dtype | None = None


_MODEL_DTYPE_POLICIES: dict[str, _ModelDTypePolicy] = {
    "google/embeddinggemma-300m": _ModelDTypePolicy(dtype=torch.bfloat16),
    "ibm-granite/granite-4.0-1b-base": _ModelDTypePolicy(cpu_dtype=torch.float32),
    "ibm-granite/granite-4.0-1b": _ModelDTypePolicy(cpu_dtype=torch.float32),
}

_ST_SUBFOLDERS = ("0_CLIPModel", "0_Transformer")


def _autoconfig(
    model_name_or_path: Union[str, os.PathLike[str]],
    trust_remote_code: bool | None = None,
):
    """Return AutoConfig, probing known sentence-transformers subfolders on failure."""
    try:
        return AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception:
        pass
    for sub in _ST_SUBFOLDERS:
        try:
            return AutoConfig.from_pretrained(
                model_name_or_path,
                subfolder=sub,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            pass
    return None


def dtype_for_model_path(
    model_name_or_path: Union[str, os.PathLike[str]],
    target_device: str | torch.device,
    trust_remote_code: bool | None = None,
) -> torch.dtype:
    """Resolve one concrete dtype before loading a model on *target_device*."""
    device_str = (
        target_device.type
        if isinstance(target_device, torch.device)
        else target_device.split(":", 1)[0]
    )
    policy = _MODEL_DTYPE_POLICIES.get(
        os.fspath(model_name_or_path), _ModelDTypePolicy()
    )
    if device_str == "cpu" and policy.cpu_dtype is not None:
        dtype = policy.cpu_dtype
    elif policy.dtype is not None:
        dtype = policy.dtype
    else:
        config = _autoconfig(model_name_or_path, trust_remote_code=trust_remote_code)
        text_config = getattr(config, "text_config", None)
        dtype = (
            getattr(text_config, "dtype", None)
            or getattr(config, "dtype", None)
            or torch.float16
        )

    if dtype == torch.float32 and device_str == "spyre":
        dtype = torch.float16

    return dtype
