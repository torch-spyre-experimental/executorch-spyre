# Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AOT ``preprocess`` for the spyre2 Triton→TTIR delegate — two staged artifacts.

Both stages frame their artifact into spyre's exact payload wire format
(``[u32 len][header json][tar]``), discriminated by ``header["artifact"]``:

- **``sdsc``** (default, on-device): the same ``mm`` partition is compiled by
  torch-spyre's device-verified ``compile_fx``/``SuperDSCScheduling`` path. We
  delegate wholesale to spyre's ``SpyreBackend.preprocess`` (reusing every v1/v2/v3
  capture path unchanged) and then annotate the header with ``artifact="sdsc"``
  plus ``ttir_provenance`` (the captured-TTIR front-half link). The reused C++
  ``SpyreBackend`` runtime consumes this exactly as a spyre payload.

- **``ktir``** (Stage 1, host-only): we capture the ``mm`` kernel's TTIR
  (``ttir_capture`` / a checked-in fixture), lower it to ``ktdp`` KTIR
  (``ttir_to_ktir``), structurally verify it against the golden
  (``ktir_verify``), and tar the KTIR MLIR. This is NOT device-runnable on the
  shipped SDK — the seam is emitted and validated so it flips to device with no
  ET-surface change once upstream ships the ktdp backend.

Select the stage with a ``CompileSpec(key="spyre2_artifact", value=b"ktir"|b"sdsc")``
on the partitioner; absent it, the device (``sdsc``) path runs.
"""

import hashlib
import io
import json
import logging
import os
import struct
import tarfile
from typing import Final, List

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram

# Reuse spyre's payload framing + AOT helpers verbatim (see module docstring).
from executorch.backends.spyre.spyre_backend import (
    SPYRE_BACKEND_ID,
    _concrete_shape,
    _pack_payload,
    _placeholder_kind,
    _retarget_edge_to_aten,
    _torch_dtype_to_str,
)
from executorch.backends.spyre2.ktir_verify import verify_matmul
from executorch.backends.spyre2.ttir_to_ktir import lower_mm_ttir_to_ktir

# spyre2's own backend id. ExecuTorch's ``to_backend`` resolves a partition's
# ``backend_id`` by matching it against ``cls.__name__`` over
# ``BackendDetails.__subclasses__()`` (backend_api.py), so this string MUST equal
# this class's name for ``Spyre2Backend.preprocess`` to be the one invoked. It is
# deliberately distinct from spyre's ``"SpyreBackend"`` (= ``SPYRE_BACKEND_ID``): with
# both packages importable, a shared id would collide on two same-named
# subclasses and route nondeterministically. The reused C++ runtime registers
# only ``"SpyreBackend"`` today, so the on-device Stage 2 path needs a macro-guarded
# second ``register_backend("Spyre2Backend", ...)`` alias (see runtime notes / T4);
# Stage 1 (KTIR) is host-only and never executed, so AOT resolution is all it
# needs.
SPYRE2_BACKEND_ID: Final[str] = "Spyre2Backend"

# CompileSpec key selecting the staged artifact. Value is b"sdsc" (default) or
# b"ktir".
_ARTIFACT_SPEC_KEY: Final[str] = "spyre2_artifact"
_KTIR_ARCNAME: Final[str] = "ktir/mm.ktir.mlir"

# fixtures/ttir/mm_fp16_64.ttir, resolved from this file (backends/spyre2/).
_PKG_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR: Final[str] = os.path.dirname(os.path.dirname(_PKG_DIR))
_TTIR_FIXTURE: Final[str] = os.path.join(
    _PKG_DIR, "fixtures", "ttir", "mm_fp16_64.ttir"
)
_KTIR_GOLDEN: Final[str] = os.path.join(
    _PKG_DIR, "fixtures", "ktir", "level4c_matmul_mlp.mlir"
)

logger = logging.getLogger(__name__)

# SPYRE2_BACKEND_ID is spyre2's routing id; SPYRE_BACKEND_ID (spyre's) is re-exported so
# provenance/payload helpers can still reference the shared runtime name.
__all__ = ["Spyre2Backend", "SPYRE2_BACKEND_ID", "SPYRE_BACKEND_ID"]


def _parse_artifact(compile_specs) -> str:
    """Read the ``spyre2_artifact`` directive; default to the device SDSC path."""
    for spec in compile_specs or []:
        if getattr(spec, "key", None) != _ARTIFACT_SPEC_KEY:
            continue
        value = spec.value
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        value = str(value).strip().lower()
        if value not in ("sdsc", "ktir"):
            raise ValueError(
                f"{_ARTIFACT_SPEC_KEY} must be 'sdsc' or 'ktir', got {value!r}"
            )
        return value
    return "sdsc"


def _io_specs(edge_program: ExportedProgram) -> tuple[list[dict], list[dict]]:
    """Extract input/output specs (shape/dtype/kind) without a spyre device.

    The Stage 1 KTIR path is host-only and must not require torch_spyre or the
    ``spyre`` device, so — unlike spyre's ``_make_spyre_example_inputs`` — this
    reads shapes/dtypes straight from placeholder/output ``meta['val']`` and
    never materializes device tensors. Same spec schema as spyre so the payload
    header is interchangeable.
    """
    gm = edge_program.graph_module
    input_specs: list[dict] = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if val is None:
            raise RuntimeError(
                f"placeholder {node.name!r} has no meta['val']; cannot build "
                "an input spec for the KTIR payload."
            )
        input_specs.append(
            {
                "shape": _concrete_shape(val, None),
                "dtype": _torch_dtype_to_str(val.dtype),
                "kind": _placeholder_kind(edge_program, node.name),
            }
        )
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    outs = output_node.args[0]
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    output_specs: list[dict] = []
    for out in outs:
        val = out.meta.get("val") if hasattr(out, "meta") else None
        if val is None:
            raise RuntimeError("output node arg has no meta['val'].")
        output_specs.append(
            {
                "shape": _concrete_shape(val, None, allow_symbolic=True),
                "dtype": _torch_dtype_to_str(val.dtype),
            }
        )
    return input_specs, output_specs


def _load_mm_ttir() -> str:
    """Return the ``mm`` TTIR: capture it if triton is present, else the fixture.

    On a Linux/pod venv with the triton wheel we capture fresh; on the Mac host
    (no wheel) we fall back to the checked-in fixture. Either way the text is a
    faithful ``tt``-dialect module the lowering can consume.
    """
    try:
        from executorch.backends.spyre2.ttir_capture import capture_mm_ttir

        return capture_mm_ttir()
    except Exception as e:  # triton missing or capture failed -> fixture
        if not os.path.isfile(_TTIR_FIXTURE):
            raise RuntimeError(
                "Stage 1 (KTIR) needs mm TTIR: capturing failed "
                f"({e!r}) and no fixture at {_TTIR_FIXTURE}. Run "
                "`python -m executorch.backends.spyre2.ttir_capture` on a host with the "
                "triton wheel to produce it."
            ) from e
        logger.info("preprocess(ktir): using checked-in TTIR fixture %s", _TTIR_FIXTURE)
        with open(_TTIR_FIXTURE) as f:
            return f.read()


def mm_ttir_provenance() -> dict:
    """Front-half provenance link recorded in the SDSC (Stage 2) header.

    Ties the device blob (produced by ``compile_fx``) back to the Triton TTIR
    the same ``mm`` was reached through. Honest about the bridge: the device
    bytes are SDSC, the TTIR is the captured front-half, not their literal
    compiler input.
    """
    prov = {"kernel": "spyre2::mm_fp16", "source": "triton-ttir"}
    if os.path.isfile(_TTIR_FIXTURE):
        with open(_TTIR_FIXTURE, "rb") as f:
            data = f.read()
        prov["ttir_fixture"] = os.path.relpath(_TTIR_FIXTURE, _PKG_DIR)
        prov["ttir_sha256"] = hashlib.sha256(data).hexdigest()
    return prov


class Spyre2Backend(BackendDetails):
    """``preprocess`` emitting a staged KTIR (host) or SDSC (device) payload."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        artifact = _parse_artifact(compile_specs)
        if artifact == "ktir":
            return Spyre2Backend._preprocess_ktir(edge_program, compile_specs)
        return Spyre2Backend._preprocess_sdsc(edge_program, compile_specs)

    # ---- Stage 2: device path (reuse spyre SDSC, annotate header) ------------
    @staticmethod
    def _preprocess_sdsc(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Delegate to spyre's proven SDSC preprocess, then annotate the header.

        Every v1/v2/v3 capture path is reused unchanged; we only unpack the
        result, inject ``artifact="sdsc"`` + ``ttir_provenance``, and re-pack —
        so the payload stays byte-compatible with the reused C++ runtime while
        carrying the Triton front-half link.
        """
        from executorch.backends.spyre.spyre_backend import SpyreBackend

        result = SpyreBackend.preprocess(edge_program, compile_specs)
        raw = result.processed_bytes
        (hlen,) = struct.unpack_from("<I", raw, 0)
        header = json.loads(raw[4 : 4 + hlen].decode("utf-8"))
        tar_bytes = raw[4 + hlen :]
        header["artifact"] = "sdsc"
        try:
            header["ttir_provenance"] = mm_ttir_provenance()
        except Exception as e:  # provenance is advisory, never fatal
            logger.warning("preprocess(sdsc): ttir_provenance skipped (%r)", e)
        logger.info(
            "preprocess(sdsc): v%s payload annotated with ttir provenance",
            header.get("version"),
        )
        return PreprocessResult(processed_bytes=_pack_payload(header, tar_bytes))

    # ---- Stage 1: host-only KTIR path ---------------------------------------
    @staticmethod
    def _preprocess_ktir(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        """Emit a structurally-verified KTIR payload (NOT device-runnable).

        TTIR (captured or fixture) -> ``ktdp`` KTIR (``ttir_to_ktir``) ->
        structural check vs the golden (``ktir_verify``) -> tar + header framed
        exactly like a spyre v1 payload, tagged ``artifact="ktir"``. The reused
        C++ runtime parses the header but has no ktdp consumer today, so a
        ``ktir`` payload is host-only by construction.
        """
        gm = edge_program.graph_module
        n_retargeted = _retarget_edge_to_aten(gm)
        input_specs, output_specs = _io_specs(edge_program)

        ttir = _load_mm_ttir()
        ktir = lower_mm_ttir_to_ktir(ttir)
        with open(_KTIR_GOLDEN) as f:
            verify_matmul(ktir, f.read())  # fails loudly on structural drift
        logger.info(
            "preprocess(ktir): lowered mm TTIR->KTIR and verified vs golden "
            "(%d bytes)",
            len(ktir),
        )

        ktir_bytes = ktir.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=_KTIR_ARCNAME)
            info.size = len(ktir_bytes)
            tf.addfile(info, io.BytesIO(ktir_bytes))
        tar_bytes = buf.getvalue()

        header = {
            "version": 1,
            "backend": SPYRE2_BACKEND_ID,
            "artifact": "ktir",
            "ktir_arcname": _KTIR_ARCNAME,
            "n_inputs": len(input_specs),
            "n_outputs": len(output_specs),
            "inputs": input_specs,
            "outputs": output_specs,
            "n_retargeted_ops": n_retargeted,
            # Naive layout kept for header-schema compatibility; the reused C++
            # runtime has no ktdp consumer, so a ktir payload is host-only.
            "pool_sticks": 0,
            "launch_layout": (
                [{"role": "input", "index": i} for i in range(len(input_specs))]
                + [{"role": "output", "index": j} for j in range(len(output_specs))]
            ),
            "device_runnable": False,
            "note": (
                "Stage 1 KTIR: structurally verified, host-only. No ktdp/linalg "
                "backend in shipped SDK 2.0; device mm goes through artifact=sdsc."
            ),
        }
        return PreprocessResult(processed_bytes=_pack_payload(header, tar_bytes))
