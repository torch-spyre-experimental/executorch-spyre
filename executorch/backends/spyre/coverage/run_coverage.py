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
"""Spyre op-coverage harness (Phase 5).

For each candidate op (see ``candidates.yaml``) this drives a tiny single-op
module through the delegate pipeline and records where it lands:

    exports    torch.export -> to_edge succeeds                 (host, torch+ET)
    compiles   SpyreBackend.preprocess lowers it to SDSC          (device/pod)
    n_kernels  how many SDSC kernels the partition fused to     (device/pod)
    runs       the .pte executes on a Spyre device              (device + runner)
    matches    on-device output == CPU fp16 reference           (device + runner)

Each gate degrades to ``skip`` (not ``fail``) when its prerequisites are
missing, so the same script is useful three ways:

  * bare host (no torch): ``--list`` prints the candidate inventory and which
    ops have a registered probe module.
  * host CI (torch + executorch, no Spyre): the ``exports`` gate runs for real;
    ``compiles``/``runs``/``matches`` report ``skip`` (no torch_spyre).
  * Spyre dev pod (FLEX_DEVICE=PF, Spyre toolchain): ``exports`` + ``compiles``
    run for real.

The probe modules deliberately isolate ONE op each: the tiny edge graph is
delegated whole (bypassing the validated-only allowlist) so we test the op
itself, not whether it is already approved.

STATUS: all four gates are implemented. ``exports`` (host) and ``compiles``
(pod) run whenever their prerequisites are present. The device gate —
``runs``/``matches`` (execute the delegated ``.pte`` on a Spyre device and diff
against the CPU fp16 reference) — is wired via :func:`_gate_device`: it builds a
whole-graph-delegated ``.pte``, writes the input ``.bin`` files + a CPU fp16
reference and a ``manifest.txt``, then subprocess-invokes the generic C++
``spyre_coverage_runner`` and parses its ``RUN:``/``MATCH:`` stdout. It runs only
when opted in (``--device``) with the runner binary present (``--runner`` or
``$EXECUTORCH_SPYRE_COVERAGE_RUNNER``) and a live device (``FLEX_DEVICE`` set); absent any of
these it degrades to ``skip`` so host CI and compile-only pod runs stay green.
Once the device gate is green for an op, promotion to ``validated: true`` in
``supported_ops.yaml`` is a deliberate, logged edit — never silent.

The compile gate calls :func:`SpyreBackend.preprocess` directly on the single-op
edge program rather than going through ``to_backend``: for a one-op module the
whole edge graph IS the subgraph a partition would hand to ``preprocess``, so
this is the tightest "does SDSC lower this op" probe and sidesteps the allowlist.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Callable, Optional

# candidates.yaml / supported_ops.yaml are parsed with the zero-dependency
# loader so --list works on the bare host (no torch, no PyYAML).
from ._yaml_compat import safe_load
from .ops_registry import load_supported_ops, validated_op_names

_COVERAGE_DIR = Path(__file__).parent
_CANDIDATES_FILE = _COVERAGE_DIR / "candidates.yaml"
_SUPPORTED_OPS_FILE = _COVERAGE_DIR / "supported_ops.yaml"


# --------------------------------------------------------------------------- #
# Probe-module registry: op name -> (build module, build example inputs).      #
# Kept tiny and single-op. Factories import torch lazily so this module (and   #
# --list) load without torch installed. Add a probe here to bring a candidate  #
# op under test; ops without a probe report exports=skip("no probe").          #
# --------------------------------------------------------------------------- #

# Each factory returns (nn.Module, tuple_of_example_inputs). Registered lazily
# via _build_probe_registry() so torch is only imported when actually probing.


def _build_probe_registry() -> dict[str, Callable[[], tuple]]:
    import torch

    F = torch.nn.functional

    def mod(fn, *inputs):
        """Wrap a forward fn + its example inputs into (module, inputs)."""

        class _M(torch.nn.Module):
            def forward(self, *a):
                return fn(*a)

        return _M().eval(), tuple(inputs)

    def h(*shape):  # a random fp16 tensor
        return torch.randn(*shape, dtype=torch.float16)

    def i(*shape, high=8):  # int64 indices/token-ids (must NOT be fp16'd)
        return torch.randint(0, high, shape, dtype=torch.int64)

    def b(*shape):  # a bool mask/predicate
        return torch.randint(0, 2, shape, dtype=torch.bool)

    reg: dict[str, Callable[[], tuple]] = {
        # --- matmul family ---
        "aten::mm": lambda: mod(lambda a, b: a @ b, h(64, 64), h(64, 64)),
        "aten::addmm": lambda: mod(
            lambda x, w, b: F.linear(x, w, b), h(1, 64), h(128, 64), h(128)
        ),
        "aten::bmm": lambda: mod(torch.bmm, h(4, 32, 48), h(4, 48, 32)),
        # --- elementwise ---
        "aten::add": lambda: mod(lambda a, b: a + b, h(1, 64), h(1, 64)),
        "aten::sub": lambda: mod(lambda a, b: a - b, h(1, 64), h(1, 64)),
        "aten::mul": lambda: mod(lambda a, b: a * b, h(1, 64), h(1, 64)),
        "aten::div": lambda: mod(lambda a, b: a / b, h(1, 64), h(1, 64)),
        "aten::neg": lambda: mod(torch.neg, h(1, 64)),
        # --- activations ---
        "aten::relu": lambda: mod(F.relu, h(1, 64)),
        "aten::silu": lambda: mod(F.silu, h(1, 64)),
        "aten::sigmoid": lambda: mod(torch.sigmoid, h(1, 64)),
        "aten::gelu": lambda: mod(F.gelu, h(1, 64)),
        # --- reductions / norm ---
        "aten::mean": lambda: mod(lambda x: x.mean(dim=-1, keepdim=True), h(1, 64)),
        "aten::pow": lambda: mod(lambda x: x.pow(2), h(1, 64)),
        "aten::rsqrt": lambda: mod(torch.rsqrt, h(1, 64).abs() + 1.0),
        "aten::_softmax": lambda: mod(lambda x: F.softmax(x, dim=-1), h(1, 64)),
        # --- shape / view ---
        "aten::permute_copy": lambda: mod(lambda x: x.permute(1, 0), h(64, 128)),
        "aten::transpose_copy": lambda: mod(lambda x: x.transpose(1, 2), h(2, 8, 16)),
        "aten::cat": lambda: mod(
            lambda a, b: torch.cat([a, b], dim=-1), h(1, 32), h(1, 32)
        ),
        "aten::clone": lambda: mod(torch.clone, h(1, 64)),
        # --- rope ---
        "aten::cos": lambda: mod(torch.cos, h(1, 64)),
        "aten::sin": lambda: mod(torch.sin, h(1, 64)),
        # --- matmul (batched/broadcast form; 2D would be aten::mm) ---
        "aten::matmul": lambda: mod(torch.matmul, h(2, 4, 32, 48), h(2, 4, 48, 32)),
        # --- norm (prefer native fused spyre customop) ---
        "aten::rms_norm": lambda: mod(
            lambda x, w: F.rms_norm(x, (64,), w), h(1, 64), h(64)
        ),
        "aten::native_layer_norm": lambda: mod(
            lambda x, w, bb: F.layer_norm(x, (64,), w, bb), h(1, 64), h(64), h(64)
        ),
        # --- view family (edge dialect lowers these to *_copy) ---
        "aten::view_copy": lambda: mod(lambda x: x.view(128, 64), h(64, 128)),
        "aten::expand_copy": lambda: mod(lambda x: x.expand(4, 64), h(1, 64)),
        "aten::slice_copy": lambda: mod(lambda x: x[:, :32], h(1, 64)),
        "aten::select_copy": lambda: mod(lambda x: x[:, 0], h(2, 64)),
        "aten::unsqueeze_copy": lambda: mod(lambda x: x.unsqueeze(0), h(1, 64)),
        "aten::_reshape_alias_copy": lambda: mod(
            lambda x: x.reshape(32, 128), h(64, 64)
        ),
        # --- shape / const ---
        "aten::full": lambda: mod(
            lambda: torch.full((1, 64), 3.0, dtype=torch.float16)
        ),
        # --- dtype (convert_element_type; fp16 -> fp32 is in the table) ---
        "aten::_to_copy": lambda: mod(lambda x: x.to(torch.float32), h(1, 64)),
        # --- masking (bool predicate must stay bool, not fp16) ---
        "aten::where": lambda: mod(torch.where, b(1, 64), h(1, 64), h(1, 64)),
        "aten::clamp": lambda: mod(lambda x: x.clamp(-1.0, 1.0), h(1, 64)),
        # --- attention (prefer fused SDSC lowering; else bmm->softmax->bmm) ---
        "aten::scaled_dot_product_attention": lambda: mod(
            F.scaled_dot_product_attention,
            h(1, 4, 16, 32),
            h(1, 4, 16, 32),
            h(1, 4, 16, 32),
        ),
        # --- embedding / kv-cache (int64 indices; RISK ops, see candidates.yaml) ---
        "aten::embedding": lambda: mod(F.embedding, i(1, 8, high=32), h(32, 64)),
        "aten::index_copy": lambda: mod(
            lambda x, idx, src: x.index_copy(0, idx, src),
            h(8, 64),
            i(4, high=8),
            h(4, 64),
        ),
    }
    return reg


# --------------------------------------------------------------------------- #
# Result model                                                                 #
# --------------------------------------------------------------------------- #

# Gate status is one of these strings.
PASS, FAIL, SKIP = "pass", "fail", "skip"


@dataclasses.dataclass
class OpResult:
    name: str
    group: str = ""
    source: str = ""
    exports: str = SKIP
    compiles: str = SKIP
    n_kernels: Optional[int] = None
    runs: str = SKIP
    matches: str = SKIP
    note: str = ""

    def as_row(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Gates                                                                        #
# --------------------------------------------------------------------------- #


def _gate_export(build) -> tuple[str, object, str]:
    """Host gate: build the probe, torch.export -> to_edge.

    Returns (status, edge_program_or_None, note).
    """
    try:
        import torch  # noqa: F401
        from executorch.exir import to_edge
    except Exception as e:  # torch / executorch not installed
        return SKIP, None, f"no torch/executorch ({type(e).__name__})"

    try:
        module, inputs = build()
        module = module.half()
        ep = __import__("torch").export.export(module, inputs)
        edge = to_edge(ep)
        return PASS, edge.exported_program(), ""
    except Exception as e:  # a real export/to_edge failure for this op
        return FAIL, None, f"export: {type(e).__name__}: {e}"


def _gate_compile(edge_program) -> tuple[str, Optional[int], str]:
    """Device/pod gate: does SpyreBackend.preprocess lower it via SDSC?

    Returns (status, n_kernels, note). ``skip`` when torch_spyre is absent
    (i.e. off-pod). ``n_kernels`` is read from the emitted payload header: a v2
    multi-kernel payload (Phase 6c) reports ``len(header["kernels"])``; a v1
    single-kernel payload reports 1. (Pre-6c the >1 case raised a guard; that
    guard is gone, so we parse the header rather than match error text.)
    """
    if edge_program is None:
        return SKIP, None, "no edge program (export gate did not pass)"
    try:
        import torch_spyre  # noqa: F401
    except Exception:
        return SKIP, None, "no torch_spyre (off-pod)"

    import json
    import struct

    from .. import spyre_backend  # local import: pulls in torch/ET stack

    try:
        result = spyre_backend.SpyreBackend.preprocess(edge_program, [])
    except Exception as e:
        return FAIL, None, f"compile: {type(e).__name__}: {e}"

    # Parse the [u32 header_len LE][header_json][tar] framing for n_kernels.
    blob = bytes(result.processed_bytes)
    (hlen,) = struct.unpack("<I", blob[:4])
    header = json.loads(blob[4 : 4 + hlen].decode("utf-8"))
    n_kernels = len(header.get("kernels", [])) or 1
    note = "multi-kernel (v2 payload)" if n_kernels > 1 else ""
    return PASS, n_kernels, note


# --------------------------------------------------------------------------- #
# Device gate helpers                                                          #
# --------------------------------------------------------------------------- #

# torch dtype -> manifest dtype token understood by spyre_coverage_runner.
_MANIFEST_DTYPE = {
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.int64": "int64",
    "torch.bool": "bool",
}


def _manifest_dtype(torch_dtype) -> Optional[str]:
    """Map a torch dtype to the runner's dtype token, or None if unsupported."""
    return _MANIFEST_DTYPE.get(str(torch_dtype))


def _dump_tensor(path: Path, t, dtype_token: str) -> None:
    """Write ``t`` as raw little-endian bytes in the manifest ``dtype_token``.

    Mirrors ``export_matmul._dump_fp16`` but generalised over the four runner
    dtypes. Float references are produced by upcasting to fp32, computing, then
    rounding to the on-device output dtype (matching the tolerance envelope).
    """
    import torch

    cast = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "int64": torch.int64,
        "bool": torch.bool,
    }[dtype_token]
    arr = t.detach().to(cast).contiguous().cpu().numpy()
    arr.tofile(str(path))


def _first_output_val(exported_program):
    """FakeTensor meta for the first output of an edge exported program."""
    out_nodes = [n for n in exported_program.graph.nodes if n.op == "output"]
    if not out_nodes:
        return None
    val = out_nodes[0].args[0]
    while isinstance(val, (tuple, list)):
        val = val[0]
    return getattr(val, "meta", {}).get("val")


def _build_artifacts(name: str, workdir: Path) -> dict:
    """Compile a whole-graph-delegated ``.pte`` for ``name`` and dump the runner
    artifacts (``model.pte`` + ``inN.bin`` + ``ref.bin`` + ``manifest.txt``).

    **Runs as its OWN process** (dispatched by ``--build-artifacts``). This is
    load-bearing: the SDSC compile starts the torch_spyre device runtime and
    CLAIMS the single Spyre PF device for this process's lifetime. By doing the
    compile here and *exiting*, the device is released before the runner
    subprocess starts — otherwise the runner's own ``start_runtime()`` aborts
    (SIGABRT via ``pthread_once``), because the device is single-holder and a
    process that has already compiled cannot coexist with a second runtime.

    Returns a JSON-serialisable dict
    ``{exports, compiles, n_kernels, out_dtype, note}``. On failure the relevant
    gate is ``FAIL``/``None`` and ``note`` explains; artifacts may be absent.

    The whole single-op edge graph is delegated regardless of the
    ``validated: true`` allowlist (same bypass rationale as the compile gate): we
    are probing whether *this op* executes on-device, not whether it is already
    approved. A CPU fp32 reference (rounded to the on-device output dtype) is the
    golden value the runner diffs against under the Phase 2 fp16 envelope.
    """
    import json as _json
    import struct

    out = {
        "exports": SKIP,
        "compiles": SKIP,
        "n_kernels": None,
        "out_dtype": None,
        "note": "",
    }

    try:
        import torch
        import torch_spyre  # noqa: F401  (registers the spyre device)
        from executorch.exir import to_edge

        from .. import spyre_backend
        from ..spyre_partitioner import SpyrePartitioner
        from torch.fx.passes.operator_support import OperatorSupportBase
    except Exception as e:
        out["note"] = f"deps missing ({type(e).__name__}: {e})"
        return out

    class _AllCallFunctions(OperatorSupportBase):
        """Delegate the entire single-op graph, bypassing the allowlist."""

        def is_node_supported(self, submodules, node) -> bool:
            return node.op == "call_function"

    probes = _build_probe_registry()
    build = probes.get(name)
    if build is None:
        out["note"] = "no probe module registered"
        return out

    def _fresh_edge():
        """A fresh (module, inputs, ref, edge) — to_backend may consume an edge,
        so the n_kernels probe and the .pte build each get their own."""
        module, inputs = build()
        ref_inputs = tuple(t.float() if t.is_floating_point() else t for t in inputs)
        ref_module = copy.deepcopy(module).float().eval()
        with torch.no_grad():
            ref = ref_module(*ref_inputs)
        if isinstance(ref, (tuple, list)):
            ref = ref[0]
        ep = torch.export.export(module.half().eval(), inputs)
        return inputs, ref, to_edge(ep)

    # export -> to_edge
    try:
        inputs, ref, edge = _fresh_edge()
    except Exception as e:
        out["exports"] = FAIL
        out["note"] = f"export: {type(e).__name__}: {e}"
        return out
    out["exports"] = PASS

    # compile (SDSC) + n_kernels via preprocess on the edge program — the same
    # tight probe the host compile gate uses. THIS CLAIMS THE DEVICE.
    try:
        result = spyre_backend.SpyreBackend.preprocess(edge.exported_program(), [])
    except Exception as e:
        out["compiles"] = FAIL
        out["note"] = f"compile: {type(e).__name__}: {e}"
        return out
    blob = bytes(result.processed_bytes)
    (hlen,) = struct.unpack("<I", blob[:4])
    header = _json.loads(blob[4 : 4 + hlen].decode("utf-8"))
    n_kernels = len(header.get("kernels", [])) or 1
    out["compiles"] = PASS
    out["n_kernels"] = n_kernels
    if n_kernels != 1:
        # multi-kernel fusion needs the Phase 6 payload; not runnable yet.
        out["note"] = "multi-kernel (v2 payload)" if n_kernels > 1 else ""
        return out

    # Single-kernel: build the actual whole-graph-delegated .pte for the runner.
    # NB: rebind `inputs` too — probes draw fresh random inputs each call, so the
    # dumped in*.bin MUST come from the same _fresh_edge() call as `ref`, else the
    # golden reference is computed from different values than the device sees.
    try:
        inputs, ref, edge = _fresh_edge()
        out_val = _first_output_val(edge.exported_program())
        part = SpyrePartitioner()
        part.op_support = _AllCallFunctions()
        et = edge.to_backend(part).to_executorch()
    except Exception as e:
        msg = str(e)
        if "no SDSC kernel was emitted" in msg:
            # Pure data-movement op (e.g. a lone contiguous slice): the compile
            # probe at the earlier step produced a v1-style payload, but the
            # whole-graph delegated build emits no compute kernel, so there is
            # nothing to run standalone. NOT a device reject — validate this op
            # in-context, when a compute subgraph containing it runs on-device.
            out["note"] = (
                "no compute kernel (pure data movement) — validate in-context, not device-rejected"
            )
        else:
            out["note"] = f"pte build: {type(e).__name__}: {e}"
        return out

    out_dtype = _manifest_dtype(getattr(out_val, "dtype", None))
    if out_dtype is None:
        out["note"] = f"unsupported output dtype {getattr(out_val, 'dtype', '?')}"
        return out
    out["out_dtype"] = out_dtype
    out_dims = [int(d) for d in ref.shape]

    # Write the .pte + input bins + typed reference + manifest.
    (workdir / "model.pte").write_bytes(et.buffer)
    manifest = ["pte model.pte"]
    for idx, t in enumerate(inputs):
        dt = _manifest_dtype(t.dtype)
        if dt is None:
            out["note"] = f"unsupported input dtype {t.dtype}"
            return out
        fname = f"in{idx}.bin"
        _dump_tensor(workdir / fname, t, dt)
        dims = list(t.shape)
        manifest.append(
            f"input {fname} {dt} {len(dims)} " + " ".join(str(int(d)) for d in dims)
        )
    _dump_tensor(workdir / "ref.bin", ref, out_dtype)
    manifest.append(
        f"output ref.bin {out_dtype} {len(out_dims)} "
        + " ".join(str(d) for d in out_dims)
    )
    (workdir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return out


def _gate_device(
    name: str,
    runner_path: Optional[str],
    keep_artifacts: bool = False,
) -> tuple[str, Optional[int], str, str, str]:
    """Device gate: compile + run a delegated ``.pte`` on-device, diff the ref.

    Returns ``(compiles, n_kernels, runs, matches, note)``. Degrades to
    ``skip`` — never ``fail`` — when a prerequisite is absent (no ``FLEX_DEVICE``,
    no runner binary), so host CI and compile-only pod runs stay green.

    Orchestrates two *sequential subprocesses*, never compiling in-process:

      1. **Builder** (``_build_artifacts`` via ``--build-artifacts``): compiles
         the whole-graph-delegated ``.pte`` and writes the runner artifacts. The
         compile claims the single Spyre device; the subprocess then exits,
         releasing it.
      2. **Runner** (``spyre_coverage_runner``): executes the ``.pte`` on-device and
         prints its ``RUN:``/``MATCH:`` verdict.

    Splitting the compile and the run into separate short-lived processes is
    mandatory: the device is single-holder, so a parent that has compiled cannot
    spawn a child that starts its own runtime without the child aborting.
    """
    if not os.environ.get("FLEX_DEVICE"):
        return (
            SKIP,
            None,
            SKIP,
            SKIP,
            "device gate skipped: FLEX_DEVICE unset (would be MOCK)",
        )
    if not runner_path or not Path(runner_path).exists():
        return (
            SKIP,
            None,
            SKIP,
            SKIP,
            "device gate skipped: no spyre_coverage_runner (--runner)",
        )

    workdir = Path(tempfile.mkdtemp(prefix="spyre_cov_"))
    try:
        # STEP 1 — build artifacts in a subprocess that EXITS (releases device).
        build_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "executorch.backends.spyre.coverage.run_coverage",
                "--build-artifacts",
                name,
                str(workdir),
            ],
            capture_output=True,
            text=True,
        )
        info = None
        for line in build_proc.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                info = json.loads(line[len("RESULT_JSON ") :])
        if info is None:
            tail = (build_proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            return (
                FAIL,
                None,
                SKIP,
                SKIP,
                f"artifact build rc={build_proc.returncode}: {tail}",
            )

        compiles = info.get("compiles", SKIP)
        n_kernels = info.get("n_kernels")
        note = info.get("note", "")

        if compiles != PASS:
            return compiles, n_kernels, SKIP, SKIP, note
        if n_kernels != 1:
            return (
                compiles,
                n_kernels,
                SKIP,
                SKIP,
                (note or f"device gate skipped: {n_kernels}-kernel (Phase 6)"),
            )

        # STEP 2 — run the .pte on-device (the device is now free).
        proc = subprocess.run(
            [str(Path(runner_path).resolve()), str(workdir)],
            capture_output=True,
            text=True,
        )
        runs = matches = SKIP
        for line in proc.stdout.splitlines():
            if line.startswith("RUN:"):
                runs = line.split(":", 1)[1].strip()
            elif line.startswith("MATCH:"):
                matches = line.split(":", 1)[1].strip()

        if runs == SKIP and matches == SKIP:
            # Runner produced no verdict — surface why (rc + last stderr line).
            # A missing manifest.txt means the *builder* never finished writing
            # the artifacts even though its compile probe reported compiles=PASS
            # (the .pte build / dtype / manifest step after the probe bailed).
            # In that case the builder's own note is the real diagnosis, so keep
            # it instead of the generic "cannot open manifest.txt" runner error.
            tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            runner_note = f"runner rc={proc.returncode}: {tail}"
            if not (workdir / "manifest.txt").exists() and note:
                note = f"artifacts incomplete (compile probe passed): {note}"
            else:
                note = runner_note
        return compiles, n_kernels, runs, matches, note
    finally:
        if keep_artifacts:
            print(f"  [device] kept artifacts in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def _run_coverage(
    only: Optional[set[str]] = None,
    do_compile: bool = True,
    do_device: bool = False,
    runner_path: Optional[str] = None,
    keep_artifacts: bool = False,
) -> list[OpResult]:
    """Run the available gates over the candidate set, return the matrix."""
    candidates = _load_candidates()
    validated = validated_op_names()
    probes = None
    results: list[OpResult] = []

    for cand in candidates:
        name = cand["name"]
        if only and name not in only:
            continue
        res = OpResult(
            name=name, group=cand.get("group", ""), source=cand.get("source", "")
        )
        if name in validated:
            res.note = "already validated"

        # Lazily build the probe registry once torch is confirmed importable.
        if probes is None:
            try:
                probes = _build_probe_registry()
            except Exception as e:
                # torch missing: no gate can run; report why and move on.
                res.exports = SKIP
                res.note = f"no probe registry: {type(e).__name__}"
                results.append(res)
                continue

        build = probes.get(name)
        if build is None:
            res.exports = SKIP
            res.note = "no probe module registered"
            results.append(res)
            continue

        res.exports, edge, note = _gate_export(build)
        if note:
            res.note = note

        # In device mode the compile MUST happen in the builder subprocess (it
        # claims the single Spyre device); an in-parent compile here would hold
        # the device and make the runner child abort. So the device gate itself
        # reports compiles + n_kernels (and runs/matches). Off-device, the
        # in-parent compile gate runs as before.
        if do_device and res.exports == PASS:
            res.compiles, res.n_kernels, res.runs, res.matches, dnote = _gate_device(
                name, runner_path, keep_artifacts=keep_artifacts
            )
            if dnote:
                res.note = dnote
        elif do_compile and res.exports == PASS:
            res.compiles, res.n_kernels, cnote = _gate_compile(edge)
            if cnote:
                res.note = cnote

        results.append(res)
    return results


# --------------------------------------------------------------------------- #
# candidates.yaml + promotion                                                  #
# --------------------------------------------------------------------------- #


def _load_candidates() -> list[dict]:
    doc = safe_load(_CANDIDATES_FILE.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("candidates"), list):
        raise ValueError(f"{_CANDIDATES_FILE}: expected a top-level 'candidates:' list")
    out = []
    # Include already-validated ops too, so the matrix shows the full picture.
    validated = {op.name: op for op in load_supported_ops()}
    seen = set()
    for entry in doc["candidates"]:
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{_CANDIDATES_FILE}: bad candidate entry {entry!r}")
        out.append(entry)
        seen.add(entry["name"])
    # Surface validated ops that are not in candidates.yaml (regression view).
    for vname, vop in validated.items():
        if vop.validated and vname not in seen:
            out.append({"name": vname, "group": "validated", "source": "supported_ops"})
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #

_MARK = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip"}


def _print_table(results: list[OpResult]) -> None:
    cols = ("name", "group", "exports", "compiles", "n_kern", "runs", "matches")
    widths = {c: len(c) for c in cols}
    rows = []
    for r in results:
        row = {
            "name": r.name,
            "group": r.group,
            "exports": _MARK.get(r.exports, r.exports),
            "compiles": _MARK.get(r.compiles, r.compiles),
            "n_kern": "" if r.n_kernels is None else str(r.n_kernels),
            "runs": _MARK.get(r.runs, r.runs),
            "matches": _MARK.get(r.matches, r.matches),
        }
        for c in cols:
            widths[c] = max(widths[c], len(row[c]))
        rows.append((row, r.note))

    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("  ".join("-" * widths[c] for c in cols))
    for row, note in rows:
        line = "  ".join(row[c].ljust(widths[c]) for c in cols)
        if note:
            line += f"   # {note}"
        print(line)

    n = len(results)
    n_exports = sum(1 for r in results if r.exports == PASS)
    n_compiles = sum(1 for r in results if r.compiles == PASS)
    n_runs = sum(1 for r in results if r.runs == PASS)
    n_matches = sum(1 for r in results if r.matches == PASS)
    print()
    print(
        f"{n} candidate(s): exports={n_exports} compiles={n_compiles} "
        f"runs={n_runs} matches={n_matches}  (blank gate == skipped: prerequisite absent)"
    )


def _emit_yaml(results: list[OpResult], path: Path) -> None:
    lines = ["# Generated by run_coverage.py -- do not edit by hand.", "matrix:"]
    for r in results:
        lines.append(f"  - name: {json.dumps(r.name)}")
        lines.append(f"    group: {json.dumps(r.group)}")
        lines.append(f"    exports: {r.exports}")
        lines.append(f"    compiles: {r.compiles}")
        lines.append(f"    n_kernels: {'null' if r.n_kernels is None else r.n_kernels}")
        lines.append(f"    runs: {r.runs}")
        lines.append(f"    matches: {r.matches}")
        if r.note:
            lines.append(f"    note: {json.dumps(r.note)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _cmd_list() -> int:
    """Print the candidate inventory + which have probes. Works without torch."""
    candidates = _load_candidates()
    try:
        probes = _build_probe_registry()
        have_torch = True
    except Exception:
        probes = {}
        have_torch = False
    validated = validated_op_names()
    print(f"{'op':<38}  {'group':<12}  {'source':<14}  probe  validated")
    print("-" * 92)
    for c in candidates:
        name = c["name"]
        has_probe = "yes" if (not have_torch or name in probes) else "no"
        if not have_torch:
            has_probe = "?"  # cannot tell without torch
        print(
            f"{name:<38}  {c.get('group',''):<12}  {c.get('source',''):<14}  "
            f"{has_probe:<5}  {'yes' if name in validated else ''}"
        )
    if not have_torch:
        print("\n(torch not importable here; probe column unknown -- run on the pod)")
    return 0


def _cmd_shadow() -> int:
    """Name-level shadow report: allowlist (supported_ops.yaml) vs torch-spyre fallbacklist.

    Observational only (changes no allowlist, no partitioning). Covers every
    op named in candidates.yaml / supported_ops.yaml / torch-spyre's fallback
    list. There is no graph here, so no root-op attribution; use
    ``EXECUTORCH_SPYRE_SHADOW_MODE=1`` on a real export for per-node edge/root output.
    """
    from ..spyre_partitioner import load_spyre_fallback_ops, shadow_decisions

    spyre_fallbacks, reason = load_spyre_fallback_ops()
    if spyre_fallbacks is None:
        print(f"shadow report unavailable: {reason}")
        return 0
    allowlist_ops = validated_op_names()
    names = (
        {c["name"] for c in _load_candidates()}
        | set(allowlist_ops)
        | set(spyre_fallbacks)
    )
    names = sorted(n for n in names if n.startswith("aten::"))
    rows = []
    for n in names:
        allow_d, spyre_d = shadow_decisions(n, allowlist_ops, spyre_fallbacks)
        if allow_d != spyre_d:
            rows.append((n, allow_d, spyre_d))
    print(f"{'op':<38}  {'allowlist':<11}  spyre")
    print("-" * 60)
    for n, a, s in rows:
        print(f"{n:<38}  {a:<11}  {s}")
    print(f"\n{len(names)} aten op(s) compared, {len(rows)} disagreement(s).")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--list", action="store_true", help="list candidate inventory and exit"
    )
    ap.add_argument(
        "--op", action="append", default=None, help="restrict to op(s) by name"
    )
    ap.add_argument(
        "--no-compile",
        action="store_true",
        help="run only the host export gate (skip preprocess/SDSC)",
    )
    ap.add_argument(
        "--emit", type=str, default=None, help="write the matrix to this YAML path"
    )
    ap.add_argument(
        "--device",
        action="store_true",
        help="run the on-device gate (runs/matches) for single-kernel passes; "
        "needs FLEX_DEVICE set and the C++ spyre_coverage_runner",
    )
    ap.add_argument(
        "--runner",
        type=str,
        default=os.environ.get("EXECUTORCH_SPYRE_COVERAGE_RUNNER"),
        help="path to the built spyre_coverage_runner "
        "(default: $EXECUTORCH_SPYRE_COVERAGE_RUNNER)",
    )
    ap.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="do not delete the per-op device tempdir (.pte/bins/manifest)",
    )
    ap.add_argument(
        "--shadow",
        action="store_true",
        help="print the allowlist vs torch-spyre fallbacklist "
        "disagreement report and exit (observational; needs torch_spyre)",
    )
    ap.add_argument(
        # Internal: invoked by _gate_device as its own subprocess so the SDSC
        # compile (which claims the single-holder Spyre device) happens in a
        # short-lived process that exits before the runner starts. Prints a
        # single 'RESULT_JSON <dict>' line for the parent to parse.
        "--build-artifacts",
        nargs=2,
        metavar=("OP", "WORKDIR"),
        default=None,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args(argv)

    if args.build_artifacts:
        name, wd = args.build_artifacts
        info = _build_artifacts(name, Path(wd))
        print("RESULT_JSON " + json.dumps(info))
        return 0

    if args.list:
        return _cmd_list()

    if args.shadow:
        return _cmd_shadow()

    only = set(args.op) if args.op else None
    results = _run_coverage(
        only=only,
        do_compile=not args.no_compile,
        do_device=args.device,
        runner_path=args.runner,
        keep_artifacts=args.keep_artifacts,
    )
    _print_table(results)
    if args.emit:
        _emit_yaml(results, Path(args.emit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
