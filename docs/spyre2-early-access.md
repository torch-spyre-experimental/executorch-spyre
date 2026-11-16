<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Spyre2 early access

The `spyre2` backend (`Spyre2Backend`) is the ExecuTorch delegate implementation for
the IBM Spyre on the Triton codegen path: `Inductor → @triton.jit → TTIR → KTIR`,
as opposed to the SuperDSC path used by the original `spyre` backend (`SpyreBackend`).

This document describes what is proven and working as of the early-access release,
what is explicitly not included, and the forward roadmap.

---

## What is proven in this release

### Triton → KTIR pipeline (off-device)

The `torch-spyre/triton` public fork provides the Triton → TTIR → KTIR frontend that
the main SDK image does not yet include. Two capabilities have been verified
off-device:

**TK0 (elementwise):** A 1D fp16 elementwise-add kernel lowers all the way from
Triton to TTIR to KTIR (`ktdp` dialect) and validates numerically via the
`ktir_cpu` reference executor: `max_abs_err 2e-4 / 5e-4`, MATCH. The
`torch-spyre/triton` backend test suite reports **915 passed, 44 skipped, 0 failed**.

**TK1 (matmul):** A `tl.dot`-based matmul kernel lowers to structured, reshape-free
KTIR — specifically `linalg.matmul` contraction plus one residual K `scf.for` loop.
Off-device numerical validation against a NumPy reference: `max_abs_err 2.17e-4`,
MATCH.

### First `mm` on a real Spyre device (T4)

A 64×64×64 fp16 matmul delegated through `Spyre2Backend` ran on a real Spyre device
and matched the CPU fp16 reference: `max_abs_err 0.09375`, within the fp16 tolerance
envelope (`atol=1e-1, rtol=2e-2`), EXIT 0.

This path uses the SDSC bridge (spyre's `compile_fx` / `SuperDSCScheduling`) with the
captured TTIR carried as front-half provenance — a staged design that holds the
KTIR artifact ready for when the SDK can consume it, while delivering a
device-verified result today.

### Off-device TTIR → KTIR lowering

The `ttir_to_ktir` module lowers captured Triton IR to KTIR for both elementwise and
matmul operations. The KTIR output has been structurally verified against a hand-
authored golden reference. The `spyre2` host-only test suite (`executorch/backends/spyre2/test/`)
is green on a standard development machine.

---

## What is NOT included in this release

The following capabilities are **not part of this early-access release**. They are
documented here so evaluators have a clear picture of the current boundary.

### On-device KTIR execution

The `dbo-opt --from-ktir` compiler pass and the OpSpec → KTIR emitter that would
consume a KTIR artifact on a real device are **not present in the current SDK pod
image**. The `ktdp` dialect is not compiled into any of the deeptools binaries on the
image. Until a future SDK image ships these tools, KTIR cannot be executed on the
device — only on the `ktir_cpu` reference executor (off-device).

This is an upstream SDK limitation, not a gap in this project. The T4 device run
works by routing through the SDSC bridge; it does not execute KTIR on device.

### Full model (Granite or any decoder) end-to-end via the Triton/KTIR path

No end-to-end model has been run through `Spyre2Backend` on device. The T4
device-verified result is a single `mm` operator. Multi-layer, multi-op, and
full-model end-to-end support via the Triton/KTIR path is future work.

### Multi-kernel, multi-tile, dynamic-shape, and mixed-dtype in spyre2

These capabilities are mature in the `spyre`/SDSC backend (phases 6 and 7). They are
**not yet ported to `spyre2`**. The spyre2 partitioner inherits the spyre partitioner's
hardware layout gates (including the multi-tile attention scramble guard), but the
multi-kernel launch DAG and static shape-bucketing wire format have not been ported to
the KTIR artifact path.

### Profiling and quantisation support

Neither Spyre profiling (`EXECUTORCH_SPYRE_PROFILE`) nor quantisation (int8 / mixed-precision) is
supported in the spyre2 backend.

---

## Roadmap

Two forward routes exist. Both are contingent on upstream changes and are provided
for informational purposes.

### Route A — OpSpec → KTIR via a future SDK image

The SDK 2.0 pod image is expected to ship `dbo-opt --from-ktir` and the associated
OpSpec → KTIR emitter in a future release. When that image is available:

1. Run the existing T5a reconciliation probe (`executorch/backends/spyre2/probes/t5a_probe.sh`)
   to confirm the tools are present.
2. Drive the T4 harness unchanged — the payload container, partitioner, and tensor
   bridge require no modification.
3. First KTIR `mm` on device.
4. Port coverage harness and partitioner; Granite end-to-end via KTIR.

### Route B — Direct `ktdp` → Dataflow IR (D-track)

An alternative that does not require a new SDK image: author the `ktdp` → Dataflow IR
lowering directly against the `dataflow` entry point that is already present on the
pod. This is independent of the OpSpec → KTIR route and provides a path to on-device
KTIR execution without waiting for an image update.

### Once either route reaches on-device KTIR

- First KTIR `mm` on a real Spyre device.
- First Granite end-to-end run via the KTIR path.
- Convergence with spyre for the upstream contribution decision.

---

## How to use spyre2 today

### Host-only tests (no device required)

```bash
cd executorch-spyre

# Use the same Python 3.12 venv from the spyre host check, or create one:
python3.12 -m venv .venv-hosttest
./.venv-hosttest/bin/python -m pip install "executorch==1.5.1" "torch==2.13.0" pytest

export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

./.venv-hosttest/bin/python -m pytest -p no:cacheprovider executorch/backends/spyre2/test/
```

The spyre2 test suite includes:
- `test_ttir_to_ktir.py` — TTIR → KTIR lowering and structural diff against the
  golden reference.
- `test_preprocess_spyre2.py` — `Spyre2Backend` resolution, `Spyre2Partitioner` routing,
  and the Stage 1 KTIR payload round-trip.

Triton-gated host tests self-skip when Triton is absent. Physical-device tests
live separately under `executorch/backends/spyre2/test_spyre/`; they require
`FLEX_DEVICE=PF` and fail rather than silently using MOCK mode.

### Device path (SDSC bridge, same as spyre)

For the device-verified T4 path, use the runbook at `executorch/backends/spyre2/probes/T4-RUNBOOK.md`.
This path runs the `Spyre2Backend`-delegated `mm` on device via the SDSC bridge, with
TTIR provenance captured in the payload.

The mandatory environment guardrails from `spyre` apply unchanged:

```bash
export HOME=/work XDG_CACHE_HOME=/work/.cache FLEX_DEVICE=PF
python -m pytest executorch/backends/spyre2/test_spyre/ -m spyre
```

See [debugging-guide.md](debugging-guide.md) for failure modes common to both
backends.

---

## See also

- [setup-reference.md](setup-reference.md) — host-only setup; [setup_pod.md](setup_pod.md) — device setup.
- [version-matrix.md](version-matrix.md) — version pins (same as spyre).
- [operator-guide.md](operator-guide.md) — op allowlist and 4-gate harness (shared
  with spyre).
- `executorch/backends/spyre2/` — spyre2 source tree.
- `executorch/backends/spyre/` — spyre source tree (partitioner, payload format, and C++ runtime
  that spyre2 reuses).
