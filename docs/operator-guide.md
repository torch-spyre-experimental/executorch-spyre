<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Operator guide

How to add a new operator to the Spyre backend, validate it end-to-end, and promote
it to the supported set.

---

## Partition modes

Which ops are delegated is decided by the partition mode (`EXECUTORCH_SPYRE_PARTITION_MODE`
env var, or `SpyrePartitioner(mode=...)`).

### Supported contract (#50)

| Mode | Status | Delegated when |
|------|--------|----------------|
| `fallbacklist` | **Production default** | the op is **not** in torch-spyre's `fallback_ops`, is **not** `validated: false` in `supported_ops.yaml`, and passes the graph-local rules |
| `allowlist` | Conservative mode; used by host CI | the op is `validated: true` in `supported_ops.yaml` and passes the graph-local rules |

Rules that hold in **both** modes:

1. **`validated: false` ops never reach the device.** The 15 entries recorded as
   unvalidated (e.g. `rms_norm`, `native_layer_norm`, `embedding`, `cat`,
   `slice_copy`, `select_copy`, `_to_copy` and its edge spelling
   `_to_dim_order_copy`, `clamp_min/max`, `maximum/minimum`, `full`,
   `constant_pad_nd`, `_scaled_mm`) stay on CPU. torch-spyre *not* listing an op
   as a CPU fallback is not evidence that the op is correct on the device, so
   `fallbacklist` mode subtracts the unvalidated set
   (`_is_graph_local_excluded`, rule 0). To start delegating one, promote it in
   `supported_ops.yaml` after a green device run (see below).
2. **Graph-local layout rules** apply: softmax-fed matmul, and wide (>1 stick)
   output write-back, stay on CPU.
3. **Fail closed.** An unknown mode raises `ValueError`. `fallbacklist` raises
   `RuntimeError` if torch-spyre's `fallback_ops` cannot be loaded; it never
   silently switches to `allowlist`.

Consequently the delegated set in `fallbacklist` mode is
`(all ops) - fallback_ops - validated:false - graph-local exclusions`, and
`allowlist` mode is a (possibly strict) subset of it plus anything torch-spyre
lists as a fallback that this repo has validated.

### Ownership and versioning of `fallback_ops`

`fallback_ops` is owned by **torch-spyre** (`torch_spyre/ops/fallbacks.py`); this
repo does not copy it. It changes between torch-spyre releases, so the delegated
set depends on the installed release. The supported release is
**v0.5.0-rc.3** (`compatibility-manifest.json`, the `third_party/torch-spyre`
submodule, and `SUPPORTED_TORCH_SPYRE` in `spyre_partitioner.py`, kept in sync by a
host test). At partition time in `fallbacklist` mode the partitioner:

- records the installed torch-spyre version, the number of fallback ops, and a
  digest of the normalized list in `SpyrePartitioner.last_report["fallback_provenance"]`
  and prints it in the `Spyre PARTITION REPORT`;
- logs a **warning** if the installed version differs from the supported one.

Use the digest to confirm two exports used the same fallback list. A mismatch is
a warning, not an error: re-run the device gate before trusting a `.pte` built
against a different release.

### Choosing a mode

- **Export for the device (default):** do nothing; `fallbacklist` needs the
  torch-spyre SDK venv.
- **Host-only work (CI, laptops without torch-spyre):** set
  `EXECUTORCH_SPYRE_PARTITION_MODE=allowlist`. The host test suite pins this itself
  (`executorch/backends/spyre/test/conftest.py`), and tests that exercise
  `fallbacklist` stub the fallback list (`test_partition_modes.py`).
- Do not assume a device result for one mode holds for the other: the delegated
  sets differ.

### The allowlist

The file `executorch/backends/spyre/coverage/supported_ops.yaml` is the data-driven source of
truth for the `allowlist` mode and for the `validated: false` exclusions in both
modes. The partitioner calls `validated_op_names()` and `unvalidated_op_names()` at
runtime, which read this YAML; in `allowlist` mode an op absent from the file, or
present with `validated: false`, is not delegated and falls back to portable CPU
kernels instead. In `fallbacklist` mode an op present with `validated: false` is
likewise kept on CPU.

Promotion to `validated: true` is a **deliberate hand edit**: it is the single
audited checkpoint that enforces the "never miscompile" contract. An op is not
promoted until it has passed all four gates of the coverage harness (see below) on a
real device. This is intentional — `validated: false` entries in `candidates.yaml`
represent ops that are candidates for promotion but have not yet been device-verified.

---

## The 4-gate coverage harness

```bash
python -m executorch.backends.spyre.coverage.run_coverage
```

The harness runs each candidate op through four sequential gates:

| Gate | What it checks |
|------|---------------|
| **Export** | `torch.export` + `to_edge` succeeds; the op is captured in the Edge IR graph. |
| **Compile** | `preprocess()` runs the Spyre compiler (`DeepTools`) and produces a `.pte` payload without error. |
| **Run** | The `.pte` executes on the Spyre device (or on CPU via the portable runtime if the device is absent). |
| **Match** | Device output matches the CPU fp16 reference within the numerical tolerance envelope (`atol=1e-1, rtol=2e-2`). |

The harness degrades gracefully: if the Spyre device or the compiler is absent, gates
that require hardware automatically record `skip` rather than `fail`. This means the
same script is useful in three contexts:

- **Bare host (laptop/CI)** — export gate runs; compile/run/match gates skip.
- **Host with compiler** — export and compile gates run; run/match skip.
- **Device pod** — all four gates run.

---

## Adding an operator: step by step

### Step 1 — Add to `candidates.yaml`

Add an entry for the op with `validated: false`:

```yaml
# executorch/backends/spyre/coverage/candidates.yaml
- op: aten.my_new_op.default
  validated: false
  notes: "fp16 only; int inputs unsupported"
```

This makes the op visible to the coverage harness without yet allowing the
partitioner to delegate it.

### Step 2 — Run the 4-gate harness on device

On a Spyre pod with `FLEX_DEVICE=PF` set:

```bash
export HOME=/work XDG_CACHE_HOME=/work/.cache FLEX_DEVICE=PF
export PYTHONPATH="/work/executorch-spyre:$PYTHONPATH"

python -m executorch.backends.spyre.coverage.run_coverage --op aten.my_new_op.default
```

Review the output for all four gates: export, compile, run, match. If any gate fails,
address the failure before proceeding.

### Step 3 — Promote to `supported_ops.yaml`

If all four gates pass, **move** the entry from `candidates.yaml` to
`supported_ops.yaml` and flip the flag:

```yaml
# executorch/backends/spyre/coverage/supported_ops.yaml
- op: aten.my_new_op.default
  validated: true
  notes: "fp16 only; int inputs unsupported"
```

This is the intentional gate. Do not set `validated: true` until all four gates have
passed on-device. Code review of this hand edit is the policy enforcement point.

### Step 4 — Add a partitioner test

Add a test case in `executorch/backends/spyre/test/test_partitioner.py` that exercises the new op
and confirms:

- The op is partitioned to the Spyre delegate (not to CPU fallback) when the input
  types are supported.
- The op falls back to CPU correctly when input types are unsupported (e.g. integer
  inputs on an fp16-only op).

The test file runs host-only (no device required) as part of the standard CI gate.

### Step 5 — Add an on-device regression check (recommended)

If the device test infrastructure supports it, add a device-gated check to the
coverage suite. This ensures future changes to the partitioner or payload format do
not silently break the op's device path.

---

## Numerical tolerance

All match comparisons use the fp16 tolerance envelope:

```
atol = 1e-1
rtol = 2e-2
```

The intrinsic residual for a single operation grows approximately as `K · ε_fp16`,
where `K` is the reduction dimension size and `ε_fp16 ≈ 9.77e-4`. For a 64-wide
matmul row, the expected worst-case `max_abs_err` is on the order of 0.06–0.1,
well within `atol=1e-1`.

If a match gate fails with `max_abs_err` significantly larger than this envelope (for
example, `≈ 85` for an attention output), the likely cause is a layout issue rather
than numerical error — see the [debugging guide](debugging-guide.md#multi-tile-attention-scramble).

---

## Op support boundary

There are three distinct outcomes for an operation at partition time:

| Outcome | Description |
|---------|-------------|
| **Device-backed** | The op is delegated to the Spyre, is not `validated: false`, the input/output types and shapes satisfy the partitioner's hardware layout gates, and it is either `validated: true` (`allowlist`) or not in torch-spyre's `fallback_ops` (`fallbacklist`). In `fallbacklist` mode an op that is absent from `supported_ops.yaml` is delegated without a per-op device sign-off. |
| **CPU fallback** | The op is not delegated in the active mode (not validated in `allowlist`; listed in `fallback_ops` in `fallbacklist`), is recorded `validated: false` (either mode), or a hardware layout gate fires (e.g. the output is wider than a single tile and would be written back tile-major-scrambled). The partitioner keeps the op on the CPU portable kernel path. Each fallback is logged; a non-zero fallback count is not an error. |
| **Unsupported / rejected** | The op type or dtype is entirely unrecognised by the backend and cannot be lowered. The partitioner rejects it; the portable runtime handles it. |

CPU fallback is expected and acceptable in real models. The three justified fallback
stages in the tiny Granite demo (`aten::embedding`, the softmax-fed context matmul,
and the wide vocabulary-projection tail) are all CPU-fallback cases that were
analysed and documented rather than treated as errors.

---

## See also

- [setup_pod.md](setup_pod.md) — environment setup; [setup-reference.md](setup-reference.md) — host-only tests and the generic export CLI.
- [debugging-guide.md](debugging-guide.md) — failure modes including the tile-major
  scramble guard.
- `executorch/backends/spyre/coverage/supported_ops.yaml` — the live allowlist.
- `executorch/backends/spyre/coverage/candidates.yaml` — ops awaiting device validation.
- `executorch/backends/spyre/test/test_partitioner.py` — host-only partitioner tests.
