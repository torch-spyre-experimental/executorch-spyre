# Spyre op-coverage workflow (Phase 5)

This directory is the machinery for growing the delegate's ATen op coverage in a
way that upholds the delegate's core contract — **fail loudly, never
miscompile**. An op is delegated only after a Spyre device has confirmed that
SDSC lowers it *and* the on-device result matches a CPU fp16 reference.

## The three files

| file | role |
|------|------|
| `candidates.yaml` | **work queue** — core-aten ops we *want* to delegate but have not yet confirmed on-device. Grouped by Granite decoder-layer role (Phase 8 dependency order), tagged by `source` provenance. |
| `supported_ops.yaml` | **source of truth for device validation** — every op the delegate knows about, each flagged `validated: true/false`. In the `allowlist` mode the partitioner delegates *only* `validated: true` entries; in the production-default `fallbacklist` mode it delegates everything torch-spyre does not list as a CPU fallback *except* `validated: false` entries. `validated: false` ops stay on CPU in both modes; see `docs/operator-guide.md#partition-modes`. |
| `run_coverage.py` | **harness** — drives a tiny single-op module for each candidate through the delegate pipeline and records where it lands (a coverage matrix). |

`ops_registry.py` loads `supported_ops.yaml` (via the zero-dependency
`_yaml_compat.py` loader) and exposes `validated_op_names()`, which
`spyre_partitioner.py` consumes directly — so the allowlist is *data driven*, not
a hand-maintained frozenset.

## The gates (and where each runs)

`run_coverage.py` runs a candidate op through a sequence of gates. Each gate
degrades to `skip` (never `fail`) when its prerequisites are missing, so the
same script is useful in three environments:

| gate | question | requires | runs on |
|------|----------|----------|---------|
| `exports` | `torch.export → to_edge` succeeds? | torch + executorch | host CI, pod |
| `compiles` | `SpyreBackend.preprocess` lowers it to SDSC? | torch_spyre toolchain | **pod only** |
| `n_kernels` | how many SDSC kernels did it fuse to? | (from `compiles`) | pod |
| `runs` | the `.pte` executes on a Spyre device? | built C++ runner + live device | **pod + device** |
| `matches` | on-device output == CPU fp16 reference? | built C++ runner + live device | **pod + device** |

Usage by environment:

- **bare host (no torch)** — `python -m ...run_coverage --list` prints the
  candidate inventory. Every gate reports `skip`.
- **host CI (torch + executorch, no Spyre)** — the `exports` gate runs for real;
  `compiles`/`runs`/`matches` report `skip` (no `torch_spyre`).
- **Spyre dev pod (`FLEX_DEVICE=PF`, Spyre toolchain)** — `exports` + `compiles`
  run for real. The device gate (`runs`/`matches`) runs when opted in with
  `--device` and given a built `spyre_coverage_runner` (`--runner PATH` or
  `$EXECUTORCH_SPYRE_COVERAGE_RUNNER`); `_gate_device` builds a whole-graph-delegated `.pte`,
  writes the input `.bin`s + a CPU reference + `manifest.txt`, and invokes the
  runner. Without `--device` (or with the runner/`FLEX_DEVICE` absent) those two
  columns report `skip`, so compile-only pod runs stay green.

## Running it

```bash
# bare inventory (works anywhere, no deps)
python -m executorch.backends.spyre.coverage.run_coverage --list

# host export gate only (torch + executorch installed, no device)
python -m executorch.backends.spyre.coverage.run_coverage --no-compile

# full host+pod gates on a Spyre dev pod
#   NOTE: do NOT set TORCH_DEVICE_BACKEND_AUTOLOAD=0 here. That flag suppresses
#   torch-spyre's PrivateUse1 "spyre" device registration, so preprocess's
#   `.to("spyre")` then fails with 'Expected one of cpu, cuda, ... at start of
#   device string: spyre' and the compiles gate goes all-FAIL. On a pod with a
#   real device (FLEX_DEVICE=PF), plain `import torch_spyre` registers spyre:0.
FLEX_DEVICE=PF python -m executorch.backends.spyre.coverage.run_coverage --emit matrix.yaml

# host+pod+device: also execute single-kernel passes on the Spyre device and
# diff against a CPU reference (needs the built runner; see spyre_runner/).
FLEX_DEVICE=PF EXECUTORCH_SPYRE_COVERAGE_RUNNER=$PWD/build-spyre/spyre_coverage_runner \
  python -m executorch.backends.spyre.coverage.run_coverage --device --emit matrix.yaml

# one op at a time while bringing it up
python -m executorch.backends.spyre.coverage.run_coverage --op aten::bmm --op aten::silu
```

## Adding a candidate op

1. Add it to `candidates.yaml` (name, `group`, `source`, optional `note`).
2. Register a tiny **single-op** probe module in `_build_probe_registry()` in
   `run_coverage.py`. Keep it minimal: the goal is to test the op itself, not a
   composite. Ops without a probe report `exports=skip("no probe")`.
3. Run the harness on the pod and read the matrix.

## Promotion — deliberate, never silent

When the device gate is green for an op (compiles to SDSC, runs, and matches the
CPU fp16 reference), it graduates to `validated: true` in `supported_ops.yaml`
with a `since_phase` note. **This is a hand edit, reviewed like any code
change** — the harness reports the evidence; a human records the promotion. In both
modes that edit is what makes the partitioner start delegating the op (via
`validated_op_names()` / `unvalidated_op_names()`), so it is the single audited
chokepoint for the "never miscompile" contract.

A candidate that lowers but fuses to more than one SDSC kernel is reported as
`compiles=pass, n_kernels>1` with a note — *not* a lowering failure. The Phase 6
multi-kernel payload (v2) + launch DAG that would carry it is host-verified and
device-compiled, but a *resident* ≥2-kernel single-delegate partition is
unreachable end-to-end today (an upstream torch-spyre fusion property), so such a
candidate still must not be promoted until that upstream split lands.
