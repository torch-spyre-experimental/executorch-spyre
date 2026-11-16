<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Debugging guide

Common failure modes, their symptoms, and how to fix or work around them.

---

## MOCK vs PF device

**Symptom:** The model runs without error, produces output that looks plausible, but
the values are wrong — for example, logits that pass a shape check but fail a
numerical match.

**Cause:** `FLEX_DEVICE` is unset (or set to a value other than `PF`). When
`FLEX_DEVICE` is not `PF`, the Spyre stack silently falls back to a mock compute stub
that mimics the delegate protocol but computes garbage. There is no error, no warning,
and no indication that a real device was not used.

**Fix:** Always set the following before running any delegate code on a pod:

```bash
export HOME=/work
export XDG_CACHE_HOME=/work/.cache
export FLEX_DEVICE=PF
```

This is the single most common source of silent failures. If you see unexpected
numerical results on a pod, verify `FLEX_DEVICE` first.

---

## Mixed-dtype tensor bridge

**Symptom:** A model with integer inputs (e.g. token index tensors passed to
`aten::embedding`) errors at the delegate boundary with a dtype mismatch.

**Background:** The delegate handles `Bool`, `Byte`, and `int64` inputs by routing
them through the `to_c10` bridge in the C++ runtime before dispatching to the device.
Operations such as `aten::embedding` are integer table lookups with no device kernel
and are expected to fall back to CPU; the mixed-dtype bridge ensures the AOT framing
can still pass integer-typed activation tensors across the delegate boundary for ops
that do have device support.

If you encounter dtype errors on a model with mixed-dtype inputs, check the `to_c10`
bridge in `CONTRIBUTING.md` for the list of handled types and the dispatch logic.
Adding a new dtype case follows the same pattern as the existing Bool/Byte entries.

---

## Multi-tile attention scramble

**Symptom:** The coverage match gate fails with a `max_abs_err` of approximately 85
on an attention output, when the expected error for fp16 compute should be ≈ 0.2 or
less.

**Background:** At `hidden` or `embed` dimension ≥ 128, or at ≥ 2 attention heads,
the SDSC path writes the attention output in tile-major order — `[ntile, S, 64]` —
which is a value-preserving permutation of the correct row-major `[S, E]` result. The
values are all present but in the wrong positions relative to the expected layout,
producing a large absolute error.

This is an upstream SDSC limitation, not a bug in this delegate. Single-tile attention
(`embed = head_dim = 64`, 1 head) is not affected and produces numerically correct
output.

**Workaround:** The partitioner enforces a hardware layout gate via
`_writes_scrambled_output`: any delegate subgraph whose output is wider than 64
elements at the host boundary is kept on CPU rather than delegated. This means the
>64-wide attention output and the wide vocabulary-projection tail in language models
both fall back to CPU automatically, without any manual intervention.

If you are adding an op and the match gate fails with `max_abs_err ≈ 85`, check
whether the op produces a wide output that reaches the host. If so, verify that
`_writes_scrambled_output` fires correctly for that op's output shape.

---

## Embedded-CPython dependency

**Symptom:** Confusion about why the C++ runtime links Python or appears to import
`torch_spyre` at runtime.

**Background:** `SpyreBackend.cpp::ensure_runtime_started()` embeds CPython and
executes:

```python
import torch_spyre
torch_spyre.start_runtime()
```

This installs the PrivateUse1 dispatch surface (including `aten::_copy_from`) that
the device runtime depends on. The torch-spyre device bring-up is wired through its
`_autoload` entry point and pybind symbols; there is no C++-callable equivalent. The
embedded interpreter therefore requires `torch` and `torch_spyre` to be importable at
runtime — point the `PYTHONPATH` to the torch-spyre SDK venv's site-packages before
running the C++ runner:

```bash
# $VENV is the torch-spyre SDK venv; $TORCH_SPYRE_SRC the directory that contains the
# torch_spyre/ package (on the pod image: /home/senuser/torch-spyre).
SP=$($VENV/bin/python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="$SP:$TORCH_SPYRE_SRC:$PYTHONPATH"
```

Without this, the runner aborts at startup with
`ModuleNotFoundError: No module named 'torch'`.

This dependency is documented as intentional. Removing it requires an upstream
torch-spyre change to provide a C++-callable `spyre::initializeDeviceRuntime()`
entry point.

---

## PyTorch version drift

**Symptom:** The C++ build links successfully but produces runtime ABI errors, or
`torch.export` produces a `.pte` that the C++ runtime cannot load.

**Background:** ExecuTorch v1.5.1 does not declare a torch pin, so `torch==2.13.0` must be
installed explicitly. The torch-spyre SDK venv ships torch 2.13.0, matching the `_C.so`
ABI. If a different torch version ends up in the environment (for example from running
the full `install_executorch.sh` examples step, which pulls in `transformers` and
`torchtune`), the build or runtime will break.

**Fix:** Install ExecuTorch core only; do not run the examples step. See
[version-matrix.md](version-matrix.md) for the exact pins.

---

## Build troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `find_package(Torch)` finds an unexpected torch version | The examples step of `install_executorch.sh` pulled in transitive deps | Reinstall `torch==2.13.0+cpu` in the build venv, or use the torch-spyre SDK venv |
| Runner link fails with undefined `flex::` / `libspyre_comms` references, or the runner aborts with `undefined symbol: _ZN4flex...` | The runner links a different `libflex` / DeepTools / comms than the `_C.so` Python imports (usually `/opt/ibm/spyre` against a torch-spyre built for another stack) | Derive the stack from the `_C.so`, configure into an empty build dir, and verify with `scripts/spyre_env.sh --check <runner>` (see [setup.md](setup-reference.md#the-spyre-stack-must-match-torch-spyres-_cso)) |
| `TORCH_SPYRE_C_SO` not found | Wrong or stale path | Do not guess: run `scripts/spyre_env.sh --cmake-args` and pass its output to CMake |
| `SPYRE_RUNTIME_DIR` not found, or headers/libs missing under it | Stale value in the build dir, or a stack different from the `_C.so`'s | Reconfigure into an empty build dir with the values from `scripts/spyre_env.sh --cmake-args` |
| `ModuleNotFoundError: No module named 'torch'` at runner start | Embedded interpreter can't find torch | Set `PYTHONPATH` to include the torch-spyre SDK venv's `site-packages` (see [Embedded-CPython dependency](#embedded-cpython-dependency)) |

---

## Device contention

Each Spyre pod holds exactly one of a shared pool of Spyre devices. The coverage
harness spawns a subprocess for the compile step, and parallel pytest execution can
create contention on that single device.

**Fix:** Run device tests sequentially rather than in parallel:

```bash
pytest -n0 executorch/backends/spyre/test/
```

Avoid running multiple device test jobs from the same pod simultaneously.

Also: **delete the pod when you are not actively using it** so the device slot is
available to others. Pod storage is ephemeral (`emptyDir`); it is wiped on delete.

---

## See also

- [setup_pod.md](setup_pod.md) — environment setup, including the mandatory guardrails.
- [setup.md](setup.md) — building `spyre_llm_runner`.
- [operator-guide.md](operator-guide.md) — numerical tolerance and the op support
  boundary.
- [version-matrix.md](version-matrix.md) — pinned versions and upgrade pitfalls.
