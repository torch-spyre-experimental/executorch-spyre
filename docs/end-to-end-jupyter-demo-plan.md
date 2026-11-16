<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# End-to-end Jupyter demo plan

## Goal

Provide one Jupyter notebook that runs on a host with an IBM Spyre and
demonstrates the complete released `spyre` (SDSC) path:

```text
PyTorch model
  -> torch.export
  -> ExecuTorch Edge IR
  -> SpyrePartitioner / SpyreBackend.preprocess
  -> Spyre-delegated .pte
  -> ExecuTorch C++ runtime
  -> Spyre device
  -> comparison with a CPU reference
```

The demo is complete only when the notebook both creates the `.pte` and executes
that same file on a physical Spyre device. Merely printing a delegated graph or
successfully compiling a JobPlan is not an end-to-end result.

## Proposed deliverables

1. `examples/notebooks/executorch_spyre_end_to_end.ipynb` — the single, linear
   demo notebook, committed with outputs cleared.
2. `examples/notebooks/README.md` — prerequisites, how to start Jupyter on the
   Spyre host, expected duration, and the success signature.
3. A small notebook test that checks cell syntax and runs the device-independent
   cells in CI. The device portion remains an explicitly marked hardware gate.
4. A link to the notebook from the top-level `README.md` and `docs/setup.md`.

The first version should use the stable `backends/spyre` SDSC backend. The
`backends/spyre2` Triton/KTIR backend is early access and does not yet provide the
same native on-device KTIR execution path, so it should not be mixed into this
demo.

## Demo workload

Use the existing deterministic 64x64 fp16 matmul from
`examples/matmul/export_matmul.py`.

This is preferable for the introductory end-to-end notebook because it:

- delegates the whole graph, making the Spyre boundary unambiguous;
- compiles and runs quickly relative to the tiny Granite example;
- has a checked CPU fp16 reference and a device-verified tolerance;
- uses the existing `spyre_runner`, avoiding notebook-only runtime logic; and
- isolates export/runtime setup failures from model coverage or CPU fallback.

The tiny Granite example remains the follow-on demo linked at the end of the
notebook, rather than the first proof that the toolchain works.

## Notebook flow

### 1. Explain the contract

State what will be produced and what constitutes success. The final observable
must be:

```text
PASS: Spyre delegate matches CPU fp16 reference
```

The notebook should identify the two distinct phases:

- **AOT preprocessing:** compile the delegated graph and package the result in
  `spyre_matmul.pte`.
- **Runtime execution:** load that `.pte` through ExecuTorch's C++ runtime and
  execute the delegate on `spyre:0`.

### 2. Preflight the host

Fail early with actionable messages. Check:

- Python 3.12, PyTorch 2.11, and ExecuTorch 1.2;
- `torch_spyre` and `backends.spyre` imports;
- `FLEX_DEVICE=PF` (never silently accept the MOCK backend);
- repository and writable artifact paths;
- the pinned versions in `compatibility-manifest.json`; and
- device visibility by moving a small fp16 tensor to `spyre:0`.

Use `/work/executorch-spyre` as the documented default repository location, but
derive the repository root when the notebook lives in the checkout and allow a
single `REPO_ROOT` override cell.

### 3. Define data and the CPU reference

Show the small `MatMul` module in the notebook. Generate `a` and `b` with a fixed
seed and compute the reference as:

```python
ref = (a.float() @ b.float()).to(torch.float16)
```

Keep `M = K = N = 64`. Display shapes, dtypes, and the first few reference
values so the input contract is visible.

### 4. Preprocess and generate the `.pte`

Show the important API calls directly rather than hiding the central operation
behind a shell command:

```python
ep = torch.export.export(model, (a, b))
edge = to_edge(ep)
delegated = edge.to_backend(SpyrePartitioner())
program = delegated.to_executorch()
```

Write `program.buffer` to `artifacts/spyre_matmul.pte`. Also write the deterministic
`a.bin`, `b.bin`, and `ref.bin` files consumed by the existing C++ runner. Reuse
small serialization helpers from `examples/matmul/export_matmul.py` where doing
so keeps the notebook readable.

This cell is the preprocessing step: `SpyreBackend.preprocess()` invokes the Spyre
compiler, captures the SDSC/JobPlan payload, and embeds it in the `.pte`.

### 5. Inspect the artifact

Before execution, assert and display:

- a delegate node is present in the lowered graph;
- `spyre_matmul.pte` exists and is non-empty;
- all four runtime artifacts exist; and
- artifact paths and byte sizes.

Do not label graph inspection alone as proof that the model ran on the Spyre.

### 6. Prepare the C++ runtime

The Python wheel can export a program but does not provide the custom
backend-linked executable needed to run this `.pte`. Use the repository's
`scripts/build_device_runtime.sh`, which also builds `spyre_runner`.

Notebook behavior should be:

1. Use an existing `/work/build-spyre/spyre_runner` when present.
2. Otherwise present and run one explicit build cell.
3. Stream build output and stop on a non-zero return code.
4. Explain that the first build takes roughly 10–20 minutes and later runs are
   cached through `/work/et-install`.

For a polished hosted demo, prebuild the runtime in the image or during host
setup. Keep the build cell so the notebook remains reproducible from a standard
Spyre SDK host.

### 7. Execute the `.pte` on Spyre

Run the exact artifact directory produced above:

```python
subprocess.run(
    ["/work/build-spyre/spyre_runner", str(artifacts_dir)],
    env={**os.environ, "FLEX_DEVICE": "PF"},
    check=True,
    text=True,
    capture_output=True,
)
```

Print stdout and assert both a zero exit code and the runner's `PASS` marker.
The runner loads `spyre_matmul.pte`, supplies `a.bin` and `b.bin`, invokes the
registered `SpyreBackend`, and compares the result with `ref.bin` using
`atol=1e-1, rtol=2e-2`.

### 8. Summarize and offer the next step

End with a compact result table containing:

- `.pte` path and size;
- backend (`spyre` / SDSC);
- hardware (`spyre:0`, `FLEX_DEVICE=PF`);
- runner return code;
- maximum absolute error parsed from runner output; and
- overall PASS/FAIL.

Point to `docs/setup-reference.md#exporting-any-model-generic-cli` for exporting a model-level workload.

## Environment and launch contract

The notebook runs on the Spyre-equipped host or pod; it is not exported on a
laptop and copied elsewhere for this first version. Keeping preprocessing and
runtime on the same pinned SDK host removes ABI and compiler-version ambiguity.

Required environment:

```bash
export HOME=/work
export XDG_CACHE_HOME=/work/.cache
export FLEX_DEVICE=PF
export PYTHONPATH=/work/executorch-spyre:$PYTHONPATH
```

Launch Jupyter from the torch-spyre Python environment, adding Jupyter only if
it is not already in the host image. Access should use SSH or cluster port
forwarding; do not expose an unauthenticated public Jupyter route.

## Error handling

Each major cell must stop immediately and explain the likely fix:

| Failure | Notebook response |
|---|---|
| `FLEX_DEVICE` is unset or not `PF` | Stop before compilation or execution; show the required export. |
| `torch_spyre` cannot import | Report the expected SDK venv and pinned Python/PyTorch versions. |
| `spyre:0` allocation fails | Report that no physical device is assigned or the runtime is unavailable. |
| Preprocessing fails | Preserve compiler output and identify the AOT preprocessing cell as the failure. |
| No delegate node | Stop; the `.pte` would not demonstrate Spyre delegation. |
| Runner is absent | Direct the user to the runtime build cell. |
| Runner returns non-zero or lacks `PASS` | Print stdout/stderr and fail the notebook. |

## Verification and acceptance criteria

The implementation is ready when all of the following are true:

- A fresh Spyre host can run the notebook top-to-bottom using documented pins.
- The notebook visibly performs `torch.export -> to_edge -> to_backend ->
  to_executorch` and writes a non-empty `.pte`.
- The lowered graph contains an ExecuTorch delegate call.
- The C++ runtime loads the newly created `.pte` rather than a checked-in
  fixture.
- Execution is guarded by `FLEX_DEVICE=PF` and reaches the physical Spyre
  device.
- Device output matches the deterministic CPU reference and prints the runner's
  `PASS` line.
- Re-running the notebook is idempotent; artifacts may be overwritten and the
  completed runtime build is reused.
- Notebook outputs, device logs, tokens, and generated `.pte` files are not
  committed.
- Device-independent notebook cells pass CI, while a separately documented
  Spyre smoke job runs the full notebook when hardware is available.

## Implementation order

1. Add the notebook using the existing matmul exporter and `spyre_runner`
   contracts without changing backend behavior.
2. Add the runtime build/cache cell and preflight diagnostics.
3. Run from a clean Spyre host and record only the expected output signature in
   documentation, keeping notebook outputs cleared.
4. Add the device-independent notebook CI check.
5. Link the notebook from the main setup documentation.
6. Optionally add a second notebook for tiny Granite after the matmul notebook
   is stable; do not expand the first notebook's acceptance gate.
