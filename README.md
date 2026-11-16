# ExecuTorch-Spyre

An [ExecuTorch](https://github.com/pytorch/executorch) backend delegate for the IBM Spyre accelerator,
built on [torch-spyre](https://github.com/torch-spyre/torch-spyre) (Apache-2.0).

## What this is

ExecuTorch-Spyre lets you export a PyTorch model to a portable `.pte` file and run it on IBM Spyre
hardware via ExecuTorch's `BackendInterface` — `torch.export` → Edge IR → Spyre partitioner
→ `preprocess()` → `.pte` → C++ runtime on device.

## Two backends

| Backend | Path | Status | Description |
|---------|------|--------|-------------|
| **spyre** (SDSC) | `executorch/backends/spyre/` | **Release candidate (v0.3.0rc1)** | SuperDSC codegen path. Device-verified: tiny single-tile `GraniteForCausalLM` runs end-to-end (`max_abs_err 0.1875`, MATCH). |
| **spyre2** (Triton/KTIR) | `executorch/backends/spyre2/` | **Early access** | Triton→TTIR→KTIR path. Proven off-device (915 backend tests, `ktir_cpu` MATCH) and first `mm` on real Spyre via SDSC bridge (`max_abs_err 0.09375`, EXIT 0). On-device KTIR execution is roadmap. |

## Quick start

To go from a Spyre pod to a working Granite run, follow these three pages in order. Each step ends with a check.

1. [docs/setup_pod.md](docs/setup_pod.md): the pod, device access and the environment guardrails.
2. [docs/setup.md](docs/setup.md): build and verify `spyre_llm_runner`.
3. [docs/hfa-decode.md](docs/hfa-decode.md): export Granite and run it.

The Jupyter demo, the host-only check, the generic export CLI and the coverage-runner device run are in
[docs/other-workflows.md](docs/other-workflows.md).

## Version pins

| Component | Version |
|-----------|---------|
| ExecuTorch | v1.5.1 |
| PyTorch | 2.13.0+cpu |
| torch-spyre | v0.5.0-rc.3 |
| Python | 3.12 |

The supported torch-spyre version is **v0.5.0-rc.3**, pinned by the
`third_party/torch-spyre` git submodule and by the SDK 2.0 image. No installable
torch-spyre package exists yet, so `pyproject.toml` only declares a lower bound in the
optional `spyre` extra; the submodule pin is authoritative. The host-only check does
not need torch-spyre. See [`docs/version-matrix.md`](docs/version-matrix.md) for the
full compatibility matrix and
[Partition modes](docs/operator-guide.md#partition-modes) for the production
partitioning contract (`fallbacklist` default, `allowlist` for host-only work).

## Documentation

- [`docs/setup_pod.md`](docs/setup_pod.md) — the pod, device access, environment guardrails
- [`docs/setup.md`](docs/setup.md) — build and verify `spyre_llm_runner`
- [`docs/hfa-decode.md`](docs/hfa-decode.md) — export and run Granite; design in [`docs/hfa-design.md`](docs/hfa-design.md)
- [`docs/setup-reference.md`](docs/setup-reference.md) — host-only tests, generic export CLI, coverage-runner build, Spyre-stack rules in full
- [`docs/other-workflows.md`](docs/other-workflows.md) — Jupyter demo, host-only check, export your own model, coverage-runner device run
- [`docs/end-to-end-jupyter-demo-plan.md`](docs/end-to-end-jupyter-demo-plan.md) — plan for a single-notebook `.pte` export and Spyre execution demo
- [`docs/operator-guide.md`](docs/operator-guide.md) — adding and validating operators
- [`docs/debugging-guide.md`](docs/debugging-guide.md) — MOCK vs PF, mixed-dtype, scramble, CPython dep
- [`docs/version-matrix.md`](docs/version-matrix.md) — version compatibility table
- [`docs/spyre2-early-access.md`](docs/spyre2-early-access.md) — spyre2 capabilities, limitations, roadmap
- [`docs/pre-scheduling-passes.md`](docs/pre-scheduling-passes.md) — adding ExecuTorch passes to torch-spyre's pre-scheduling pipeline

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). DCO sign-off is required on every commit (`git commit -s`).
Credit for work that predates the initial import commit is recorded in [`CONTRIBUTORS.md`](CONTRIBUTORS.md).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
