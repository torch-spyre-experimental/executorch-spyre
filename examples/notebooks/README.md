# ExecuTorch-Spyre Notebooks

This directory contains interactive Jupyter notebooks demonstrating the end-to-end workflow for compiling and executing PyTorch models on IBM Spyre hardware using ExecuTorch.

## Overview: Two-Stage Execution Workflow

The demo is split into two distinct stages:

1. **Part 1 — Ahead-of-Time (AOT) `.pte` Generation ([`01_generate_pte.ipynb`](01_generate_pte.ipynb))**:
   - Takes a PyTorch model and partitions supported subgraphs for the Spyre backend (`SpyrePartitioner`).
   - Runs AOT preprocessing (`SpyreBackend.preprocess`) using the Spyre compiler (DeepTools) to generate compiled payloads and packs the final `.pte` artifact alongside input/reference tensors.
   - Can run on any environment with `torch-spyre` / DeepTools (compiler mode with `FLEX_DEVICE=MOCK`).

2. **Part 2 — On-Device Execution ([`02_run_on_device.ipynb`](02_run_on_device.ipynb))**:
   - Loads the generated `.pte` artifact and input tensors.
   - Executes through the ExecuTorch C++ runtime on physical Spyre hardware (`FLEX_DEVICE=PF`, `spyre:0`).
   - Compares on-device output with deterministic CPU fp16 reference data.
   - **Requires physical Spyre hardware access** (Spyre dev pod).

---

## Environment Setup & Prerequisites

### Hardware Requirements
- **Part 1 (Export / AOT Preprocessing)**: Requires the `torch-spyre` compiler stack (available in Spyre dev pods or systems with the Spyre SDK).
- **Part 2 (Device Execution)**: Requires access to physical IBM Spyre hardware (`/dev/vfio/*` or PCIe device nodes) in a Spyre dev pod.

### Software & Version Pinning
Ensure your environment uses the supported version pins:
- **Python**: `3.12`
- **PyTorch**: `2.13.0`
- **ExecuTorch**: `1.5.1`

---

## Step-by-Step Setup

### 1. Create a Dedicated Python 3.12 Virtual Environment

Create and activate a standalone Python 3.12 virtual environment (or use the pre-configured `/home/senuser/torch-spyre/.venv` on the Spyre dev pod):

```bash
# Create virtual environment
python3.12 -m venv .venv-notebook
source .venv-notebook/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install "torch==2.13.0" "executorch==1.5.1" ipykernel jupyterlab
```

### 2. Install ExecuTorch-Spyre in Editable Mode

Install the `executorch-spyre` repository package into your virtual environment:

```bash
cd /path/to/executorch-spyre
pip install -e .
```

### 3. Register the Jupyter Kernel

Register the dedicated environment as a named Jupyter kernel:

```bash
python -m ipykernel install --user --name executorch-spyre --display-name "Python 3 (ExecuTorch-Spyre SDK)"
```

### 4. Launch JupyterLab

Start JupyterLab:

```bash
jupyter lab --ip 0.0.0.0 --no-browser
```

> **Note**: Use SSH tunneling or cluster port forwarding to access Jupyter securely.

---

## Running the Notebooks

1. Open [`01_generate_pte.ipynb`](01_generate_pte.ipynb).
2. Ensure the kernel is set to **Python 3 (ExecuTorch-Spyre SDK)** (`executorch-spyre`).
3. Run all cells in `01_generate_pte.ipynb`. Cell 6 will release the Python runtime.
4. **Important**: Shut down the Part 1 kernel (**Kernel → Shut Down Kernel**) before proceeding. The physical VFIO device file descriptor is only released when the Python process exits.
5. Open [`02_run_on_device.ipynb`](02_run_on_device.ipynb), select the same kernel, and run all cells.
6. The test passes when the runner outputs:
   ```text
   PASS: Spyre delegate matches CPU fp16 reference
   ```

---

## Environment Variable Overrides (Optional)

You can customize scratch and artifact paths via environment variables:

| Variable | Default | Description |
|---|---|---|
| `EXECUTORCH_SPYRE_REPO_ROOT` | Current working directory or repo root | Root directory of the `executorch-spyre` repository |
| `EXECUTORCH_SPYRE_WORK_HOME` | `Path.cwd() / "work"` (or `/tmp/work`) | Base working directory for build artifacts |
| `EXECUTORCH_SPYRE_ARTIFACTS` | `$EXECUTORCH_SPYRE_WORK_HOME/artifacts` | Output directory for `.pte` and tensor binaries |
| `EXECUTORCH_SPYRE_RUN_BUILD` | `$EXECUTORCH_SPYRE_WORK_HOME/build-spyre` | Directory containing built `spyre_runner` binary |
| `FLEX_DEVICE` | `PF` (for device execution) / `MOCK` (for compiler-only) | Device driver execution mode |
