<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Other workflows

> This content was moved here from [README.md](../README.md) during the docs reorganization; it is unchanged. For the Granite quick start, see [setup_pod.md](setup_pod.md), [setup.md](setup.md) and [hfa-decode.md](hfa-decode.md).

## Quick start

### End-to-end Jupyter demo on Spyre

The [demo notebooks](../examples/notebooks/README.md) demonstrate generating a Spyre-delegated `.pte`
([Part 1: 01_generate_pte.ipynb](../examples/notebooks/01_generate_pte.ipynb)) and executing that
same artifact on a physical Spyre device ([Part 2: 02_run_on_device.ipynb](../examples/notebooks/02_run_on_device.ipynb)).
See the [notebook instructions](../examples/notebooks/README.md).

### Host-only sanity check (no Spyre device required)

```bash
python3.12 -m venv .venv-hosttest
.venv-hosttest/bin/pip install "executorch==1.5.1" "torch==2.13.0+cpu" pytest
export PYTHONPATH="$PWD:$PYTHONPATH" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
.venv-hosttest/bin/python -m pytest -p no:cacheprovider executorch/backends/spyre/test/
```

The host suite pins `EXECUTORCH_SPYRE_PARTITION_MODE=allowlist` itself; see
[Partition modes](operator-guide.md#partition-modes) for the production contract
(`fallbacklist` default, `allowlist` for host-only work).

Expected: **at least 99 passed, at most 2 skipped** after
`pip install --no-deps -e .`. Physical-device tests are collected separately
from `executorch/backends/spyre/test_spyre/` and never skip inside the host suite.

### Export your own model

```bash
.venv-hosttest/bin/pip install --no-deps -e .
# Host has no torch-spyre, so use the allowlist partition mode (the production default,
# fallbacklist, needs torch-spyre's fallback_ops and fails closed without it).
EXECUTORCH_SPYRE_PARTITION_MODE=allowlist .venv-hosttest/bin/executorch-spyre-export --model linear --inspect   # device-free partition report
```

`--model` also takes `pkg.module:Class`, `file.py:Class`, or a Hugging Face model id. See
[Exporting any model](setup-reference.md#exporting-any-model-generic-cli).

### Device run (requires Spyre pod access)

See [`setup_pod.md`](setup_pod.md) and [`setup-reference.md`](setup-reference.md) for environment setup, then:

```bash
# On the Spyre dev pod, in the exporter venv, with FLEX_DEVICE=PF exported:
nohup executorch-spyre-export --model linear --out-dir /work/artifacts-linear \
  > /work/run.log 2>&1 &
tail -f /work/run.log
( cd /work/artifacts-linear && /work/build-spyre/spyre_coverage_runner . )
```
