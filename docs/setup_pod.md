<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Setting up a Spyre pod

> This content was moved here from [setup.md](setup.md) and [hfa-decode.md](hfa-decode.md) during the docs reorganization; it is unchanged. To build the runner, continue with [setup.md](setup.md); to export and run Granite, see [hfa-decode.md](hfa-decode.md).

Creating the pod itself follows the cluster administrator's procedure; this page starts from a pod you can already log in to.

## Device environment (Spyre pod)

### What the pod provides, and what you build

The pod image ships **one** Python environment, and everything else lives under
`/work`, which is an `emptyDir`: deleting the pod wipes it. This guide does not
depend on any other pre-existing venv, so it works on any pod layout, including
one where you cannot create `/work` (see
[Using a different directory layout](#using-a-different-directory-layout)).

| Piece | Where it comes from |
|-------|---------------------|
| `/home/senuser/torch-spyre/.venv` | **In the image.** torch-spyre SDK venv: torch 2.13.0, `torch_spyre`, the DeepTools compiler and device driver, plus `cmake` and `ninja`. Use it for `preprocess()`, every step that calls the Spyre compiler, and the C++ build. |
| ExecuTorch Python (`executorch.exir`, `to_edge`) | The SDK venv or, if `executorch` is not importable there, a small overlay directory you create (step 2 below). |
| ExecuTorch C++ libraries and `spyre_coverage_runner` | **Built by you** with `scripts/build_device_runtime.sh` (step 3 below). |

> **Older docs mention `/work/et-venv` and `/work/aot-venv`.** Those were
> hand-built on an earlier dev pod (ExecuTorch v1.2.0, torch 2.11.0) and are not
> part of the image, so they will not exist on a fresh pod. The current stack
> (ExecuTorch v1.5.1, torch 2.13.0) does not need them: `et-venv` only supplied
> ExecuTorch core and `cmake`, both of which the SDK venv or the build script now
> provide, and `aot-venv` was a `--no-deps` overlay, recreated in step 2.

### Bring up a fresh pod

Run these once per pod, after exporting the guardrails below.

**1. Check the SDK venv.**

```bash
VENV=/home/senuser/torch-spyre/.venv
$VENV/bin/python -c "import torch, torch_spyre; print(torch.__version__)"   # expect 2.13.x
$VENV/bin/cmake --version && $VENV/bin/ninja --version
$VENV/bin/python -c "from executorch.exir import to_edge; print('executorch AOT OK')"
```

If the last command succeeds, skip step 2.

**2. Only if `executorch` is not importable: create the AOT overlay.** Install
the wheel into a plain directory with `--no-deps`, so pip never touches the
image's torch (the version-matrix explains why that torch must not change):

```bash
python3.12 -m pip install --no-deps --target /work/aot-overlay "executorch==1.5.1"
export PYTHONPATH="/work/aot-overlay:$PYTHONPATH"
$VENV/bin/python -c "from executorch.exir import to_edge; print('AOT overlay OK')"
```

If the import check fails with `ModuleNotFoundError` for a pure-Python module,
install that module the same way (`--no-deps --target /work/aot-overlay`). The
earlier v1.2.0 overlay needed `ruamel.yaml`, `ruamel.yaml.clib`, `parameterized`
and `flatbuffers`. Do not install anything that would pull in `torch`.

**3. Build the C++ runtime.** `scripts/build_device_runtime.sh` clones
ExecuTorch v1.5.1, builds and installs it with the SDK venv's `cmake`, and
builds `spyre_coverage_runner`. See [Building the C++ runtime](setup-reference.md#building-the-c-runtime).

After a pod deletion, repeat steps 2 and 3. The SDK venv in the image is
unaffected.

### Using a different directory layout

Nothing here requires the `/work` prefix. Point the build script at any writable
directory with its environment variables (`REPO`, `ET_SRC`, `ET_INSTALL`,
`RUN_BUILD`, `VENV`, `HOME_DIR`, `XDG_CACHE_HOME`), and substitute your own path
for `/work/aot-overlay` and `/work/executorch-spyre` in the commands in this guide:

```bash
REPO=/tmp/work/executorch-spyre ET_SRC=/tmp/work/src/executorch \
ET_INSTALL=/tmp/work/et-install RUN_BUILD=/tmp/work/build-spyre \
HOME_DIR=/tmp/work bash scripts/build_device_runtime.sh
```

### Mandatory environment guardrails

Set these variables in **every** pod shell before running any delegate code:

```bash
export HOME=/work
export XDG_CACHE_HOME=/work/.cache
export FLEX_DEVICE=PF
```

**`FLEX_DEVICE=PF` is mandatory.** Without it the stack silently falls back to a
mock compute stub whose outputs are garbage. This is the single most common source
of silent failures. See the [debugging guide](debugging-guide.md#mock-vs-pf-device)
for details.

Also put the repo root on `PYTHONPATH` (the `backends` package is an implicit
namespace package — no `__init__.py`):

```bash
export PYTHONPATH="/work/executorch-spyre:$PYTHONPATH"
```

To use the [generic export CLI](setup-reference.md#exporting-any-model-generic-cli) on the pod,
install the repo editable into the venv that runs `preprocess()`. `--no-deps`
keeps the image's torch build; `--no-build-isolation` avoids fetching
setuptools if the pod has no PyPI access:

```bash
/home/senuser/torch-spyre/.venv/bin/pip install --no-deps --no-build-isolation -e /work/executorch-spyre
```

Run the physical-device regression suite explicitly; it is never part of the
host test command:

```bash
/home/senuser/torch-spyre/.venv/bin/python -m pytest \
  executorch/backends/spyre/test_spyre/ -m spyre
```

The suite performs a `spyre:0` allocation before compiling anything and fails
immediately if `FLEX_DEVICE` is not exactly `PF`.


### Profile script

The pod image's profile script (moved from `hfa-decode.md`, section 1.1):

```bash
source /etc/profile.d/ibm-aiu-setup.sh 2>/dev/null
```
