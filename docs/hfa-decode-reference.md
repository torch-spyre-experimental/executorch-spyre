<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# HFA decode: environment, build and run (sections 1.1-1.3)

> This content was moved here from [hfa-decode.md](hfa-decode.md) during the docs reorganization; it is unchanged. The current build steps are in [setup.md](setup.md); the export and run steps are in [hfa-decode.md](hfa-decode.md).

## 1. Quick Start

### 1.1 Environment Setup

Configure your shell environment before exporting or running models:

```bash
export VENV=/path/to/.venv
export PATH="$VENV/bin:$PATH"
export ET_HOME=/path/to/executorch-spyre
export PYTHONPATH="$ET_HOME:/path/to/torch-spyre"
export HF_ADAPTERS_ROOT=/path/to/hf-adapters

export FLEX_DEVICE=PF
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# Which Spyre stack was YOUR torch-spyre built against? Its lib dirs must come FIRST on
# LD_LIBRARY_PATH (the profile script above only knows the pod image's /opt/ibm stack).
#   torch-spyre from the pod image : leave SPYRE_STACK empty
#   torch-spyre built against a    : SPYRE_STACK=/path/to/sentient   (the checkout that holds
#   sentient checkout                runtime/, deeptools/ and spyre_comms/)
# Never copy a fixed /opt/ibm list: it breaks a torch-spyre built against another stack.
export SPYRE_STACK=${SPYRE_STACK:-}
if [ -n "$SPYRE_STACK" ]; then
  export LD_LIBRARY_PATH="$SPYRE_STACK/spyre_comms/lib:$SPYRE_STACK/runtime/lib:$SPYRE_STACK/deeptools/lib:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$VENV/lib64/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Must print the path of torch_spyre/_C.so before you build or run anything (see setup.md,
# "The Spyre stack must match torch-spyre's _C.so"). "undefined symbol: _ZN4flex..." means
# the wrong stack comes first on LD_LIBRARY_PATH: fix SPYRE_STACK above.
python -c "import torch, torch_spyre._C as c; print(c.__file__)"

cd $ET_HOME
```

---

### 1.2 Export a Model to PTE

Use [`export_hfa.py`](../executorch/backends/spyre/export_hfa.py:1) with `--llm-wrapper` to export a Hugging Face checkpoint to an ExecuTorch `.pte` binary compatible with standard LLM runners:

```bash
# Set your model checkpoint path
export CKPT=/path/to/granite-3.3-8b-instruct
export OUT_DIR=/tmp/et-decode-llm

# Export decode program with StandardLLMWrapper
python executorch/backends/spyre/export_hfa.py \
    --ckpt $CKPT \
    --mode decode \
    --llm-wrapper \
    --seq 128 \
    --stage pte \
    --out-dir $OUT_DIR
```

The output file will be `$OUT_DIR/spyre_hfa_decode_llm.pte`.

---

### 1.3 Run Inference

`spyre_llm_runner` drives the `--llm-wrapper` PTE through the standard
ExecuTorch LLM runner stack (`TextLLMRunner` / `create_text_llm_runner`) and
generates text from a prompt.  It is **not** built by default (CMake option
`SPYRE_BUILD_LLM_RUNNER`, default `OFF`) because it needs an ExecuTorch
*installation* (`cmake --install`) with the LLM runner extension.

```text
executorch-spyre ($ET_HOME)
└── third_party/executorch ($ET_SRC)
         │
         │ build + install
         ▼
$HOME/et-install-llm ($ET_INSTALL)
         │
         ▼
spyre_llm_runner
```

`ET_SRC` is the ExecuTorch source tree. `ET_INSTALL` is the installation prefix
generated from `ET_SRC` via `cmake --install` and does not need to be located
under the source tree.

#### Prerequisites

The ExecuTorch install at `$ET_INSTALL` must have been configured with:

```text
-DEXECUTORCH_BUILD_EXTENSION_LLM=ON
-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON
-DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
```

in addition to the options used for the base install (data loader, tensor,
portable kernels).  Such an install provides `extension_llm_runner`,
`extension_llm_sampler`, and the separate `tokenizers` package (see
[Building an LLM-runner-enabled ExecuTorch install](setup-reference.md#building-an-llm-runner-enabled-executorch-install)).
A minimal base install without LLM extensions will fail during configuration
with a message listing the missing targets.

#### Variables

| Variable | Meaning |
|---|---|
| `ET_HOME` | `executorch-spyre` repository root. |
| `ET_SRC` | ExecuTorch source tree (`$ET_HOME/third_party/executorch`). |
| `ET_INSTALL` | Destination install directory produced by `cmake --install` from `ET_SRC`. Must contain `lib64/cmake/ExecuTorch/executorch-config.cmake` (or `lib/...`) and `lib64/cmake/tokenizers/tokenizers-config.cmake`. |
| `RUN_BUILD` | Output directory for this runner build. |

```bash
export ET_HOME=/path/to/executorch-spyre
export ET_SRC=$ET_HOME/third_party/executorch
export ET_INSTALL=$HOME/et-install-llm
export RUN_BUILD=/tmp/build-spyre-llm

# Fail early if the install layout is not what CMake expects
ls $ET_INSTALL/lib64/cmake/ExecuTorch/executorch-config.cmake \
   $ET_INSTALL/lib64/cmake/tokenizers/tokenizers-config.cmake 2>/dev/null || \
ls $ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake \
   $ET_INSTALL/lib/cmake/tokenizers/tokenizers-config.cmake

# Discover the cmake config dir (works for both lib/ and lib64/ layouts).
export ET_CMAKE_DIR=$(find "$ET_INSTALL" -name executorch-config.cmake | head -1 | xargs dirname)

# The runner must link the same libflex / DeepTools / libspyre_comms as the
# torch_spyre/_C.so that Python imports. `source` derives them from that _C.so and
# exports TORCH_SPYRE_C_SO, TORCH_SPYRE_CSRC and SPYRE_{RUNTIME,DEEPTOOLS,SENLIB,COMMS}_DIR
# into this shell (setup.md, "The Spyre stack must match torch-spyre's _C.so").
# It fails, and exports nothing, if the import check above does not pass.
source $ET_HOME/scripts/spyre_env.sh
env | grep -E '^(TORCH_SPYRE_C_SO|TORCH_SPYRE_CSRC|SPYRE_RUNTIME_DIR|SPYRE_DEEPTOOLS_DIR|SPYRE_SENLIB_DIR|SPYRE_COMMS_DIR)='   # six lines expected

# Always configure into an EMPTY build directory (an old CMakeCache.txt keeps its values).
rm -rf "${RUN_BUILD:?}"

# Configure with the opt-in flag. CMake reads the six variables from the environment and
# prints each as "-- Spyre stack: NAME = value   [env]"; with none set it stops with a
# FATAL_ERROR that says how to derive them. A value left in an old CMakeCache.txt is
# printed as [cache] and, if it differs from the environment, stops the configure.
cmake -S "$ET_HOME/executorch/backends/spyre/spyre_runner" -B "$RUN_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ET_INSTALL;$(python -c 'import torch,os; print(os.path.dirname(torch.__file__))')/share/cmake" \
    -Dexecutorch_DIR="$ET_CMAKE_DIR" \
    -DEXECUTORCH_ROOT="$ET_SRC" \
    -DSPYRE_RUNNER_WITH_PORTABLE_OPS=ON \
    -DSPYRE_BUILD_LLM_RUNNER=ON

cmake --build "$RUN_BUILD" --target spyre_llm_runner

# Verify: the runner and _C.so must resolve the same libflex and libspyre_comms.
"$ET_HOME/scripts/spyre_env.sh" --check "$RUN_BUILD/spyre_llm_runner"

# Run inference with text prompt
"$RUN_BUILD/spyre_llm_runner" \
    --pte $OUT_DIR/spyre_hfa_decode_llm.pte \
    --tokenizer $CKPT/tokenizer.json \
    --prompt "Write a short poem about computing:" \
    --max-new 32
```
