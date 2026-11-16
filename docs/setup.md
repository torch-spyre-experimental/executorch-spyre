<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Building `spyre_llm_runner`

This page takes you from a Spyre pod to a built and verified `spyre_llm_runner`.
Every step ends with a check; do not go on until it passes. Then export and run a
model with [hfa-decode.md](hfa-decode.md).

- Need a pod, device access or the environment guardrails first? See [setup_pod.md](setup_pod.md).
- Everything else that used to be on this page (host-only tests, the generic export CLI,
  the coverage-runner build, the long form of the Spyre-stack rules) is in
  [setup-reference.md](setup-reference.md).

Use a fresh login shell (`env -i HOME=$HOME bash -l`). The clone must include the
submodules (`git clone --recurse-submodules ...`, or `git submodule update --init --recursive`).

## 0. Set the paths for your machine

This is the only block you edit.

```bash
export VENV=/path/to/.venv                  # the venv where `import torch_spyre` works
export ET_HOME=/path/to/executorch-spyre
export ET_SRC=$ET_HOME/third_party/executorch
export ET_INSTALL=$HOME/et-install-llm
export RUN_BUILD=/tmp/build-spyre-llm
export CKPT=/path/to/granite-3.3-8b-instruct
export TORCH_SPYRE_REPO=/path/to/torch-spyre
export SPYRE_STACK=${SPYRE_STACK:-}         # see the table
export PATH="$VENV/bin:$PATH"
```

Which Spyre stack was **your** torch-spyre built against? Its library directories must
come first on `LD_LIBRARY_PATH`.

| Your torch-spyre | `SPYRE_STACK` | Where the Spyre libraries come from |
|---|---|---|
| Installed in the pod image | leave empty | the pod's profile script puts `/opt/ibm/...` on `LD_LIBRARY_PATH` (see [setup_pod.md](setup_pod.md)) |
| Built against a sentient checkout | `/path/to/sentient` (the checkout that holds `runtime/`, `deeptools/` and `spyre_comms/`) | `$SPYRE_STACK/{spyre_comms,runtime,deeptools}/lib`, listed first |

Never copy a fixed `/opt/ibm` list: it breaks a torch-spyre built against another stack.

## 1. `LD_LIBRARY_PATH` and `import torch_spyre._C`

```bash
export PYTHONPATH="$ET_HOME:$TORCH_SPYRE_REPO"
export FLEX_DEVICE=PF
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

if [ -n "$SPYRE_STACK" ]; then
  export LD_LIBRARY_PATH="$SPYRE_STACK/spyre_comms/lib:$SPYRE_STACK/runtime/lib:$SPYRE_STACK/deeptools/lib:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$VENV/lib64/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"

python -c "import torch, torch_spyre._C as c; print(c.__file__)"
```

**Check:** the last command prints the path of `torch_spyre/_C...so`. An
`undefined symbol: _ZN4flex...` error means the wrong stack comes first: fix `SPYRE_STACK`
and repeat this step.

## 2. ExecuTorch install with the LLM runner

`spyre_llm_runner` needs an ExecuTorch **install** built with the LLM extensions.

```bash
cmake -S "$ET_SRC" -B "$ET_SRC/cmake-out-llm" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ET_INSTALL" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPYTHON_EXECUTABLE="$VENV/bin/python" \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON

cmake --build "$ET_SRC/cmake-out-llm" -j$(nproc) --target install
```

**Check:** all three libraries are found.

```bash
for lib in libextension_llm_runner.a libextension_llm_sampler.a libtokenizers.a; do
  found=$(find "$ET_INSTALL" -name "$lib" | head -1)
  [ -n "$found" ] && echo "OK   $lib -> $found" \
                   || echo "MISS $lib — install incomplete"
done
```

Expect three `OK` lines (`lib64/` on RHEL-family hosts, `lib/` elsewhere). The flags and the
install layout are explained in [setup-reference.md](setup-reference.md#building-an-llm-runner-enabled-executorch-install).
Skip this step if `$ET_INSTALL` already passes the check.

## 3. Derive the Spyre stack from `_C.so`

The runner must link the same libflex, DeepTools and libspyre_comms as the `_C.so` that
Python imports. Derive them; never type them.

```bash
source "$ET_HOME/scripts/spyre_env.sh"
env | grep -E '^(TORCH_SPYRE_C_SO|TORCH_SPYRE_CSRC|SPYRE_RUNTIME_DIR|SPYRE_DEEPTOOLS_DIR|SPYRE_SENLIB_DIR|SPYRE_COMMS_DIR)='
```

**Check:** exactly six lines are printed. It must be `source`, not `bash`: executing the
script only prints `export` lines and sets nothing in your shell.

## 4. Configure in an empty build directory

```bash
rm -rf "${RUN_BUILD:?}"      # an old CMakeCache.txt keeps its values

export ET_CMAKE_DIR=$(find "$ET_INSTALL" -name executorch-config.cmake | head -1 | xargs dirname)

cmake -S "$ET_HOME/executorch/backends/spyre/spyre_runner" -B "$RUN_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$ET_INSTALL;$(python -c 'import torch,os; print(os.path.dirname(torch.__file__))')/share/cmake" \
    -Dexecutorch_DIR="$ET_CMAKE_DIR" \
    -DEXECUTORCH_ROOT="$ET_SRC" \
    -DSPYRE_RUNNER_WITH_PORTABLE_OPS=ON \
    -DSPYRE_BUILD_LLM_RUNNER=ON
```

With CMake 3.24 or newer, `cmake --fresh -S ... -B "$RUN_BUILD" ...` does the same as the
`rm -rf`. (This repo's CMake files declare 3.19, so `rm -rf` is the form that works everywhere.)

**Check:** the output has six lines `-- Spyre stack: NAME = value   [env]` and
`spyre_llm_runner: enabled`. `[cache]` or `[-D]` instead of `[env]` means an old build
directory was reused: go back to the `rm -rf`.

## 5. Build

```bash
cmake --build "$RUN_BUILD" --target spyre_llm_runner
```

**Check:** the build finishes without errors; step 6 confirms the result.

## 6. Verify the runner against `_C.so`

```bash
"$ET_HOME/scripts/spyre_env.sh" --check "$RUN_BUILD/spyre_llm_runner"
```

**Check:** it reports that the runner and `_C.so` resolve the same libflex and
libspyre_comms. Then continue with [hfa-decode.md](hfa-decode.md); keep this shell open
(it holds the variables the next page uses).

## The three most common failures

| What you see | Cause | Fix |
|---|---|---|
| `Spyre stack: stale value from an existing CMakeCache.txt ...`, or `Inconsistent Spyre stack` listing `/home/senuser` or `/opt/ibm` paths | `$RUN_BUILD` holds the cache of an earlier configure | `rm -rf "${RUN_BUILD:?}"` (or `cmake --fresh` with CMake 3.24+), then redo step 4 |
| `undefined symbol: _ZN4flex...` when importing `torch_spyre._C`, or `ldd` shows `/opt/ibm` first | `/opt/ibm` comes before the sentient directories on `LD_LIBRARY_PATH` | Set `SPYRE_STACK` and redo step 1 in a fresh shell; the sentient directories must come first |
| Step 3 printed `export ...` lines but `env` shows nothing | The script was run with `bash`, not `source` | `source "$ET_HOME/scripts/spyre_env.sh"` |
