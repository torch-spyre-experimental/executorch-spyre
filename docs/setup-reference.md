<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

> This content was moved here from [setup.md](setup.md) during the docs reorganization; it is unchanged. The runner build is in [setup.md](setup.md); pod setup is in [setup_pod.md](setup_pod.md).

# Setup guide

Two distinct environments are needed depending on what you want to do:

- **Host-only** — a laptop or CI machine with no Spyre hardware. Sufficient for
  running the Python AOT unit tests (partitioner, payload framing, coverage harness).
- **Device (Spyre pod)** — required for the DeepTools compile step, the C++ device
  runtime, and any end-to-end model run.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | **3.12** (the version CI and the device pod use. The macOS arm64 wheel is built for macOS 14 or later; older macOS fails to resolve. Use `python3.12`.) |
| ExecuTorch | `1.5.1` |
| PyTorch | `2.13.0+cpu` (must match the torch-spyre `_C.so` ABI on-device) |
| torch-spyre | `v0.5.0-rc.3` (SDK 2.0 image; pinned by the `third_party/torch-spyre` submodule). Not needed for the host-only check. |
| Spyre device access | A Spyre development pod. Building and running on device currently requires a development build of the IBM Spyre Software Stack. Internal build instructions are available to IBM employees through internal documentation channels (see [torch-spyre](https://github.com/torch-spyre/torch-spyre)). |

---

## Cloning the repository

```bash
git clone --recurse-submodules git@github.com:torch-spyre-experimental/executorch-spyre.git
cd executorch-spyre
```

`--recurse-submodules` populates `third_party/torch-spyre` (pinned to `v0.5.0-rc.3`,
the supported torch-spyre version) and `third_party/executorch` — needed for the C++ build.
The `third_party/executorch` submodule is pinned to the same ExecuTorch version
(`1.5.1`) as the `executorch` Python package; `scripts/build_device_runtime.sh`
clones ExecuTorch `v1.5.1` itself.
The authoritative versions are in [`version-matrix.md`](version-matrix.md).
If you already cloned without the flag, run:

```bash
git submodule update --init --recursive
```

---

## Host-only check (no device)

This 60-second sanity check confirms the AOT logic — partitioner allowlist, payload
framing, coverage harness — before spending a device slot. It requires only
`torch`, `executorch`, and `pytest`; no Spyre hardware, no pod.

### Create the test venv

```bash

python3.12 -m venv .venv-hosttest
./.venv-hosttest/bin/python -m pip install --upgrade pip
./.venv-hosttest/bin/python -m pip install "executorch==1.5.1" "torch==2.13.0" pytest
# The delegate itself, editable: makes `executorch.backends.spyre` importable
# from anywhere and installs the `executorch-spyre-export` command.
./.venv-hosttest/bin/python -m pip install --no-deps -e .
```

> **Python version note.** On macOS, Homebrew's default `python3` may be 3.13 or
> later. Install `python3.12` explicitly (`brew install python@3.12`) and use it
> here. The `executorch==1.5.1` macOS wheel targets macOS 14 or later.

### Run the tests

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

./.venv-hosttest/bin/python -m pytest -p no:cacheprovider executorch/backends/spyre/test/
```

Use `python -m pytest` (not a bare `pytest`) so the test runner belongs to the
same interpreter that has `torch` and `executorch`.

**Expected result:** at least `99 passed`, with at most `2 skipped` for optional
host dependencies. Physical-device tests are not collected by this command.

The suite mocks the compiler seam and contains no `torch_spyre` imports, so it
never touches hardware. Physical-device regressions live under
`executorch/backends/spyre/test_spyre/` and require `FLEX_DEVICE=PF`.

---

## Exporting any model (generic CLI)

`executorch-spyre-export` (or `python -m executorch.backends.spyre.export`, which is
the same CLI) lowers a model to a Spyre-delegated `.pte`:
`torch.export` → `to_edge` → `.to_backend(SpyrePartitioner())` →
`.to_executorch()`. Whatever the partitioner does not delegate falls back to
the portable CPU kernels, and the CLI reports it. It is available after
`pip install -e .` (see above).

`--model` takes one of three kinds of spec (`--source` overrides the guess):

| Spec | Source | Example inputs |
|------|--------|----------------|
| A name from upstream ExecuTorch's `MODEL_NAME_TO_MODEL` (`linear`, `add`, `softmax`, `mv2`, …) | `registry` | The model's `get_example_inputs()` |
| `pkg.module:Attr` or `path/to/file.py:Attr`, where `Attr` is an `nn.Module` class, a factory, an instance, or an `EagerModelBase` | `module` | `get_example_inputs()` if defined, else `--input` |
| A Hugging Face model id or a local checkpoint dir | `hf` | Random `input_ids [--batch, --seq]`; the output is the logits |

Floating-point weights and inputs are cast to `--dtype` (default `fp16`). Some
catalogue models need extra packages (`torchvision`, `transformers`); HF models
need `transformers`.

```bash
# Device-free: trace and partition, print delegated / CPU-fallback ops, write nothing.
executorch-spyre-export --model linear --inspect
executorch-spyre-export --model executorch.backends.spyre.export_mixed:MixedUnsupported \
    --input 1,64 --inspect

# Build the .pte (runs the SDSC compile, so on the pod with FLEX_DEVICE=PF).
executorch-spyre-export --model ./my_model.py:Net --model-kwargs '{"dim": 64}' \
    --input 1,64 --out-dir /work/artifacts-net
executorch-spyre-export --model ibm-granite/granite-3.3-2b-instruct --seq 128 \
    --out-dir /work/artifacts-granite
```

`--input SHAPE[:DTYPE]` is repeatable, one per forward argument, for example
`--input 1x16:int64 --input 1,16,64`. The tensors are random: floats from
`randn`, integers and bools 0/1. The supported DTYPEs are `fp16`, `fp32`,
`int64` and `bool`.

The CLI writes these files to `--out-dir`:

| File | Contents |
|------|----------|
| `spyre_<name>.pte` | The program |
| `input<i>.bin` | Each input as raw bytes |
| `ref.bin` | CPU reference for output 0, computed in fp32 and cast to `--dtype` (`--no-reference` skips it) |
| `manifest.txt` | Manifest for the generic runner |
| `<name>_inputs.json` | Shapes, dtypes and the delegated / CPU-fallback op sets |

Run the result on the device with the manifest-driven runner. It compares
output 0 against `ref.bin` within `atol=1e-1, rtol=2e-2`:

```bash
( cd /work/artifacts-net && /work/build-spyre/spyre_coverage_runner . )
```

`aten.silu` is kept un-decomposed by default (`--preserve-ops`), so its fp16->fp32
opmath decomposition does not insert CPU-falling-back casts. If the partitioner leaves a preserved op on CPU, the CLI
refuses to build, because portable has no kernel for it. Re-run with
`--preserve-ops none` in that case. Pass `-v` to log each CPU-fallback node,
and see `--help` for the remaining options.

For models with repeated decoder blocks, each block is compiled as its own
delegate by default, which keeps full-depth models under the compile timeout.
`EXECUTORCH_SPYRE_PARTITION_GRANULARITY=model` restores the single-delegate behaviour; see
[Partition granularity](hfa-design.md#partition-granularity).

Limitations: export is static-shape only. Neither `--buckets` nor KV-cache /
`export_llm` LLM export is supported yet. HF models are traced as one
full-sequence forward pass without a cache.

---

## Building the C++ runtime

The C++ build requires the ExecuTorch source tree, the Spyre SDK stack, and the
torch-spyre `_C.so`. The canonical one-shot script handles everything
(ET source clone, CMake configure + install, runner compile):

```bash
nohup bash scripts/build_device_runtime.sh \
  > /work/build-device-runtime.log 2>&1 &
tail -f /work/build-device-runtime.log
```

Paths default to the standard pod layout and are individually overridable via
environment variables (`REPO`, `ET_SRC`, `ET_INSTALL`, `RUN_BUILD`, `HOME_DIR`,
`XDG_CACHE_HOME`). `VENV` has no default: set it, or activate the torch-spyre venv
(its `sys.prefix` is then used). The Spyre stack is derived with
`scripts/spyre_env.sh` (see the next section), and the built runner is checked against
the `_C.so`; `LD_LIBRARY_PATH` must therefore already let `import torch_spyre._C`
work. See the script header for details.

> **Why `EXECUTORCH_BUILD_PORTABLE_OPS=ON`?** Any model that has CPU-fallback ops
> (for example `aten::sin`, `aten::embedding`, or the softmax-fed context matmul in
> Granite) requires the portable kernels library at runtime. Without it those ops
> fail with error `0x14` (`OperatorMissing`). The script always links
> `portable_ops_lib` into `spyre_coverage_runner` for this reason.

### The Spyre stack must match torch-spyre's `_C.so`

`spyre_llm_runner` and `spyre_coverage_runner` run in the same process as
`torch_spyre/_C.so` and call into it. **The runner must link the same `libflex`,
DeepTools and `libspyre_comms` as the `_C.so` that Python imports.** If it links a
different copy (typically the `/opt/ibm/spyre` stack, while your torch-spyre was
built against a newer one), the build fails with undefined references to
`flex::` / `libspyre_comms` symbols, or the runner fails at start-up with
`undefined symbol`. Do not take these paths from a default or from another pod's
notes; derive them from the `_C.so`:

1. **Make `import torch_spyre._C` work first.** `LD_LIBRARY_PATH` must list the
   Spyre stack your torch-spyre was built against, before any other. On the pod
   image that is `/opt/ibm/spyre/...` (the profile script sets it). For a torch-spyre
   built against another checkout, put that checkout's `runtime/lib`,
   `deeptools/lib` and `spyre_comms/lib` first. Check:

   ```bash
   # Pod-image torch-spyre: leave SPYRE_STACK empty. torch-spyre built against a
   # sentient checkout: SPYRE_STACK=/path/to/sentient.
   export SPYRE_STACK=${SPYRE_STACK:-}
   if [ -n "$SPYRE_STACK" ]; then
     export LD_LIBRARY_PATH="$SPYRE_STACK/spyre_comms/lib:$SPYRE_STACK/runtime/lib:$SPYRE_STACK/deeptools/lib:$LD_LIBRARY_PATH"
   fi
   python -c "import torch, torch_spyre._C as c; print(c.__file__)"
   ```

   An `undefined symbol: _ZN4flex...` here means the wrong stack comes first.

2. **Derive the variables from that `_C.so`** with `scripts/spyre_env.sh`:

   ```bash
   source scripts/spyre_env.sh              # exports the six variables into THIS shell
   scripts/spyre_env.sh --print             # human-readable table (prints, exports nothing)
   scripts/spyre_env.sh --cmake-args        # -D... arguments, one per line
   eval "$(scripts/spyre_env.sh)"           # same as `source`, from a script or subshell
   ```

   `source` exports; **executing** the script (`scripts/spyre_env.sh`, without
   `source`) cannot change your shell, so it prints `export ...` lines instead.
   Either way diagnostics go to stderr and a failure leaves your shell untouched.
   Then configure without any `-D` for these: CMake reads them from the environment
   and prints each as `-- Spyre stack: NAME = value   [env]`.

   | Variable | Derived as |
   |---|---|
   | `TORCH_SPYRE_C_SO` | the single `_C*.so` in the package `importlib.util.find_spec("torch_spyre")` locates (no import needed) |
   | `TORCH_SPYRE_CSRC` | `csrc/` next to it (`prepare_kernel.h` must exist) |
   | `SPYRE_RUNTIME_DIR` | install root of the `libflex.so` that `ldd <_C.so>` resolves |
   | `SPYRE_DEEPTOOLS_DIR` | install root of `libspyrecode-host-functions.so` (not `libutil.so`: several DeepTools trees ship a `libutil.so`) |
   | `SPYRE_SENLIB_DIR` | install root of `libsenlib.so` |
   | `SPYRE_COMMS_DIR` | install root of `libspyre_comms.so.1` |

   "Install root" is the parent of the `lib` directory holding the library. The
   script also imports `torch_spyre._C` under the current `LD_LIBRARY_PATH` and fails
   if that does not work. If all six variables are already set it uses them as given.
   If only some are set it stops, because that is usually stale values from an
   earlier shell.

3. **Always configure into an empty build directory.** CMake keeps the values it
   was given in `CMakeCache.txt`, and a cached value wins over the environment, so
   reusing a directory silently keeps an earlier stack. If you do, configure stops
   with "stale value from an existing CMakeCache.txt; reconfigure with `cmake --fresh`
   or an empty build dir" and lists the cached and the environment value of each
   variable that differs. `cmake --fresh` needs CMake 3.24 or newer (the CMakeLists
   here declare 3.19, so it is not guaranteed); `rm -rf` works with every version:

   ```bash
   rm -rf "${RUN_BUILD:?}"
   ```

4. **Verify after the build.** The runner and `_C.so` must resolve the same
   libraries:

   ```bash
   scripts/spyre_env.sh --check "$RUN_BUILD/spyre_llm_runner"
   ```

   It prints, for `libflex`, `libspyre_comms` and DeepTools, the file `ldd` resolves
   for the `_C.so` and for the runner, and exits non-zero on any difference. The
   same comparison by hand: `ldd "$TORCH_SPYRE_C_SO" | grep -E 'libflex|libspyre_comms'`
   against `ldd "$RUN_BUILD/spyre_llm_runner" | grep -E 'libflex|libspyre_comms'`.

**What `LD_LIBRARY_PATH` still does.** The runner executables are linked with a
`BUILD_RPATH` (a `DT_RPATH`) made of the derived `SPYRE_*_DIR/lib` directories, the
`_C.so` directory and the directory of the `libaiupti` that `ldd <_C.so>` resolves.
The loader searches it before `LD_LIBRARY_PATH`, so the **runner is independent of
`LD_LIBRARY_PATH`**: `env -u LD_LIBRARY_PATH ldd "$RUN_BUILD/spyre_llm_runner"` still
resolves the stack it was configured against, and a stray `/opt/ibm` entry cannot
substitute another `libflex`. **Python is not**: the export step (`preprocess()`
imports `torch_spyre._C`, which has no RPATH of its own) and step 1 above still depend
on `LD_LIBRARY_PATH`. `--check` compares the runner with the `_C.so` as loaded under
your current `LD_LIBRARY_PATH`, so it also reports a shell whose Python would load a
different stack than the runner.

At configure time CMake prints every `Spyre stack:` value with its source (`-D` for
a `-D` on the command line, `env` for the environment, `cache` for a value already in
`CMakeCache.txt`) and fails if `ldd <_C.so>` resolves `libflex` or `libspyre_comms`
from a directory other than `${SPYRE_RUNTIME_DIR}/lib` / `${SPYRE_COMMS_DIR}/lib`.
The same variables are accepted from the environment when no `-D` is given, so a
stale `export` in your shell is visible in that printout.

For a **host-only C++ build** (the payload-parser unit tests, no device libraries):

```bash
cmake -S executorch/backends/spyre -B /tmp/spyre-hosttest-build \
  -DEXECUTORCH_BUILD_SPYRE=OFF \
  -DEXECUTORCH_BUILD_SPYRE_TESTS=ON
cmake --build /tmp/spyre-hosttest-build
ctest --test-dir /tmp/spyre-hosttest-build --output-on-failure
```

On macOS, `nlohmann/json` and GoogleTest can be installed via Homebrew
(`brew install nlohmann-json googletest`). On a pod without system GoogleTest,
CMake falls back to the copy vendored in the ExecuTorch source tree (no network
fetch needed).

---

## Building an LLM-runner-enabled ExecuTorch install

The standard `build_device_runtime.sh` script produces only the core ExecuTorch
runtime plus `spyre_coverage_runner`. Building `spyre_llm_runner` requires an
additional four CMake flags that pull in the tokenizer library, the sampler, and
the full text-LLM-runner stack.

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

The relationship between the directories is:
- **`ET_HOME`**: `executorch-spyre` repository root.
- **`ET_SRC`**: ExecuTorch source tree submodule (`$ET_HOME/third_party/executorch`).
- **`ET_INSTALL`**: ExecuTorch destination install directory produced by `cmake --install` from `ET_SRC` (e.g. `$HOME/et-install-llm`).

### Required extra CMake flags (on top of the base build)

```text
-DEXECUTORCH_BUILD_EXTENSION_LLM=ON
-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON
```

`EXECUTORCH_BUILD_EXTENSION_LLM` is the gate that builds the tokenizers
sub-library (sentencepiece + re2 + PCRE2 + abseil).
`EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER` depends on it and builds
`extension_llm_runner` (the text-LLM runner stack) plus `extension_llm_sampler`.
Both flags must be set together; CMake enforces this dependency.

### Complete configure command

Set the path variables before running CMake. `ET_SRC` points to the ExecuTorch
source tree submodule (`$ET_HOME/third_party/executorch`); `ET_INSTALL` is the
destination directory where the LLM-enabled ExecuTorch headers and libraries will
be installed:

```bash
export ET_HOME=/path/to/executorch-spyre
export ET_SRC=$ET_HOME/third_party/executorch   # ExecuTorch source tree
export ET_INSTALL=$HOME/et-install-llm          # destination for the new install
```

```bash
# Variables assumed from the environment:
#   $VENV       — path to the torch-spyre SDK venv
#   $ET_SRC     — ExecuTorch source tree ($ET_HOME/third_party/executorch)
#   $ET_INSTALL — install prefix; set above

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
```

### Complete build and install command

```bash
cmake --build "$ET_SRC/cmake-out-llm" -j$(nproc) --target install
```

The `--target install` combines build and install in one step. Use `-j$(nproc)`
(or a fixed count such as `-j8`) to parallelise the compilation; the tokenizer
third-party libraries (abseil, re2, sentencepiece, PCRE2) contribute the most
build time.

### Verify the install tree

After install, confirm the three LLM-specific components are present:

```bash
for lib in libextension_llm_runner.a libextension_llm_sampler.a libtokenizers.a; do
  found=$(find "$ET_INSTALL" -name "$lib" | head -1)
  [ -n "$found" ] && echo "OK   $lib -> $found" \
                   || echo "MISS $lib — install incomplete"
done
```

Expected output (paths use `lib64` on RHEL-family hosts; `lib` elsewhere):

```
OK   libextension_llm_runner.a  -> <install>/lib64/libextension_llm_runner.a
OK   libextension_llm_sampler.a -> <install>/lib64/libextension_llm_sampler.a
OK   libtokenizers.a            -> <install>/lib64/libtokenizers.a
```

The install tree also includes two CMake config trees used by downstream
consumers:

```
<install>/lib64/cmake/ExecuTorch/executorch-config.cmake   ← executorch_DIR
<install>/lib64/cmake/tokenizers/tokenizers-config.cmake   ← found via CMAKE_PREFIX_PATH
```

### Building `spyre_llm_runner` against the install

Once `$ET_INSTALL` contains the LLM-runner libraries, add
`-DSPYRE_BUILD_LLM_RUNNER=ON` to the runner configure step:

```bash
export RUN_BUILD=/tmp/build-spyre-llm

# Discover the cmake config dir (works for both lib/ and lib64/ layouts).
export ET_CMAKE_DIR=$(find "$ET_INSTALL" -name executorch-config.cmake | head -1 | xargs dirname)

# Torch cmake tree (needed for find_package(Torch)).
export TORCH_CMAKE=$(
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  "$VENV/bin/python" -c \
  "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))"
)

# Spyre stack of the _C.so that Python imports (see the section above): exports the six
# TORCH_SPYRE_* / SPYRE_* variables into this shell. `python -c "import torch_spyre._C"`
# must already work (LD_LIBRARY_PATH, step 1 above).
source "$ET_HOME/scripts/spyre_env.sh"
env | grep -E '^(TORCH_SPYRE_C_SO|TORCH_SPYRE_CSRC|SPYRE_RUNTIME_DIR|SPYRE_DEEPTOOLS_DIR|SPYRE_SENLIB_DIR|SPYRE_COMMS_DIR)='   # six lines

# Always an empty build directory (an old CMakeCache.txt keeps its values; configure
# stops with "stale value from an existing CMakeCache.txt" if they differ).
rm -rf "${RUN_BUILD:?}"

cmake -S "$ET_HOME/executorch/backends/spyre/spyre_runner" -B "$RUN_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$ET_INSTALL;$TORCH_CMAKE" \
  -Dexecutorch_DIR="$ET_CMAKE_DIR" \
  -DEXECUTORCH_ROOT="$ET_SRC" \
  -DSPYRE_RUNNER_WITH_PORTABLE_OPS=ON \
  -DSPYRE_BUILD_LLM_RUNNER=ON

cmake --build "$RUN_BUILD" -j$(nproc)

# The runner and _C.so must resolve the same libflex / libspyre_comms.
"$ET_HOME/scripts/spyre_env.sh" --check "$RUN_BUILD/spyre_llm_runner"
```

CMake prints `spyre_llm_runner: enabled (executorch_DIR=…)` during configure if
all three imported targets (`extension_llm_runner`, `extension_llm_sampler`,
`tokenizers::tokenizers`) are found. If any are missing it prints a
`FATAL_ERROR` that names each missing target and the install flags needed to
provide it.

> **`TORCH_DEVICE_BACKEND_AUTOLOAD=0`** — set this when invoking Python just to
> query the Torch cmake path. Without it, Python tries to import `torch_spyre`,
> which fails outside the full pod environment and causes the path query to
> return an empty string.

### Install tree summary

```
<install>/
├── include/
│   ├── executorch/…                  (ET runtime + extension headers)
│   └── pytorch/tokenizers/…          (tokenizer public headers)
└── lib64/                            (lib/ on non-RHEL hosts)
    ├── libexecutorch_core.a
    ├── libexecutorch.a
    ├── libportable_ops_lib.a
    ├── libportable_kernels.a
    ├── libextension_module_static.a
    ├── libextension_tensor.a
    ├── libextension_data_loader.a
    ├── libextension_flat_tensor.a
    ├── libextension_named_data_map.a
    ├── libextension_llm_runner.a      ← LLM runner stack
    ├── libextension_llm_sampler.a     ← sampler (temperature / top-p)
    ├── libtokenizers.a                ← HF / SPM / tiktoken tokenizer
    ├── libre2.a  libsentencepiece.a   ← tokenizer dependencies
    ├── libabsl_*.a  libpcre2-8.a      ← tokenizer dependencies
    └── cmake/
        ├── ExecuTorch/executorch-config.cmake
        └── tokenizers/tokenizers-config.cmake
```

---

## See also

- [version-matrix.md](version-matrix.md) — pinned component versions and known
  upgrade pitfalls.
- [operator-guide.md](operator-guide.md) — adding and validating new operators.
- [debugging-guide.md](debugging-guide.md) — common failure modes and fixes.
- [spyre2-early-access.md](spyre2-early-access.md) — Triton/KTIR codegen path (early
  access).
