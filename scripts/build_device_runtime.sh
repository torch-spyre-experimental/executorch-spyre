#!/usr/bin/env bash
# build_device_runtime.sh — rebuild the ExecuTorch C++ device runtime +
# spyre_coverage_runner, from scratch, in one command.
#
# WHY THIS EXISTS
#   The dev pod's /work is an emptyDir: a pod DELETE wipes the ExecuTorch build,
#   the runner binary, and the et-* venvs. The AOT Python stack survives (it
#   lives in the image's torch-spyre venv), but the C++ runtime that actually
#   executes a .pte on device does not. This script reconstructs it
#   deterministically so a device run is never blocked on "remember how we built
#   the runtime last time".
#
#   Environment setup: see docs/setup_pod.md for the pod and docs/setup.md for the runner build.
#
#   ONE deliberate design choice: PORTABLE OPS ARE BUILT **ON**. Any model with
#   CPU-fallback ops (e.g. aten::sin, aten::embedding) requires portable_ops_lib
#   at runtime. Without it those nodes fail with 0x14 (OperatorMissing). See
#   docs/setup-reference.md — "Building the C++ runtime" for details.
#
# PINS (see compatibility-manifest.json)
#   ExecuTorch v1.5.1 == torch 2.13.0 (SDK image torch-spyre 2.0).
#
# USAGE
#   Runs long (ET C++ build ~10-20 min on 128 cpu). Drive it detached and poll:
#     nohup bash scripts/build_device_runtime.sh \
#       > /work/build-device-runtime.log 2>&1 &
#   Then watch for the final "DEVICE RUNTIME READY" / "RC=" marker in the log.
#
#   VENV: the torch-spyre venv (torch 2.13 + cmake + ninja). There is no default:
#   set VENV, or activate the venv so `python` on PATH is its interpreter (VENV is then
#   its sys.prefix). The script fails clearly if neither gives a usable venv.
#
#   SPYRE STACK: the runner must link the same libflex / DeepTools / libspyre_comms
#   as the torch_spyre/_C.so that $VENV/bin/python imports. Stage [6] derives them
#   with scripts/spyre_env.sh and checks the built runner against that _C.so (see
#   docs/setup-reference.md, "The Spyre stack must match torch-spyre's _C.so"). LD_LIBRARY_PATH
#   must therefore already let `import torch_spyre._C` work. If TORCH_SPYRE_C_SO,
#   TORCH_SPYRE_CSRC and SPYRE_{RUNTIME,DEEPTOOLS,SENLIB,COMMS}_DIR are all set in the
#   environment they are used as given (nothing is derived); a partial set fails.
#
#   PATH OVERRIDES (env): REPO, ET_SRC, ET_INSTALL, RUN_BUILD, VENV, HOME_DIR,
#   XDG_CACHE_HOME. Defaults are the /work dev-pod layout. Override to retarget
#   to a different layout, e.g.:
#     REPO=/tmp/work/executorch-spyre ET_SRC=/tmp/work/src/executorch \
#     ET_INSTALL=/tmp/work/et-install RUN_BUILD=/tmp/work/build-spyre \
#     HOME_DIR=/tmp/work bash scripts/build_device_runtime.sh
set -euo pipefail

# Resolve repo root from this script's location (scripts/).
_SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)

# --- paths (pod) -------------------------------------------------------------
# All overridable via env so a different pod layout can retarget them WITHOUT
# editing this script. The defaults are the canonical dev-pod paths.
# VENV (device-linked torch 2.13 + cmake/ninja) has no default: a hard-coded pod path
# silently selected the wrong stack. Take it from the environment, else from the
# active python.
if [ -z "${VENV:-}" ]; then
  if _active_py=$(command -v python 2>/dev/null) && [ -n "$_active_py" ] \
     && VENV=$("$_active_py" -c 'import sys; print(sys.prefix)' 2>/dev/null) && [ -n "$VENV" ]; then
    echo "VENV not set; using the active python's prefix: $VENV ($_active_py)"
  else
    echo "FATAL: VENV is not set and no 'python' on PATH to derive it from." >&2
    echo "       Set VENV=/path/to/torch-spyre/.venv, or activate that venv first." >&2
    exit 2
  fi
fi
PY=$VENV/bin/python
CMAKE=$VENV/bin/cmake
REPO=${REPO:-"$_SCRIPT_DIR/.."}
ET_SRC=${ET_SRC:-${HOME:-/tmp/work}/src/executorch}     # ET v1.5.1 source (submodules initialised)
ET_BUILD=$ET_SRC/cmake-out
ET_INSTALL=${ET_INSTALL:-${HOME:-/tmp/work}/et-install} # ExecuTorch CMake CONFIG + static libs land here
RUN_BUILD=${RUN_BUILD:-${HOME:-/tmp/work}/build-spyre}    # spyre_runner build dir
ET_TAG=v1.5.1

export HOME=${HOME_DIR:-/work}                 # HOME_DIR override (bare HOME is easily inherited)
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$HOME/.cache}
mkdir -p "$XDG_CACHE_HOME"

echo "=== [0] preflight ==="
test -x "$PY"    || { echo "FATAL: no python at $PY"; exit 2; }
test -x "$CMAKE" || { echo "FATAL: no cmake at $CMAKE"; exit 2; }
"$PY" -c "import torch; assert torch.__version__.startswith('2.13'), torch.__version__; print('torch', torch.__version__)"

echo "=== [1] ET source present? (clone v1.5.1 if missing) ==="
if [ ! -f "$ET_SRC/CMakeLists.txt" ]; then
  echo "  cloning ExecuTorch $ET_TAG -> $ET_SRC"
  mkdir -p "$(dirname "$ET_SRC")"
  git clone --branch "$ET_TAG" --depth 1 https://github.com/pytorch/executorch.git "$ET_SRC"
  ( cd "$ET_SRC" && git submodule update --init --recursive --depth 1 --jobs 8 )
else
  echo "  found existing $ET_SRC ($(du -sh "$ET_SRC" | cut -f1))"
fi

echo "=== [2] buck2 needs python 'zstd' in the build venv ==="
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
if ! "$PY" -c "import zstd" 2>/dev/null; then
  echo "  installing zstd -> $SP"
  pip3 install --no-input --target "$SP" zstd
fi
"$PY" -c "import zstd; print('  zstd OK')"

echo "=== [3] configure ExecuTorch (PORTABLE OPS ON) ==="
# Idempotent: skip the (long) ET build if a complete install is already present.
ET_CFG=$(find "$ET_INSTALL" -name executorch-config.cmake 2>/dev/null | head -1 || true)
if [ -n "$ET_CFG" ] && [ -f "$ET_INSTALL/lib64/libportable_ops_lib.a" -o -f "$ET_INSTALL/lib/libportable_ops_lib.a" ]; then
  echo "  ExecuTorch already installed ($ET_CFG) — skipping stages [3][4]"
  SKIP_ET=1
fi
if [ -z "${SKIP_ET:-}" ]; then
rm -rf "${ET_BUILD:?}" "${ET_INSTALL:?}"
"$CMAKE" -S "$ET_SRC" -B "$ET_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ET_INSTALL" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DPYTHON_EXECUTABLE="$PY" \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_PORTABLE_OPS=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON

echo "=== [4] build + install ExecuTorch ==="
"$CMAKE" --build "$ET_BUILD" -j"$(nproc)" --target install
fi  # end SKIP_ET guard

echo "=== [5] sanity: the libs the runner links must exist ==="
for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
           libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
  found=$(find "$ET_INSTALL" -name "$lib" | head -1)
  [ -n "$found" ] && echo "  OK  $lib -> $found" || { echo "  MISSING $lib"; exit 3; }
done
# ET v1.2.0 installs to lib64/ on this image -> discover the config dir, don't assume lib/.
ET_CFG=$(find "$ET_INSTALL" -name executorch-config.cmake 2>/dev/null | head -1)
[ -n "$ET_CFG" ] && echo "  OK  executorch-config.cmake -> $ET_CFG" \
  || { echo "  MISSING executorch-config.cmake"; exit 3; }
ET_CMAKE_DIR=$(dirname "$ET_CFG")

echo "=== [6] build the spyre_coverage_runner (portable ops linked ON) ==="
TORCH_CMAKE=$("$PY" -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share','cmake'))")
# The Spyre stack comes from the _C.so that $PY imports, never from a default.
echo "  deriving the Spyre stack from the torch_spyre/_C.so imported by $PY"
SPYRE_ARGS=$(PYTHON="$PY" bash "$_SCRIPT_DIR/spyre_env.sh" --cmake-args) || {
  echo "FATAL: could not derive the Spyre stack (see the message above)." >&2
  echo "       Make 'import torch_spyre._C' work first: LD_LIBRARY_PATH must list the Spyre" >&2
  echo "       stack your torch-spyre was built against before any other (docs/setup.md, step 1)." >&2
  exit 5
}
mapfile -t SPYRE_CMAKE_ARGS <<<"$SPYRE_ARGS"
printf '  %s\n' "${SPYRE_CMAKE_ARGS[@]}"
rm -rf "${RUN_BUILD:?}"                # never reuse a build dir: a stale CMakeCache.txt keeps old values
"$CMAKE" -S "$REPO/executorch/backends/spyre/spyre_runner" -B "$RUN_BUILD" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PY" \
  -DCMAKE_PREFIX_PATH="$ET_INSTALL;$TORCH_CMAKE" \
  -Dexecutorch_DIR="$ET_CMAKE_DIR" \
  -DEXECUTORCH_ROOT="$ET_SRC" \
  -DSPYRE_RUNNER_WITH_PORTABLE_OPS=ON \
  "${SPYRE_CMAKE_ARGS[@]}"
"$CMAKE" --build "$RUN_BUILD" -j"$(nproc)"

BIN="$RUN_BUILD/spyre_coverage_runner"
test -x "$BIN" || { echo "FATAL: runner not built at $BIN"; exit 4; }
echo "  checking that the runner and _C.so resolve the same libflex / libspyre_comms"
PYTHON="$PY" bash "$_SCRIPT_DIR/spyre_env.sh" --check "$BIN" || {
  echo "FATAL: $BIN does not resolve the same Spyre libraries as _C.so." >&2
  exit 5
}
echo "=== [7] confirm aten::sin.out is linked into the runner ==="
if strings "$BIN" | grep -q "aten::sin.out"; then
  echo "  OK  aten::sin.out present in runner symbol/string table"
else
  echo "  WARN aten::sin.out string not found via 'strings' (op registry is runtime; not necessarily fatal)"
fi

echo "DEVICE RUNTIME READY: $BIN"
echo "RC=0"
