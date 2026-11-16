#!/usr/bin/env bash
# test_build_device_runtime.sh — hermetic unit tests for scripts/build_device_runtime.sh
#
# Tests run on any host with bash >= 4 and standard POSIX tools (mktemp, find,
# mkdir, etc.).  No Spyre pod, no device, no real cmake or python required.
# All external tools (python, cmake, git, pip3, strings, nproc, du) are replaced
# with minimal stubs that live on a prepended $PATH inside each test's temp dir.
#
# USAGE
#   bash scripts/tests/test_build_device_runtime.sh
#   # exits 0 on all pass, 1 on first failure (with details)
set -euo pipefail

# ---------------------------------------------------------------------------
# Framework (tiny, no dependencies)
# ---------------------------------------------------------------------------
PASS=0; FAIL=0
_FAILURES=()

pass() { PASS=$((PASS+1)); echo "  PASS  $1"; }
fail() {
  FAIL=$((FAIL+1))
  _FAILURES+=("$1: $2")
  echo "  FAIL  $1: $2"
}

assert_exit() {
  local name=$1 want=$2; shift 2
  local got
  got=$("$@" 2>&1; echo "EXIT:$?") || true
  local code=${got##*EXIT:}
  if [ "$code" = "$want" ]; then pass "$name"; else fail "$name" "expected exit $want, got $code"; fi
}

assert_output_contains() {
  local name=$1 pattern=$2; shift 2
  local out
  out=$("$@" 2>&1) || true
  if echo "$out" | grep -q "$pattern"; then
    pass "$name"
  else
    fail "$name" "expected pattern '$pattern' not found in output"
  fi
}

assert_output_not_contains() {
  local name=$1 pattern=$2; shift 2
  local out
  out=$("$@" 2>&1) || true
  if echo "$out" | grep -q "$pattern"; then
    fail "$name" "unexpected pattern '$pattern' found in output"
  else
    pass "$name"
  fi
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Escape hatch of scripts/spyre_env.sh: with all six variables set in the environment
# nothing is derived (the stub python above cannot import torch_spyre). T10 covers the
# derivation failure; the other tests cover the rest of the script.
SPYRE_VARS=(TORCH_SPYRE_C_SO=/stack/torch_spyre/_C.so
            TORCH_SPYRE_CSRC=/stack/torch_spyre/csrc
            SPYRE_RUNTIME_DIR=/stack/runtime
            SPYRE_DEEPTOOLS_DIR=/stack/deeptools
            SPYRE_SENLIB_DIR=/stack/senlib
            SPYRE_COMMS_DIR=/stack/comms)

# Resolve the script under test relative to this test file's location.
SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
SUT="$SCRIPT_DIR/../build_device_runtime.sh"

# Create a temp workspace; register cleanup.
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# make_stub_dir: create a directory of stub executables and return its path.
# Usage: DIR=$(make_stub_dir <name>)
make_stub_dir() {
  local d="$TMPROOT/$1/bin"
  mkdir -p "$d"
  echo "$d"
}

# make_python_stub: write a python stub that reports the given torch version.
# Usage: make_python_stub <bindir> <torch_version>
make_python_stub() {
  local d=$1 ver=$2
  cat > "$d/python" <<PYEOF
#!/usr/bin/env bash
# Minimal python stub
if echo "\$*" | grep -q "torch.__version__"; then
  echo "torch $ver"
  # Simulate the assert: fail if version doesn't start with 2.13
  echo "$ver" | grep -q "^2\\.13" || exit 1
  exit 0
fi
if echo "\$*" | grep -q "site.getsitepackages"; then
  echo "$TMPROOT/sitepackages"
  exit 0
fi
if echo "\$*" | grep -q "import zstd"; then
  exit 0
fi
if echo "\$*" | grep -q "import torch"; then
  echo "torch $ver"
  exit 0
fi
# torch cmake path query
if echo "\$*" | grep -q "share.*cmake"; then
  echo "$TMPROOT/torch_cmake"
  exit 0
fi
exit 0
PYEOF
  chmod +x "$d/python"
  mkdir -p "$TMPROOT/sitepackages" "$TMPROOT/torch_cmake"
}

# make_cmake_stub_ok: cmake that always succeeds.
# If --target install is passed, writes the expected sentinel files into ET_INSTALL.
make_cmake_stub_ok() {
  local d=$1
  cat > "$d/cmake" <<'CMEOF'
#!/usr/bin/env bash
# Minimal cmake stub — succeeds silently.
# If we see --target install, populate ET_INSTALL with sentinel files.
if echo "$*" | grep -q -- "--target install"; then
  if [ -n "${ET_INSTALL:-}" ]; then
    mkdir -p "$ET_INSTALL/lib"
    for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
               libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
      touch "$ET_INSTALL/lib/$lib"
    done
    mkdir -p "$ET_INSTALL/lib/cmake/ExecuTorch"
    touch "$ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake"
  fi
fi
# If second positional cmake call (build runner), create the binary.
if echo "$*" | grep -q "spyre_runner"; then
  : # handled by --build below
fi
if echo "$*" | grep -q -- "--build"; then
  # Extract -B value from the configure call isn't easy here; runner binary
  # is created by the make_runner_cmake_stub variant instead.
  :
fi
exit 0
CMEOF
  chmod +x "$d/cmake"
}

# make_runner_cmake_stub: cmake that on --build writes the runner binary into RUN_BUILD.
make_runner_cmake_stub() {
  local d=$1
  cat > "$d/cmake" <<'CMEOF'
#!/usr/bin/env bash
if echo "$*" | grep -q -- "--target install"; then
  if [ -n "${ET_INSTALL:-}" ]; then
    mkdir -p "$ET_INSTALL/lib"
    for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
               libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
      touch "$ET_INSTALL/lib/$lib"
    done
    mkdir -p "$ET_INSTALL/lib/cmake/ExecuTorch"
    touch "$ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake"
  fi
fi
if echo "$*" | grep -q -- "--build"; then
  if [ -n "${RUN_BUILD:-}" ]; then
    mkdir -p "$RUN_BUILD"
    # Create the runner binary
    printf '#!/bin/sh\nexit 0\n' > "$RUN_BUILD/spyre_coverage_runner"
    chmod +x "$RUN_BUILD/spyre_coverage_runner"
  fi
fi
exit 0
CMEOF
  chmod +x "$d/cmake"
}

# make_common_stubs: write git, pip3, strings, nproc, du stubs.
make_common_stubs() {
  local d=$1
  # git: clone just creates a CMakeLists.txt in the target, submodule update is noop
  cat > "$d/git" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "clone" ]; then
  # last positional arg is the target dir
  tgt="${@: -1}"
  mkdir -p "$tgt"
  touch "$tgt/CMakeLists.txt"
  exit 0
fi
exit 0
EOF
  # pip3: noop
  printf '#!/bin/sh\nexit 0\n' > "$d/pip3"
  # strings: print a known string that includes aten::sin.out
  printf '#!/bin/sh\necho "aten::sin.out"\n' > "$d/strings"
  # nproc: return 4
  printf '#!/bin/sh\necho 4\n' > "$d/nproc"
  # ldd: every binary "resolves" the same libs, so spyre_env.sh --check passes
  printf '#!/bin/sh\nprintf "\\tlibflex.so => /stack/lib/libflex.so (0x0)\\n\\tlibspyre_comms.so.1 => /stack/lib/libspyre_comms.so.1 (0x0)\\n\\tlibspyrecode-host-functions.so => /stack/lib/libspyrecode-host-functions.so (0x0)\\n"\n' > "$d/ldd"
  # du: return something harmless
  printf '#!/bin/sh\necho "100M\t$1"\n' > "$d/du"
  # find: delegate to real find (already on PATH after our stubs)
  # (no stub needed — we keep real find)
  chmod +x "$d/git" "$d/pip3" "$d/strings" "$d/nproc" "$d/du" "$d/ldd"
}

# run_sut: run the SUT with given env overrides, stubs prepended to PATH.
# Usage: run_sut <stub_bin_dir> [env_var=val ...]
run_sut() {
  local stub_bin=$1; shift
  env PATH="$stub_bin:$PATH" "${SPYRE_VARS[@]}" \
      HOME_DIR="$TMPROOT/home" \
      XDG_CACHE_HOME="$TMPROOT/cache" \
      "$@" \
      bash "$SUT"
}

# ---------------------------------------------------------------------------
# T1 — Preflight exits 2 when python is absent
# ---------------------------------------------------------------------------
echo; echo "--- T1: python absent -> exit 2 ---"
{
  D=$(make_stub_dir t1)
  make_common_stubs "$D"
  # cmake stub present, python absent (VENV points nowhere)
  printf '#!/bin/sh\nexit 0\n' > "$D/cmake"; chmod +x "$D/cmake"
  T1OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV=/nonexistent \
            bash "$SUT" 2>&1) && T1RC=0 || T1RC=$?
  if [ "$T1RC" = "2" ] && echo "$T1OUT" | grep -q "FATAL: no python"; then
    pass "T1: python absent -> exit 2"
  else
    fail "T1: python absent -> exit 2" "got exit $T1RC; output: $T1OUT"
  fi
}

# ---------------------------------------------------------------------------
# T2 — Preflight exits 2 when cmake is absent
# ---------------------------------------------------------------------------
echo; echo "--- T2: cmake absent -> exit 2 ---"
{
  D=$(make_stub_dir t2)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  # deliberately NO cmake stub
  T2OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            bash "$SUT" 2>&1) && T2RC=0 || T2RC=$?
  if [ "$T2RC" = "2" ] && echo "$T2OUT" | grep -q "FATAL: no cmake"; then
    pass "T2: cmake absent -> exit 2"
  else
    fail "T2: cmake absent -> exit 2" "got exit $T2RC; output: $T2OUT"
  fi
}

# ---------------------------------------------------------------------------
# T3 — Wrong torch version fails preflight
# ---------------------------------------------------------------------------
echo; echo "--- T3: wrong torch version -> non-zero exit ---"
{
  D=$(make_stub_dir t3)
  make_common_stubs "$D"
  make_python_stub "$D" "2.11.0"   # wrong version
  printf '#!/bin/sh\nexit 0\n' > "$D/cmake"; chmod +x "$D/cmake"
  T3OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            bash "$SUT" 2>&1) && T3RC=0 || T3RC=$?
  if [ "$T3RC" != "0" ]; then
    pass "T3: wrong torch version -> non-zero exit"
  else
    fail "T3: wrong torch version -> non-zero exit" "script exited 0 with torch 2.11.0"
  fi
}

# ---------------------------------------------------------------------------
# T4 — Existing ET install skips stages 3+4
# ---------------------------------------------------------------------------
echo; echo "--- T4: existing ET install -> skip stages 3/4 ---"
{
  D=$(make_stub_dir t4)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  make_runner_cmake_stub "$D"
  ET_INSTALL="$TMPROOT/t4-et-install"
  RUN_BUILD_T4="$TMPROOT/t4-run-build"
  ET_SRC_T4="$TMPROOT/t4-et-src"
  # Pre-populate sentinel files so SKIP_ET triggers
  mkdir -p "$ET_INSTALL/lib/cmake/ExecuTorch" "$ET_INSTALL/lib"
  touch "$ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake"
  for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
             libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
    touch "$ET_INSTALL/lib/$lib"
  done
  mkdir -p "$ET_SRC_T4"; touch "$ET_SRC_T4/CMakeLists.txt"
  T4OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            ET_INSTALL="$ET_INSTALL" \
            ET_SRC="$ET_SRC_T4" \
            RUN_BUILD="$RUN_BUILD_T4" \
            bash "$SUT" 2>&1) && T4RC=0 || T4RC=$?
  if echo "$T4OUT" | grep -q "skipping stages \[3\]\[4\]"; then
    pass "T4: existing ET install -> skip stages 3/4"
  else
    fail "T4: existing ET install -> skip stages 3/4" "skip message not found; output: $T4OUT"
  fi
}

# ---------------------------------------------------------------------------
# T5 — Missing lib causes exit 3
# ---------------------------------------------------------------------------
echo; echo "--- T5: missing lib -> exit 3 ---"
{
  D=$(make_stub_dir t5)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  # cmake stub that installs all libs EXCEPT libportable_ops_lib.a on --target install.
  # We cannot use make_runner_cmake_stub here because it writes the complete set.
  _T5_INSTALL="$TMPROOT/t5-et-install"
  cat > "$D/cmake" <<CMEOF
#!/usr/bin/env bash
if echo "\$*" | grep -q -- "--target install"; then
  mkdir -p "$_T5_INSTALL/lib/cmake/ExecuTorch" "$_T5_INSTALL/lib"
  touch "$_T5_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake"
  for lib in libexecutorch_core.a libportable_kernels.a \\
             libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
    touch "$_T5_INSTALL/lib/\$lib"
  done
  # libportable_ops_lib.a deliberately NOT created
fi
exit 0
CMEOF
  chmod +x "$D/cmake"
  ET_SRC_T5="$TMPROOT/t5-et-src"
  mkdir -p "$ET_SRC_T5"; touch "$ET_SRC_T5/CMakeLists.txt"
  T5OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            ET_INSTALL="$_T5_INSTALL" \
            ET_SRC="$ET_SRC_T5" \
            RUN_BUILD="$TMPROOT/t5-run-build" \
            bash "$SUT" 2>&1) && T5RC=0 || T5RC=$?
  if [ "$T5RC" = "3" ] && echo "$T5OUT" | grep -q "MISSING libportable_ops_lib.a"; then
    pass "T5: missing lib -> exit 3"
  else
    fail "T5: missing lib -> exit 3" "got exit $T5RC; output: $T5OUT"
  fi
}

# ---------------------------------------------------------------------------
# T6 — Missing executorch-config.cmake causes exit 3
# ---------------------------------------------------------------------------
echo; echo "--- T6: missing executorch-config.cmake -> exit 3 ---"
{
  D=$(make_stub_dir t6)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  # cmake stub that installs libs but NOT the config file
  cat > "$D/cmake" <<'CMEOF'
#!/usr/bin/env bash
if echo "$*" | grep -q -- "--target install"; then
  if [ -n "${ET_INSTALL:-}" ]; then
    mkdir -p "$ET_INSTALL/lib"
    for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
               libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
      touch "$ET_INSTALL/lib/$lib"
    done
    # executorch-config.cmake deliberately NOT written
  fi
fi
exit 0
CMEOF
  chmod +x "$D/cmake"
  ET_INSTALL="$TMPROOT/t6-et-install"
  ET_SRC_T6="$TMPROOT/t6-et-src"
  mkdir -p "$ET_SRC_T6"; touch "$ET_SRC_T6/CMakeLists.txt"
  T6OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            ET_INSTALL="$ET_INSTALL" \
            ET_SRC="$ET_SRC_T6" \
            RUN_BUILD="$TMPROOT/t6-run-build" \
            bash "$SUT" 2>&1) && T6RC=0 || T6RC=$?
  if [ "$T6RC" = "3" ] && echo "$T6OUT" | grep -q "MISSING executorch-config.cmake"; then
    pass "T6: missing executorch-config.cmake -> exit 3"
  else
    fail "T6: missing executorch-config.cmake -> exit 3" "got exit $T6RC; output: $T6OUT"
  fi
}

# ---------------------------------------------------------------------------
# T7 — cmake succeeds but runner binary not produced -> exit 4
# ---------------------------------------------------------------------------
echo; echo "--- T7: runner not built -> exit 4 ---"
{
  D=$(make_stub_dir t7)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  ET_INSTALL="$TMPROOT/t7-et-install"
  ET_SRC_T7="$TMPROOT/t7-et-src"
  # cmake installs ET but does NOT create the runner binary
  cat > "$D/cmake" <<'CMEOF'
#!/usr/bin/env bash
if echo "$*" | grep -q -- "--target install"; then
  if [ -n "${ET_INSTALL:-}" ]; then
    mkdir -p "$ET_INSTALL/lib/cmake/ExecuTorch" "$ET_INSTALL/lib"
    touch "$ET_INSTALL/lib/cmake/ExecuTorch/executorch-config.cmake"
    for lib in libexecutorch_core.a libportable_ops_lib.a libportable_kernels.a \
               libextension_module_static.a libextension_tensor.a libextension_data_loader.a; do
      touch "$ET_INSTALL/lib/$lib"
    done
  fi
fi
# --build for runner: succeed but write NO binary
exit 0
CMEOF
  chmod +x "$D/cmake"
  mkdir -p "$ET_SRC_T7"; touch "$ET_SRC_T7/CMakeLists.txt"
  T7OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            ET_INSTALL="$ET_INSTALL" \
            ET_SRC="$ET_SRC_T7" \
            RUN_BUILD="$TMPROOT/t7-run-build" \
            bash "$SUT" 2>&1) && T7RC=0 || T7RC=$?
  if [ "$T7RC" = "4" ] && echo "$T7OUT" | grep -q "FATAL: runner not built"; then
    pass "T7: runner not built -> exit 4"
  else
    fail "T7: runner not built -> exit 4" "got exit $T7RC; output: $T7OUT"
  fi
}

# ---------------------------------------------------------------------------
# T8 — Happy path: DEVICE RUNTIME READY + RC=0
# ---------------------------------------------------------------------------
echo; echo "--- T8: happy path -> DEVICE RUNTIME READY + RC=0 ---"
{
  D=$(make_stub_dir t8)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  make_runner_cmake_stub "$D"
  ET_INSTALL="$TMPROOT/t8-et-install"
  ET_SRC_T8="$TMPROOT/t8-et-src"
  RUN_BUILD_T8="$TMPROOT/t8-run-build"
  mkdir -p "$ET_SRC_T8"; touch "$ET_SRC_T8/CMakeLists.txt"
  T8OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            ET_INSTALL="$ET_INSTALL" \
            ET_SRC="$ET_SRC_T8" \
            RUN_BUILD="$RUN_BUILD_T8" \
            bash "$SUT" 2>&1) && T8RC=0 || T8RC=$?
  if [ "$T8RC" = "0" ] \
     && echo "$T8OUT" | grep -q "DEVICE RUNTIME READY" \
     && echo "$T8OUT" | grep -q "RC=0"; then
    pass "T8: happy path -> DEVICE RUNTIME READY + RC=0"
  else
    fail "T8: happy path -> DEVICE RUNTIME READY + RC=0" "got exit $T8RC; output: $T8OUT"
  fi
}

# ---------------------------------------------------------------------------
# T9 — PATH overrides are respected (no access outside temp dirs)
# ---------------------------------------------------------------------------
echo; echo "--- T9: env overrides respected ---"
{
  D=$(make_stub_dir t9)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"
  make_runner_cmake_stub "$D"
  REPO_T9="$TMPROOT/t9-repo"
  ET_INSTALL_T9="$TMPROOT/t9-et-install"
  ET_SRC_T9="$TMPROOT/t9-et-src"
  RUN_BUILD_T9="$TMPROOT/t9-run-build"
  mkdir -p "$REPO_T9/executorch/backends/spyre/spyre_runner"
  mkdir -p "$ET_SRC_T9"; touch "$ET_SRC_T9/CMakeLists.txt"
  T9OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" "${SPYRE_VARS[@]}" \
            VENV="$D/.." \
            REPO="$REPO_T9" \
            ET_SRC="$ET_SRC_T9" \
            ET_INSTALL="$ET_INSTALL_T9" \
            RUN_BUILD="$RUN_BUILD_T9" \
            bash "$SUT" 2>&1) && T9RC=0 || T9RC=$?
  # Verify overrides were honoured: runner binary must land in RUN_BUILD_T9,
  # not the default /tmp/work/build-spyre.
  if [ "$T9RC" = "0" ] && [ -x "$RUN_BUILD_T9/spyre_coverage_runner" ]; then
    pass "T9: env overrides respected"
  else
    fail "T9: env overrides respected" "got exit $T9RC; runner not found at $RUN_BUILD_T9/spyre_coverage_runner; output: $T9OUT"
  fi
}

# ---------------------------------------------------------------------------
# T10 — Stack derivation fails -> clear failure (exit 5), no cmake configure
# ---------------------------------------------------------------------------
echo; echo "--- T10: derivation of the Spyre stack fails -> exit 5 + message ---"
{
  D=$(make_stub_dir t10)
  make_common_stubs "$D"
  make_python_stub "$D" "2.13.0"      # stub cannot import torch_spyre -> derives nothing
  make_cmake_stub_ok "$D"            # never writes a runner binary
  ET_INSTALL="$TMPROOT/t10-et-install"
  ET_SRC_T10="$TMPROOT/t10-et-src"
  RUN_BUILD_T10="$TMPROOT/t10-run-build"
  mkdir -p "$ET_SRC_T10"; touch "$ET_SRC_T10/CMakeLists.txt"
  # No SPYRE_VARS: the real derivation runs against the stub python.
  T10OUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" \
             VENV="$D/.." ET_INSTALL="$ET_INSTALL" ET_SRC="$ET_SRC_T10" \
             RUN_BUILD="$RUN_BUILD_T10" \
             bash "$SUT" 2>&1) && T10RC=0 || T10RC=$?
  if [ "$T10RC" = "5" ] \
     && echo "$T10OUT" | grep -q "FATAL: could not derive the Spyre stack" \
     && echo "$T10OUT" | grep -q "spyre_env.sh: could not derive" \
     && [ ! -e "$RUN_BUILD_T10/spyre_coverage_runner" ]; then
    pass "T10: derivation failure -> exit 5, clear message, nothing built"
  else
    fail "T10: derivation failure" "got exit $T10RC; output: $T10OUT"
  fi

  # A partial set of variables (stale shell) is refused as well.
  T10BOUT=$(env PATH="$D:$PATH" HOME_DIR="$TMPROOT/home" XDG_CACHE_HOME="$TMPROOT/cache" \
              SPYRE_RUNTIME_DIR=/stale VENV="$D/.." ET_INSTALL="$ET_INSTALL" \
              ET_SRC="$ET_SRC_T10" RUN_BUILD="$RUN_BUILD_T10" \
              bash "$SUT" 2>&1) && T10BRC=0 || T10BRC=$?
  if [ "$T10BRC" = "5" ] && echo "$T10BOUT" | grep -q "only some variables are set"; then
    pass "T10b: partial SPYRE_* set -> exit 5, names the missing ones"
  else
    fail "T10b: partial SPYRE_* set" "got exit $T10BRC; output: $T10BOUT"
  fi

  # VENV neither set nor derivable (no python on PATH) -> clear exit 2.
  T10COUT=$(env -i PATH="/nonexistent" HOME_DIR="$TMPROOT/home" \
              XDG_CACHE_HOME="$TMPROOT/cache" /bin/bash "$SUT" 2>&1) && T10CRC=0 || T10CRC=$?
  if [ "$T10CRC" = "2" ] && echo "$T10COUT" | grep -q "FATAL: VENV is not set"; then
    pass "T10c: no VENV and no python -> exit 2, no /home/senuser fallback"
  else
    fail "T10c: no VENV" "got exit $T10CRC; output: $T10COUT"
  fi
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "Results: $PASS passed, $FAIL failed"
if [ "${#_FAILURES[@]}" -gt 0 ]; then
  echo
  echo "Failures:"
  for f in "${_FAILURES[@]}"; do echo "  - $f"; done
  exit 1
fi
exit 0
