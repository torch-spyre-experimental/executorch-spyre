#!/usr/bin/env bash
# test_spyre_cmake_stack.sh — tests for the "Spyre stack" block of
# executorch/backends/spyre/CMakeLists.txt: the source labels ([env], [cache], [-D]) and
# the stale-CMakeCache.txt error.
#
# Runs the REAL cmake against a fake torch_spyre package, a fake Spyre stack and a stub
# `ldd`, in a temp dir. No device, torch, torch-spyre or ExecuTorch checkout is needed:
# the configure is only asserted up to the stack block (it fails or stops later for
# lack of Torch / ExecuTorch, which is expected and not looked at).
#
# USAGE
#   bash scripts/tests/test_spyre_cmake_stack.sh
set -euo pipefail

PASS=0; FAIL=0; _FAILURES=()
pass() { PASS=$((PASS+1)); echo "  PASS  $1"; }
fail() { FAIL=$((FAIL+1)); _FAILURES+=("$1: $2"); echo "  FAIL  $1: $2"; }

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
SRC=$(cd "$SCRIPT_DIR/../../executorch/backends/spyre"; pwd)
command -v cmake >/dev/null || { echo "cmake not found"; exit 2; }
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# --- fixture ---------------------------------------------------------------
FX="$TMPROOT/fx"
PKG="$FX/site/torch_spyre"
mkdir -p "$PKG/csrc" "$FX/bin" "$FX/et/tools/cmake" "$FX/json/include/nlohmann"
: > "$PKG/_C.so"
: > "$PKG/csrc/prepare_kernel.h"
: > "$FX/json/include/nlohmann/json.hpp"
echo 'macro(executorch_target_link_options_shared_lib)
endmacro()' > "$FX/et/tools/cmake/Utils.cmake"
# Two complete stacks: A (what the environment names) and B (what an old cache names).
for s in A B; do
  for d in runtime deeptools senlib comms; do mkdir -p "$FX/stack$s/$d/lib"; done
  : > "$FX/stack$s/runtime/lib/libflex.so"
  : > "$FX/stack$s/comms/lib/libspyre_comms.so.1"
done
# ldd stub: _C.so resolves stack A (LDD_STACK=B switches it).
cat > "$FX/bin/ldd" <<LDD
#!/usr/bin/env bash
s=\${LDD_STACK:-A}
echo "	libflex.so => $FX/stack\$s/runtime/lib/libflex.so (0x1)"
echo "	libspyre_comms.so.1 => $FX/stack\$s/comms/lib/libspyre_comms.so.1 (0x1)"
LDD
chmod +x "$FX/bin/ldd"

# configure STACK BUILD_DIR [extra cmake args...]: env names STACK (A|B); output in $OUT.
OUT="$TMPROOT/out"
configure() {
  local s=$1 b=$2; shift 2
  env -i HOME="$HOME" PATH="$FX/bin:$PATH" LDD_STACK="${LDD_STACK:-A}" \
      TORCH_SPYRE_C_SO="$PKG/_C.so" TORCH_SPYRE_CSRC="$PKG/csrc" \
      SPYRE_RUNTIME_DIR="$FX/stack$s/runtime" SPYRE_DEEPTOOLS_DIR="$FX/stack$s/deeptools" \
      SPYRE_SENLIB_DIR="$FX/stack$s/senlib" SPYRE_COMMS_DIR="$FX/stack$s/comms" \
      cmake -S "$SRC" -B "$b" -DEXECUTORCH_ROOT="$FX/et" -DNLOHMANN_JSON_DIR="$FX/json" "$@" \
      >"$OUT" 2>&1 || true
}
labels() { grep -E 'Spyre stack: (TORCH_SPYRE|SPYRE_)[A-Z_]+ = ' "$OUT" | sed -E 's/.*\[(.*)\]$/\1/' | sort | uniq -c | tr -s ' ' | tr '\n' ';'; }
STALE_MSG="stale value from an existing CMakeCache.txt; reconfigure with"

echo "--- C1: empty build dir -> six values labelled [env]"
B1="$TMPROOT/b1"; configure A "$B1"
[ "$(labels)" = " 6 env;" ] && pass "C1: six [env] lines" || fail "C1" "labels: $(labels)"

echo "--- C2: same environment, existing cache -> [cache], no error"
configure A "$B1"
[ "$(labels)" = " 6 cache;" ] && ! grep -q "stale value" "$OUT" \
  && pass "C2: six [cache] lines, no stale error" || fail "C2" "labels: $(labels)"

echo "--- C3: environment changed, existing cache -> stale error, before the ldd check"
configure B "$B1"
if grep -q "$STALE_MSG" "$OUT" && grep -q '`cmake --fresh` or an empty build dir' "$OUT" \
   && grep -q "SPYRE_RUNTIME_DIR" "$OUT" && grep -q "$FX/stackA/runtime" "$OUT" \
   && grep -q "$FX/stackB/runtime" "$OUT" && ! grep -q "Inconsistent Spyre stack" "$OUT" \
   && ! grep -q '\[-D\]' "$OUT"; then
  pass "C3: stale message names the cached and environment values; no [-D], no ldd error"
else
  fail "C3" "$(sed -n '1,40p' "$OUT")"
fi

echo "--- C4: a cache written by an older checkout (values, no source markers) -> [cache]/stale, never [-D]"
# This is the reported failure: the environment had the right values, an old
# CMakeCache.txt held others, and the old code printed the cached ones as "[-D]".
B4="$TMPROOT/b4"; mkdir -p "$B4"
{
  echo "TORCH_SPYRE_C_SO:STRING=$PKG/_C.so"
  echo "TORCH_SPYRE_CSRC:STRING=$PKG/csrc"
  echo "SPYRE_RUNTIME_DIR:STRING=$FX/stackB/runtime"
  echo "SPYRE_DEEPTOOLS_DIR:STRING=$FX/stackB/deeptools"
  echo "SPYRE_SENLIB_DIR:STRING=$FX/stackB/senlib"
  echo "SPYRE_COMMS_DIR:STRING=$FX/stackB/comms"
} > "$B4/CMakeCache.txt"
configure A "$B4"
if grep -q "$STALE_MSG" "$OUT" && ! grep -q '\[-D\]' "$OUT"; then
  pass "C4: stale error, no [-D] label"
else
  fail "C4" "$(sed -n '1,20p' "$OUT")"
fi

echo "--- C5: empty build dir with the new environment -> works ([env])"
B5="$TMPROOT/b5"; configure B "$B5" ; LDD_STACK=B configure B "$B5"
rm -rf "$B5"; LDD_STACK=B configure B "$B5"
[ "$(labels)" = " 6 env;" ] && ! grep -q "stale value" "$OUT" \
  && pass "C5: new stack in an empty dir configures" || fail "C5" "labels: $(labels)"

echo "--- C6: stale cache and a stack that ldd does not resolve -> stale error comes first"
configure A "$B1"            # cache = A
LDD_STACK=A configure B "$B1"  # env = B, ldd says A: both would fail
grep -q "$STALE_MSG" "$OUT" && ! grep -q "Inconsistent Spyre stack" "$OUT" \
  && pass "C6: stale error reported before the consistency check" || fail "C6" "$(sed -n '1,30p' "$OUT")"

echo "--- C7: explicit -D on the command line is labelled [-D]"
B7="$TMPROOT/b7"
configure A "$B7" -DSPYRE_COMMS_DIR="$FX/stackA/comms"
grep -q "SPYRE_COMMS_DIR = $FX/stackA/comms   \[-D\]" "$OUT" \
  && pass "C7: -D labelled [-D]" || fail "C7" "$(grep 'SPYRE_COMMS_DIR' "$OUT")"

echo "--- C8: cmake --fresh clears a stale cache (CMake >= 3.24 only)"
ver=$(cmake --version | sed -n '1s/.* //p')
if [ "$(printf '%s\n3.24\n' "$ver" | sort -V | head -1)" = "3.24" ]; then
  configure A "$B1"; configure B "$B1"   # stale again
  grep -q "$STALE_MSG" "$OUT" || fail "C8" "no stale error to clear"
  configure B "$B1" --fresh
  [ "$(labels)" = " 6 env;" ] && ! grep -q "stale value" "$OUT" \
    && pass "C8: --fresh reconfigures (cmake $ver)" || fail "C8" "labels: $(labels)"
else
  echo "  SKIP  C8: cmake $ver has no --fresh (needs 3.24)"
fi

echo
echo "Results: $PASS passed, $FAIL failed"
if [ "${#_FAILURES[@]}" -gt 0 ]; then printf '  - %s\n' "${_FAILURES[@]}"; exit 1; fi
