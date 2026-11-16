#!/usr/bin/env bash
# test_spyre_env.sh — hermetic tests for scripts/spyre_env.sh (sourced vs executed).
#
# No device, torch or torch-spyre needed: a fake torch_spyre package, a stub `ldd` and a
# stub `python` (real python3 for the derivation, a canned answer for the import check)
# live in a temp dir.
#
# USAGE
#   bash scripts/tests/test_spyre_env.sh
set -euo pipefail

PASS=0; FAIL=0; _FAILURES=()
pass() { PASS=$((PASS+1)); echo "  PASS  $1"; }
fail() { FAIL=$((FAIL+1)); _FAILURES+=("$1: $2"); echo "  FAIL  $1: $2"; }

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
SUT="$SCRIPT_DIR/../spyre_env.sh"
REAL_PY=$(command -v python3)
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT

# --- fixture ---------------------------------------------------------------
FX="$TMPROOT/fx"
PKG="$FX/site/torch_spyre"
mkdir -p "$PKG/csrc" "$FX/bin"
: > "$PKG/_C.cpython-312-x86_64-linux-gnu.so"
: > "$PKG/csrc/prepare_kernel.h"
for d in runtime deeptools senlib comms; do mkdir -p "$FX/stack/$d/lib"; done
: > "$FX/stack/runtime/lib/libflex.so"
: > "$FX/stack/deeptools/lib/libspyrecode-host-functions.so"
: > "$FX/stack/senlib/lib/libsenlib.so"
: > "$FX/stack/comms/lib/libspyre_comms.so.1"
C_SO="$PKG/_C.cpython-312-x86_64-linux-gnu.so"

# ldd stub: every target resolves the same fake stack, unless LDD_FLEX overrides libflex.
cat > "$FX/bin/ldd" <<LDD
#!/usr/bin/env bash
echo "	libflex.so => \${LDD_FLEX:-$FX/stack/runtime/lib/libflex.so} (0x1)"
echo "	libspyrecode-host-functions.so => $FX/stack/deeptools/lib/libspyrecode-host-functions.so (0x1)"
echo "	libsenlib.so => $FX/stack/senlib/lib/libsenlib.so (0x1)"
echo "	libspyre_comms.so.1 => $FX/stack/comms/lib/libspyre_comms.so.1 (0x1)"
LDD
# python stub: "-" (derivation script on stdin) runs for real; "-c" (import check) answers.
cat > "$FX/bin/python" <<PY
#!/usr/bin/env bash
if [ "\${1:-}" = "-c" ]; then
  [ -z "\${FAKE_IMPORT_FAILS:-}" ] || { echo "ImportError: undefined symbol: _ZN4flex19createComputeParams" >&2; exit 1; }
  echo "$C_SO"; exit 0
fi
exec "$REAL_PY" "\$@"
PY
chmod +x "$FX/bin/ldd" "$FX/bin/python"
touch "$FX/runner"

VARS=(TORCH_SPYRE_C_SO TORCH_SPYRE_CSRC SPYRE_RUNTIME_DIR SPYRE_DEEPTOOLS_DIR
      SPYRE_SENLIB_DIR SPYRE_COMMS_DIR)
# Clean env: none of the six set; stubs first on PATH; fake package importable.
cleanenv() { env -u TORCH_SPYRE_C_SO -u TORCH_SPYRE_CSRC -u SPYRE_RUNTIME_DIR \
  -u SPYRE_DEEPTOOLS_DIR -u SPYRE_SENLIB_DIR -u SPYRE_COMMS_DIR \
  PATH="$FX/bin:$PATH" PYTHONPATH="$FX/site" "$@"; }

# --- E: executed ------------------------------------------------------------
echo "--- executed ---"
OUT=$(cleanenv bash "$SUT" 2>"$TMPROOT/err") && RC=0 || RC=$?
[ "$RC" = 0 ] && pass "E1: exit 0" || fail "E1" "rc=$RC: $(cat "$TMPROOT/err")"

if [ "$(grep -vc '^export [A-Z_]*=' <<<"$OUT")" = 0 ] && [ "$(grep -c '^export ' <<<"$OUT")" = 6 ]; then
  pass "E2: stdout is exactly six export lines"
else fail "E2" "stdout: $OUT"; fi

grep -q "import OK" "$TMPROOT/err" && pass "E3: diagnostics are on stderr" || fail "E3" "stderr: $(cat "$TMPROOT/err")"

# eval-ing the output into a fresh shell yields the right values
GOT=$(env -i PATH=/usr/bin:/bin bash -c "eval \"\$1\"; echo \$SPYRE_RUNTIME_DIR \$SPYRE_COMMS_DIR \$TORCH_SPYRE_C_SO" _ "$OUT")
[ "$GOT" = "$FX/stack/runtime $FX/stack/comms $C_SO" ] && pass "E4: eval of the output sets the values" || fail "E4" "$GOT"

# executing must NOT change the caller's environment
LEAK=$(cleanenv bash -c "bash '$SUT' >/dev/null 2>&1; echo \"\${SPYRE_RUNTIME_DIR:-unset}\"")
[ "$LEAK" = unset ] && pass "E5: executing exports nothing into the caller" || fail "E5" "$LEAK"

CA=$(cleanenv bash "$SUT" --cmake-args 2>/dev/null)
grep -qx -- "-DSPYRE_RUNTIME_DIR=$FX/stack/runtime" <<<"$CA" && [ "$(wc -l <<<"$CA")" = 6 ] \
  && pass "E6: --cmake-args prints six -D lines" || fail "E6" "$CA"

cleanenv bash "$SUT" --print 2>/dev/null | grep -q "SPYRE_COMMS_DIR *= $FX/stack/comms" \
  && pass "E7: --print" || fail "E7" "no table"

cleanenv bash "$SUT" --check "$FX/runner" >/dev/null 2>"$TMPROOT/err" && pass "E8: --check OK" \
  || fail "E8" "$(cat "$TMPROOT/err")"
OUT=$(cleanenv env FAKE_IMPORT_FAILS=1 bash "$SUT" 2>"$TMPROOT/err") && RC=0 || RC=$?
if [ "$RC" != 0 ] && [ -z "$OUT" ] && grep -q "import torch_spyre._C' fails" "$TMPROOT/err"; then
  pass "E9: failing import -> non-zero, empty stdout, explanation on stderr"
else fail "E9" "rc=$RC stdout=$OUT"; fi

# --- S: sourced -------------------------------------------------------------
echo "--- sourced ---"
# Run in an interactive-like shell with `set +e` and a marker option state; print what the
# caller sees afterwards.
src_run() { # $1 = extra args to source; remaining env from cleanenv
  cleanenv bash -c '
    set +e +u; set +o pipefail
    sut=$1; extra=${2:-}; out=$3; err=$4; set --      # `source` inherits $@: start clean
    before="$-"
    source "$sut" $extra >"$out" 2>"$err"; rc=$?
    echo "rc=$rc"
    echo "opts_same=$([ "$before" = "$-" ] && echo yes || echo no)"
    echo "pipefail=$(set -o | awk "\$1==\"pipefail\"{print \$2}")"
    for v in TORCH_SPYRE_C_SO TORCH_SPYRE_CSRC SPYRE_RUNTIME_DIR SPYRE_DEEPTOOLS_DIR SPYRE_SENLIB_DIR SPYRE_COMMS_DIR; do
      if declare -p $v 2>/dev/null | grep -q "^declare -x"; then echo "exported:$v=${!v}"; else echo "notexported:$v"; fi
    done
    echo "helper_left=$(declare -F _spyre_env_run >/dev/null && echo yes || echo no)"
    echo "tmp_left=$(compgen -v | grep -c "^_spyre_env\|^_SPYRE_ENV")"
    echo "MODE_leak=${MODE:-unset} VARS_leak=${VARS:-unset}"
    echo "still_alive"
  ' _ "$SUT" "${1:-}" "$TMPROOT/so" "$TMPROOT/se"
}
R=$(src_run) || true
grep -q '^rc=0$' <<<"$R" && pass "S1: source returns 0" || fail "S1" "$R"
ok=1; for v in "${VARS[@]}"; do grep -q "^exported:$v=" <<<"$R" || ok=0; done
[ $ok = 1 ] && pass "S2: all six variables exported into the caller" || fail "S2" "$R"
grep -q "^exported:SPYRE_COMMS_DIR=$FX/stack/comms$" <<<"$R" && grep -q "^exported:TORCH_SPYRE_C_SO=$C_SO$" <<<"$R" \
  && pass "S3: values are the derived ones" || fail "S3" "$R"
[ ! -s "$TMPROOT/so" ] && pass "S4: nothing on stdout when sourced" || fail "S4" "$(cat "$TMPROOT/so")"
grep -q "import OK" "$TMPROOT/se" && pass "S5: diagnostics on stderr when sourced" || fail "S5" "$(cat "$TMPROOT/se")"
grep -q '^opts_same=yes$' <<<"$R" && grep -q '^pipefail=off$' <<<"$R" \
  && pass "S6: no set -e/-u/pipefail leaks into the caller" || fail "S6" "$R"
grep -q '^helper_left=no$' <<<"$R" && grep -q '^tmp_left=0$' <<<"$R" && grep -q '^MODE_leak=unset VARS_leak=unset$' <<<"$R" \
  && pass "S7: no helper function or temporary variable left behind" || fail "S7" "$R"

# failing import: source returns non-zero, caller survives, exports nothing
R=$(cleanenv env FAKE_IMPORT_FAILS=1 bash -c '
  sut=$1; err=$2; set --; set +e; source "$sut" 2>"$err"; rc=$?
  echo "rc=$rc"; echo "RUNTIME=${SPYRE_RUNTIME_DIR:-unset}"; echo still_alive' _ "$SUT" "$TMPROOT/se") || true
if grep -qv '^rc=0$' <<<"$(grep '^rc=' <<<"$R")" && grep -q '^RUNTIME=unset$' <<<"$R" \
   && grep -q still_alive <<<"$R" && grep -q "fails" "$TMPROOT/se"; then
  pass "S8: failing import -> source returns non-zero, shell survives, nothing exported"
else fail "S8" "$R"; fi

# a caller running with `set -e` must not be killed by a failing source in an `if`
R=$(cleanenv env FAKE_IMPORT_FAILS=1 bash -c '
  sut=$1; set --; set -e; if source "$sut" 2>/dev/null; then echo bad; else echo handled; fi; echo still_alive' _ "$SUT") || true
grep -q handled <<<"$R" && grep -q still_alive <<<"$R" && pass "S9: works under the caller's set -e" || fail "S9" "$R"

# sourced with a non-export mode: prints, exports nothing
R=$(cleanenv bash -c 'sut=$1; set --; source "$sut" --cmake-args 2>/dev/null; echo "RUNTIME=${SPYRE_RUNTIME_DIR:-unset}"' _ "$SUT")
grep -q '^-DSPYRE_RUNTIME_DIR=' <<<"$R" && grep -q '^RUNTIME=unset$' <<<"$R" \
  && pass "S10: sourced --cmake-args prints and exports nothing" || fail "S10" "$R"

# escape hatch: all six preset -> sourcing keeps them
R=$(env -i PATH="$FX/bin:/usr/bin:/bin" TORCH_SPYRE_C_SO=/x/_C.so TORCH_SPYRE_CSRC=/x/csrc \
      SPYRE_RUNTIME_DIR=/x/r SPYRE_DEEPTOOLS_DIR=/x/d SPYRE_SENLIB_DIR=/x/s SPYRE_COMMS_DIR=/x/c \
      bash -c 'sut=$1; set --; source "$sut" 2>/dev/null; echo "rc=$? $SPYRE_RUNTIME_DIR"' _ "$SUT")
[ "$R" = "rc=0 /x/r" ] && pass "S11: all six preset -> kept as given" || fail "S11" "$R"

# partial set: error
R=$(cleanenv env SPYRE_RUNTIME_DIR=/x/r bash -c 'sut=$1; err=$2; set --; source "$sut" 2>"$err"; echo "rc=$?"' _ "$SUT" "$TMPROOT/se")
[ "$R" != "rc=0" ] && grep -q "only some variables" "$TMPROOT/se" \
  && pass "S12: partial preset -> non-zero + message" || fail "S12" "$R"

# -h / unknown option, both modes
{ bash "$SUT" -h 2>&1 || true; } | grep -q "source scripts/spyre_env.sh" && pass "H1: --help documents sourcing" || fail "H1" "no usage"
{ bash "$SUT" --bogus 2>&1 || true; } | grep -q "unknown option" && pass "H2: unknown option rejected" || fail "H2" ""

# The `env | grep` the docs print after sourcing: exactly the six names, even when the
# caller also exports SPYRE_STACK (the old '^(TORCH_SPYRE|SPYRE_)' matched that too).
echo "--- documented grep ---"
DOC_GREP='^(TORCH_SPYRE_C_SO|TORCH_SPYRE_CSRC|SPYRE_RUNTIME_DIR|SPYRE_DEEPTOOLS_DIR|SPYRE_SENLIB_DIR|SPYRE_COMMS_DIR)='
REPO_ROOT=$(cd "$(dirname "$SUT")/.."; pwd)
g1_ok=1
for f in docs/setup.md docs/setup-reference.md docs/hfa-decode-reference.md; do
  git -C "$REPO_ROOT" grep -qF -e "grep -E '$DOC_GREP'" -- "$f" || g1_ok=0
done
# and the old broad pattern must not be documented anywhere
if git -C "$REPO_ROOT" grep -qF -e "grep -E '^(TORCH_SPYRE|SPYRE_)" -- docs README.md; then g1_ok=0; fi
if [ "$g1_ok" = 1 ]; then
  pass "G1: setup.md, setup-reference.md and hfa-decode-reference.md use the six-name pattern; the broad one is gone"
else
  fail "G1" "six-name pattern missing from a doc, or the old broad pattern is still documented"
fi
R=$(cleanenv env SPYRE_STACK=/x/sentient bash -c '
  set +e; sut=$1; pat=$2; set --; source "$sut" 2>/dev/null
  echo "new=$(env | grep -cE "$pat")"
  echo "new_has_stack=$(env | grep -E "$pat" | grep -c "^SPYRE_STACK=")"
  echo "old_has_stack=$(env | grep -E "^(TORCH_SPYRE|SPYRE_)" | grep -c "^SPYRE_STACK=")"' _ "$SUT" "$DOC_GREP")
grep -q '^new=6$' <<<"$R" && grep -q '^new_has_stack=0$' <<<"$R" && grep -q '^old_has_stack=1$' <<<"$R" \
  && pass "G2: documented grep prints exactly six lines and skips SPYRE_STACK (the old pattern matched it)" || fail "G2" "$R"

echo
echo "Results: $PASS passed, $FAIL failed"
if [ "${#_FAILURES[@]}" -gt 0 ]; then printf '  - %s\n' "${_FAILURES[@]}"; exit 1; fi
