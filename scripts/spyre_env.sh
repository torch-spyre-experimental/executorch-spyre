#!/usr/bin/env bash
# Copyright 2026 The ExecuTorch-Spyre Authors.
# Licensed under the Apache License, Version 2.0.
#
# spyre_env.sh -- derive the Spyre device-build variables from the torch_spyre/_C.so
# that Python imports. Single source of truth for docs/setup.md and
# scripts/build_device_runtime.sh.
#
# WHY
#   spyre_llm_runner / spyre_coverage_runner run in the same process as, and call
#   into, torch_spyre/_C.so. They must therefore link the same libflex, DeepTools
#   and libspyre_comms that _C.so loads. Picking those paths independently (from a
#   default, from docs, from a different SDK tree) produces undefined references at
#   link time or an ImportError at run time. This script reads the answer from the
#   dynamic loader instead of guessing.
#
# USAGE (bash only)
#   source scripts/spyre_env.sh [--exports]
#       SOURCED: exports the six variables below into the calling shell. Nothing is
#       printed on stdout; diagnostics go to stderr. On failure it returns non-zero and
#       leaves your shell options and variables untouched. Use this in a terminal.
#
#   scripts/spyre_env.sh [--exports | --cmake-args | --print | --check BINARY]
#       EXECUTED: a child process cannot export into your shell, so it prints.
#
#     --exports     (default) `export NAME='value'` lines, for eval:
#                       eval "$(scripts/spyre_env.sh)"
#     --cmake-args  one `-DNAME=value` per line:
#                       mapfile -t SPYRE_CMAKE_ARGS < <(scripts/spyre_env.sh --cmake-args)
#                       cmake ... "${SPYRE_CMAKE_ARGS[@]}"
#     --print       a human-readable table.
#     --check BIN   after the build: compare what `ldd BIN` and `ldd $TORCH_SPYRE_C_SO`
#                   resolve for libflex / libspyre_comms / DeepTools. Prints both and
#                   exits 1 on any difference. Resolution uses the current
#                   LD_LIBRARY_PATH, i.e. what a run from this shell would load.
#
#   --cmake-args, --print and --check behave the same when sourced (they print / check;
#   they export nothing).
#
#   Diagnostics go to stderr, so stdout is always safe to eval / capture.
#
# VARIABLES
#   TORCH_SPYRE_C_SO     the _C*.so that `import torch_spyre._C` loads
#   TORCH_SPYRE_CSRC     <torch_spyre package>/csrc
#   SPYRE_RUNTIME_DIR    <root> of the libflex.so that ldd resolves for _C.so
#   SPYRE_DEEPTOOLS_DIR  <root> of libspyrecode-host-functions.so (NOT libutil.so:
#                        that name exists in several DeepTools trees)
#   SPYRE_SENLIB_DIR     <root> of libsenlib.so
#   SPYRE_COMMS_DIR      <root> of libspyre_comms.so.1
#   ("<root>" is the parent of the directory that holds the library, which must be
#   named `lib`.)
#
# INPUTS
#   PYTHON               interpreter to use (default: `python` on PATH). Its
#                        PYTHONPATH must make `torch_spyre` importable.
#   LD_LIBRARY_PATH      decides which libflex / DeepTools / comms `ldd` reports,
#                        exactly as it will at run time. It must list the Spyre stack
#                        that torch-spyre was built against FIRST. This script fails if
#                        `import torch_spyre._C` does not work under it.
#
# ESCAPE HATCH
#   If all six variables are already set in the environment, nothing is derived and
#   they are emitted as given. If only some are set the script fails: a partial set is
#   almost always stale values left in a shell.
# END-OF-HEADER

# All work happens in _spyre_env_run, which is always run in a subshell: `set -e`, `exit`
# and its variables can never leak into a shell that sources this file.
_spyre_env_run() {
set -euo pipefail

MODE=${1:---exports}
TARGET=${2:-}
VARS=(TORCH_SPYRE_C_SO TORCH_SPYRE_CSRC SPYRE_RUNTIME_DIR SPYRE_DEEPTOOLS_DIR
      SPYRE_SENLIB_DIR SPYRE_COMMS_DIR)

die() { echo "spyre_env.sh: $*" >&2; exit 1; }
case "$MODE" in
  --exports|--cmake-args|--print) ;;
  --check) [ -n "$TARGET" ] || die "--check needs the binary to inspect" ;;
  -h|--help) sed -n '2,/^# END-OF-HEADER/p' "$_SPYRE_ENV_SELF" | sed '$d;s/^# \{0,1\}//' ; exit 0 ;;
  *) die "unknown option '$MODE' (use --exports, --cmake-args or --print)" ;;
esac

declare -A V=()
n_set=0
for k in "${VARS[@]}"; do
  if [ -n "${!k:-}" ]; then V[$k]=${!k}; n_set=$((n_set+1)); fi
done

if [ "$n_set" -eq "${#VARS[@]}" ]; then
  echo "spyre_env.sh: all of ${VARS[*]} are set in the environment; not deriving." >&2
elif [ "$n_set" -ne 0 ]; then
  unset_list=()
  for k in "${VARS[@]}"; do [ -n "${V[$k]:-}" ] || unset_list+=("$k"); done
  die "only some variables are set in the environment (missing: ${unset_list[*]}). Set all six or none: a partial set is usually stale values from an earlier shell. Unset them, or run 'unset ${VARS[*]}'."
else
  PYTHON=${PYTHON:-python}
  command -v "$PYTHON" >/dev/null 2>&1 || die "no interpreter '$PYTHON' on PATH. Activate the torch-spyre venv or set PYTHON=/path/to/python."

  derived=$("$PYTHON" - <<'PYEOF'
import glob, importlib.util, os, subprocess, sys

def fail(msg):
    sys.exit(msg)

spec = importlib.util.find_spec("torch_spyre")          # does not import the package
if spec is None or not spec.submodule_search_locations:
    fail("torch_spyre is not importable with this interpreter (%s). "
         "Check PYTHONPATH / the active venv." % sys.executable)
pkg = list(spec.submodule_search_locations)[0]
sos = sorted(glob.glob(os.path.join(pkg, "_C*.so")))
if len(sos) != 1:
    fail("expected exactly one _C*.so in %s, found %d: %s" % (pkg, len(sos), sos))
so = sos[0]
csrc = os.path.join(pkg, "csrc")
if not os.path.isfile(os.path.join(csrc, "prepare_kernel.h")):
    fail("%s/prepare_kernel.h not found: this torch_spyre has no csrc/ next to it "
         "(installed wheel?). The runner needs the sources it was built from." % csrc)

ldd = subprocess.run(["ldd", so], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     universal_newlines=True).stdout
res = {}
for line in ldd.splitlines():
    if "=>" not in line:
        continue
    name, rest = [s.strip() for s in line.split("=>", 1)]
    res[name] = None if rest.startswith("not found") else rest.split()[0]

WANT = [("libflex.so", "SPYRE_RUNTIME_DIR"),
        ("libspyrecode-host-functions.so", "SPYRE_DEEPTOOLS_DIR"),
        ("libsenlib.so", "SPYRE_SENLIB_DIR"),
        ("libspyre_comms.so.1", "SPYRE_COMMS_DIR")]
out = {"TORCH_SPYRE_C_SO": so, "TORCH_SPYRE_CSRC": csrc}
for lib, var in WANT:
    p = res.get(lib)
    if not p:
        fail("ldd %s does not resolve %s (%s). LD_LIBRARY_PATH is missing the Spyre "
             "stack that this torch-spyre was built against." %
             (so, lib, "not found" if lib in res else "not listed"))
    libdir = os.path.dirname(p)
    if os.path.basename(libdir) != "lib":
        fail("%s resolves to %s, which is not under a 'lib' directory; cannot derive "
             "the install root for %s." % (lib, p, var))
    out[var] = os.path.dirname(libdir)
for k, v in out.items():
    print("%s=%s" % (k, v))
PYEOF
  ) || die "could not derive the Spyre stack (see above)."

  while IFS='=' read -r k v; do
    [ -n "$k" ] && V[$k]=$v
  done <<<"$derived"
  for k in "${VARS[@]}"; do
    [ -n "${V[$k]:-}" ] || die "could not derive $k: '$PYTHON' printed no value for it (is torch_spyre importable with it?)."
  done

  # ldd says where the loader *would* look. Prove that the interpreter really loads
  # this _C.so with the current LD_LIBRARY_PATH (a wrong stack fails here with
  # 'undefined symbol: _ZN4flex...', which is the failure this script exists to catch).
  if ! imp=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -c \
        'import torch, torch_spyre._C as c; print(c.__file__)' 2>&1); then
    echo "$imp" | tail -n 5 >&2
    die "'import torch_spyre._C' fails with the current LD_LIBRARY_PATH. Usually a different Spyre stack (e.g. /opt/ibm/spyre) comes first on LD_LIBRARY_PATH than the one torch-spyre was built against. Put the right stack's lib dirs first and retry."
  fi
  imp=$(tail -n 1 <<<"$imp")
  [ "$(readlink -f "$imp")" = "$(readlink -f "${V[TORCH_SPYRE_C_SO]}")" ] \
    || die "Python imports $imp but the derivation used ${V[TORCH_SPYRE_C_SO]}."
  echo "spyre_env.sh: derived from ${V[TORCH_SPYRE_C_SO]} (import OK)" >&2
fi

if [ "$MODE" = "--check" ]; then
  [ -e "$TARGET" ] || die "--check: no such file: $TARGET"
  bad=0
  ldd "$TARGET" >/dev/null 2>&1 || die "--check: 'ldd $TARGET' failed (not a dynamic executable?)."
  for lib in libflex.so libspyre_comms.so.1 libspyrecode-host-functions.so; do
    a=$(ldd "${V[TORCH_SPYRE_C_SO]}" 2>&1 | awk -v l="$lib" '$1==l {print $3}' || true)
    b=$(ldd "$TARGET" 2>&1 | awk -v l="$lib" '$1==l {print $3}' || true)
    ra=$(readlink -f "${a:-/nonexistent}" 2>/dev/null || true)
    rb=$(readlink -f "${b:-/nonexistent}" 2>/dev/null || true)
    if [ -n "$a" ] && [ "$ra" = "$rb" ]; then st=OK; else st=MISMATCH; bad=1; fi
    printf '%-9s %s\n    _C.so : %s\n    runner: %s\n' "$st" "$lib" "${a:-<not resolved>}" "${b:-<not resolved>}"
  done
  [ "$bad" -eq 0 ] || die "$TARGET and ${V[TORCH_SPYRE_C_SO]} do not resolve the same libraries. LD_LIBRARY_PATH and the -D values used at configure time disagree; rebuild in an EMPTY build dir after fixing LD_LIBRARY_PATH."
  echo "spyre_env.sh: $TARGET and _C.so resolve the same libflex / libspyre_comms / DeepTools" >&2
  exit 0
fi

case "$MODE" in
  --exports)    for k in "${VARS[@]}"; do printf 'export %s=%q\n' "$k" "${V[$k]}"; done ;;
  --cmake-args) for k in "${VARS[@]}"; do printf -- '-D%s=%s\n' "$k" "${V[$k]}"; done ;;
  --print)      for k in "${VARS[@]}"; do printf '%-20s = %s\n' "$k" "${V[$k]}"; done ;;
esac
}

_SPYRE_ENV_SELF=${BASH_SOURCE[0]}
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  # Sourced: derive in a subshell (stdout captured, stderr passes through), then export
  # into this shell. Only the six variables are left behind.
  if [[ "${1:---exports}" == "--exports" ]]; then
    _spyre_env_out=$( _spyre_env_run --exports ) && _spyre_env_rc=0 || _spyre_env_rc=$?
    if [[ $_spyre_env_rc -eq 0 ]]; then eval "$_spyre_env_out"; fi
  else
    ( _spyre_env_run "$@" ) && _spyre_env_rc=0 || _spyre_env_rc=$?
  fi
  # $_spyre_env_rc is expanded into the string before the cleanup runs.
  eval "unset -f _spyre_env_run; unset _SPYRE_ENV_SELF _spyre_env_out _spyre_env_rc; return $_spyre_env_rc"
else
  # Executed: a child cannot export into the parent, so print (eval-able on stdout).
  _spyre_env_run "$@"
fi
