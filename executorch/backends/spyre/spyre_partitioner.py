# Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Op-support (capability-based) partitioner for the Spyre delegate.

Phase 3 replaces the Phase 2 whole-graph tagging with real partial delegation:
an :class:`~torch.fx.passes.operator_support.OperatorSupportBase` names the
edge-dialect ops the Spyre/DeepTools compiler can lower, and ExecuTorch's
capability-based partitioner (via
``generate_partitions_from_list_of_nodes``) grows the largest connected
subgraphs over those supported nodes. Each proposed partition becomes one
delegate call; **unsupported nodes stay on CPU** (ExecuTorch's portable
kernels). Nothing is dropped silently — every ``call_function`` node that falls
back to CPU is logged.

In ``allowlist`` mode the op allowlist (:data:`_SUPPORTED_ATEN_OPS`) is keyed
on the *core-aten* op
name (``op.name()`` → e.g. ``"aten::addmm"``) so it is stable across the
edge-dialect ``EdgeOpOverload`` wrapper (whose ``._op`` is the underlying
``OpOverload``) and plain aten overloads alike. Since Phase 5 it is **data
driven**: it is the set of ``validated: true`` ops in
``coverage/supported_ops.yaml`` (loaded via :func:`coverage.ops_registry.
validated_op_names`), which is the checked-in source of truth. An op is marked
validated only after an on-device compile+run confirmed SDSC lowers it and the
result matched the CPU fp16 reference (Phase 2 matmul + Phase 3 MLP); candidate
ops sit at ``validated: false`` until the coverage harness promotes them. If
SDSC ever rejects a delegated op, ``preprocess`` fails loudly rather than
miscompiling.

Two partition modes share this machinery (see :func:`resolve_partition_mode`)
and the production contract is defined in ``docs/operator-guide.md``:

* ``fallbacklist`` (**production default**): delegate everything except ops
  torch-spyre declares a CPU fallback for (``fallback_ops``) and except every
  ``validated: false`` op in ``supported_ops.yaml``, which is kept on CPU by a
  graph-local rule (:data:`_UNVALIDATED_ATEN_OPS`). A torch-spyre version other
  than :data:`SUPPORTED_TORCH_SPYRE` is logged as a warning and recorded in
  ``SpyrePartitioner.last_report["fallback_provenance"]``.
* ``allowlist`` (conservative / CI mode): delegate only ``validated: true`` ops.

Graph-local exclusions (:func:`_is_graph_local_excluded`) apply in both modes.
"""

import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.utils import tag_mutated_buffer
from executorch.exir.backend.canonical_partitioners.pattern_op_partitioner import (
    generate_partitions_from_list_of_nodes,
)
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from torch.export.exported_program import ExportedProgram
from torch.fx.passes.operator_support import OperatorSupportBase

from .spyre_backend import SPYRE_BACKEND_ID
from .spyre_block_partition import (
    PartitionGroup,
    build_partition_report,
    discover_block_container,
    format_partition_report,
    parse_chunk_size,
    resolve_partition_granularity,
    split_partitions_by_block,
)
from .coverage.ops_registry import unvalidated_op_names, validated_op_names

logger = logging.getLogger(__name__)
# Partition diagnostics are emitted at INFO level on this logger.  Enable with:
#
#   logging.getLogger("executorch.backends.spyre.spyre_partitioner").setLevel(logging.INFO)
#
# The partitioning strategy is controlled by the EXECUTORCH_SPYRE_PARTITION_MODE env var:
#   EXECUTORCH_SPYRE_PARTITION_MODE=fallbacklist   (production default) delegate everything not in torch-spyre
#                                     fallback_ops and not `validated: false` in supported_ops.yaml
#   EXECUTORCH_SPYRE_PARTITION_MODE=allowlist      (conservative / CI) delegate only `validated: true` ops
#
# Typical output lines:
#   PARTITION MODE: fallbacklist
#   SpyrePartitioner: 3 partition(s), 105/112 compute nodes delegated, 7 on CPU.
#     CPU fallback: embedding (edge=aten::embedding, root=torch.embedding, reason=torch-spyre fallback)
#   SHADOW MISMATCH
#     edge=aten::cat  root=torch.cat  allowlist=CPU  spyre=delegate  nodes=2


# Core-aten op names (op.name()) the Spyre SDSC back-end is confirmed to lower.
# Derived from coverage/supported_ops.yaml (the source of truth) — exactly the
# entries flagged `validated: true`. In allowlist mode the partitioner delegates
# only these; new ops are promoted there after an on-device compile+run in the
# coverage harness. In fallbacklist mode this set is only used for the shadow
# comparison and is not a delegation gate.
_SUPPORTED_ATEN_OPS: frozenset[str] = validated_op_names()

# Entries flagged `validated: false` in supported_ops.yaml: known compile
# failures, numeric mismatches (rms_norm, _to_copy), stick-layout scrambles
# (innermost-dim cat) or zero-kernel data movement. They stay on CPU in BOTH
# modes (see _is_graph_local_excluded rule 0), so fallbacklist mode cannot
# delegate an op the project has recorded as unsafe.
_UNVALIDATED_ATEN_OPS: frozenset[str] = unvalidated_op_names()

# Edge-dialect spellings of an unvalidated op. to_edge() lowers ``aten::_to_copy``
# to a dim-order cast with the same (failing) fp16->fp32 behaviour.
_UNVALIDATED_ALIASES: dict[str, str] = {
    "dim_order_ops::_to_dim_order_copy": "aten::_to_copy",
}

# The torch-spyre release this repo is validated against. Must match
# ``torch_spyre`` in compatibility-manifest.json (checked by a host test).
SUPPORTED_TORCH_SPYRE = "0.5.0rc3"


def _aten_name(node) -> Optional[str]:
    """Return the *base* core-aten op name for a call_function node, or ``None``.

    Handles both the edge-dialect ``EdgeOpOverload`` (real op at ``._op``) and a
    plain ``OpOverload``. ``op.name()`` yields the fully-qualified name *including
    the overload* for typed overloads — ``"aten::add.Tensor"``,
    ``"aten::pow.Tensor_Scalar"``, ``"aten::mean.dim"`` — but omits it for the
    ``.default`` overload (``"aten::addmm"``). The allowlist / ``supported_ops.
    yaml`` is keyed on the *base* name (``"aten::add"``), matching how the
    coverage harness probes op families, so we strip the overload suffix here.
    Without this, every typed-overload op silently failed the allowlist check and
    fell back to CPU despite being device-validated.
    """
    target = node.target
    op = getattr(target, "_op", target)  # unwrap EdgeOpOverload -> OpOverload
    name = getattr(op, "name", None)
    if not callable(name):
        return None
    try:
        full = op.name()  # "aten::add.Tensor" | "aten::mm" | "ns::op[.overload]"
    except Exception:  # pragma: no cover - defensive
        return None
    # Drop the overload suffix: "ns::op.overload" -> "ns::op". Base op names never
    # contain a '.', so splitting the post-'::' segment on the first '.' is safe.
    if "::" in full:
        namespace, opname = full.split("::", 1)
        return f"{namespace}::{opname.split('.', 1)[0]}"
    return full


def _torch_op_name(node) -> Optional[str]:
    """Return the high-level PyTorch op name (e.g. ``'torch.embedding'``) for a node, or ``None``.

    Inspects ``node.meta['torch_fn']`` first (the user-level PyTorch function/method),
    falling back to ``node.meta['original_aten']`` (the pre-decomposition ATen op).
    Returns ``None`` if neither metadata field is present or decipherable.
    """
    meta = getattr(node, "meta", None)
    if not isinstance(meta, dict):
        return None

    # 1. Preferred: node.meta['torch_fn']
    torch_fn = meta.get("torch_fn")
    if torch_fn is not None:
        if isinstance(torch_fn, (tuple, list)):
            if len(torch_fn) >= 2 and isinstance(torch_fn[1], str) and torch_fn[1]:
                raw = torch_fn[1].rsplit(".", 1)[-1]
                return f"torch.{raw}"
            if len(torch_fn) >= 1 and isinstance(torch_fn[0], str) and torch_fn[0]:
                raw = re.sub(r"_\d+$", "", torch_fn[0])
                return f"torch.{raw}"
        elif isinstance(torch_fn, str) and torch_fn:
            if torch_fn.startswith("torch."):
                return torch_fn
            raw = torch_fn.rsplit(".", 1)[-1]
            return f"torch.{raw}"
        elif callable(torch_fn):
            name = getattr(torch_fn, "__name__", None)
            if name:
                return f"torch.{name}"

    # 2. Fallback: node.meta['original_aten']
    orig_aten = meta.get("original_aten")
    if orig_aten is not None:
        name_fn = getattr(orig_aten, "name", None)
        if callable(name_fn):
            try:
                full = orig_aten.name()
            except Exception:  # pragma: no cover - defensive
                full = str(orig_aten)
        else:
            full = str(orig_aten)

        if "::" in full:
            opname = full.split("::", 1)[1].split(".", 1)[0]
        elif full.startswith("aten."):
            parts = full.split(".")
            opname = parts[1] if len(parts) > 1 else parts[0]
        else:
            opname = full.split(".", 1)[0]

        return f"torch.{opname.lstrip('_')}"

    return None


# Matmul-family ops that the device delegate cannot lower when *fed a softmax
# output*. An `mm` whose operand is an `aten::_softmax` result fails at run time
# with 0x11 (NotImplemented), categorically — any operand position, any softmax
# axis, terminal or not. `mm` and `softmax` each pass in isolation; only the
# `_softmax -> mm` operand edge is unlowerable. Until the delegate gains that
# lowering, we keep the softmax-fed matmul on CPU (a justified, logged fallback)
# rather than let it 0x11 at run time. This is the SDPA context matmul `probs @ v`.
_MATMUL_ATEN_OPS: frozenset[str] = frozenset(
    {"aten::mm", "aten::bmm", "aten::matmul", "aten::addmm"}
)


def _consumes_softmax(node) -> bool:
    """True if any direct input to ``node`` is an ``aten::_softmax`` output."""
    return any(
        _aten_name(producer) == "aten::_softmax" for producer in node.all_input_nodes
    )


# SDSC stick width. A delegate partition that writes a tensor *wider than one
# stick* back to the host is stored tile-major -- laid out [ntile, rows, 64]
# instead of row-major [rows, ntile*64] -- a silent numeric miscompile
# (multiset-preserving positional scramble). On-device intermediates wider than
# a stick are fine (they stay in device layout and downstream device ops read
# them correctly); only the *host write-back* of a delegate output scrambles.
# So the constraint is scoped to nodes whose value reaches the graph output,
# not to every wide op (e.g. the 128-wide MLP intermediate stays delegated).
_STICK = 64

# Layout-preserving "view" ops: they alias or rescale a tensor's *same*
# underlying values (a reshape/transpose/permute is a view; ``div`` by a scalar
# is elementwise) without changing the stick (last-dim) identity. Walking a
# node's user chain through only these reaches the graph ``output`` iff the
# node's *own* output tensor -- that exact value, unchanged in width -- is what
# gets written back to the host. A matmul is deliberately NOT here: it *consumes*
# its operands and *births a new tensor of a new width*, so a wide value feeding
# a matmul is not itself host-written (the matmul's output is). Walking through
# a matmul would misattribute the matmul's output-reaching to its operands and
# wrongly refuse on-device intermediates (e.g. the 128-wide MLP activation that
# feeds a 64-wide down-projection). ``mul``/``add`` are also excluded: RMSNorm is
# ``x * rsqrt(...)`` and the residual is an ``add``, so admitting them would
# bridge the walk through every norm and residual and mark the whole net
# output-bound.
_VIEW_TAIL_OPS: frozenset[str] = frozenset(
    {
        "aten::view_copy",
        "aten::_reshape_alias_copy",
        "aten::transpose_copy",
        "aten::permute_copy",
        "aten::div",
        # ExecuTorch-Spyre specific: alias_copy is inserted by the edge compiler as a
        # transparent alias of the output tensor; it must be traced through so that
        # wide delegate outputs that reach the graph via alias_copy -> output are
        # correctly identified by _writes_scrambled_output().
        "aten::alias_copy",
        # dim_order_ops::_to_dim_order_copy is a layout-conversion op injected by
        # to_edge(); it is width-preserving and must be transparent to the view-chain
        # traversal so that wide lm_head outputs are detected as scrambled.
        "dim_order_ops::_to_dim_order_copy",
    }
)


def _last_dim(node) -> Optional[int]:
    """Size of ``node``'s output stick axis (last dim), or None if unknown."""
    val = node.meta.get("val")
    try:
        return int(val.shape[-1])
    except (AttributeError, IndexError, TypeError):
        return None


def _is_wide(node) -> bool:
    """True if ``node``'s output spans more than one 64-wide stick."""
    last = _last_dim(node)
    return last is not None and last > _STICK


def _reaches_output_via_views(node) -> bool:
    """True if ``node``'s own output value reaches the graph ``output`` through
    only layout-preserving view ops (never through a matmul, which births a new
    tensor of a new width). So this is True exactly for nodes whose emitted
    tensor -- unchanged in width -- is (aliased into) the model output that gets
    written back to the host.
    """
    seen = set()
    stack = list(node.users)
    while stack:
        user = stack.pop()
        if user in seen:
            continue
        seen.add(user)
        if user.op == "output":
            return True
        if user.op == "call_function" and _aten_name(user) in _VIEW_TAIL_OPS:
            stack.extend(user.users)
    return False


def _writes_scrambled_output(node) -> bool:
    """True if delegating ``node`` would write a >1-stick tensor back to the host.

    Two host-write shapes both scramble (a tile-major ``[ntile, S, 64]`` reorder
    of the row-major result):

      1. ``node``'s own wide output reaches the graph output through view ops --
         the terminal vocab-projection ``mm`` and its ``div``/reshape tail.
      2. ``node``'s wide output feeds a wide, output-bound matmul that will
         itself be kept on CPU by rule 1 -- e.g. the tied-weight ``transpose_copy``
         feeding that projection ``mm``. Its wide output would cross the delegate
         boundary into the CPU matmul and scramble on the way out.

    A wide value that only feeds an on-device matmul whose *own* output is <= 1
    stick (the 128-wide MLP activation -> 64-wide down-projection) stays a device
    intermediate and writes back correctly, so it is intentionally NOT caught.
    """
    if not _is_wide(node):
        return False
    if _reaches_output_via_views(node):
        return True
    return any(
        user.op == "call_function"
        and _aten_name(user) in _MATMUL_ATEN_OPS
        and _is_wide(user)
        and _reaches_output_via_views(user)
        for user in node.users
    )


def _is_graph_local_excluded(node) -> bool:
    """Integration-layer extension point for graph-local delegation exclusions.

    Returns ``True`` if ``node`` should be forced to CPU fallback for a reason
    that is specific to the ExecuTorch-Spyre integration layer (not expressible
    as a simple op-name allowlist entry).  ``False`` means "no integration-layer
    objection; let the allowlist decide."

    This function is the *only* place in the partitioner where graph-local
    heuristics live.  New rules belong here; capability gaps (things torch-spyre
    should learn to compile or raise ``Unsupported`` on) do not belong here and
    should be filed upstream instead.

    Current rules:

    0. **unvalidated op** — the op is recorded as ``validated: false`` in
       ``supported_ops.yaml`` (it failed a device compile, produced wrong
       numbers, or scrambles stick layout), so it stays on CPU.  This makes
       ``fallbacklist`` mode safe by construction: torch-spyre not listing an op
       as a CPU fallback is not evidence that the op is correct on device.

    1. **softmax-fed matmul** — a matmul whose direct operand is an
       ``aten::_softmax`` output 0x11s (NotImplemented) on device.  Refused
       here so partitioning surfaces it as a CPU fallback rather than a
       run-time crash.

    2. **wide-output writeback scramble** — a delegate output wider than one
       64-element stick is written back tile-major-scrambled to the host.
       Terminal >1-stick projections are kept on CPU so the delegate emits
       only single-stick outputs.

    3. **state-scatter exemption** — ``aten::index_put`` whose base tensor is
       a graph ``placeholder`` (KV-cache scatter into a ``kv_state`` buffer)
       is exempt from rule 2: its output is captured on-device (device layout
       preserved, never read back through the host), so the scrambled host
       copy is never consumed.

    Note: fp32 dtype is intentionally *not* excluded here.  torch-spyre's
    SDSC compiler handles IEEE_FP32 natively (``DataFormats.IEEE_FP32`` in
    ``codegen/superdsc.py``); fp32 nodes that cannot be compiled surface as
    ``Unsupported`` from ``preprocess`` rather than a silent CPU fallback.
    """
    name = _aten_name(node)

    # Rule 0: recorded-as-unsafe ops never reach preprocess.
    if _UNVALIDATED_ALIASES.get(name, name) in _UNVALIDATED_ATEN_OPS:
        return True

    # Rule 1: softmax-fed matmul.
    if name in _MATMUL_ATEN_OPS and _consumes_softmax(node):
        return True

    # Rules 2 + 3: wide-output writeback, with state-scatter exemption.
    is_state_scatter = (
        name == "aten::index_put"
        and bool(node.args)
        and getattr(node.args[0], "op", None) == "placeholder"
    )
    if not is_state_scatter and _writes_scrambled_output(node):
        return True

    return False


# --------------------------------------------------------------------------- #
# Shadow mode (observational only)                                             #
# --------------------------------------------------------------------------- #
#
# Compares, per node, the decision that *drives* partitioning today
# (``supported_ops.yaml`` allowlist) against the decision torch-spyre's
# ``torch_spyre/ops/fallbacks.py`` implies. It never influences partitioning:
# it runs after ``partition()`` has finished tagging, inside a catch-all guard,
# and is OFF unless ``EXECUTORCH_SPYRE_SHADOW_MODE`` is truthy (or ``shadow=True`` is passed
# to :class:`SpyrePartitioner`).
#
# Granularity caveat: ``fallbacks.py`` only lists ops that torch-spyre *knows*
# fall back to CPU. "Not in fallback_ops" is therefore reported as ``delegate``
# ("torch-spyre does not declare a CPU fallback"), which is weaker than "torch-
# spyre can compile it". Both sides are compared at base-op granularity
# (overload suffix stripped), like the allowlist. Graph-local exclusions
# (:func:`_is_graph_local_excluded`) apply to either future and are not part of
# the comparison.

SHADOW_ENV_VAR = "EXECUTORCH_SPYRE_SHADOW_MODE"
_DELEGATE = "delegate"
_CPU = "CPU"


def _shadow_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get(SHADOW_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _normalize_fallback_op(op) -> Optional[str]:
    """Normalize one ``fallback_ops`` entry to a base name (``"aten::cumsum"``).

    ``fallback_ops`` mixes ``OpOverload`` objects (``op.name()`` ->
    ``"aten::cumsum.out"``), ``OpOverloadPacket`` objects and plain strings
    (``"spyre::triu_cpu"``). Anything not decipherable yields ``None``.
    """
    if isinstance(op, str):
        full = op
    else:
        name_fn = getattr(op, "name", None)
        try:
            full = (
                name_fn()
                if callable(name_fn)
                else getattr(op, "_qualified_op_name", None)
            )
        except Exception:
            return None
    if not isinstance(full, str) or "::" not in full:
        return None
    namespace, opname = full.split("::", 1)
    return f"{namespace}::{opname.split('.', 1)[0]}"


def _normalize_fallback_ops(ops: Iterable) -> frozenset[str]:
    return frozenset(n for n in map(_normalize_fallback_op, ops) if n is not None)


def load_spyre_fallback_ops() -> tuple[Optional[frozenset[str]], str]:
    """Return ``(base op names torch-spyre falls back to CPU for, reason)``.

    No hard dependency: on any failure returns ``(None, reason)``. Importing
    ``torch_spyre.ops.fallbacks`` raises ``RuntimeError`` (not ``ImportError``)
    unless the ``spyre`` device was registered first, so ``_autoload()`` is
    called (as :meth:`SpyreBackend.preprocess` does) and every ``Exception`` is
    caught.
    """
    try:
        import torch_spyre

        torch_spyre._autoload()
        from torch_spyre.ops.fallbacks import fallback_ops
    except ImportError as e:
        return None, f"torch_spyre not importable ({e})"
    except Exception as e:  # device/runtime registration problems
        return None, f"torch_spyre fallbacks unavailable ({type(e).__name__}: {e})"
    return _normalize_fallback_ops(fallback_ops), ""


def spyre_fallback_provenance(fallbacks: frozenset[str]) -> dict:
    """Identify the fallback list a ``fallbacklist`` partition was built from.

    ``fallback_ops`` is owned by torch-spyre and changes between releases, which
    changes what gets delegated. Recording the installed torch-spyre version and a
    digest of the normalized list makes that visible at export time (see
    ``SpyrePartitioner.last_report["fallback_provenance"]``).
    """
    try:
        from importlib.metadata import version

        installed = version("torch-spyre")
    except Exception:
        try:
            import torch_spyre

            installed = str(getattr(torch_spyre, "__version__", "unknown"))
        except Exception:
            installed = "unknown"
    digest = hashlib.sha256("\n".join(sorted(fallbacks)).encode()).hexdigest()[:12]
    return {
        "torch_spyre": installed,
        "supported_torch_spyre": SUPPORTED_TORCH_SPYRE,
        "version_matches": _normalize_version(installed)
        == _normalize_version(SUPPORTED_TORCH_SPYRE),
        "n_fallback_ops": len(fallbacks),
        "fallback_ops_digest": digest,
    }


def _normalize_version(v: str) -> str:
    """``0.5.0-rc.3`` / ``v0.5.0rc3`` / ``0.5.0rc3+g1234`` -> ``0.5.0rc3``."""
    v = v.strip().lstrip("v").split("+", 1)[0]
    return re.sub(r"[-_.]?rc[-_.]?", "rc", v)


def shadow_decisions(
    name: str, yaml_ops: frozenset[str], spyre_fallbacks: frozenset[str]
) -> tuple[str, str]:
    """``(allowlist_decision, spyre_decision)`` for a base op name."""
    return (
        _DELEGATE if name in yaml_ops else _CPU,
        _CPU if name in spyre_fallbacks else _DELEGATE,
    )


@dataclass
class ShadowReport:
    """Aggregated shadow comparison. Mismatches are de-duplicated by
    ``(edge, root, allowlist, spyre)`` with a node count, to bound log volume."""

    total: int = 0
    agree: int = 0
    mismatches: Counter = field(default_factory=Counter)

    def format_mismatches(self) -> List[str]:
        out = []
        for (edge, root, allowlist_d, spyre_d), n in sorted(
            self.mismatches.items(), key=lambda kv: tuple(x or "" for x in kv[0])
        ):
            out.append(
                "SHADOW MISMATCH\n"
                f"  edge={edge}\n"
                f"  root={root or '?'}\n"
                f"  allowlist={allowlist_d}\n"
                f"  spyre={spyre_d}\n"
                f"  nodes={n}"
            )
        return out


def compute_shadow_report(
    graph_module, yaml_ops: frozenset[str], spyre_fallbacks: frozenset[str]
) -> ShadowReport:
    """Compare allowlist vs torch-spyre fallbacklist decisions for every ``call_function`` node."""
    report = ShadowReport()
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = _aten_name(node)
        if name is None:
            continue
        report.total += 1
        allowlist_d, spyre_d = shadow_decisions(name, yaml_ops, spyre_fallbacks)
        if allowlist_d == spyre_d:
            report.agree += 1
        else:
            report.mismatches[(name, _torch_op_name(node), allowlist_d, spyre_d)] += 1
    return report


def _run_shadow(graph_module) -> Optional[ShadowReport]:
    """Run + log the shadow comparison. Never raises, never mutates the graph."""
    try:
        spyre_fallbacks, reason = load_spyre_fallback_ops()
        if spyre_fallbacks is None:
            logger.info("SHADOW: comparison skipped: %s", reason)
            return None
        report = compute_shadow_report(
            graph_module, _SUPPORTED_ATEN_OPS, spyre_fallbacks
        )
        for msg in report.format_mismatches():
            logger.info("%s", msg)
        logger.info(
            "SHADOW: %d/%d compute nodes agree (allowlist vs torch-spyre fallbacklist), "
            "%d node(s) in %d distinct mismatch(es).",
            report.agree,
            report.total,
            report.total - report.agree,
            len(report.mismatches),
        )
        return report
    except Exception as e:  # pragma: no cover - shadow must never break partition()
        logger.info("SHADOW: comparison failed: %s: %s", type(e).__name__, e)
        return None


# --------------------------------------------------------------------------- #
# Partition modes                                                              #
# --------------------------------------------------------------------------- #
#
# Two coexisting partitioning strategies (graph-local rules apply to both):
#
#   "allowlist"    op in supported_ops.yaml              -> delegate, else CPU
#                  (conservative; only ops with confirmed on-device validation)
#   "fallbacklist" op in torch-spyre ``fallback_ops``    -> CPU,      else delegate
#                  (torch-spyre owns capability; delegate everything it does not
#                   explicitly declare a CPU fallback for)
#
# The default (and the supported production contract) is ``fallbacklist``; the
# ``validated: false`` ops are excluded from it by a graph-local rule.  Override
# process-wide with the ``EXECUTORCH_SPYRE_PARTITION_MODE`` env var or an explicit ``mode=``
# constructor argument:
#
#   EXECUTORCH_SPYRE_PARTITION_MODE=allowlist python export.py ...
#   SpyrePartitioner(mode="allowlist")

PARTITION_MODE_ENV_VAR = "EXECUTORCH_SPYRE_PARTITION_MODE"
MODE_ALLOWLIST = "allowlist"
MODE_FALLBACKLIST = "fallbacklist"
_PARTITION_MODES = (MODE_ALLOWLIST, MODE_FALLBACKLIST)

_REASON_YAML = "not in supported_ops.yaml"
_REASON_SPYRE = "torch-spyre fallback"
_REASON_LOCAL = "graph-local rule"
_REASON_UNVALIDATED = "validated: false in supported_ops.yaml"
_REASON_NON_ATEN = "not an aten op"


def resolve_partition_mode(explicit: Optional[str] = None) -> str:
    """Explicit ``mode`` > ``EXECUTORCH_SPYRE_PARTITION_MODE`` env var > ``"fallbacklist"``.

    Accepted values: ``"allowlist"``, ``"fallbacklist"``.  Raises
    :exc:`ValueError` on anything else.
    """
    mode = explicit
    if mode is None:
        mode = (
            os.environ.get(PARTITION_MODE_ENV_VAR, "").strip().lower()
            or MODE_FALLBACKLIST
        )
    if mode not in _PARTITION_MODES:
        raise ValueError(
            f"unknown Spyre partition mode {mode!r}; expected one of {_PARTITION_MODES}"
        )
    return mode


class _SpyreOperatorSupport(OperatorSupportBase):
    """Decides, per node, whether the Spyre delegate may take it.

    ``allowlist`` mode: delegate iff the op is in ``supported_ops.yaml``.
    ``fallbacklist`` mode: delegate unless torch-spyre declares a CPU fallback
    for the op (``spyre_fallbacks``). Graph-local exclusions apply to both.
    """

    def __init__(
        self,
        mode: str = MODE_FALLBACKLIST,
        spyre_fallbacks: Optional[frozenset[str]] = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.spyre_fallbacks = spyre_fallbacks or frozenset()

    def cpu_reason(self, node) -> Optional[str]:
        """Why ``node`` must stay on CPU, or ``None`` if it may be delegated."""
        if node.op != "call_function":
            return _REASON_NON_ATEN
        name = _aten_name(node)
        if name is None:
            return _REASON_NON_ATEN
        if self.mode == MODE_FALLBACKLIST:
            if name in self.spyre_fallbacks:
                return _REASON_SPYRE
        elif name not in _SUPPORTED_ATEN_OPS:
            return _REASON_YAML
        if _UNVALIDATED_ALIASES.get(name, name) in _UNVALIDATED_ATEN_OPS:
            return _REASON_UNVALIDATED
        if _is_graph_local_excluded(node):
            return _REASON_LOCAL
        return None

    def is_node_supported(self, submodules, node) -> bool:
        return self.cpu_reason(node) is None


class SpyrePartitioner(Partitioner):
    """Capability-based partitioner: delegate supported ops, CPU-fallback rest.

    Uses ExecuTorch's ``generate_partitions_from_list_of_nodes`` (which wraps
    ``torch.fx``'s :class:`CapabilityBasedPartitioner`) to grow maximal
    connected subgraphs over allowlisted ops. With ``granularity="model"`` each
    proposed partition becomes one delegate; with the default
    ``granularity="block"`` each proposed partition is further cut along
    decoder-block boundaries (see :mod:`spyre_block_partition`) so the compile
    unit is one block.  Each delegate is tagged ``tag{id}[_{j}]`` and mapped to
    a single :class:`DelegationSpec` naming :data:`SPYRE_BACKEND_ID`;
    ``to_backend`` then hands each tagged subgraph to ``SpyreBackend.preprocess``
    and replaces it with one delegate call node.  Untagged ``call_function``
    nodes remain on CPU and are logged.
    """

    def __init__(
        self,
        compile_specs: List[CompileSpec] | None = None,
        shadow: Optional[bool] = None,
        mode: Optional[str] = None,
        granularity: Optional[str] = None,
    ) -> None:
        """``mode``: ``"fallbacklist"`` (production default; delegate everything
        except ops torch-spyre declares a CPU fallback for and ops recorded as
        ``validated: false``) or ``"allowlist"`` (conservative / CI mode;
        delegate only ``validated: true`` ops in ``supported_ops.yaml``). ``None`` defers to the
        ``EXECUTORCH_SPYRE_PARTITION_MODE`` env var, then ``"fallbacklist"``.
        ``"fallbacklist"`` raises :exc:`RuntimeError` if torch-spyre's fallback
        list cannot be loaded — it never silently falls back to the allowlist.

        ``granularity``: ``"block"`` (default; one delegate per decoder block)
        or ``"model"`` (monolithic: one delegate per maximal connected
        subgraph).  ``None`` defers to ``EXECUTORCH_SPYRE_PARTITION_GRANULARITY``, then
        ``"block"``.  Graphs without repeated-block metadata behave the same
        under both.
        """
        super().__init__()
        # Observational only; None -> decided by the EXECUTORCH_SPYRE_SHADOW_MODE env var.
        self._shadow = shadow
        self.mode = resolve_partition_mode(mode)
        self.granularity = resolve_partition_granularity(granularity)
        self.last_report: Optional[dict] = None
        self.compile_specs: List[CompileSpec] = compile_specs or []
        self.delegation_spec = DelegationSpec(SPYRE_BACKEND_ID, self.compile_specs)
        spyre_fallbacks = None
        self.fallback_provenance: Optional[dict] = None
        if self.mode == MODE_FALLBACKLIST:
            spyre_fallbacks, reason = load_spyre_fallback_ops()
            if spyre_fallbacks is None:
                raise RuntimeError(
                    f"SpyrePartitioner(mode='fallbacklist') needs torch-spyre's "
                    f"fallback_ops: {reason}. Use mode='allowlist' for the "
                    f"supported_ops.yaml path."
                )
            self.fallback_provenance = spyre_fallback_provenance(spyre_fallbacks)
            if not self.fallback_provenance["version_matches"]:
                logger.warning(
                    "torch-spyre %s is installed but this repo is validated against "
                    "%s; the fallback_ops list (and so the set of delegated ops) "
                    "may differ. fallback_ops digest=%s.",
                    self.fallback_provenance["torch_spyre"],
                    SUPPORTED_TORCH_SPYRE,
                    self.fallback_provenance["fallback_ops_digest"],
                )
        self.op_support = _SpyreOperatorSupport(self.mode, spyre_fallbacks)

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        graph_module = exported_program.graph_module

        partitions = generate_partitions_from_list_of_nodes(
            graph_module, op_support=self.op_support
        )

        container = None
        chunk_size = parse_chunk_size(self.granularity)
        if chunk_size is not None:
            container = discover_block_container(n for p in partitions for n in p.nodes)
            groups = split_partitions_by_block(
                graph_module, partitions, container, chunk_size=chunk_size
            )
        else:
            groups = [
                PartitionGroup(
                    tag=f"tag{p.id}", origin=p.id, blocks=[], nodes=list(p.nodes)
                )
                for p in partitions
            ]

        partition_tags: dict[str, DelegationSpec] = {}
        tagged = set()
        for group in groups:
            partition_tags[group.tag] = self.delegation_spec
            for node in group.nodes:
                node.meta["delegation_tag"] = group.tag
                tagged.add(node)

        # No silent coverage gaps: report every compute node left on CPU.
        fallback = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node not in tagged
        ]
        n_compute = sum(1 for n in graph_module.graph.nodes if n.op == "call_function")
        # Diagnostics below: see logger comment near the top of this file.
        logger.info("PARTITION MODE: %s", self.mode)
        logger.info("PARTITION GRANULARITY: %s", self.granularity)
        logger.info(
            "SpyrePartitioner: %d partition(s), %d/%d compute nodes delegated, "
            "%d on CPU.",
            len(groups),
            n_compute - len(fallback),
            n_compute,
            len(fallback),
        )
        # op_support may be replaced by a caller (e.g. the coverage harness) with
        # an object that has no cpu_reason(); degrade to a generic reason.
        cpu_reason = getattr(self.op_support, "cpu_reason", None)
        for node in fallback:
            reason = cpu_reason(node) if cpu_reason else None
            logger.info(
                "  CPU fallback: %s (edge=%s, root=%s, reason=%s)",
                node.name,
                _aten_name(node) or "?",
                _torch_op_name(node) or "?",
                reason or "not partitioned",
            )

        if _shadow_enabled(self._shadow):
            _run_shadow(graph_module)

        # Standard ExecuTorch mutable-buffer ownership: a registered buffer that
        # is mutated inside a partition (e.g. a KV cache written by index_put)
        # is tagged with that partition, so the delegate sees it as
        # BUFFER / BUFFER_MUTATION instead of the top-level program threading it
        # through as a USER_INPUT / USER_OUTPUT round trip. Must run after all
        # node tagging above: it copies the tag from the buffer's users.
        tag_mutated_buffer(exported_program)

        # Report after ownership tagging so KV-buffer ownership is visible.
        self.last_report = build_partition_report(
            exported_program,
            groups,
            mode=self.mode,
            granularity=self.granularity,
            container=container,
            n_cpu=len(fallback),
        )
        self.last_report["fallback_provenance"] = self.fallback_provenance
        for line in format_partition_report(self.last_report):
            logger.info("%s", line)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )
