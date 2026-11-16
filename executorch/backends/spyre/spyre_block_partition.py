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
"""Block-level (per decoder layer) delegation for the Spyre partitioner.

``CapabilityBasedPartitioner`` grows the *largest* connected supported subgraph,
so a full-model export collapses into one giant delegate (for a 40-layer
Granite: ~942 SDSCs, ~75 s in dbo-opt, then a timeout).  That is an incidental
result of the greedy algorithm, not an architectural requirement.  This module
splits each maximal partition along decoder-block boundaries so the compile
unit becomes one block.

Block recovery uses only ``node.meta["nn_module_stack"]`` (present on every
edge-dialect node; ``source_fn_stack`` does not survive ``torch.export``):

  * the *block container* is the module path ``<prefix>`` for which the stack
    contains ``<prefix>.<int>`` entries with the most distinct indices (a
    ``ModuleList`` such as ``model.layers``); no family-specific names;
  * a node's block key is the ``<int>`` of that entry (``None`` if absent);
  * keyless nodes (embedding, rope gather, final norm, lm_head, ...) join the
    nearest preceding keyed node in graph order, or the first block when they
    precede all of them, so N blocks yield exactly N delegates.

Splitting a valid partition can create a cycle between its own sub-groups
(block i -> block j -> block i); such sub-groups are merged (SCC) and logged.
A cycle can only form inside one original partition, so merging is local.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

GRANULARITY_ENV_VAR = "EXECUTORCH_SPYRE_PARTITION_GRANULARITY"
GRANULARITY_BLOCK = "block"
GRANULARITY_MODEL = "model"
PARTITION_GRANULARITIES = (GRANULARITY_BLOCK, GRANULARITY_MODEL)

# ``<prefix>.<int>`` where the int is a whole path component (not the tail only:
# a helper may call ``layers.3.attn`` without ``layers.3`` ever being a stack
# entry).
_INDEXED = re.compile(r"\.(?P<idx>\d+)(?=\.|$)")


def resolve_partition_granularity(explicit: Optional[str] = None) -> str:
    """Explicit arg > ``EXECUTORCH_SPYRE_PARTITION_GRANULARITY`` env var > ``"block"``."""
    import os

    value = explicit
    if value is None:
        value = (
            os.environ.get(GRANULARITY_ENV_VAR, "").strip().lower() or GRANULARITY_BLOCK
        )
    else:
        value = value.strip().lower()

    if value in PARTITION_GRANULARITIES:
        return value
    if value.startswith("chunk:"):
        n_str = value[len("chunk:") :]
        try:
            n = int(n_str)
            if n < 1:
                raise ValueError
        except ValueError:
            raise ValueError(
                f"invalid Spyre partition granularity {value!r}; "
                f"expected chunk:<N> with N >= 1"
            ) from None
        return f"chunk:{n}"
    raise ValueError(
        f"unknown Spyre partition granularity {value!r}; "
        f"expected one of {PARTITION_GRANULARITIES} or 'chunk:<N>'"
    )


def parse_chunk_size(granularity: str) -> Optional[int]:
    """Return the integer chunk size for 'block' (1) or 'chunk:<N>', or None for 'model'."""
    if granularity == GRANULARITY_BLOCK:
        return 1
    if granularity.startswith("chunk:"):
        return int(granularity.split(":", 1)[1])
    return None


@dataclass
class PartitionGroup:
    """One delegate: the nodes that share ``tag``."""

    tag: Optional[str] = None
    origin: int = -1  # id of the greedy partition this group was cut from
    blocks: List[int] = field(
        default_factory=list
    )  # block indices owned (empty when no block metadata)
    nodes: List = field(default_factory=list)


def _stack_paths(node) -> List[str]:
    stack = node.meta.get("nn_module_stack")
    if not stack:
        return []
    return [v[0] if isinstance(v, (tuple, list)) else str(v) for v in stack.values()]


def discover_block_container(nodes: Iterable) -> Optional[str]:
    """Module path of the repeated-block container, or ``None``.

    Candidates are every ``<prefix>`` followed by an integer path component in
    some ``nn_module_stack`` path.

    Heuristic Design & Rationale:
    -----------------------------
    1. Why largest repeated indexed container:
       Transformer architectures (e.g., Llama, Granite, Mistral, GPT) structure
       their backbone as a primary list of identical decoder layers (e.g.,
       ``model.layers.0`` .. ``model.layers.39``). In any exported whole-model
       graph, the module container representing the decoder backbone will contain
       the highest multiplicity of distinct indices (e.g., 32, 40, or 80 layers).

    2. Why ties prefer shallower containers (``-p.count(".")``):
       If two containers exhibit identical index counts (for example, in a model
       with N layers where an internal sub-structure also has N items), the
       container higher up in the module hierarchy (fewer dot-separated segments)
       corresponds to the enclosing transformer layer rather than a nested
       component.

    3. Filtering out nested ModuleLists (heads, experts, per-head projections):
       Candidates nested *inside* another indexed container (e.g.
       ``model.layers.0.mlp.experts.2`` or ``layers.0.self_attn.q_heads.4``) are
       pruned by the ``top_level`` filter because their prefix contains an
       ancestor indexed segment (``layers.<digit>``). Thus, sub-block structures
       are never misidentified as decoder layers even in single- or few-layer graphs.

    4. Why this is a heuristic rather than an exported annotation:
       ``torch.export`` flattens the PyTorch ``nn.Module`` hierarchy into a flat
       FX graph, retaining module paths only as diagnostic strings in
       ``node.meta["nn_module_stack"]``. Because PyTorch export does not provide
       a formal "decoder block" semantic tag, inferring the container prefix from
       module stack paths allows automatic, zero-config partitioning across diverse
       model families (HuggingFace, Torchtune, custom wrappers) while preserving
       the option to explicitly pass an override ``container`` parameter.

    Returns the winning container prefix string (e.g. ``"model.layers"``), or
    ``None`` if no repeated container (>= 2 indices) is found.
    """
    indices: Dict[str, set] = defaultdict(set)
    for node in nodes:
        for path in _stack_paths(node):
            for m in _INDEXED.finditer(path):
                indices[path[: m.start()]].add(int(m["idx"]))
    top_level = [
        p
        for p in indices
        if not any(
            q != p
            and p.startswith(q + ".")
            and p[len(q) + 1 :].split(".", 1)[0].isdigit()
            for q in indices
        )
    ]
    top_level = [p for p in top_level if len(indices[p]) >= 2]
    if not top_level:
        return None
    return max(top_level, key=lambda p: (len(indices[p]), -p.count(".")))


def block_of(node, container: str) -> Optional[int]:
    """Block index of ``node`` under ``container``, or ``None``."""
    prefix = container + "."
    for path in _stack_paths(node):
        if path.startswith(prefix):
            head = path[len(prefix) :].split(".", 1)[0]
            if head.isdigit():
                return int(head)
    return None


def _fill_keys(keys: Sequence[Optional[int]]) -> List[Optional[int]]:
    """Forward-fill ``None`` keys; leading ``None`` take the first real key."""
    first = next((k for k in keys if k is not None), None)
    out: List[Optional[int]] = []
    prev = first
    for k in keys:
        if k is not None:
            prev = k
        out.append(prev)
    return out


def _sccs(adj: Dict[int, set]) -> List[List[int]]:
    """Tarjan SCCs of a small graph (recursion depth <= #groups)."""
    index: Dict[int, int] = {}
    low: Dict[int, int] = {}
    on_stack: set = set()
    stack: List[int] = []
    out: List[List[int]] = []
    counter = [0]

    def visit(v: int) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in adj[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in adj:
        if v not in index:
            visit(v)
    return out


def _split_partition(
    pid: int,
    nodes: List,
    order: Dict,
    container: Optional[str],
    chunk_size: int = 1,
) -> List[PartitionGroup]:
    """Cut one greedy partition into per-block or per-chunk groups (acyclic)."""
    nodes = sorted(nodes, key=order.__getitem__)
    if container is None:
        return [PartitionGroup(origin=pid, blocks=[], nodes=nodes)]
    keys = _fill_keys([block_of(n, container) for n in nodes])
    if all(k is None for k in keys):
        return [PartitionGroup(origin=pid, blocks=[], nodes=nodes)]

    # gid per distinct cluster key, in first-appearance order.
    gid_of_key: Dict[Optional[int], int] = {}
    node_gid: Dict = {}
    for n, k in zip(nodes, keys):
        cluster_id = (k // chunk_size) if k is not None else None
        node_gid[n] = gid_of_key.setdefault(cluster_id, len(gid_of_key))

    adj: Dict[int, set] = {g: set() for g in gid_of_key.values()}
    for n in nodes:
        for a in n.all_input_nodes:
            if a in node_gid and node_gid[a] != node_gid[n]:
                adj[node_gid[a]].add(node_gid[n])

    # Merge any cycle between sub-groups (SCC condensation is acyclic).
    remap = {g: g for g in adj}
    key_of_gid = {g: k for k, g in gid_of_key.items()}
    for comp in _sccs(adj):
        if len(comp) > 1:
            logger.warning(
                "block partition: partition %d: blocks %s form a dependency "
                "cycle; merging them into one delegate",
                pid,
                sorted(key_of_gid[g] for g in comp),
            )
            root = min(comp)
            for g in comp:
                remap[g] = root
    node_gid = {n: remap[g] for n, g in node_gid.items()}

    grouped: Dict[int, PartitionGroup] = {}
    for n, k in zip(nodes, keys):
        g = grouped.setdefault(
            node_gid[n], PartitionGroup(origin=pid, blocks=[], nodes=[])
        )
        g.nodes.append(n)
        if k is not None and k not in g.blocks:
            g.blocks.append(k)
    return list(grouped.values())


def split_partitions_by_block(
    graph_module,
    partitions: Sequence,
    container: Optional[str] = None,
    chunk_size: int = 1,
) -> List[PartitionGroup]:
    """Split greedy ``partitions`` along decoder-block boundaries.

    ``container`` overrides auto-discovery.  Groups are tagged ``tag{pid}``
    when a partition stays whole, else ``tag{pid}_{j}``.
    """
    order = {n: i for i, n in enumerate(graph_module.graph.nodes)}
    if container is None:
        all_nodes = [n for p in partitions for n in p.nodes]
        container = discover_block_container(all_nodes)
    logger.info("block partition: container=%s", container)

    groups: List[PartitionGroup] = []
    for p in partitions:
        parts = _split_partition(
            p.id, list(p.nodes), order, container, chunk_size=chunk_size
        )
        for j, g in enumerate(parts):
            g.tag = f"tag{p.id}" if len(parts) == 1 else f"tag{p.id}_{j}"
        groups.extend(parts)
    return groups


# --------------------------------------------------------------------------- #
# Diagnostics                                                                  #
# --------------------------------------------------------------------------- #


def kv_ownership(exported_program) -> Dict[str, List[str]]:
    """Mutated-buffer FQN -> sorted distinct delegation tags of its users.

    A well-formed result maps every buffer to exactly one tag: the delegate
    that owns the state.  More than one tag means the state is shared across
    delegates (a correctness hazard); an empty list means no delegated user.
    """
    sig = exported_program.graph_signature
    mutated = set(sig.buffers_to_mutate.values())
    out: Dict[str, List[str]] = {}
    for node in exported_program.graph.nodes:
        if node.op != "placeholder":
            continue
        fqn = sig.inputs_to_buffers.get(node.name)
        if fqn is None or fqn not in mutated:
            continue
        tags = {u.meta.get("delegation_tag") for u in node.users}
        out[fqn] = sorted(t for t in tags if t is not None)
    return out


def build_partition_report(
    exported_program,
    groups: Sequence[PartitionGroup],
    *,
    mode: str,
    granularity: str,
    container: Optional[str],
    n_cpu: int,
) -> dict:
    """Structured partition report (see :func:`format_partition_report`)."""
    ownership = kv_ownership(exported_program)
    kv_by_tag: Dict[str, List[str]] = defaultdict(list)
    for fqn, tags in ownership.items():
        for t in tags:
            kv_by_tag[t].append(fqn)
    return {
        "mode": mode,
        "granularity": granularity,
        "block_container": container,
        "n_partitions": len(groups),
        "n_cpu_nodes": n_cpu,
        "partitions": [
            {
                "tag": g.tag,
                "origin": g.origin,
                "n_nodes": len(g.nodes),
                "blocks": list(g.blocks),
                "kv_buffers": sorted(kv_by_tag.get(g.tag, [])),
            }
            for g in groups
        ],
        "kv_ownership": ownership,
        "kv_violations": sorted(f for f, t in ownership.items() if len(t) > 1),
    }


def format_partition_report(report: dict) -> List[str]:
    lines = [
        "=== Spyre PARTITION REPORT ===",
        f"  granularity={report['granularity']} mode={report['mode']} "
        f"block_container={report['block_container']}",
        f"  partitions={report['n_partitions']} cpu_nodes={report['n_cpu_nodes']}",
    ]
    for p in report["partitions"]:
        blocks = p["blocks"]
        span = (
            f"{blocks[0]}..{blocks[-1]}"
            if len(blocks) > 1
            else (str(blocks[0]) if blocks else "-")
        )
        lines.append(
            f"  {p['tag']:>10s}: {p['n_nodes']:5d} nodes  blocks={span:<7s} "
            f"kv={','.join(p['kv_buffers']) or '-'}"
        )
    prov = report.get("fallback_provenance")
    if prov:
        lines.append(
            f"  torch-spyre={prov['torch_spyre']} (supported={prov['supported_torch_spyre']}, "
            f"match={prov['version_matches']}) fallback_ops={prov['n_fallback_ops']} "
            f"digest={prov['fallback_ops_digest']}"
        )
    if report["kv_violations"]:
        lines.append(
            f"  KV OWNERSHIP VIOLATION (state shared across delegates): "
            f"{report['kv_violations']}"
        )
    return lines
