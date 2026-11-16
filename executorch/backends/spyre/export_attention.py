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
"""Export a GQA attention block as a Spyre-delegated ``.pte`` (Phase 8, 8c).

This is the first *multi-primitive* subgraph on device: Q/K/V projections ->
RoPE -> scaled-dot-product attention (math decomposition) -> output projection,
with grouped-query attention. It stitches together every 8a/8b workaround into
one graph that must **fully delegate** and match a CPU fp16 reference on device
within the fp16 envelope (``atol=1e-1, rtol=2e-2``).

Module (``AttentionBlock``), all fp16, **per-head projections / 2D-mm**, **B=1**::

    x [1,S,E] -> x2d [S,E]                                # free leading-1 drop (B==1)
      k_j = rope(k_head[j](x2d))  for j in kv heads        # each [S,D], dense contiguous
      v_j =      v_head[j](x2d)                             # each [S,D]
      for each query head h (kv head j = h // rep):
        q_h = rope(q_head[h](x2d))                          # [S,D], dense contiguous
        ah  = softmax(mm(q_h, k_jᵀ)*scale + mask) @ v_j     # 2D-`mm` SDPA math, [S,D]
        y  += o_head[h](ah)                                 # Linear(D,E); weight-split o_proj
      -> y  [1,S,E]

**Q/K/V/O are all SPLIT into per-head ``Linear`` projections.** A single
``Linear(E, H*D)`` projection's ``[S, H*D]`` output interleaves the heads on the
innermost dim, so slicing one head off dim 2 of the ``[1,S,H,D]`` view yields a
``[S,D]`` tensor with **row-stride ``H*D`` (heads gappy, not stick-aligned)**. Per-head
projections make each head a dense contiguous ``[S,D]`` (row-stride ``D`` == exactly
one stick), so every matmul operand is stick-aligned.

**Everything is 2D ``mm``, not 3D ``bmm``.** With ``B=1`` (decoder inference) the
per-head attention is a chain of 2D ``[S,D]`` matmuls. This is not cosmetic:
o_proj on a 3D ``[1,S,D]`` context (a ``bmm`` output) lowers on device to a
``batchmatmul`` with ``shared_weight_unit_bmm`` that **SIGABRTs dxp_standalone**.

Reuses the generic ``spyre_coverage_runner`` (manifest-driven, multi-input,
multi-case) rather than a bespoke C++ runner: this script emits ``manifest.txt``
alongside the ``.bin`` inputs and the CPU ``ref.bin``.

Run in the AOT overlay venv. ``--inspect`` runs the partition
device-free (prints the edge graph + any CPU-fallback nodes) so delegation is
confirmed before a device compile is spent. Building the ``.pte`` needs the Spyre
toolchain (``dxp_standalone``); running it needs ``FLEX_DEVICE=PF``.
"""

import argparse
import json
import math
import os

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_details import CompileSpec

from executorch.backends.spyre import SpyrePartitioner
from executorch.backends.spyre.rope import (
    apply_rope,
    build_rope_tables,
    rotation_matrix,
)

# Phase 7 Track 2 bucketing: default seq-len buckets. Each compiles one JobPlan;
# the runtime selects by input shape. Opt in with --buckets; empty keeps the
# byte-identical single-shape path.
SEQ_BUCKETS = [8, 16, 32]

# Compile-spec key mirrored in spyre_backend._BUCKET_COMPILE_SPEC_KEY.
_BUCKET_COMPILE_SPEC_KEY = "spyre_dynamic_buckets"


class AttentionBlock(torch.nn.Module):
    """GQA attention: per-head Q/K/V proj -> RoPE + 2D-mm SDPA -> summed o_proj, fp16.

    Per-head / clone-free (see module docstring): Q/K/V/O are each split into
    per-head ``Linear`` projections so every head is a dense contiguous ``[S,D]``
    (no gappy head ``slice_copy``), attended per-head with 2D ``mm`` (``B==1``);
    the per-head o_proj outputs are summed. ``cos``/``sin``/``mask`` are forward
    inputs (seq-dependent, host-precomputed); ``scale``/``rot`` are fp16 buffers
    (seq-independent). ``n_heads`` must be a multiple of ``n_kv_heads``
    (``n_kv_heads == n_heads`` is plain MHA).
    """

    def __init__(self, embed_dim: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) not divisible by n_heads ({n_heads})"
            )
        if n_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) not a multiple of n_kv_heads ({n_kv_heads})"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = embed_dim // n_heads
        d = self.head_dim
        self.q_heads = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim, d, bias=False) for _ in range(n_heads)]
        )
        self.k_heads = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim, d, bias=False) for _ in range(n_kv_heads)]
        )
        self.v_heads = torch.nn.ModuleList(
            [torch.nn.Linear(embed_dim, d, bias=False) for _ in range(n_kv_heads)]
        )
        self.o_heads = torch.nn.ModuleList(
            [torch.nn.Linear(d, embed_dim, bias=False) for _ in range(n_heads)]
        )
        # Seq-independent constants ride in as fp16 buffers (clean lifted consts).
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(d), dtype=torch.float16)
        )
        self.register_buffer("rot", rotation_matrix(d, dtype=torch.float16))
        # 8c bisection: when set to one of qh/ks/vs/raw/scores/probs/ah/yh, forward
        # truncates the HEAD-0 chain at that link and returns the 2D intermediate
        # (instead of the o_proj-summed [1,S,E] output). Lets a device dump localize
        # exactly which link of the per-head SDPA math diverges. None => full output.
        self._emit = None

    def forward(self, x, cos, sin, mask):
        b, s, _ = x.shape
        hq, hkv, d = self.n_heads, self.n_kv_heads, self.head_dim
        rep = hq // hkv
        if b != 1:
            raise ValueError(f"AttentionBlock is B=1 only (2D-mm path), got B={b}")
        x2d = x.reshape(s, self.embed_dim)  # [S,E]
        ks = [
            apply_rope(self.k_heads[j](x2d), cos, sin, self.rot) for j in range(hkv)
        ]  # [S,D]
        vs = [self.v_heads[j](x2d) for j in range(hkv)]  # [S,D]
        if self._emit is not None:
            qh0 = apply_rope(self.q_heads[0](x2d), cos, sin, self.rot)  # [S,D]
            if self._emit == "qh":
                return qh0
            if self._emit == "ks":
                return ks[0]
            if self._emit == "vs":
                return vs[0]
            raw = torch.mm(qh0, ks[0].transpose(0, 1))  # [S,S]
            if self._emit == "raw":
                return raw
            scores0 = raw * self.scale + mask  # [S,S]
            if self._emit == "scores":
                return scores0
            probs0 = torch.softmax(scores0, dim=-1)  # [S,S]
            if self._emit == "probs":
                return probs0
            ah0 = torch.mm(probs0, vs[0])  # [S,D]
            if self._emit == "ah":
                return ah0
            return self.o_heads[0](ah0)  # [S,E], "yh"
        y = None
        for h in range(hq):
            j = h // rep  # shared KV head for this query head (GQA)
            qh = apply_rope(self.q_heads[h](x2d), cos, sin, self.rot)  # [S,D]
            scores = torch.mm(qh, ks[j].transpose(0, 1)) * self.scale + mask  # [S,S]
            ah = torch.mm(torch.softmax(scores, dim=-1), vs[j])  # [S,D]
            yh = self.o_heads[h](ah)  # [S,E]
            y = yh if y is None else y + yh
        return y.reshape(b, s, self.embed_dim)  # [1,S,E]


def _ref_forward(mod: AttentionBlock, x, cos, sin, mask):
    """CPU fp16 reference: fp32 compute rounded to fp16 (CPU has no fp16 kernels)."""
    b, s, _ = x.shape
    hq, hkv, d = mod.n_heads, mod.n_kv_heads, mod.head_dim
    xf = x.float()
    p = {k_: v_.float() for k_, v_ in mod.state_dict().items()}
    cosf, sinf, rotf = cos.float(), sin.float(), mod.rot.float()
    scale = float(mod.scale)

    q_weight = torch.cat([p[f"q_heads.{h}.weight"] for h in range(hq)], dim=0)
    k_weight = torch.cat([p[f"k_heads.{j}.weight"] for j in range(hkv)], dim=0)
    v_weight = torch.cat([p[f"v_heads.{j}.weight"] for j in range(hkv)], dim=0)

    def proj(w, h):
        return torch.nn.functional.linear(xf, w).view(b, s, h, d).transpose(1, 2)

    def rope(t):  # t: [B,H,S,D]
        bb, hh, ss, dd = t.shape
        rotated = (t.reshape(bb * hh * ss, dd) @ rotf).reshape(bb, hh, ss, dd)
        return t * cosf + rotated * sinf

    q = rope(proj(q_weight, hq))
    k = rope(proj(k_weight, hkv))
    v = proj(v_weight, hkv)
    if hq != hkv:
        rep = hq // hkv
        k = k[:, :, None].expand(b, hkv, rep, s, d).reshape(b, hq, s, d)
        v = v[:, :, None].expand(b, hkv, rep, s, d).reshape(b, hq, s, d)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale + mask.float()
    ctx = torch.matmul(torch.softmax(scores, dim=-1), v)  # [B,Hq,S,D]
    ctx = ctx.transpose(1, 2).reshape(b, s, hq * d)
    o_weight = torch.cat([p[f"o_heads.{h}.weight"] for h in range(hq)], dim=1)
    y = torch.nn.functional.linear(ctx, o_weight)
    return y.to(torch.float16)


def _ref_emit(mod: AttentionBlock, x, cos, sin, mask, emit: str):
    """CPU fp16 reference for a HEAD-0 intermediate, mirroring the forward emit path."""
    b, s, _ = x.shape
    xf = x.float().reshape(s, mod.embed_dim)  # [S,E]
    p = {k_: v_.float() for k_, v_ in mod.state_dict().items()}
    cosf, sinf, rotf = cos.float(), sin.float(), mod.rot.float()
    scale = float(mod.scale)

    def lin(name):  # xf @ Wᵀ, [S,E]@[E,D] -> [S,D]
        return xf @ p[name].transpose(0, 1)

    def rope2d(t):  # [S,D]
        return t * cosf + (t @ rotf) * sinf

    qh = rope2d(lin("q_heads.0.weight"))
    ks0 = rope2d(lin("k_heads.0.weight"))
    vs0 = lin("v_heads.0.weight")
    if emit == "qh":
        return qh.to(torch.float16)
    if emit == "ks":
        return ks0.to(torch.float16)
    if emit == "vs":
        return vs0.to(torch.float16)
    raw = qh @ ks0.transpose(0, 1)
    if emit == "raw":
        return raw.to(torch.float16)
    scores = raw * scale + mask.float()
    if emit == "scores":
        return scores.to(torch.float16)
    probs = torch.softmax(scores, dim=-1)
    if emit == "probs":
        return probs.to(torch.float16)
    ah = probs @ vs0
    if emit == "ah":
        return ah.to(torch.float16)
    return (ah @ p["o_heads.0.weight"].transpose(0, 1)).to(torch.float16)  # yh [S,E]


def _dump_fp16(path: str, t: torch.Tensor) -> None:
    """Write a contiguous fp16 tensor as raw little-endian bytes."""
    t.detach().to(torch.float16).contiguous().cpu().numpy().tofile(path)


def _causal_mask(s: int) -> torch.Tensor:
    """fp16 additive causal mask [S,S]: 0 on/below diagonal, -inf above."""
    return torch.triu(torch.full((s, s), float("-inf")), diagonal=1).to(torch.float16)


def _make_case_inputs(embed_dim: int, head_dim: int, seq: int, seed: int):
    """Host-precompute (x, cos, sin, mask) for one sequence length."""
    torch.manual_seed(seed + seq)  # vary per bucket so shapes are distinct data
    x = torch.randn(1, seq, embed_dim, dtype=torch.float16)
    cos, sin = build_rope_tables(seq, head_dim, dtype=torch.float16)  # [S,D]
    mask = _causal_mask(seq)
    return x, cos, sin, mask


def _partition_report(edge_prog) -> tuple[list[str], list[str]]:
    """Return (delegated_ops, cpu_fallback_ops) for a device-free delegation check."""
    from executorch.backends.spyre.spyre_partitioner import _aten_name

    result = SpyrePartitioner().partition(edge_prog)
    nodes = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    delegated, fallback = [], []
    for n in nodes:
        (delegated if n.meta.get("delegation_tag") is not None else fallback).append(
            _aten_name(n) or str(n.target)
        )
    return delegated, fallback


def _fake_tensor_vals(node):
    """The FakeTensor value(s) a node produces (unwrapping list/tuple outputs)."""
    v = node.meta.get("val")
    if v is None:
        return []
    seq = v if isinstance(v, (list, tuple)) else [v]
    return [t for t in seq if hasattr(t, "shape") and hasattr(t, "dtype")]


def _fake_bytes(t):
    """Byte size of one FakeTensor, or None if any dim is symbolic (dynamic)."""
    n = 1
    try:
        for d in t.shape:
            n *= int(d)  # SymInt (dynamic) raises -> report None
    except Exception:
        return None
    return n * int(t.element_size())


def _partition_boundary_report(edge_prog) -> dict:
    """Device-free delegate-boundary host-copy budget for one forward.

    Runs only the partitioner (``.partition()``, no ``preprocess`` / device
    toolchain), then measures what crosses every delegate boundary -- exactly the
    H2D/D2H copies ``SpyreBackend::execute`` performs per partition.
    """
    from executorch.backends.spyre.spyre_partitioner import (
        _aten_name,
        _consumes_softmax,
    )

    result = SpyrePartitioner().partition(edge_prog)
    ep = result.tagged_exported_program
    graph = ep.graph_module.graph

    sig = getattr(ep, "graph_signature", None)
    param_names: set[str] = set()
    for attr in (
        "inputs_to_parameters",
        "inputs_to_buffers",
        "inputs_to_lifted_tensor_constants",
    ):
        m = getattr(sig, attr, None)
        if isinstance(m, dict):
            param_names |= set(m.keys())

    def _kind(node) -> str:
        if node.op != "placeholder":
            return "activation"
        return "param" if node.name in param_names else "user_input"

    partitions: "dict[str, list]" = {}
    fallback_nodes = []
    n_delegated = 0
    for n in graph.nodes:
        if n.op != "call_function":
            continue
        tag = n.meta.get("delegation_tag")
        if tag is None:
            fallback_nodes.append(n)
        else:
            partitions.setdefault(str(tag), []).append(n)
            n_delegated += 1

    parts = []
    total_h2d = 0
    total_d2h = 0
    for tag, pnodes in partitions.items():
        pset = set(pnodes)
        seen_in = set()
        in_bytes = 0
        split = {"param_bytes": 0, "user_input_bytes": 0, "activation_bytes": 0}
        n_in = 0
        for n in pnodes:
            for inp in n.all_input_nodes:
                if inp in pset or inp in seen_in:
                    continue
                seen_in.add(inp)
                n_in += 1
                for t in _fake_tensor_vals(inp):
                    b = _fake_bytes(t)
                    if b is None:
                        in_bytes = None
                    elif in_bytes is not None:
                        in_bytes += b
                    key = _kind(inp) + "_bytes"
                    if b is None:
                        split[key] = None
                    elif split[key] is not None:
                        split[key] += b
        n_out = 0
        out_bytes = 0
        for n in pnodes:
            if not any(u not in pset for u in n.users):
                continue
            n_out += 1
            for t in _fake_tensor_vals(n):
                b = _fake_bytes(t)
                if b is None:
                    out_bytes = None
                elif out_bytes is not None:
                    out_bytes += b
        if in_bytes is not None and total_h2d is not None:
            total_h2d += in_bytes
        else:
            total_h2d = None
        if out_bytes is not None and total_d2h is not None:
            total_d2h += out_bytes
        else:
            total_d2h = None
        parts.append(
            {
                "tag": tag,
                "n_in": n_in,
                "in_bytes": in_bytes,
                "in_param_bytes": split["param_bytes"],
                "in_user_input_bytes": split["user_input_bytes"],
                "in_activation_bytes": split["activation_bytes"],
                "n_out": n_out,
                "out_bytes": out_bytes,
            }
        )

    stages = {"embedding": 0, "context_mm": 0, "vocab_tail": 0}
    for n in fallback_nodes:
        name = _aten_name(n)
        if name == "aten::embedding":
            stages["embedding"] += 1
        elif name == "aten::mm" and _consumes_softmax(n):
            stages["context_mm"] += 1
        else:
            stages["vocab_tail"] += 1

    return {
        "n_partitions": len(partitions),
        "n_delegated_nodes": n_delegated,
        "n_fallback_nodes": len(fallback_nodes),
        "fallback_ops": sorted(
            {_aten_name(n) or str(n.target) for n in fallback_nodes}
        ),
        "fallback_stages": stages,
        "partitions": parts,
        "total_h2d_bytes": total_h2d,
        "total_d2h_bytes": total_d2h,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-kv-heads", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--buckets",
        default="",
        help=(
            "Comma-separated seq-len buckets, e.g. '8,16,32'. When set, the seq "
            "axis of x/cos/sin/mask is exported dynamic and preprocess emits one "
            "JobPlan per bucket (v3). Empty => static single-shape .pte."
        ),
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="Device-free: trace + partition, print delegated/fallback ops, exit.",
    )
    ap.add_argument(
        "--emit",
        default="",
        choices=["", "qh", "ks", "vs", "raw", "scores", "probs", "ah", "yh"],
        help=(
            "8c bisection: truncate the HEAD-0 SDPA chain at this link and emit the "
            "2D intermediate as the .pte output (single static seq; forbids --buckets). "
            "Dump it on device (EXECUTORCH_SPYRE_DUMP_OUT) and compare vs ref to localize the "
            "diverging link. Empty => full o_proj-summed attention output."
        ),
    )
    args = ap.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]
    if args.emit and buckets:
        raise SystemExit("--emit is single-shape only; drop --buckets")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    head_dim = args.embed_dim // args.n_heads
    mod = AttentionBlock(args.embed_dim, args.n_heads, args.n_kv_heads).eval().half()
    if args.emit:
        mod._emit = args.emit  # trace the truncated head-0 chain

    seqs = sorted(set(buckets)) if buckets else [args.seq]
    trace_seq = max(seqs)
    tx, tcos, tsin, tmask = _make_case_inputs(
        args.embed_dim, head_dim, trace_seq, args.seed
    )

    if buckets:
        seq_dim = torch.export.Dim("seq", min=min(buckets), max=max(buckets))
        dynamic_shapes = {
            "x": {1: seq_dim},
            "cos": {0: seq_dim},
            "sin": {0: seq_dim},
            "mask": {0: seq_dim, 1: seq_dim},
        }
        ep = torch.export.export(
            mod, (tx, tcos, tsin, tmask), dynamic_shapes=dynamic_shapes
        )
        spec_value = json.dumps(
            {
                "arg_dims": [
                    {"arg": 0, "axis": 1},
                    {"arg": 1, "axis": 0},
                    {"arg": 2, "axis": 0},
                    {"arg": 3, "axis": 0},
                    {"arg": 3, "axis": 1},
                ],
                "buckets": buckets,
            }
        ).encode("utf-8")
        compile_specs = [CompileSpec(_BUCKET_COMPILE_SPEC_KEY, spec_value)]
    else:
        ep = torch.export.export(mod, (tx, tcos, tsin, tmask))
        compile_specs = None

    edge = to_edge(ep)

    if args.inspect:
        delegated, fallback = _partition_report(edge.exported_program())
        print(f"delegated graph:\n{edge.exported_program().graph}")
        print(f"\ndelegated ops ({len(delegated)}): {sorted(set(delegated))}")
        print(f"CPU-fallback ops ({len(fallback)}): {sorted(set(fallback))}")
        if fallback:
            print("RESULT: FALLBACK -- graph does not fully delegate")
        else:
            print("RESULT: FULLY DELEGATED")
        return

    edge = edge.to_backend(SpyrePartitioner(compile_specs))
    et = edge.to_executorch()

    pte_path = os.path.join(args.out_dir, "spyre_attention.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    manifest = ["pte spyre_attention.pte"]
    meta_cases = []
    for seq in seqs:
        x, cos, sin, mask = _make_case_inputs(args.embed_dim, head_dim, seq, args.seed)
        with torch.no_grad():
            ref = (
                _ref_emit(mod, x, cos, sin, mask, args.emit)
                if args.emit
                else _ref_forward(mod, x, cos, sin, mask)
            )
        tag = f"s{seq}"
        files = {
            "x": f"x_{tag}.bin",
            "cos": f"cos_{tag}.bin",
            "sin": f"sin_{tag}.bin",
            "mask": f"mask_{tag}.bin",
            "ref": f"ref_{tag}.bin",
        }
        _dump_fp16(os.path.join(args.out_dir, files["x"]), x)
        _dump_fp16(os.path.join(args.out_dir, files["cos"]), cos)
        _dump_fp16(os.path.join(args.out_dir, files["sin"]), sin)
        _dump_fp16(os.path.join(args.out_dir, files["mask"]), mask)
        _dump_fp16(os.path.join(args.out_dir, files["ref"]), ref)
        if len(seqs) > 1:
            manifest.append(f"case {tag}")
        manifest.append(f"input {files['x']} fp16 3 1 {seq} {args.embed_dim}")
        manifest.append(f"input {files['cos']} fp16 2 {seq} {head_dim}")
        manifest.append(f"input {files['sin']} fp16 2 {seq} {head_dim}")
        manifest.append(f"input {files['mask']} fp16 2 {seq} {seq}")
        if args.emit:
            odims = " ".join(str(dd) for dd in ref.shape)
            manifest.append(f"output {files['ref']} fp16 {ref.dim()} {odims}")
        else:
            manifest.append(f"output {files['ref']} fp16 3 1 {seq} {args.embed_dim}")
        meta_cases.append({"seq": seq, **files})

    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    meta = {
        "pte": "spyre_attention.pte",
        "model": "GQA attention block (qkv proj -> RoPE -> sdpa_math -> o proj)",
        "dtype": "float16",
        "embed_dim": args.embed_dim,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": head_dim,
        "inputs_order": ["x", "cos", "sin", "mask"],
        "cases": meta_cases,
    }
    if buckets:
        meta["dynamic"] = {"axis": "seq", "buckets": buckets}
    with open(os.path.join(args.out_dir, "attention_inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes)")
    print(f"seqs: {seqs}  buckets: {buckets or 'static'}")
    delegated, fallback = _partition_report(to_edge(ep).exported_program())
    print(f"delegated ops: {sorted(set(delegated))}")
    print(f"CPU-fallback ops: {sorted(set(fallback)) or 'none'}")


if __name__ == "__main__":
    main()
