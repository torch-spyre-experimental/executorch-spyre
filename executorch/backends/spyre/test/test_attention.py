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
"""Unit tests for the device-safe attention primitives (Phase 8, 8b).

Device-free. Guarantees:

  1. **Numerics** -- ``sdpa_math`` matches a reference fp32 SDPA (MHA and GQA,
     with and without a causal mask) within the fp16 envelope.
  2. **Delegation** -- traced to the edge dialect, the decomposition contains no
     fused ``scaled_dot_product_attention``, no ``cat``, and no 4D ``matmul``;
     every op is on the validated allowlist, so it fully delegates.
  3. ``kv_append`` refuses the innermost/stick dim and concats the sequence dim.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402

from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    _SUPPORTED_ATEN_OPS,
    _aten_name,
    _consumes_softmax,
    SpyrePartitioner,
)
from executorch.backends.spyre.attention import kv_append, sdpa_math  # noqa: E402
from executorch.backends.spyre.export_attention import (  # noqa: E402
    AttentionBlock,
    _make_case_inputs,
    _ref_forward,
)

ATOL, RTOL = 1e-1, 2e-2


def _sdpa_ref(q, k, v, mask=None, scale=None):
    """Reference SDPA in fp32 with explicit GQA expansion."""
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    qf, kf, vf = q.float(), k.float(), v.float()
    if hq != hkv:
        rep = hq // hkv
        kf = kf[:, :, None].expand(b, hkv, rep, sk, d).reshape(b, hq, sk, d)
        vf = vf[:, :, None].expand(b, hkv, rep, sk, d).reshape(b, hq, sk, d)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores + mask.float()
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).to(q.dtype)


# --- Numerics ---------------------------------------------------------------


def test_sdpa_math_matches_reference_mha():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    k = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    v = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    got = sdpa_math(q, k, v)
    want = _sdpa_ref(q, k, v)
    assert got.shape == (1, 4, 8, 64)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL)


def test_sdpa_math_matches_reference_gqa():
    # 8 query heads share 2 KV heads (GQA rep=4).
    torch.manual_seed(1)
    q = torch.randn(1, 8, 8, 64, dtype=torch.float16)
    k = torch.randn(1, 2, 8, 64, dtype=torch.float16)
    v = torch.randn(1, 2, 8, 64, dtype=torch.float16)
    got = sdpa_math(q, k, v)
    want = _sdpa_ref(q, k, v)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL)


def test_sdpa_math_matches_reference_causal_mask():
    torch.manual_seed(2)
    s, d = 8, 64
    q = torch.randn(1, 4, s, d, dtype=torch.float16)
    k = torch.randn(1, 4, s, d, dtype=torch.float16)
    v = torch.randn(1, 4, s, d, dtype=torch.float16)
    mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1).to(torch.float16)
    got = sdpa_math(q, k, v, mask=mask)
    want = _sdpa_ref(q, k, v, mask=mask)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL)


def test_sdpa_math_rejects_bad_gqa_ratio():
    q = torch.randn(1, 6, 4, 8)
    k = torch.randn(1, 4, 4, 8)
    v = torch.randn(1, 4, 4, 8)
    with pytest.raises(ValueError):
        sdpa_math(q, k, v)


# --- Delegation -------------------------------------------------------------


class _SDPAModule(torch.nn.Module):
    def __init__(self, hq=4, hkv=4, s=8, d=64, causal=False) -> None:
        super().__init__()
        self.hkv = hkv
        # scale rides in as an fp16 registered buffer, NOT a Python float: a
        # float lifts an fp32 constant + a _to_dim_order_copy cast, and a
        # scalar tensor built inside forward trips a _clone_dim_order contiguity
        # clone -- both fall off the allowlist. A buffer is a clean lifted
        # constant (this is how export_attention.py / 8c must pass scale too).
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(d), dtype=torch.float16)
        )
        if causal:
            mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
            self.register_buffer("mask", mask.to(torch.float16))
        else:
            self.mask = None

    def forward(self, q, k, v):
        return sdpa_math(q, k, v, mask=self.mask, scale=self.scale)


def _sdpa_edge(hq=4, hkv=4, s=8, d=64, causal=False):
    mod = _SDPAModule(hq, hkv, s, d, causal).eval()
    q = torch.randn(1, hq, s, d, dtype=torch.float16)
    k = torch.randn(1, hkv, s, d, dtype=torch.float16)
    v = torch.randn(1, hkv, s, d, dtype=torch.float16)
    exported = torch.export.export(mod, (q, k, v))
    return to_edge(exported).exported_program()


def _op_names(edge):
    return {
        _aten_name(n) for n in edge.graph_module.graph.nodes if n.op == "call_function"
    }


def test_sdpa_math_has_no_blocked_ops():
    names = _op_names(_sdpa_edge(causal=True))
    assert "aten::scaled_dot_product_attention" not in names
    assert "aten::cat" not in names
    # 4D matmul is compile-blocked; the decomposition must use bmm instead.
    assert "aten::matmul" not in names
    assert "aten::bmm" in names, f"expected bmm in {sorted(n for n in names if n)}"


def test_sdpa_math_fully_delegates():
    edge = _sdpa_edge(causal=True)
    result = SpyrePartitioner().partition(edge)
    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    assert len(compute) > 0
    fallback = {_aten_name(n) for n in compute if n not in tagged}
    assert len(tagged) == len(compute), f"SDPA CPU fallback for: {fallback}"
    used = {_aten_name(n) for n in compute}
    assert used <= _SUPPORTED_ATEN_OPS, f"off allowlist: {used - _SUPPORTED_ATEN_OPS}"


def test_sdpa_math_gqa_delegates():
    # GQA regroups query heads under their shared KV head via contiguous
    # reshapes -- NOT a repeat_kv expand_copy (which forces a _clone_dim_order
    # contiguity copy off the allowlist). So the GQA graph must fully delegate
    # with no expand_copy and no clone.
    edge = _sdpa_edge(hq=8, hkv=2)
    result = SpyrePartitioner().partition(edge)
    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    assert len(compute) > 0
    fallback = {_aten_name(n) for n in compute if n not in tagged}
    assert len(tagged) == len(compute), f"GQA SDPA CPU fallback for: {fallback}"
    used = {_aten_name(n) for n in compute}
    assert "aten::expand_copy" not in used, "GQA regroup must not use expand_copy"
    assert used <= _SUPPORTED_ATEN_OPS, f"off allowlist: {used - _SUPPORTED_ATEN_OPS}"


# --- AttentionBlock (8c export module) --------------------------------------


def _block_edge(embed_dim=256, hq=4, hkv=2, seq=16):
    mod = AttentionBlock(embed_dim, hq, hkv).eval().half()
    x, cos, sin, mask = _make_case_inputs(embed_dim, embed_dim // hq, seq, seed=0)
    exported = torch.export.export(mod, (x, cos, sin, mask))
    return to_edge(exported).exported_program()


def test_attention_block_matches_reference():
    embed_dim, hq, hkv, seq = 256, 4, 2, 16
    mod = AttentionBlock(embed_dim, hq, hkv).eval().half()
    x, cos, sin, mask = _make_case_inputs(embed_dim, embed_dim // hq, seq, seed=0)
    with torch.no_grad():
        got = mod(x, cos, sin, mask)
        want = _ref_forward(mod, x, cos, sin, mask)
    assert got.shape == (1, seq, embed_dim)
    assert torch.allclose(got.float(), want.float(), atol=ATOL, rtol=RTOL)


def test_attention_block_has_no_blocked_ops():
    names = _op_names(_block_edge())
    for blocked in (
        "aten::scaled_dot_product_attention",
        "aten::cat",
        "aten::matmul",  # 4D matmul compile-blocked; decomposition uses bmm
        "aten::_to_copy",
        "aten::rms_norm",
    ):
        assert blocked not in names, f"blocked op {blocked} in graph"
    # dim_order layout ops are off the allowlist -> would force CPU fallback.
    assert not any(
        "dim_order" in (n or "") for n in names
    ), f"unexpected dim_order op in {sorted(n for n in names if n)}"


def _assert_only_context_mm_falls_back(edge, n_heads):
    # Single-tile block (embed == 64, one 64-wide output stick -> no scramble):
    # everything delegates EXCEPT the per-head softmax-fed context mm (probs @ v),
    # which the partitioner refuses. So the
    # only CPU fallback is exactly `n_heads` softmax-fed mms; everything else is
    # delegated and on the allowlist.
    result = SpyrePartitioner().partition(edge)
    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    fallback = [n for n in compute if n not in tagged]
    assert len(compute) > 0
    fb_names = {_aten_name(n) for n in fallback}
    assert fb_names == {"aten::mm"}, f"unexpected CPU fallback: {fb_names}"
    assert all(_consumes_softmax(n) for n in fallback), "non-softmax-fed mm fell back"
    assert (
        len(fallback) == n_heads
    ), f"expected {n_heads} context mms, got {len(fallback)}"
    used = {_aten_name(n) for n in tagged}
    assert used <= _SUPPORTED_ATEN_OPS, f"off allowlist: {used - _SUPPORTED_ATEN_OPS}"


def test_attention_block_fully_delegates():
    # Single-tile GQA (hkv < hq, embed == 64): only the per-head context mms
    # fall back; wider embeds are the documented multi-tile output scramble (see
    # test_attention_block_multitile_output_not_fully_delegated).
    _assert_only_context_mm_falls_back(
        _block_edge(embed_dim=64, hq=4, hkv=2), n_heads=4
    )


def test_attention_block_mha_fully_delegates():
    # Single-tile plain MHA (hkv == hq, embed == 64): same clean fallback.
    _assert_only_context_mm_falls_back(
        _block_edge(embed_dim=64, hq=4, hkv=4), n_heads=4
    )


def test_attention_block_multitile_output_not_fully_delegated():
    # A multi-tile block (embed == 256 => the [1,S,256] output spans 4 sticks) is
    # the documented output-store scramble: a >1-stick delegate output writes back
    # tile-major-scrambled. The scramble guard must refuse the terminal wide
    # projection, so the block is NOT fully delegated. (Single-tile is the only
    # supported config -- see the two *_fully_delegates tests; the node-level
    # guard is correct-and-complete for a single-tile backbone with a wide
    # terminal projection, e.g. tiny Granite's vocab head, not for a fully-wide
    # backbone. This only asserts the guard fires, not that multi-tile is usable.)
    edge = _block_edge(embed_dim=256, hq=4, hkv=4)
    result = SpyrePartitioner().partition(edge)
    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    assert len(tagged) < len(compute), "multi-tile wide output must not fully delegate"


# --- kv_append --------------------------------------------------------------


def test_kv_append_concats_sequence_dim():
    past = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    new = torch.randn(1, 4, 1, 64, dtype=torch.float16)
    out = kv_append(past, new, dim=2)
    assert out.shape == (1, 4, 9, 64)
    assert torch.equal(out[:, :, :8], past)
    assert torch.equal(out[:, :, 8:], new)


def test_kv_append_rejects_innermost_dim():
    past = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    new = torch.randn(1, 4, 8, 64, dtype=torch.float16)
    with pytest.raises(ValueError):
        kv_append(past, new, dim=-1)
    with pytest.raises(ValueError):
        kv_append(past, new, dim=3)
