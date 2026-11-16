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
"""Unit tests for the device-safe RoPE helper (Phase 8, 8b).

Device-free: pure ``torch`` + the AOT edge stack. Two guarantees:

  1. **Numerics** -- ``apply_rope`` (matmul-based ``rotate_half``) equals the
     textbook ``cat``-based HF RoPE on CPU, within the fp16 envelope.
  2. **Delegation** -- traced into the edge dialect, RoPE contains **no**
     ``aten::cat`` (the innermost-dim cat that scrambles on device) and every
     op it does contain is on the validated allowlist, so it fully delegates.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402

from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    _SUPPORTED_ATEN_OPS,
    _aten_name,
    SpyrePartitioner,
)
from executorch.backends.spyre.rope import (  # noqa: E402
    apply_rope,
    build_rope_tables,
    rotation_matrix,
)

# fp16 envelope (project-wide).
ATOL, RTOL = 1e-1, 2e-2


# --- Reference: textbook HF/Llama RoPE (the cat-based rotate_half) ----------


def _rotate_half_ref(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_ref(x, cos, sin):
    return x * cos + _rotate_half_ref(x) * sin


# --- Numerics ---------------------------------------------------------------


def test_rotation_matrix_equals_rotate_half():
    # x @ R must reproduce rotate_half(x) exactly (it is the same linear map).
    torch.manual_seed(0)
    d = 64
    x = torch.randn(2, 3, 5, d)
    rot = rotation_matrix(d, dtype=torch.float32)
    got = x.reshape(-1, d) @ rot
    want = _rotate_half_ref(x).reshape(-1, d)
    assert torch.equal(got, want)


def test_rotation_matrix_rejects_odd_dim():
    with pytest.raises(ValueError):
        rotation_matrix(63)


def test_apply_rope_matches_reference_fp16():
    torch.manual_seed(0)
    b, h, s, d = 1, 4, 8, 64
    x = torch.randn(b, h, s, d, dtype=torch.float16)
    cos, sin = build_rope_tables(s, d, dtype=torch.float16)
    rot = rotation_matrix(d, dtype=torch.float16)

    got = apply_rope(x, cos, sin, rot)
    want = _apply_rope_ref(x, cos, sin)  # cat-based reference, same fp16 inputs

    assert got.shape == (b, h, s, d)
    assert torch.allclose(got, want, atol=ATOL, rtol=RTOL)


def test_build_rope_tables_shape_and_dtype():
    cos, sin = build_rope_tables(seq_len=16, head_dim=64)
    assert cos.shape == (16, 64) and sin.shape == (16, 64)
    assert cos.dtype == torch.float16 and sin.dtype == torch.float16
    # Non-interleaved layout: the two halves share frequencies -> cos[:, :32]
    # equals cos[:, 32:].
    assert torch.equal(cos[:, :32], cos[:, 32:])


# --- Delegation: no innermost cat, fully on the allowlist -------------------


class _RoPEModule(torch.nn.Module):
    """Wraps ``apply_rope`` with ``cos``/``sin``/``R`` as fp16 buffers.

    Holding the tables as buffers (not Python floats) keeps the whole graph
    fp16, so ``to_edge`` inserts no ``_to_dim_order_copy`` cast (which fails the
    device match -- see the RMSNormMath authoring rule).
    """

    def __init__(self, seq_len: int, head_dim: int) -> None:
        super().__init__()
        cos, sin = build_rope_tables(seq_len, head_dim, dtype=torch.float16)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("rot", rotation_matrix(head_dim, dtype=torch.float16))

    def forward(self, x):
        return apply_rope(x, self.cos, self.sin, self.rot)


def _rope_edge(seq_len=8, head_dim=64, heads=4):
    mod = _RoPEModule(seq_len, head_dim).eval()
    x = torch.randn(1, heads, seq_len, head_dim, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    return to_edge(exported).exported_program()


def test_rope_has_no_innermost_cat():
    # The whole point of the matmul rotate_half: the device graph must contain
    # no aten::cat (which scrambles along the stick dim).
    edge = _rope_edge()
    names = {
        _aten_name(n) for n in edge.graph_module.graph.nodes if n.op == "call_function"
    }
    assert (
        "aten::cat" not in names
    ), f"RoPE leaked a cat: {sorted(n for n in names if n)}"


def test_rope_fully_delegates():
    # Every compute op RoPE lowers to must be validated, so it delegates with
    # zero CPU fallback.
    edge = _rope_edge()
    result = SpyrePartitioner().partition(edge)

    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    assert len(compute) > 0
    fallback = {_aten_name(n) for n in compute if n not in tagged}
    assert len(tagged) == len(compute), f"RoPE CPU fallback for: {fallback}"

    # And each op RoPE uses is genuinely on the allowlist.
    used = {_aten_name(n) for n in compute}
    assert (
        used <= _SUPPORTED_ATEN_OPS
    ), f"unexpected ops off allowlist: {used - _SUPPORTED_ATEN_OPS}"
