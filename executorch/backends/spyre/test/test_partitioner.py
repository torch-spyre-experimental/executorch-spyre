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
"""Unit tests for the capability-based ``SpyrePartitioner``.

Device-free: they need only ``torch`` + ``executorch`` (the AOT export/edge
stack), no ``torch_spyre`` and no Spyre device. They exercise (a) the op-name
allowlist plumbing (``_aten_name`` unwrap + ``_SUPPORTED_ATEN_OPS`` membership),
(b) full delegation of an all-supported MLP, and (c) CPU fallback for an
unsupported op (``aten::sin``) in a mixed graph — the "no silent coverage gaps"
guarantee.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402

from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    _SUPPORTED_ATEN_OPS,
    _aten_name,
    _torch_op_name,
    SpyrePartitioner,
)

# ``MixedUnsupported`` (``relu(sin(fc(x)))``) is defined in export_mixed as the
# single source of truth -- the same module the mixed-workload .pte and device
# run use -- so the fallback op set asserted here and run on device cannot drift.
from executorch.backends.spyre.export_mixed import MixedUnsupported  # noqa: E402


# --- Modules under test ----------------------------------------------------


class MLP(torch.nn.Module):
    """Every op (addmm / permute_copy / relu) is on the Spyre allowlist."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class RMSNormMath(torch.nn.Module):
    """RMSNorm authored as explicit math (the Phase-8 8b workaround).

    The fused ``aten::rms_norm`` op compiles and runs on-device but is
    numerically outside the fp16 envelope (matches=FAIL, 8a follow-up), so it
    stays off the allowlist. Its decomposition — ``x * rsqrt(mean(x**2) + eps)
    * w`` — is entirely ops the 8a sweep validated (mul / pow / mean / add /
    rsqrt). So an RMSNorm written this way must be *fully* delegated with no CPU
    fallback.

    ``eps`` is held as an **fp16 buffer**, not a Python float: a float literal is
    lifted by ``torch.export`` as an fp32 constant, and ``to_edge`` then inserts a
    ``dim_order_ops::_to_dim_order_copy`` fp32->fp16 cast to feed the fp16 add —
    an op in the ``_to_copy`` family that fails the device match (matches=FAIL),
    so it would not delegate. Keeping every constant in fp16 keeps the whole
    graph on-device. This is a real authoring rule for the decoder layer (8d).
    """

    def __init__(self, dim: int = 64, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float16))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float16))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


# --- Helpers ---------------------------------------------------------------


def _edge_program(mod: torch.nn.Module):
    """Export ``mod`` (fp16, static ``[1, 64]``) and lower to the edge dialect."""
    mod = mod.eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    return to_edge(exported).exported_program()


def _tag_counts(exported_program):
    """Return ``(tagged, compute)`` node counts for call_function nodes."""
    compute = [
        n for n in exported_program.graph_module.graph.nodes if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    return tagged, compute


# --- _aten_name / allowlist plumbing ---------------------------------------


def test_supported_ops_include_validated_core():
    # The core ops confirmed end-to-end on a Spyre device (Phase 2/3) must
    # remain in the allowlist — regressing any of these breaks a shipped path.
    for op in ("aten::mm", "aten::addmm", "aten::relu", "aten::permute_copy"):
        assert op in _SUPPORTED_ATEN_OPS


def test_supported_ops_include_phase8_promotions():
    # 8a device sweep (2026-07-27) validated these on-device (runs+matches
    # within the fp16 envelope). They back the RMSNorm-math / gated-MLP /
    # attention-softmax paths the decoder layer needs.
    for op in (
        "aten::add",
        "aten::sub",
        "aten::mul",
        "aten::div",
        "aten::neg",
        "aten::pow",
        "aten::rsqrt",
        "aten::mean",
        "aten::silu",
        "aten::gelu",
        "aten::sigmoid",
        "aten::view_copy",
        "aten::expand_copy",
        "aten::transpose_copy",
        "aten::unsqueeze_copy",
        "aten::_reshape_alias_copy",
        "aten::clone",
        "aten::clamp",
        "aten::_softmax",
        # 8a follow-up: bmm/matmul were runs=FAIL until the single-kernel
        # workspace tensor was routed through the v2 payload (a delegate-side
        # layout bug, not an SDSC limitation). They back attention (8c).
        "aten::bmm",
        "aten::matmul",
    ):
        assert op in _SUPPORTED_ATEN_OPS, f"{op} should be validated after 8a"


def test_ops_not_yet_device_validated_stay_off_allowlist():
    # Contract: an op flips to the allowlist ONLY after a green on-device run.
    # These either fail the device run/match or are compile-blocked (8b queue),
    # so the partitioner must still leave them on CPU.
    # NOTE: aten::scaled_dot_product_attention graduated to validated=true in
    # phase 9 and was removed from this list.
    for op in (
        "aten::rms_norm",  # runs=PASS matches=FAIL (8a follow-up) — use RMSNorm-math instead
        "aten::cat",  # innermost-dim cat scrambles (8a follow-up) — seq-dim ok, but off allowlist so RoPE dim=-1 cat can't silently delegate
        "aten::_to_copy",  # matches=FAIL max_abs_err 4.13 (8a follow-up) — avoided via fp16 constants
        "aten::slice_copy",  # no standalone SDSC kernel (8a follow-up) — validate in-context 8c/8d
        "aten::embedding",  # compile-blocked (8b)
        "aten::index_copy",  # compile-blocked (8b)
    ):
        assert op not in _SUPPORTED_ATEN_OPS, f"{op} is not device-validated yet"


def test_aten_name_unwraps_edge_overload():
    # Every delegated MLP node's edge overload must unwrap to a name in the set.
    edge = _edge_program(MLP())
    seen = set()
    for node in edge.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = _aten_name(node)
        assert name is not None, f"could not recover aten name for {node.target}"
        assert name.startswith("aten::")
        seen.add(name)
    # MLP is addmm (x2) + permute_copy (x2) + relu — all in the allowlist.
    assert seen <= _SUPPORTED_ATEN_OPS
    assert {"aten::addmm", "aten::permute_copy", "aten::relu"} <= seen


def test_aten_name_strips_typed_overload():
    # Regression: typed overloads (add.Tensor, pow.Tensor_Scalar, mean.dim, ...)
    # must normalise to their base name so they match the base-name allowlist.
    # Before this, only .default-overload ops matched and every elementwise /
    # reduction op silently fell back to CPU despite being device-validated.
    edge = _edge_program(RMSNormMath())
    names = {
        _aten_name(n) for n in edge.graph_module.graph.nodes if n.op == "call_function"
    }
    # None of the recovered names may carry an overload suffix.
    for name in names:
        if name is None:
            continue
        _, opname = name.split("::", 1)
        assert "." not in opname, f"{name} still carries an overload suffix"
    # The typed-overload ops in RMSNorm-math resolve to their validated base name.
    assert {"aten::add", "aten::mul", "aten::pow", "aten::mean"} <= names


def test_aten_name_none_for_placeholder():
    edge = _edge_program(MLP())
    placeholder = next(
        n for n in edge.graph_module.graph.nodes if n.op == "placeholder"
    )
    assert _aten_name(placeholder) is None


def test_torch_op_name_from_meta():
    class DummyNode:
        def __init__(self, meta):
            self.meta = meta

    # 1. torch_fn tuple format
    assert (
        _torch_op_name(DummyNode({"torch_fn": ("embedding_1", "function.embedding")}))
        == "torch.embedding"
    )
    assert (
        _torch_op_name(
            DummyNode(
                {"torch_fn": ("index_copy_1", "builtin_function_or_method.index_copy")}
            )
        )
        == "torch.index_copy"
    )
    assert (
        _torch_op_name(
            DummyNode({"torch_fn": ("softmax_1", "builtin_function_or_method.softmax")})
        )
        == "torch.softmax"
    )
    assert (
        _torch_op_name(
            DummyNode({"torch_fn": ("mm_1", "builtin_function_or_method.mm")})
        )
        == "torch.mm"
    )

    # 2. torch_fn single-item tuple / string / callable
    assert (
        _torch_op_name(DummyNode({"torch_fn": ("embedding_1",)})) == "torch.embedding"
    )
    assert (
        _torch_op_name(DummyNode({"torch_fn": "torch.embedding"})) == "torch.embedding"
    )
    assert _torch_op_name(DummyNode({"torch_fn": "embedding"})) == "torch.embedding"
    assert _torch_op_name(DummyNode({"torch_fn": torch.sin})) == "torch.sin"

    # 3. Fallback to original_aten
    assert (
        _torch_op_name(DummyNode({"original_aten": "aten::index_copy"}))
        == "torch.index_copy"
    )
    assert (
        _torch_op_name(DummyNode({"original_aten": "aten._softmax.default"}))
        == "torch.softmax"
    )
    assert (
        _torch_op_name(DummyNode({"original_aten": "aten.embedding.default"}))
        == "torch.embedding"
    )

    # 4. Graceful handling of missing / empty metadata
    assert _torch_op_name(DummyNode({})) is None
    assert _torch_op_name(DummyNode(None)) is None
    assert _torch_op_name(object()) is None
    assert _torch_op_name(None) is None


def test_torch_op_name_on_exported_nodes():
    class AttributionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 8)

        def forward(self, x, y, idx, val):
            a = self.emb(x)
            b = torch.index_copy(a, 0, idx, val)
            c = torch.softmax(b, dim=-1)
            d = torch.mm(c, y)
            return d

    mod = AttributionModule().eval()
    x = torch.tensor([0, 1], dtype=torch.long)
    y = torch.randn(8, 8)
    idx = torch.tensor([0, 1], dtype=torch.long)
    val = torch.randn(2, 8)

    edge = to_edge(torch.export.export(mod, (x, y, idx, val))).exported_program()
    roots = {
        _torch_op_name(n)
        for n in edge.graph_module.graph.nodes
        if n.op == "call_function"
    }
    assert "torch.embedding" in roots
    assert "torch.index_copy" in roots
    assert "torch.softmax" in roots
    assert "torch.mm" in roots


# --- partition() behaviour -------------------------------------------------


def test_mlp_fully_delegated():
    edge = _edge_program(MLP())
    result = SpyrePartitioner().partition(edge)

    assert len(result.partition_tags) >= 1
    tagged, compute = _tag_counts(result.tagged_exported_program)
    assert len(compute) > 0
    # All-supported graph => every compute node delegated, none on CPU.
    assert len(tagged) == len(compute)


def test_mixed_graph_falls_back_on_unsupported_op():
    edge = _edge_program(MixedUnsupported())
    result = SpyrePartitioner(mode="allowlist").partition(edge)

    tagged, compute = _tag_counts(result.tagged_exported_program)
    tagged_names = {_aten_name(n) for n in tagged}
    fallback_names = {_aten_name(n) for n in compute if n not in tagged}

    # sin is not allowlisted -> stays on CPU; supported neighbours are delegated.
    assert "aten::sin" not in tagged_names
    assert any("sin" in (name or "") for name in fallback_names)
    assert len(tagged) > 0
    assert len(tagged) < len(compute)


def test_rmsnorm_math_fully_delegated():
    # RMSNorm-as-math (8b workaround for the unrunnable fused rms_norm) is built
    # only from 8a-validated ops, so it must delegate with zero CPU fallback.
    edge = _edge_program(RMSNormMath())
    result = SpyrePartitioner().partition(edge)

    tagged, compute = _tag_counts(result.tagged_exported_program)
    assert len(compute) > 0
    assert len(tagged) == len(compute), (
        "RMSNorm-math must fully delegate; CPU fallback for: "
        f"{ {_aten_name(n) for n in compute if n not in tagged} }"
    )


def test_partition_tags_map_to_spyre_backend():
    from executorch.backends.spyre.spyre_partitioner import SPYRE_BACKEND_ID

    edge = _edge_program(MLP())
    result = SpyrePartitioner().partition(edge)
    for spec in result.partition_tags.values():
        assert spec.backend_id == SPYRE_BACKEND_ID
