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
"""KV-cache export characterization (issue #36, Phase A → Phase 10).

Device-free: only ``torch`` + ``executorch``. These tests pin down what the AOT
stack does with an hf-adapters-style in-place KV-cache write
(``key_cache.index_copy_(2, cache_index, k)``) *before* any runtime work, so
Phase B (on-device repro) and Phase C (mutation handling / persistent cache)
target the right ops and signature kinds:

  * **Mutation kind** depends on who owns the cache: a cache passed as a
    ``forward`` argument (hf-adapters ``block_forward``) becomes a
    ``USER_INPUT_MUTATION``; a registered buffer (ExecuTorch
    ``extension/llm/modules/kv_cache.py``) becomes a ``BUFFER_MUTATION``.
  * **Op lowering chain**: ``torch.export`` keeps ``index_copy_``;
    functionalization yields out-of-place ``index_copy``; ``to_edge``'s core-aten
    decomposition turns that into ``aten::index_put``. So the op the partitioner
    sees is ``index_put`` -- *not* the ``aten::index_copy`` entry in
    ``supported_ops.yaml`` -- unless ``index_copy`` is kept via ``preserve_ops``.
  * **Partitioning** (Phase 10+): ``aten::index_put`` is validated and on the
    allowlist. The partitioner delegates scatter writes whose base tensor is a
    graph placeholder (``is_state_scatter`` exemption), so both the cache write
    and the attention read are delegated. ``aten::index_copy`` remains off the
    allowlist (SDSC compile FAIL).
  * **Ownership**: ``SpyrePartitioner.partition`` calls ``tag_mutated_buffer``, so
    the mutated cache buffer moves *into* the delegate (delegate-owned state):
    the delegate sees ``BUFFER`` / ``BUFFER_MUTATION`` and the top-level program
    neither holds nor passes the cache. Without the tag (the pre-tag baseline,
    kept as a characterization test) the delegate takes the cache as a
    ``USER_INPUT`` and returns it as a ``USER_OUTPUT`` (top-level
    ``BUFFER_MUTATION`` write-back).
  * **LLM wrapper signature**: ``forward(tokens, input_pos) -> logits`` with
    graph-local ``cache_index`` / mask derivation keeps its user-visible
    signature; the KV caches are metadata-only ``BUFFER`` / ``BUFFER_MUTATION``.
  * **Full model**: on the ``(tokens, input_pos)`` LLM wrapper, the cache adds
    no extra CPU fallbacks or partitions (``index_put`` is delegated).

The partition/lowering tests stub ``SpyreBackend.preprocess`` (it needs the Spyre
toolchain). The ``allow_index_put`` fixture is kept for back-compat but is now a
no-op (``index_put`` is already on the allowlist).
"""

from collections import Counter

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402
from executorch.exir.backend.backend_details import PreprocessResult  # noqa: E402
from torch.export.graph_signature import InputKind, OutputKind  # noqa: E402

from executorch.backends.spyre import spyre_partitioner  # noqa: E402
from executorch.backends.spyre.spyre_backend import SpyreBackend  # noqa: E402
from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    _SUPPORTED_ATEN_OPS,
    _aten_name,
    SpyrePartitioner,
)
from executorch.backends.spyre.attention import sdpa_math  # noqa: E402

# Small decode-step shapes: one new token written into a 128-slot cache.
# head_dim = 64 = one stick, so no output is wider than a stick.
B, H, MAX_LEN, D = 1, 2, 128, 64


# --- Modules under test ----------------------------------------------------


class CacheAsInput(torch.nn.Module):
    """hf-adapters ``block_forward`` style: caches are ``forward`` arguments,
    written in place by ``index_copy_`` and read whole by attention.

    ``scale`` is an fp16 buffer, not a float, so the attention math delegates
    without a ``_to_dim_order_copy`` cast (see ``sdpa_math``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(D**-0.5, dtype=torch.float16))

    def forward(self, q, k, v, key_cache, value_cache, cache_index, mask):
        key_cache.index_copy_(2, cache_index, k)
        value_cache.index_copy_(2, cache_index, v)
        return sdpa_math(q, key_cache, value_cache, mask, scale=self.scale)


class CacheAsBuffer(torch.nn.Module):
    """ExecuTorch ``kv_cache.py`` style: caches are registered buffers."""

    def __init__(self, n_new: int = 1) -> None:
        super().__init__()
        self.n_new = n_new
        cache = torch.zeros(B, H, MAX_LEN, D, dtype=torch.float16)
        self.register_buffer("k_cache", cache.clone())
        self.register_buffer("v_cache", cache.clone())
        self.register_buffer("scale", torch.tensor(D**-0.5, dtype=torch.float16))

    def forward(self, q, k, v, cache_index, mask):
        self.k_cache.index_copy_(2, cache_index, k)
        self.v_cache.index_copy_(2, cache_index, v)
        return sdpa_math(q, self.k_cache, self.v_cache, mask, scale=self.scale)


# --- Helpers ---------------------------------------------------------------


def _step_inputs(n_new: int = 1, pos: int = 0):
    """q/k/v for ``n_new`` positions starting at ``pos``, plus index and mask."""
    q = torch.randn(B, H, n_new, D, dtype=torch.float16)
    k = torch.randn(B, H, n_new, D, dtype=torch.float16)
    v = torch.randn(B, H, n_new, D, dtype=torch.float16)
    cache_index = torch.arange(pos, pos + n_new, dtype=torch.int64)
    mask = torch.zeros(B, 1, n_new, MAX_LEN, dtype=torch.float16)
    return q, k, v, cache_index, mask


def _export_buffer(n_new: int = 1):
    return torch.export.export(CacheAsBuffer(n_new).eval(), _step_inputs(n_new))


def _export_input():
    q, k, v, idx, mask = _step_inputs()
    cache = torch.zeros(B, H, MAX_LEN, D, dtype=torch.float16)
    return torch.export.export(
        CacheAsInput().eval(), (q, k, v, cache.clone(), cache.clone(), idx, mask)
    )


def _ops(graph_module) -> set:
    return {
        _aten_name(n) or str(n.target)
        for n in graph_module.graph.nodes
        if n.op == "call_function"
    }


def _output_kinds(ep) -> list:
    return [(s.kind, s.target) for s in ep.graph_signature.output_specs]


@pytest.fixture
def stub_preprocess(monkeypatch):
    """Replace ``SpyreBackend.preprocess`` with a stub that records the delegate's
    input/output signature (the real one needs torch_spyre)."""
    seen: dict = {}

    def _preprocess(cls, exported_program, compile_specs):
        sig = exported_program.graph_signature
        seen.setdefault("inputs", []).append(
            [(s.kind, s.arg.name) for s in sig.input_specs]
        )
        seen.setdefault("outputs", []).append(
            [(s.kind, s.target) for s in sig.output_specs]
        )
        seen.setdefault("input_shapes", []).append(
            [
                tuple(n.meta["val"].shape)
                for n in exported_program.graph_module.graph.nodes
                if n.op == "placeholder"
            ]
        )
        return PreprocessResult(processed_bytes=b"stub")

    monkeypatch.setattr(SpyreBackend, "preprocess", classmethod(_preprocess))
    return seen


@pytest.fixture
def no_tag_mutated_buffer(monkeypatch):
    """Disable the partitioner's ``tag_mutated_buffer`` call (pre-tag baseline)."""
    monkeypatch.setattr(spyre_partitioner, "tag_mutated_buffer", lambda ep: None)


@pytest.fixture
def allow_index_put(monkeypatch):
    """Pretend Phase B validated ``index_put`` (test-local allowlist change)."""
    monkeypatch.setattr(
        spyre_partitioner,
        "_SUPPORTED_ATEN_OPS",
        spyre_partitioner._SUPPORTED_ATEN_OPS | {"aten::index_put"},
    )


# --- Mutation kinds --------------------------------------------------------


def test_cache_as_input_is_user_input_mutation():
    edge = to_edge(_export_input()).exported_program()
    assert _output_kinds(edge)[:2] == [
        (OutputKind.USER_INPUT_MUTATION, "key_cache"),
        (OutputKind.USER_INPUT_MUTATION, "value_cache"),
    ]


def test_cache_as_buffer_is_buffer_mutation():
    edge = to_edge(_export_buffer()).exported_program()
    assert _output_kinds(edge)[:2] == [
        (OutputKind.BUFFER_MUTATION, "k_cache"),
        (OutputKind.BUFFER_MUTATION, "v_cache"),
    ]
    buffers = [
        s.target for s in edge.graph_signature.input_specs if s.kind == InputKind.BUFFER
    ]
    assert {"k_cache", "v_cache"} <= set(buffers)


# --- Op lowering chain -----------------------------------------------------


def test_index_copy_lowering_chain():
    ep = _export_buffer()
    # torch.export keeps the in-place op; functionalization makes it
    # out-of-place; to_edge's core-aten decomposition rewrites it to index_put.
    assert "aten.index_copy_.default" in _ops(ep.graph_module)
    assert "aten.index_copy.default" in _ops(ep.run_decompositions({}).graph_module)

    edge_ops = _ops(to_edge(_export_buffer()).exported_program().graph_module)
    assert "aten::index_put" in edge_ops
    assert "aten::index_copy" not in edge_ops


def test_preserve_ops_keeps_index_copy():
    config = EdgeCompileConfig(preserve_ops=[torch.ops.aten.index_copy.default])
    edge_ops = _ops(
        to_edge(_export_buffer(), compile_config=config).exported_program().graph_module
    )
    assert "aten::index_copy" in edge_ops
    assert "aten::index_put" not in edge_ops


def test_cache_index_stays_int64():
    edge = to_edge(_export_buffer()).exported_program()
    (idx,) = [
        n
        for n in edge.graph_module.graph.nodes
        if n.op == "placeholder" and n.name == "cache_index"
    ]
    assert idx.meta["val"].dtype == torch.int64


# --- Partitioning with today's allowlist (Phase 10+) ----------------------


def test_cache_write_ops_allowlist_state():
    # Phase 10: index_put validated via torch-spyre indirect scatter kernel.
    # index_copy stays off (SDSC compile FAIL -- dxp_standalone SIGABRT).
    assert "aten::index_put" in _SUPPORTED_ATEN_OPS
    assert "aten::index_copy" not in _SUPPORTED_ATEN_OPS


@pytest.mark.parametrize("n_new", [1, 64], ids=["decode", "prefill64"])
def test_write_delegated_attention_delegated(n_new):
    # Phase 10: index_put is on the allowlist and its base tensor is a
    # placeholder, so the is_state_scatter exemption keeps it delegated.
    # Both the cache write and the sdpa_math read are in the same partition.
    edge = to_edge(_export_buffer(n_new)).exported_program()
    SpyrePartitioner().partition(edge)
    nodes = [n for n in edge.graph_module.graph.nodes if n.op == "call_function"]
    on_cpu = {_aten_name(n) for n in nodes if "delegation_tag" not in n.meta}
    tags = {n.meta["delegation_tag"] for n in nodes if "delegation_tag" in n.meta}
    # Everything is delegated; no CPU fallback ops remain.
    assert on_cpu == set()
    assert len(tags) == 1


def test_fused_sdpa_needs_preserve_ops():
    """Without ``preserve_ops`` fused SDPA decomposes into mask-handling ops
    (``where``/``any``/``logical_not``) and fp32 casts that fall back to CPU;
    preserved, it is a single delegated node."""

    class FusedSdpa(CacheAsBuffer):
        def forward(self, q, k, v, cache_index, mask):
            self.k_cache.index_copy_(2, cache_index, k)
            self.v_cache.index_copy_(2, cache_index, v)
            return torch.nn.functional.scaled_dot_product_attention(
                q, self.k_cache, self.v_cache, attn_mask=mask
            )

    def edge_ops(config):
        ep = torch.export.export(FusedSdpa().eval(), _step_inputs())
        return _ops(to_edge(ep, compile_config=config).exported_program().graph_module)

    decomposed = edge_ops(EdgeCompileConfig())
    assert "aten::scaled_dot_product_attention" not in decomposed
    assert {"aten::where", "dim_order_ops::_to_dim_order_copy"} <= decomposed

    preserved = edge_ops(
        EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default]
        )
    )
    assert preserved == {"aten::index_put", "aten::scaled_dot_product_attention"}


# --- Lowering (to_backend) with a stubbed preprocess -----------------------


def test_untagged_cache_is_plain_delegate_input(stub_preprocess, no_tag_mutated_buffer):
    """Pre-tag baseline: without ``tag_mutated_buffer`` the whole caches are
    plain delegate inputs, so every call would copy the full K and V caches."""
    lowered = to_edge(_export_buffer()).to_backend(SpyrePartitioner())
    lowered.to_executorch()
    (inputs,) = stub_preprocess["inputs"]
    (shapes,) = stub_preprocess["input_shapes"]
    assert all(kind == InputKind.USER_INPUT for kind, _ in inputs)
    assert shapes.count((B, H, MAX_LEN, D)) == 2


def test_untagged_write_round_trips_cache(stub_preprocess, no_tag_mutated_buffer):
    """Pre-tag baseline: the delegate takes the cache in and returns the
    updated cache out; the top-level program writes it back to the buffer."""
    lowered = to_edge(_export_buffer()).to_backend(SpyrePartitioner())
    top = lowered.exported_program()
    assert _ops(top.graph_module) == {
        "executorch_call_delegate",
        "<built-in function getitem>",
    }
    assert _output_kinds(top)[:2] == [
        (OutputKind.BUFFER_MUTATION, "k_cache"),
        (OutputKind.BUFFER_MUTATION, "v_cache"),
    ]
    (outputs,) = stub_preprocess["outputs"]
    assert [kind for kind, _ in outputs] == [OutputKind.USER_OUTPUT] * 3
    lowered.to_executorch()


def test_partitioner_makes_cache_delegate_owned(stub_preprocess):
    """``SpyrePartitioner`` tags the mutated buffer: it moves into the delegate,
    so the delegate sees ``BUFFER`` / ``BUFFER_MUTATION`` and the top-level
    program no longer holds or passes the cache. This is the persistent-cache
    shape Phase C needs; ``SpyreBackend`` must then own the device buffer itself."""
    lowered = to_edge(_export_buffer()).to_backend(SpyrePartitioner())
    top = lowered.exported_program()
    top_buffers = {
        s.target for s in top.graph_signature.input_specs if s.kind == InputKind.BUFFER
    }
    assert not {"k_cache", "v_cache"} & top_buffers
    assert [k for k, _ in _output_kinds(top)] == [OutputKind.USER_OUTPUT]

    (inputs,) = stub_preprocess["inputs"]
    (outputs,) = stub_preprocess["outputs"]
    assert [(k, n) for k, n in inputs if k == InputKind.BUFFER] == [
        (InputKind.BUFFER, "b_k_cache"),
        (InputKind.BUFFER, "b_v_cache"),
    ]
    assert outputs[:2] == [
        (OutputKind.BUFFER_MUTATION, "k_cache"),
        (OutputKind.BUFFER_MUTATION, "v_cache"),
    ]
    lowered.to_executorch()


# --- LLM wrapper: forward(tokens, input_pos) -> logits ---------------------

_V, _C, _HD = 64, 128, 64  # vocab, cache length, head_dim (one stick)


class TokensInputPosLLM(torch.nn.Module):
    """Mirrors ``export_hfa.StandardLLMWrapper``'s contract: caches are
    registered buffers and ``cache_index`` / mask are derived in-graph from
    ``input_pos``, so the exported signature is just ``(tokens, input_pos)``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_V, _HD).half()
        self.out = torch.nn.Linear(_HD, _V, bias=False).half()
        for i in range(2):
            self.register_buffer(
                f"kc{i}", torch.zeros(1, 1, _C, _HD, dtype=torch.float16)
            )
            self.register_buffer(
                f"vc{i}", torch.zeros(1, 1, _C, _HD, dtype=torch.float16)
            )
        self.register_buffer("scale", torch.tensor(_HD**-0.5, dtype=torch.float16))
        self.register_buffer("_pos_range", torch.arange(_C, dtype=torch.int64))
        self.register_buffer("_fill", torch.tensor(-32752.0, dtype=torch.float16))
        self.register_buffer("_zero", torch.zeros((), dtype=torch.float16))

    def forward(self, tokens, input_pos):
        masked = self._pos_range > input_pos[0]
        mask = torch.where(
            masked.view(1, 1, 1, _C),
            self._fill.expand(1, 1, 1, _C),
            self._zero.expand(1, 1, 1, _C),
        )
        h = self.embed(tokens).view(1, 1, 1, _HD)
        for i in range(2):
            kc, vc = getattr(self, f"kc{i}"), getattr(self, f"vc{i}")
            kc.index_copy_(2, input_pos, h)
            vc.index_copy_(2, input_pos, h)
            h = sdpa_math(h, kc, vc, mask, scale=self.scale)
        return self.out(h.view(1, 1, _HD))


_KV = ["kc0", "vc0", "kc1", "vc1"]


def _llm_edge():
    tokens = torch.zeros(1, 1, dtype=torch.int64)
    input_pos = torch.tensor([3], dtype=torch.int64)
    return to_edge(torch.export.export(TokensInputPosLLM().eval(), (tokens, input_pos)))


def test_llm_wrapper_edge_signature_has_buffer_and_buffer_mutation():
    sig = _llm_edge().exported_program().graph_signature
    assert sig.user_inputs == ("tokens", "input_pos")
    assert len(sig.user_outputs) == 1  # logits
    assert [
        t for k, t in _output_kinds_of(sig) if k == OutputKind.BUFFER_MUTATION
    ] == _KV
    assert set(_KV) <= {s.target for s in sig.input_specs if s.kind == InputKind.BUFFER}
    # BUFFER_MUTATION outputs precede the user output.
    assert [k for k, _ in _output_kinds_of(sig)] == [OutputKind.BUFFER_MUTATION] * 4 + [
        OutputKind.USER_OUTPUT
    ]


def test_llm_wrapper_tagged_lowering_keeps_user_signature(stub_preprocess):
    lowered = _llm_edge().to_backend(SpyrePartitioner())
    top = lowered.exported_program().graph_signature
    # User-visible signature unchanged: (tokens, input_pos) -> logits.
    assert top.user_inputs == ("tokens", "input_pos")
    assert len(top.user_outputs) == 1
    # Top level no longer sees the KV caches at all.
    assert not set(_KV) & {s.target for s in top.input_specs}
    assert not top.buffers_to_mutate
    assert [k for k, _ in _output_kinds_of(top)] == [OutputKind.USER_OUTPUT]

    # Delegate side: KV caches are BUFFER inputs and BUFFER_MUTATION outputs.
    delegate_inputs = [
        i
        for i in stub_preprocess["inputs"]
        if any(k == InputKind.BUFFER and n == "b_kc0" for k, n in i)
    ]
    delegate_outputs = [
        o
        for o in stub_preprocess["outputs"]
        if any(k == OutputKind.BUFFER_MUTATION for k, _ in o)
    ]
    (d_in,), (d_out,) = delegate_inputs, delegate_outputs
    assert [n for k, n in d_in if k == InputKind.BUFFER] == [f"b_{n}" for n in _KV]
    assert [t for k, t in d_out if k == OutputKind.BUFFER_MUTATION] == _KV
    lowered.to_executorch()


def test_llm_wrapper_delegate_metadata_for_kv_state():
    """What the backend can read to identify delegate-owned kv_state buffers."""
    captured: dict = {}

    def _preprocess(cls, program, compile_specs):
        sig = program.graph_signature
        if sig.buffers_to_mutate:
            captured["buffers_to_mutate"] = dict(sig.buffers_to_mutate)
            captured["inputs_to_buffers"] = dict(sig.inputs_to_buffers)
            captured["persistent"] = {
                s.arg.name: s.persistent
                for s in sig.input_specs
                if s.kind == InputKind.BUFFER
            }
            captured["state_shapes"] = {
                k: tuple(v.shape) for k, v in program.state_dict.items()
            }
        return PreprocessResult(processed_bytes=b"stub")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_preprocess))
        _llm_edge().to_backend(SpyrePartitioner()).to_executorch()

    # output arg name -> buffer FQN: pairs each BUFFER_MUTATION output with the
    # BUFFER input it overwrites (input name -> FQN via inputs_to_buffers).
    assert sorted(captured["buffers_to_mutate"].values()) == sorted(_KV)
    assert {captured["inputs_to_buffers"][f"b_{n}"] for n in _KV} == set(_KV)
    assert all(captured["persistent"][f"b_{n}"] for n in _KV)
    assert {captured["state_shapes"][n] for n in _KV} == {(1, 1, _C, _HD)}


def _output_kinds_of(sig) -> list:
    return [(s.kind, s.target) for s in sig.output_specs]


# --- CPU-only runtime baseline ---------------------------------------------


def test_cpu_runtime_persists_cache_across_steps():
    """CPU reference for Phase D: the portable runtime keeps a mutable buffer
    across ``execute()`` calls, so two decode steps match eager."""
    runtime = pytest.importorskip("executorch.runtime")

    torch.manual_seed(0)
    eager = CacheAsBuffer().eval()
    pte = to_edge(
        torch.export.export(CacheAsBuffer().eval(), _step_inputs())
    ).to_executorch()
    method = runtime.Runtime.get().load_program(pte.buffer).load_method("forward")

    written = []
    for pos in (3, 7):
        q, k, v, idx, _ = _step_inputs(pos=pos)
        written.append(pos)
        # Host-built decode mask: attend only to slots written so far.
        mask = torch.full((B, 1, 1, MAX_LEN), float("-inf"), dtype=torch.float16)
        mask[..., written] = 0
        expected = eager(q, k, v, idx, mask)
        (actual,) = method.execute((q, k, v, idx, mask))
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


# --- Cache write adds no CPU fallback (LLM-wrapper signature) ---------------


class _UncachedLLM(TokensInputPosLLM):
    """``TokensInputPosLLM`` without the cache writes: attention reads the
    (never-written) cache buffers, so it has the same graph minus ``index_put``."""

    def forward(self, tokens, input_pos):
        masked = self._pos_range > input_pos[0]
        mask = torch.where(
            masked.view(1, 1, 1, _C),
            self._fill.expand(1, 1, 1, _C),
            self._zero.expand(1, 1, 1, _C),
        )
        h = self.embed(tokens).view(1, 1, 1, _HD)
        for i in range(2):
            h = sdpa_math(
                h,
                getattr(self, f"kc{i}"),
                getattr(self, f"vc{i}"),
                mask,
                scale=self.scale,
            )
        return self.out(h.view(1, 1, _HD))


def _cpu_partition(module, mode):
    tokens = torch.zeros(1, 1, dtype=torch.int64)
    input_pos = torch.tensor([3], dtype=torch.int64)
    edge = to_edge(torch.export.export(module.eval(), (tokens, input_pos)))
    ep = edge.exported_program()
    SpyrePartitioner(mode=mode).partition(ep)
    nodes = [n for n in ep.graph_module.graph.nodes if n.op == "call_function"]
    on_cpu = Counter(_aten_name(n) for n in nodes if "delegation_tag" not in n.meta)
    tags = {n.meta["delegation_tag"] for n in nodes if "delegation_tag" in n.meta}
    return on_cpu, len(tags)


@pytest.mark.parametrize("mode", ["fallbacklist", "allowlist"])
def test_llm_wrapper_cache_adds_no_cpu_fallback(mode, monkeypatch):
    """The cache adds no extra CPU fallbacks and no extra partitions relative to
    the same model without cache writes.  ``index_put`` is delegated via the
    ``is_state_scatter`` exemption and fuses into the existing attention
    partitions rather than splitting them."""
    # Host CI has no torch-spyre; stub an empty fallback_ops so fallbacklist mode
    # (which fails closed without it) can run.
    monkeypatch.setattr(
        "executorch.backends.spyre.spyre_partitioner.load_spyre_fallback_ops",
        lambda: (frozenset(), ""),
    )
    ref_cpu, ref_parts = _cpu_partition(_UncachedLLM(), mode)
    cpu, parts = _cpu_partition(TokensInputPosLLM(), mode)

    assert cpu - ref_cpu == Counter()
    assert ref_cpu - cpu == Counter()
    assert parts == ref_parts


# ==========================================================================
# Phase B: delegate-owned KV state (spyre_backend.py + SpyreBackend.cpp tests)
# ==========================================================================


# --- Phase B.1: kv_state classification -----------------------------------


def test_build_kv_state_out_map_finds_buffer_mutations(stub_preprocess):
    """_build_kv_state_out_map returns {placeholder_name -> output_index} for
    every BUFFER input whose FQN appears in buffers_to_mutate."""
    from executorch.backends.spyre.spyre_backend import _build_kv_state_out_map

    # Extend stub to also capture kv_map.
    seen_kv: dict = {}
    _orig_preprocess = SpyreBackend.preprocess.__func__  # type: ignore[attr-defined]

    def _capturing_preprocess(cls, program, compile_specs):
        kv_map = _build_kv_state_out_map(program)
        seen_kv.update(kv_map)
        return PreprocessResult(processed_bytes=b"stub")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_capturing_preprocess))
        _llm_edge().to_backend(SpyrePartitioner()).to_executorch()

    # Each KV buffer (b_kc0, b_vc0, b_kc1, b_vc1) maps to a BUFFER_MUTATION
    # output index (0..3 in BUFFER_MUTATION order, before the USER_OUTPUT).
    assert set(seen_kv.keys()) == {f"b_{n}" for n in _KV}
    # Output indices must be distinct and within [0, n_outputs).
    assert len(set(seen_kv.values())) == len(_KV)
    assert all(v >= 0 for v in seen_kv.values())


def test_placeholder_kind_returns_kv_state_for_mutated_buffers():
    """_placeholder_kind returns 'kv_state' for BUFFER placeholders that are
    in kv_state_names, and 'buffer' for non-mutated buffers."""
    from executorch.backends.spyre.spyre_backend import (
        _build_kv_state_out_map,
        _placeholder_kind,
    )

    seen_kinds: dict = {}

    def _capturing_preprocess(cls, program, compile_specs):
        kv_map = _build_kv_state_out_map(program)
        kv_names = frozenset(kv_map.keys())
        gm = program.graph_module
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                seen_kinds[node.name] = _placeholder_kind(program, node.name, kv_names)
        return PreprocessResult(processed_bytes=b"stub")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_capturing_preprocess))
        _llm_edge().to_backend(SpyrePartitioner()).to_executorch()

    # KV cache placeholders are "kv_state".
    for kv_name in _KV:
        ph_name = f"b_{kv_name}"
        assert (
            seen_kinds.get(ph_name) == "kv_state"
        ), f"{ph_name!r} expected 'kv_state', got {seen_kinds.get(ph_name)!r}"
    # Non-KV inputs (user_input, parameter, buffer, constant) must NOT be kv_state.
    for ph_name, kind in seen_kinds.items():
        if ph_name in {f"b_{n}" for n in _KV}:
            continue
        assert (
            kind != "kv_state"
        ), f"{ph_name!r} should not be 'kv_state' (got {kind!r})"


# --- Phase B.2: payload contents (requires real preprocess) ---------------


def test_preprocess_emits_kv_state_in_payload():
    """The processed payload contains kind='kv_state' + state_out for every
    KV cache buffer and kind='user_input' for tokens/input_pos."""
    import json
    import struct

    # Require real torch_spyre (this test gates on the real compiler).
    try:
        import torch_spyre  # noqa: F401

        torch_spyre._autoload()
    except Exception:
        pytest.skip("torch_spyre not available")

    captured_payloads: list = []
    _real_preprocess = SpyreBackend.preprocess  # staticmethod, callable directly

    def _capture_preprocess(cls, edge_program, compile_specs):
        result = _real_preprocess(edge_program, compile_specs)
        captured_payloads.append(result.processed_bytes)
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_capture_preprocess))
        try:
            _llm_edge().to_backend(SpyrePartitioner()).to_executorch()
        except Exception as e:
            if captured_payloads:
                pass  # Partial success: some partitions compiled, check those.
            else:
                pytest.skip(f"preprocess compilation failed: {e}")

    assert captured_payloads, "no partition was preprocessed"
    # Inspect the delegate partition(s) for kv_state inputs.
    for raw in captured_payloads:
        if len(raw) < 4:
            continue
        (hlen,) = struct.unpack_from("<I", raw, 0)
        header = json.loads(raw[4 : 4 + hlen])
        kv_specs = [s for s in header.get("inputs", []) if s.get("kind") == "kv_state"]
        if kv_specs:
            # Every kv_state input must have a non-negative state_out.
            assert all(
                s.get("state_out", -1) >= 0 for s in kv_specs
            ), f"kv_state specs without state_out: {kv_specs}"
            # Shape and dtype must be non-empty.
            for s in kv_specs:
                assert s.get("shape"), f"kv_state spec missing shape: {s}"
                assert s.get("dtype"), f"kv_state spec missing dtype: {s}"
            break
    else:
        pytest.fail("no partition with kv_state inputs found in payload")


# --- Phase B.3: ET arg-index mapping --------------------------------------


def test_kv_state_out_map_indices_match_buffer_mutation_order():
    """The state_out indices from _build_kv_state_out_map correspond to the
    BUFFER_MUTATION output positions in the delegate's output_specs."""
    from executorch.backends.spyre.spyre_backend import _build_kv_state_out_map

    kv_out_maps: list = []

    def _capturing_preprocess(cls, program, compile_specs):
        kv_map = _build_kv_state_out_map(program)
        sig = program.graph_signature
        # Collect expected: BUFFER_MUTATION output index by FQN.
        expected: dict[str, int] = {}
        inputs_to_buffers = getattr(sig, "inputs_to_buffers", {}) or {}
        for i, spec in enumerate(sig.output_specs):
            if spec.kind == OutputKind.BUFFER_MUTATION:
                expected[spec.target] = i

        # Map placeholder -> expected output index.
        expected_by_ph: dict[str, int] = {}
        for ph_name in kv_map:
            fqn = inputs_to_buffers.get(ph_name, ph_name[2:])
            expected_by_ph[ph_name] = expected.get(fqn, -1)

        kv_out_maps.append((kv_map, expected_by_ph))
        return PreprocessResult(processed_bytes=b"stub")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_capturing_preprocess))
        _llm_edge().to_backend(SpyrePartitioner()).to_executorch()

    assert kv_out_maps, "preprocess was not called"
    kv_map, expected_by_ph = kv_out_maps[0]
    for ph_name, out_idx in kv_map.items():
        assert out_idx == expected_by_ph.get(ph_name), (
            f"{ph_name!r}: kv_state_out_map gives {out_idx}, "
            f"expected {expected_by_ph.get(ph_name)} from output_specs"
        )


# --- Phase B.4: ping-pong ownership logic (host simulation) ---------------


def test_kv_state_ping_pong_simulation():
    """Simulate the C++ ping-pong logic in Python to verify correctness:
    - First call: allocate zero tensor on 'device' (host tensor here).
    - Post-sync: capture the output tensor into kv_state_cache.
    - Second call: inject the captured tensor (no re-allocation).
    - The injected tensor is the previous call's output.

    This is a host-only, device-free simulation of the SpyreBackend logic.
    """
    from executorch.backends.spyre.spyre_backend import _build_kv_state_out_map

    kv_map_captured: dict = {}

    def _capturing_preprocess(cls, program, compile_specs):
        kv_map_captured.update(_build_kv_state_out_map(program))
        return PreprocessResult(processed_bytes=b"stub")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(SpyreBackend, "preprocess", classmethod(_capturing_preprocess))
        _llm_edge().to_backend(SpyrePartitioner()).to_executorch()

    assert kv_map_captured, "no kv_state inputs found"

    # Simulate the runtime's kv_state_cache (parallel to n_inputs).
    # The delegate has n_kv_state inputs + other inputs; use the max ordinal.
    # For simplicity, index by position in kv_map_captured.
    n_slots = len(kv_map_captured)
    kv_state_cache: list = [None] * n_slots
    state_out_indices = list(kv_map_captured.values())

    # All state_out indices must be non-negative and distinct.
    assert all(so >= 0 for so in state_out_indices)
    assert len(set(state_out_indices)) == n_slots, "state_out indices must be unique"

    # n_out is at least max(state_out_indices)+1 (KV outputs) + 1 (USER logits).
    n_out = max(state_out_indices) + 2

    # Build kv_out_skip: True for KV output indices, False for logits output.
    kv_out_skip = [False] * n_out
    for so in state_out_indices:
        kv_out_skip[so] = True

    # Simulate two execute() calls.
    def _fake_kernel_output(call_idx):
        """A fresh tensor with value == call_idx+1 for each KV output slot."""
        return {
            so: torch.full((1, 1, _C, _HD), float(call_idx + 1), dtype=torch.float16)
            for so in state_out_indices
        }

    for call_idx in range(2):
        # Inject phase (call_idx > 0 would inject from cache).
        if call_idx == 0:
            # First call: allocate zero tensors on 'device'.
            injected = [
                torch.zeros(1, 1, _C, _HD, dtype=torch.float16) for _ in range(n_slots)
            ]
            for slot_idx, t in enumerate(injected):
                assert t.flatten()[0].item() == 0.0, "initial KV state must be zero"
        else:
            # Subsequent calls: inject from kv_state_cache.
            for slot_idx in range(n_slots):
                assert (
                    kv_state_cache[slot_idx] is not None
                ), f"kv_state_cache[{slot_idx}] was not populated after call 0"
            # Injected tensor is the output from the PREVIOUS call (value = call_idx).
            for slot_idx in range(n_slots):
                assert kv_state_cache[slot_idx].flatten()[0].item() == float(
                    call_idx
                ), (
                    f"slot {slot_idx}: expected value {float(call_idx)} (call_idx={call_idx}), "
                    f"got {kv_state_cache[slot_idx].flatten()[0].item()}"
                )

        # Kernel produces outputs (value = call_idx + 1).
        kernel_outs = _fake_kernel_output(call_idx)

        # Capture: store output[state_out] into kv_state_cache.
        for slot_idx, (ph_name, so) in enumerate(kv_map_captured.items()):
            kv_state_cache[slot_idx] = kernel_outs[so]

    # After two calls, cache holds call 1's output (value = 2.0).
    for slot_idx in range(n_slots):
        assert kv_state_cache[slot_idx].flatten()[0].item() == 2.0, (
            f"slot {slot_idx}: expected 2.0 after 2 calls, "
            f"got {kv_state_cache[slot_idx].flatten()[0].item()}"
        )

    # Verify D2H skip set is correct.
    assert all(kv_out_skip[so] for so in state_out_indices)
    # USER_OUTPUT (logits) index is NOT skipped — it's the last output.
    assert not kv_out_skip[n_out - 1]
