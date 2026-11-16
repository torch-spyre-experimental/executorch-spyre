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
"""Block-level delegation (``granularity="block"``, the default).

Host-only.  A small LLM-shaped module (``layers`` ModuleList, per-layer
``kc{i}`` / ``vc{i}`` mutable KV buffers, fused SDPA, ``(tokens, input_pos)``
signature) stands in for the hf-adapters Granite wrapper.  ``SpyreBackend.
preprocess`` is stubbed, and the torch-spyre fallback list is injected, so no
Spyre device is needed.

Covers: partition count (N layers -> N delegates; ``model`` -> 1), block
boundaries (no delegate spans two blocks), KV ownership (``kc{i}``/``vc{i}``
owned by the delegate of block ``i`` only), the diagnostics report, and that
lowering still yields one ``DelegateCall`` per block.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

import torch.nn.functional as F  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402
from executorch.exir.backend.backend_details import PreprocessResult  # noqa: E402
from torch.export.graph_signature import InputKind  # noqa: E402

from executorch.backends.spyre import spyre_partitioner as ap  # noqa: E402
from executorch.backends.spyre import spyre_block_partition as bp  # noqa: E402
from executorch.backends.spyre.spyre_backend import SpyreBackend  # noqa: E402
from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner  # noqa: E402

_V, _C, _HD = 64, 128, 64


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = torch.nn.Linear(_HD, _HD, bias=False).half()
        self.k = torch.nn.Linear(_HD, _HD, bias=False).half()
        self.v = torch.nn.Linear(_HD, _HD, bias=False).half()
        self.o = torch.nn.Linear(_HD, _HD, bias=False).half()
        # Per-block mutable KV state, like hf-adapters' kc{i}/vc{i}.
        self.register_buffer("kc", torch.zeros(1, 1, _C, _HD, dtype=torch.float16))
        self.register_buffer("vc", torch.zeros(1, 1, _C, _HD, dtype=torch.float16))

    def forward(self, h, input_pos, mask):
        q = self.q(h).view(1, 1, 1, _HD)
        self.kc.index_copy_(2, input_pos, self.k(h).view(1, 1, 1, _HD))
        self.vc.index_copy_(2, input_pos, self.v(h).view(1, 1, 1, _HD))
        a = F.scaled_dot_product_attention(q, self.kc, self.vc, attn_mask=mask)
        return h + self.o(a.view(1, 1, _HD))


class _LLM(torch.nn.Module):
    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_V, _HD).half()
        self.layers = torch.nn.ModuleList(_Block() for _ in range(n_layers))
        self.out = torch.nn.Linear(_HD, _V, bias=False).half()

    def forward(self, tokens, input_pos, mask):
        h = self.embed(tokens).view(1, 1, _HD)
        for layer in self.layers:
            h = layer(h, input_pos, mask)
        return self.out(h)


def _edge(n_layers: int):
    ex = (
        torch.zeros(1, 1, dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
        torch.zeros(1, 1, 1, _C, dtype=torch.float16),
    )
    ep = torch.export.export(_LLM(n_layers).eval(), ex)
    return to_edge(
        ep,
        compile_config=EdgeCompileConfig(
            preserve_ops=[torch.ops.aten.scaled_dot_product_attention.default]
        ),
    )


@pytest.fixture(autouse=True)
def _no_torch_spyre(monkeypatch):
    monkeypatch.setattr(ap, "load_spyre_fallback_ops", lambda: (frozenset(), ""))
    monkeypatch.delenv(bp.GRANULARITY_ENV_VAR, raising=False)
    monkeypatch.delenv(ap.PARTITION_MODE_ENV_VAR, raising=False)


@pytest.fixture
def stub_preprocess(monkeypatch):
    seen: list = []

    def _preprocess(cls, exported_program, compile_specs):
        sig = exported_program.graph_signature
        seen.append([s.target for s in sig.input_specs if s.kind == InputKind.BUFFER])
        return PreprocessResult(processed_bytes=b"stub")

    monkeypatch.setattr(SpyreBackend, "preprocess", classmethod(_preprocess))
    return seen


def _partition(n_layers: int, granularity=None):
    edge = _edge(n_layers)
    ep = edge.exported_program()
    p = SpyrePartitioner(granularity=granularity)
    res = p.partition(ep)
    return p, ep, res


def _tag_of(ep, fqn: str):
    sig = ep.graph_signature
    (node,) = [
        n
        for n in ep.graph.nodes
        if n.op == "placeholder" and sig.inputs_to_buffers.get(n.name) == fqn
    ]
    return node.meta.get("delegation_tag")


# --- granularity resolution -------------------------------------------------


def test_default_granularity_is_block():
    assert bp.resolve_partition_granularity(None) == bp.GRANULARITY_BLOCK
    assert SpyrePartitioner().granularity == "block"


def test_env_var_selects_granularity_and_explicit_wins(monkeypatch):
    monkeypatch.setenv(bp.GRANULARITY_ENV_VAR, "model")
    assert bp.resolve_partition_granularity(None) == "model"
    assert bp.resolve_partition_granularity("block") == "block"
    assert SpyrePartitioner().granularity == "model"


def test_unknown_granularity_rejected():
    with pytest.raises(ValueError, match="unknown Spyre partition granularity"):
        SpyrePartitioner(granularity="layer")


@pytest.mark.parametrize(
    "gran, expected",
    [("chunk:1", "chunk:1"), ("chunk:2", "chunk:2"), ("chunk:8", "chunk:8")],
)
def test_chunk_granularity_parsing_valid(gran, expected):
    assert bp.resolve_partition_granularity(gran) == expected
    assert SpyrePartitioner(granularity=gran).granularity == expected
    assert bp.parse_chunk_size(gran) == int(expected.split(":")[1])


@pytest.mark.parametrize(
    "gran", ["chunk:0", "chunk:-2", "chunk:foo", "chunk:", "chunk:-1"]
)
def test_chunk_granularity_parsing_invalid(gran):
    with pytest.raises(ValueError, match="invalid Spyre partition granularity"):
        bp.resolve_partition_granularity(gran)


# --- partition count --------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 2, 3, 5])
def test_block_granularity_yields_one_delegate_per_layer(n_layers):
    _, _, res = _partition(n_layers)
    assert len(res.partition_tags) == n_layers


@pytest.mark.parametrize("n_layers", [2, 4])
def test_model_granularity_is_monolithic(n_layers):
    p, _, res = _partition(n_layers, granularity="model")
    assert len(res.partition_tags) == 1
    assert p.last_report["block_container"] is None


def test_single_layer_model_equals_block():
    _, _, blk = _partition(1, "block")
    _, _, mono = _partition(1, "model")
    assert len(blk.partition_tags) == len(mono.partition_tags) == 1


# --- block boundaries -------------------------------------------------------


def test_block_container_is_discovered_not_hardcoded():
    p, _, _ = _partition(3)
    assert p.last_report["block_container"] == "layers"


def test_no_delegate_spans_two_blocks():
    p, ep, _ = _partition(4)
    container = p.last_report["block_container"]
    per_tag: dict = {}
    for n in ep.graph.nodes:
        tag = n.meta.get("delegation_tag")
        if n.op != "call_function" or tag is None:
            continue
        per_tag.setdefault(tag, set()).add(bp.block_of(n, container))
    assert len(per_tag) == 4
    for tag, blocks in per_tag.items():
        # Keyless nodes (embedding / final lm_head) are folded into a neighbour
        # block, so each delegate has exactly one real block index.
        assert len(blocks - {None}) == 1, (tag, blocks)
    # ... and the four delegates cover the four distinct blocks.
    assert {next(iter(b - {None})) for b in per_tag.values()} == {0, 1, 2, 3}


def test_keyless_nodes_fold_into_neighbours_without_extra_delegates():
    # embedding precedes block 0 and lm_head follows the last block. lm_head is
    # delegable and must not create a delegate of its own. embedding is recorded
    # ``validated: false`` in supported_ops.yaml (no SDSC kernel), so the partition
    # contract keeps it on CPU in every mode: it is the only CPU node and must not
    # create a delegate either.
    p, _, res = _partition(2)
    assert len(res.partition_tags) == 2
    assert p.last_report["n_cpu_nodes"] == 1


def test_fill_keys_and_scc_helpers():
    assert bp._fill_keys([None, None, 0, None, 1, None]) == [0, 0, 0, 0, 1, 1]
    assert bp._fill_keys([None, None]) == [None, None]
    comps = bp._sccs({0: {1}, 1: {0}, 2: set()})
    assert sorted(map(sorted, comps)) == [[0, 1], [2]]


# --- KV ownership -----------------------------------------------------------


@pytest.mark.parametrize("n_layers", [2, 3])
def test_kv_buffers_owned_by_their_own_block(n_layers):
    p, ep, _ = _partition(n_layers)
    tags = []
    for i in range(n_layers):
        tk, tv = _tag_of(ep, f"layers.{i}.kc"), _tag_of(ep, f"layers.{i}.vc")
        assert tk is not None and tk == tv, f"layer {i}: kc/vc split across delegates"
        tags.append(tk)
    assert len(set(tags)) == n_layers  # distinct delegate per block
    assert p.last_report["kv_violations"] == []
    # Report agrees: every partition lists exactly its own kc/vc.
    for part in p.last_report["partitions"]:
        (blk,) = part["blocks"]
        assert part["kv_buffers"] == sorted([f"layers.{blk}.kc", f"layers.{blk}.vc"])


def test_lowering_gives_one_delegate_call_per_block_with_owned_state(stub_preprocess):
    n_layers = 3
    lowered = _edge(n_layers).to_backend(SpyrePartitioner())
    top = lowered.exported_program().graph_signature
    assert top.user_inputs == ("tokens", "input_pos", "mask")
    assert not top.buffers_to_mutate  # state moved into the delegates
    assert not [s for s in top.input_specs if s.kind == InputKind.BUFFER]

    # One preprocess (= one payload) per block; each sees only its own KV.
    assert len(stub_preprocess) == n_layers
    kv_per_delegate = sorted(
        sorted(t for t in bufs if t.rsplit(".", 1)[-1] in ("kc", "vc"))
        for bufs in stub_preprocess
    )
    assert kv_per_delegate == [
        [f"layers.{i}.kc", f"layers.{i}.vc"] for i in range(n_layers)
    ]

    et = lowered.to_executorch()
    plan = et.executorch_program.execution_plan[0]
    n_calls = sum(
        1
        for ch in plan.chains
        for ins in ch.instructions
        if type(ins.instr_args).__name__ == "DelegateCall"
    )
    assert len(plan.delegates) == n_layers
    assert n_calls == n_layers


def test_model_granularity_lowering_is_single_delegate(stub_preprocess):
    lowered = _edge(3).to_backend(SpyrePartitioner(granularity="model"))
    assert len(stub_preprocess) == 1
    assert (
        len(lowered.to_executorch().executorch_program.execution_plan[0].delegates) == 1
    )


# --- diagnostics ------------------------------------------------------------


def test_partition_report_contents(caplog):
    with caplog.at_level(logging.INFO, logger=ap.logger.name):
        p, _, _ = _partition(3)
    rep = p.last_report
    assert rep["granularity"] == "block" and rep["n_partitions"] == 3
    assert [x["blocks"] for x in rep["partitions"]] == [[0], [1], [2]]
    assert all(x["n_nodes"] > 0 for x in rep["partitions"])
    msgs = [r.getMessage() for r in caplog.records]
    assert "PARTITION GRANULARITY: block" in msgs
    assert "=== Spyre PARTITION REPORT ===" in msgs
    assert any("layers.2.kc" in m for m in msgs)
    assert any("3 partition(s)" in m for m in msgs)


# --- chunk granularity tests ------------------------------------------------


@pytest.mark.parametrize(
    "n_layers, granularity, expected_delegates",
    [
        (1, "block", 1),
        (1, "chunk:2", 1),
        (1, "chunk:4", 1),
        (1, "model", 1),
        (2, "block", 2),
        (2, "chunk:2", 1),
        (2, "chunk:4", 1),
        (2, "model", 1),
        (3, "block", 3),
        (3, "chunk:2", 2),
        (3, "chunk:4", 1),
        (3, "model", 1),
        (4, "block", 4),
        (4, "chunk:2", 2),
        (4, "chunk:4", 1),
        (4, "model", 1),
        (5, "block", 5),
        (5, "chunk:2", 3),
        (5, "chunk:4", 2),
        (5, "model", 1),
        (8, "block", 8),
        (8, "chunk:2", 4),
        (8, "chunk:4", 2),
        (8, "model", 1),
    ],
)
def test_chunk_granularity_partition_counts(n_layers, granularity, expected_delegates):
    p, _, res = _partition(n_layers, granularity=granularity)
    assert len(res.partition_tags) == expected_delegates
    assert p.last_report["n_partitions"] == expected_delegates


def test_chunk_granularity_block_distribution():
    # 5 layers with chunk:2 -> 3 delegates: [0, 1], [2, 3], [4]
    p, _, _ = _partition(5, granularity="chunk:2")
    assert [x["blocks"] for x in p.last_report["partitions"]] == [[0, 1], [2, 3], [4]]

    # 6 layers with chunk:4 -> 2 delegates: [0, 1, 2, 3], [4, 5]
    p, _, _ = _partition(6, granularity="chunk:4")
    assert [x["blocks"] for x in p.last_report["partitions"]] == [[0, 1, 2, 3], [4, 5]]


def test_chunk_granularity_kv_buffers_ownership():
    p, ep, _ = _partition(4, granularity="chunk:2")
    assert p.last_report["kv_violations"] == []
    assert len(p.last_report["partitions"]) == 2
    # First delegate owns blocks 0 and 1 KV buffers
    assert p.last_report["partitions"][0]["kv_buffers"] == [
        "layers.0.kc",
        "layers.0.vc",
        "layers.1.kc",
        "layers.1.vc",
    ]
    # Second delegate owns blocks 2 and 3 KV buffers
    assert p.last_report["partitions"][1]["kv_buffers"] == [
        "layers.2.kc",
        "layers.2.vc",
        "layers.3.kc",
        "layers.3.vc",
    ]


def test_chunk_granularity_lowering(stub_preprocess):
    n_layers = 4
    lowered = _edge(n_layers).to_backend(SpyrePartitioner(granularity="chunk:2"))
    top = lowered.exported_program().graph_signature
    assert top.user_inputs == ("tokens", "input_pos", "mask")
    assert not top.buffers_to_mutate
    assert not [s for s in top.input_specs if s.kind == InputKind.BUFFER]

    assert len(stub_preprocess) == 2
    kv_per_delegate = sorted(
        sorted(t for t in bufs if t.rsplit(".", 1)[-1] in ("kc", "vc"))
        for bufs in stub_preprocess
    )
    assert kv_per_delegate == [
        ["layers.0.kc", "layers.0.vc", "layers.1.kc", "layers.1.vc"],
        ["layers.2.kc", "layers.2.vc", "layers.3.kc", "layers.3.vc"],
    ]

    et = lowered.to_executorch()
    plan = et.executorch_program.execution_plan[0]
    n_calls = sum(
        1
        for ch in plan.chains
        for ins in ch.instructions
        if type(ins.instr_args).__name__ == "DelegateCall"
    )
    assert len(plan.delegates) == 2
    assert n_calls == 2
