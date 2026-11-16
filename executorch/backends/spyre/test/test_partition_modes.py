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
"""Host tests for the partition-mode contract (#50).

The contract under test (see ``docs/operator-guide.md#partition-modes``):

* ``fallbacklist`` is the production default; ``allowlist`` is the conservative /
  CI mode.
* ``fallbacklist`` needs torch-spyre's ``fallback_ops`` and fails closed
  (``RuntimeError``) when they cannot be loaded -- it never silently uses the
  allowlist.
* Every ``validated: false`` op in ``supported_ops.yaml`` and every graph-local
  unsafe pattern stays on CPU in **both** modes.

No torch-spyre is installed on the host, so torch-spyre's ``fallback_ops`` is
stubbed by patching ``load_spyre_fallback_ops``.
"""

import json
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

import torch.nn.functional as F  # noqa: E402
from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402

from executorch.backends.spyre import spyre_partitioner as ap  # noqa: E402
from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    MODE_ALLOWLIST,
    MODE_FALLBACKLIST,
    PARTITION_MODE_ENV_VAR,
    SUPPORTED_TORCH_SPYRE,
    SpyrePartitioner,
    _aten_name,
    resolve_partition_mode,
    shadow_decisions,
    spyre_fallback_provenance,
)
from executorch.backends.spyre.coverage.ops_registry import (  # noqa: E402
    all_op_names,
    validated_op_names,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


# --- Modules under test -----------------------------------------------------


class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Sin(torch.nn.Module):
    """``sin`` is in neither supported_ops.yaml nor (in these tests) fallback_ops."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)

    def forward(self, x):
        return torch.relu(torch.sin(self.fc(x)))


class Embedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(32, 64)
        self.fc = torch.nn.Linear(64, 64)

    def forward(self, ids):
        return self.fc(self.emb(ids))


class RMSNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(64))

    def forward(self, x):
        return F.rms_norm(x, (64,), self.w) + x


class LayerNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n = torch.nn.LayerNorm(64)

    def forward(self, x):
        return self.n(x) + x


class InnermostCat(torch.nn.Module):
    """RoPE ``rotate_half``-style innermost-dim concat (SDSC stick scramble)."""

    def forward(self, x):
        return torch.cat([-x[:, 32:], x[:, :32]], dim=-1) + x


class Slice(torch.nn.Module):
    def forward(self, x):
        return x[:, :32] * 2


class Select(torch.nn.Module):
    def forward(self, x):
        return x[0] * 2


class Cast(torch.nn.Module):
    def forward(self, x):
        return x.float() * 2


class SoftmaxFedMatmul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.fc(torch.softmax(x, -1)) + x


class WideOutput(torch.nn.Module):
    """A single linear whose 128-wide result is written straight back to the host."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(64, 128)

    def forward(self, x):
        return self.fc(x)


# --- Helpers ----------------------------------------------------------------

# Only the norm modules need their op kept un-decomposed. Preserving ops that a
# module does not use makes every later ``to_edge`` call in the process slower
# (0.15 s -> 6 s by the 12th call), so preserve per module, not globally.
_PRESERVE_FOR = {
    RMSNorm: [torch.ops.aten.rms_norm.default],
    LayerNorm: [torch.ops.aten.native_layer_norm.default],
}


def _edge(mod, x=None):
    preserve = _PRESERVE_FOR.get(type(mod), [])
    mod = mod.eval().half()
    if x is None:
        x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    return to_edge(
        exported, compile_config=EdgeCompileConfig(preserve_ops=preserve)
    ).exported_program()


def _stub_fallbacks(monkeypatch, ops=()):
    monkeypatch.setattr(ap, "load_spyre_fallback_ops", lambda: (frozenset(ops), ""))


def _cpu_ops(mode, mod, x=None, fallbacks=()):
    """Base op names left on CPU after partitioning ``mod`` in ``mode``."""
    ep = _edge(mod, x)
    result = SpyrePartitioner(mode=mode).partition(ep)
    graph = result.tagged_exported_program.graph_module.graph
    return {
        _aten_name(n)
        for n in graph.nodes
        if n.op == "call_function"
        and _aten_name(n) is not None
        and n.meta.get("delegation_tag") is None
    }


# --- Mode resolution --------------------------------------------------------


def test_default_mode_is_fallbacklist(monkeypatch):
    monkeypatch.delenv(PARTITION_MODE_ENV_VAR, raising=False)
    assert resolve_partition_mode() == MODE_FALLBACKLIST


def test_env_var_selects_mode(monkeypatch):
    monkeypatch.setenv(PARTITION_MODE_ENV_VAR, " AllowList ")
    assert resolve_partition_mode() == MODE_ALLOWLIST


def test_explicit_mode_beats_env(monkeypatch):
    monkeypatch.setenv(PARTITION_MODE_ENV_VAR, "allowlist")
    assert resolve_partition_mode("fallbacklist") == MODE_FALLBACKLIST


@pytest.mark.parametrize("bad", ["", "deny", "allow-list", "FALLBACK"])
def test_unknown_mode_raises(bad):
    with pytest.raises(ValueError, match="unknown Spyre partition mode"):
        resolve_partition_mode(bad)


def test_unknown_mode_in_env_raises(monkeypatch):
    monkeypatch.setenv(PARTITION_MODE_ENV_VAR, "bogus")
    with pytest.raises(ValueError):
        SpyrePartitioner()


# --- Fail-closed ------------------------------------------------------------


def test_fallbacklist_fails_closed_without_fallback_ops(monkeypatch):
    monkeypatch.setattr(
        ap, "load_spyre_fallback_ops", lambda: (None, "torch_spyre not importable")
    )
    with pytest.raises(RuntimeError, match="needs torch-spyre's fallback_ops"):
        SpyrePartitioner(mode="fallbacklist")


def test_fallbacklist_default_fails_closed_without_fallback_ops(monkeypatch):
    # The production default (no mode, no env) must not degrade to the allowlist.
    monkeypatch.delenv(PARTITION_MODE_ENV_VAR, raising=False)
    monkeypatch.setattr(ap, "load_spyre_fallback_ops", lambda: (None, "boom"))
    with pytest.raises(RuntimeError):
        SpyrePartitioner()


def test_fallbacklist_fails_closed_when_torch_spyre_missing():
    # Un-stubbed: this host has no torch_spyre, so the real loader must report
    # (None, reason) and the partitioner must refuse to construct.
    fallbacks, reason = ap.load_spyre_fallback_ops()
    if fallbacks is not None:
        pytest.skip("torch_spyre is installed on this host")
    assert reason
    with pytest.raises(RuntimeError):
        SpyrePartitioner(mode="fallbacklist")


def test_allowlist_does_not_need_fallback_ops(monkeypatch):
    def boom():
        raise AssertionError("allowlist mode must not load torch-spyre fallback_ops")

    monkeypatch.setattr(ap, "load_spyre_fallback_ops", boom)
    assert SpyrePartitioner(mode="allowlist").mode == MODE_ALLOWLIST


# --- Both modes -------------------------------------------------------------


@pytest.mark.parametrize("mode", [MODE_ALLOWLIST, MODE_FALLBACKLIST])
def test_mlp_fully_delegated_in_both_modes(mode, monkeypatch):
    _stub_fallbacks(monkeypatch)
    assert _cpu_ops(mode, MLP()) == set()


# --- Disagreement between allowlist and fallback list -----------------------


def test_op_outside_yaml_is_cpu_in_allowlist_but_delegated_in_fallbacklist(monkeypatch):
    _stub_fallbacks(monkeypatch)  # torch-spyre declares no fallbacks
    assert _cpu_ops(MODE_ALLOWLIST, Sin()) == {"aten::sin"}
    assert _cpu_ops(MODE_FALLBACKLIST, Sin()) == set()


def test_op_in_fallback_ops_is_cpu_in_fallbacklist_but_delegated_in_allowlist(
    monkeypatch,
):
    _stub_fallbacks(monkeypatch, {"aten::relu"})  # relu is validated: true
    assert "aten::relu" in validated_op_names()
    assert _cpu_ops(MODE_ALLOWLIST, MLP()) == set()
    assert _cpu_ops(MODE_FALLBACKLIST, MLP()) == {"aten::relu"}


def test_fallback_ops_entries_are_normalized_to_base_names(monkeypatch):
    # fallback_ops mixes OpOverload objects, packets and plain strings; an
    # overload-qualified entry must still match the base name the node resolves to.
    assert ap._normalize_fallback_ops(
        [torch.ops.aten.sin.default, torch.ops.aten.cumsum, "spyre::triu_cpu", 7]
    ) == frozenset({"aten::sin", "aten::cumsum", "spyre::triu_cpu"})


def test_shadow_decisions_report_the_disagreement():
    yaml_ops = frozenset({"aten::mm"})
    spyre = frozenset({"aten::mm", "aten::sin"})
    assert shadow_decisions("aten::mm", yaml_ops, spyre) == ("delegate", "CPU")
    assert shadow_decisions("aten::sin", yaml_ops, spyre) == ("CPU", "CPU")
    assert shadow_decisions("aten::add", yaml_ops, spyre) == ("CPU", "delegate")


def test_shadow_report_counts_mismatches(monkeypatch):
    ep = _edge(Sin())
    report = ap.compute_shadow_report(
        ep.graph_module, validated_op_names(), frozenset()
    )
    # allowlist=CPU / spyre=delegate for aten::sin, and nothing else disagrees.
    assert [k[0] for k in report.mismatches] == ["aten::sin"]
    assert report.agree == report.total - 1


# --- Unvalidated and unsafe ops stay on CPU in BOTH modes -------------------

_UNSAFE_CASES = [
    # (id, module factory, input, op that must stay on CPU)
    ("embedding", Embedding, torch.zeros(1, 4, dtype=torch.int64), "aten::embedding"),
    ("rms_norm", RMSNorm, None, "aten::rms_norm"),
    ("native_layer_norm", LayerNorm, None, "aten::native_layer_norm"),
    ("innermost_cat", InnermostCat, None, "aten::cat"),
    ("slice_copy", Slice, None, "aten::slice_copy"),
    ("select_copy", Select, torch.randn(2, 64), "aten::select_copy"),
    ("to_copy", Cast, None, "dim_order_ops::_to_dim_order_copy"),
    ("softmax_fed_matmul", SoftmaxFedMatmul, None, "aten::addmm"),
    ("wide_output_writeback", WideOutput, None, "aten::addmm"),
]


@pytest.mark.parametrize("mode", [MODE_ALLOWLIST, MODE_FALLBACKLIST])
@pytest.mark.parametrize(
    ("factory", "x", "op"),
    [c[1:] for c in _UNSAFE_CASES],
    ids=[c[0] for c in _UNSAFE_CASES],
)
def test_unsafe_op_stays_on_cpu_in_both_modes(mode, factory, x, op, monkeypatch):
    # torch-spyre declares *no* fallbacks: in fallbacklist mode only this
    # repo's own rules can keep the op off the device.
    _stub_fallbacks(monkeypatch)
    if x is not None and x.is_floating_point():
        x = x.half()
    assert op in _cpu_ops(mode, factory(), x)


def test_unvalidated_set_matches_yaml_and_is_disjoint_from_validated():
    assert ap._UNVALIDATED_ATEN_OPS == all_op_names() - validated_op_names()
    assert ap._UNVALIDATED_ATEN_OPS.isdisjoint(ap._SUPPORTED_ATEN_OPS)
    # The ops #50 names explicitly.
    for op in (
        "aten::rms_norm",
        "aten::embedding",
        "aten::cat",
        "aten::slice_copy",
        "aten::select_copy",
        "aten::_to_copy",
        "aten::native_layer_norm",
    ):
        assert op in ap._UNVALIDATED_ATEN_OPS


def test_cpu_reason_names_the_rule(monkeypatch):
    _stub_fallbacks(monkeypatch)
    part = SpyrePartitioner(mode="fallbacklist")
    nodes = {
        _aten_name(n): n
        for n in _edge(RMSNorm()).graph_module.graph.nodes
        if n.op == "call_function"
    }
    assert part.op_support.cpu_reason(nodes["aten::rms_norm"]) == ap._REASON_UNVALIDATED
    assert part.op_support.cpu_reason(nodes["aten::add"]) is None


def test_zero_kernel_view_ops_validated_are_not_blocked_in_fallbacklist(monkeypatch):
    # The unvalidated rule must not swallow validated view ops: the vocab-style
    # MLP above stays fully delegated, and view_copy/permute_copy are validated.
    _stub_fallbacks(monkeypatch)
    assert {"aten::view_copy", "aten::permute_copy"} <= validated_op_names()
    assert _cpu_ops(MODE_FALLBACKLIST, MLP()) == set()


# --- Provenance of the fallback list ----------------------------------------


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ("0.5.0rc3", True),
        ("0.5.0-rc.3", True),
        ("v0.5.0rc3+gabc123", True),
        ("0.5.0rc1", False),
        ("0.6.0.dev112", False),
        ("unknown", False),
    ],
)
def test_provenance_version_match(monkeypatch, installed, expected):
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda _name: installed)
    prov = spyre_fallback_provenance(frozenset({"aten::sin"}))
    assert prov["torch_spyre"] == installed
    assert prov["version_matches"] is expected
    assert prov["supported_torch_spyre"] == SUPPORTED_TORCH_SPYRE


def test_provenance_digest_tracks_the_list():
    a = spyre_fallback_provenance(frozenset({"aten::sin", "aten::cos"}))
    b = spyre_fallback_provenance(frozenset({"aten::cos", "aten::sin"}))
    c = spyre_fallback_provenance(frozenset({"aten::sin"}))
    assert a["fallback_ops_digest"] == b["fallback_ops_digest"]
    assert a["fallback_ops_digest"] != c["fallback_ops_digest"]
    assert a["n_fallback_ops"] == 2


def test_version_mismatch_warns_and_is_recorded_in_report(monkeypatch, caplog):
    import importlib.metadata as md

    _stub_fallbacks(monkeypatch)
    monkeypatch.setattr(md, "version", lambda _name: "0.6.0.dev112")
    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        part = SpyrePartitioner(mode="fallbacklist")
    assert "0.6.0.dev112" in caplog.text and SUPPORTED_TORCH_SPYRE in caplog.text

    part.partition(_edge(MLP()))
    prov = part.last_report["fallback_provenance"]
    assert prov["version_matches"] is False
    assert prov["torch_spyre"] == "0.6.0.dev112"


def test_version_match_does_not_warn(monkeypatch, caplog):
    import importlib.metadata as md

    _stub_fallbacks(monkeypatch)
    monkeypatch.setattr(md, "version", lambda _name: "0.5.0rc3")
    with caplog.at_level(logging.WARNING, logger=ap.logger.name):
        SpyrePartitioner(mode="fallbacklist")
    assert caplog.text == ""


def test_allowlist_report_has_no_fallback_provenance():
    part = SpyrePartitioner(mode="allowlist")
    part.partition(_edge(MLP()))
    assert part.last_report["fallback_provenance"] is None


def test_supported_torch_spyre_matches_compatibility_manifest():
    manifest = json.loads((_REPO_ROOT / "compatibility-manifest.json").read_text())
    assert ap._normalize_version(manifest["torch_spyre"]) == ap._normalize_version(
        SUPPORTED_TORCH_SPYRE
    )
