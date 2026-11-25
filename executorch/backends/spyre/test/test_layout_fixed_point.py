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
"""Host tests for the kv_state layout fixed-point loop (``_iterate_input_layouts``).

No device and no torch-spyre: ``_capture_one_compile`` is replaced by a fake
kernel that picks each state-output layout from a policy, and spyre tensors are
replaced by a stand-in that only carries a layout dict. That is enough to drive
the loop's real control flow (convergence check, rebuild, iteration budget).
"""

import json
import logging
import re
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

import executorch.backends.spyre.spyre_backend as ab  # noqa: E402

KV_SHAPE = [1, 8, 512, 128]

# The Granite decode layouts: K's fixed point is the default layout, V's is the
# default with device dims 1 and 3 swapped. The default upload carries
# valid_elements while the compiler's outputs do not, so the convergence check
# must be blind to valid_elements (as in the real runs).
_STRIDES = [524288, 65536, 128, 1]
DEFAULT_IN = {
    "device_size": [8, 512, 2, 1, 64],
    "stride_map": [65536, 128, 64, -1, 1],
    "device_dtype": 0,
    "element_arrangement": 0,
    "valid_elements": [[[0], 8], [[1], 512], [[2, 4], 128], [[3], 1]],
    "strides": _STRIDES,
}
DEFAULT_OUT = {**DEFAULT_IN, "valid_elements": []}
SWAPPED = {
    **DEFAULT_OUT,
    "device_size": [8, 1, 2, 512, 64],
    "stride_map": [65536, -1, 64, 128, 1],
}
OTHER = {
    **DEFAULT_OUT,
    "device_size": [8, 2, 1, 512, 64],
    "stride_map": [65536, 64, -1, 128, 1],
}


class _T:
    """Stand-in for a spyre tensor: only the layout and dtype matter here."""

    dtype = "bfloat16"

    def __init__(self, layout):
        self.layout = layout


class _Graph:
    """A fake delegate: optional user inputs first, then kv_state placeholders."""

    def __init__(self, state_names, user_inputs=("tokens",)):
        names = list(user_inputs) + list(state_names)
        self.names = names
        self.state_names = list(state_names)  # state_out j <-> state_names[j]
        self.gm = types.SimpleNamespace(
            graph=types.SimpleNamespace(
                nodes=[types.SimpleNamespace(op="placeholder", name=n) for n in names]
            )
        )
        self.example_inputs, self.input_specs = self.fresh()

    def fresh(self):
        """New (example_inputs, input_specs), as ``_make_spyre_example_inputs`` makes."""
        example_inputs, input_specs = [], []
        for n in self.names:
            if n in self.state_names:
                example_inputs.append(_T(DEFAULT_IN))
                input_specs.append(
                    {
                        "shape": list(KV_SHAPE),
                        "dtype": "bfloat16",
                        "kind": "kv_state",
                        "state_out": self.state_names.index(n),
                    }
                )
            else:
                example_inputs.append(_T(None))
                input_specs.append(
                    {
                        "shape": [1, 1],
                        "dtype": "int64",
                        "kind": "user_input",
                        "state_out": -1,
                    }
                )
        return example_inputs, input_specs

    def idx(self, name):
        return self.names.index(name)

    def spec(self, name):
        return self.input_specs[self.idx(name)]


class _Kernel:
    """Fake ``_capture_one_compile``.

    ``policy(name, n_call, in_layout)`` returns the state-output layout the
    kernel writes for the state buffer ``name``; ``n_call`` counts compiles of
    this kernel from 0 and ``in_layout`` is the layout it was compiled with.
    """

    def __init__(self, graph, policy):
        self.graph = graph
        self.policy = policy
        self.calls = []  # per compile: {name: input layout}

    def __call__(self, gm, iter_inputs, cfx, ac, kr):
        g = self.graph
        ins = {n: iter_inputs[g.idx(n)].layout for n in g.state_names}
        n_call = len(self.calls)
        self.calls.append(ins)
        output_specs = [
            {
                "shape": list(KV_SHAPE),
                "dtype": "bfloat16",
                "device_layout": self.policy(n, n_call, ins[n]),
            }
            for n in g.state_names
        ]
        output_specs.append({"shape": [1, 1, 49159], "dtype": "float16"})  # logits
        return [], [], {}, {}, output_specs


def _granite_policy(name, n_call, in_layout):
    """K ends at the default layout, V at the swapped one, whatever the input."""
    return SWAPPED if "vc" in name else DEFAULT_OUT


@pytest.fixture(autouse=True)
def _clean_seed_cache():
    ab._reset_layout_seed_cache()
    yield
    ab._reset_layout_seed_cache()


@pytest.fixture
def fake_compile(monkeypatch):
    """Patch the device-bound helpers; return ``install(graph, policy) -> _Kernel``."""
    monkeypatch.setattr(ab, "_device_layout_spec", lambda t: t.layout)
    monkeypatch.setattr(
        ab, "_spyre_tensor_with_layout", lambda layout, shape, dtype: _T(layout)
    )

    def install(graph, policy=_granite_policy):
        kernel = _Kernel(graph, policy)
        monkeypatch.setattr(ab, "_capture_one_compile", kernel)
        return kernel

    return install


def _run(graph, **kw):
    return ab._iterate_input_layouts(
        graph.gm, graph.example_inputs, graph.input_specs, None, None, None, **kw
    )


def test_unseeded_delegate_takes_two_passes(fake_compile):
    g = _Graph(["b__inner_kc0", "b__inner_vc0"])
    kernel = fake_compile(g)
    _run(g)
    assert len(kernel.calls) == 2
    # Pass 1 compiles the default upload; pass 2 compiles V at its fixed point.
    assert kernel.calls[0] == {"b__inner_kc0": DEFAULT_IN, "b__inner_vc0": DEFAULT_IN}
    assert kernel.calls[1] == {"b__inner_kc0": DEFAULT_IN, "b__inner_vc0": SWAPPED}
    assert g.spec("b__inner_kc0")["device_layout"] == DEFAULT_OUT
    assert g.spec("b__inner_vc0")["device_layout"] == SWAPPED
    assert "device_layout" not in g.spec("tokens")


def test_default_fixed_point_compiles_once(fake_compile):
    g = _Graph(["b__inner_kc0", "b__inner_vc0"])
    kernel = fake_compile(g, lambda name, n_call, in_layout: DEFAULT_OUT)
    _run(g)
    assert len(kernel.calls) == 1


def test_non_convergence_raises_after_exactly_max_iters(fake_compile):
    g = _Graph(["b__inner_kc0", "b__inner_vc0"])

    def never_settles(name, n_call, in_layout):
        if "vc" not in name:
            return DEFAULT_OUT
        return SWAPPED if n_call % 2 == 0 else OTHER

    kernel = fake_compile(g, never_settles)
    with pytest.raises(RuntimeError, match="did not converge"):
        _run(g, max_iters=3)
    assert len(kernel.calls) == 3


def test_each_pass_logs_number_and_wall_time(fake_compile, caplog):
    g = _Graph(["b__inner_kc0", "b__inner_vc0"])
    fake_compile(g)
    with caplog.at_level(logging.INFO, logger=ab.logger.name):
        _run(g)
    lines = [
        r.getMessage() for r in caplog.records if "input-layout pass" in r.getMessage()
    ]
    assert len(lines) == 2
    assert re.search(
        r"pass 1/4: 2 kv_state input\(s\), 1 mismatched, compile \d+\.\ds", lines[0]
    )
    assert re.search(
        r"pass 2/4: 2 kv_state input\(s\), 0 mismatched, compile \d+\.\ds", lines[1]
    )


# --- seed cache ----------------------------------------------------------------

K0, V0, K1, V1 = "b__inner_kc0", "b__inner_vc0", "b__inner_kc1", "b__inner_vc1"


def _v_key(name="b__inner_vc#"):
    return (tuple(KV_SHAPE), "bfloat16", ab._layout_key(DEFAULT_IN), name)


def test_cache_reset_between_tests_starts_empty():
    assert ab._layout_seed_cache == {}


def test_state_names_are_normalised_across_layers():
    assert ab._normalize_state_name("b__inner_kc0") == "b__inner_kc#"
    assert ab._normalize_state_name("b__inner_vc13") == "b__inner_vc#"
    assert ab._normalize_state_name("b_kc1") == "b_kc#"


def test_converged_v_is_cached_and_k_is_not(fake_compile):
    g = _Graph([K0, V0])
    fake_compile(g)
    _run(g)
    # Only V mismatched at pass 1, so only V is cached (keyed by the layer-free name).
    assert list(ab._layout_seed_cache) == [_v_key()]
    assert ab._layout_seed_cache[_v_key()] == SWAPPED


def test_seed_hit_converges_in_one_compile(fake_compile):
    first = _Graph([K0, V0])
    fake_compile(first)
    _run(first)

    # Another layer of the same model, another delegate.
    second = _Graph([K1, V1])
    kernel = fake_compile(second)
    _run(second)
    assert len(kernel.calls) == 1
    assert kernel.calls[0] == {K1: DEFAULT_IN, V1: SWAPPED}
    assert second.spec(K1)["device_layout"] == DEFAULT_OUT
    assert second.spec(V1)["device_layout"] == SWAPPED


def test_input_that_converged_at_pass_1_is_never_seeded(fake_compile):
    g = _Graph([K0, V0])
    fake_compile(g)
    _run(g)
    # K converged at pass 1 in the first run, so nothing was stored for it.
    second = _Graph([K1, V1])
    kernel = fake_compile(second)
    _run(second)
    assert kernel.calls[0][K1] == DEFAULT_IN
    assert _v_key("b__inner_kc#") not in ab._layout_seed_cache
    assert all("kc" not in key[3] for key in ab._layout_seed_cache)


def test_wrong_seed_is_rejected_and_converges(fake_compile):
    ab._layout_seed_cache[_v_key()] = OTHER  # wrong for this kernel
    g = _Graph([K0, V0])
    kernel = fake_compile(g)
    _run(g)
    # Pass 1 ran with the wrong seed, pass 2 with the real fixed point: the
    # rejected seed costs one pass of the budget, no refund.
    assert len(kernel.calls) == 2
    assert kernel.calls[0][V0] == OTHER
    assert kernel.calls[1][V0] == SWAPPED
    assert g.spec(V0)["device_layout"] == SWAPPED
    # The bad entry was replaced by the converged layout.
    assert ab._layout_seed_cache[_v_key()] == SWAPPED


def test_rejecting_a_seed_keeps_the_good_ones(fake_compile):
    other_shape = [1, 4, 512, 128]
    good_key = (tuple(other_shape), "bfloat16", ab._layout_key(DEFAULT_IN), "b_good#")
    ab._layout_seed_cache[good_key] = SWAPPED
    ab._layout_seed_cache[_v_key()] = OTHER
    g = _Graph([K0, V0])
    fake_compile(g)
    _run(g)
    assert ab._layout_seed_cache[good_key] == SWAPPED


def test_one_wrong_seed_does_not_unseed_the_right_one(fake_compile):
    # Two V-like buffers: one cache entry right, one wrong. The right one keeps
    # its seed on the recompile instead of restarting from the default.
    g = _Graph([K0, "b__inner_vc0", "b__inner_wc0"])

    def policy(name, n_call, in_layout):
        return DEFAULT_OUT if "kc" in name else SWAPPED

    kernel = fake_compile(g, policy)
    ab._layout_seed_cache[_v_key("b__inner_vc#")] = SWAPPED
    ab._layout_seed_cache[_v_key("b__inner_wc#")] = OTHER
    _run(g)
    assert len(kernel.calls) == 2
    assert kernel.calls[0]["b__inner_vc0"] == SWAPPED
    assert kernel.calls[1]["b__inner_vc0"] == SWAPPED
    assert kernel.calls[1]["b__inner_wc0"] == SWAPPED
    assert ab._layout_seed_cache[_v_key("b__inner_vc#")] == SWAPPED
    assert ab._layout_seed_cache[_v_key("b__inner_wc#")] == SWAPPED


def _never_settles(name, n_call, in_layout):
    if "vc" not in name:
        return DEFAULT_OUT
    return SWAPPED if n_call % 2 == 0 else OTHER


def test_non_convergence_with_a_seed_raises_after_exactly_max_iters(fake_compile):
    ab._layout_seed_cache[_v_key()] = SWAPPED
    g = _Graph([K0, V0])

    def disagrees_with_the_seed(name, n_call, in_layout):
        if "vc" not in name:
            return DEFAULT_OUT
        return OTHER if n_call % 2 == 0 else SWAPPED

    kernel = fake_compile(g, disagrees_with_the_seed)
    with pytest.raises(RuntimeError, match="did not converge") as ei:
        _run(g, max_iters=3)
    # The rejected seed's pass counts: no refund.
    assert len(kernel.calls) == 3
    assert kernel.calls[0][V0] == SWAPPED
    # The message tells the truth about the seed.
    assert "Seeded from the layout cache: ['b__inner_vc0']" in str(ei.value)
    assert "rejected at pass 1: ['b__inner_vc0']" in str(ei.value)
    # The rejected entry is gone and nothing was stored from a failed run.
    assert ab._layout_seed_cache == {}


def test_non_convergence_without_a_seed_stores_nothing(fake_compile):
    g = _Graph([K0, V0])
    kernel = fake_compile(g, _never_settles)
    with pytest.raises(RuntimeError, match="did not converge") as ei:
        _run(g, max_iters=3)
    assert len(kernel.calls) == 3
    assert "Seeded from the layout cache: []" in str(ei.value)
    assert ab._layout_seed_cache == {}


def test_seed_without_a_state_output_layout_is_reverted(fake_compile, monkeypatch):
    # The kernel reports no layout for V's state output: the seed cannot be
    # checked, so the input goes back to the default upload (which is what the
    # runtime feeds when no device_layout is recorded).
    ab._layout_seed_cache[_v_key()] = SWAPPED
    g = _Graph([K0, V0])
    kernel = fake_compile(g)

    def no_v_layout(gm, iter_inputs, cfx, ac, kr):
        out = list(kernel(gm, iter_inputs, cfx, ac, kr))
        out[4][g.state_names.index(V0)].pop("device_layout")
        return tuple(out)

    monkeypatch.setattr(ab, "_capture_one_compile", no_v_layout)
    _run(g)
    assert len(kernel.calls) == 2
    assert kernel.calls[0][V0] == SWAPPED
    assert kernel.calls[1][V0] == DEFAULT_IN
    assert "device_layout" not in g.spec(V0)
    assert _v_key() not in ab._layout_seed_cache


def test_pass_log_lists_seeded_inputs(fake_compile, caplog):
    ab._layout_seed_cache[_v_key()] = SWAPPED
    g = _Graph([K0, V0])
    fake_compile(g)
    with caplog.at_level(logging.INFO, logger=ab.logger.name):
        _run(g)
    (line,) = [
        r.getMessage() for r in caplog.records if "input-layout pass" in r.getMessage()
    ]
    assert "pass 1/4" in line and "0 mismatched" in line
    assert "seeded ['b__inner_vc0']" in line and "seed rejected []" in line


def test_graph_without_placeholder_names_does_not_use_the_cache(fake_compile):
    g = _Graph([K0, V0])
    kernel = fake_compile(g)
    ab._layout_seed_cache[_v_key()] = SWAPPED
    g.gm = None  # e.g. the host fakes in test_preprocess
    _run(g)
    assert len(kernel.calls) == 2  # no seed: default, then fixed point
    assert kernel.calls[0][V0] == DEFAULT_IN


def test_placeholder_count_mismatch_warns_with_both_counts(fake_compile, caplog):
    g = _Graph([K0, V0])
    kernel = fake_compile(g)
    ab._layout_seed_cache[_v_key()] = SWAPPED
    g.gm.graph.nodes.pop()  # 2 placeholders for 3 input specs
    with caplog.at_level(logging.DEBUG, logger=ab.logger.name):
        _run(g)
    (rec,) = [r for r in caplog.records if "seed cache skipped" in r.getMessage()]
    assert rec.levelno == logging.WARNING
    assert "2 graph placeholders but 3 input specs" in rec.getMessage()
    assert kernel.calls[0][V0] == DEFAULT_IN  # and the cache really was not used


def test_missing_graph_logs_at_debug_not_warning(fake_compile, caplog):
    g = _Graph([K0, V0])
    fake_compile(g)
    g.gm = None
    with caplog.at_level(logging.DEBUG, logger=ab.logger.name):
        _run(g)
    (rec,) = [r for r in caplog.records if "seed cache skipped" in r.getMessage()]
    assert rec.levelno == logging.DEBUG


def test_matching_graph_logs_no_skip_message(fake_compile, caplog):
    g = _Graph([K0, V0])
    fake_compile(g)
    with caplog.at_level(logging.DEBUG, logger=ab.logger.name):
        _run(g)
    assert not [r for r in caplog.records if "seed cache skipped" in r.getMessage()]


# --- bucketed path -------------------------------------------------------------


def test_bucketed_path_shares_the_cache_across_buckets(
    fake_compile, monkeypatch, tmp_path
):
    g = _Graph([K0, V0])
    kernel = fake_compile(g)
    code_dir = tmp_path / "spyreCodeDir"
    code_dir.mkdir()
    cfx = ac = kr = None

    def make_inputs(edge, shape_overrides=None):
        example_inputs, input_specs = g.fresh()
        return example_inputs, input_specs, [{"shape": [1], "dtype": "float32"}]

    monkeypatch.setattr(ab, "_retarget_edge_to_aten", lambda gm: 0)
    monkeypatch.setattr(ab, "_make_spyre_example_inputs", make_inputs)
    monkeypatch.setattr(ab, "_resolve_dynamic_dims", lambda ep, gm, arg_dims: [(0, 0)])
    monkeypatch.setattr(ab, "_build_v2_layout", lambda runs, i, o: ([], [], 0))
    monkeypatch.setattr(ab, "_spyrecode_dir", lambda d: str(code_dir))
    # The fake kernel emits one dir per compile; _remove_dirs ignores missing ones.
    real = kernel.__call__

    def with_dir(gm, iter_inputs, cfx, ac, kr):
        out = list(real(gm, iter_inputs, cfx, ac, kr))
        out[0] = [str(tmp_path / "never_created")]
        return tuple(out)

    monkeypatch.setattr(ab, "_capture_one_compile", with_dir)

    spec = [
        types.SimpleNamespace(
            key="spyre_dynamic_buckets",
            value=json.dumps({"arg_dims": [{"arg": 0, "axis": 0}], "buckets": [1, 2]}),
        )
    ]
    edge = types.SimpleNamespace(graph_module=g.gm)
    result = ab.SpyreBackend._preprocess_impl(edge, spec, cfx, ac, kr)
    assert result.processed_bytes
    # Bucket 0 needs 2 compiles; bucket 1 is seeded from it and needs 1.
    assert len(kernel.calls) == 3
    assert kernel.calls[2][V0] == SWAPPED
    assert ab._layout_seed_cache[_v_key()] == SWAPPED
