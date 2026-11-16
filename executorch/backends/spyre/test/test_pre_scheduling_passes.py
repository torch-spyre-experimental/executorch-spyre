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
"""Host tests for ``spyre_pre_scheduling_passes`` (no torch-spyre, no device).

``CustomPreSchedulingPasses`` is replaced by fake classes that mirror the
structure of torch-spyre main (64a818de) and v0.5.0-rc.3 (46001de): a plain
class whose ``__init__`` assigns ``self.passes = [...]`` and whose ``uuid()``
is derived from that list. Real-class and frontend-only integration tests live
under ``test_spyre/``.
"""

import json
import threading
from unittest import mock

import pytest

pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.backends.spyre import spyre_pre_scheduling_passes as sps  # noqa: E402

pytestmark = pytest.mark.host

MAIN_NAMES = [
    "splice_while_loops",
    "deadcode_elimination",
    "propagate_named_dims",
    "validate_named_dims",
    "assign_dim_hints",
    "insert_bmm_padding",
    "split_multi_ops",
    "propagate_spyre_tensor_layouts",
    "validate_ops",
    "optimize_restickify_locations",
    "reorder_nonstick_dims",
    "finalize_layouts",
    "insert_restickify",
    "validate_no_restickify_on_mutation_targets",
    "enforce_indirect_access_layout",
    "reorder_nonstick_dims_mutation",
    "insert_post_mutation_restickify",
    "insert_restickify_padding",
    "dedup_and_promote_constants",
    "_maybe_coarse_tile_span_overflow",
    "plan_sharded_hbm",
    "span_reduction",
    "_distribute_work",
    "_maybe_scratchpad_planning",
    "elide_proven_read_copies",
]
RC3_NAMES = [
    "splice_while_loops",
    "deadcode_elimination",
    "propagate_named_dims",
    "validate_named_dims",
    "assign_dim_hints",
    "_maybe_reorder_unhinted_interlopers",
    "_maybe_coarse_tile_hints",
    "insert_bmm_padding",
    "split_multi_ops",
    "propagate_spyre_tensor_layouts",
    "reorder_nonstick_dims",
    "validate_ops",
    "optimize_restickify_locations",
    "finalize_layouts",
    "insert_restickify",
    "validate_no_restickify_on_mutation_targets",
    "enforce_indirect_access_layout",
    "insert_post_mutation_restickify",
    "insert_restickify_padding",
    "dedup_and_promote_constants",
    "_maybe_coarse_tile_span_overflow",
    "span_reduction",
    "_distribute_work",
    "_maybe_scratchpad_planning",
    "elide_proven_read_copies",
]


def _named(name):
    def fn(graph):
        graph.append(name)

    fn.__name__ = name
    return fn


def _get_pass_name(fn):
    # Same precedence as torch-spyre passes.py (main :122-133, rc.3 :112-).
    pattern_name = getattr(getattr(fn, "__self__", None), "pass_name", None)
    if isinstance(pattern_name, str) and pattern_name:
        return pattern_name
    if hasattr(fn, "__name__"):
        return fn.__name__
    if hasattr(fn, "__func__"):
        return fn.__func__.__name__
    return type(fn).__name__


def _make_fake(names):
    base = [_named(n) for n in names]

    class CustomPreSchedulingPasses:
        def __init__(self):
            self.passes = list(base)

        def __call__(self, graph):
            for p in self.passes:
                p(graph)

        def uuid(self):
            return tuple(_get_pass_name(p) for p in self.passes)

    return CustomPreSchedulingPasses


def _target(cls, reserved=("dedup_and_promote_constants",)):
    return sps._Target(
        cls, _get_pass_name, frozenset(reserved), lambda: "fake torch-spyre"
    )


@pytest.fixture(params=["main", "rc3"])
def fake(request, monkeypatch):
    names = MAIN_NAMES if request.param == "main" else RC3_NAMES
    cls = _make_fake(names)
    reserved = ("dedup_and_promote_constants",) if request.param == "main" else ()
    target = _target(cls, reserved)
    monkeypatch.setattr(sps, "_target_resolver", lambda: target)
    monkeypatch.setattr(sps, "_registry", {})
    monkeypatch.setattr(sps, "_cache_logged", False)
    cls.names = names
    return cls


def _order(cls):
    inst = cls()
    return [_get_pass_name(p) for p in inst.passes]


def _run(cls):
    trace = []
    cls()(trace)
    return trace


# ---------------------------------------------------------------- insertion


def test_after_and_before(fake):
    sps.register_pre_scheduling_pass(_named("p_after"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("p_before"), before="span_reduction")
    with sps.spyre_pass_scope(["p_after", "p_before"]):
        order = _order(fake)
    base = fake.names
    i = base.index("assign_dim_hints")
    j = base.index("span_reduction")
    expected = base[: i + 1] + ["p_after"] + base[i + 1 : j] + ["p_before"] + base[j:]
    assert order == expected
    with sps.spyre_pass_scope(["p_after"]):
        assert _run(fake)[i + 1] == "p_after"


def test_decorator_form_returns_fn(fake):
    @sps.register_pre_scheduling_pass(after="assign_dim_hints")
    def deco_pass(graph):
        graph.append("deco_pass")

    assert deco_pass.__name__ == "deco_pass"
    with sps.spyre_pass_scope(["deco_pass"]):
        assert "deco_pass" in _run(fake)


def test_name_override_uses_given_name(fake):
    def f(graph):
        graph.append("x")

    sps.register_pre_scheduling_pass(f, after="assign_dim_hints", name="renamed")
    with sps.spyre_pass_scope(["renamed"]):
        assert "renamed" in _order(fake)
        assert "f" not in _order(fake)
    assert [p.name for p in sps.list_pre_scheduling_passes()] == ["renamed"]


def test_chained_anchors_and_same_anchor_order(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("b"), after="a")
    sps.register_pre_scheduling_pass(_named("c"), before="a")
    sps.register_pre_scheduling_pass(_named("d"), after="assign_dim_hints")
    # list order is irrelevant: anchors decide
    with sps.spyre_pass_scope(["d", "b", "c", "a"]):
        order = _order(fake)
    i = order.index("assign_dim_hints")
    assert order[i : i + 5] == ["assign_dim_hints", "c", "a", "b", "d"]
    assert len(order) == len(fake.names) + 4


def test_anchor_to_unselected_registered_pass_errors(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("b"), after="a")
    with sps.spyre_pass_scope(["b"]):
        with pytest.raises(sps.PreSchedulingPassError, match="not selected"):
            fake()


def test_api_requires_exactly_one_anchor(fake):
    with pytest.raises(ValueError, match="exactly one"):
        sps.register_pre_scheduling_pass(_named("x"))
    with pytest.raises(ValueError, match="exactly one"):
        sps.register_pre_scheduling_pass(_named("x"), after="a", before="b")


def test_duplicate_registration_errors(fake):
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with pytest.raises(ValueError, match="already registered"):
        sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")


# ------------------------------------------------------------------ errors


def test_missing_anchor_lists_available(fake):
    sps.register_pre_scheduling_pass(_named("x"), after="no_such_pass")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError) as e:
            fake()
    msg = str(e.value)
    assert "'no_such_pass' for pass 'x' appears 0 times" in msg
    assert "assign_dim_hints" in msg and "fake torch-spyre" in msg


def test_anchor_only_in_other_version_is_missing(fake):
    # plan_sharded_hbm: main only; _maybe_coarse_tile_hints: rc.3 only.
    only = "plan_sharded_hbm" if "plan_sharded_hbm" in fake.names else None
    other = "_maybe_coarse_tile_hints"
    anchor = only or other
    absent = other if only else "plan_sharded_hbm"
    sps.register_pre_scheduling_pass(_named("ok"), after=anchor)
    sps.register_pre_scheduling_pass(_named("bad"), after=absent)
    with sps.spyre_pass_scope(["ok"]):
        assert "ok" in _order(fake)
    with sps.spyre_pass_scope(["bad"]):
        with pytest.raises(sps.PreSchedulingPassError, match="appears 0 times"):
            fake()


def test_ambiguous_anchor(fake):
    orig = fake.__init__

    def init(self):
        orig(self)
        self.passes.append(_named("assign_dim_hints"))

    fake.__init__ = init
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError, match="appears 2 times"):
            fake()


def test_name_collision_with_torch_spyre_pass(fake):
    sps.register_pre_scheduling_pass(
        _named("span_reduction"), name="span_reduction", after="assign_dim_hints"
    )
    with sps.spyre_pass_scope(["span_reduction"]):
        with pytest.raises(sps.PreSchedulingPassError, match="collides"):
            fake()


def test_name_collision_with_passes_run_expanded(fake):
    # _PASSES_RUN_EXPANDED (main) is name based; reserved even if the pass
    # list did not hold it.
    cls = _make_fake([n for n in fake.names if n != "dedup_and_promote_constants"])
    tgt = _target(cls, ("dedup_and_promote_constants",))
    sps.register_pre_scheduling_pass(
        _named("dedup_and_promote_constants"), after="assign_dim_hints"
    )
    with mock.patch.object(sps, "_target_resolver", lambda: tgt):
        with sps.spyre_pass_scope(["dedup_and_promote_constants"]):
            with pytest.raises(sps.PreSchedulingPassError, match="collides"):
                cls()


def test_anchor_registered_later_errors(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="b")
    sps.register_pre_scheduling_pass(_named("b"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["a", "b"]):
        with pytest.raises(sps.PreSchedulingPassError, match="registered later"):
            fake()


def test_structural_check_passes_not_a_list(fake):
    orig = fake.__init__

    def init(self):
        orig(self)
        self.passes = tuple(self.passes)

    fake.__init__ = init
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError, match="expected a list"):
            fake()


def test_structural_check_no_passes_attribute(fake):
    def init(self):
        pass

    fake.__init__ = init
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError, match="NoneType"):
            fake()


def test_structural_check_non_callable_entry(fake):
    orig = fake.__init__

    def init(self):
        orig(self)
        self.passes.append("not callable")

    fake.__init__ = init
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError, match="not callable"):
            fake()


def test_errors_include_version_info(fake):
    sps.register_pre_scheduling_pass(_named("x"), after="nope")
    with sps.spyre_pass_scope(["x"]):
        with pytest.raises(sps.PreSchedulingPassError, match=r"\[fake torch-spyre\]"):
            fake()


# ------------------------------------------------------------- name scoping


def test_unknown_scope_name_lists_registered(fake):
    sps.register_pre_scheduling_pass(_named("known"), after="assign_dim_hints")
    with pytest.raises(sps.PreSchedulingPassError) as e:
        with sps.spyre_pass_scope(["nope"]):
            pass
    assert "nope" in str(e.value) and "['known']" in str(e.value)


def test_duplicate_scope_names_error(fake):
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with pytest.raises(sps.PreSchedulingPassError, match="duplicate"):
        with sps.spyre_pass_scope(["x", "x"]):
            pass


def test_list_pre_scheduling_passes(fake):
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("y"), before="x")
    assert sps.list_pre_scheduling_passes() == (
        sps.PassInfo("x", "assign_dim_hints", "after"),
        sps.PassInfo("y", "x", "before"),
    )


# -------------------------------------------------------------- scope rules


def test_no_scope_is_identical_to_unpatched(fake):
    baseline = _order(fake)
    uuid0 = fake().uuid()
    sps.register_pre_scheduling_pass(_named("x"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["x"]):
        assert "x" in _order(fake)
        assert fake().uuid() != uuid0  # injected pass enters the key
    # wrapper stays installed (idempotent), but must be inert
    assert _order(fake) == baseline
    assert fake().uuid() == uuid0
    assert _run(fake) == baseline


def test_empty_scope_is_noop(fake):
    cls_init = fake.__init__
    with sps.spyre_pass_scope([]):
        assert fake.__init__ is cls_init  # not even installed
        assert _order(fake) == fake.names


def test_nested_scopes_union_and_restore(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("b"), after="finalize_layouts")
    with sps.spyre_pass_scope(["a"]):
        o1 = _order(fake)
        with sps.spyre_pass_scope(["b"]):
            o2 = _order(fake)
        assert _order(fake) == o1
    assert "a" in o1 and "b" not in o1
    assert "a" in o2 and "b" in o2
    assert _order(fake) == fake.names


def test_scope_restored_after_exception(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with pytest.raises(KeyError):
        with sps.spyre_pass_scope(["a"]):
            raise KeyError
    assert sps._ACTIVE.get() is None
    assert _order(fake) == fake.names


def test_concurrent_threads_with_different_scopes(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    sps.register_pre_scheduling_pass(_named("b"), after="assign_dim_hints")
    inside = threading.Barrier(3)
    results = {}

    errors = []

    def worker(key, names):
        try:
            with sps.spyre_pass_scope(names):
                inside.wait(timeout=10)  # both scopes open at once
                results[key] = _order(fake)
                inside.wait(timeout=10)
        except BaseException as e:  # a swallowed thread error must fail the test
            errors.append(e)

    ts = [
        threading.Thread(target=worker, args=("a", ["a"])),
        threading.Thread(target=worker, args=("b", ["b"])),
    ]
    for t in ts:
        t.start()
    inside.wait(timeout=10)
    # main thread: no scope, same moment
    results["none"] = _order(fake)
    inside.wait(timeout=10)
    for t in ts:
        t.join()
    assert errors == []
    assert "a" in results["a"] and "b" not in results["a"]
    assert "b" in results["b"] and "a" not in results["b"]
    assert results["none"] == fake.names


def test_new_thread_does_not_inherit_scope(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    out = {}
    with sps.spyre_pass_scope(["a"]):
        t = threading.Thread(target=lambda: out.update(o=_order(fake)))
        t.start()
        t.join()
    assert out["o"] == fake.names


def test_require_applied_detects_skipped_pipeline(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with pytest.raises(sps.PreSchedulingPassError, match="did not run"):
        with sps.spyre_pass_scope(["a"]) as scope:
            scope.require_applied()  # no pipeline was built
    with sps.spyre_pass_scope(["a"]) as scope:
        fake()
        scope.require_applied()


def test_require_applied_noop_for_empty_scope(fake):
    with sps.spyre_pass_scope([]) as scope:
        scope.require_applied()


# --------------------------------------------------------- patch mechanics


def test_install_is_idempotent(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["a"]):
        first = fake.__init__
    with sps.spyre_pass_scope(["a"]):
        with sps.spyre_pass_scope(["a"]):
            assert fake.__init__ is first
    assert fake.__init__ is first
    assert getattr(first, sps._WRAPPER_MARK) is first
    with sps.spyre_pass_scope(["a"]):
        assert _order(fake).count("a") == 1


def test_not_installed_at_import_or_before_first_scope(fake):
    orig = fake.__init__
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    assert fake.__init__ is orig
    with sps.spyre_pass_scope(["a"]):
        assert fake.__init__ is not orig


def _reassigning_init(cls):
    """Mimics torch-spyre's tests (test_loop_scheduler_review.py:381-391)."""
    real_init = cls.__init__

    def init(self):
        real_init(self)
        self.passes = [p for p in self.passes] + [_named("mock_extra")]

    return init


def test_coexistence_other_patch_applied_before_ours(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with mock.patch.object(fake, "__init__", _reassigning_init(fake)):
        with sps.spyre_pass_scope(["a"]):
            order = _order(fake)
        assert "a" in order and "mock_extra" in order
    # mock restored over our layer: still inert and still usable afterwards
    assert _order(fake) == fake.names
    with sps.spyre_pass_scope(["a"]):
        assert "a" in _order(fake) and "mock_extra" not in _order(fake)


def test_coexistence_other_patch_applied_after_ours(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["a"]):
        assert "a" in _order(fake)
    with mock.patch.object(fake, "__init__", _reassigning_init(fake)):
        # the mock sits above our wrapper and reassigns self.passes; a new
        # scope layers a fresh wrapper over it.
        with sps.spyre_pass_scope(["a"]):
            order = _order(fake)
        assert order.count("a") == 1 and "mock_extra" in order
        assert order.index("a") == order.index("assign_dim_hints") + 1
    assert _order(fake) == fake.names


def test_other_patch_swapped_in_while_scope_open_fails_loudly(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with pytest.raises(sps.PreSchedulingPassError, match="replaced by another"):
        with sps.spyre_pass_scope(["a"]) as scope:
            fake.__init__ = lambda self: setattr(self, "passes", [])  # replaced
            scope.require_applied()


def test_original_init_is_looked_up_each_time_not_captured_at_import(fake):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with sps.spyre_pass_scope(["a"]):
        pass
    # replace the class' __init__ wholesale (e.g. reload); we must re-wrap
    fake.__init__ = _make_fake(fake.names).__init__
    with sps.spyre_pass_scope(["a"]):
        assert "a" in _order(fake)


# --------------------------------------------------------------- CompileSpec


def test_compile_spec_roundtrip(fake):
    spec = sps.pass_compile_spec(["b", "a"])
    assert spec.key == "spyre_pre_scheduling_passes"
    assert isinstance(spec.value, bytes)
    assert json.loads(spec.value) == ["b", "a"]
    assert sps.parse_pass_names([spec]) == ("b", "a")


def test_parse_ignores_other_specs_and_absent_key(fake):
    other = sps.CompileSpec("spyre_dynamic_buckets", b"[1]")
    assert sps.parse_pass_names([other]) == ()
    assert sps.parse_pass_names([]) == ()
    assert sps.parse_pass_names(None) == ()


def test_duplicate_names_in_compile_spec_error(fake):
    with pytest.raises(sps.PreSchedulingPassError, match="duplicate"):
        sps.pass_compile_spec(["a", "a"])
    bad = sps.CompileSpec(sps.PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY, b'["a","a"]')
    with pytest.raises(sps.PreSchedulingPassError, match="duplicate"):
        sps.parse_pass_names([bad])


@pytest.mark.parametrize(
    "value", [b"not json", b'{"a": 1}', b"[1, 2]", b'"abc"', b"\xff"]
)
def test_malformed_compile_spec_errors(fake, value):
    spec = sps.CompileSpec(sps.PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY, value)
    with pytest.raises(sps.PreSchedulingPassError):
        sps.parse_pass_names([spec])


def test_two_pass_specs_error(fake):
    s = sps.pass_compile_spec(["a"])
    with pytest.raises(sps.PreSchedulingPassError, match="multiple"):
        sps.parse_pass_names([s, s])


# ----------------------------------------------------------- torch settings


def test_fx_compile_mode_guard(fake, monkeypatch):
    import torch._inductor.compile_fx as cfx

    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    for attr, value in (
        ("fx_compile_mode", cfx.FxCompileMode.SUBPROCESS),
        ("fx_compile_mode", cfx.FxCompileMode.SERIALIZE),
        ("fx_compile_async", True),
        ("fx_compile_progressive", True),
    ):
        with monkeypatch.context() as m:
            m.setattr(cfx, attr, value)
            with pytest.raises(sps.PreSchedulingPassError, match="in-process"):
                with sps.spyre_pass_scope(["a"]):
                    pass
            # empty scope never inspects the mode
            with sps.spyre_pass_scope([]):
                pass


def test_caches_disabled_inside_scope_and_restored(fake):
    import torch._functorch.config as fcfg
    import torch._inductor.config as icfg

    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    keys = [
        (icfg, "fx_graph_cache"),
        (icfg, "fx_graph_remote_cache"),
        (fcfg, "enable_autograd_cache"),
        (fcfg, "enable_remote_autograd_cache"),
    ]
    before = [getattr(c, k) for c, k in keys]
    with (
        icfg.patch({"fx_graph_cache": True}),
        fcfg.patch({"enable_autograd_cache": True}),
    ):
        with sps.spyre_pass_scope(["a"]):
            assert [getattr(c, k) for c, k in keys] == [False] * 4
            with sps.spyre_pass_scope(["a"]):  # nested: refcounted
                assert icfg.fx_graph_cache is False
            assert icfg.fx_graph_cache is False
        assert icfg.fx_graph_cache is True
        assert fcfg.enable_autograd_cache is True
    assert [getattr(c, k) for c, k in keys] == before


def test_empty_scope_leaves_caches_alone(fake):
    import torch._inductor.config as icfg

    with icfg.patch({"fx_graph_cache": True}):
        with sps.spyre_pass_scope([]):
            assert icfg.fx_graph_cache is True


def test_cache_disable_logged_once_at_info(fake, caplog):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    with caplog.at_level("INFO", logger=sps.logger.name):
        for _ in range(3):
            with sps.spyre_pass_scope(["a"]):
                pass
    msgs = [r for r in caplog.records if "disabling the" in r.getMessage()]
    assert len(msgs) == 1 and msgs[0].levelname == "INFO"
    assert "FX graph cache" in msgs[0].getMessage()


def test_second_compile_in_scope_runs_pass_again(fake):
    """Cache gate: a second 'compile' of the same graph is not served from the
    FX / AOTAutograd caches while a scope with passes is open.

    The real-compile version of this test is in test_spyre/ (frontend-only);
    here the gate is the torch config the cache lookup consults
    (compile_fx.py:1026-1032 and autograd_cache.py:137-140 in torch 2.x).
    """
    import torch._functorch.config as fcfg
    import torch._inductor.config as icfg

    runs = []

    def counting(graph):
        runs.append(1)

    sps.register_pre_scheduling_pass(counting, after="assign_dim_hints")

    def compile_once(graph):
        # What a cache-aware compile does: a hit skips building the pipeline.
        cached = icfg.fx_graph_cache and fcfg.enable_autograd_cache
        if not cached:
            fake()(graph)

    with (
        icfg.patch({"fx_graph_cache": True}),
        fcfg.patch({"enable_autograd_cache": True}),
    ):
        with sps.spyre_pass_scope(["counting"]):
            compile_once([])
            compile_once([])
        assert len(runs) == 2
        compile_once([])  # outside the scope the cache is back on
        assert len(runs) == 2


# ------------------------------------------------- SpyreBackend.preprocess


@pytest.fixture
def stub_torch_spyre(monkeypatch):
    """Stub the torch_spyre modules ``preprocess`` imports, so it runs on host."""
    import sys
    import types

    pkg = types.ModuleType("torch_spyre")
    pkg._autoload = lambda: None
    execution = types.ModuleType("torch_spyre.execution")
    ac = types.ModuleType("torch_spyre.execution.async_compile")
    kr = types.ModuleType("torch_spyre.execution.kernel_runner")
    pkg.execution, execution.async_compile, execution.kernel_runner = execution, ac, kr
    for name, mod in (
        ("torch_spyre", pkg),
        ("torch_spyre.execution", execution),
        ("torch_spyre.execution.async_compile", ac),
        ("torch_spyre.execution.kernel_runner", kr),
    ):
        monkeypatch.setitem(sys.modules, name, mod)


def _stub_edge():
    import types

    graph = types.SimpleNamespace(nodes=[])
    return types.SimpleNamespace(graph_module=types.SimpleNamespace(graph=graph))


def _preprocess_with_recorder(monkeypatch, compile_specs, fake_cls=None):
    from executorch.backends.spyre import spyre_backend as backend

    seen = {}

    def impl(edge_program, specs, cfx, ac, kr):
        seen["scope"] = sps._ACTIVE.get()
        seen["specs"] = specs
        if fake_cls is not None:
            seen["order"] = _order(fake_cls)
        return "RESULT"

    monkeypatch.setattr(backend.SpyreBackend, "_preprocess_impl", staticmethod(impl))
    out = backend.SpyreBackend.preprocess(_stub_edge(), compile_specs)
    return out, seen


def test_preprocess_without_spec_enters_no_scope(fake, monkeypatch, stub_torch_spyre):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    orig_init = fake.__init__
    out, seen = _preprocess_with_recorder(monkeypatch, [], fake)
    assert out == "RESULT"
    assert seen["scope"] is None
    assert seen["order"] == fake.names
    assert fake.__init__ is orig_init  # nothing patched: behaviour as before
    # other (e.g. bucket) specs alone also leave it untouched
    other = sps.CompileSpec("spyre_dynamic_buckets", b"[1]")
    _, seen = _preprocess_with_recorder(monkeypatch, [other], fake)
    assert seen["scope"] is None and seen["specs"] == [other]
    assert fake.__init__ is orig_init


def test_preprocess_with_spec_runs_impl_inside_scope(
    fake, monkeypatch, stub_torch_spyre
):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")

    from executorch.backends.spyre import spyre_backend as backend

    def impl(edge_program, specs, cfx, ac, kr):
        fake()  # what the compile does: build the pipeline under the scope
        return "RESULT"

    monkeypatch.setattr(backend.SpyreBackend, "_preprocess_impl", staticmethod(impl))
    spec = sps.pass_compile_spec(["a"])
    assert backend.SpyreBackend.preprocess(_stub_edge(), [spec]) == "RESULT"
    assert sps._ACTIVE.get() is None  # scope left again
    assert _order(fake) == fake.names


def test_preprocess_with_spec_fails_if_pipeline_never_built(
    fake, monkeypatch, stub_torch_spyre
):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    spec = sps.pass_compile_spec(["a"])
    with pytest.raises(sps.PreSchedulingPassError, match="did not run"):
        _preprocess_with_recorder(monkeypatch, [spec])


def test_preprocess_unknown_pass_name_lists_registered(
    fake, monkeypatch, stub_torch_spyre
):
    sps.register_pre_scheduling_pass(_named("a"), after="assign_dim_hints")
    spec = sps.CompileSpec(sps.PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY, b'["zzz"]')
    with pytest.raises(sps.PreSchedulingPassError, match=r"zzz.*\['a'\]"):
        _preprocess_with_recorder(monkeypatch, [spec])
