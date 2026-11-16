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
"""Pre-scheduling pass framework against the REAL torch-spyre (no device).

Integration-tested on torch-spyre main only. rc.3 (46001de) is "structurally
compatible, not integration-tested": it has no ``_flat_operations`` / ``OpsView``
and is covered by the fake-class host tests. These tests skip, with a reason,
when torch-spyre is missing or is not a main-style release.

Compiles run with ``TORCH_SPYRE_FRONTEND_ONLY`` (config ``frontend_only``): the
whole frontend, including the pass pipeline, runs and the backend compiler is
skipped. The resulting kernel launch raises the documented error
(``kernel_runner.py:83-89`` on main); any other error fails the test.
"""

import contextlib
import importlib.util
import logging
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")
pytest.importorskip("torch_spyre", reason="torch_spyre is not importable")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

ts_passes = pytest.importorskip("torch_spyre._inductor.passes")
if not hasattr(ts_passes, "_flat_operations"):
    pytest.skip(
        "torch-spyre without _flat_operations/OpsView (e.g. v0.5.0-rc.3): "
        "structurally compatible, covered by the fake-class host tests",
        allow_module_level=True,
    )

import torch._functorch.config as functorch_config  # noqa: E402
import torch._inductor.compile_fx as cfx  # noqa: E402
import torch._inductor.config as inductor_config  # noqa: E402
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from torch_spyre._inductor import config as spyre_config  # noqa: E402
from torch_spyre._inductor.pass_utils import format_operations  # noqa: E402

from executorch.backends.spyre import spyre_pre_scheduling_passes as sps  # noqa: E402
from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner  # noqa: E402
from executorch.backends.spyre.test.test_preprocess import MLP  # noqa: E402

pytestmark = pytest.mark.spyre_frontend

ANCHOR = "assign_dim_hints"
PASS_NAME = "et_probe_pass"
FRONTEND_ONLY_MSG = "TORCH_SPYRE_FRONTEND_ONLY=1"


@pytest.fixture(autouse=True)
def frontend_only_env(monkeypatch):
    """Frontend-only, one CP-SAT worker (LX allocation is only deterministic
    with one), and every cache that could skip the pipeline off, so the
    with/without comparison below compares two real pipeline runs."""
    monkeypatch.setattr(sps, "_registry", {})
    with contextlib.ExitStack() as stack:
        stack.enter_context(spyre_config.patch({"frontend_only": True, "num_cpus": 1}))
        stack.enter_context(
            inductor_config.patch(
                {"fx_graph_cache": False, "fx_graph_remote_cache": False}
            )
        )
        stack.enter_context(functorch_config.patch({"enable_autograd_cache": False}))
        yield


class Recorder:
    """Counts compile_fx calls, pass runs per call, pass lists, post-pipeline IR."""

    def __init__(self):
        self.compile_calls = 0
        self.pass_runs = []  # compile_fx call index (1-based) per pass run
        self.pass_lists = []  # ordered pass names per pipeline run
        self.ir = []  # per pipeline run: (op names, formatted IR, device layouts)

    def probe_pass(self, graph):
        # Read-only: get_read_writes(), never op_read_writes (memoizes).
        for op in graph.operations:
            op.get_read_writes()
        self.pass_runs.append(self.compile_calls)


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()

    real_compile_fx = cfx.compile_fx

    def counting_compile_fx(*args, **kwargs):
        rec.compile_calls += 1
        return real_compile_fx(*args, **kwargs)

    monkeypatch.setattr(cfx, "compile_fx", counting_compile_fx)

    real_call = ts_passes.CustomPreSchedulingPasses.__call__

    def recording_call(self, graph):
        rec.pass_lists.append([ts_passes._get_pass_name(p) for p in self.passes])
        real_call(self, graph)
        ops = list(ts_passes.all_ops(graph.operations))
        rec.ir.append(
            (
                [op.get_operation_name() for op in ops],
                format_operations(graph.operations),
                [
                    repr(getattr(getattr(op, "layout", None), "device_layout", None))
                    for op in ops
                ],
            )
        )

    monkeypatch.setattr(ts_passes.CustomPreSchedulingPasses, "__call__", recording_call)

    def et_probe_pass(graph):
        rec.probe_pass(graph)

    et_probe_pass.__name__ = PASS_NAME
    sps.register_pre_scheduling_pass(et_probe_pass, after=ANCHOR)
    return rec


def _lower(compile_specs):
    mod = MLP().eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    try:
        to_edge_transform_and_lower(
            exported, partitioner=[SpyrePartitioner(compile_specs=compile_specs)]
        )
    except Exception as e:
        # The kernel launch in frontend-only mode raises this; nothing else is OK.
        chain, cur = [], e
        while cur is not None and cur not in chain:
            chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        if not any(FRONTEND_ONLY_MSG in str(c) for c in chain):
            raise
    else:  # pragma: no cover
        pytest.fail("expected the frontend-only kernel launch error, got none")


def test_pass_runs_once_per_compile_at_requested_position(recorder):
    _lower([sps.pass_compile_spec([PASS_NAME])])
    assert recorder.compile_calls >= 1
    # exactly once per compile_fx call (per GraphLowering), not per delegate
    assert sorted(recorder.pass_runs) == list(range(1, recorder.compile_calls + 1))
    assert len(recorder.pass_lists) == recorder.compile_calls
    for names in recorder.pass_lists:
        assert names.count(PASS_NAME) == 1
        assert names.index(PASS_NAME) == names.index(ANCHOR) + 1


def test_compile_without_spec_does_not_run_pass(recorder):
    _lower([])
    assert recorder.compile_calls >= 1
    assert recorder.pass_runs == []
    assert all(PASS_NAME not in names for names in recorder.pass_lists)


def test_result_identical_with_and_without_pass(recorder):
    _lower([])
    baseline_ir, baseline_lists = list(recorder.ir), list(recorder.pass_lists)
    assert baseline_ir and all(
        ops and text and layouts for ops, text, layouts in baseline_ir
    )
    assert any(lay != "None" for _, _, layouts in baseline_ir for lay in layouts)
    recorder.ir.clear()
    recorder.pass_lists.clear()
    _lower([sps.pass_compile_spec([PASS_NAME])])
    assert recorder.pass_runs  # it did run this time
    # pass list aside, the compiled result is the same
    assert recorder.ir == baseline_ir
    assert [
        [n for n in names if n != PASS_NAME] for names in recorder.pass_lists
    ] == baseline_lists


def test_scope_leaves_nothing_behind(recorder):
    _lower([sps.pass_compile_spec([PASS_NAME])])
    runs = len(recorder.pass_runs)
    assert sps._ACTIVE.get() is None
    _lower([])  # wrapper stays installed but must be inert
    assert len(recorder.pass_runs) == runs


def test_recompile_inside_scope_still_runs_pass(recorder):
    """A re-compile inside a scope still runs the pass, with the FX graph and
    AOTAutograd caches ON outside the scope.

    This does NOT prove a prevented cache hit: a negative control (scope's
    cache-disabling removed) still passes, and no cache hit was reproduced on
    this path. The test shows the pass runs on every compile and that the cache
    settings are restored. That the scope switches the cache gates off (and
    back on) is asserted in
    test/test_pre_scheduling_passes.py::test_caches_disabled_inside_scope_and_restored.
    """
    with (
        inductor_config.patch({"fx_graph_cache": True}),
        functorch_config.patch({"enable_autograd_cache": True}),
    ):
        spec = [sps.pass_compile_spec([PASS_NAME])]
        _lower(spec)
        first = len(recorder.pass_runs)
        _lower(spec)
        assert first >= 1
        assert len(recorder.pass_runs) == 2 * first
        assert inductor_config.fx_graph_cache is True  # restored after the scope


def test_real_class_uuid_and_pass_list_in_and_out_of_scope(recorder):
    """Real CustomPreSchedulingPasses: out of scope the pass list and uuid() are
    unpatched; in scope the injected pass is present and changes uuid()."""
    cls = ts_passes.CustomPreSchedulingPasses
    base = cls()
    base_names = [ts_passes._get_pass_name(p) for p in base.passes]
    base_uuid = base.uuid()
    assert PASS_NAME not in base_names

    with sps.spyre_pass_scope([PASS_NAME]):
        inst = cls()
        names = [ts_passes._get_pass_name(p) for p in inst.passes]
        assert names.count(PASS_NAME) == 1
        assert names.index(PASS_NAME) == names.index(ANCHOR) + 1
        assert [n for n in names if n != PASS_NAME] == base_names
        assert inst.uuid() != base_uuid

    after = cls()  # wrapper stays installed; must be inert out of scope
    assert [ts_passes._get_pass_name(p) for p in after.passes] == base_names
    assert after.uuid() == base_uuid


def test_real_class_anchor_errors_name_torch_spyre(recorder):
    sps.register_pre_scheduling_pass(
        lambda g: None, after="no_such_anchor", name="bad_anchor_pass"
    )
    with sps.spyre_pass_scope(["bad_anchor_pass"]):
        with pytest.raises(sps.PreSchedulingPassError) as e:
            ts_passes.CustomPreSchedulingPasses()
    msg = str(e.value)
    assert "no_such_anchor" in msg and ANCHOR in msg and "torch_spyre" in msg


def test_hello_world_example_runs_once_per_compile_and_keeps_ir(recorder, caplog):
    """examples/pre_scheduling_pass/hello_pre_scheduling_pass.py is read-only."""
    path = Path(__file__).parents[5] / "examples/pre_scheduling_pass"
    spec = importlib.util.spec_from_file_location(
        "hello_pre_scheduling_pass", path / "hello_pre_scheduling_pass.py"
    )
    spec.loader.exec_module(importlib.util.module_from_spec(spec))  # registers it

    _lower([])
    baseline_ir, baseline_calls = list(recorder.ir), recorder.compile_calls
    recorder.ir.clear()
    recorder.pass_lists.clear()
    with caplog.at_level(logging.INFO, logger="hello_pre_scheduling_pass"):
        _lower([sps.pass_compile_spec(["example_count_ops"])])
    logged = [r for r in caplog.records if "example_count_ops:" in r.getMessage()]
    assert len(logged) == len(recorder.pass_lists) >= 1  # once per compile_fx call
    assert len(logged) == recorder.compile_calls - baseline_calls
    assert all(" 0 ops" not in r.getMessage() for r in logged)  # it did count
    assert recorder.ir == baseline_ir
