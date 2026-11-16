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
"""Unit tests for the AOT ``preprocess`` helpers in ``spyre_backend``.

The tests here are device-free: they exercise the pure helpers
(``_torch_dtype_to_str``, ``_pack_payload`` wire format, ``_retarget_edge_to_aten``
edge→aten rewrite, and ``_placeholder_kind`` lifted-input classification) with
only ``torch`` + ``executorch``. End-to-end SDSC compilation tests live in the
sibling ``test_spyre`` suite.
"""

import glob
import io
import json
import struct
import tarfile
import tempfile
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402

from executorch.backends.spyre.spyre_backend import (  # noqa: E402
    SPYRE_BACKEND_ID,
    _build_v2_layout,
    _build_v3_header,
    _capture_one_compile,
    _check_capture_counts,
    _describe_compile_failure,
    _pack_payload,
    _placeholder_kind,
    _retarget_edge_to_aten,
    _torch_dtype_to_str,
)


# --- Module under test (shared with the partitioner tests) -----------------


class MLP(torch.nn.Module):
    """Every op is on the Spyre allowlist; carries lifted params + buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _edge_program(mod: torch.nn.Module):
    mod = mod.eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    return to_edge(exported).exported_program()


class LinearCumsumLinear(torch.nn.Module):
    """Phase 6a's 2-kernel vehicle: ``cumsum`` falls back to CPU, forcing a
    boundary between the two linears -> two SDSC kernels."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.cumsum(self.fc1(x), dim=-1))


# --- _torch_dtype_to_str ---------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.int64, "int64"),
        (torch.uint8, "uint8"),
        (torch.bool, "bool"),
    ],
)
def test_torch_dtype_to_str(dtype, expected):
    # The C++ side keys tensor construction on these bare names; the "torch."
    # prefix must be stripped exactly.
    assert _torch_dtype_to_str(dtype) == expected


# --- _pack_payload wire format ---------------------------------------------


def test_pack_payload_frames_header_then_tar():
    header = {"version": 1, "backend": SPYRE_BACKEND_ID, "n_inputs": 3}
    tar_bytes = b"\x00\x01\x02fake-tar-bytes\xff"

    payload = _pack_payload(header, tar_bytes)

    # [u32 header_len LE][header_json][tar] — mirror C++ SpyreBackend::init.
    (header_len,) = struct.unpack("<I", payload[:4])
    header_json = payload[4 : 4 + header_len]
    trailing = payload[4 + header_len :]

    assert json.loads(header_json.decode("utf-8")) == header
    assert trailing == tar_bytes


def test_pack_payload_roundtrips_a_real_tar():
    # A payload whose tar actually opens — the shape the C++ delegate untars.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        data = b'{"JobExecPlan": {}, "JobPreparationPlan": {}}'
        info = tarfile.TarInfo("spyreCodeDir/spyrecode.json")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    tar_bytes = buf.getvalue()

    header = {"version": 1, "spyrecode_subdir": "spyreCodeDir"}
    payload = _pack_payload(header, tar_bytes)

    (header_len,) = struct.unpack("<I", payload[:4])
    recovered_tar = payload[4 + header_len :]
    with tarfile.open(fileobj=io.BytesIO(recovered_tar), mode="r") as tf:
        names = tf.getnames()
    assert "spyreCodeDir/spyrecode.json" in names


def test_pack_payload_header_len_is_utf8_byte_count():
    # A non-ASCII value must be counted in *bytes*, not characters, or the C++
    # split point drifts.
    header = {"note": "café-µ"}  # multi-byte utf-8
    payload = _pack_payload(header, b"tail")
    (header_len,) = struct.unpack("<I", payload[:4])
    assert header_len == len(json.dumps(header).encode("utf-8"))
    assert payload[4 + header_len :] == b"tail"


# --- _retarget_edge_to_aten ------------------------------------------------


def test_retarget_edge_to_aten_rewrites_edge_overloads():
    from torch._ops import OpOverload

    edge = _edge_program(MLP())
    gm = edge.graph_module

    call_fns = [n for n in gm.graph.nodes if n.op == "call_function"]
    assert call_fns, "expected compute nodes in the edge graph"

    n_retargeted = _retarget_edge_to_aten(gm)

    # Every edge overload is rewritten; the count matches the compute nodes that
    # carried an edge op (MLP: addmm x2, permute_copy x2, relu).
    assert n_retargeted > 0
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        # After retargeting, targets are plain core-aten OpOverloads (no edge
        # wrapper carrying a ``._op`` attribute).
        assert isinstance(node.target, OpOverload)
        assert node.target.name().startswith("aten::")


# --- _placeholder_kind -----------------------------------------------------


def test_placeholder_kind_classifies_lifted_inputs():
    edge = _edge_program(MLP())
    kinds = {
        node.name: _placeholder_kind(edge, node.name)
        for node in edge.graph_module.graph.nodes
        if node.op == "placeholder"
    }

    # MLP lifts two weights + two biases (params) and one user input.
    assert "user_input" in kinds.values()
    assert list(kinds.values()).count("parameter") >= 4
    # No placeholder is left unclassified as something outside the taxonomy.
    assert set(kinds.values()) <= {"parameter", "buffer", "constant", "user_input"}


def test_placeholder_kind_name_convention_fallback():
    # Even with no signature maps, the p_/b_/c_ lifting prefixes classify.

    class _NoSig:
        graph_signature = None

    stub = _NoSig()
    assert _placeholder_kind(stub, "p_fc1_weight") == "parameter"
    assert _placeholder_kind(stub, "b_running_mean") == "buffer"
    assert _placeholder_kind(stub, "c_lifted_const0") == "constant"
    assert _placeholder_kind(stub, "x") == "user_input"


def test_placeholder_kind_ms_prefix_is_a_plain_user_input():
    # The ``ms_`` mutable_state probe convention is gone: such a name is no
    # longer special-cased, and no placeholder is ever classified mutable_state.
    class _NoSig:
        graph_signature = None

    assert _placeholder_kind(_NoSig(), "ms_state0") == "user_input"


# --- _make_spyre_example_inputs (Phase 7 mixed-dtype, host-only) -----------


def test_make_spyre_example_inputs_preserves_mixed_dtypes(monkeypatch):
    # Phase 7 mixed-dtype policy: an int64-indexed embedding feeding an fp16
    # matmul. The synthesized example inputs must keep the token-ids at int64
    # (indices are never downcast) while the float weights are fp16 -- the AOT
    # half of the policy. Only ``.to("spyre")`` in the helper is device-bound and
    # dtype selection is device-independent, so retarget the examples to CPU to
    # run this on the Mac.
    import executorch.backends.spyre.spyre_backend as spyre_backend

    monkeypatch.setattr(spyre_backend, "_SPYRE_DEVICE", "cpu")

    class MixedDtype(torch.nn.Module):
        """int64 token-ids -> fp16 embedding lookup -> fp16 linear."""

        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(32, 64)
            self.proj = torch.nn.Linear(64, 16)

        def forward(self, ids):
            return self.proj(self.emb(ids))

    mod = MixedDtype().eval().half()
    ids = torch.zeros(1, 8, dtype=torch.int64)  # token-ids: int64, NOT downcast
    exported = torch.export.export(mod, (ids,))
    edge = to_edge(exported).exported_program()

    inputs, input_specs, output_specs = spyre_backend._make_spyre_example_inputs(edge)

    spec_dtypes = [s["dtype"] for s in input_specs]
    # The lone int64 user input (ids) survives; the float weights become fp16 and
    # nothing is silently left at fp32.
    assert "int64" in spec_dtypes
    assert spec_dtypes.count("float16") >= 1
    assert "float32" not in spec_dtypes

    # The example tensors carry the matching dtypes -- int64 zero-filled indices,
    # randomized fp16 weights.
    tensor_dtypes = {t.dtype for t in inputs}
    assert torch.int64 in tensor_dtypes
    assert torch.float16 in tensor_dtypes

    # The int64 input is exactly the ids placeholder (a user_input), never a
    # lifted weight.
    id_specs = [s for s in input_specs if s["dtype"] == "int64"]
    assert id_specs and all(s["kind"] == "user_input" for s in id_specs)

    # The fp16 matmul's result stays fp16 on the output side.
    assert output_specs[0]["dtype"] == "float16"


# --- _build_v2_layout (host-only, replays the Phase 6a observation) --------


def _mk_tensor_rec(data_ptr, shape, dtype_str, is_output_pos):
    return {
        "is_tensor": True,
        "data_ptr": data_ptr,
        "dtype_str": dtype_str,
        "shape": list(shape),
        "dim": len(shape),
        "is_output_pos": is_output_pos,
    }


def test_build_v2_layout_from_6a_records():
    # Replay the exact per-arg records Phase 6a captured for
    # ``fc1 -> cumsum(dim=-1) -> fc2`` (report tables, section "The three data
    # points"). Buffer identity keys on data_ptr; the trailing tensor is the
    # written output (is_output_pos=True). This validates the classifier fix on
    # the Mac -- no Spyre device needed.
    POOL = 94759151221056  # [256] uint8 scratch, SHARED across both kernels
    FC1_W, FC1_B = 94759133450336, 94759133109248  # delegate inputs idx 0, 1
    FC2_W, FC2_B = 94759134011664, 94759133528000  # delegate inputs idx 2, 3
    X = 94759133244352  # delegate input idx 4
    FC1_OUT = 94759151220416  # produced by k0, host-round-tripped (not consumed)
    CUMSUM_OUT = 94759141340608  # DIFFERENT data_ptr; consumed by k1
    FINAL_OUT = 94759131740640  # delegate output idx 0

    # Kernel 0 (fc1 addmm): [pool, fc1.w, x, fc1.b, fc1-result]
    k0 = [
        _mk_tensor_rec(POOL, [256], "uint8", False),
        _mk_tensor_rec(FC1_W, [128, 64], "float16", False),
        _mk_tensor_rec(X, [64, 64], "float16", False),
        _mk_tensor_rec(FC1_B, [128], "float16", False),
        _mk_tensor_rec(FC1_OUT, [64, 128], "float16", True),
    ]
    # Kernel 1 (fc2 addmm): [pool, fc2.w, cumsum-result, fc2.b, final-result]
    k1 = [
        _mk_tensor_rec(POOL, [256], "uint8", False),
        _mk_tensor_rec(FC2_W, [64, 128], "float16", False),
        _mk_tensor_rec(CUMSUM_OUT, [64, 128], "float16", False),
        _mk_tensor_rec(FC2_B, [64], "float16", False),
        _mk_tensor_rec(FINAL_OUT, [64, 64], "float16", True),
    ]

    input_ptr_to_idx = {FC1_W: 0, FC1_B: 1, FC2_W: 2, FC2_B: 3, X: 4}
    output_ptr_to_idx = {FINAL_OUT: 0}

    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0, k1], input_ptr_to_idx, output_ptr_to_idx
    )

    # Two kernels, one shared scratch pool.
    assert len(kernels) == 2
    assert pool_sticks == 256
    assert kernels[0]["subdir"] == "spyreCodeDir_0"
    assert kernels[1]["subdir"] == "spyreCodeDir_1"

    k0_layout = kernels[0]["launch_layout"]
    k1_layout = kernels[1]["launch_layout"]

    # Kernel 0: pool, three inputs, then the produced fc1 intermediate.
    assert k0_layout[0] == {"role": "pool"}
    assert k0_layout[1] == {"role": "input", "index": 0}
    assert k0_layout[2] == {"role": "input", "index": 4}
    assert k0_layout[3] == {"role": "input", "index": 1}
    assert k0_layout[4]["role"] == "intermediate"
    assert k0_layout[4]["produces"] is True

    # Kernel 1: the cumsum result is a CONSUMED intermediate -- the Phase 6a bug
    # mislabeled it "output". It must be role="intermediate", produces=False.
    assert k1_layout[0] == {"role": "pool"}
    assert k1_layout[1] == {"role": "input", "index": 2}
    assert k1_layout[2]["role"] == "intermediate"
    assert k1_layout[2]["produces"] is False
    assert k1_layout[3] == {"role": "input", "index": 3}
    # The final result is the delegate output, NOT an intermediate.
    assert k1_layout[4] == {"role": "output", "index": 0}

    # Two distinct intermediate buffer ids (differing data_ptrs), both the
    # host-round-trip edge -> resident:false, neither a delegate output.
    assert len(intermediates) == 2
    produced = k0_layout[4]["buffer"]
    consumed = k1_layout[2]["buffer"]
    assert produced != consumed
    for im in intermediates:
        assert im["resident"] is False
        assert im["delegate_output"] is False
        assert im["dtype"] == "float16"
        assert im["shape"] == [64, 128]


def test_build_v2_layout_collapses_resident_edge():
    # Complement to the 6a host-round-trip case: when a producer and consumer
    # share the SAME data_ptr (a genuine device-resident edge), the two args
    # collapse to ONE buffer id marked resident:true.
    POOL = 1000
    RESIDENT = 2000  # produced by k0, consumed by k1 -- same device buffer
    OUT = 3000

    k0 = [
        _mk_tensor_rec(POOL, [128], "uint8", False),
        _mk_tensor_rec(RESIDENT, [8, 8], "float16", True),
    ]
    k1 = [
        _mk_tensor_rec(POOL, [128], "uint8", False),
        _mk_tensor_rec(RESIDENT, [8, 8], "float16", False),
        _mk_tensor_rec(OUT, [8, 8], "float16", True),
    ]

    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0, k1], input_ptr_to_idx={}, output_ptr_to_idx={OUT: 0}
    )

    assert pool_sticks == 128
    assert len(intermediates) == 1
    assert intermediates[0]["resident"] is True
    # Both kernels reference the single buffer id; k0 produces, k1 consumes.
    prod = kernels[0]["launch_layout"][1]
    cons = kernels[1]["launch_layout"][1]
    assert prod["role"] == cons["role"] == "intermediate"
    assert prod["buffer"] == cons["buffer"] == intermediates[0]["id"]
    assert prod["produces"] is True
    assert cons["produces"] is False


def test_build_v2_layout_single_kernel_workspace_is_intermediate():
    # Phase 8 8a follow-up: bmm/matmul compile to a SINGLE kernel that allocates
    # an internal workspace/accumulator tensor. That tensor is neither a delegate
    # input nor a delegate output, so the naive v1 classifier ("input | pool |
    # else->output") mislabels it "output index 1" -- but n_outputs==1, so the
    # C++ execute() binds one past the outputs array and returns Error::Internal.
    # preprocess() detects this by probing _build_v2_layout on the single kernel:
    # a non-empty intermediates list means "route through v2, not v1". This
    # replays the exact bmm launch order the device payload showed
    # ([pool, workspace, in1, in0, out0]) and asserts the workspace is captured as
    # an intermediate, not a phantom output. Device-free -- runs on the Mac.
    POOL = 5000  # [512] uint8 scratch pool (pool_sticks == 512 on device)
    IN0, IN1 = 6000, 6001  # the two bmm operands (delegate inputs 0, 1)
    WORKSPACE = 7000  # internal accumulator: neither a delegate input nor output
    OUT = 8000  # the single delegate output (index 0)

    # Launch order mirrors the device-dumped bmm layout.
    k0 = [
        _mk_tensor_rec(POOL, [512], "uint8", False),
        _mk_tensor_rec(WORKSPACE, [3, 32, 32], "float16", True),
        _mk_tensor_rec(IN1, [3, 48, 32], "float16", False),
        _mk_tensor_rec(IN0, [3, 32, 48], "float16", False),
        _mk_tensor_rec(OUT, [3, 32, 32], "float16", True),
    ]

    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0], input_ptr_to_idx={IN0: 0, IN1: 1}, output_ptr_to_idx={OUT: 0}
    )

    # The probe preprocess() uses: a single kernel WITH an intermediate must NOT
    # take the v1 path.
    assert len(intermediates) >= 1, "workspace must be captured as an intermediate"
    assert len(kernels) == 1
    assert pool_sticks == 512

    layout = kernels[0]["launch_layout"]
    # The workspace tensor is role=intermediate (produced this kernel), never a
    # second output -- this is the exact bug the v1->v2 routing fixes.
    ws = layout[1]
    assert ws["role"] == "intermediate"
    assert ws["produces"] is True
    # Exactly one delegate output survives, at index 0. No phantom "output 1".
    outputs = [a for a in layout if a.get("role") == "output"]
    assert outputs == [{"role": "output", "index": 0}]
    # The two real operands stay inputs.
    inputs = sorted(a["index"] for a in layout if a.get("role") == "input")
    assert inputs == [0, 1]


def test_build_v2_layout_single_kernel_one_output():
    # The shape the removed v1 path used to cover: one kernel, [pool, inputs...,
    # output]. v2 gives one kernels[] entry, no intermediates, and a lifted
    # top-level pool_sticks.
    POOL, W, X, OUT = 100, 200, 300, 400
    k0 = [
        _mk_tensor_rec(POOL, [64], "uint8", False),
        _mk_tensor_rec(W, [8, 8], "float16", False),
        _mk_tensor_rec(X, [1, 8], "float16", False),
        _mk_tensor_rec(OUT, [1, 8], "float16", True),
    ]
    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0], input_ptr_to_idx={W: 0, X: 1}, output_ptr_to_idx={OUT: 0}
    )
    assert intermediates == []
    assert pool_sticks == 64
    assert kernels == [
        {
            "subdir": "spyreCodeDir_0",
            "launch_layout": [
                {"role": "pool"},
                {"role": "input", "index": 0},
                {"role": "input", "index": 1},
                {"role": "output", "index": 0},
            ],
        }
    ]


def test_build_v2_layout_output_index_follows_pointer_not_launch_order():
    # The removed v1 classifier numbered outputs by a running counter in launch
    # order. v2 uses the delegate output index of the matching data_ptr, so a
    # kernel that writes output 1 before output 0 keeps the right indices.
    IN, OUT0, OUT1 = 10, 20, 30
    k0 = [
        _mk_tensor_rec(IN, [4], "float16", False),
        _mk_tensor_rec(OUT1, [4], "float16", True),
        _mk_tensor_rec(OUT0, [4], "float16", True),
    ]
    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0], input_ptr_to_idx={IN: 0}, output_ptr_to_idx={OUT0: 0, OUT1: 1}
    )
    assert intermediates == []
    assert pool_sticks == 0
    assert kernels[0]["launch_layout"] == [
        {"role": "input", "index": 0},
        {"role": "output", "index": 1},
        {"role": "output", "index": 0},
    ]


def test_build_v2_layout_pool_has_no_per_entry_sticks():
    # v1 put "sticks" on the pool launch entry; v2 emits a bare {"role": "pool"}
    # and carries the size once, at the top level.
    POOL, OUT = 1, 2
    k0 = [
        _mk_tensor_rec(POOL, [512], "uint8", False),
        _mk_tensor_rec(OUT, [4], "float16", True),
    ]
    kernels, _, pool_sticks = _build_v2_layout(
        [k0], input_ptr_to_idx={}, output_ptr_to_idx={OUT: 0}
    )
    assert pool_sticks == 512
    assert kernels[0]["launch_layout"][0] == {"role": "pool"}


# --- capture guard (host-only) ---------------------------------------------


def test_check_capture_counts_accepts_one_launch_per_dir():
    _check_capture_counts(["d0"], [[]])
    _check_capture_counts(["d0", "d1"], [[], []])


def test_build_v2_layout_rejects_empty_captured_run():
    # A launch with no recorded args would yield an empty launch_layout, which the
    # C++ parser rejects; the exporter must fail first.
    with pytest.raises(RuntimeError, match="no arguments"):
        _build_v2_layout([[]], {}, {})
    with pytest.raises(RuntimeError, match="kernel 1 "):
        _build_v2_layout([[{"is_tensor": False}], []], {}, {})


@pytest.mark.parametrize("dirs,runs", [(["d0"], []), ([], [[]]), (["d0"], [[], []])])
def test_check_capture_counts_rejects_mismatch(dirs, runs):
    with pytest.raises(RuntimeError, match="SpyreSDSCKernelRunner.run") as ei:
        _check_capture_counts(dirs, runs)
    # Both counts are reported so the failure says which side is off.
    assert f"observed {len(runs)}" in str(ei.value)
    assert f"{len(dirs)} SpyreCode dir(s)" in str(ei.value)


def _fake_compile_env(launches, fail_on_call=None):
    """Fake (cfx, ac, kr) whose compile emits one dir and runs ``launches`` kernels.

    The emitted dir holds a ``spyreCodeDir`` like the real codegen. ``fail_on_call``
    (1-based) makes that ``compile_fx`` call raise after emitting its dir, as a
    crashed ``dxp_standalone`` does.
    """
    import os

    class Runner:
        def run(self, *args):
            return None

    ac = types.SimpleNamespace(get_output_dir=lambda name, *a, **k: "/unused")
    kr = types.SimpleNamespace(SpyreSDSCKernelRunner=Runner)
    calls = []

    def compile_fx(gm, example_inputs):
        calls.append(1)
        # the patched version returns an owned dir
        owned = ac.get_output_dir("k0")
        os.makedirs(os.path.join(owned, "spyreCodeDir"))
        with open(os.path.join(owned, "spyreCodeDir", "spyrecode.json"), "w") as f:
            f.write("{}")
        if fail_on_call == len(calls):
            raise RuntimeError("fake dxp_standalone crash")

        def compiled(*args):
            for _ in range(launches):
                Runner().run(*args)
            return torch.zeros(1)

        return compiled

    return types.SimpleNamespace(compile_fx=compile_fx), ac, kr


def _spyre_sdsc_dirs(root):
    return set(glob.glob(f"{root}/spyre_sdsc_*"))


def _redirect_tempdir(monkeypatch, tmp_path):
    """Point tempfile at tmp_path (tempdir is cached; setting TMPDIR mid-session
    would not take effect) and return the real temp dir's current spyre_sdsc_* set,
    so a test can check nothing leaks into either place."""
    real_tmp = tempfile.gettempdir()
    before = _spyre_sdsc_dirs(real_tmp)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    return real_tmp, before


def test_capture_one_compile_raises_when_run_hook_never_fires(monkeypatch, tmp_path):
    # The guard raises after the dir was emitted; the dir must not be left behind.
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    cfx, ac, kr = _fake_compile_env(launches=0)
    with pytest.raises(RuntimeError, match="observed 0"):
        _capture_one_compile(None, [torch.zeros(1)], cfx, ac, kr)
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before


def test_capture_one_compile_removes_dirs_when_compile_raises(monkeypatch, tmp_path):
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    cfx, ac, kr = _fake_compile_env(launches=1, fail_on_call=1)
    original_get_output_dir = ac.get_output_dir
    with pytest.raises(RuntimeError, match="fake dxp_standalone crash"):
        _capture_one_compile(None, [torch.zeros(1)], cfx, ac, kr)
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before
    assert ac.get_output_dir is original_get_output_dir  # patches still undone


def test_capture_one_compile_accepts_matching_launch(monkeypatch, tmp_path):
    real_tmp = tempfile.gettempdir()
    before = _spyre_sdsc_dirs(real_tmp)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    cfx, ac, kr = _fake_compile_env(launches=1)
    dirs, runs, *_ = _capture_one_compile(None, [torch.zeros(1)], cfx, ac, kr)
    assert len(dirs) == len(runs) == 1
    assert all(d.startswith(str(tmp_path)) for d in dirs)
    assert _spyre_sdsc_dirs(real_tmp) == before
    # On success the caller owns the dirs (it still has to tar them).
    assert _spyre_sdsc_dirs(tmp_path) == set(dirs)


# --- AOT leaves no spyre_sdsc_* dir behind (host-only) -----------------------


def _preprocess_env(monkeypatch, bucketed=False):
    """Stub the device-bound helpers so _preprocess_impl runs against the fakes."""
    import executorch.backends.spyre.spyre_backend as spyre_backend

    def make_inputs(edge, shape_overrides=None):
        return (
            [torch.zeros(1)],
            [{"shape": [1], "dtype": "float16", "kind": "user_input", "state_out": -1}],
            [{"shape": [1], "dtype": "float32"}],
        )

    monkeypatch.setattr(spyre_backend, "_retarget_edge_to_aten", lambda gm: 0)
    monkeypatch.setattr(spyre_backend, "_make_spyre_example_inputs", make_inputs)
    if bucketed:
        monkeypatch.setattr(
            spyre_backend, "_resolve_dynamic_dims", lambda ep, gm, arg_dims: [(0, 0)]
        )
    return spyre_backend.SpyreBackend, types.SimpleNamespace(graph_module=None)


def _tar_names(payload):
    (header_len,) = struct.unpack("<I", payload[:4])
    with tarfile.open(fileobj=io.BytesIO(payload[4 + header_len :]), mode="r") as tf:
        return tf.getnames()


def test_preprocess_impl_success_removes_dirs_and_keeps_payload(monkeypatch, tmp_path):
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    backend, edge = _preprocess_env(monkeypatch)
    cfx, ac, kr = _fake_compile_env(launches=1)
    result = backend._preprocess_impl(edge, [], cfx, ac, kr)
    assert "spyreCodeDir_0/spyrecode.json" in _tar_names(result.processed_bytes)
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before


def test_preprocess_impl_guard_failure_removes_dirs(monkeypatch, tmp_path):
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    backend, edge = _preprocess_env(monkeypatch)
    cfx, ac, kr = _fake_compile_env(launches=0)
    with pytest.raises(RuntimeError, match="observed 0"):
        backend._preprocess_impl(edge, [], cfx, ac, kr)
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before


def test_preprocess_impl_failure_after_capture_removes_dirs(monkeypatch, tmp_path):
    import executorch.backends.spyre.spyre_backend as spyre_backend

    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    backend, edge = _preprocess_env(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("layout build failed")

    monkeypatch.setattr(spyre_backend, "_build_v2_layout", boom)
    cfx, ac, kr = _fake_compile_env(launches=1)
    with pytest.raises(RuntimeError, match="layout build failed"):
        backend._preprocess_impl(edge, [], cfx, ac, kr)
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before


_BUCKET_SPEC = [
    types.SimpleNamespace(
        key="spyre_dynamic_buckets",
        value=json.dumps({"arg_dims": [{"arg": 0, "axis": 0}], "buckets": [1, 2]}),
    )
]


def test_preprocess_bucketed_success_removes_all_bucket_dirs(monkeypatch, tmp_path):
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    backend, edge = _preprocess_env(monkeypatch, bucketed=True)
    cfx, ac, kr = _fake_compile_env(launches=1)
    result = backend._preprocess_impl(edge, _BUCKET_SPEC, cfx, ac, kr)
    names = _tar_names(result.processed_bytes)
    assert "bucket0/spyreCodeDir_0/spyrecode.json" in names
    assert "bucket1/spyreCodeDir_0/spyrecode.json" in names
    assert _spyre_sdsc_dirs(tmp_path) == set()
    assert _spyre_sdsc_dirs(real_tmp) == before


def test_preprocess_bucketed_failure_in_later_bucket_removes_earlier_dirs(
    monkeypatch, tmp_path
):
    real_tmp, before = _redirect_tempdir(monkeypatch, tmp_path)
    backend, edge = _preprocess_env(monkeypatch, bucketed=True)
    cfx, ac, kr = _fake_compile_env(launches=1, fail_on_call=2)
    with pytest.raises(RuntimeError, match="fake dxp_standalone crash"):
        backend._preprocess_impl(edge, _BUCKET_SPEC, cfx, ac, kr)
    assert _spyre_sdsc_dirs(tmp_path) == set()  # bucket 0's dir is gone too
    assert _spyre_sdsc_dirs(real_tmp) == before


# --- _build_v3_header (host-only, Phase 7 Track 2 bucket grouping) ----------


def _single_kernel_bucket(m, pool_ptr, w_ptr, x_ptr, out_ptr):
    """Build one bucket's (kernels, intermediates, pool_sticks, specs) via the
    same _build_v2_layout path preprocess uses -- a single-kernel matmul at a
    concrete M so the two buckets differ only in the dynamic dim."""
    # launch order: [pool, weight(input 0), x(input 1), result(output 0)]
    k0 = [
        _mk_tensor_rec(pool_ptr, [128], "uint8", False),
        _mk_tensor_rec(w_ptr, [64, 64], "float16", False),
        _mk_tensor_rec(x_ptr, [m, 64], "float16", False),
        _mk_tensor_rec(out_ptr, [m, 64], "float16", True),
    ]
    kernels, intermediates, pool_sticks = _build_v2_layout(
        [k0], input_ptr_to_idx={w_ptr: 0, x_ptr: 1}, output_ptr_to_idx={out_ptr: 0}
    )
    input_specs = [
        {"shape": [64, 64], "dtype": "float16", "kind": "parameter"},
        {"shape": [m, 64], "dtype": "float16", "kind": "user_input"},
    ]
    output_specs = [{"shape": [m, 64], "dtype": "float16"}]
    return kernels, intermediates, pool_sticks, input_specs, output_specs


def test_build_v3_header_groups_two_buckets():
    # Replay the exact shape preprocess._preprocess_bucketed hands to
    # _build_v3_header: two buckets (M=128, M=256) over one dynamic dim
    # (input index 1, axis 0). Each bucket is a self-contained v2 layout under
    # its own subdir_prefix. Pure + device-free -- runs on the Mac.
    dynamic_input_dims = [[1, 0]]  # delegate input 1, axis 0 varies across buckets

    b0_k, b0_im, b0_pool, b0_in, b0_out = _single_kernel_bucket(
        128, pool_ptr=1000, w_ptr=2000, x_ptr=3000, out_ptr=4000
    )
    b1_k, b1_im, b1_pool, b1_in, b1_out = _single_kernel_bucket(
        256, pool_ptr=1000, w_ptr=2000, x_ptr=3001, out_ptr=4001
    )
    buckets = [
        {
            "shape_key": [128],
            "subdir_prefix": "bucket0",
            "pool_sticks": b0_pool,
            "kernels": b0_k,
            "intermediates": b0_im,
            "inputs": b0_in,
            "outputs": b0_out,
        },
        {
            "shape_key": [256],
            "subdir_prefix": "bucket1",
            "pool_sticks": b1_pool,
            "kernels": b1_k,
            "intermediates": b1_im,
            "inputs": b1_in,
            "outputs": b1_out,
        },
    ]

    header = _build_v3_header(
        dynamic_input_dims,
        buckets,
        rep_input_specs=b0_in,
        rep_output_specs=b0_out,
        n_retargeted=1,
    )

    # v3 identity + the dynamic-dim directive the runtime reads to pick a bucket.
    assert header["version"] == 3
    assert header["backend"] == SPYRE_BACKEND_ID
    assert header["dynamic_input_dims"] == [[1, 0]]

    # Two distinct buckets, distinct shape_keys and subdir_prefixes -- the
    # runtime keys bucket selection on shape_key, extraction on subdir_prefix.
    hb = header["buckets"]
    assert len(hb) == 2
    assert [b["shape_key"] for b in hb] == [[128], [256]]
    assert [b["subdir_prefix"] for b in hb] == ["bucket0", "bucket1"]
    assert len({b["subdir_prefix"] for b in hb}) == 2

    # Each bucket carries a complete v2 sub-schema: one kernel with its own
    # spyreCodeDir_0, a pool, and per-bucket input/output specs at the bucket M.
    for b, m in zip(hb, (128, 256)):
        assert len(b["kernels"]) == 1
        assert b["kernels"][0]["subdir"] == "spyreCodeDir_0"
        assert b["pool_sticks"] == 128
        assert b["outputs"][0]["shape"] == [m, 64]
        # The user input's dynamic axis reflects this bucket's concrete size.
        user_in = [s for s in b["inputs"] if s.get("kind") == "user_input"]
        assert user_in and user_in[0]["shape"] == [m, 64]

    # Legacy top-level fields mirror the first bucket so a v3 payload still
    # degrades to v2 semantics (n_inputs/n_outputs/inputs/outputs).
    assert header["n_inputs"] == len(b0_in)
    assert header["n_outputs"] == len(b0_out)
    assert header["inputs"] == b0_in
    assert header["outputs"] == b0_out
    assert header["n_retargeted_ops"] == 1


def test_build_v3_header_survives_pack_roundtrip():
    # The assembled v3 header must frame + JSON-roundtrip through _pack_payload
    # exactly (the C++ delegate re-parses this). Reuse a minimal 1-bucket header.
    b_k, b_im, b_pool, b_in, b_out = _single_kernel_bucket(
        128, pool_ptr=1000, w_ptr=2000, x_ptr=3000, out_ptr=4000
    )
    header = _build_v3_header(
        [[1, 0]],
        [
            {
                "shape_key": [128],
                "subdir_prefix": "bucket0",
                "pool_sticks": b_pool,
                "kernels": b_k,
                "intermediates": b_im,
                "inputs": b_in,
                "outputs": b_out,
            }
        ],
        rep_input_specs=b_in,
        rep_output_specs=b_out,
        n_retargeted=1,
    )
    payload = _pack_payload(header, b"tar-bytes")
    (header_len,) = struct.unpack("<I", payload[:4])
    recovered = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    assert recovered == header
    assert recovered["version"] == 3
    assert payload[4 + header_len :] == b"tar-bytes"


# --- _describe_compile_failure (#50: name the offending node) ----------------


def test_describe_compile_failure_names_suspect_node_and_target():
    gm = _edge_program(MLP()).graph_module
    relu = next(
        n for n in gm.graph.nodes if n.op == "call_function" and "relu" in n.name
    )
    msg = _describe_compile_failure(
        gm, RuntimeError(f"InductorError: KeyError 'No FX node for {relu.name}'")
    )
    assert f"suspect node(s): {relu.name} (target=relu)" in msg
    assert "RuntimeError" in msg
    assert "addmm x2" in msg  # op histogram of the subgraph


def test_describe_compile_failure_matches_on_op_name():
    gm = _edge_program(MLP()).graph_module
    msg = _describe_compile_failure(gm, ValueError("lowering for addmm failed"))
    assert "target=addmm" in msg


def test_describe_compile_failure_without_a_named_node():
    gm = _edge_program(MLP()).graph_module
    msg = _describe_compile_failure(gm, RuntimeError("opaque"))
    assert "none named in the error text" in msg
    assert "subgraph ops:" in msg
