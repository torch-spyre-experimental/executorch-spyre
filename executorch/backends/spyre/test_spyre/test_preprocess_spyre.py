"""SDSC preprocessing regressions that require physical Spyre hardware."""

import io
import json
import struct
import tarfile

import pytest
import torch

from executorch.exir import to_edge_transform_and_lower

from executorch.backends.spyre.spyre_backend import SPYRE_BACKEND_ID, SpyreBackend
from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner
from executorch.backends.spyre.test.test_preprocess import (
    LinearCumsumLinear,
    MLP,
    _edge_program,
)

pytestmark = pytest.mark.spyre


def test_preprocess_emits_parseable_payload():
    mod = MLP().eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[SpyrePartitioner()]
    ).to_executorch()

    (delegate,) = lowered.executorch_program.execution_plan[0].delegates
    payload = bytes(
        lowered.executorch_program.backend_delegate_data[delegate.processed.index].data
    )
    (header_len,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))

    # The static path always emits v2, even for a single kernel.
    # NOTE: updated for v2, not run on device (preflight guard: torch-spyre
    # version mismatch).
    assert header["version"] == 2
    assert header["backend"] == SPYRE_BACKEND_ID
    assert header["n_inputs"] == 5
    assert header["n_outputs"] == 1
    assert "spyrecode_subdir" not in header
    assert len(header["kernels"]) == 1
    assert header["kernels"][0]["subdir"] == "spyreCodeDir_0"

    recovered_tar = payload[4 + header_len :]
    with tarfile.open(fileobj=io.BytesIO(recovered_tar), mode="r") as tf:
        names = tf.getnames()
    assert any(n.startswith("spyreCodeDir_0/") for n in names)
    assert any("spyrecode.json" in name for name in names)


def test_preprocess_emits_v2_multikernel_payload():
    edge = _edge_program(LinearCumsumLinear())
    result = SpyreBackend.preprocess(edge, [])
    payload = bytes(result.processed_bytes)

    (header_len,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    assert header["version"] == 2
    assert header["backend"] == SPYRE_BACKEND_ID
    assert len(header["kernels"]) == 2
    assert header["intermediates"]

    recovered_tar = payload[4 + header_len :]
    with tarfile.open(fileobj=io.BytesIO(recovered_tar), mode="r") as tf:
        names = tf.getnames()
    for subdir in ("spyreCodeDir_0", "spyreCodeDir_1"):
        assert any(name.startswith(f"{subdir}/") for name in names)
        assert any(subdir in name and "spyrecode.json" in name for name in names)
