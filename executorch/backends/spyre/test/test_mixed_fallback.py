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
"""Host (device-free) tests for the simplest Spyre/CPU **mixed** workload.

The device run (``scripts/run_mixed_fallback.sh``) proves the ``.pte`` executes
end-to-end -- ``sin`` on CPU while ``fc``/``relu`` delegate to Spyre. These tests
pin, without a device, the AOT structure that run depends on:

  1. **only ``aten::sin`` falls back** (a generic Rule-1 allowlist miss) and
     every delegated node is on the allowlist;
  2. the interior CPU op splits the graph into **two delegate partitions**
     (two delegate calls with a device->host->device hand-off);
  3. the **"no silent coverage gaps"** guarantee -- every non-delegated compute
     node is logged by the partitioner (the guide's headline contract);
The SDSC ``.pte`` build belongs to the sibling ``test_spyre`` suite.
"""

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402

from executorch.backends.spyre.spyre_partitioner import (  # noqa: E402
    _SUPPORTED_ATEN_OPS,
    _aten_name,
    SpyrePartitioner,
)
from executorch.backends.spyre.export_mixed import MixedUnsupported  # noqa: E402


def _edge_program(mod: torch.nn.Module):
    """Export ``mod`` (fp16, static ``[1, 64]``) and lower to the edge dialect."""
    mod = mod.eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    return to_edge(exported).exported_program()


def _tag_split(result):
    compute = [
        n
        for n in result.tagged_exported_program.graph_module.graph.nodes
        if n.op == "call_function"
    ]
    tagged = [n for n in compute if n.meta.get("delegation_tag") is not None]
    fallback = [n for n in compute if n not in tagged]
    return compute, tagged, fallback


def test_only_sin_falls_back():
    result = SpyrePartitioner(mode="allowlist").partition(
        _edge_program(MixedUnsupported())
    )
    compute, tagged, fallback = _tag_split(result)

    assert len(compute) > 0
    # Exactly aten::sin leaves the delegate; nothing else.
    assert {_aten_name(n) for n in fallback} == {"aten::sin"}
    # The fc (addmm/permute_copy) + relu neighbours all delegate, and every
    # delegated node is on the device-validated allowlist.
    assert len(tagged) > 0
    delegated_names = {_aten_name(n) for n in tagged}
    assert delegated_names <= _SUPPORTED_ATEN_OPS
    assert {"aten::addmm", "aten::permute_copy", "aten::relu"} <= delegated_names


def test_two_delegate_partitions():
    # The interior CPU sin cuts the graph into two delegate calls (fc before,
    # relu after) -- the structural claim the device run relies on.
    result = SpyrePartitioner(mode="allowlist").partition(
        _edge_program(MixedUnsupported())
    )
    assert len(result.partition_tags) == 2


def test_silence_guarantee(caplog):
    # The guide's headline contract: no silent coverage gaps. Every compute node
    # left on CPU must be logged. Assert one "CPU fallback:" line per fallback
    # node, and that the summary reports exactly one node on CPU.
    edge = _edge_program(MixedUnsupported())
    with caplog.at_level(
        logging.INFO, logger="executorch.backends.spyre.spyre_partitioner"
    ):
        result = SpyrePartitioner(mode="allowlist").partition(edge)
    _, _, fallback = _tag_split(result)

    fallback_lines = [
        r.getMessage() for r in caplog.records if "CPU fallback:" in r.getMessage()
    ]
    assert len(fallback_lines) == len(fallback)
    assert any("sin" in line for line in fallback_lines)
    assert any("root=torch.sin" in line for line in fallback_lines)
    # The summary line reports the CPU count truthfully.
    assert any(f"{len(fallback)} on CPU" in r.getMessage() for r in caplog.records)
