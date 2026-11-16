"""Mixed CPU/Spyre compilation regressions."""

import pytest
import torch

from executorch.exir import to_edge

from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner
from executorch.backends.spyre.export_mixed import MixedUnsupported

pytestmark = pytest.mark.spyre


def test_pte_builds():
    mod = MixedUnsupported().eval().half()
    x = torch.randn(1, 64, dtype=torch.float16)
    exported = torch.export.export(mod, (x,))
    program = to_edge(exported).to_backend(SpyrePartitioner()).to_executorch()
    assert len(program.buffer) > 0
