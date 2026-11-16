"""Spyre2 SDSC bridge regressions that require physical Spyre hardware."""

import pytest
from executorch.exir.backend.backend_details import CompileSpec

from executorch.backends.spyre2 import Spyre2Backend
from executorch.backends.spyre2.test.test_preprocess_spyre2 import (
    _edge_program,
    _unpack,
)

pytestmark = pytest.mark.spyre


def test_sdsc_preprocess_annotates_ttir_provenance():
    edge = _edge_program()
    result = Spyre2Backend.preprocess(edge, [CompileSpec("spyre2_artifact", b"sdsc")])
    header, _ = _unpack(bytes(result.processed_bytes))

    assert header["artifact"] == "sdsc"
    assert header["version"] in (1, 2)
    assert header["ttir_provenance"] == {
        "kernel": "spyre2::mm_fp16",
        "source": "triton-ttir",
    }
