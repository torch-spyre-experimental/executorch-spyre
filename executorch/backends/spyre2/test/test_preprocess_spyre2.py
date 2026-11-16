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
"""spyre2 ``preprocess`` + routing tests.

Host-only (torch + executorch, no triton / no Spyre device):
- ``Spyre2Backend`` resolves under the id its partitioner emits (the
  ``backend_id == cls.__name__`` contract), distinct from spyre's;
- the Stage 1 KTIR ``preprocess`` frames a parseable, verified payload;
- ``_parse_artifact`` selects the stage.

The Stage 2 SDSC ``preprocess`` end-to-end lives in the sibling ``test_spyre``
suite.
"""

import io
import json
import struct
import tarfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir import to_edge  # noqa: E402
from executorch.exir.backend.backend_details import (  # noqa: E402
    BackendDetails,
    CompileSpec,
)

from executorch.backends.spyre2 import (  # noqa: E402
    SPYRE2_BACKEND_ID,
    Spyre2Backend,
    Spyre2Partitioner,
)
from executorch.backends.spyre2.spyre2_backend import _parse_artifact  # noqa: E402


class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


def _edge_program(m=64, k=64, n=64):
    mod = MatMul().eval()
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    exported = torch.export.export(mod, (a, b))
    return to_edge(exported).exported_program()


def _unpack(payload: bytes):
    (hlen,) = struct.unpack_from("<I", payload, 0)
    header = json.loads(payload[4 : 4 + hlen].decode("utf-8"))
    tar_bytes = payload[4 + hlen :]
    return header, tar_bytes


# --- backend id / resolution contract --------------------------------------


def test_backend_id_matches_class_name():
    # ExecuTorch's to_backend matches DelegationSpec.backend_id against
    # cls.__name__ over BackendDetails.__subclasses__(); the id MUST equal the
    # class name or Spyre2Backend.preprocess is never invoked.
    assert SPYRE2_BACKEND_ID == Spyre2Backend.__name__ == "Spyre2Backend"


def test_spyre2_id_distinct_from_spyre():
    # A shared id would collide on two same-named subclasses and route
    # nondeterministically. spyre2 must be its own id.
    from executorch.backends.spyre.spyre_backend import SPYRE_BACKEND_ID

    assert SPYRE2_BACKEND_ID != SPYRE_BACKEND_ID


def test_spyre2_backend_is_registered_subclass():
    # Importing executorch.backends.spyre2 must make Spyre2Backend discoverable to the resolver.
    names = {c.__name__ for c in BackendDetails.__subclasses__()}
    assert "Spyre2Backend" in names


def test_partitioner_tags_spyre2_backend_id():
    # The re-pointed partitioner must delegate to "Spyre2Backend", not spyre's id.
    part = Spyre2Partitioner()
    assert part.delegation_spec.backend_id == SPYRE2_BACKEND_ID


def test_partitioner_forwards_compile_specs():
    specs = [CompileSpec("spyre2_artifact", b"ktir")]
    part = Spyre2Partitioner(specs)
    assert part.delegation_spec.compile_specs == specs


# --- _parse_artifact --------------------------------------------------------


def test_parse_artifact_defaults_to_sdsc():
    assert _parse_artifact([]) == "sdsc"
    assert _parse_artifact(None) == "sdsc"


def test_parse_artifact_reads_ktir():
    assert _parse_artifact([CompileSpec("spyre2_artifact", b"ktir")]) == "ktir"


def test_parse_artifact_rejects_unknown():
    with pytest.raises(ValueError, match="sdsc.*ktir|ktir"):
        _parse_artifact([CompileSpec("spyre2_artifact", b"bogus")])


# --- Stage 1 KTIR preprocess (host-only) -----------------------------------


def test_ktir_preprocess_frames_verified_payload():
    edge = _edge_program()
    specs = [CompileSpec("spyre2_artifact", b"ktir")]
    result = Spyre2Backend.preprocess(edge, specs)
    header, tar_bytes = _unpack(bytes(result.processed_bytes))

    assert header["version"] == 1
    assert header["backend"] == SPYRE2_BACKEND_ID
    assert header["artifact"] == "ktir"
    assert header["device_runnable"] is False
    assert header["n_inputs"] == 2  # a, b (no lifted params for a bare matmul)
    assert header["n_outputs"] == 1

    # The tar carries the KTIR MLIR, and it re-opens.
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
        names = tf.getnames()
        assert header["ktir_arcname"] in names
        ktir = tf.extractfile(header["ktir_arcname"]).read().decode("utf-8")
    assert "linalg.matmul" in ktir
    assert "vector<64xf16>" in ktir


def test_ktir_payload_roundtrips_through_pack():
    # The header must JSON-roundtrip byte-exactly (the C++ delegate re-parses it).
    edge = _edge_program()
    result = Spyre2Backend.preprocess(edge, [CompileSpec("spyre2_artifact", b"ktir")])
    payload = bytes(result.processed_bytes)
    header, tar_bytes = _unpack(payload)
    # Re-frame and compare: same header dict, same tar tail.
    from executorch.backends.spyre.spyre_backend import _pack_payload

    assert _pack_payload(header, tar_bytes) == payload
