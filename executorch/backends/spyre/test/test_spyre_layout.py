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
"""Host-only tests for the ET-side layout boundary (``spyre_layout``).

No Spyre device or torch-spyre needed: a ``FakeStl`` stands in for
``SpyreTensorLayout`` (the helpers only read its attributes).  The ``_legacy_*``
functions below are verbatim copies of the pre-refactor implementations from
``spyre_backend.py`` and serve as a differential oracle proving the Phase 1
refactor did not change behaviour.

The golden JSON strings are duplicated in ``SpyreLayoutTest.cpp``; the two suites
therefore pin the same wire bytes from both languages.
"""

import json
import logging
import re
from pathlib import Path

import pytest

from executorch.backends.spyre import spyre_layout
from executorch.backends.spyre.spyre_layout import (
    LAYOUT_FIELDS,
    layout_key,
    layout_spec_from_stl,
    layouts_equal_for_audit,
    resolve_host_strides,
)

_RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


class FakeStl:
    def __init__(
        self,
        device_size,
        stride_map,
        device_dtype=1,
        element_arrangement=0,
        valid_elements=None,
    ):
        self.device_size = device_size
        self.stride_map = stride_map
        self.device_dtype = device_dtype
        self.element_arrangement = element_arrangement
        self.valid_elements = valid_elements or {}


def _stl_tile_major():
    return FakeStl(
        device_size=[2, 64, 64],
        stride_map=[64, 1, 4096],
        device_dtype=3,
        element_arrangement=1,
        valid_elements={(0, 1): 100, (2,): 7},
    )


# Golden wire bytes. MUST stay identical to kGoldenLayoutJson in SpyreLayoutTest.cpp.
GOLDEN_SPEC = {
    "device_size": [2, 64, 64],
    "stride_map": [64, 1, 4096],
    "device_dtype": 3,
    "element_arrangement": 1,
    "valid_elements": [[[0, 1], 100], [[2], 7]],
    "strides": [128, 64, 1],
}
GOLDEN_JSON = (
    '{"device_size": [2, 64, 64], "stride_map": [64, 1, 4096], '
    '"device_dtype": 3, "element_arrangement": 1, '
    '"valid_elements": [[[0, 1], 100], [[2], 7]], "strides": [128, 64, 1]}'
)


# --- verbatim pre-refactor implementations (differential oracle) ------------


def _legacy_spec(stl, strides):
    return {
        "device_size": [int(x) for x in stl.device_size],
        "stride_map": [int(x) for x in stl.stride_map],
        "device_dtype": int(stl.device_dtype),
        "element_arrangement": int(stl.element_arrangement),
        "valid_elements": [
            [[int(d) for d in dims], int(n)] for dims, n in stl.valid_elements.items()
        ],
        "strides": [int(s) for s in strides],
    }


def _legacy_strides(layout, shape):
    shape = [int(s) for s in shape]
    strides = [int(s) for s in (layout.get("strides") or [])]
    if len(strides) != len(shape):
        strides = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * max(shape[d + 1], 1)
    return strides


# --- layout_key oracle --------------------------------------------------------


def _expected_layout_key(layout):
    """Independent restatement of ``layout_key``: every schema field except
    ``valid_elements``, which the key ignores (3f31690)."""
    if layout is None:
        return None
    return (
        tuple(layout["device_size"]),
        tuple(layout["stride_map"]),
        int(layout["device_dtype"]),
        int(layout["element_arrangement"]),
        tuple(layout.get("strides") or ()),
    )


# --- schema ------------------------------------------------------------------


def test_schema_field_names_and_order_are_frozen():
    assert LAYOUT_FIELDS == (
        "device_size",
        "stride_map",
        "device_dtype",
        "element_arrangement",
        "valid_elements",
        "strides",
    )


def test_python_schema_matches_cpp_header():
    """LAYOUT_FIELDS must equal kLayoutFields in runtime/SpyreLayout.h."""
    text = (_RUNTIME / "SpyreLayout.h").read_text()
    block = re.search(r"BEGIN_LAYOUT_FIELDS(.*?)END_LAYOUT_FIELDS", text, re.S).group(1)
    assert tuple(re.findall(r'"([a-z_]+)"', block)) == LAYOUT_FIELDS


def test_spec_keys_are_exactly_the_schema_in_order():
    spec = layout_spec_from_stl(_stl_tile_major(), [128, 64, 1])
    assert tuple(spec.keys()) == LAYOUT_FIELDS


# --- serialization -----------------------------------------------------------


def test_serialize_matches_golden_json_bytes():
    spec = layout_spec_from_stl(_stl_tile_major(), [128, 64, 1])
    assert spec == GOLDEN_SPEC
    # Byte-for-byte payload compatibility: exactly what _pack_payload would emit.
    assert json.dumps(spec) == GOLDEN_JSON


def test_serialize_coerces_to_plain_ints():
    class NumLike:
        def __init__(self, v):
            self.v = v

        def __int__(self):
            return self.v

    stl = FakeStl(
        [NumLike(4), NumLike(8)],
        [NumLike(8), NumLike(1)],
        NumLike(2),
        NumLike(0),
        {(NumLike(0),): NumLike(3)},
    )
    spec = layout_spec_from_stl(stl, [NumLike(8), NumLike(1)])
    assert json.dumps(spec)  # would raise on non-serializable
    assert all(type(x) is int for x in spec["device_size"] + spec["strides"])


def test_serialize_matches_legacy_on_variants():
    for stl, strides in [
        (_stl_tile_major(), [128, 64, 1]),
        (FakeStl([1, 1], [-1, 0]), [1]),
        (FakeStl([128], [1], valid_elements={}), []),
    ]:
        assert layout_spec_from_stl(stl, strides) == _legacy_spec(stl, strides)


def test_device_layout_spec_none_for_non_spyre_tensor():
    torch = pytest.importorskip("torch")
    assert spyre_layout.device_layout_spec(torch.zeros(2)) is None
    assert spyre_layout.device_layout_spec(None) is None


# --- deserialization / round trip -------------------------------------------


def test_golden_json_roundtrips_through_json():
    spec = json.loads(GOLDEN_JSON)
    assert spec == GOLDEN_SPEC
    assert json.dumps(spec) == GOLDEN_JSON


def test_schema_roundtrip_via_fake_stl():
    """spec -> rebuild-STL-args -> spec is the identity (what
    spyre_tensor_with_layout feeds to SpyreTensorLayout)."""
    spec = json.loads(GOLDEN_JSON)
    rebuilt = FakeStl(
        list(spec["device_size"]),
        list(spec["stride_map"]),
        spec["device_dtype"],
        spec["element_arrangement"],
        {tuple(dims): int(n) for dims, n in spec["valid_elements"]},
    )
    assert layout_spec_from_stl(rebuilt, spec["strides"]) == spec


@pytest.mark.parametrize(
    "shape,layout_strides,expected",
    [
        ([4, 8, 16], [128, 16, 1], [128, 16, 1]),  # rank matches -> layout wins
        ([4, 8, 16], [], [128, 16, 1]),  # absent -> row-major
        ([4, 8, 16], [1, 1], [128, 16, 1]),  # rank mismatch -> row-major
        ([2, 0, 3], [], [3, 3, 1]),  # zero dim clamps to 1
        ([], [], []),
        ([5], [], [1]),
    ],
)
def test_resolve_host_strides(shape, layout_strides, expected):
    layout = {"strides": layout_strides}
    assert resolve_host_strides(layout, shape) == expected
    assert resolve_host_strides(layout, shape) == _legacy_strides(layout, shape)


def test_resolve_host_strides_missing_strides_key():
    assert resolve_host_strides({}, [2, 3]) == [3, 1]


# --- equality helpers (audit semantics, NOT SpyreTensorLayout::operator==) ---


def _base():
    return json.loads(GOLDEN_JSON)


def test_layout_key_none():
    assert layout_key(None) is None
    assert layouts_equal_for_audit(None, None)
    assert not layouts_equal_for_audit(None, _base())


def test_layout_key_is_hashable_and_matches_oracle():
    spec = _base()
    assert hash(layout_key(spec)) is not None
    assert layout_key(spec) == _expected_layout_key(spec)


def test_layout_key_ignores_valid_elements(caplog):
    """``valid_elements`` is not part of the key (so it cannot keep the layout
    fixed-point loop from converging); a non-empty value is logged."""
    a = _base()
    for other_valid in (
        list(reversed(a["valid_elements"])),  # order
        [[[0, 1], 99], [[2], 7]],  # value
        [],  # absent
    ):
        b = _base()
        b["valid_elements"] = other_valid
        assert layout_key(a) == layout_key(b)
        assert layouts_equal_for_audit(a, b)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=spyre_layout.logger.name):
        layout_key(_base())
    assert "ignoring valid_elements" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=spyre_layout.logger.name):
        layout_key({**_base(), "valid_elements": []})
    assert "ignoring valid_elements" not in caplog.text


@pytest.mark.parametrize(
    "field,new",
    [
        ("device_size", [2, 64, 32]),
        ("stride_map", [64, 1, 2048]),
        ("device_dtype", 4),
        ("element_arrangement", 0),
        ("strides", [128, 64, 2]),
    ],
)
def test_layout_key_sensitive_to_every_schema_field(field, new):
    a, b = _base(), _base()
    b[field] = new
    assert not layouts_equal_for_audit(a, b)
    assert layout_key(a) != layout_key(b)
    assert layout_key(b) == _expected_layout_key(b)


def test_layout_key_differs_from_stl_operator_eq_on_strides():
    """Documents the KNOWN divergence (out of scope to fix in Phase 1).

    ``SpyreTensorLayout::operator==`` compares only device_size / stride_map /
    device_dtype / element_arrangement.  The audit key additionally compares
    the host strides, so two layouts that are operator==-equal can still be
    audit-different.  Modelled here by mirroring operator==.  (``valid_elements``
    is ignored by both; see ``test_layout_key_ignores_valid_elements``.)
    """

    def stl_operator_eq(x, y):
        return all(
            x[f] == y[f]
            for f in (
                "device_size",
                "stride_map",
                "device_dtype",
                "element_arrangement",
            )
        )

    a, c = _base(), _base()
    c["strides"] = [1, 2, 3]
    assert stl_operator_eq(a, c)
    assert not layouts_equal_for_audit(a, c)


def test_spyre_backend_reexports_same_objects():
    """The private names in spyre_backend are aliases of the spyre_layout helpers,
    so existing imports/call sites keep working and cannot diverge."""
    pytest.importorskip("torch")
    pytest.importorskip("executorch.exir")
    from executorch.backends.spyre import spyre_backend

    assert spyre_backend._layout_key is spyre_layout.layout_key
    assert spyre_backend._device_layout_spec is spyre_layout.device_layout_spec
    assert (
        spyre_backend._spyre_tensor_with_layout is spyre_layout.spyre_tensor_with_layout
    )
    assert spyre_backend._SPYRE_DEVICE == spyre_layout.SPYRE_DEVICE == "spyre"
