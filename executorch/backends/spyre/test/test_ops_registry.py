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
"""Unit tests for the data-driven op allowlist (Phase 5).

Dependency-free: these need **neither** ``torch``/``executorch`` **nor**
``PyYAML`` — that is the whole point. ``coverage/_yaml_compat.py`` must parse
``supported_ops.yaml`` in the bare host-CI interpreter so importing the
partitioner never drags a new dependency into every ExecuTorch export. When
PyYAML *is* installed the two parsers are cross-checked so the fallback cannot
silently drift from real YAML semantics.
"""

import textwrap

import pytest

from executorch.backends.spyre.coverage import _yaml_compat
from executorch.backends.spyre.coverage._yaml_compat import _minimal_load, safe_load
from executorch.backends.spyre.coverage.ops_registry import (
    _SUPPORTED_OPS_FILE,
    SupportedOp,
    all_op_names,
    load_supported_ops,
    validated_op_names,
)


# --- registry contract -----------------------------------------------------


def test_validated_core_ops_present():
    # The four ops confirmed end-to-end on a Spyre device (Phase 2/3).
    validated = validated_op_names()
    for op in ("aten::mm", "aten::addmm", "aten::relu", "aten::permute_copy"):
        assert op in validated


def test_validated_is_subset_of_all():
    assert validated_op_names() <= all_op_names()


def test_only_flagged_ops_are_validated():
    # Nothing is delegated unless explicitly validated: true.
    ops = load_supported_ops()
    validated = validated_op_names()
    for op in ops:
        assert (op.name in validated) == op.validated


def test_partitioner_uses_registry():
    # The partitioner's frozenset must be exactly the validated set. Guarded on
    # executorch since importing the partitioner pulls in the ET backend stack.
    pytest.importorskip("executorch")
    from executorch.backends.spyre.spyre_partitioner import _SUPPORTED_ATEN_OPS

    assert _SUPPORTED_ATEN_OPS == validated_op_names()


def test_entries_are_well_formed():
    for op in load_supported_ops():
        assert isinstance(op, SupportedOp)
        assert op.name.startswith("aten::")
        assert isinstance(op.validated, bool)


# --- fallback parser: schema enforcement -----------------------------------


def test_minimal_load_parses_list_of_mappings():
    doc = _minimal_load(
        textwrap.dedent(
            """
            ops:
              - name: "aten::mm"
                validated: true
                since_phase: 2
              - name: "aten::bmm"
                validated: false
            """
        )
    )
    assert doc == {
        "ops": [
            {"name": "aten::mm", "validated": True, "since_phase": 2},
            {"name": "aten::bmm", "validated": False},
        ]
    }


def test_minimal_load_scalars_and_comments():
    doc = _minimal_load(
        textwrap.dedent(
            """
            # a comment
            top: 1.5    # trailing comment
            ops:
              - name: 'aten::relu'   # colon-free note
                note: plain string
            """
        )
    )
    assert doc["top"] == 1.5
    assert doc["ops"][0] == {"name": "aten::relu", "note": "plain string"}


def test_minimal_load_rejects_list_item_without_parent():
    with pytest.raises(ValueError):
        _minimal_load("- name: orphan\n")


def test_minimal_load_rejects_missing_colon():
    with pytest.raises(ValueError):
        _minimal_load("ops:\n  - just_a_word\n")


# --- fallback vs PyYAML cross-check ----------------------------------------


def test_fallback_matches_pyyaml_on_supported_ops():
    """When PyYAML is installed, the strict fallback must agree with it byte for
    byte on the real file — the guard against silent drift."""
    if not _yaml_compat.HAVE_PYYAML:
        pytest.skip("PyYAML not installed; nothing to cross-check against")
    import yaml

    text = _SUPPORTED_OPS_FILE.read_text(encoding="utf-8")
    assert _minimal_load(text) == yaml.safe_load(text)


def test_safe_load_uses_pyyaml_when_available():
    text = 'ops:\n  - name: "aten::mm"\n    validated: true\n'
    assert safe_load(text) == {"ops": [{"name": "aten::mm", "validated": True}]}
