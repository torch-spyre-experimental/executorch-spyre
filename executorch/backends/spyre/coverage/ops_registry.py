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
"""The data-driven op registry for the Spyre partitioner.

``supported_ops.yaml`` (next to this file) is the checked-in **source of
truth** for which core-aten ops have (or have not) been validated on a Spyre
device. Each entry::

    - name: "aten::mm"       # core-aten op.name()
      validated: true         # confirmed on a Spyre device end-to-end?
      since_phase: 2          # phase that validated / added it (informational)
      note: "64x64 fp16 matmul"

It is used in both partition modes (see ``docs/operator-guide.md``):

* ``allowlist`` mode delegates **only** ``validated: true`` ops (see
  :func:`validated_op_names`).
* ``fallbacklist`` mode (the production default) delegates everything torch-spyre
  does not declare a CPU fallback for, **except** the ``validated: false`` ops
  (see :func:`unvalidated_op_names`): those stay on CPU in both modes.

An op is claimed only after an on-device compile+run has confirmed SDSC accepts
it and the result matches. The coverage harness (:mod:`run_coverage`) is what
flips ``validated`` from ``false`` to ``true`` on a green device run;
unvalidated entries are the queue.

Loading uses :mod:`_yaml_compat` (PyYAML when present, strict subset parser
otherwise) so importing the partitioner adds no hard PyYAML dependency.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ._yaml_compat import safe_load

_SUPPORTED_OPS_FILE = Path(__file__).with_name("supported_ops.yaml")


@dataclass(frozen=True)
class SupportedOp:
    """One entry from ``supported_ops.yaml``."""

    name: str
    validated: bool
    since_phase: Optional[int] = None
    note: str = ""


def _parse(doc: dict, source: str) -> list[SupportedOp]:
    if not isinstance(doc, dict):
        raise ValueError(f"{source}: top level must be a mapping, got {type(doc)}")
    ops_raw = doc.get("ops")
    if not isinstance(ops_raw, list):
        raise ValueError(f"{source}: missing or non-list 'ops:' key")
    ops: list[SupportedOp] = []
    seen: set[str] = set()
    for i, entry in enumerate(ops_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: ops[{i}] is not a mapping: {entry!r}")
        name = entry.get("name")
        if not isinstance(name, str) or "::" not in name:
            raise ValueError(
                f"{source}: ops[{i}].name must be a core-aten name like "
                f"'aten::mm', got {name!r}"
            )
        if name in seen:
            raise ValueError(f"{source}: duplicate op entry {name!r}")
        seen.add(name)
        validated = entry.get("validated", False)
        if not isinstance(validated, bool):
            raise ValueError(
                f"{source}: ops[{i}].validated must be a bool, got {validated!r}"
            )
        since_phase = entry.get("since_phase")
        if since_phase is not None and not isinstance(since_phase, int):
            raise ValueError(f"{source}: ops[{i}].since_phase must be an int or absent")
        ops.append(
            SupportedOp(
                name=name,
                validated=validated,
                since_phase=since_phase,
                note=str(entry.get("note", "")),
            )
        )
    return ops


@functools.lru_cache(maxsize=None)
def load_supported_ops(path: Optional[str] = None) -> tuple[SupportedOp, ...]:
    """Load and validate ``supported_ops.yaml`` (cached).

    ``path`` overrides the default file (used by tests). Returns a tuple so the
    result is hashable/immutable and safe to cache.
    """
    p = Path(path) if path is not None else _SUPPORTED_OPS_FILE
    doc = safe_load(p.read_text(encoding="utf-8"))
    return tuple(_parse(doc, source=str(p)))


def validated_op_names(path: Optional[str] = None) -> frozenset[str]:
    """The set of op names the partitioner may delegate (``validated: true``)."""
    return frozenset(op.name for op in load_supported_ops(path) if op.validated)


def unvalidated_op_names(path: Optional[str] = None) -> frozenset[str]:
    """Ops listed with ``validated: false``: never delegated in either mode."""
    return frozenset(op.name for op in load_supported_ops(path) if not op.validated)


def all_op_names(path: Optional[str] = None) -> frozenset[str]:
    """Every op name in the file, validated or not."""
    return frozenset(op.name for op in load_supported_ops(path))
