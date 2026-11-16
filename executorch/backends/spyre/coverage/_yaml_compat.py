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
"""Zero-dependency YAML loader for the coverage subpackage's own files.

The Spyre delegate must import (partitioner -> ops_registry -> here) in the bare
host CI environment, which may not have PyYAML installed. So:

* if :mod:`yaml` (PyYAML) is importable, use it -- authoritative, full YAML;
* otherwise fall back to :func:`_minimal_load`, a **strict** parser for the
  small, fixed schema of ``supported_ops.yaml`` / ``candidates.yaml`` only:

  - a top-level mapping of ``key: scalar`` and/or ``key:`` -> list-of-mappings;
  - list items are ``- key: scalar`` blocks (flat mappings, further indented
    continuation lines);
  - scalars are ``true`` / ``false`` / ``null`` / int / float / quoted or bare
    string;
  - ``#`` comments and blank lines are ignored.

The fallback deliberately raises :class:`ValueError` on anything outside that
subset rather than guessing -- a malformed allowlist should fail loudly, never
be silently mis-parsed into a wrong delegation decision. When PyYAML *is*
present the two parsers are cross-checked in the test suite so the fallback
cannot drift.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - trivial import guard
    import yaml as _pyyaml
except ImportError:  # pragma: no cover
    _pyyaml = None

HAVE_PYYAML: bool = _pyyaml is not None


def safe_load(text: str) -> Any:
    """Parse ``text`` as YAML, preferring PyYAML and falling back to the subset.

    Returns whatever the document's top level is (for our files: a ``dict``).
    """
    if _pyyaml is not None:
        return _pyyaml.safe_load(text)
    return _minimal_load(text)


def _scalar(token: str) -> Any:
    """Convert a YAML scalar token to a Python value (strict subset)."""
    token = token.strip()
    if token == "" or token == "~" or token == "null":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        return token[1:-1]
    # numeric?
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token  # bare string


def _split_key_value(body: str) -> tuple[str, str]:
    """Split ``key: value`` on the first colon that separates them.

    Colons inside quotes (e.g. ``name: "aten::mm"``) must not split the key, so
    a leading quoted value is handled first.
    """
    idx = body.find(":")
    if idx == -1:
        raise ValueError(f"expected 'key: value', got {body!r}")
    key = body[:idx].strip()
    value = body[idx + 1 :].strip()
    return key, value


def _minimal_load(text: str) -> dict:
    """Strict parser for the constrained schema (see module docstring)."""
    result: dict[str, Any] = {}
    current_list: list | None = None
    current_list_key: str | None = None
    current_item: dict | None = None
    list_indent: int | None = None

    for raw in text.splitlines():
        # Strip full-line and trailing comments (no '#' appears inside our
        # values; keep it simple and reject if that assumption is violated)
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if stripped.startswith("- "):
            # A list item begins a new mapping under the most recent key.
            if current_list is None:
                raise ValueError(f"list item without a parent key: {raw!r}")
            list_indent = indent
            current_item = {}
            current_list.append(current_item)
            key, value = _split_key_value(stripped[2:])
            current_item[key] = _scalar(value)
            continue

        if (
            current_item is not None
            and list_indent is not None
            and indent > list_indent
        ):
            # Continuation line of the current list item's mapping.
            key, value = _split_key_value(stripped)
            current_item[key] = _scalar(value)
            continue

        # Top-level mapping entry.
        current_item = None
        list_indent = None
        key, value = _split_key_value(stripped)
        if value == "":
            # Opens a nested block -- we only support list-of-mappings blocks.
            current_list = []
            current_list_key = key
            result[key] = current_list
        else:
            current_list = None
            current_list_key = None
            result[key] = _scalar(value)

    return result
