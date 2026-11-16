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
"""Pytest path setup for the spyre2 AOT test suite.

In the open-source repo, ``backends`` is a regular package rooted at the repo
root. Both ``backends.spyre2`` (this package) and ``backends.spyre`` (spyre, which
spyre2 reuses for the payload framing + partitioner) resolve as long as the repo
root is on ``sys.path``.

Layout::

    executorch-spyre/
      executorch/backends/
        spyre/       <- spyre (reused by executorch.backends.spyre2)
        spyre2/
          test/conftest.py   <- this file
"""

import sys
from pathlib import Path

# test/ -> spyre2/ -> backends/ -> executorch/ -> executorch-spyre/  (the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[4]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import executorch.backends
    _our_backends = str(_REPO_ROOT / "executorch" / "backends")
    if _our_backends not in executorch.backends.__path__:
        executorch.backends.__path__.append(_our_backends)
except ImportError:
    pass


def pytest_collection_modifyitems(items):
    """Classify every test collected from this directory as host-only."""
    import pytest

    for item in items:
        if Path(item.path).is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.host)
