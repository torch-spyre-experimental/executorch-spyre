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
"""Pytest fixtures / path setup for the Spyre delegate AOT test suite.

``backends`` is an *implicit namespace package* (there is deliberately no
``backends/__init__.py``), so ``from executorch.backends.spyre import ...`` only resolves
when the ``executorch-spyre`` project root is on ``sys.path``. Tests may be
invoked from any cwd (repo root, ``executorch-spyre/``, or the test dir itself),
so we insert that root here rather than relying on the collector's rootdir.

Layout::

    executorch-spyre/                 <- inserted onto sys.path (this file's 4th parent)
      executorch/backends/spyre/
        spyre_backend.py
        spyre_partitioner.py
        test/
          conftest.py               <- this file
"""

import sys
from pathlib import Path

import pytest

# test/ -> spyre/ -> backends/ -> executorch/ -> executorch-spyre/
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import executorch.backends

    _our_backends = str(_PROJECT_ROOT / "executorch" / "backends")
    if _our_backends not in executorch.backends.__path__:
        executorch.backends.__path__.append(_our_backends)
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _pin_partition_env(monkeypatch):
    """Make the host suite deterministic regardless of the caller's environment.

    The code default is ``fallbacklist`` (the production contract), which needs
    torch-spyre's ``fallback_ops`` and fails closed without it. Host tests have
    no torch-spyre, so tests that do not care about the mode run in ``allowlist``
    and tests that do care pass ``mode=`` explicitly (see
    ``test_partition_modes.py``). Individual tests may override the variable.
    """
    monkeypatch.setenv("EXECUTORCH_SPYRE_PARTITION_MODE", "allowlist")


def pytest_collection_modifyitems(items):
    """Classify every test collected from this directory as host-only."""
    for item in items:
        if Path(item.path).is_relative_to(Path(__file__).parent):
            item.add_marker(pytest.mark.host)
