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
"""Tests that need torch-spyre but NOT a physical device.

The parent ``test_spyre/conftest.py`` has an autouse session fixture,
``require_physical_spyre``, that raises unless ``FLEX_DEVICE=PF`` and a device
probe succeeds. Tests here run the torch-spyre *frontend* only
(``TORCH_SPYRE_FRONTEND_ONLY``), so this directory overrides that one fixture
with a check that torch-spyre is importable. Device tests elsewhere in
``test_spyre/`` are unaffected: a fixture override only applies below this
directory.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_physical_spyre():
    """Override: frontend-only tests need torch-spyre, not a device."""
    try:
        import torch_spyre
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"torch_spyre is not importable: {e}", allow_module_level=False)
    # TORCH_DEVICE_BACKEND_AUTOLOAD=0 leaves the device and compile wrapper
    # unregistered; _autoload() is idempotent.
    torch_spyre._autoload()
