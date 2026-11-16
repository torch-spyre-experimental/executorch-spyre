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
"""TTIR capture tests — **gated on the triton wheel** (run on the Linux/pod venv).

The triton wheel does not install on the macOS/arm host (plan risk R2), so these
skip there. On the pod they confirm the AST->TTIR frontend is reachable without a
GPU/Spyre backend and that the captured TTIR matches the single-mm-tile grammar
the Stage 1 lowering consumes — the same grammar the checked-in fixture carries.
"""

import pytest

triton = pytest.importorskip("triton")

from executorch.backends.spyre2.ttir_capture import capture_mm_ttir  # noqa: E402
from executorch.backends.spyre2.ttir_to_ktir import (  # noqa: E402
    _require_ttir_grammar,
    lower_mm_ttir_to_ktir,
)
from executorch.backends.spyre2.ktir_verify import verify_matmul  # noqa: E402


def test_capture_produces_tt_dialect():
    ttir = capture_mm_ttir()
    assert "tt.func" in ttir or "tt.dot" in ttir
    # Version stamped in the header so schema drift (R1) is traceable.
    assert f"triton=={triton.__version__}" in ttir


def test_captured_ttir_matches_mm_tile_grammar():
    ttir = capture_mm_ttir()
    # Must satisfy the lowering's grammar gate (1 dot, >=2 load, >=1 store).
    _require_ttir_grammar(ttir)  # raises on mismatch


def test_captured_ttir_lowers_and_verifies():
    import os

    ttir = capture_mm_ttir()
    ktir = lower_mm_ttir_to_ktir(ttir)
    golden_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fixtures", "ktir", "level4c_matmul_mlp.mlir",
    )
    with open(golden_path) as f:
        verify_matmul(ktir, f.read())
