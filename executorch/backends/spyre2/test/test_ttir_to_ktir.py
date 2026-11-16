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
"""Stage 1 host-only tests: TTIR->KTIR lowering + structural verification.

Pure Python — no torch / triton / MLIR / Spyre device. Reads the checked-in TTIR
fixture and the golden KTIR, so these run on the Mac with nothing but stdlib.
"""

import os

import pytest

from executorch.backends.spyre2.kernels.mm_fp16 import MM_K, MM_M, MM_N
from executorch.backends.spyre2.ktir_verify import (
    MATMUL_SIGNATURE,
    canonical_signature,
    split_functions,
    verify_matmul,
)
from executorch.backends.spyre2.ttir_to_ktir import lower_mm_ttir_to_ktir

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TTIR_FIXTURE = os.path.join(_PKG_DIR, "fixtures", "ttir", "mm_fp16_64.ttir")
_KTIR_GOLDEN = os.path.join(
    _PKG_DIR, "fixtures", "ktir", "level4c_matmul_mlp.mlir"
)


@pytest.fixture(scope="module")
def ttir_text() -> str:
    with open(_TTIR_FIXTURE) as f:
        return f.read()


@pytest.fixture(scope="module")
def golden_ktir() -> str:
    with open(_KTIR_GOLDEN) as f:
        return f.read()


# --- fixtures exist and match the MVP grammar ------------------------------


def test_ttir_fixture_has_single_mm_tile_grammar(ttir_text):
    # The lowering's grammar gate expects exactly one tt.dot, >=2 tt.load, >=1
    # tt.store. If the fixture drifts from that, the lowering would silently
    # emit wrong KTIR, so assert it here too.
    assert ttir_text.count("tt.dot") == 1
    assert ttir_text.count("tt.load") >= 2
    assert ttir_text.count("tt.store") >= 1


# --- lowering emits the canonical matmul KTIR ------------------------------


def test_lower_emits_matmul_signature(ttir_text):
    ktir = lower_mm_ttir_to_ktir(ttir_text)
    funcs = split_functions(ktir)
    assert len(funcs) == 1, f"expected one emitted function, got {sorted(funcs)}"
    (_, body), = funcs.items()
    assert canonical_signature(body) == MATMUL_SIGNATURE


def test_lower_uses_single_stick_types_for_64_tile(ttir_text):
    # M=K=N=64 => one 64-elem stick along K and N: activation <64x1xvf16>,
    # weight <1x1xvf16>, output <64x1xvf16>.
    ktir = lower_mm_ttir_to_ktir(ttir_text)
    assert "vector<64xf16>" in ktir
    assert f"{MM_M}x1xvector<64xf16>" in ktir  # activation + output
    assert "1x1xvector<64xf16>" in ktir  # weight (K/64 x N/64)


def test_emitted_ktir_verifies_against_golden(ttir_text, golden_ktir):
    # The Stage 1 oracle: the emitted mm reduces to the same op grammar as a
    # golden matmul function, and uses the vector<64xf16> stick type.
    ktir = lower_mm_ttir_to_ktir(ttir_text)
    verify_matmul(ktir, golden_ktir)  # raises on structural drift


def test_golden_contains_a_matmul_signature(golden_ktir):
    # Guards the oracle itself: at least one golden function must reduce to the
    # canonical matmul signature, else verify_matmul would be vacuous.
    sigs = {n: canonical_signature(b) for n, b in split_functions(golden_ktir).items()}
    assert any(s == MATMUL_SIGNATURE for s in sigs.values()), sigs


# --- grammar gate rejects malformed / non-mm TTIR -------------------------


def test_lower_rejects_ttir_without_dot():
    bad = "module {\n  tt.func @k() {\n    tt.load %p\n    tt.store %q\n    tt.return\n  }\n}"
    with pytest.raises(ValueError, match="tt.dot"):
        lower_mm_ttir_to_ktir(bad)


def test_lower_rejects_ttir_with_too_few_loads():
    bad = (
        "module {\n  tt.func @k() {\n    tt.load %a\n    tt.dot %a, %a\n"
        "    tt.store %c\n    tt.return\n  }\n}"
    )
    with pytest.raises(ValueError, match="tt.load"):
        lower_mm_ttir_to_ktir(bad)


def test_lower_rejects_non_stick_aligned_shape(ttir_text):
    # A K that is not a multiple of the 64-element stick must fail loudly.
    with pytest.raises(ValueError, match="stick"):
        lower_mm_ttir_to_ktir(ttir_text, k=96)


# --- verify_matmul catches structural drift --------------------------------


def test_verify_rejects_missing_store(golden_ktir):
    # Drop the store from an otherwise-valid emission: signature no longer
    # matches MATMUL_SIGNATURE.
    ktir = lower_mm_ttir_to_ktir_no_store()
    with pytest.raises(AssertionError):
        verify_matmul(ktir, golden_ktir)


def lower_mm_ttir_to_ktir_no_store() -> str:
    """A deliberately broken single-function KTIR (matmul but no store)."""
    return (
        "module {\n"
        "func.func @mm() {\n"
        "  %t = ktdp.get_compute_tile_id : index\n"
        "  %v = ktdp.construct_memory_view @a {stick = vector<64xf16>}\n"
        "  %d = ktdp.load %v : tensor<64x1xvector<64xf16>>\n"
        "  %r = linalg.matmul %d, %d : ... -> ...\n"
        "  return\n"
        "}\n"
        "}\n"
    )
