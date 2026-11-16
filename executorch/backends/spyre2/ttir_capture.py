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
"""Capture TTIR (Triton IR, the ``tt`` MLIR dialect) for the ``mm`` kernel.

This is the "follow ExecuTorch-GPU down to TTIR" half of spyre2. Triton's
AST→TTIR frontend is **target-independent**: compiling a ``@triton.jit`` kernel
to the ``ttir`` stage needs neither a GPU nor the (upstream-only) Triton Spyre
backend. We stop at TTIR and hand it to the Stage 1 lowering (``ttir_to_ktir``).

The captured TTIR text is checked in as a fixture
(``backends/spyre2/fixtures/ttir/mm_fp16_64.ttir``) with the producing triton
version recorded in a header comment, because the ``tt`` dialect's textual
format drifts across triton releases — the fixture lets the Mac-side lowering
tests run with no triton wheel installed (see risks R1/R2 in the plan).

Usage (on a host with the triton wheel)::

    python -m executorch.backends.spyre2.ttir_capture --out backends/spyre2/fixtures/ttir/mm_fp16_64.ttir
"""

import argparse
import os

TTIR_KEY = "ttir"


def capture_mm_ttir() -> str:
    """Compile the fixed 64x64x64 fp16 ``mm`` kernel to TTIR and return the text.

    Builds a ``triton.compiler.ASTSource`` for ``_mm_fp16_kernel`` and compiles
    only as far as the ``ttir`` stage, reading ``compiled.asm["ttir"]``. Raises a
    clear error if the triton wheel is missing.
    """
    try:
        import triton
        from triton.compiler import ASTSource
    except ImportError as e:  # pragma: no cover - host without triton
        raise RuntimeError(
            "capture_mm_ttir requires the `triton` wheel; install it in the AOT "
            "venv (Linux/pod). Mac-side KTIR tests read the checked-in fixture "
            "instead and do not call this."
        ) from e

    from executorch.backends.spyre2.kernels.mm_fp16 import MM_K, MM_M, MM_N, _mm_fp16_kernel

    # Contiguous row-major strides for the fixed tile.
    strides = {
        "stride_am": MM_K,
        "stride_ak": 1,
        "stride_bk": MM_N,
        "stride_bn": 1,
        "stride_cm": MM_N,
        "stride_cn": 1,
    }
    # A/B/C are fp16 pointers; the dimension + stride params are tl.constexpr.
    signature = {
        "A": "*fp16",
        "B": "*fp16",
        "C": "*fp16",
        "M": "constexpr",
        "N": "constexpr",
        "K": "constexpr",
        **{k: "constexpr" for k in strides},
    }
    constexprs = {"M": MM_M, "N": MM_N, "K": MM_K, **strides}

    src = ASTSource(fn=_mm_fp16_kernel, signature=signature, constexprs=constexprs)
    compiled = triton.compile(src)
    ttir = compiled.asm[TTIR_KEY]

    version = getattr(triton, "__version__", "unknown")
    header = (
        f"// TTIR for spyre2 mm_fp16 (M={MM_M} K={MM_K} N={MM_N}, fp16)\n"
        f"// Produced by triton=={version} via executorch.backends.spyre2.ttir_capture.\n"
        f"// Frontend (AST->TTIR) is target-independent: no GPU / Spyre backend.\n"
    )
    return header + ttir


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture mm TTIR to a fixture file.")
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fixtures", "ttir", "mm_fp16_64.ttir",
        ),
        help="Path to write the captured TTIR text.",
    )
    args = ap.parse_args()
    text = capture_mm_ttir()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(text)
    print(f"wrote {args.out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
