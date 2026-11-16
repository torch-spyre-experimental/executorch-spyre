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
"""Hand-written fp16 ``mm`` Triton kernel — the TTIR source of truth for spyre2.

This is the spyre2 analogue of ExecuTorch's in-repo CUDA Triton templates
(``repos/executorch/backends/cuda/triton/kernels/int4_matmul.py``): a
``@triton.jit`` kernel wrapped as a ``@triton_op`` with a ``register_fake``, so
it participates in ``torch.export`` / AOTInductor exactly like the GPU path.

Why hand-written rather than Inductor's autotuned GEMM template: we want a
**deterministic, single-program** TTIR with no autotune permutations and
``tl.arange``-based loads (not ``make_tensor_descriptor``), so each ``tt.load``
maps cleanly onto one ``ktdp.construct_memory_view`` + ``ktdp.load`` in the
Stage 1 TTIR→KTIR lowering, and ``tl.dot`` maps onto a single ``linalg.matmul``.
``aten.mm`` is *not* in ExecuTorch's ``EDGE_TO_TRITON_KERNELS`` (only sdpa/topk),
so there is no upstream mm kernel to conflict with — we own the TTIR shape.

MVP shape: a single 64x64x64 fp16 tile. On Spyre one stick = 64 fp16 elements,
so K and N each collapse to exactly one stick — the degenerate single-core case
of the golden ``level4c_matmul_mlp.mlir`` tiling. The kernel therefore uses a
single program (grid ``(1,)``) with no K-loop and no block/tile arithmetic.

Import side effects: importing this module registers ``triton::mm_fp16`` in the
``torch.library``. It requires the ``triton`` wheel; guard imports at call sites
that must run without it (the KTIR structural tests read a checked-in fixture and
never import this module).
"""

import torch

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - host without the triton wheel
    _HAS_TRITON = False


# Fixed MVP tile: one stick along K and N (64 fp16 elems == 1 stick).
MM_M = 64
MM_K = 64
MM_N = 64


if _HAS_TRITON:

    @triton.jit
    def _mm_fp16_kernel(
        A,  # [M, K] fp16 activations
        B,  # [K, N] fp16 weights
        C,  # [M, N] fp16 output
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
    ):
        """C[M,N] = A[M,K] @ B[K,N], one program computes the whole tile.

        Single-tile MVP: ``M==N==K==64`` fit one stick along K and N, so there is
        no program-id fan-out and no K-loop — exactly the structure the Stage 1
        lowering expects (one load per operand, one ``tl.dot``, one store).
        """
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        offs_k = tl.arange(0, K)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs)  # [M, K] fp16
        b = tl.load(b_ptrs)  # [K, N] fp16

        # fp16 inputs, fp32 accumulate, round back to fp16 (matches the CPU
        # reference in export_matmul.py, which accumulates in fp32).
        acc = tl.dot(a, b, out_dtype=tl.float32)

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(tl.float16))

    @triton_op("spyre2::mm_fp16", mutates_args={})
    def mm_fp16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """fp16 ``mm``: ``out = a @ b`` for the fixed 64x64x64 MVP tile."""
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"inner dims disagree: {a.shape} @ {b.shape}"
        out = torch.empty(M, N, dtype=torch.float16, device=a.device)
        wrap_triton(_mm_fp16_kernel)[(1,)](
            a,
            b,
            out,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
        )
        return out

    @mm_fp16.register_fake
    def _mm_fp16_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, _ = a.shape
        _, N = b.shape
        return torch.empty(M, N, dtype=torch.float16, device=a.device)
