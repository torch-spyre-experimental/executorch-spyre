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
"""Device-safe Rotary Position Embedding (RoPE) for the Spyre delegate.

Phase 8 (8b) blocked-op workaround. Two Granite/Llama RoPE ingredients do not
lower to SDSC as written:

  1. ``aten::cos`` / ``aten::sin`` emit **no SDSC kernel** (8a sweep). But the
     RoPE tables depend only on *position* and *head_dim*, never on activations
     -- so we **host-precompute** ``cos``/``sin`` once and feed them to the
     delegate as fp16 inputs. RoPE on-device then reduces to ``mul``/``add``
     plus the ``rotate_half`` below, all 8a-validated ops.

  2. The textbook ``rotate_half`` is::

         x1, x2 = x[..., :d//2], x[..., d//2:]
         return torch.cat((-x2, x1), dim=-1)          # <-- innermost-dim cat

     and the 8a follow-up proved **innermost/stick-dim ``cat`` scrambles on
     device** (max_abs_err 3.53, 60/64 elements outside the fp16 envelope) --
     an SDSC stick-layout bug. A blanket ``cat`` allowlist entry would silently
     miscompile exactly this. So we express ``rotate_half`` as a **constant
     matmul**: ``rotate_half(x) = x @ R`` where ``R`` is a fixed ``[D, D]``
     sign-permutation matrix. ``matmul`` is 8a-validated; there is no ``cat`` on
     the device path at all.

The 4D ``matmul`` overload is compile-blocked upstream (``insert_bmm_padding``,
torch-spyre #2434), so ``apply_rope`` collapses the leading ``[B, H, S]`` dims
and rotates through a **2D ``mm``** (validated), then restores the shape. The
innermost dim ``D`` -- the stick dim -- is never split or concatenated, which is
why this path is layout-safe where the cat was not.

This module is host/CPU pure PyTorch: the tables and ``R`` are built here (the
``cos``/``sin``/host ``cat`` all run on CPU at trace time). ``apply_rope`` uses
only device-validated ops, so it delegates fully when traced into an attention
graph (device-confirmed in 8c).
"""

from __future__ import annotations

import torch


def rotation_matrix(head_dim: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Constant ``[D, D]`` matrix ``R`` with ``x @ R == rotate_half(x)``.

    ``rotate_half(x) = [-x2, x1]`` where ``x1, x2`` are the two halves of the
    last dim. As a linear map ``(x @ R)[j] = sum_i x[i] R[i, j]``:

      * for ``j in [0, D/2)``  we need ``-x[j + D/2]`` -> ``R[j + D/2, j] = -1``
      * for ``j in [D/2, D)``  we need ``x[j - D/2]``  -> ``R[j - D/2, j] = +1``

    i.e. ``R`` is two signed off-diagonal identity blocks (``+I`` upper-right,
    ``-I`` lower-left). Using a matmul instead of ``cat([-x2, x1], dim=-1)``
    avoids the innermost-dim concat that scrambles on device (8a follow-up).
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    r = torch.zeros(head_dim, head_dim, dtype=dtype)
    k = torch.arange(half)
    r[k + half, k] = -1.0  # lower-left block: (x @ R)[:half] = -x2
    r[k, k + half] = 1.0  # upper-right block: (x @ R)[half:] =  x1
    return r


def build_rope_tables(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host-precompute ``(cos, sin)`` RoPE tables, each ``[seq_len, head_dim]``.

    Non-interleaved (GPT-NeoX / Llama / Granite) layout: the ``head_dim/2``
    inverse frequencies are duplicated across the two halves so ``cos``/``sin``
    line up elementwise with both ``x`` and ``rotate_half(x)``. Computed in fp32
    for accuracy, then cast to ``dtype`` (fp16) for the delegate input -- the
    ``cos``/``sin`` calls run here on CPU, never on device.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, D] -- host-side cat, CPU only
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rot: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to ``x`` using device-validated ops only.

    Args:
        x:   activations with a trailing head dim, any rank ``[..., D]`` (fp16).
             Both the ``[B, H, S, D]`` head-major layout and the head-sliced
             ``[B, S, D]`` layout (8c) are supported -- only the last dim matters.
        cos: ``build_rope_tables`` cosine table, broadcastable to ``x`` (e.g.
             ``[S, D]`` for either layout above; ``[1, 1, S, D]`` also works).
        sin: matching sine table.
        rot: ``rotation_matrix(D)`` constant ``[D, D]``.

    Returns ``x * cos + rotate_half(x) * sin`` with ``rotate_half`` computed as
    ``x @ rot``. All leading dims are collapsed so the rotate is a **2D ``mm``**
    (the 4D ``matmul`` overload is compile-blocked); the innermost/stick dim
    ``D`` is preserved throughout, so the layout stays device-safe. Because the
    input is contiguous, the collapse/restore reshapes have identity ``dim_order``
    (free views -- no ``_clone_dim_order`` contiguity copy).
    """
    d = x.shape[-1]
    x2d = x.reshape(-1, d)  # collapse ALL leading dims; keep stick dim D
    rotated = (x2d @ rot).reshape(x.shape)
    return x * cos + rotated * sin
