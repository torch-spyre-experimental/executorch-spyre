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
"""Device-safe attention primitives for the Spyre delegate (Phase 8, 8b).

Two blocked-op workarounds live here, both built from 8a-validated ops so they
delegate fully (device-confirmed in 8c):

  * ``sdpa_math`` -- the fused ``aten::scaled_dot_product_attention`` compile-
    fails on an unsupported stick expression. We force the **math
    decomposition** ``scores = q @ kᵀ * scale (+ mask) -> softmax -> @ v``. The
    4D ``matmul`` overload is *also* compile-blocked upstream
    (``insert_bmm_padding``), so we collapse the ``[B, H]`` head dims into the
    batch axis and run both products as **3D ``bmm``** (8a-validated). GQA is
    handled by regrouping the query heads under their shared KV head (a
    clone-free contiguous reshape), not a ``repeat_kv`` broadcast -- see
    ``sdpa_math`` for why.

  * ``kv_append`` -- ``aten::index_copy`` (the in-place KV-cache write) SIGABRTs
    the standalone compiler. The workaround is a **``cat`` along the sequence
    dim**. The 8a follow-up proved sequence-dim ``cat`` is numerically correct
    on device (the innermost/stick-dim ``cat`` is what scrambles). NOTE: ``cat``
    stays ``validated: false`` (the allowlist is op-name-based and cannot
    distinguish the safe seq-dim case from the unsafe innermost one), so a
    ``kv_append`` used in a delegated graph falls back to CPU (logged) rather
    than miscompiling. It is off the single-forward critical path
    (KV concat is a decode-time concern); provided for the incremental-decode
    path and as the crash-free replacement for ``index_copy``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def sdpa_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Scaled-dot-product attention as explicit math, device-safe.

    Args:
        q: ``[B, Hq, Sq, D]`` query (fp16).
        k: ``[B, Hkv, Sk, D]`` key.
        v: ``[B, Hkv, Sk, D]`` value. ``Hq`` must be a multiple of ``Hkv``
           (grouped-query attention); ``Hq == Hkv`` is plain MHA.
        mask: optional additive mask broadcastable to ``[B, Hq, Sq, Sk]`` (e.g.
              a causal mask with ``-inf`` above the diagonal). Added to the
              scores before softmax.
        scale: softmax temperature; defaults to ``1/sqrt(D)``. Pass a float for
               host use; for a fully-delegated graph pass an **fp16 tensor held
               as a registered buffer** in the caller (see the scale note below).

    Returns ``[B, Hq, Sq, D]``.

    The whole path is reshape + ``transpose_copy`` + ``bmm`` + ``mul`` + ``add``
    + ``_softmax`` -- all 8a-validated. No fused SDPA, no 4D matmul, no ``cat``,
    and (via the head-regroup below) no GQA ``expand_copy``/clone.
    """
    b, hq, sq, d = q.shape
    _, hkv, sk, _ = k.shape
    if hq % hkv != 0:
        raise ValueError(f"Hq ({hq}) must be a multiple of Hkv ({hkv})")
    rep = hq // hkv
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    # How ``scale`` enters the graph decides whether the mul delegates. Passing a
    # Python float is numerically correct but NOT delegation-clean: torch.export
    # lifts the float as an fp32 0-d constant and emits aten::mul.Tensor, so
    # to_edge inserts a dim_order_ops::_to_dim_order_copy to downcast fp32->fp16
    # (off the allowlist -> CPU-fallback split; _to_copy is itself a device
    # numeric-fail op). Creating the scalar tensor *inside* this function
    # (``q.new_tensor(scale)``) fixes the dtype but still trips a
    # dim_order_ops::_clone_dim_order contiguity clone on the run-time-created
    # constant. The clean path is to pass ``scale`` as an fp16 tensor that is
    # module state (a registered buffer) in the caller -- it then rides in like
    # the mask/RoPE tables as a lifted constant with no clone and no cast. So:
    # for a fully-delegated graph, callers MUST pass ``scale`` as an fp16
    # buffer; the float default stays for host numerics.

    # GQA without a repeat_kv broadcast. The textbook path expands each KV head
    # `rep` times (expand_copy) then reshapes -- but that reshape merges a
    # stride-0 broadcast dim, forcing a dim_order_ops::_clone_dim_order
    # contiguity copy that falls off the allowlist. Instead we **regroup the
    # query heads under their shared KV head**: view Q as [B, Hkv, rep, Sq, D]
    # (Hq splits contiguously into (Hkv, rep)) and fold (rep, Sq) into one axis,
    # so each KV head's single bmm serves all `rep` of its query heads. Query
    # head i uses KV head i // rep -- identical to HF repeat_kv -- but every
    # reshape here is a contiguous split/merge of adjacent dims, so there is no
    # clone, no expand, and MHA is just the rep == 1 case.
    qc = q.reshape(b * hkv, rep * sq, d)  # [B*Hkv, rep*Sq, D]
    kc = k.reshape(b * hkv, sk, d)  # [B*Hkv, Sk, D]
    vc = v.reshape(b * hkv, sk, d)

    # scores = q @ kᵀ * scale  -> [B*Hkv, rep*Sq, Sk] -> [B, Hq, Sq, Sk]
    scores = torch.bmm(qc, kc.transpose(-2, -1)) * scale
    scores = scores.reshape(b, hq, sq, sk)
    if mask is not None:
        scores = scores + mask
    probs = torch.softmax(scores, dim=-1)

    # context = probs @ v  -> [B*Hkv, rep*Sq, D] -> [B, Hq, Sq, D]
    ctx = torch.bmm(probs.reshape(b * hkv, rep * sq, sk), vc)
    return ctx.reshape(b, hq, sq, d)


def kv_append(past: torch.Tensor, new: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """Append ``new`` KV to ``past`` along the sequence dim (index_copy-free).

    ``past``/``new`` are ``[B, H, S, D]``; ``dim=2`` is the sequence axis. This
    is the crash-free replacement for ``index_copy`` (which SIGABRTs the
    standalone compiler). The concat is along a non-innermost dim, which the 8a
    follow-up verified is numerically correct on device -- unlike the
    innermost/stick-dim concat. See the module docstring for the delegation
    caveat (``cat`` stays off the allowlist, so this falls back to CPU when it
    appears in a delegated graph).
    """
    if dim == -1 or dim == past.dim() - 1:
        raise ValueError(
            "kv_append must not concat along the innermost/stick dim "
            "(it scrambles on device); use the sequence dim"
        )
    return torch.cat((past, new), dim=dim)
