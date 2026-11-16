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
"""Stage 1 (host-only): faithful TTIR -> KTIR (``ktdp``) lowering for ``mm``.

This is the "from TTIR, provide a Spyre-specific implementation" half. It
mirrors the four TTIR->KTIR passes documented in the knowledgebase
(``triton-compiler-pipeline.md``: LowerDescriptorMemory / LowerComputeOps /
ConvertFunctions / DistributeWork) for a single ``mm`` tile:

    tt.func            -> func.func
    tt.get_program_id  -> ktdp.get_compute_tile_id            (DistributeWork)
    tt.load  A / B     -> ktdp.construct_memory_view + ktdp.load  (LowerDescriptorMemory)
    tt.dot             -> linalg.matmul                       (LowerComputeOps)
    tt.store C         -> ktdp.construct_memory_view + ktdp.store (LowerDescriptorMemory)
    tt.return          -> return                              (ConvertFunctions)

Scope: the fixed 64x64x64 fp16 MVP tile. On Spyre one stick = 64 fp16 elements,
so K and N each collapse to a single stick — the degenerate single-core case of
the golden ``level4c_matmul_mlp.mlir`` (whose ``%row_start = tile_id * 16``
tiling is the multi-core generalization we deliberately omit here).

**This lowering is host-only and is NOT device-runnable on the shipped SDK 2.0.**
The shipped ``hcc-opt`` registers no ``ktdp`` / ``linalg`` / ``tt`` dialects and
``SpyreAsyncCompile.triton()`` raises, so there is no compiler or runtime for a
KTIR artifact. The emitted MLIR is verified *structurally* (``ktir_verify``)
against the hand-authored golden and framed into spyre's payload container so the
seam is ready when upstream ships the ktdp backend. Device execution of ``mm``
today goes through the Stage 2 SDSC bridge (``spyre2_backend``), not this file.
"""

import re

from executorch.backends.spyre2.kernels.mm_fp16 import MM_K, MM_M, MM_N

STICK = 64  # fp16 elements per stick (128-byte aligned) on Spyre.


def _require_ttir_grammar(ttir_text: str) -> None:
    """Assert the captured TTIR has the single-``mm``-tile op grammar we lower.

    We do not re-parse TTIR into an AST (the ``tt`` textual format drifts across
    triton releases — plan risk R1). Instead we confirm the ops the lowering
    consumes are present exactly as expected for one tile: a ``tt.dot``, two
    loads, and a store. A mismatch means the kernel changed shape and the
    textual lowering below would silently emit wrong KTIR, so we fail loudly.

    Comment lines (``//`` — both triton's own header and any prose in a fixture)
    are stripped before counting so op tokens mentioned in prose never inflate
    the tallies.
    """
    code = "\n".join(
        line for line in ttir_text.splitlines() if not line.lstrip().startswith("//")
    )
    n_dot = len(re.findall(r"\btt\.dot\b", code))
    n_load = len(re.findall(r"\btt\.load\b", code))
    n_store = len(re.findall(r"\btt\.store\b", code))
    problems = []
    if n_dot != 1:
        problems.append(f"expected exactly 1 tt.dot, found {n_dot}")
    if n_load < 2:
        problems.append(f"expected >=2 tt.load (A, B), found {n_load}")
    if n_store < 1:
        problems.append(f"expected >=1 tt.store (C), found {n_store}")
    if problems:
        raise ValueError(
            "TTIR does not match the single-mm-tile grammar this lowering "
            "handles: " + "; ".join(problems) + ". The mm kernel or triton "
            "version likely changed; re-check ttir_capture / the fixture."
        )


def _sticks(elems: int, axis: str) -> int:
    if elems % STICK != 0:
        raise ValueError(
            f"{axis} dim {elems} is not a multiple of the {STICK}-element stick; "
            "the single-tile MVP lowering requires stick-aligned dims."
        )
    return elems // STICK


def lower_mm_ttir_to_ktir(
    ttir_text: str,
    *,
    m: int = MM_M,
    k: int = MM_K,
    n: int = MM_N,
    func_name: str = "mm",
) -> str:
    """Lower captured ``mm`` TTIR to a ``ktdp``-dialect KTIR MLIR string.

    ``ttir_text`` is validated for the expected op grammar; the emitted KTIR is
    derived from the (fixed, stick-aligned) ``m``/``k``/``n`` shape. Stick
    layout, mirroring the golden: activation ``<M x K/64>``, weight
    ``<K/64 x N/64>``, output ``<M x N/64>``, each element a ``vector<64xf16>``.
    """
    _require_ttir_grammar(ttir_text)

    k_sticks = _sticks(k, "K")
    n_sticks = _sticks(n, "N")
    act_ty = f"{m}x{k_sticks}xvector<{STICK}xf16>"
    wt_ty = f"{k_sticks}x{n_sticks}xvector<{STICK}xf16>"
    out_ty = f"{m}x{n_sticks}xvector<{STICK}xf16>"

    return f"""// KTIR lowered from TTIR by executorch.backends.spyre2.ttir_to_ktir (host-only).
// mm: [{m}x{k}] x [{k}x{n}] -> [{m}x{n}] fp16, single {STICK}-elem-stick tile.
// NOT device-runnable on shipped SDK 2.0 (no ktdp/linalg/tt in hcc-opt);
// structurally verified against backends/spyre2/fixtures/ktir/level4c_matmul_mlp.mlir.
module {{

func.func @{func_name}() {{
  %tile_id = ktdp.get_compute_tile_id : index

  %view_act = ktdp.construct_memory_view @activation
      [0 : {m}, 0 : {k_sticks}]
      {{memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<{STICK}xf16>}}
      : ktdp.memory_view<{act_ty}>

  %view_weight = ktdp.construct_memory_view @weight
      [0 : {k_sticks}, 0 : {n_sticks}]
      {{memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<{STICK}xf16>}}
      : ktdp.memory_view<{wt_ty}>

  %view_output = ktdp.construct_memory_view @output
      [0 : {m}, 0 : {n_sticks}]
      {{memory_space = #ktdp.memory_space<off_chip>,
       stick = vector<{STICK}xf16>}}
      : ktdp.memory_view<{out_ty}>

  %act_data = ktdp.load %view_act : tensor<{act_ty}>
  %wt_data  = ktdp.load %view_weight : tensor<{wt_ty}>

  %result = linalg.matmul %act_data, %wt_data
      : tensor<{act_ty}>, tensor<{wt_ty}>
        -> tensor<{out_ty}>

  ktdp.store %result, %view_output : tensor<{out_ty}>

  return
}}

}} // module
"""
