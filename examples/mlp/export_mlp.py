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
"""Export a tiny fp16 MLP as a Spyre-delegated ``.pte`` (Phase 3 exit signal).

Module: ``Linear(64,128) -> ReLU -> Linear(128,64)``, fp16, static shape
``[1,64]``. A multi-op graph with real weights -- unlike the Phase 2 single-op
matmul, this exercises:

  * the **op-support partitioner** growing a connected subgraph over
    ``addmm``/``relu``/``permute_copy`` and CPU-falling-back anything else,
  * ``preprocess`` handling **lifted params/buffers**: the subgraph's
    placeholders are the user input ``x`` *plus* the four lifted params
    (``fc1.weight``, ``fc1.bias``, ``fc2.weight``, ``fc2.bias``). ExecuTorch
    stores those params in the ``.pte`` and passes them to the delegate's
    ``execute()`` as EValues in placeholder order, so the on-device harness only
    feeds the single user input ``x``.

Pipeline: ``torch.export`` -> ``to_edge`` -> ``.to_backend(SpyrePartitioner())``
-> ``.to_executorch()``. Also writes ``x.bin`` (user input) and ``ref.bin`` (CPU
fp16 reference) as raw little-endian fp16 for the ``spyre_mlp_runner`` on-device
compare.

Run in the AOT overlay venv. The SDSC codegen inside
``preprocess`` runs during ``to_backend`` and needs the Spyre toolchain
(``dxp_standalone``); on-device execution of the resulting ``.pte`` needs
FLEX_DEVICE=PF.
"""

import argparse
import json
import os

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_details import CompileSpec

from executorch.backends.spyre import SpyrePartitioner

# Phase 7 Track 2: default batch buckets for the dynamic batch dim of `x`. Each
# bucket compiles one concrete JobPlan; the runtime selects by input shape. Opt
# in with --buckets; empty keeps the byte-identical static Phase 3 path.
BATCH_BUCKETS = [1, 4, 16]

# Compile-spec key mirrored in spyre_backend._BUCKET_COMPILE_SPEC_KEY.
_BUCKET_COMPILE_SPEC_KEY = "spyre_dynamic_buckets"


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _dump_fp16(path: str, t: torch.Tensor) -> None:
    """Write a contiguous fp16 tensor as raw little-endian bytes."""
    arr = t.detach().to(torch.float16).contiguous().cpu().numpy()
    arr.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--in-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--buckets",
        default="",
        help=(
            "Comma-separated batch buckets, e.g. '1,4,16'. When set, dim 0 of "
            "`x` is exported dynamic and preprocess emits one JobPlan per bucket "
            "(v3 payload). Empty => static single-shape .pte (Phase 3)."
        ),
    )
    args = ap.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    x = torch.randn(args.batch, args.in_dim, dtype=torch.float16)

    mod = MLP(args.in_dim, args.hidden, args.out_dim).eval().half()

    # CPU fp16 reference: run the module in fp32 and round to fp16 (CPU has no
    # native fp16 linear kernel), matching the Phase 2 reference convention.
    with torch.no_grad():
        params32 = {k: v.float() for k, v in mod.state_dict().items()}
        h = torch.nn.functional.linear(
            x.float(), params32["fc1.weight"], params32["fc1.bias"]
        )
        h = torch.relu(h)
        y = torch.nn.functional.linear(h, params32["fc2.weight"], params32["fc2.bias"])
        ref = y.to(torch.float16)

    if buckets:
        # Mark dim 0 (batch) of `x` dynamic. Bucket sizes must lie within bound.
        # Trace with a representative batch > 1: PyTorch's 0/1 specialization would
        # otherwise bake batch=1 into the graph as a constant and reject the Dim.
        # The Dim still permits min(buckets) (incl. 1); each bucket later compiles
        # its own concrete JobPlan, so batch=1 is served by its own bucket at runtime.
        batch_dim = torch.export.Dim("batch", min=min(buckets), max=max(buckets))
        trace_batch = max(buckets)
        trace_x = torch.randn(trace_batch, args.in_dim, dtype=torch.float16)
        ep = torch.export.export(mod, (trace_x,), dynamic_shapes={"x": {0: batch_dim}})
        spec_value = json.dumps(
            {"arg_dims": [{"arg": 0, "axis": 0}], "buckets": buckets}
        ).encode("utf-8")
        compile_specs = [CompileSpec(_BUCKET_COMPILE_SPEC_KEY, spec_value)]
    else:
        ep = torch.export.export(mod, (x,))
        compile_specs = None

    edge = to_edge(ep)
    edge = edge.to_backend(SpyrePartitioner(compile_specs))
    et = edge.to_executorch()

    pte_path = os.path.join(args.out_dir, "spyre_mlp.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    _dump_fp16(os.path.join(args.out_dir, "x.bin"), x)
    _dump_fp16(os.path.join(args.out_dir, "ref.bin"), ref)

    meta = {
        "pte": "spyre_mlp.pte",
        "model": "Linear->ReLU->Linear",
        "dtype": "float16",
        "inputs": [{"file": "x.bin", "shape": [args.batch, args.in_dim]}],
        "output": {"file": "ref.bin", "shape": [args.batch, args.out_dim]},
    }
    if buckets:
        meta["dynamic"] = {"arg": 0, "axis": 0, "buckets": buckets}
    with open(os.path.join(args.out_dir, "mlp_inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes)")
    print(f"delegated graph:\n{edge.exported_program().graph}")
    print(f"ref[0,:4] = {ref[0, :4].tolist()}")


if __name__ == "__main__":
    main()
