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
"""Export a single-op fp16 matmul as a Spyre-delegated ``.pte`` (Phase 2 exit).

Pipeline: ``torch.export`` -> ``to_edge`` -> ``.to_backend(SpyrePartitioner())``
-> ``.to_executorch()``. The whole graph is delegated to ``SpyreBackend``, whose
``preprocess`` bakes the SpyreCode/JobPlan blob into the delegate payload.

Also writes the inputs and a CPU fp16 reference output as raw little-endian
fp16 so the standalone ``spyre_runner`` can feed the same inputs on-device and
assert the result matches within fp16 tolerance.

Run in the AOT overlay venv: torch-spyre + executorch coexist.
Compilation is independent of FLEX_DEVICE (that only gates on-device execution).
"""

import argparse
import json
import os

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_details import CompileSpec

from executorch.backends.spyre import SpyrePartitioner

# Default shape buckets for the dynamic M (row/seq) dim. Each
# bucket compiles one concrete JobPlan; the runtime picks by input shape. Opt in
# with --buckets; empty keeps the byte-identical static Phase 2 path.
SEQ_BUCKETS = [128, 256, 512]

# Compile-spec key mirrored in spyre_backend._BUCKET_COMPILE_SPEC_KEY. Its JSON
# value names the dynamic dim(s) by forward user-input index + axis and the
# concrete bucket sizes to compile.
_BUCKET_COMPILE_SPEC_KEY = "spyre_dynamic_buckets"


class MatMul(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def _dump_fp16(path: str, t: torch.Tensor) -> None:
    """Write a contiguous fp16 tensor as raw little-endian bytes."""
    arr = t.detach().to(torch.float16).contiguous().cpu().numpy()
    arr.tofile(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--buckets",
        default="",
        help=(
            "Comma-separated M-dim buckets, e.g. '128,256,512'. When set, dim 0 "
            "of `a` is exported dynamic and preprocess emits one JobPlan per "
            "bucket (v3 payload). Empty => static single-shape .pte (Phase 2)."
        ),
    )
    args = ap.parse_args()

    buckets = [int(x) for x in args.buckets.split(",") if x.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    a = torch.randn(args.m, args.k, dtype=torch.float16, generator=gen)
    b = torch.randn(args.k, args.n, dtype=torch.float16, generator=gen)

    mod = MatMul().eval()

    # CPU fp16 reference: accumulate in fp32, round to fp16 (CPU has no native
    # fp16 matmul kernel). This is the golden value the on-device run must match.
    ref = (a.float() @ b.float()).to(torch.float16)

    if buckets:
        # Mark dim 0 (M / row / seq) of `a` dynamic; `b` stays static. Bucket
        # sizes must lie within the Dim bound.
        m_dim = torch.export.Dim("m", min=min(buckets), max=max(buckets))
        dynamic_shapes = {"a": {0: m_dim}, "b": None}
        ep = torch.export.export(mod, (a, b), dynamic_shapes=dynamic_shapes)
        spec_value = json.dumps(
            {"arg_dims": [{"arg": 0, "axis": 0}], "buckets": buckets}
        ).encode("utf-8")
        compile_specs = [CompileSpec(_BUCKET_COMPILE_SPEC_KEY, spec_value)]
    else:
        ep = torch.export.export(mod, (a, b))
        compile_specs = None

    edge = to_edge(ep)
    edge = edge.to_backend(SpyrePartitioner(compile_specs))
    et = edge.to_executorch()

    pte_path = os.path.join(args.out_dir, "spyre_matmul.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    _dump_fp16(os.path.join(args.out_dir, "a.bin"), a)
    _dump_fp16(os.path.join(args.out_dir, "b.bin"), b)
    _dump_fp16(os.path.join(args.out_dir, "ref.bin"), ref)

    meta = {
        "pte": "spyre_matmul.pte",
        "op": "aten.matmul.default",
        "dtype": "float16",
        "inputs": [
            {"file": "a.bin", "shape": [args.m, args.k]},
            {"file": "b.bin", "shape": [args.k, args.n]},
        ],
        "output": {"file": "ref.bin", "shape": [args.m, args.n]},
    }
    if buckets:
        meta["dynamic"] = {"arg": 0, "axis": 0, "buckets": buckets}
    with open(os.path.join(args.out_dir, "inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes)")
    print(f"delegated graph:\n{edge.exported_program().graph}")
    print(f"ref[0,:4] = {ref[0, :4].tolist()}")


if __name__ == "__main__":
    main()
