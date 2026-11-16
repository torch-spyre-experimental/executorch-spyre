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
"""Export a single-op fp16 matmul as a **spyre2**-delegated ``.pte``.

Same pipeline as spyre's ``export_matmul`` (``torch.export`` -> ``to_edge`` ->
``.to_backend(...)`` -> ``.to_executorch()``) but routed to
:class:`~executorch.backends.spyre2.spyre2_partitioner.Spyre2Partitioner`, so ``to_backend`` invokes
:meth:`~executorch.backends.spyre2.spyre2_backend.Spyre2Backend.preprocess` (id ``"Spyre2Backend"``)
rather than spyre's ``SpyreBackend``.

``--artifact`` selects the staged payload via a
``CompileSpec("spyre2_artifact", ...)`` forwarded to the delegate:

- ``sdsc`` (default): the on-device Stage 2 path — the ``mm`` partition is
  compiled by torch-spyre's device-verified ``compile_fx``/``SuperDSCScheduling``
  (reused from spyre) and annotated with the captured-TTIR provenance. Run in the
  AOT overlay venv (torch-spyre + executorch); device execution then needs the
  ``"Spyre2Backend"`` runtime alias (see runtime notes) and ``FLEX_DEVICE=PF``.
- ``ktir``: the host-only Stage 1 path — TTIR is lowered to ``ktdp`` KTIR and
  structurally verified against the golden. The resulting ``.pte`` is NOT
  device-runnable on the shipped SDK (no ktdp/linalg/tt backend); it exists to
  exercise and freeze the TTIR->KTIR seam.

Like spyre, also writes ``a.bin`` / ``b.bin`` / ``ref.bin`` (raw little-endian
fp16) so the reused ``spyre_runner`` can feed the same inputs on-device and assert
the result matches the CPU fp16 reference (``sdsc`` only).
"""

import argparse
import json
import os

import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_details import CompileSpec

from executorch.backends.spyre2 import Spyre2Partitioner

# Mirror of executorch.backends.spyre2.spyre2_backend._ARTIFACT_SPEC_KEY.
_ARTIFACT_SPEC_KEY = "spyre2_artifact"


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
        "--artifact",
        choices=("sdsc", "ktir"),
        default="sdsc",
        help=(
            "Staged payload: 'sdsc' (device, Stage 2) or 'ktir' (host-only, "
            "Stage 1). Default sdsc."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)

    a = torch.randn(args.m, args.k, dtype=torch.float16, generator=gen)
    b = torch.randn(args.k, args.n, dtype=torch.float16, generator=gen)

    mod = MatMul().eval()

    # CPU fp16 reference: accumulate in fp32, round to fp16 (CPU has no native
    # fp16 matmul kernel). This is the golden value the on-device run must match.
    ref = (a.float() @ b.float()).to(torch.float16)

    ep = torch.export.export(mod, (a, b))
    compile_specs = [
        CompileSpec(_ARTIFACT_SPEC_KEY, args.artifact.encode("utf-8"))
    ]

    edge = to_edge(ep)
    edge = edge.to_backend(Spyre2Partitioner(compile_specs))
    et = edge.to_executorch()

    pte_path = os.path.join(args.out_dir, "spyre2_matmul.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    _dump_fp16(os.path.join(args.out_dir, "a.bin"), a)
    _dump_fp16(os.path.join(args.out_dir, "b.bin"), b)
    _dump_fp16(os.path.join(args.out_dir, "ref.bin"), ref)

    meta = {
        "pte": "spyre2_matmul.pte",
        "op": "aten.matmul.default",
        "dtype": "float16",
        "artifact": args.artifact,
        "device_runnable": args.artifact == "sdsc",
        "inputs": [
            {"file": "a.bin", "shape": [args.m, args.k]},
            {"file": "b.bin", "shape": [args.k, args.n]},
        ],
        "output": {"file": "ref.bin", "shape": [args.m, args.n]},
    }
    with open(os.path.join(args.out_dir, "inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes, artifact={args.artifact})")
    print(f"delegated graph:\n{edge.exported_program().graph}")
    if args.artifact == "ktir":
        print("NOTE: artifact=ktir is host-only; the .pte is not device-runnable.")
    print(f"ref[0,:4] = {ref[0, :4].tolist()}")


if __name__ == "__main__":
    main()
