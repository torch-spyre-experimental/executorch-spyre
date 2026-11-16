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
"""Export the *simplest* Spyre/CPU **mixed** workload as one ``.pte``.

Every other end-to-end device artifact is fully delegated
(``export_matmul``/``export_mlp``/``export_attention``). This script is the
minimal honest proof of the generic **CPU-fallback** path: a graph where some ops delegate to Spyre and exactly one
op falls back to ExecuTorch's portable CPU kernels, run end-to-end on device and
matched against a CPU fp16 reference.

Module: ``relu(sin(fc(x)))`` (:class:`MixedUnsupported`, shared with
``test_partitioner``). Partitioning splits it into::

    fc (addmm + permute_copy)  -> delegate partition 1  (Spyre)
    sin                        -> CPU  (not allowlisted; portable kernel)
    relu                       -> delegate partition 2  (Spyre)

i.e. two delegate calls with an interior device->host->device hand-off around
one CPU node. ``sin`` is a pure **Rule 1** fallback (a generic allowlist miss),
distinct from ``run_8c_fallback.sh`` which exercises only **Rule 2** (a
softmax-fed ``mm``).

Pipeline: ``torch.export`` -> ``to_edge`` -> ``.to_backend(SpyrePartitioner())``
-> ``.to_executorch()``. Emits ``x.bin`` (user input) + ``ref.bin`` (CPU fp16
reference) + ``manifest.txt`` for the generic ``spyre_coverage_runner`` (see
``backends/spyre/spyre_runner/spyre_coverage_runner.cpp``).

The runner MUST be linked with ``portable_ops_lib`` (default
``SPYRE_RUNNER_WITH_PORTABLE_OPS=ON``) so the CPU ``sin`` node executes; without it
the fallback node fails at run time with ``0x14`` (OperatorMissing). SDSC codegen
inside ``preprocess`` runs during ``to_backend`` and needs the Spyre toolchain;
on-device execution needs ``FLEX_DEVICE=PF``.
"""

import argparse
import json
import os

import torch
from executorch.exir import to_edge

from executorch.backends.spyre import SpyrePartitioner
from executorch.backends.spyre.export_attention import _dump_fp16, _partition_report


class MixedUnsupported(torch.nn.Module):
    """``relu(sin(fc(x)))`` -- ``torch.sin`` is not allowlisted, so it must fall
    back to CPU while the linear (``addmm``/``permute_copy``) and ``relu``
    delegate to Spyre. The single source of truth for this module: the
    partitioner unit test imports it from here.
    """

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(torch.sin(self.fc(x)))


# The only op expected to leave the delegate: a generic Rule-1 allowlist miss.
_EXPECTED_FALLBACK: frozenset[str] = frozenset({"aten::sin"})


def _cpu_ref(mod: MixedUnsupported, x: torch.Tensor) -> torch.Tensor:
    """CPU fp16 reference: compute in fp32 (CPU has no native fp16 linear) and
    round to fp16, matching the project reference convention (export_mlp.py).
    """
    with torch.no_grad():
        params32 = {k: v.float() for k, v in mod.state_dict().items()}
        h = torch.nn.functional.linear(
            x.float(), params32["fc.weight"], params32["fc.bias"]
        )
        y = torch.relu(torch.sin(h))
        return y.to(torch.float16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts-mixed")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="device-free delegation check: assert some ops delegate AND only "
        "aten::sin falls back, then exit without building a .pte.",
    )
    ap.add_argument(
        "--print-program",
        action="store_true",
        help="print the ExecuTorch program graph (after to_backend + to_executorch) "
        "and exit without writing any files. Needs torch_spyre for the SDSC compile.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    x = torch.randn(args.batch, args.dim, dtype=torch.float16)
    mod = MixedUnsupported(args.dim).eval().half()

    ep = torch.export.export(mod, (x,))
    edge = to_edge(ep)

    if args.inspect:
        delegated, fallback = _partition_report(edge.exported_program())
        print(f"delegated graph:\n{edge.exported_program().graph}")
        print(f"\ndelegated ops ({len(delegated)}): {sorted(set(delegated))}")
        print(f"CPU-fallback ops ({len(fallback)}): {sorted(set(fallback))}")
        # Gate: this is a MIXED workload, so both halves must be present --
        # at least one delegated op AND a fallback set that is exactly aten::sin.
        fb = set(fallback)
        if not delegated:
            print("RESULT: UNEXPECTED -- nothing delegated (not a mixed workload)")
        elif fb != set(_EXPECTED_FALLBACK):
            print(
                f"RESULT: UNEXPECTED FALLBACK {sorted(fb)} -- expected exactly "
                f"{sorted(_EXPECTED_FALLBACK)}"
            )
        else:
            print(
                f"RESULT: OK -- {len(delegated)} op(s) delegated to Spyre, "
                f"CPU fallback is exactly {sorted(fb)}"
            )
        return

    edge = edge.to_backend(SpyrePartitioner())
    et = edge.to_executorch()

    if args.print_program:
        print(et.exported_program().graph_module.graph)
        return

    pte_path = os.path.join(args.out_dir, "spyre_mixed.pte")
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    ref = _cpu_ref(mod, x)
    _dump_fp16(os.path.join(args.out_dir, "x.bin"), x)
    _dump_fp16(os.path.join(args.out_dir, "ref.bin"), ref)

    # Generic spyre_coverage_runner manifest (see spyre_coverage_runner.cpp:24-60).
    manifest = [
        "pte spyre_mixed.pte",
        f"input x.bin fp16 {x.dim()} " + " ".join(str(d) for d in x.shape),
        f"output ref.bin fp16 {ref.dim()} " + " ".join(str(d) for d in ref.shape),
    ]
    with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    # Companion JSON (human/tooling); the device runner consumes only manifest.txt.
    meta = {
        "pte": "spyre_mixed.pte",
        "model": "relu(sin(Linear))",
        "dtype": "float16",
        "inputs": [{"file": "x.bin", "shape": list(x.shape)}],
        "output": {"file": "ref.bin", "shape": list(ref.shape)},
        "expected_cpu_fallback": sorted(_EXPECTED_FALLBACK),
    }
    with open(os.path.join(args.out_dir, "mixed_inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes)")
    print(f"ref[0,:4] = {ref[0, :4].tolist()}")


if __name__ == "__main__":
    main()
