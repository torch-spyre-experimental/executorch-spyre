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
"""ExecuTorch backend delegate for the IBM Spyre — Triton→TTIR path (spyre2).

Sibling of ``backends.spyre`` (spyre, the device-verified SuperDSC path). The whole
ExecuTorch-facing surface (partitioner, ``.pte`` wire format, C++ runtime,
tensor bridge, device runner, coverage harness) is codegen-independent and is
**reused** from spyre — imported or symlinked, never copied. Only what
``preprocess`` puts in the payload tar differs per stage:

- **Stage 1 (KTIR, host-only):** a faithful TTIR→KTIR (``ktdp``) lowering for
  ``mm`` (``ttir_to_ktir``), structurally verified against the golden KTIR
  (``ktir_verify``). Not device-runnable on the shipped SDK (no ``ktdp``/
  ``linalg``/``tt`` in ``hcc-opt``); framed into spyre's payload container with
  ``header["artifact"] = "ktir"`` so the seam is ready when upstream ships the
  ktdp backend.
- **Stage 2 (SDSC, on-device):** the same ``mm`` partition runs on the Spyre by
  bridging to torch-spyre's device-verified ``compile_fx``/``SuperDSCScheduling``
  path (reused from spyre), carrying the captured TTIR as front-half provenance
  (``header["artifact"] = "sdsc"``).

Exposes:
- ``Spyre2Backend``      : ``BackendDetails`` whose ``preprocess`` emits either a
                         KTIR (Stage 1) or SDSC (Stage 2) payload.
- ``Spyre2Partitioner``  : spyre's ``SpyrePartitioner`` op support + gates, re-pointed
                         to delegate to ``Spyre2Backend`` (id ``"Spyre2Backend"``).
- ``SpyrePartitioner`` : re-exported verbatim from spyre (op support + hardware
                         layout gates are properties of the Spyre, not the codegen).
- ``SPYRE2_BACKEND_ID``  : spyre2's routing/registration id (``"Spyre2Backend"``).
- ``SPYRE_BACKEND_ID``   : spyre's id, re-exported for the shared runtime name.
"""

__all__ = [
    "Spyre2Backend",
    "Spyre2Partitioner",
    "SpyrePartitioner",
    "SPYRE2_BACKEND_ID",
    "SPYRE_BACKEND_ID",
]


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    if name in ("Spyre2Backend", "SPYRE2_BACKEND_ID", "SPYRE_BACKEND_ID"):
        from .spyre2_backend import SPYRE2_BACKEND_ID, SPYRE_BACKEND_ID, Spyre2Backend

        return {
            "Spyre2Backend": Spyre2Backend,
            "SPYRE2_BACKEND_ID": SPYRE2_BACKEND_ID,
            "SPYRE_BACKEND_ID": SPYRE_BACKEND_ID,
        }[name]
    if name in ("Spyre2Partitioner", "SpyrePartitioner"):
        from .spyre2_partitioner import Spyre2Partitioner, SpyrePartitioner

        return {
            "Spyre2Partitioner": Spyre2Partitioner,
            "SpyrePartitioner": SpyrePartitioner,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
