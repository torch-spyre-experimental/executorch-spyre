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
"""ExecuTorch backend delegate for the IBM Spyre accelerator (AOT side).

Exposes:
- ``SpyreBackend``       : ``BackendDetails`` whose ``preprocess`` drives the
                         torch-spyre DeepTools compiler and emits a SpyreCode
                         (JobPlan) blob as the delegate payload.
- ``SpyrePartitioner`` : an op-support (capability-based) partitioner that tags
                         the largest connected subgraphs of SDSC-supported ops
                         for the ``SpyreBackend`` delegate; unsupported ops fall
                         back to CPU (portable kernels).
- ``SPYRE_BACKEND_ID``   : the backend id string, shared with the C++ runtime
                         registration (must match exactly).
"""

# NOTE: imports are lazy (PEP 562). ``spyre_backend`` / ``spyre_partitioner`` pull in
# torch + the ExecuTorch stack, but the ``executorch.backends.spyre.coverage`` subpackage (the
# op-coverage harness, the data-driven allowlist, and their dependency-free
# tests) must import on a bare host with neither torch nor PyYAML installed --
# that is the whole point of the zero-dependency YAML loader. Importing this
# parent package eagerly would defeat that, so ``SpyreBackend`` / ``SpyrePartitioner``
# are resolved only on first attribute access.

__all__ = ["SpyreBackend", "SpyrePartitioner", "SPYRE_BACKEND_ID"]


def __getattr__(name):  # PEP 562 module-level lazy attribute access
    if name in ("SpyreBackend", "SPYRE_BACKEND_ID"):
        from .spyre_backend import SPYRE_BACKEND_ID, SpyreBackend

        return {"SpyreBackend": SpyreBackend, "SPYRE_BACKEND_ID": SPYRE_BACKEND_ID}[
            name
        ]
    if name == "SpyrePartitioner":
        from .spyre_partitioner import SpyrePartitioner

        return SpyrePartitioner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
