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
"""spyre2 partitioner — reuse spyre's op support + gates, re-point the backend id.

Op support and the Spyre hardware-layout refusal gates (softmax-fed matmul,
wide-output scramble) are properties of the *accelerator*, not of the codegen
path, so the partitioning logic carries over unchanged. ``executorch.backends.spyre``
(spyre) resolves alongside ``executorch.backends.spyre2`` (this package).

The **only** change from spyre is the delegated ``backend_id``. ExecuTorch's
``to_backend`` resolves a partition to a backend by matching its
``DelegationSpec.backend_id`` against ``cls.__name__`` over
``BackendDetails.__subclasses__()``. spyre's :class:`SpyrePartitioner` hardcodes
``SPYRE_BACKEND_ID == "SpyreBackend"``, which resolves to spyre's ``SpyreBackend`` — so
to route the same partitions to :class:`~executorch.backends.spyre2.spyre2_backend.Spyre2Backend`
we subclass it and swap ``delegation_spec`` for one naming
:data:`~executorch.backends.spyre2.spyre2_backend.SPYRE2_BACKEND_ID` (``"Spyre2Backend"``). Nothing
else (op allowlist, ``_writes_scrambled_output`` / softmax gates, tagging loop)
is duplicated.
"""

from typing import List

from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.backend.partitioner import DelegationSpec

# Reused verbatim from spyre (op support + hardware-layout gates).
from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner
from executorch.backends.spyre2.spyre2_backend import SPYRE2_BACKEND_ID

__all__ = ["Spyre2Partitioner", "SpyrePartitioner", "SPYRE2_BACKEND_ID"]


class Spyre2Partitioner(SpyrePartitioner):
    """spyre's :class:`SpyrePartitioner` with the delegated id set to spyre2's.

    Inherits ``partition()`` and the entire ``_SpyreOperatorSupport`` allowlist /
    refusal logic; overrides only the :class:`DelegationSpec` so every tagged
    partition routes to ``Spyre2Backend.preprocess`` (id ``"Spyre2Backend"``) instead
    of spyre's ``SpyreBackend``. ``compile_specs`` (e.g. the
    ``spyre2_artifact=b"ktir"|b"sdsc"`` selector) are forwarded to the delegate.
    """

    def __init__(self, compile_specs: List[CompileSpec] | None = None) -> None:
        super().__init__(compile_specs)
        # Re-point: spyre2's backend id, same forwarded compile_specs.
        self.delegation_spec = DelegationSpec(SPYRE2_BACKEND_ID, self.compile_specs)
