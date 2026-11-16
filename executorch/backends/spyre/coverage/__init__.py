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
"""ATen-op coverage tooling for the Spyre delegate.

This subpackage is the Phase 5 "coverage foundation":

* :mod:`ops_registry` — the **data-driven allowlist**. ``supported_ops.yaml`` is
  the checked-in source of truth for which core-aten ops the partitioner
  delegates; the partitioner delegates only the ``validated: true`` entries.
* ``candidates.yaml`` — the *candidate* op inventory mined from torch-spyre's
  own lowering tables on the Spyre pod (see ``README.md``), the raw material the
  harness works through.
* :mod:`run_coverage` — the harness that takes each candidate op through
  ``export -> to_edge -> SpyrePartitioner -> preprocess`` (host "compiles?"
  gate) and, on a Spyre device, ``execute`` vs a CPU fp16 reference (device
  "runs? / matches?'' gate), emitting a coverage matrix.

The registry loader has **no hard PyYAML dependency**: it uses PyYAML when it is
installed (the pod / ExecuTorch env) and falls back to a small strict parser for
the constrained schema of these two files otherwise (the bare host CI), so
importing the partitioner never drags a new dependency into every ET export.
"""
