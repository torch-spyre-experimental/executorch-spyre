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
"""Hello-world pre-scheduling pass: count ops and restickifies, change nothing.

Importing this module registers ``example_count_ops``; nothing else imports it.
Select it for a delegate through a CompileSpec::

    import hello_pre_scheduling_pass  # registers the pass
    from executorch.backends.spyre.spyre_pre_scheduling_passes import pass_compile_spec
    partitioner = SpyrePartitioner(compile_specs=[pass_compile_spec(["example_count_ops"])])

See docs/pre-scheduling-passes.md.
"""

import logging

import torch

from executorch.backends.spyre.spyre_pre_scheduling_passes import (
    pass_compile_spec,
    register_pre_scheduling_pass,
)

logger = logging.getLogger(__name__)


# `assign_dim_hints` is in the pass list of torch-spyre main and v0.5.0-rc.3.
@register_pre_scheduling_pass(after="assign_dim_hints")
def example_count_ops(graph):
    """Log the number of ops and of restickify ops. Read-only."""
    try:  # main: graph.operations may be an OpsView; rc.3 has no all_ops
        from torch_spyre._inductor.passes import all_ops
    except ImportError:

        def all_ops(ops):
            return ops

    ops = list(all_ops(graph.operations))
    restickify = torch.ops.spyre.restickify.default
    n_restickify = sum(
        any(o.target is restickify for o in getattr(op, "origins", ())) for op in ops
    )
    logger.info("example_count_ops: %d ops, %d restickify", len(ops), n_restickify)


if __name__ == "__main__":
    print(pass_compile_spec(["example_count_ops"]))
