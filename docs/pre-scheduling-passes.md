# Adding passes to torch-spyre's pre-scheduling pipeline

`executorch.backends.spyre.spyre_pre_scheduling_passes` lets the ExecuTorch
Spyre backend insert its own passes into torch-spyre's
`CustomPreSchedulingPasses` (the LoopLevel IR pipeline that runs on the
`GraphLowering` right before the Inductor `Scheduler` is built). torch-spyre has
no registration hook; this module adds one from the outside by wrapping
`CustomPreSchedulingPasses.__init__`.

## Support status

| torch-spyre | status |
|---|---|
| main (`64a818de`) | **integration-tested** (frontend-only, `test_spyre/frontend_only/`) |
| v0.5.0-rc.3 (`46001de`) | **structurally compatible, not integration-tested** (fake-class unit tests; structure verified statically) |

## Usage

```python
from executorch.backends.spyre.spyre_pre_scheduling_passes import (
    pass_compile_spec, register_pre_scheduling_pass,
)
from executorch.backends.spyre.spyre_partitioner import SpyrePartitioner

@register_pre_scheduling_pass(after="assign_dim_hints")
def my_pass(graph):                      # Callable[[GraphLowering], None]
    for op in graph.operations:
        op.get_read_writes()             # read-only

partitioner = SpyrePartitioner(compile_specs=[pass_compile_spec(["my_pass"])])
```

Registering only records the pass. It runs when a `CompileSpec` with key
`spyre_pre_scheduling_passes` (value: JSON list of names, as bytes) reaches
`SpyreBackend.preprocess`, which enters `spyre_pass_scope(names)` around the
torch-spyre compile. Without that spec `preprocess` behaves as before. The
module imports torch-spyre lazily, on the first scope with passes.

API: `register_pre_scheduling_pass(fn=None, *, after=None, before=None,
name=None)`, `spyre_pass_scope(names)`, `list_pre_scheduling_passes()`,
`pass_compile_spec(names)`, `parse_pass_names(compile_specs)`.

Execution order comes from the **anchors**, never from the order of the names
in the list. Passes with the same anchor keep registration order. Duplicate
names in the list raise. An anchor is a pass name from torch-spyre's list or an
earlier-registered, also selected, pass.

## Quickstart

A read-only example that counts ops and restickifies and logs them at INFO is in
`examples/pre_scheduling_pass/hello_pre_scheduling_pass.py`. It registers
`example_count_ops` (after `assign_dim_hints`, present on main and rc.3) only
when that module is imported; nothing in the package imports it. Import it, then
pass `pass_compile_spec(["example_count_ops"])` to the partitioner. It is
exercised by a frontend-only test.

## Pass contract

* `Callable[[GraphLowering], None]` with a unique `__name__`. Names must not
  collide with a torch-spyre pass or (main) `_PASSES_RUN_EXPANDED`.
* Edit `graph.operations` in place; never rebind it.
* After IR edits call `invalidate_op_read_writes` where the memo may be stale.
  Read-only passes use `op.get_read_writes()`, not `op_read_writes` (memoizes
  into `op.__dict__`).
* **main only:** `graph.operations` may be a `loop_node.OpsView` (iterate
  `all_ops` for leaf ops); rebinding it raises in `_flat_operations`.
  **rc.3:** none of `all_ops` / `OpsView` / `_flat_operations` exists; passes
  see a plain list.

### Anchors

`splice_while_loops, deadcode_elimination, propagate_named_dims,
validate_named_dims, assign_dim_hints, insert_bmm_padding, split_multi_ops,
propagate_spyre_tensor_layouts, validate_ops, optimize_restickify_locations,
reorder_nonstick_dims, finalize_layouts, insert_restickify,
validate_no_restickify_on_mutation_targets, enforce_indirect_access_layout,
reorder_nonstick_dims_mutation, insert_post_mutation_restickify,
insert_restickify_padding, dedup_and_promote_constants,
_maybe_coarse_tile_span_overflow, plan_sharded_hbm, span_reduction,
_distribute_work, _maybe_scratchpad_planning, elide_proven_read_copies` (main).
rc.3 lacks `plan_sharded_hbm` and `reorder_nonstick_dims_mutation` and adds
`_maybe_reorder_unhinted_interlopers` and `_maybe_coarse_tile_hints`.

## Mechanism

`CustomPreSchedulingPasses()` is built once per `enable_spyre_context` entry
(main `patches.py:216`; rc.3 `patches.py:177`), from a class imported inside the
function, so replacing the class attribute is effective. `self.passes` is a list
built in `__init__` (main `passes.py:556`, rc.3 `passes.py:515`), pass names come
from `_get_pass_name` (main `passes.py:122-133`). `uuid()` (main
`passes.py:753`, hashing pass source files via `_uuid`, `:175-182`) exists on the
class, but nothing on this path feeds it into an Inductor cache key: the instance
is a closure variable in `enable_spyre_context` (main `patches.py:216`) and only
its `__call__` is used. Changing the pass list therefore does not invalidate
Inductor's caches; the framework therefore disables them inside a scope as a
defensive measure (see below).

The wrapper is installed on first scope entry, not at import. It calls the
original `__init__`, then inserts passes only if a scope is active; out of
scope the pass list, `uuid()` and output are those of unpatched torch-spyre.
Before inserting it checks that `self.passes` is a list of callables and that
each anchor appears exactly once, and fails with the torch-spyre path/version/HEAD
in the message. The wrapper always looks up the `__init__` it wraps at install
time and is re-installed on the next scope entry if another patch
(`mock.patch.object`) replaced it; a scope fails loudly (`require_applied()`)
if no pipeline was built under it.

## Scope, threads and caches

* The scope is a `ContextVar`: compiles on the calling thread see it; a new
  thread does not inherit it (use `contextvars.copy_context().run`).
* A scope with passes raises under `TORCHINDUCTOR_FX_COMPILE_MODE`
  subprocess/serialize, or async/progressive FX compile.
* Inside a scope with passes the Inductor FX graph cache and the AOTAutograd
  cache are switched off (restored on exit, INFO-logged once). This is a
  defensive measure: a cache hit would skip the pipeline, but **no cache hit was
  reproduced on this ExecuTorch path**. The switch itself is covered by a host
  test (`test_caches_disabled_inside_scope_and_restored`); the frontend-only
  test only shows that a re-compile inside a scope still runs the pass. The
  switch is process-global torch config, so compiles on other threads lose those
  caches while a scope is open.
* torch-spyre `sdsc_cache` and `spyre_kernel_cache` stay on: both are keyed on
  post-pipeline artifacts.

## Failure modes

| situation | error |
|---|---|
| unknown / duplicate name | `PreSchedulingPassError` listing registered names |
| anchor missing/ambiguous/unselected | `anchor 'x' for pass 'y' appears N times in ...passes (need exactly 1)` + available names |
| name collision | `pass name 'x' collides with a torch-spyre pass name or a reserved name` |
| `passes` not a list / entries not callable | `...passes is X, expected a list ...` |
| out-of-process FX compile | `pre-scheduling passes need the in-process FX compile` |
| wrapper displaced / pipeline never built | `... did not run` / `replaced by another patch ...` |
