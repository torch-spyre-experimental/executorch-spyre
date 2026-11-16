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
"""AOT ``preprocess`` for the Spyre ExecuTorch delegate.

Drives torch-spyre's DeepTools compiler from an ExecuTorch *edge* program and
serializes the resulting SpyreCode (JobPlan) directory as the delegate payload.

Reuse seam: everything from
``torch._inductor.compile_fx.compile_fx`` downward
(Inductor lowering -> ``SuperDSCScheduling`` -> DeepTools -> SpyreCode dir ->
``prepare_kernel``) is graph-source-agnostic. So ``preprocess``:

  1. retargets edge-dialect op overloads back to core-aten (Inductor lowering
     knows ``aten.mm.default``, not ``executorch_exir_dialects_edge__ops_*``);
  2. synthesizes fp16 example inputs on the ``spyre`` device for *every*
     subgraph placeholder -- user inputs **and** lifted params/buffers alike
     (see ``_make_spyre_example_inputs``);
  3. runs ``compile_fx`` once into an **explicit output dir we own** (by
     redirecting torch-spyre's ``get_output_dir`` at a ``mkdtemp`` we control),
     so the emitted SpyreCode dir is captured directly -- no inductor-cache
     scraping;
  4. tars each emitted ``code_dir/spyreCodeDir`` as ``spyreCodeDir_{i}`` (the
     ``spyrecode.json`` + ``init_binary.bin`` that ``prepare_kernel`` consumes at
     runtime);
  5. prepends a small JSON header (I/O arity + shapes/dtypes) so the C++
     runtime can bind ET tensors <-> Spyre tensors without re-deriving them.

Payload wire format (``processed_bytes``)::

    [u32 header_len, little-endian] [header_json utf-8] [tar bytes]

Each ``header["inputs"][i]`` entry may carry an OPTIONAL ``"device_layout"``
object (same schema as ``header["outputs"][j]["device_layout"]``; see
``_device_layout_spec``), currently only for ``kv_state`` inputs. Readers
must treat its absence as "no layout recorded" (default layout); payloads that
omit it remain valid, so the field is purely additive.

Scope: multi-op subgraphs from the capability partitioner, lifted
params/buffers passed as runtime delegate inputs (in placeholder order). The
static path always emits a v2 payload (a ``kernels`` array, one entry per emitted
SDSC kernel, even for a single kernel); a bucket spec emits v3. The v1
single-kernel payload is no longer emitted, though the C++ runtime still parses
it. ``preprocess`` fails loudly if the number of observed kernel launches differs
from the number of emitted SpyreCode dirs.
"""

import copy
import json
import logging
import os
import re
import shutil
import struct
import tarfile
import tempfile
import threading
import time
from typing import Final, List

import torch
from executorch.backends.spyre.spyre_layout import (
    SPYRE_DEVICE,
    device_layout_spec,
    layout_key,
    spyre_tensor_with_layout,
)
from executorch.backends.spyre.spyre_pre_scheduling_passes import (
    parse_pass_names,
    spyre_pass_scope,
)
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind

# Backend id string. MUST match the name passed to ``register_backend`` in the
# C++ runtime (runtime/SpyreBackend.cpp) and the id used by SpyrePartitioner.
SPYRE_BACKEND_ID: Final[str] = "SpyreBackend"

# Subdirectory (inside the emitted SpyreCode dir) that prepare_kernel consumes.
_SPYRECODE_SUBDIR: Final[str] = "spyreCodeDir"

# Device string registered by torch-spyre (PrivateUse1).
_SPYRE_DEVICE: Final[str] = SPYRE_DEVICE

logger = logging.getLogger(__name__)

# Per-process delegate compile counter (thread-safe).  Incremented once per
# ``SpyreBackend.preprocess`` call so that "DELEGATE_COMPILE #N" lines in the
# export log can be matched back to the partition-report rows (tag0, tag1, …).
_delegate_compile_counter = 0
_delegate_compile_lock = threading.Lock()


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


# SpyreTensorLayout handling lives in ``spyre_layout`` (the single ET-side layout
# boundary). The private names below are kept so existing call sites and imports
# are unchanged. NOTE: ``_layout_key`` is the layout-convergence identity and is NOT
# ``SpyreTensorLayout::operator==`` -- see the ``spyre_layout`` module docstring.
_device_layout_spec = device_layout_spec
_layout_key = layout_key
_spyre_tensor_with_layout = spyre_tensor_with_layout


def _attach_output_layouts(output_specs: list, outs) -> None:
    """Record each compiled delegate output's device layout in its output spec."""
    for j, o in enumerate(outs):
        if j < len(output_specs):
            spec = _device_layout_spec(o)
            if spec is not None:
                output_specs[j]["device_layout"] = spec


def _remove_dirs(dirs) -> None:
    """Best-effort removal of the ``spyre_sdsc_*`` dirs a compile pass emitted."""
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


# Upper bound on compile passes spent searching for the kv_state layout
# fixed point. A well-behaved kernel needs 2 (default layout -> kernel layout ->
# confirmed); 1 if the default already matches. Hitting the bound is an error.
_MAX_LAYOUT_ITERS: Final[int] = 4


# Per-process cache of kv_state layout fixed points, so that only the first
# delegate (or bucket) that needs a non-default layout pays the extra compile
# pass. Key: (shape, dtype, layout_key(default upload layout), normalised
# placeholder name). Value: the converged ``device_layout`` spec dict. Only inputs
# whose fixed point differs from the default layout are stored, and only those
# are ever seeded. A wrong seed costs one pass of the ``_MAX_LAYOUT_ITERS`` budget
# and is dropped; it can never produce a mismatched payload because the loop
# still verifies every input against the compiled state output.
_layout_seed_cache: "dict[tuple, dict]" = {}
_layout_seed_lock = threading.Lock()

_LAYER_INDEX_RE = re.compile(r"\d+")


def _reset_layout_seed_cache() -> None:
    """Forget every cached layout (tests; also useful in long-lived processes)."""
    with _layout_seed_lock:
        _layout_seed_cache.clear()


def _normalize_state_name(name: str) -> str:
    """``b__inner_kc0`` -> ``b__inner_kc#``: layers of one model share a key."""
    return _LAYER_INDEX_RE.sub("#", name)


def _placeholder_names(gm, n_inputs: int) -> "list[str] | None":
    """Placeholder names in graph order, or None when they are unavailable (then the
    seed cache is not used and the loop behaves as without it).

    No graph (the host fakes in test_preprocess) is logged at debug level. A
    placeholder count that differs from the ``n_inputs`` input specs is logged as
    a warning: on a real export it means the layout seed cache is off and the
    per-delegate compile saving will be missing.
    """
    nodes = getattr(getattr(gm, "graph", None), "nodes", None)
    if nodes is None:
        logger.debug("layout seed cache skipped: no graph for placeholder names")
        return None
    names = [n.name for n in nodes if n.op == "placeholder"]
    if len(names) != n_inputs:
        logger.warning(
            "layout seed cache skipped: %d graph placeholders but %d input specs; "
            "every delegate pays the full fixed-point passes",
            len(names),
            n_inputs,
        )
        return None
    return names


def _iterate_input_layouts(
    gm: torch.fx.GraphModule,
    example_inputs: list,
    input_specs: list,
    cfx,
    ac,
    kr,
    *,
    max_iters: int = _MAX_LAYOUT_ITERS,
) -> "tuple[list, list, dict, dict, list]":
    """Compile until each kv_state input layout equals its state-output layout.

    A kv_state input ``i`` (``kind == "kv_state"``, ``state_out = j >= 0``)
    is fed on step N+1 with the kernel's own output ``j`` from step N. The kernel
    writes that output in a layout of its choosing (e.g. V's device dims 1 and 3
    swapped), whereas the step-1 example/initial input is a default-layout
    upload. For the loop to be
    sound we need the fixed point::

        layout(input i) == layout(output state_out[i])

    Algorithm (each pass is one full ``_capture_one_compile``):

      1. Compile with the current example inputs (initially default layout).
      2. For every kv_state input compare ``layout(example_inputs[i])`` with
         the compiled ``output_specs[state_out]["device_layout"]``.
      3. All equal -> converged: record that layout as
         ``input_specs[i]["device_layout"]`` and return this compile's artifacts.
      4. Otherwise rebuild the mismatching example inputs with the output layout
         (``spyre_tensor_with_layout``), discard this pass's SpyreCode dirs, and
         recompile.

    Seeding: before pass 1, a kv_state input whose fixed point an earlier call
    already found (``_layout_seed_cache``, per process, shared by the static and
    bucketed paths) starts at that layout instead of the default upload. Only
    inputs that mismatched at pass 1 are ever stored or seeded; an input that
    converged at the default layout is not. The cache key is
    ``(shape, dtype, layout_key(default upload), name)`` with the digits in the
    placeholder name normalised (``b__inner_vc0`` -> ``b__inner_vc#``). A seed is
    only a starting point: pass 1 still compares it with the compiled output
    layout. A wrong seed is dropped from the cache (that entry only), the input
    is rebuilt from the output layout as in step 4, and the pass counts against
    ``max_iters`` (no refund). A seed whose state output reports no layout cannot
    be checked and is reverted to the default upload. ``valid_elements`` takes no
    part in any of these comparisons.

    Graphs with no kv_state input (or whose state outputs carry no device
    layout) compile exactly once and ``input_specs`` is left untouched -> payload
    byte-identical to before. Shape/dtype disagreement between a state input and
    its output, and failure to converge within ``max_iters`` passes, raise
    ``RuntimeError`` (never silently ship a mismatched layout).

    Returns the same 5-tuple as ``_capture_one_compile`` for the converged pass.
    """
    iter_inputs = list(example_inputs)
    history: "list[dict[int, tuple]]" = []

    # Seed from the per-process cache: a kv_state input whose fixed point an
    # earlier delegate/bucket already found starts at that layout instead of the
    # default upload. seed_keys/default_keys exist only when the cache is usable.
    names = _placeholder_names(gm, len(input_specs))
    seed_keys: dict[int, tuple] = {}  # input idx -> cache key
    default_keys: dict[int, tuple] = {}  # input idx -> layout_key(default upload)
    seeded: dict[int, tuple] = {}  # input idx -> layout_key(seed)
    for i, spec in enumerate(input_specs):
        if names is None:
            break
        if spec.get("kind") != "kv_state" or spec.get("state_out", -1) < 0:
            continue
        default_layout = _device_layout_spec(iter_inputs[i])
        if default_layout is None:
            continue
        default_keys[i] = _layout_key(default_layout)
        seed_keys[i] = (
            tuple(spec["shape"]),
            spec["dtype"],
            default_keys[i],
            _normalize_state_name(names[i]),
        )
        with _layout_seed_lock:
            seed = _layout_seed_cache.get(seed_keys[i])
        if seed is not None:
            iter_inputs[i] = _spyre_tensor_with_layout(
                seed, spec["shape"], iter_inputs[i].dtype
            )
            seeded[i] = _layout_key(seed)
    seeded_names = [names[i] for i in seeded]
    rejected: list[str] = []
    pass1_mismatched: set[int] = set()

    for iteration in range(1, max_iters + 1):
        t_pass = time.monotonic()
        (
            captured_dirs,
            captured_runs,
            input_ptr_to_idx,
            output_ptr_to_idx,
            output_specs,
        ) = _capture_one_compile(gm, iter_inputs, cfx, ac, kr)
        compile_s = time.monotonic() - t_pass

        try:
            # kv_state input idx -> compiled state-output layout (dict).
            targets: dict[int, dict] = {}
            for i, spec in enumerate(input_specs):
                if spec.get("kind") != "kv_state":
                    continue
                j = spec.get("state_out", -1)
                if j < 0 or j >= len(output_specs):
                    continue
                out_layout = output_specs[j].get("device_layout")
                if out_layout is None:
                    continue
                if spec["shape"] != output_specs[j]["shape"] or (
                    spec["dtype"] != output_specs[j]["dtype"]
                ):
                    raise RuntimeError(
                        f"preprocess: {spec['kind']} input {i} "
                        f"({spec['shape']}, {spec['dtype']}) cannot alias state output "
                        f"{j} ({output_specs[j]['shape']}, {output_specs[j]['dtype']}); "
                        "no layout fixed point exists."
                    )
                targets[i] = out_layout

            mismatched = {
                i: out_layout
                for i, out_layout in targets.items()
                if _layout_key(_device_layout_spec(iter_inputs[i]))
                != _layout_key(out_layout)
            }
            # A seed with no state-output layout to check it against cannot be
            # verified: it must go back to the default upload, which is what the
            # runtime will feed when no device_layout is recorded.
            unverified = [i for i in seeded if i not in targets]
            history.append({i: _layout_key(l) for i, l in targets.items()})

            if iteration == 1:
                pass1_mismatched = set(mismatched)
                # Judge the seeds. A rejected seed is dropped from the cache (only
                # that entry); the input is rebuilt below like any mismatch and the
                # pass counts against max_iters.
                for i in seeded:
                    if i in mismatched or i in unverified:
                        rejected.append(names[i])
                        with _layout_seed_lock:
                            _layout_seed_cache.pop(seed_keys[i], None)

            logger.info(
                "preprocess: input-layout pass %d/%d: %d kv_state input(s), "
                "%d mismatched, compile %.1fs, seeded %s, seed rejected %s",
                iteration,
                max_iters,
                len(targets),
                len(mismatched),
                compile_s,
                seeded_names,
                rejected,
            )

            if not mismatched and not unverified:
                for i, out_layout in targets.items():
                    input_specs[i]["device_layout"] = out_layout
                    # Cache only inputs that mismatched at pass 1 and whose fixed
                    # point is not the default upload; an input that converges at
                    # pass 1 (today: K) is never cached, so never seeded.
                    if (
                        i in seed_keys
                        and i in pass1_mismatched
                        and _layout_key(out_layout) != default_keys[i]
                    ):
                        with _layout_seed_lock:
                            _layout_seed_cache[seed_keys[i]] = copy.deepcopy(out_layout)
                return (
                    captured_dirs,
                    captured_runs,
                    input_ptr_to_idx,
                    output_ptr_to_idx,
                    output_specs,
                )

            # Not converged: this pass's artifacts were compiled against stale input
            # layouts. Drop them before recompiling (or raising).
            _remove_dirs(captured_dirs)

            if iteration == max_iters:
                raise RuntimeError(
                    f"preprocess: kv_state input layouts did not converge to "
                    f"the state-output layouts within {max_iters} compile passes "
                    f"(still mismatched: inputs {sorted(mismatched)}). The kernel "
                    "keeps choosing a different output layout than the input it was "
                    "compiled with, so state fed back on step N+1 would be misread. "
                    f"Seeded from the layout cache: {seeded_names} "
                    f"(rejected at pass 1: {rejected}). "
                    f"Per-pass target layout keys: {history!r}"
                )

            for i, out_layout in mismatched.items():
                iter_inputs[i] = _spyre_tensor_with_layout(
                    out_layout, input_specs[i]["shape"], iter_inputs[i].dtype
                )
            for i in unverified:
                iter_inputs[i] = example_inputs[i]
            for i in mismatched.keys() | set(unverified):
                seeded.pop(i, None)

        except BaseException:
            _remove_dirs(captured_dirs)
            raise

    raise AssertionError("unreachable")  # pragma: no cover (max_iters >= 1)


def _pack_payload(header: dict, tar_bytes: bytes) -> bytes:
    """Serialize the delegate payload wire format.

    Layout (little-endian): ``[u32 header_len][header_json utf-8][tar bytes]``.
    This is the exact framing the C++ ``SpyreBackend::init`` parses, so keeping it
    in one small helper lets the round-trip be unit-tested without a device.
    """
    header_json = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(header_json)) + header_json + tar_bytes


def _retarget_edge_to_aten(graph_module: torch.fx.GraphModule) -> int:
    """Rewrite edge-dialect op overloads to their core-aten counterparts.

    ``EdgeOpOverload`` wraps the real ``OpOverload`` at ``._op``. Inductor's
    lowering table is keyed on core-aten ops, so we swap the target in place.

    Hardened for multi-op subgraphs: only nodes whose ``_op`` resolves to a real
    ``torch._ops.OpOverload`` are retargeted. A ``call_function`` node that is
    neither an ``EdgeOpOverload`` (no ``._op``) nor already a plain
    ``OpOverload`` is *flagged* (logged) and left untouched rather than
    blind-swapped -- if it is something Inductor cannot lower, ``compile_fx``
    fails loudly downstream instead of us silently mangling the graph.

    Returns the number of nodes retargeted (for logging / sanity).
    """
    from torch._ops import OpOverload

    retargeted = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        underlying = getattr(target, "_op", None)
        if isinstance(underlying, OpOverload):
            # Edge-dialect overload wrapping a core-aten op: swap to the aten op.
            node.target = underlying
            retargeted += 1
        elif isinstance(target, OpOverload):
            # Already a plain aten OpOverload (nothing to retarget).
            continue
        else:
            # Not an edge/aten op overload -- e.g. a builtin (operator.getitem),
            # a higher-order op, or an unexpected callable. Leave it for Inductor
            # to accept or reject; just make the surprise visible.
            logger.warning(
                "preprocess: node %r has non-OpOverload target %r; leaving as-is "
                "(compile_fx will reject it if unsupported).",
                node.name,
                target,
            )
    graph_module.recompile()
    return retargeted


def _build_kv_state_out_map(exported_program: ExportedProgram) -> "dict[str, int]":
    """Map KV-state buffer placeholder names to their BUFFER_MUTATION output index.

    A placeholder is KV state when:
      1. It is classified as a ``buffer`` (``inputs_to_buffers`` present, or
         ``b_`` prefix in the extracted subgraph).
      2. Its buffer FQN appears as a value in ``buffers_to_mutate``
         (meaning a BUFFER_MUTATION output writes the updated cache back).

    The output index is derived from the ``output_specs`` ordering — the
    position of the BUFFER_MUTATION output spec whose ``target`` matches the
    FQN. This is reliable because ExecuTorch preserves BUFFER_MUTATION output
    order in the delegate's extracted program.

    Returns ``{}`` when ``buffers_to_mutate`` is absent or empty (non-KV
    subgraphs, legacy payloads).  Callers fall through to the normal
    non-state path (``state_out`` stays -1).

    Note: in the *extracted subgraph* (what ``preprocess`` receives) the
    ``buffers_to_mutate`` field is populated by ExecuTorch's lowering pass
    (``lower_to_backend_module``).  The placeholder names use the ``b_``
    prefix convention (e.g. ``b_k_cache``).
    """
    sig = getattr(exported_program, "graph_signature", None)
    if sig is None:
        return {}
    buffers_to_mutate = getattr(sig, "buffers_to_mutate", None) or {}
    if not buffers_to_mutate:
        return {}

    # Build FQN -> output_index from BUFFER_MUTATION output specs.
    fqn_to_out_idx: dict[str, int] = {}
    output_specs = getattr(sig, "output_specs", [])
    out_idx = 0
    for spec in output_specs:
        if spec.kind == OutputKind.BUFFER_MUTATION:
            # spec.target is the buffer FQN (e.g. "k_cache")
            fqn_to_out_idx.setdefault(spec.target, out_idx)
        out_idx += 1

    # inputs_to_buffers maps placeholder_name -> buffer FQN.
    # Only present in the extracted subgraph when the partition lowering
    # preserves the mapping; fall back to ``b_`` prefix convention.
    inputs_to_buffers = getattr(sig, "inputs_to_buffers", None) or {}

    result: dict[str, int] = {}
    # Walk input_specs to find BUFFER inputs.
    for spec in getattr(sig, "input_specs", []):
        if spec.kind != InputKind.BUFFER:
            continue
        ph_name = spec.arg.name  # e.g. "b_k_cache"
        fqn = inputs_to_buffers.get(ph_name)
        if fqn is None:
            # Fallback: strip the "b_" prefix to recover the FQN.
            fqn = ph_name[2:] if ph_name.startswith("b_") else ph_name
        if fqn in buffers_to_mutate.values():
            # This buffer is mutated (KV state). Find its output index.
            out_idx_for_fqn = fqn_to_out_idx.get(fqn)
            if out_idx_for_fqn is not None:
                result[ph_name] = out_idx_for_fqn
                logger.debug(
                    "_build_kv_state_out_map: placeholder %r (fqn=%r) -> output[%d]",
                    ph_name,
                    fqn,
                    out_idx_for_fqn,
                )
    return result


def _placeholder_kind(
    exported_program: ExportedProgram,
    name: str,
    kv_state_names: "frozenset[str] | None" = None,
) -> str:
    """Classify a placeholder as user_input / parameter / buffer / constant / kv_state.

    ``kv_state`` is emitted for delegate-owned KV cache buffers: a registered
    buffer that the delegate also mutates (BUFFER + BUFFER_MUTATION in the
    extracted subgraph signature). The SpyreBackend runtime allocates these
    buffers once on the device (zero-initialized), never copies them to/from
    the host after the first call, and captures each call's output back into
    the slot — ping-pong ownership.

    Two sources, in order:

    1. The program's ``graph_signature`` maps (authoritative when populated).
    2. torch.export's lifted-input **name convention** as a fallback. This is
       needed because ``preprocess`` receives the *extracted partition's*
       exported program, whose signature marks every delegate input as
       ``USER_INPUT`` (the top-level param/buffer mapping does not survive
       partition extraction) -- but the placeholder names still carry the
       lifting prefix: ``p_`` parameter, ``b_`` buffer, ``c_`` lifted constant.

    Any placeholder matching neither source is treated as a user input.
    """
    # kv_state: delegate-owned mutable buffers (BUFFER + BUFFER_MUTATION).
    # Must be checked before the generic "buffer" classification below.
    if kv_state_names and name in kv_state_names:
        return "kv_state"

    sig = getattr(exported_program, "graph_signature", None)
    if sig is not None:
        if name in (getattr(sig, "inputs_to_parameters", None) or {}):
            return "parameter"
        if name in (getattr(sig, "inputs_to_buffers", None) or {}):
            return "buffer"
        if name in (getattr(sig, "inputs_to_lifted_tensor_constants", None) or {}):
            return "constant"

    # Fallback: torch.export lifted-input naming convention (survives the
    # partition extraction that clears the signature maps above).
    if name.startswith("p_"):
        return "parameter"
    if name.startswith("b_") and (kv_state_names is None or name not in kv_state_names):
        return "buffer"
    if name.startswith("c_"):
        return "constant"
    return "user_input"


def _concrete_shape(val, overrides, *, allow_symbolic: bool = False) -> list:
    """Resolve a fake tensor's shape to concrete ints.

    ``overrides`` maps *axis -> concrete size* for dims that are symbolic in the
    exported program -- the bucketing case, where an export carries a
    ``Dim(\"seq\")`` and each bucket pins it to one value. A dim that is symbolic
    with no override is an error for **inputs** (``allow_symbolic=False``): the
    bucket loop must supply a concrete size for every dynamic input dim before
    ``compile_fx`` can specialize. For **outputs** (``allow_symbolic=True``) we
    tolerate a symbolic dim and record ``None``, because the v3 path backfills
    concrete output shapes from the compiled result (``_capture_one_compile``).
    """
    dims: list = []
    for axis, s in enumerate(val.shape):
        if overrides and axis in overrides:
            dims.append(int(overrides[axis]))
            continue
        try:
            dims.append(int(s))
        except (TypeError, ValueError):
            if allow_symbolic:
                dims.append(None)
            else:
                raise RuntimeError(
                    f"placeholder has a symbolic size at axis {axis} with no "
                    "bucket override; static bucketing must supply a "
                    "concrete size for every dynamic input dim."
                )
    return dims


def _make_spyre_example_inputs(
    exported_program: ExportedProgram,
    shape_overrides: "dict | None" = None,
) -> tuple[list[torch.Tensor], list[dict], list[dict]]:
    """Build dtype-preserving spyre-device inputs for **every** placeholder.

    Each example tensor keeps the placeholder's true dtype (from
    ``node.meta[\"val\"].dtype``): floats are randomized, non-floats (int64
    token-ids/positions, bool/uint8 masks) are zero-filled. This is the AOT
    half of the mixed-dtype policy -- only float tensors are ever
    downcast to fp16 (by the module ``.half()`` in the export vehicle), never
    integer indices or boolean masks.

    ``shape_overrides`` (bucketing) maps a *placeholder ordinal*
    (its 0-based index among placeholders, == its delegate-input index) to a
    ``{axis: size}`` dict that pins otherwise-symbolic dims to a concrete bucket
    size. It is ``None`` for the static (v2) path, where every dim is already
    concrete in ``meta[\"val\"]``.

    Returns ``(inputs, input_specs, output_specs)`` where each input spec is
    ``{\"shape\": [...], \"dtype\": \"<name>\", \"kind\": \"<user_input|parameter|
    buffer|constant|kv_state>\", \"state_out\": <int>}``; ``state_out``
    is the delegate output index that captures this input's updated value, or ``-1``
    for non-state inputs. A ``kv_state`` spec may later gain
    an optional ``\"device_layout\"`` entry (see ``_iterate_input_layouts``); it is
    absent here and omitted from the payload when no layout was recorded.

    A partitioned subgraph's placeholders include lifted params/buffers (weights,
    biases) as well as user inputs. ExecuTorch passes the stored constant tensors
    to ``execute()`` as delegate-input EValues in **placeholder order**, and the
    C++ delegate copies each arg to the device -- so ``preprocess`` treats *all*
    placeholders uniformly as runtime inputs and emits example tensors for them
    in graph order. Values are irrelevant to SDSC codegen (only shape/dtype
    matter); the ``kind`` is recorded for transparency and to verify arg
    ordering on-device.
    """
    gm = exported_program.graph_module
    # kv_state_out_map: BUFFER+BUFFER_MUTATION (signature-derived, delegate-owned KV).
    kv_state_out_map = _build_kv_state_out_map(exported_program)
    kv_state_names: frozenset[str] = frozenset(kv_state_out_map.keys())
    inputs: list[torch.Tensor] = []
    input_specs: list[dict] = []
    ordinal = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if val is None:
            raise RuntimeError(
                f"placeholder {node.name!r} has no meta['val']; cannot "
                "synthesize an example input for compile_fx."
            )
        overrides = (shape_overrides or {}).get(ordinal)
        shape = _concrete_shape(val, overrides)
        dtype = val.dtype
        if dtype.is_floating_point:
            t = torch.randn(shape, dtype=dtype)
        else:
            t = torch.zeros(shape, dtype=dtype)
        inputs.append(t.to(_SPYRE_DEVICE))
        kind = _placeholder_kind(exported_program, node.name, kv_state_names)
        # state_out: the delegate output that captures this input's updated value
        # (kv_state only: BUFFER+BUFFER_MUTATION from the signature); -1 otherwise.
        state_out_idx = kv_state_out_map.get(node.name, -1)
        input_specs.append(
            {
                "shape": list(shape),
                "dtype": _torch_dtype_to_str(dtype),
                "kind": kind,
                "state_out": state_out_idx,
            }
        )
        ordinal += 1

    output_specs: list[dict] = []
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    outs = output_node.args[0]
    if not isinstance(outs, (tuple, list)):
        outs = (outs,)
    for out in outs:
        val = out.meta.get("val") if hasattr(out, "meta") else None
        if val is None:
            raise RuntimeError("output node arg has no meta['val'].")
        output_specs.append(
            {
                "shape": _concrete_shape(val, None, allow_symbolic=True),
                "dtype": _torch_dtype_to_str(val.dtype),
            }
        )
    return inputs, input_specs, output_specs


def _infer_output_positions(runner, run_args) -> set:
    """Best-effort: which positional args does this kernel *write* (vs read)?

    This is the signal that separates a *produced* buffer from a *consumed*
    intermediate -- the fix for the classifier bug that mislabeled a
    consumed intermediate (the cumsum result feeding kernel 2) as an output
    because it keyed only on "not an input, not the pool -> output".

    The reliable source is the JobPlan's output arity/descriptors on the runner;
    we probe a few candidate attributes and, absent them, fall back to the
    Inductor calling convention: the compiled ``call()``
    wrapper allocates fresh output buffer(s) and appends them at the *end* of the
    launch args, so the trailing tensor arg is the (sole) produced buffer.

    (Follow-up: confirm the real ``SpyreSDSCKernelRunner`` output-descriptor
    attribute name and drop the trailing-arg fallback if a firmer signal exists.)

    Returns a set of positional indices (into ``run_args``) that are outputs.
    """
    tensor_positions = [
        i for i, a in enumerate(run_args) if isinstance(a, torch.Tensor)
    ]
    if not tensor_positions:
        return set()
    for attr in ("num_outputs", "n_outputs", "num_output_tensors"):
        n = getattr(runner, attr, None)
        if isinstance(n, int) and 0 < n <= len(tensor_positions):
            return set(tensor_positions[-n:])
    # Fallback: the trailing tensor arg is the produced output buffer.
    return {tensor_positions[-1]}


def _build_v2_layout(
    captured_runs: list, input_ptr_to_idx: dict, output_ptr_to_idx: dict
) -> tuple[list, list, int]:
    """Reconstruct the multi-kernel (v2) launch DAG from captured run records.

    Pure and torch-free -- it consumes the plain per-arg records ``_spy_run``
    captures (a tensor record has ``{is_tensor, data_ptr, dtype_str, shape, dim,
    is_output_pos}``; a non-tensor record has ``{is_tensor: False}``) so the
    trickiest logic is host-unit-testable without a Spyre device.

    ``captured_runs`` is one entry per kernel, in launch order. For each arg, in
    order:

      1. ``data_ptr`` is a delegate input  -> ``{\"role\":\"input\",\"index\":i}``
      2. ``data_ptr`` is a delegate output -> ``{\"role\":\"output\",\"index\":j}``
      3. 1-D uint8 buffer                  -> ``{\"role\":\"pool\"}`` (asserted shared
         across kernels; sticks lifted to a single top-level ``pool_sticks``)
      4. otherwise it is an inter-kernel intermediate. A buffer id is assigned by
         ``data_ptr``; ``produces`` comes from ``is_output_pos``. If the *same*
         ``data_ptr`` is written by one kernel and read by another the edge is
         device-``resident`` (one id); differing ``data_ptr`` values (the
         host-round-trip case) yield distinct ids, each ``resident: false``.

    Returns ``(kernels, intermediates, pool_sticks)``.
    """
    kernels: list = []
    intermediates: list = []
    ptr_to_buf_id: dict = {}
    pool_sticks = None
    pool_ptr = None

    for records in captured_runs:
        if not records:
            # The runtime rejects a kernel with an empty launch_layout (it no
            # longer falls back to a naive inputs-then-outputs order), so fail at
            # export time rather than emitting a .pte that cannot load.
            raise RuntimeError(
                f"preprocess: kernel {len(kernels)} was launched with no "
                "arguments, so no launch_layout can be recorded for it"
            )
        layout: list = []
        for rec in records:
            if not rec.get("is_tensor"):
                layout.append({"role": "other"})
                continue
            ptr = rec["data_ptr"]
            if ptr in input_ptr_to_idx:
                layout.append({"role": "input", "index": input_ptr_to_idx[ptr]})
            elif ptr in output_ptr_to_idx:
                layout.append({"role": "output", "index": output_ptr_to_idx[ptr]})
            elif rec["dtype_str"] == "uint8" and rec["dim"] == 1:
                sticks = int(rec["shape"][0])
                if pool_sticks is None:
                    pool_sticks, pool_ptr = sticks, ptr
                else:
                    # Exactly one scratch pool is shared across kernels.
                    assert ptr == pool_ptr, (
                        "expected a single shared scratch pool across kernels; "
                        f"saw distinct pool data_ptrs {pool_ptr} vs {ptr}"
                    )
                    assert sticks == pool_sticks, (
                        f"pool sticks disagree across kernels: "
                        f"{pool_sticks} vs {sticks}"
                    )
                layout.append({"role": "pool"})
            else:
                if ptr in ptr_to_buf_id:
                    # Same device buffer seen in an earlier kernel: a genuine
                    # device-resident producer->consumer edge.
                    buf_id = ptr_to_buf_id[ptr]
                    intermediates[buf_id]["resident"] = True
                else:
                    buf_id = len(intermediates)
                    ptr_to_buf_id[ptr] = buf_id
                    entry = {
                        "id": buf_id,
                        "shape": list(rec["shape"]),
                        "dtype": rec["dtype_str"],
                        "resident": False,
                        "delegate_output": ptr in output_ptr_to_idx,
                    }
                    if "const_value" in rec and not rec.get("is_output_pos"):
                        # Read-only constant: the runtime fills it with this
                        # value rather than leaving it uninitialized.
                        entry["init_value"] = rec["const_value"]
                    intermediates.append(entry)
                layout.append(
                    {
                        "role": "intermediate",
                        "buffer": buf_id,
                        "produces": bool(rec.get("is_output_pos")),
                    }
                )
        kernels.append(
            {"subdir": f"{_SPYRECODE_SUBDIR}_{len(kernels)}", "launch_layout": layout}
        )

    return kernels, intermediates, (pool_sticks or 0)


# --- Static bucketing (v3 wire format) -------------------------------------


_BUCKET_COMPILE_SPEC_KEY: Final[str] = "spyre_dynamic_buckets"


def _parse_bucket_spec(compile_specs) -> "dict | None":
    """Extract the bucket directive from the delegate's compile specs.

    The partitioner attaches a ``CompileSpec(key=\"spyre_dynamic_buckets\", value=...)``
    whose JSON payload is ``{\"arg_dims\": [{\"arg\": k, \"axis\": a}, ...],
    \"buckets\": [...]}``::

      * ``arg_dims`` names the dynamic dim(s) by *forward user-input index* and
        axis (``arg`` counts only real inputs, not lifted params/buffers);
      * ``buckets`` is the list of concrete sizes to compile a JobPlan for --
        either bare ints (single dynamic dim) or per-dim tuples.

    Returns the parsed dict, or ``None`` when no bucket spec is present (the
    static v2 path).
    """
    for spec in compile_specs or []:
        if getattr(spec, "key", None) != _BUCKET_COMPILE_SPEC_KEY:
            continue
        value = spec.value
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        return json.loads(value)
    return None


def _normalize_buckets(buckets, ndims: int) -> list:
    """Coerce a bucket list to a list of concrete ``shape_key`` lists.

    A single dynamic dim may be given as bare ints (``[128, 256, 512]``); several
    dims require per-dim tuples (``[[128, 8], [256, 8]]``). Every key must have
    exactly ``ndims`` entries.
    """
    out: list = []
    for b in buckets:
        key = [int(x) for x in b] if isinstance(b, (list, tuple)) else [int(b)]
        if len(key) != ndims:
            raise ValueError(
                f"bucket {b!r} has {len(key)} dims but {ndims} dynamic dim(s) "
                "were declared"
            )
        out.append(key)
    if not out:
        raise ValueError("bucket spec declared no buckets")
    return out


def _resolve_dynamic_dims(
    exported_program: ExportedProgram, gm: torch.fx.GraphModule, arg_dims: list
) -> list:
    """Map export-time ``{\"arg\": k, \"axis\": a}`` specs to (placeholder_ordinal, axis).

    ``arg`` indexes the module's forward user inputs (0-based, signature order);
    the k-th ``user_input`` placeholder is that arg. Lifted params/buffers are
    skipped so bucket dims track real inputs, not weights. The returned ordinal
    is the placeholder's 0-based index among *all* placeholders == its
    delegate-input index, which is what the runtime reads to pick a bucket.
    """
    kv_state_names: frozenset[str] = frozenset(
        _build_kv_state_out_map(exported_program).keys()
    )
    user_positions: list = []
    ordinal = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if (
            _placeholder_kind(exported_program, node.name, kv_state_names)
            == "user_input"
        ):
            user_positions.append(ordinal)
        ordinal += 1
    resolved: list = []
    for ad in arg_dims:
        a = int(ad["arg"])
        if a >= len(user_positions):
            raise ValueError(
                f"bucket arg index {a} out of range: the subgraph has "
                f"{len(user_positions)} user input(s)"
            )
        resolved.append((user_positions[a], int(ad["axis"])))
    return resolved


def _spyrecode_dir(code_dir: str) -> str:
    """Return ``<code_dir>/spyreCodeDir`` (the JobPlan bundle prepare_kernel eats)."""
    d = os.path.join(code_dir, _SPYRECODE_SUBDIR)
    if not os.path.isdir(d):
        raise RuntimeError(
            f"preprocess: expected {_SPYRECODE_SUBDIR!r} under {code_dir!r}, "
            "but it is missing."
        )
    return d


def _release_runner_jobplan(runner) -> None:
    """Release device Program memory associated with a SpyreSDSCKernelRunner.

    SpyreSDSCKernelRunner lazily initializes `_jobplan` on first execution,
    holding a reference to a `torch_spyre._C.JobPlan` (and its underlying
    Flex device allocation). Torch-spyre does not currently provide a public
    `release()` or `close()` method on SpyreSDSCKernelRunner. Setting `_jobplan`
    to None releases the C++ JobPlan handle so device memory is immediately freed.
    """
    if hasattr(runner, "release_jobplan"):
        runner.release_jobplan()
    elif hasattr(runner, "_jobplan"):
        runner._jobplan = None


def _check_capture_counts(captured_dirs: list, captured_runs: list) -> None:
    """Require one observed kernel launch per emitted SpyreCode dir.

    ``_build_v2_layout`` derives ``kernels[]`` from ``captured_runs`` while the
    tar is built from ``captured_dirs``; if the two disagree the payload would
    pair launch layouts with the wrong SpyreCode dirs (or have none at all).
    """
    if len(captured_runs) == len(captured_dirs):
        return
    raise RuntimeError(
        f"preprocess: observed {len(captured_runs)} SpyreSDSCKernelRunner.run "
        f"launch(es) but {len(captured_dirs)} SpyreCode dir(s) were emitted; "
        "they must match one-to-one. If no launch was observed, the "
        "SpyreSDSCKernelRunner.run hook did not fire (torch-spyre may have "
        "changed how kernels are launched)."
    )


def _op_base_name(target) -> str:
    """Return the bare aten op name (e.g. ``addmm``) for an FX call target.

    Handles torch ``OpOverload`` and ExecuTorch ``EdgeOpOverload`` (both expose
    ``name()`` as a *method* returning ``"aten::addmm"``) as well as plain
    callables / strings.
    """
    n = getattr(target, "name", None)
    if callable(n):
        n = n()
    if not isinstance(n, str):
        n = getattr(target, "__name__", "") or ""
    parts = n.replace("::", ".").split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


def _describe_compile_failure(gm: torch.fx.GraphModule, exc: BaseException) -> str:
    """Name the FX node(s) / op target(s) a compile failure most likely refers to.

    Inductor errors rarely say which delegated node broke. This returns a
    message that (a) lists every node whose name or op appears in the error text
    ("suspects") and (b) summarizes the subgraph's op histogram, so a failure in
    ``preprocess`` can be traced back to a concrete ``node.name`` / ``node.target``
    and fed to ``supported_ops.yaml`` or an upstream issue. Pure and device-free.
    """
    text = f"{type(exc).__name__}: {exc}"
    suspects = []
    hist: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        base = _op_base_name(node.target)
        hist[base] = hist.get(base, 0) + 1
        name_hit = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(node.name)}(?![A-Za-z0-9_])", text
        )
        op_hit = re.search(rf"(?<![A-Za-z0-9_]){re.escape(base)}(?![A-Za-z0-9_])", text)
        if name_hit or op_hit:
            suspects.append(f"{node.name} (target={base})")
    lines = [f"compile failed: {text}"]
    if suspects:
        lines.append("  suspect node(s): " + ", ".join(suspects[:10]))
    else:
        lines.append("  suspect node(s): none named in the error text")
    ops = ", ".join(f"{k} x{v}" for k, v in sorted(hist.items()))
    lines.append(f"  subgraph ops: {ops or '(none)'}")
    return "\n".join(lines)


def _capture_one_compile(gm, example_inputs, cfx, ac, kr):
    """Compile one concrete example-input set and capture what preprocess needs.

    A faithful extraction of the capture machinery ``preprocess`` has always
    used: redirect ``get_output_dir`` so each emitted SpyreCode dir lands in a
    tempdir we own, intercept every ``SpyreSDSCKernelRunner.run`` to record the
    per-arg launch layout (keyed on ``data_ptr``), then compile + run once to
    trigger SDSC codegen. A single call reproduces the verified static-path behavior;
    the bucket loop calls it once per shape bucket.

    Returns ``(captured_dirs, captured_runs, input_ptr_to_idx,
    output_ptr_to_idx, output_specs)`` -- ``output_specs`` read from the concrete
    compiled result (so dynamic output dims come out concrete per bucket).
    """
    captured_dirs: list[str] = []
    original_get_output_dir = ac.get_output_dir

    def _patched_get_output_dir(kernel_name, *args, **kwargs):
        owned = tempfile.mkdtemp(prefix=f"spyre_sdsc_{kernel_name}_")
        captured_dirs.append(owned)
        return owned

    input_ptr_to_idx = {t.data_ptr(): i for i, t in enumerate(example_inputs)}
    captured_runs: list[list[dict]] = []
    executed_runners: set = set()
    original_run = kr.SpyreSDSCKernelRunner.run

    def _spy_run(self, *run_args, **run_kwargs):
        executed_runners.add(self)
        out_positions = _infer_output_positions(self, run_args)
        records: list[dict] = []
        for pos, a in enumerate(run_args):
            if not isinstance(a, torch.Tensor):
                records.append({"is_tensor": False})
                continue
            rec = {
                "is_tensor": True,
                "data_ptr": a.data_ptr(),
                "dtype_str": _torch_dtype_to_str(a.dtype),
                "shape": [int(s) for s in a.shape],
                "dim": a.dim(),
                "is_output_pos": pos in out_positions,
            }
            # A read-only single-element operand that is not a delegate input is
            # a compile-time constant (e.g. spyre_constant_tensor for the SDPA
            # scale). Capture its value so the runtime can materialize it
            # instead of handing the kernel an uninitialized buffer.
            if (
                a.numel() == 1
                and pos not in out_positions
                and a.data_ptr() not in input_ptr_to_idx
            ):
                rec["const_value"] = float(a.detach().cpu().reshape(()).item())
            records.append(rec)
        captured_runs.append(records)
        return original_run(self, *run_args, **run_kwargs)

    try:
        ac.get_output_dir = _patched_get_output_dir
        kr.SpyreSDSCKernelRunner.run = _spy_run
        try:
            compiled = cfx.compile_fx(gm, example_inputs)
            raw_out = compiled(*example_inputs)  # trigger SDSC codegen
        except Exception as e:
            # Keep the original exception type (callers may match on it) and
            # attach the node-level diagnosis as a note. The diagnosis is
            # best-effort: it must never replace the compile error it describes.
            try:
                diagnosis = _describe_compile_failure(gm, e)
            except Exception:
                logger.debug(
                    "preprocess: could not diagnose compile failure", exc_info=True
                )
            else:
                logger.error("preprocess: %s", diagnosis)
                e.add_note("preprocess: " + diagnosis)
            raise
        finally:
            ac.get_output_dir = original_get_output_dir
            kr.SpyreSDSCKernelRunner.run = original_run
            for r in executed_runners:
                _release_runner_jobplan(r)
            executed_runners.clear()
            import gc

            gc.collect()

        _check_capture_counts(captured_dirs, captured_runs)

        outs = raw_out if isinstance(raw_out, (tuple, list)) else (raw_out,)
        output_ptr_to_idx: dict[int, int] = {}
        output_specs: list[dict] = []
        for j, o in enumerate(outs):
            if isinstance(o, torch.Tensor):
                output_ptr_to_idx.setdefault(o.data_ptr(), j)
                output_specs.append(
                    {
                        "shape": [int(s) for s in o.shape],
                        "dtype": _torch_dtype_to_str(o.dtype),
                    }
                )
        _attach_output_layouts(output_specs, outs)
        return (
            captured_dirs,
            captured_runs,
            input_ptr_to_idx,
            output_ptr_to_idx,
            output_specs,
        )
    except BaseException:
        _remove_dirs(captured_dirs)
        raise


def _build_v3_header(
    dynamic_input_dims: list,
    buckets: list,
    rep_input_specs: list,
    rep_output_specs: list,
    n_retargeted: int,
) -> dict:
    """Assemble the v3 (bucketed) delegate header.

    Pure and device-free so the grouping can be unit-tested on the Mac. ``v3`` is
    ``v2`` + a ``buckets`` array (each entry the v2 kernels/intermediates/pool
    sub-schema plus a ``shape_key`` and ``subdir_prefix``) + a top-level
    ``dynamic_input_dims`` (``[[input_index, axis], ...]``) telling the runtime
    which arg dim to read to select a bucket. ``inputs``/``outputs`` mirror the
    first bucket so a v3 payload degrades to v2 semantics on the legacy fields.
    """
    return {
        "version": 3,
        "backend": SPYRE_BACKEND_ID,
        "n_inputs": len(rep_input_specs),
        "n_outputs": len(rep_output_specs),
        "inputs": rep_input_specs,
        "outputs": rep_output_specs,
        "n_retargeted_ops": n_retargeted,
        "dynamic_input_dims": dynamic_input_dims,
        "buckets": buckets,
    }


class SpyreBackend(BackendDetails):
    """``preprocess`` that emits a SpyreCode/JobPlan blob for the Spyre delegate."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        import torch_spyre  # noqa: F401  registers the "spyre" PrivateUse1 device

        torch_spyre._autoload()
        import torch._inductor.compile_fx as cfx
        import torch._inductor.config as _inductor_config
        import torch_spyre.execution.async_compile as ac
        import torch_spyre.execution.kernel_runner as kr

        global _delegate_compile_counter
        with _delegate_compile_lock:
            delegate_idx = _delegate_compile_counter
            _delegate_compile_counter += 1

        n_nodes = sum(
            1 for n in edge_program.graph_module.graph.nodes if n.op == "call_function"
        )

        # Inductor's async-compile subprocess pool (SubprocPool) is driven by
        # compile_threads (default 32). When the pool's readiness probe completes
        # before _submit_backend_compiler is reached, SpyreAsyncCompile routes
        # dbo-opt invocations into the pool. The SubprocPool worker unpickles
        # _run_backend_compiler, which imports torch_spyre._inductor.constants
        # -> torch_spyre._C, requiring the sentient shared libraries on
        # LD_LIBRARY_PATH. On a large graph (e.g. 40-layer LLM) the lowering
        # phase is long enough that the probe completes before the first SDSC
        # compile, making the subprocess path non-deterministically active.
        # preprocess is always synchronous (compile + run in one shot), so
        # there is no benefit from parallelism here; pin to 1 thread to
        # guarantee inline (in-process) compilation on every graph size.
        # Optional ExecuTorch-owned passes for torch-spyre's pre-scheduling
        # pipeline, selected by name through a CompileSpec. With no such spec
        # this is a no-op and preprocess behaves exactly as before.
        pass_names = parse_pass_names(compile_specs)
        t0 = time.monotonic()
        with (
            _inductor_config.patch({"compile_threads": 1}),
            spyre_pass_scope(pass_names) as pass_scope,
        ):
            result = SpyreBackend._preprocess_impl(
                edge_program, compile_specs, cfx, ac, kr
            )
            pass_scope.require_applied()
        elapsed = time.monotonic() - t0
        print(
            f"DELEGATE_COMPILE #{delegate_idx}: {elapsed:.1f}s  nodes={n_nodes}",
            flush=True,
        )
        return result

    @staticmethod
    def _preprocess_impl(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
        cfx,
        ac,
        kr,
    ) -> PreprocessResult:
        # A bucket directive routes to the v3 (bucketed) path,
        # which compiles one JobPlan per shape bucket. Absent it, fall through to
        # the static (v2) path below.
        bucket_spec = _parse_bucket_spec(compile_specs)
        if bucket_spec is not None:
            return SpyreBackend._preprocess_bucketed(
                edge_program, bucket_spec, cfx, ac, kr
            )

        gm = edge_program.graph_module
        n_retargeted = _retarget_edge_to_aten(gm)

        example_inputs, input_specs, output_specs = _make_spyre_example_inputs(
            edge_program
        )

        # ---- Compile + fixed-point input-layout iteration ---------------------
        # Recompiles until each kv_state input's layout equals the layout of
        # its state output (input_specs[i]["device_layout"] is recorded on
        # convergence; raises RuntimeError past _MAX_LAYOUT_ITERS). Graphs with
        # no kv_state compile exactly once, as before.
        (
            captured_dirs,
            captured_runs,
            input_ptr_to_idx,
            output_ptr_to_idx,
            compiled_output_specs,
        ) = _iterate_input_layouts(gm, example_inputs, input_specs, cfx, ac, kr)
        try:
            # Keep the graph-derived output_specs (unchanged payload content) and
            # only graft on the compiled device layouts, as _attach_output_layouts did.
            for j, cs in enumerate(compiled_output_specs):
                if j < len(output_specs) and "device_layout" in cs:
                    output_specs[j]["device_layout"] = cs["device_layout"]

            logger.info(
                "preprocess: output device layouts %r",
                [s.get("device_layout") for s in output_specs],
            )
            logger.info(
                "preprocess: kv_state input layouts %r",
                [s.get("device_layout") for s in input_specs],
            )

            if not captured_dirs:
                raise RuntimeError(
                    "preprocess: no SDSC kernel was emitted (get_output_dir was never "
                    "called). compile_fx may not have reached SDSC codegen -- check "
                    "that every op in the subgraph is a supported SDSC lowering."
                )

            # Every static-path payload is v2, including a single SDSC kernel. A
            # single kernel that allocates a workspace tensor (bmm/matmul accumulator)
            # is not representable in v1, so the v2 ``intermediates`` table is the
            # only layout emitted here.
            # ---- v2 payload (one or more kernels) ---------------------------------
            # A partition may compile to >1 SDSC kernel (a host-fallback op
            # such as ``cumsum`` forces a split), or to a single one. Tar each
            # kernel's spyreCodeDir under a distinct ``spyreCodeDir_{i}`` arcname
            # (i in [0, N)) and emit the v2 header (a ``kernels`` array over one
            # shared scratch pool, plus an ``intermediates`` table of
            # activation/workspace buffers the C++ delegate allocates).
            logger.info(
                "preprocess: %d SDSC kernels emitted -> v2 multi-kernel payload (%r)",
                len(captured_dirs),
                captured_dirs,
            )
            kernels, intermediates, pool_sticks = _build_v2_layout(
                captured_runs, input_ptr_to_idx, output_ptr_to_idx
            )
            logger.info(
                "preprocess: v2 layout kernels=%r intermediates=%r (pool_sticks=%d)",
                kernels,
                intermediates,
                pool_sticks,
            )

            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                tar_path = tmp.name
            try:
                with tarfile.open(tar_path, "w") as tf:
                    for i, code_dir in enumerate(captured_dirs):
                        tf.add(
                            _spyrecode_dir(code_dir), arcname=f"{_SPYRECODE_SUBDIR}_{i}"
                        )
                with open(tar_path, "rb") as f:
                    tar_bytes = f.read()
            finally:
                os.unlink(tar_path)

            header = {
                "version": 2,
                "backend": SPYRE_BACKEND_ID,
                "n_inputs": len(input_specs),
                "n_outputs": len(output_specs),
                "inputs": input_specs,
                "outputs": output_specs,
                "n_retargeted_ops": n_retargeted,
                "pool_sticks": pool_sticks,
                "kernels": kernels,
                "intermediates": intermediates,
            }
            processed = _pack_payload(header, tar_bytes)
            return PreprocessResult(processed_bytes=processed)
        finally:
            _remove_dirs(captured_dirs)

    @staticmethod
    def _preprocess_bucketed(
        edge_program: ExportedProgram,
        bucket_spec: dict,
        cfx,
        ac,
        kr,
    ) -> PreprocessResult:
        """Static bucketing (v3 wire format): compile one concrete JobPlan set per shape bucket.

        Spyre's ``prepareKernel`` compiles for a FIXED shape, and torch-spyre
        concretizes the SDSC to the example-input shape (confirmed on device --
        true symbolic JobPlans are upstream-blocked, torch-spyre #2434/#220/
        #1371-1373). So a dynamic seq/batch dim is served by compiling the
        subgraph once per declared bucket size and selecting at runtime by input
        shape.

        Each bucket reuses the verified static capture + layout machinery
        unchanged (``_capture_one_compile`` + ``_build_v2_layout``, even for a
        single-kernel bucket -- it just yields one ``spyreCodeDir_0``). The one
        tar carries every bucket's kernels under ``bucket{b}/spyreCodeDir_{i}``;
        the v3 header lists the buckets with their ``shape_key`` +
        ``subdir_prefix`` and a top-level ``dynamic_input_dims`` telling the
        runtime which arg dim(s) to read to pick a bucket.
        """
        gm = edge_program.graph_module
        # Retarget edge->aten ONCE on the shared graph module; every bucket
        # compile runs against the same (already-core-aten) graph.
        n_retargeted = _retarget_edge_to_aten(gm)

        arg_dims = bucket_spec.get("arg_dims") or []
        if not arg_dims:
            raise ValueError(
                "bucket spec has no 'arg_dims'; cannot tell which input dim(s) "
                "vary across buckets."
            )
        resolved = _resolve_dynamic_dims(edge_program, gm, arg_dims)
        bucket_keys = _normalize_buckets(
            bucket_spec.get("buckets") or [], len(resolved)
        )
        dynamic_input_dims = [[ordinal, axis] for (ordinal, axis) in resolved]

        logger.info(
            "preprocess: bucketed (v3) path -- dynamic_input_dims=%r, buckets=%r",
            dynamic_input_dims,
            bucket_keys,
        )

        buckets: list[dict] = []
        # (arcname, on-disk spyreCodeDir) pairs, tarred together at the end.
        tar_entries: list[tuple[str, str]] = []
        rep_input_specs: "list | None" = None
        rep_output_specs: "list | None" = None
        # Every bucket's emitted SpyreCode dirs; removed once the tar is built.
        all_dirs: list[str] = []

        try:
            for b, key in enumerate(bucket_keys):
                # Map each resolved (placeholder_ordinal, axis) to this bucket's size.
                shape_overrides: dict[int, dict[int, int]] = {}
                for (ordinal, axis), size in zip(resolved, key):
                    shape_overrides.setdefault(ordinal, {})[axis] = int(size)

                example_inputs, input_specs, _ = _make_spyre_example_inputs(
                    edge_program, shape_overrides=shape_overrides
                )
                # Fixed-point input-layout iteration (same as the static path).
                (
                    captured_dirs,
                    captured_runs,
                    input_ptr_to_idx,
                    output_ptr_to_idx,
                    output_specs,
                ) = _iterate_input_layouts(gm, example_inputs, input_specs, cfx, ac, kr)
                all_dirs.extend(captured_dirs)

                if not captured_dirs:
                    raise RuntimeError(
                        f"preprocess: bucket {key!r} emitted no SDSC kernel "
                        "(get_output_dir was never called)."
                    )

                # Reuse the v2 layout builder for every bucket -- a single-kernel
                # bucket just yields one kernel with subdir 'spyreCodeDir_0'.
                kernels, intermediates, pool_sticks = _build_v2_layout(
                    captured_runs, input_ptr_to_idx, output_ptr_to_idx
                )

                subdir_prefix = f"bucket{b}"
                for i, code_dir in enumerate(captured_dirs):
                    arcname = f"{subdir_prefix}/{_SPYRECODE_SUBDIR}_{i}"
                    tar_entries.append((arcname, _spyrecode_dir(code_dir)))

                buckets.append(
                    {
                        "shape_key": key,
                        "subdir_prefix": subdir_prefix,
                        "pool_sticks": pool_sticks,
                        "kernels": kernels,
                        "intermediates": intermediates,
                        "inputs": input_specs,
                        "outputs": output_specs,
                    }
                )
                logger.info(
                    "preprocess: bucket %d shape_key=%r -> %d kernel(s), "
                    "pool_sticks=%d",
                    b,
                    key,
                    len(kernels),
                    pool_sticks,
                )
                if rep_input_specs is None:
                    rep_input_specs, rep_output_specs = input_specs, output_specs

            with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
                tar_path = tmp.name
            try:
                with tarfile.open(tar_path, "w") as tf:
                    for arcname, code_dir in tar_entries:
                        tf.add(code_dir, arcname=arcname)
                with open(tar_path, "rb") as f:
                    tar_bytes = f.read()
            finally:
                os.unlink(tar_path)

            header = _build_v3_header(
                dynamic_input_dims,
                buckets,
                rep_input_specs,
                rep_output_specs,
                n_retargeted,
            )
            processed = _pack_payload(header, tar_bytes)
            return PreprocessResult(processed_bytes=processed)
        finally:
            _remove_dirs(all_dirs)
