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

"""Add ExecuTorch-owned passes to torch-spyre's pre-scheduling (LoopLevel IR) pipeline.

torch-spyre runs ``CustomPreSchedulingPasses`` on the ``GraphLowering`` right
before the Inductor ``Scheduler`` is built. The pass list is private to that
class (``self.passes``, built in ``__init__``) and torch-spyre offers no
registration hook. This module adds one from the outside:

* ``register_pre_scheduling_pass`` records a pass function and an anchor
  (``after=`` / ``before=`` another pass, by name).
* ``spyre_pass_scope(names)`` makes compiles started *inside* the ``with`` block
  (and only those) build their pipeline with the named passes inserted.
* ``CustomPreSchedulingPasses.__init__`` is wrapped once, idempotently, on the
  first scope entry (never at import). Outside a scope the wrapper does nothing,
  so the pass list, ``uuid()`` and the compiled output are those of unpatched
  torch-spyre.
* The ET backend carries pass NAMES in a ``CompileSpec`` (see
  ``pass_compile_spec``); ``SpyreBackend.preprocess`` enters the scope.

Example::

    from executorch.backends.spyre.spyre_pre_scheduling_passes import (
        register_pre_scheduling_pass, pass_compile_spec,
    )

    @register_pre_scheduling_pass(after="assign_dim_hints")
    def my_pass(graph):          # Callable[[GraphLowering], None]
        for op in graph.operations:
            op.get_read_writes()  # read-only

    SpyrePartitioner(compile_specs=[pass_compile_spec(["my_pass"])])

The module imports nothing from torch or torch-spyre at import time.

Pass contract
-------------
* Signature ``Callable[[GraphLowering], None]``; unique ``__name__``.
* Mutate ``graph.operations`` in place. Never rebind it.
* After IR edits, call ``invalidate_op_read_writes`` where the cached
  read/write sets may be stale. Read-only passes should use
  ``op.get_read_writes()``, not ``op_read_writes`` (which memoizes into
  ``op.__dict__``).
* Order: execution order is determined by the anchors, never by the order of
  names given to ``spyre_pass_scope`` / the CompileSpec. Several passes with
  the same anchor keep their *registration* order (``after=X``: X, p1, p2;
  ``before=X``: p1, p2, X). An anchor is a name in torch-spyre's list or the
  name of an earlier-registered, also selected, pass.
* Anchor names are the ``_get_pass_name`` names of torch-spyre's list. They
  differ between releases; a missing/ambiguous anchor raises with the actual
  list. Main (64a818de): ``splice_while_loops``, ``deadcode_elimination``,
  ``propagate_named_dims``, ``validate_named_dims``, ``assign_dim_hints``,
  ``insert_bmm_padding``, ``split_multi_ops``, ``propagate_spyre_tensor_layouts``,
  ``validate_ops``, ``optimize_restickify_locations``,
  ``reorder_nonstick_dims``, ``finalize_layouts``, ``insert_restickify``,
  ``validate_no_restickify_on_mutation_targets``,
  ``enforce_indirect_access_layout``, ``reorder_nonstick_dims_mutation``,
  ``insert_post_mutation_restickify``, ``insert_restickify_padding``,
  ``dedup_and_promote_constants``, ``_maybe_coarse_tile_span_overflow``,
  ``plan_sharded_hbm``, ``span_reduction``, ``_distribute_work``,
  ``_maybe_scratchpad_planning``, ``elide_proven_read_copies``. rc.3 (46001de)
  has the same set except that it lacks ``plan_sharded_hbm`` and
  ``reorder_nonstick_dims_mutation``, and adds
  ``_maybe_reorder_unhinted_interlopers`` and ``_maybe_coarse_tile_hints``.
* Main only: ``graph.operations`` may be a ``loop_node.OpsView`` while a pass
  runs (iterate ``all_ops`` for leaf ops), a rebind raises in
  ``_flat_operations``, and ``_PASSES_RUN_EXPANDED`` names are reserved
  (torch-spyre ``passes.py:460-493``). rc.3 has none of these; passes see the
  plain list.

Scope, threads, caches
----------------------
* A scope lives in a ``contextvars.ContextVar``. The torch-spyre pipeline is
  constructed on the thread that calls ``compile_fx``, so it is sufficient
  there. A new ``threading.Thread`` does NOT inherit the scope (use
  ``contextvars.copy_context().run``); concurrent threads with different scopes
  are independent.
* Out-of-process/async FX compile (``TORCHINDUCTOR_FX_COMPILE_MODE`` =
  subprocess/serialize, async, progressive) would build the pipeline outside
  the scope, so a scope with passes raises in that situation.
* Inside a scope with passes the Inductor FX graph cache and the AOTAutograd
  cache are disabled (scoped ``config.patch``, restored on exit; INFO-logged
  once). Those caches can return a result without building the pipeline, so the
  passes would be skipped silently. The torch-spyre ``sdsc_cache`` and
  ``spyre_kernel_cache`` are keyed on post-pipeline artifacts and stay enabled.
  The patched torch config is process-global: compiles on other threads
  lose those caches for as long as any scope with passes is open.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from executorch.exir.backend.compile_spec_schema import CompileSpec

logger = logging.getLogger(__name__)

PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY: Final[str] = "spyre_pre_scheduling_passes"

__all__ = [
    "PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY",
    "PassInfo",
    "PreSchedulingPassError",
    "list_pre_scheduling_passes",
    "parse_pass_names",
    "pass_compile_spec",
    "register_pre_scheduling_pass",
    "spyre_pass_scope",
]

_WRAPPER_MARK: Final[str] = "_spyre_et_pre_scheduling_wrapper"


class PreSchedulingPassError(RuntimeError):
    """A pass scope/registry/structure problem. Never swallowed."""


class PassInfo(NamedTuple):
    name: str
    anchor: str
    position: str  # "after" | "before"


@dataclass(frozen=True)
class _Registered:
    name: str
    fn: Callable[..., None]
    anchor: str
    position: str


_registry: Dict[str, _Registered] = {}
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def register_pre_scheduling_pass(
    fn: Optional[Callable[..., None]] = None,
    *,
    after: Optional[str] = None,
    before: Optional[str] = None,
    name: Optional[str] = None,
):
    """Register ``fn`` to run ``after``/``before`` the named pass.

    Usable as ``register_pre_scheduling_pass(fn, after="x")`` or as a decorator
    ``@register_pre_scheduling_pass(after="x")``. Returns ``fn`` unchanged.
    Registering does not activate anything; see :func:`spyre_pass_scope`.
    """
    if (after is None) == (before is None):
        raise ValueError("exactly one of `after` / `before` is required")
    anchor = after if after is not None else before
    position = "after" if after is not None else "before"
    if not isinstance(anchor, str) or not anchor:
        raise ValueError(f"anchor must be a non-empty pass name, got {anchor!r}")

    def _register(f: Callable[..., None]) -> Callable[..., None]:
        if not callable(f):
            raise TypeError(f"pass must be callable, got {f!r}")
        pass_name = name if name is not None else getattr(f, "__name__", None)
        if not isinstance(pass_name, str) or not pass_name:
            raise ValueError(f"cannot derive a pass name for {f!r}; give name=")
        installed = f
        if pass_name != getattr(f, "__name__", None):
            installed = _renamed(f, pass_name)
        with _lock:
            if pass_name in _registry:
                raise ValueError(
                    f"pre-scheduling pass {pass_name!r} already registered"
                )
            _registry[pass_name] = _Registered(pass_name, installed, anchor, position)
        return f

    return _register if fn is None else _register(fn)


def _renamed(fn: Callable[..., None], new_name: str) -> Callable[..., None]:
    """Thin wrapper whose ``__name__`` is ``new_name``.

    torch-spyre names a pass by ``__name__`` and hashes ``inspect.getfile`` of
    each pass for ``uuid()``. ``_pass_sources`` (set by torch-spyre's own
    ``@_runs``: main passes.py:370-374, rc.3 passes.py:346-350) makes it hash
    the real pass's file rather than this module.
    """

    def wrapper(graph: Any) -> None:
        fn(graph)

    wrapper.__name__ = new_name
    wrapper.__qualname__ = new_name
    setattr(wrapper, "_pass_sources", (fn,))
    return wrapper


def list_pre_scheduling_passes() -> Tuple[PassInfo, ...]:
    """Registered passes in registration order (for debugging/error messages)."""
    with _lock:
        return tuple(PassInfo(r.name, r.anchor, r.position) for r in _registry.values())


def _registered_names() -> List[str]:
    with _lock:
        return list(_registry)


# ---------------------------------------------------------------------------
# CompileSpec
# ---------------------------------------------------------------------------


def pass_compile_spec(names: Sequence[str]) -> CompileSpec:
    """``CompileSpec`` selecting the registered passes ``names``.

    The value is a JSON list of names as UTF-8 bytes. Order is irrelevant
    (anchors decide execution order); duplicates raise.
    """
    checked = _check_names(names)
    return CompileSpec(
        PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY, json.dumps(list(checked)).encode()
    )


def parse_pass_names(compile_specs: Optional[Iterable[CompileSpec]]) -> Tuple[str, ...]:
    """Pass names carried by ``compile_specs`` (empty when the key is absent)."""
    found = [
        s
        for s in (compile_specs or [])
        if getattr(s, "key", None) == PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY
    ]
    if not found:
        return ()
    if len(found) > 1:
        raise PreSchedulingPassError(
            f"multiple {PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY!r} CompileSpecs; "
            "pass one spec with all names"
        )
    value = found[0].value
    try:
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        parsed = json.loads(value)
    except (UnicodeDecodeError, ValueError, TypeError) as e:
        raise PreSchedulingPassError(
            f"{PRE_SCHEDULING_PASSES_COMPILE_SPEC_KEY!r} CompileSpec is not a JSON "
            f"list of names: {e}"
        ) from e
    return _check_names(parsed)


def _check_names(names: Any) -> Tuple[str, ...]:
    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
        raise PreSchedulingPassError(
            f"pass names must be a list/tuple of str, got {type(names).__name__}"
        )
    bad = [n for n in names if not isinstance(n, str) or not n]
    if bad:
        raise PreSchedulingPassError(f"pass names must be non-empty str, got {bad!r}")
    dups = sorted({n for n in names if list(names).count(n) > 1})
    if dups:
        raise PreSchedulingPassError(f"duplicate pass names: {dups}")
    return tuple(names)


# ---------------------------------------------------------------------------
# Target (torch-spyre) resolution -- lazy, replaceable in tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    cls: type
    get_pass_name: Callable[[Callable[..., Any]], str]
    reserved_names: frozenset
    describe: Callable[[], str]


@functools.lru_cache(maxsize=None)
def _git_head(directory: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", directory, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        head = out.stdout.strip()
        return head if out.returncode == 0 and head else "unavailable"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _describe_torch_spyre() -> str:
    try:
        pkg = importlib.import_module("torch_spyre")
    except Exception as e:  # pragma: no cover - only when torch-spyre is broken
        return f"torch-spyre not importable ({e})"
    path = getattr(pkg, "__file__", None) or "?"
    version = "unknown"
    try:
        from importlib import metadata

        for dist in ("torch_spyre", "torch-spyre"):
            try:
                version = metadata.version(dist)
                break
            except metadata.PackageNotFoundError:
                continue
    except Exception:  # pragma: no cover
        pass
    head = _git_head(os.path.dirname(os.path.dirname(path))) if path != "?" else "n/a"
    return f"torch-spyre version={version} git HEAD={head} path={path}"


def _resolve_target() -> _Target:
    """Import torch-spyre's passes module (first use only)."""
    mod = importlib.import_module("torch_spyre._inductor.passes")
    cls = getattr(mod, "CustomPreSchedulingPasses", None)
    if cls is None:
        raise PreSchedulingPassError(
            "torch_spyre._inductor.passes has no CustomPreSchedulingPasses; "
            f"unsupported torch-spyre. {_describe_torch_spyre()}"
        )
    get_name = getattr(mod, "_get_pass_name", None) or _fallback_pass_name
    reserved = frozenset(getattr(mod, "_PASSES_RUN_EXPANDED", ()))
    return _Target(cls, get_name, reserved, _describe_torch_spyre)


def _fallback_pass_name(fn: Callable[..., Any]) -> str:
    return getattr(fn, "__name__", type(fn).__name__)


# Seam for unit tests (fake class mirroring torch-spyre's structure).
_target_resolver: Callable[[], _Target] = _resolve_target


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


class _Scope:
    """One ``spyre_pass_scope`` activation."""

    __slots__ = ("names", "parent", "instances", "target")

    def __init__(self, names: Tuple[str, ...], parent: Optional["_Scope"]):
        self.names = names
        self.parent = parent
        self.instances = 0  # pipelines built under this scope
        self.target: Optional[_Target] = None

    def effective_names(self) -> frozenset:
        out, s = set(), self
        while s is not None:
            out.update(s.names)
            s = s.parent
        return frozenset(out)

    def hit(self) -> None:
        s: Optional[_Scope] = self
        while s is not None:
            s.instances += 1
            s = s.parent

    def require_applied(self) -> None:
        """Raise unless a pipeline was built under this scope.

        Guards against the wrapper having been removed by someone else's patch
        while the scope was open: the compile would run without the passes.
        """
        if self.target is not None:
            cls = self.target.cls
            top = cls.__dict__.get("__init__", cls.__init__)
            if getattr(top, _WRAPPER_MARK, None) is not top:
                raise PreSchedulingPassError(
                    "CustomPreSchedulingPasses.__init__ was replaced by another "
                    "patch while a pass scope was open; the patch may have "
                    "discarded the injected passes. Refusing to continue "
                    f"[{self.target.describe()}]"
                )
        if self.names and self.instances == 0:
            raise PreSchedulingPassError(
                f"pass scope {list(self.names)} was active but no "
                "CustomPreSchedulingPasses was constructed under it, so the "
                "passes did not run (compile skipped the pipeline, or "
                "CustomPreSchedulingPasses.__init__ was replaced by another "
                "patch). Refusing to continue without the passes."
            )


_ACTIVE: contextvars.ContextVar[Optional[_Scope]] = contextvars.ContextVar(
    "spyre_pre_scheduling_scope", default=None
)


@contextlib.contextmanager
def spyre_pass_scope(names: Sequence[str]) -> Iterator[_Scope]:
    """Compiles started inside this block build their pipeline with ``names``.

    Nested scopes combine (union); leaving restores the previous state exactly.
    An empty ``names`` is a no-op: nothing is checked, patched or disabled.
    The yielded scope has ``require_applied()``: call it after the compile to
    fail loudly if no pipeline was built under the scope (so the passes did not
    run) or if another patch displaced our ``__init__`` wrapper meanwhile.
    """
    checked = _check_names(names)
    parent = _ACTIVE.get()
    scope = _Scope(checked, parent)
    if not checked:
        yield scope
        return

    known = _registered_names()
    unknown = [n for n in checked if n not in known]
    if unknown:
        raise PreSchedulingPassError(
            f"unknown pre-scheduling pass(es) {unknown}; registered: {known}"
        )
    _check_fx_compile_mode()
    target = _target_resolver()
    _ensure_installed(target)
    scope.target = target
    with _caches_disabled():
        token = _ACTIVE.set(scope)
        try:
            yield scope
        finally:
            _ACTIVE.reset(token)


def _check_fx_compile_mode() -> None:
    try:
        import torch._inductor.compile_fx as cfx

        normal = cfx.FxCompileMode.NORMAL
        mode = cfx.fx_compile_mode
        is_async = cfx.fx_compile_async
        progressive = cfx.fx_compile_progressive
    except (ImportError, AttributeError) as e:
        raise PreSchedulingPassError(
            "cannot determine torch._inductor.compile_fx FX compile mode "
            f"({e}); refusing to run passes that might not reach the compile"
        ) from e
    if mode != normal or is_async or progressive:
        raise PreSchedulingPassError(
            "pre-scheduling passes need the in-process FX compile "
            f"(fx_compile_mode={mode}, async={is_async}, progressive={progressive}); "
            "unset TORCHINDUCTOR_FX_COMPILE_MODE or drop the pass scope"
        )


# -- caches -----------------------------------------------------------------

_cache_lock = threading.Lock()
_cache_depth = 0
_cache_saved: List[Tuple[Any, str, Any]] = []
_cache_logged = False

_CACHE_KEYS: Final[Tuple[Tuple[str, Tuple[str, ...]], ...]] = (
    ("torch._inductor.config", ("fx_graph_cache", "fx_graph_remote_cache")),
    (
        "torch._functorch.config",
        ("enable_autograd_cache", "enable_remote_autograd_cache"),
    ),
)


@contextlib.contextmanager
def _caches_disabled() -> Iterator[None]:
    """Refcounted disable of the FX graph and AOTAutograd caches.

    Equivalent to a scoped ``config.patch`` (the first scope saves the values
    and sets them False, the last one to leave restores them), but not built on
    ``config.patch`` itself: its ``ConfigPatch`` keeps the saved values in a
    ``ContextVar`` and asserts if exited from a different thread/context than
    it was entered in, which overlapping scopes on several threads do.
    """
    global _cache_depth, _cache_logged
    with _cache_lock:
        if _cache_depth == 0:
            saved: List[Tuple[Any, str, Any]] = []
            try:
                for modname, keys in _CACHE_KEYS:
                    cfg = importlib.import_module(modname)
                    for k in keys:
                        if not hasattr(cfg, k):
                            logger.warning(
                                "%s.%s does not exist in this torch; cannot "
                                "disable it for the pass scope",
                                modname,
                                k,
                            )
                            continue
                        saved.append((cfg, k, getattr(cfg, k)))
                        setattr(cfg, k, False)
            except BaseException:
                for cfg, k, v in reversed(saved):
                    setattr(cfg, k, v)
                raise
            _cache_saved[:] = saved
            if not _cache_logged:
                _cache_logged = True
                logger.info(
                    "Spyre pre-scheduling passes are active: disabling the "
                    "Inductor FX graph cache and the AOTAutograd cache inside the "
                    "scope (a cache hit would skip the pass pipeline). Restored "
                    "on exit."
                )
        _cache_depth += 1
    try:
        yield
    finally:
        with _cache_lock:
            _cache_depth -= 1
            if _cache_depth == 0:
                for cfg, k, v in reversed(_cache_saved):
                    setattr(cfg, k, v)
                _cache_saved.clear()


# ---------------------------------------------------------------------------
# __init__ wrap
# ---------------------------------------------------------------------------


def _ensure_installed(target: _Target) -> None:
    """Wrap ``target.cls.__init__`` unless our wrapper is already the top one.

    If another patch has since replaced ``__init__`` (``mock.patch.object``
    around, or restored over, ours), the next scope entry layers a fresh
    wrapper on top. Layers cooperate: a pipeline that already holds the
    selected passes is left alone, one whose ``self.passes`` was reassigned
    after our layer ran is processed again.
    """
    cls = target.cls
    with _lock:
        current = cls.__dict__.get("__init__", cls.__init__)
        if getattr(current, _WRAPPER_MARK, None) is current:
            return
        wrapper = _make_wrapper(current, target)
        setattr(cls, "__init__", wrapper)


def _make_wrapper(orig: Callable[..., None], target: _Target):
    # updated=(): do not copy orig.__dict__, so the identity marker below is
    # never inherited from (or leaked to) another wrapper.
    @functools.wraps(orig, updated=())
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        orig(self, *args, **kwargs)
        scope = _ACTIVE.get()
        if scope is not None and scope.names:
            _apply(self, target, scope)

    setattr(__init__, _WRAPPER_MARK, __init__)
    return __init__


def _fail(target: _Target, msg: str) -> PreSchedulingPassError:
    return PreSchedulingPassError(f"{msg} [{target.describe()}]")


def _apply(self: Any, target: _Target, scope: _Scope) -> None:
    names = scope.effective_names()
    with _lock:
        selected = [r for r in _registry.values() if r.name in names]
        registered = list(_registry)
    missing = sorted(names - {r.name for r in selected})
    if missing:
        raise _fail(
            target,
            f"unknown pre-scheduling pass(es) {missing}; registered: {registered}",
        )

    passes = getattr(self, "passes", None)
    if not isinstance(passes, list):
        raise _fail(
            target,
            f"{type(self).__name__}.passes is {type(passes).__name__}, expected a "
            "list built in __init__; this torch-spyre is not supported",
        )
    bad = [i for i, p in enumerate(passes) if not callable(p)]
    if bad:
        raise _fail(
            target, f"{type(self).__name__}.passes entries {bad} are not callable"
        )

    # Another layer (or this one on a re-entered __init__) already did the job.
    if all(any(p is r.fn for p in passes) for r in selected):
        scope.hit()
        return

    base_names = [target.get_pass_name(p) for p in passes]
    selected_names = {r.name for r in selected}
    for r in selected:
        if r.name in base_names or r.name in target.reserved_names:
            raise _fail(
                target,
                f"pass name {r.name!r} collides with a torch-spyre pass name or a "
                "reserved name (_PASSES_RUN_EXPANDED)",
            )

    before: Dict[str, List[_Registered]] = {}
    after: Dict[str, List[_Registered]] = {}
    for r in selected:  # registration order, which also orders chained anchors
        if r.anchor in selected_names:
            earlier = [s.name for s in selected[: selected.index(r)]]
            if r.anchor not in earlier:
                raise _fail(
                    target,
                    f"pass {r.name!r} is anchored to {r.anchor!r}, which was "
                    "registered later; anchors must be registered earlier",
                )
        else:
            count = base_names.count(r.anchor)
            if count != 1:
                with _lock:
                    unselected = r.anchor in _registry
                hint = (
                    " (it is a registered pass that is not selected in this scope)"
                    if unselected
                    else ""
                )
                raise _fail(
                    target,
                    f"anchor {r.anchor!r} for pass {r.name!r} appears {count} times "
                    f"in {type(self).__name__}.passes (need exactly 1){hint}; "
                    f"available: {base_names}",
                )
        (after if r.position == "after" else before).setdefault(r.anchor, []).append(r)

    def render(fn: Callable[..., Any], name: str) -> List[Callable[..., Any]]:
        out: List[Callable[..., Any]] = []
        for r in before.get(name, ()):
            out += render(r.fn, r.name)
        out.append(fn)
        for r in after.get(name, ()):
            out += render(r.fn, r.name)
        return out

    rendered: List[Callable[..., Any]] = []
    for p, n in zip(passes, base_names):
        rendered += render(p, n)
    passes[:] = rendered
    scope.hit()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Spyre pre-scheduling pass list (%d): %s",
            len(passes),
            [target.get_pass_name(p) for p in passes],
        )
