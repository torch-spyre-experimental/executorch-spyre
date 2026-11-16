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
"""ET-side ``SpyreTensorLayout`` abstraction boundary (AOT / Python half).

This module is the single place in ExecuTorch-Spyre's Python code that knows how
to (a) serialize a torch-spyre ``SpyreTensorLayout`` into the payload's
``device_layout`` JSON object, (b) compare two such objects, and (c) rebuild a
spyre tensor that carries exactly such a layout.  The C++ mirror lives in
``runtime/SpyreLayout.h`` (pure schema/parse, host-testable) and
``runtime/SpyreLayoutBridge.{h,cpp}`` (device-bound ``SpyreTensorLayout``
construction).

Schema (``device_layout`` object; ``LAYOUT_FIELDS`` is the authoritative list,
and its order is the JSON key order emitted today -- do not reorder)::

    device_size          [int, ...]               SpyreTensorLayout.device_size
    stride_map           [int, ...]               SpyreTensorLayout.stride_map
    device_dtype         int                      DataFormats enum value
    element_arrangement  int                      ElementArrangement enum value
    valid_elements       [[[int, ...], int], ...] (device-dim group -> n valid)
    strides              [int, ...]               HOST strides of the tensor

``strides`` is not part of ``SpyreTensorLayout``; it is the host tensor's
strides, recorded because the runtime allocates with them.

Two different notions of "layout equality" exist -- do NOT conflate them
-----------------------------------------------------------------------
* ``layout_key`` / ``layouts_equal_for_audit`` (this module) is the identity
  used by the AOT convergence check (``spyre_backend._iterate_input_layouts``,
  which also keys its seed cache on it; the ``audit`` in the function name is
  historical).  It compares ``device_size``, ``stride_map``, ``device_dtype``,
  ``element_arrangement`` **and the host ``strides``**.  It does **not**
  compare ``valid_elements``: the default upload carries them and the
  compiler's output layouts do not, so including them would never converge.
  A non-empty ``valid_elements`` is only logged.
* ``spyre::SpyreTensorLayout::operator==`` (torch-spyre C++, also exposed as
  Python ``SpyreTensorLayout.__eq__``) compares only ``device_size``,
  ``stride_map``, ``device_dtype`` and ``element_arrangement``.  It ignores
  ``valid_elements`` and knows nothing about host strides.  The C++ delegate
  does not compare layouts at capture time (kv_state capture checks shape and
  dtype only), so nothing at runtime uses this one.

So the convergence key is *stricter* than ``operator==`` in the host strides
(it compares them, ``operator==`` does not) and equal to it on the other four
fields; neither looks at ``valid_elements``.  Key-equal implies
``operator==``-equal, not the other way round.  This divergence is known and
intentionally preserved; reconciling it is future work.

``torch`` / ``torch_spyre`` are imported lazily so the pure helpers can be
unit-tested on a host with neither a Spyre device nor torch-spyre installed.
"""

from typing import Final
import logging

logger = logging.getLogger(__name__)

# Device string registered by torch-spyre (PrivateUse1).
SPYRE_DEVICE: Final[str] = "spyre"

# Authoritative ``device_layout`` schema field names, in emitted order.
LAYOUT_FIELDS: Final[tuple] = (
    "device_size",
    "stride_map",
    "device_dtype",
    "element_arrangement",
    "valid_elements",
    "strides",
)


def layout_spec_from_stl(stl, strides) -> dict:
    """Serialize an ``stl`` (SpyreTensorLayout-like) + host ``strides`` to a spec.

    ``stl`` only needs the attributes ``device_size``, ``stride_map``,
    ``device_dtype``, ``element_arrangement`` and ``valid_elements`` (a
    ``{dims_tuple: n}`` mapping), so this is testable without torch-spyre.
    """
    return {
        "device_size": [int(x) for x in stl.device_size],
        "stride_map": [int(x) for x in stl.stride_map],
        "device_dtype": int(stl.device_dtype),
        "element_arrangement": int(stl.element_arrangement),
        "valid_elements": [
            [[int(d) for d in dims], int(n)] for dims, n in stl.valid_elements.items()
        ],
        "strides": [int(s) for s in strides],
    }


def device_layout_spec(t) -> "dict | None":
    """Serialize the *actual* device layout the compiled kernel wrote ``t`` with.

    The compiled Spyre kernel may produce a delegate output in a non-default
    (e.g. device dims permuted) ``SpyreTensorLayout``. The runtime must allocate the output
    with that same layout, otherwise the device->host copy-back (which interprets
    the buffer through the tensor's own layout) scrambles the data. Returns
    ``None`` for a non-Spyre tensor (runtime then keeps the default layout).
    """
    import torch

    if not isinstance(t, torch.Tensor) or t.device.type != SPYRE_DEVICE:
        return None
    from torch_spyre._C import get_spyre_tensor_layout

    return layout_spec_from_stl(get_spyre_tensor_layout(t), t.stride())


def layout_key(layout: "dict | None"):
    """Hashable convergence identity of a ``device_layout`` spec dict (None -> None).

    Compares device_size / stride_map / device_dtype / element_arrangement
    plus the host strides the runtime allocates with.  ``valid_elements`` is
    deliberately left out (a non-empty value is logged).  This is NOT
    ``SpyreTensorLayout::operator==``: that one ignores host strides (see the
    module docstring).
    """
    if layout is None:
        return None

    if layout.get("valid_elements"):
        logger.warning(
            "layout_key: ignoring valid_elements during layout convergence: %s",
            layout["valid_elements"],
        )

    return (
        tuple(layout["device_size"]),
        tuple(layout["stride_map"]),
        int(layout["device_dtype"]),
        int(layout["element_arrangement"]),
        #        tuple(
        #            sorted((tuple(dims), int(n)) for dims, n in layout["valid_elements"])
        #        ),
        tuple(layout.get("strides") or ()),
    )


def layouts_equal_for_audit(a: "dict | None", b: "dict | None") -> bool:
    """Convergence-check equality: ``layout_key(a) == layout_key(b)``.

    Stricter than ``SpyreTensorLayout::operator==`` (see the module docstring).
    """
    return layout_key(a) == layout_key(b)


def resolve_host_strides(layout: dict, shape) -> "list[int]":
    """Host strides to allocate with: the layout's when its rank matches ``shape``,
    else contiguous row-major (same rule as C++ ``resolve_host_strides``)."""
    shape = [int(s) for s in shape]
    strides = [int(s) for s in (layout.get("strides") or [])]
    if len(strides) != len(shape):
        strides = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * max(shape[d + 1], 1)
    return strides


def spyre_tensor_with_layout(layout: dict, shape, dtype):
    """Allocate an (uninitialised) spyre tensor carrying exactly ``layout``.

    Same construction the C++ delegate uses (``spyre_empty_from_payload_layout``):
    host strides come from the layout when their rank matches, else row-major.
    """
    import torch
    from torch_spyre._C import (
        DataFormats,
        ElementArrangement,
        SpyreTensorLayout,
        spyre_empty_with_layout,
    )

    stl = SpyreTensorLayout(
        list(layout["device_size"]),
        list(layout["stride_map"]),
        DataFormats(int(layout["device_dtype"])),
        {tuple(dims): int(n) for dims, n in layout["valid_elements"]},
        ElementArrangement(int(layout["element_arrangement"])),
    )
    shape = [int(s) for s in shape]
    strides = resolve_host_strides(layout, shape)
    return spyre_empty_with_layout(
        tuple(shape), tuple(strides), dtype, stl, torch.device(SPYRE_DEVICE)
    )
