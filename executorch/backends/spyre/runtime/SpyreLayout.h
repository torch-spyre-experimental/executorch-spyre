// Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ET-side SpyreTensorLayout abstraction boundary, pure (schema) half.
//
// This header is the single C++ definition of the payload ``device_layout``
// schema: the ``OutputLayout`` POD, its field names, the JSON parser, a JSON
// serializer (used by host tests to prove the schema round-trips), and the
// host-stride defaulting rule. It is deliberately free of ExecuTorch,
// torch-spyre, ATen and CPython so it can be unit tested off-device (see
// test/SpyreLayoutTest.cpp). The device-bound half -- building a real
// ``::spyre::SpyreTensorLayout`` / device tensor from an ``OutputLayout`` --
// lives in SpyreLayoutBridge.{h,cpp}. The Python mirror is ``spyre_layout.py``.
//
// Schema (``kLayoutFields`` is authoritative; order == Python emit order):
//
//   device_size          [int, ...] SpyreTensorLayout::device_size stride_map
//   [int, ...]               SpyreTensorLayout::stride_map device_dtype int
//   DataFormats enum value element_arrangement  int ElementArrangement enum
//   value valid_elements       [[[int, ...], int], ...] (device-dim group -> n
//   valid) strides              [int, ...]               HOST strides of the
//   tensor
//
// ``strides`` is not part of SpyreTensorLayout; it is the host tensor's
// strides.
//
// NOTE on equality: the AOT layout audit (spyre_layout.py::layout_key) compares
// all six fields above. ``::spyre::SpyreTensorLayout::operator==`` compares
// only device_size, stride_map, device_dtype and element_arrangement; it
// ignores valid_elements and knows nothing about host strides. They are
// DIFFERENT predicates. The runtime does not compare layouts at capture time
// (the kv_state capture checks shape and dtype only), and this header
// intentionally provides no equality helper so that the two are never silently
// conflated.

#ifndef EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUT_H_
#define EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>

namespace executorch {
namespace backends {
namespace spyre {

// Authoritative ``device_layout`` field names, in Python emit order. The Python
// test suite parses this array out of this header and compares it against
// spyre_layout.LAYOUT_FIELDS, so the two languages cannot drift silently.
// BEGIN_LAYOUT_FIELDS
inline constexpr const char* kLayoutFields[] = {
    "device_size",         "stride_map",     "device_dtype",
    "element_arrangement", "valid_elements", "strides",
};
// END_LAYOUT_FIELDS

// Device-side layout of one delegate output, as recorded at AOT from the tensor
// the compiled kernel actually produced (``SpyreTensorLayout`` + host strides).
// The kernel may write an output in a non-default (tile-major) device layout;
// the runtime must allocate the output with the SAME layout so the device->host
// copy-back interprets the buffer correctly. Plain ints/vectors keep this
// header dependency-free: ``device_dtype`` / ``element_arrangement`` are the
// integer values of torch-spyre's DataFormats / ElementArrangement enums, and
// ``valid_elements`` is a list of (device-dim group -> valid element count).
// ``present`` is false when the payload carries no layout (older payloads / CPU
// outputs); the runtime then keeps its default-layout allocation.
struct OutputLayout {
  bool present = false;
  std::vector<std::int64_t> device_size;
  std::vector<std::int64_t> stride_map;
  std::int64_t device_dtype = 0;
  std::int64_t element_arrangement = 0;
  std::vector<std::pair<std::vector<std::int64_t>, std::int64_t>>
      valid_elements;
  std::vector<std::int64_t> strides;  // host strides of the output tensor
};

// Device-side layout of one delegate input. Same wire schema and fields as an
// output's ``device_layout``; currently only emitted for ``kv_state`` inputs
// (header ``"inputs"[i]["device_layout"]``). ``present`` is false when the
// payload carries no layout for that input (non-state inputs, older payloads),
// in which case the runtime keeps its default-layout allocation.
using InputLayout = OutputLayout;

inline std::vector<std::int64_t> parse_i64_array(const nlohmann::json& arr) {
  std::vector<std::int64_t> v;
  for (const auto& s : arr) v.push_back(s.get<std::int64_t>());
  return v;
}

// Parse one ``device_layout`` JSON object (shared by outputs and inputs).
inline OutputLayout parse_device_layout(const nlohmann::json& dl) {
  OutputLayout ol;
  ol.present = true;
  ol.device_size = parse_i64_array(dl.at("device_size"));
  ol.stride_map = parse_i64_array(dl.at("stride_map"));
  ol.device_dtype = dl.value("device_dtype", static_cast<std::int64_t>(0));
  ol.element_arrangement =
      dl.value("element_arrangement", static_cast<std::int64_t>(0));
  if (dl.contains("valid_elements")) {
    for (const auto& ve : dl["valid_elements"]) {
      ol.valid_elements.emplace_back(parse_i64_array(ve.at(0)),
                                     ve.at(1).get<std::int64_t>());
    }
  }
  if (dl.contains("strides")) ol.strides = parse_i64_array(dl["strides"]);
  return ol;
}

// Inverse of parse_device_layout. The runtime never emits payloads (the AOT
// Python side does); this exists so host tests can prove the schema round-trips
// (JSON -> OutputLayout -> JSON). All six fields are always written.
inline nlohmann::json layout_to_json(const OutputLayout& ol) {
  nlohmann::json ve = nlohmann::json::array();
  for (const auto& e : ol.valid_elements) {
    ve.push_back(nlohmann::json::array({e.first, e.second}));
  }
  nlohmann::json j;
  j["device_size"] = ol.device_size;
  j["stride_map"] = ol.stride_map;
  j["device_dtype"] = ol.device_dtype;
  j["element_arrangement"] = ol.element_arrangement;
  j["valid_elements"] = std::move(ve);
  j["strides"] = ol.strides;
  return j;
}

// Host strides to allocate with: the payload's when their rank matches
// ``sizes``, else contiguous row-major. Mirrors spyre_layout.py
// ``resolve_host_strides``.
inline std::vector<std::int64_t> resolve_host_strides(
    const OutputLayout& ol, const std::vector<std::int64_t>& sizes) {
  std::vector<std::int64_t> strides = ol.strides;
  if (strides.size() != sizes.size()) {
    strides.assign(sizes.size(), 1);
    for (std::int64_t d = static_cast<std::int64_t>(sizes.size()) - 2; d >= 0;
         --d) {
      strides[d] = strides[d + 1] * std::max<std::int64_t>(sizes[d + 1], 1);
    }
  }
  return strides;
}

}  // namespace spyre
}  // namespace backends
}  // namespace executorch

#endif  // EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUT_H_
