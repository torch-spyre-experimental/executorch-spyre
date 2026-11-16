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

#include "SpyreLayoutBridge.h"

#include <map>
#include <optional>
#include <utility>

// spyre_mem.h pulls module.h (pybind11); forward-declare the one symbol we need
// to keep pybind out of this TU. Signature confirmed against
// torch_spyre/csrc/spyre_mem.h (same declaration SpyreBackend.cpp carries for
// its pool allocation).
namespace spyre {
at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout,
                                   std::optional<c10::Device> device_opt);
}  // namespace spyre

namespace executorch {
namespace backends {
namespace spyre {

::spyre::SpyreTensorLayout stl_from_payload_layout(const OutputLayout& ol) {
  std::map<std::vector<int64_t>, int64_t> valid;
  for (const auto& ve : ol.valid_elements) valid[ve.first] = ve.second;
  return ::spyre::SpyreTensorLayout(
      ol.device_size, ol.stride_map, static_cast<DataFormats>(ol.device_dtype),
      std::move(valid),
      static_cast<::spyre::ElementArrangement>(ol.element_arrangement));
}

at::Tensor spyre_empty_from_payload_layout(
    const OutputLayout& ol, const std::vector<std::int64_t>& sizes,
    c10::ScalarType dtype, c10::Device dev) {
  const std::vector<int64_t> strides = resolve_host_strides(ol, sizes);
  return ::spyre::spyre_empty_with_layout(sizes, strides, dtype,
                                          stl_from_payload_layout(ol), dev);
}

}  // namespace spyre
}  // namespace backends
}  // namespace executorch
