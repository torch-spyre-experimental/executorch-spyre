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
// ET-side SpyreTensorLayout abstraction boundary, device-bound half. Turns the
// pure schema struct (SpyreLayout.h::OutputLayout) into torch-spyre objects.
// These are the ONLY places in the runtime that construct a SpyreTensorLayout
// from a payload ``device_layout``. Python mirror: spyre_layout.py.

#ifndef EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUTBRIDGE_H_
#define EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUTBRIDGE_H_

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <vector>

#include "SpyreLayout.h"
// ::spyre::SpyreTensorLayout + ::spyre::ElementArrangement (pybind-free
// header).
#include "spyre_tensor_impl.h"

namespace executorch {
namespace backends {
namespace spyre {

// Build the SpyreTensorLayout described by a payload ``device_layout``.
::spyre::SpyreTensorLayout stl_from_payload_layout(const OutputLayout& ol);

// Allocate a Spyre device tensor whose device layout is the payload layout.
// Host strides come from the payload when their rank matches, else row-major
// (see resolve_host_strides in SpyreLayout.h).
at::Tensor spyre_empty_from_payload_layout(
    const OutputLayout& ol, const std::vector<std::int64_t>& sizes,
    c10::ScalarType dtype, c10::Device dev);

}  // namespace spyre
}  // namespace backends
}  // namespace executorch

#endif  // EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYRELAYOUTBRIDGE_H_
