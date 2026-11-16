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
// Phase 3 exit-criterion harness: load the Spyre-delegated MLP .pte
// (Linear(64,128) -> ReLU -> Linear(128,64), fp16), feed ONLY the user input
// x, run it on the Spyre device (via the linked-in SpyreBackend), and assert
// the output matches the baked CPU fp16 reference within the Phase 2 envelope.
//
// The subgraph's lifted params (fc1/fc2 weight+bias) are stored inside the
// .pte and supplied to the delegate's execute() as EValues by ExecuTorch, in
// placeholder order (verified AOT-side: x sits at delegate-input index 3, the
// four params at 0/1/2/4). So at the *method* level we bind a single input --
// x -- and ET routes the stored consts into the remaining delegate positions.
//
//   ./spyre_mlp_runner <artifacts-dir>
//
// Requires FLEX_DEVICE=PF at runtime (unset -> MOCK stub -> garbage compute).

#include <c10/util/Half.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

namespace {

std::vector<uint16_t> read_fp16(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamsize n = ifs.tellg();
  ifs.seekg(0);
  std::vector<uint16_t> buf(static_cast<size_t>(n) / sizeof(uint16_t));
  ifs.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

float half_to_float(uint16_t bits) {
  return static_cast<float>(c10::Half(bits, c10::Half::from_bits()));
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = (argc > 1) ? argv[1] : "artifacts";
  // Shapes fixed by export_mlp.py defaults: x[1,64] -> y[1,64].
  const int64_t B = 1, IN = 64, OUT = 64;

  auto x = read_fp16(dir + "/x.bin");
  auto ref = read_fp16(dir + "/ref.bin");

  TensorPtr tx = from_blob(x.data(), {B, IN}, ScalarType::Half);

  Module module(dir + "/spyre_mlp.pte");
  const auto result = module.forward({EValue(*tx)});
  if (result.error() != Error::Ok) {
    fprintf(stderr, "forward() failed: 0x%x\n",
            static_cast<unsigned>(result.error()));
    return 1;
  }

  const Tensor out = result->at(0).toTensor();
  const auto* out_data = out.const_data_ptr<uint16_t>();
  const size_t numel = static_cast<size_t>(B * OUT);
  if (out.numel() != static_cast<ssize_t>(numel) || ref.size() != numel) {
    fprintf(stderr, "size mismatch: out=%zd ref=%zu expected=%zu\n",
            out.numel(), ref.size(), numel);
    return 1;
  }

  // Per-element allclose: |got - exp| <= atol + rtol*|exp|. Same envelope as
  // the Phase 2 matmul harness (atol=1e-1, rtol=2e-2). The MLP's two matmuls
  // reduce over K=64 then K=128 in reduced precision while the CPU reference
  // accumulates in fp32; the ReLU is exact, so the residual is again dominated
  // by fp16 matmul accumulation.
  const double kAtol = 1e-1, kRtol = 2e-2;
  double max_abs = 0.0, worst_got = 0.0, worst_exp = 0.0;
  size_t worst_i = 0, n_fail = 0;
  for (size_t i = 0; i < numel; ++i) {
    const float got = half_to_float(out_data[i]);
    const float exp = half_to_float(ref[i]);
    const double abs_err = std::fabs(got - exp);
    if (abs_err > max_abs) {
      max_abs = abs_err;
      worst_got = got;
      worst_exp = exp;
      worst_i = i;
    }
    if (abs_err > kAtol + kRtol * std::fabs(exp)) {
      ++n_fail;
    }
  }
  const bool pass = (n_fail == 0);

  printf("out[0,:4] = %.4f %.4f %.4f %.4f\n", half_to_float(out_data[0]),
         half_to_float(out_data[1]), half_to_float(out_data[2]),
         half_to_float(out_data[3]));
  printf("ref[0,:4] = %.4f %.4f %.4f %.4f\n", half_to_float(ref[0]),
         half_to_float(ref[1]), half_to_float(ref[2]), half_to_float(ref[3]));
  printf("max_abs_err = %.6f at [%zu] (got %.4f vs ref %.4f)\n", max_abs,
         worst_i, worst_got, worst_exp);
  printf("elements outside |got-exp| <= %.g + %.g*|exp|: %zu / %zu\n", kAtol,
         kRtol, n_fail, numel);
  printf("%s\n", pass ? "PASS: Spyre MLP delegate matches CPU fp16 reference"
                      : "FAIL: output diverges from reference");
  return pass ? 0 : 1;
}
