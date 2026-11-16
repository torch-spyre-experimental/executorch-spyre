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
// Phase 5 device gate: a GENERIC, manifest-driven runner for the op-coverage
// harness (coverage/run_coverage.py). Unlike spyre_runner / spyre_mlp_runner,
// which hardcode one op's shapes and dtypes, this reads a flat manifest
// describing an arbitrary single-op delegated .pte, binds all of its method
// inputs, runs it on the Spyre device (via the whole-archive-linked
// SpyreBackend), and compares the output against a CPU fp16 reference produced
// AOT-side by the harness.
//
//   ./spyre_coverage_runner <artifacts-dir>   (default "artifacts")
//
// The artifacts dir must contain a manifest.txt written by run_coverage's
// _gate_device. Line grammar (whitespace-separated, one record per line, '#'
// comments and blank lines ignored):
//
//   pte    <file>
//   case   <name>
//   input  <file> <dtype> <ndim> <dim0> <dim1> ...
//   output <file> <dtype> <ndim> <dim0> <dim1> ...
//
// dtype is one of: fp16 fp32 int64 bool. Inputs are bound to the method in the
// order listed (which is the exported forward-arg / placeholder order). A
// single output line names the CPU reference; the on-device result at index 0
// is diffed against it.
//
// Phase 7 Track 2 (bucketing): an optional `case <name>` record starts a new
// evaluation group. Each case carries its own input(s) + output; the shared
// `.pte` is forwarded once PER case, so one bucketed module is exercised across
// several input shapes (each must select its own shape bucket and match CPU). A
// manifest with no `case` records is one implicit case -- byte-identical to the
// Phase 5 single-shape behavior.
//
// Contract on stdout (parsed by the harness):
//   RUN: pass      -- every case's module.forward() returned Error::Ok
//   RUN: fail      -- some case's forward() errored (its MATCH is then skip)
//   MATCH: pass    -- every case's output within tolerance of its reference
//   MATCH: fail    -- some case's output diverged / size mismatch
//   MATCH: skip    -- not evaluated (its forward failed)
// With >1 case, per-case `RUN[<name>]: ...` / `MATCH[<name>]: ...` lines are
// also emitted before the aggregate line; the aggregate passes iff every case
// passes.
//
// Exit codes mirror the sibling runners: 0 = all cases RUN pass & MATCH pass; 1
// = any forward error, size mismatch, or MATCH fail; 2 = a file could not be
// opened / parsed.
//
// Float outputs use the Phase 2 envelope |got-exp| <= atol + rtol*|exp| with
// atol=1e-1, rtol=2e-2 (fp16 matmul accumulates in reduced precision while the
// reference accumulates in fp32). int64 / bool outputs are compared exactly.
//
// Requires FLEX_DEVICE=PF at runtime (unset -> MOCK stub -> garbage compute);
// the harness only invokes this runner when FLEX_DEVICE is set.

#include <c10/util/Half.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
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

// Tolerances shared with the Phase 2 matmul / Phase 3 MLP harnesses.
constexpr double kAtol = 1e-1, kRtol = 2e-2;

enum class DType { Half, Float, Long, Bool, BFloat16 };

struct TensorSpec {
  std::string file;
  DType dtype;
  std::vector<int64_t> dims;
};

// One evaluation group: the input(s) bound for a single forward() and the CPU
// reference its output is diffed against. A non-bucketed manifest has exactly
// one (unnamed) case; a v3/bucketed manifest has one per input shape.
struct Case {
  std::string name;
  std::vector<TensorSpec> inputs;
  TensorSpec output;
  bool have_output = false;
};

const char* dtype_name(DType d) {
  switch (d) {
    case DType::Half:
      return "fp16";
    case DType::Float:
      return "fp32";
    case DType::Long:
      return "int64";
    case DType::Bool:
      return "bool";
    case DType::BFloat16:
      return "bf16";
  }
  return "?";
}

bool parse_dtype(const std::string& s, DType* out) {
  if (s == "fp16") {
    *out = DType::Half;
    return true;
  }
  if (s == "fp32") {
    *out = DType::Float;
    return true;
  }
  if (s == "int64") {
    *out = DType::Long;
    return true;
  }
  if (s == "bool") {
    *out = DType::Bool;
    return true;
  }
  if (s == "bf16") {
    *out = DType::BFloat16;
    return true;
  }
  return false;
}

ScalarType to_scalar_type(DType d) {
  switch (d) {
    case DType::Half:
      return ScalarType::Half;
    case DType::Float:
      return ScalarType::Float;
    case DType::Long:
      return ScalarType::Long;
    case DType::Bool:
      return ScalarType::Bool;
    case DType::BFloat16:
      return ScalarType::BFloat16;
  }
  return ScalarType::Half;
}

size_t elem_size(DType d) {
  switch (d) {
    case DType::Half:
      return 2;
    case DType::Float:
      return 4;
    case DType::Long:
      return 8;
    case DType::Bool:
      return 1;
    case DType::BFloat16:
      return 2;
  }
  return 0;
}

int64_t numel(const std::vector<int64_t>& dims) {
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

float half_to_float(uint16_t bits) {
  return static_cast<float>(c10::Half(bits, c10::Half::from_bits()));
}

// Read a whole binary file into a byte buffer, or exit(2) on open failure.
std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) {
    fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamsize n = ifs.tellg();
  ifs.seekg(0);
  std::vector<uint8_t> buf(static_cast<size_t>(n));
  ifs.read(reinterpret_cast<char*>(buf.data()), n);
  return buf;
}

// Parse a `<file> <dtype> <ndim> <dims...>` tail into a TensorSpec.
bool parse_tensor_spec(std::istringstream& iss, TensorSpec* spec) {
  std::string dtype_s;
  int ndim = -1;
  if (!(iss >> spec->file >> dtype_s >> ndim)) return false;
  if (!parse_dtype(dtype_s, &spec->dtype)) {
    fprintf(stderr, "unknown dtype '%s'\n", dtype_s.c_str());
    return false;
  }
  if (ndim < 0) return false;
  spec->dims.clear();
  for (int i = 0; i < ndim; ++i) {
    int64_t d;
    if (!(iss >> d)) return false;
    spec->dims.push_back(d);
  }
  return true;
}

// Compare on-device output bytes against the reference bytes for `spec`.
// Returns true if within tolerance (float) or exactly equal (int/bool).
bool compare(const TensorSpec& spec, const Tensor& out,
             const std::vector<uint8_t>& ref) {
  const int64_t n = numel(spec.dims);
  // The CPU reference must match the manifest's declared shape exactly.
  if (ref.size() != static_cast<size_t>(n) * elem_size(spec.dtype)) {
    fprintf(stderr, "ref size mismatch: ref_bytes=%zu expected numel=%lld\n",
            ref.size(), static_cast<long long>(n));
    return false;
  }
  // The device output must hold AT LEAST the logical numel. For a bucketed
  // (v3) .pte, ExecuTorch statically plans the delegate output at the largest
  // bucket's shape, so a smaller bucket's run returns an over-allocated buffer
  // with the true [b,...] result in the leading contiguous elements (batch is
  // the outermost dim). We compare that logical prefix -- exactly the slice a
  // real consumer takes from the input shape. A short buffer is a real error.
  if (out.numel() < static_cast<ssize_t>(n)) {
    fprintf(stderr, "size mismatch: out=%zd < expected numel=%lld\n",
            out.numel(), static_cast<long long>(n));
    return false;
  }
  if (out.numel() > static_cast<ssize_t>(n))
    printf(
        "note: comparing first %lld of %zd elems (static upper-bound output)\n",
        static_cast<long long>(n), out.numel());

  size_t n_fail = 0;
  double max_abs = 0.0;
  switch (spec.dtype) {
    case DType::BFloat16: {
      const auto* got = out.const_data_ptr<uint16_t>();
      const auto* exp = reinterpret_cast<const uint16_t*>(ref.data());
      for (int64_t i = 0; i < n; ++i) {
        const double g = static_cast<float>(
            c10::BFloat16(got[i], c10::BFloat16::from_bits()));
        const double e = static_cast<float>(
            c10::BFloat16(exp[i], c10::BFloat16::from_bits()));
        const double err = std::fabs(g - e);
        max_abs = std::max(max_abs, err);
        if (err > kAtol + kRtol * std::fabs(e)) ++n_fail;
      }
      break;
    }
    case DType::Half: {
      const auto* got = out.const_data_ptr<uint16_t>();
      const auto* exp = reinterpret_cast<const uint16_t*>(ref.data());
      for (int64_t i = 0; i < n; ++i) {
        const double g = half_to_float(got[i]), e = half_to_float(exp[i]);
        const double err = std::fabs(g - e);
        max_abs = std::max(max_abs, err);
        if (err > kAtol + kRtol * std::fabs(e)) ++n_fail;
      }
      break;
    }
    case DType::Float: {
      const auto* got = out.const_data_ptr<float>();
      const auto* exp = reinterpret_cast<const float*>(ref.data());
      for (int64_t i = 0; i < n; ++i) {
        const double err = std::fabs(static_cast<double>(got[i]) - exp[i]);
        max_abs = std::max(max_abs, err);
        if (err > kAtol + kRtol * std::fabs(static_cast<double>(exp[i])))
          ++n_fail;
      }
      break;
    }
    case DType::Long: {
      const auto* got = out.const_data_ptr<int64_t>();
      const auto* exp = reinterpret_cast<const int64_t*>(ref.data());
      for (int64_t i = 0; i < n; ++i)
        if (got[i] != exp[i]) ++n_fail;
      break;
    }
    case DType::Bool: {
      const auto* got = out.const_data_ptr<bool>();
      const auto* exp = reinterpret_cast<const bool*>(ref.data());
      for (int64_t i = 0; i < n; ++i)
        if (got[i] != exp[i]) ++n_fail;
      break;
    }
  }

  if (spec.dtype == DType::Half || spec.dtype == DType::Float ||
      spec.dtype == DType::BFloat16)
    printf("max_abs_err = %.6f (dtype %s, numel %lld)\n", max_abs,
           dtype_name(spec.dtype), static_cast<long long>(n));
  printf("elements outside tolerance: %zu / %lld\n", n_fail,
         static_cast<long long>(n));
  return n_fail == 0;
}

// Bind one case's inputs, forward the (already-loaded) module, and compare the
// output against the case's reference. `suffix` labels the per-case stdout
// lines
// ("" for the sole case). Sets *ran/*matched. exit(2) on a file/size error,
// since that is a harness/manifest bug rather than a device failure.
void run_case(Module& module, const std::string& dir, const Case& c,
              const std::string& suffix, bool* ran, bool* matched) {
  *ran = false;
  *matched = false;

  // Byte buffers must outlive the TensorPtrs, so keep them (and the TensorPtrs)
  // alive through forward().
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  buffers.reserve(c.inputs.size());
  tensors.reserve(c.inputs.size());
  evalues.reserve(c.inputs.size());
  for (const auto& in : c.inputs) {
    buffers.push_back(read_bytes(dir + "/" + in.file));
    const auto& buf = buffers.back();
    const size_t want =
        static_cast<size_t>(numel(in.dims)) * elem_size(in.dtype);
    if (buf.size() != want) {
      fprintf(stderr, "input %s: %zu bytes, expected %zu\n", in.file.c_str(),
              buf.size(), want);
      std::exit(2);
    }
    // from_blob takes SizesType (int32) sizes; our manifest dims are int64.
    std::vector<executorch::aten::SizesType> sizes(in.dims.begin(),
                                                   in.dims.end());
    tensors.push_back(from_blob(const_cast<uint8_t*>(buf.data()), sizes,
                                to_scalar_type(in.dtype)));
    evalues.emplace_back(*tensors.back());
  }

  const auto result = module.forward(evalues);
  if (result.error() != Error::Ok) {
    fprintf(stderr, "forward() failed: 0x%x\n",
            static_cast<unsigned>(result.error()));
    printf("RUN%s: fail\n", suffix.c_str());
    printf("MATCH%s: skip\n", suffix.c_str());
    return;
  }
  *ran = true;
  printf("RUN%s: pass\n", suffix.c_str());

  const Tensor out = result->at(0).toTensor();
  if (std::getenv("EXECUTORCH_SPYRE_DUMP_OUT")) {  // raw device outputs, for
                                                   // offline layout diagnosis
    for (size_t oi = 0; oi < result->size(); ++oi) {
      if (!result->at(oi).isTensor()) continue;
      const Tensor t = result->at(oi).toTensor();
      std::ofstream of(
          dir + "/out" + suffix + (oi ? "_" + std::to_string(oi) : "") + ".bin",
          std::ios::binary);
      of.write(static_cast<const char*>(t.const_data_ptr()),
               static_cast<std::streamsize>(t.nbytes()));
    }
  }
  const std::vector<uint8_t> ref = read_bytes(dir + "/" + c.output.file);
  *matched = compare(c.output, out, ref);
  printf("MATCH%s: %s\n", suffix.c_str(), *matched ? "pass" : "fail");
}

// Phase 9 perf loop (opt-in via EXECUTORCH_SPYRE_PERF_ITERS=N>0; default 0
// keeps this runner byte-identical to the Phase 5 gate). Times `iters+1`
// forwards of one case on a FRESH Module so the first (cold) forward pays this
// partition-set's lazy init + prepareKernel of every partition; the next
// `iters` are steady-state (warm). The embedded-CPython one-time cost is
// already paid by the verification Module above (std::call_once), so it appears
// once in the [SPYRE_PROFILE] cpython_init_ms line, not here. Set
// EXECUTORCH_SPYRE_PROFILE=1 alongside to get the per-phase decomposition of
// each forward. exit(1) on a forward error (the verification pass already ran).
void run_perf(const std::string& dir, const std::string& pte, const Case& c,
              int iters) {
  Module module(dir + "/" + pte);

  // Bind the case's inputs once and reuse them across every timed forward.
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<TensorPtr> tensors;
  std::vector<EValue> evalues;
  buffers.reserve(c.inputs.size());
  tensors.reserve(c.inputs.size());
  evalues.reserve(c.inputs.size());
  for (const auto& in : c.inputs) {
    buffers.push_back(read_bytes(dir + "/" + in.file));
    const auto& buf = buffers.back();
    std::vector<executorch::aten::SizesType> sizes(in.dims.begin(),
                                                   in.dims.end());
    tensors.push_back(from_blob(const_cast<uint8_t*>(buf.data()), sizes,
                                to_scalar_type(in.dtype)));
    evalues.emplace_back(*tensors.back());
  }

  using Clock = std::chrono::steady_clock;
  auto timed_forward = [&]() -> double {
    const auto t0 = Clock::now();
    const auto r = module.forward(evalues);
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    if (r.error() != Error::Ok) {
      fprintf(stderr, "perf forward() failed: 0x%x\n",
              static_cast<unsigned>(r.error()));
      std::exit(1);
    }
    return ms;
  };

  // First forward includes lazy init + prepareKernel of every partition.
  const double cold_ms = timed_forward();
  std::vector<double> warm;
  warm.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) warm.push_back(timed_forward());

  std::sort(warm.begin(), warm.end());
  const double wmin = warm.front();
  const double wmax = warm.back();
  const double wmedian = warm[warm.size() / 2];
  double wsum = 0.0;
  for (double v : warm) wsum += v;
  const double wmean = wsum / static_cast<double>(warm.size());

  printf(
      "PERF: cold_ms=%.3f warm_min_ms=%.3f warm_median_ms=%.3f "
      "warm_mean_ms=%.3f warm_max_ms=%.3f iters=%d case=%s\n",
      cold_ms, wmin, wmedian, wmean, wmax, iters,
      c.name.empty() ? "-" : c.name.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = (argc > 1) ? argv[1] : "artifacts";
  const std::string manifest_path = dir + "/manifest.txt";

  std::ifstream mf(manifest_path);
  if (!mf) {
    fprintf(stderr, "cannot open %s\n", manifest_path.c_str());
    return 2;
  }

  std::string pte;
  std::vector<Case> cases;
  // The case currently being filled. `input`/`output` before any `case` record
  // open an implicit first case, preserving the Phase 5 single-shape grammar.
  auto current = [&]() -> Case& {
    if (cases.empty()) cases.emplace_back();
    return cases.back();
  };

  std::string line;
  while (std::getline(mf, line)) {
    // Strip comments and skip blanks.
    const size_t hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);
    std::istringstream iss(line);
    std::string kind;
    if (!(iss >> kind)) continue;
    if (kind == "pte") {
      iss >> pte;
    } else if (kind == "case") {
      Case c;
      iss >> c.name;  // optional label; blank is fine
      cases.push_back(std::move(c));
    } else if (kind == "input") {
      TensorSpec s;
      if (!parse_tensor_spec(iss, &s)) {
        fprintf(stderr, "bad input line: %s\n", line.c_str());
        return 2;
      }
      current().inputs.push_back(std::move(s));
    } else if (kind == "output") {
      Case& c = current();
      if (!parse_tensor_spec(iss, &c.output)) {
        fprintf(stderr, "bad output line: %s\n", line.c_str());
        return 2;
      }
      c.have_output = true;
    } else {
      fprintf(stderr, "unknown manifest record '%s'\n", kind.c_str());
      return 2;
    }
  }

  if (pte.empty() || cases.empty()) {
    fprintf(stderr, "manifest missing pte/input/output records\n");
    return 2;
  }
  for (size_t i = 0; i < cases.size(); ++i) {
    if (cases[i].inputs.empty() || !cases[i].have_output) {
      fprintf(stderr, "case %zu missing input/output records\n", i);
      return 2;
    }
  }

  // One Module (one init -> one SpyreHandle, possibly multi-bucket) forwarded
  // once per case, so a bucketed .pte is exercised across every input shape.
  Module module(dir + "/" + pte);
  const bool multi = cases.size() > 1;
  bool all_ran = true, all_matched = true;
  for (const auto& c : cases) {
    std::string suffix;
    if (multi)
      suffix = "[" + (c.name.empty() ? std::string("?") : c.name) + "]";
    bool ran = false, matched = false;
    run_case(module, dir, c, suffix, &ran, &matched);
    all_ran = all_ran && ran;
    all_matched = all_matched && matched;
  }

  if (multi) {
    // Aggregate lines for the harness's grep: pass iff every case passed.
    printf("RUN: %s\n", all_ran ? "pass" : "fail");
    printf("MATCH: %s\n", all_ran ? (all_matched ? "pass" : "fail") : "skip");
  }

  // Phase 9 (opt-in): after the verification pass, time cold vs steady-state
  // forwards of the first case. Default (unset / 0) leaves the exit contract
  // and stdout of the Phase 5 gate untouched. This never changes the exit code.
  const char* perf_env = std::getenv("EXECUTORCH_SPYRE_PERF_ITERS");
  const int perf_iters = perf_env ? std::atoi(perf_env) : 0;
  if (perf_iters > 0) {
    run_perf(dir, pte, cases.front(), perf_iters);
  }

  return (all_ran && all_matched) ? 0 : 1;
}
