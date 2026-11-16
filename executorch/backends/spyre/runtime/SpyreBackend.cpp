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
// Runtime side of the Spyre ExecuTorch delegate.
//
//   init()    : parse the AOT payload header, extract the tarred SpyreCode
//               dir(s), start the Spyre runtime (once), prepareKernel() each
//               into a JobPlan, stash the vector in an SpyreHandle in the ET
//               runtime arena (Phase 6d: N JobPlans prepared at init).
//   execute() : bridge ET portable tensors <-> Spyre (PrivateUse1) tensors,
//               allocate the shared scratch pool once, launch each prepared
//               JobPlan in wire order (threading resident intermediates
//               producer->consumer on-device), synchronize, copy device outputs
//               back into ET output tensors. A non-resident inter-kernel edge
//               is rejected (NotImplemented) — see classify_launch in
//               SpyrePayload.h.
//   destroy() : run the JobPlans' destructors and remove the temp dir.
//
// Payload wire format (produced by spyre_backend.py::preprocess):
//   [u32 header_len, little-endian] [header_json utf-8] [tar bytes]
// The tar (what spyre_backend.py emits for static compiles) carries one
// "spyreCodeDir_{i}/" per kernel (spyrecode.json + init_binary.bin), under a
// "bucket{b}/" prefix for a bucketed payload. Each is fed to prepareKernel().
// v1 payloads (one top-level "spyreCodeDir/") are rejected at init. See
// SpyrePayload.h for the schema.
//
// Linking (see CMakeLists.txt): this TU is compiled into a shared lib that is
// whole-archive-linked so the static register_backend() global ctor below is
// not stripped. It links torch-spyre's _C.so (exports ::spyre::prepareKernel /
// launchJobPlan / startRuntime with default visibility) + ${TORCH_LIBRARIES} +
// executorch_core + the Spyre runtime stack.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Host-only payload parser (the [u32 len][json][tar] wire format). Kept in its
// own dependency-free header so it can be unit tested off-device; see
// test/SpyrePayloadTest.cpp. Pulls nlohmann/json.hpp.
#include "SpyrePayload.h"
// ET-side SpyreTensorLayout boundary (payload device_layout -> torch-spyre).
#include "SpyreLayoutBridge.h"

// torch-spyre C++ headers. These pull flex/flex.hpp, job_plan.h,
// spyre_stream.h and nlohmann/json.hpp, but crucially NOT module.h (which
// includes pybind11). startRuntime() lives in module.h, so we forward-declare
// it below instead of including that header.
#include "prepare_kernel.h"  // ::spyre::prepareKernel -> std::unique_ptr<JobPlan>
#include "spyre_kernel.h"  // ::spyre::launchJobPlan(const JobPlan&, args[, stream])
#include "spyre_stream.h"  // ::spyre::synchronizeDevice / getCurrentStream
// ::spyre::SpyreTensorLayout + ::spyre::ElementArrangement, and (transitively
// via <util/sendefs.h>) the DataFormats enum. This header is pybind-free.
#include "spyre_tensor_impl.h"

// ATen (from ${TORCH_LIBRARIES}) for the tensor bridge.
#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>

// spyre_mem.h pulls module.h (pybind11); forward-declare the two symbols we
// need here to keep pybind out of this TU. Signatures confirmed against
// torch_spyre/csrc/spyre_mem.h.
namespace spyre {
at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout,
                                   std::optional<c10::Device> device_opt);
// Purely on-device fill (FillAsync on the current stream): no host buffer, no
// H2D. Takes the tensor's composite address, so it does not read the layout.
at::Tensor spyre_fill_tensor(const at::Tensor& self, double value);
}  // namespace spyre

// ExecuTorch backend runtime.
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

// torch-spyre registers the bulk of its PrivateUse1 device support from Python
// at import time (torch_spyre/__init__.py::_autoload): the device module, the
// AutogradPrivateUse1/PrivateUse1 dispatch-key decomposition kernels, and the
// aten eager kernels — critically aten::_copy_from (torch_spyre/ops/eager.py),
// which is what our tensor bridge needs to move data cpu<->spyre. These live in
// torch.library / @register_kernel decorators and re-enter pybind
// (_C.copy_tensor etc.), so there is no plain C++ symbol we can bind them from.
// Rather than reproduce that surface by hand, we embed CPython and `import
// torch_spyre` once at startup, running the exact init path the working
// pure-Python flow uses.
#include <Python.h>

namespace executorch {
namespace backends {
namespace spyre {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::Backend;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::BackendInterface;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::register_backend;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

namespace {

// --- Phase 9 perf instrumentation (env-gated, zero cost when off) -----------
//
// EXECUTORCH_SPYRE_PROFILE=1 turns on wall-clock timing of the init()/execute()
// phases, emitted as `[SPYRE_PROFILE] ...` ET_LOG(Info) lines. Each line
// carries the delegate instance id (below) so the several partitions a real
// model fragments into — one per CPU-fallback boundary — are disambiguated in
// the log. When the var is unset the only added work is a single cached getenv
// at first use and a branch; control flow and semantics of init()/execute() are
// unchanged (no wire format change either — SpyrePayload.h and the C++ gtests
// are untouched).
using ProfileClock = std::chrono::steady_clock;

bool profile_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("EXECUTORCH_SPYRE_PROFILE");
    // Any non-empty value except a literal "0" enables it.
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return on;
}

double profile_ms_since(const ProfileClock::time_point& t0) {
  return std::chrono::duration<double, std::milli>(ProfileClock::now() - t0)
      .count();
}

// One id per delegated partition, assigned in init() order. Lets the log tie an
// execute() line back to the partition whose init() line has the same id.
std::atomic<int> g_profile_delegate_seq{0};

// torch-spyre's Python autoload sets up the full PrivateUse1 dispatch surface
// (device module + aten::_copy_from + dispatch-key kernels) and starts the
// runtime; the bare executor_runner never imports torch_spyre, so we drive that
// same init once here by embedding CPython.
//
// Phase 4 investigation — is this embedded-CPython dependency removable?
//   No, not without an upstream torch-spyre change. The device bring-up we need
//   (backend rename, device-module registration, the dispatch-key + _copy_from
//   kernel registrations, and start_runtime()) is wired through torch-spyre's
//   `_autoload` Python entry point and pybind-exported symbols; there is no
//   C++-callable entry point that performs the same registration. So a pure-C++
//   runner still has to go through the interpreter. This is INTENTIONAL and
//   upstream-blocked, not a shortcut. Recommendation on record: file a
//   torch-spyre feature request for a non-pybind
//   `spyre::initializeDeviceRuntime()` a native embedder can call; once that
//   exists this whole function collapses to a single C++ call and the embedded
//   interpreter (and its libtorch-Python dependency) can be dropped.
//
// Runtime requirement: the embedded interpreter uses the host libpython, which
// starts with an empty module path — torch and torch_spyre must be reachable on
// PYTHONPATH (venv site-packages + the torch-spyre repo root) or the
// PyRun_SimpleString below raises ModuleNotFoundError and the ET_CHECK aborts.
// See documentation for setup instructions.
//
// After Py_Initialize() the calling (main) thread holds the GIL and keeps it:
// the ATen ops in execute() dispatch back into these Python-registered kernels,
// which need the GIL held. We deliberately never release it (single-threaded
// runner), so those re-entrant kernel calls Just Work.
void ensure_runtime_started() {
  static std::once_flag once;
  std::call_once(once, []() {
    // Phase 9: this call_once body is the one-time embedded-CPython cost — the
    // top removal candidate on record (blocked on an upstream non-pybind
    // spyre::initializeDeviceRuntime(); see the Phase 4 note above). Time it
    // distinctly so the report can quantify exactly what that upstream request
    // would save. Logged once (only the first partition's init() reaches this
    // body; later partitions no-op through call_once).
    const bool prof = profile_enabled();
    const auto t0 = ProfileClock::now();
    if (!Py_IsInitialized()) {
      Py_Initialize();
    }
    // import torch first (torch_spyre imports it), then torch_spyre (runs
    // _autoload: rename_privateuse1_backend, _register_device_module, the
    // dispatch-key + eager kernel registrations incl. _copy_from). Finally
    // start_runtime() — the same call _SpyreImpl._lazy_init() makes; it is
    // std::call_once on the C++ side so this is idempotent.
    const int rc = PyRun_SimpleString(
        "import torch\n"
        "import torch_spyre\n"
        "torch_spyre._autoload()\n"
        "torch_spyre._C.start_runtime()\n");
    ET_CHECK_MSG(
        rc == 0,
        "Spyre delegate: embedded torch_spyre import / start_runtime failed "
        "(see Python traceback above)");
    if (prof) {
      ET_LOG(Info,
             "[SPYRE_PROFILE] cpython_init_ms=%.3f (one-time: Py_Initialize + "
             "import torch + import torch_spyre + start_runtime)",
             profile_ms_since(t0));
    }
  });
}

// ET portable ScalarType -> c10::ScalarType. Explicit (not a reinterpret) so a
// future divergence between the enums surfaces as a compile/lookup error rather
// than silent corruption. Covers the float compute dtypes plus the int/bool
// dtypes real decoders bind as delegate I/O (int64 token-ids/positions, bool
// masks -- Phase 7 mixed-dtype). Kept in sync with dtype_from_str below.
c10::ScalarType to_c10(ScalarType t) {
  switch (t) {
    case ScalarType::Half:
      return c10::kHalf;
    case ScalarType::Float:
      return c10::kFloat;
    case ScalarType::BFloat16:
      return c10::kBFloat16;
    case ScalarType::Int:
      return c10::kInt;
    case ScalarType::Long:
      return c10::kLong;
    case ScalarType::Bool:
      return c10::kBool;
    case ScalarType::Byte:
      return c10::kByte;
    default:
      ET_CHECK_MSG(false, "Spyre delegate: unsupported ScalarType %d",
                   static_cast<int>(t));
      return c10::kHalf;  // unreachable
  }
}

// Bare wire dtype name (Intermediate::dtype, e.g. "float16") ->
// c10::ScalarType. These are ``str(dtype).removeprefix("torch.")`` on the AOT
// side, so no "torch." prefix. Used to allocate the id-indexed inter-kernel
// device buffers (Phase 6e); the delegate I/O tensors use to_c10(ScalarType)
// instead since ET hands those in already typed.
c10::ScalarType dtype_from_str(const std::string& s) {
  if (s == "float16") return c10::kHalf;
  if (s == "float32") return c10::kFloat;
  if (s == "bfloat16") return c10::kBFloat16;
  if (s == "int32") return c10::kInt;
  if (s == "int64") return c10::kLong;
  if (s == "uint8") return c10::kByte;
  if (s == "bool") return c10::kBool;
  ET_CHECK_MSG(false, "Spyre delegate: unsupported intermediate dtype '%s'",
               s.c_str());
  return c10::kHalf;  // unreachable
}

std::vector<int64_t> et_sizes(const Tensor& t) {
  std::vector<int64_t> sizes;
  sizes.reserve(t.dim());
  for (ssize_t i = 0; i < t.dim(); ++i) {
    sizes.push_back(static_cast<int64_t>(t.size(i)));
  }
  return sizes;
}

struct PreparedBucket;  // fwd: defined below alongside SpyreHandle.

// Render "(inputI[axis A]=N, ...)" naming the observed size of each dynamic dim
// for a no-match diagnostic (execute() appends the available bucket
// shape_keys). Kept tiny and string-only so it never touches the device.
std::string describe_dynamic_shape(
    const std::vector<std::pair<std::size_t, std::int64_t>>& dims,
    const std::vector<int64_t>& observed) {
  std::string s = "(";
  for (std::size_t d = 0; d < dims.size(); ++d) {
    if (d) s += ", ";
    s += "input" + std::to_string(dims[d].first) + "[axis " +
         std::to_string(dims[d].second) + "]=" + std::to_string(observed[d]);
  }
  s += ")";
  return s;
}

// LaunchEntry (one JobPlan launch slot) is defined in SpyrePayload.h alongside
// the parser that fills it.

// stl_from_payload_layout / spyre_empty_from_payload_layout (payload
// ``device_layout`` -> SpyreTensorLayout / device tensor) live in
// SpyreLayoutBridge.{h,cpp}, the single ET-side layout boundary.

// One prepared kernel: a compiled JobPlan paired with the launch sequence that
// replays its args in the compiler's expected order. A v2 kernel set yields N
// of these, launched in vector order (Phase 6e); a single kernel is N==1.
struct PreparedKernel {
  std::unique_ptr<::spyre::JobPlan> plan;
  std::vector<LaunchEntry> launch_layout;
};

// One prepared shape bucket (Phase 7 Track 2): the compiled JobPlan(s) for one
// concrete dynamic-dim size, plus that bucket's own scratch pool and
// inter-kernel intermediate table. A v2 (non-bucketed) payload yields exactly
// one bucket with an empty ``shape_key``, which execute() always selects. A v3
// payload yields one bucket per declared size; execute() picks by matching the
// runtime shape of the dynamic input(s) against ``shape_key``.
struct PreparedBucket {
  std::vector<std::int64_t>
      shape_key;  // concrete dynamic-dim sizes; empty=always
  std::vector<PreparedKernel> kernels;
  std::int64_t pool_sticks = 0;
  std::vector<Intermediate> intermediates;
  // Concrete output shape per delegate output for this bucket. ExecuTorch
  // preallocates the delegate's output EValue at the upper-bound trace shape
  // (the largest bucket), so execute() resizes each output to its selected
  // bucket's true shape before the copy-back. Empty => fall back to the ET
  // output shape.
  std::vector<std::vector<std::int64_t>> output_shapes;
  // Device layout each output must be allocated with (recorded at AOT from the
  // compiled kernel's real output tensor). Empty/absent => default layout.
  std::vector<OutputLayout> output_layouts;
  // AOT-recorded dtype per delegate output (bare wire string, e.g. "float16").
  // Parallel to output_shapes. Empty string => absent in payload (skip check).
  // execute() verifies the ET EValue's scalar_type() matches this to catch a
  // compile-time fp32/fp16 mismatch before it becomes a silent bit-reinterpret.
  std::vector<std::string> output_dtypes;
};

// Runtime state for one delegated subgraph. Lives in the ET runtime arena
// (allocateInstance + placement-new); destroyed explicitly in destroy().
//
// Phase 6d made this multi-kernel: init() prepares every kernel in the payload.
// Phase 6e wired the N>1 launch DAG (walking each kernel's launch_layout,
// allocating the shared pool once, threading inter-kernel intermediates
// producer->consumer on-device). Phase 7 Track 2 made it multi-bucket: init()
// prepares every kernel of every shape bucket, and execute() selects the bucket
// whose ``shape_key`` matches the runtime shape of the dynamic input(s) before
// running the (unchanged) per-bucket launch sequence. A non-bucketed payload
// has exactly one bucket with an empty shape_key, so the Phase-2/3/6 path is
// byte-identical.
struct SpyreHandle {
  std::vector<PreparedBucket> buckets;
  // Which delegate input dim(s) select the bucket at runtime: each pair is
  // (delegate_input_index, axis). Empty for a non-bucketed payload.
  std::vector<std::pair<std::size_t, std::int64_t>> dynamic_input_dims;
  size_t n_inputs = 0;
  size_t n_outputs = 0;
  // Whether the N>1 launch DAG is actually runnable, computed once at init()
  // from the parsed payload (classify_launch: launchable iff EVERY bucket is).
  // A non-resident inter-kernel edge or a dangling intermediate id makes it
  // non-kOk; execute() rejects those loudly rather than launch with an
  // uninitialized buffer. Always kOk when every bucket has a single kernel.
  Launchability launchability = Launchability::kOk;
  std::string launch_reject;  // human-readable reason when launchability != kOk
  std::string tmpdir;  // extracted SpyreCode dir root; removed in destroy()
  int profile_id =
      0;  // Phase 9: delegate instance id for [SPYRE_PROFILE] lines

  // --- Persistent-weight cache (Step 1 of stateful Flex-session evolution) ---
  //
  // Model parameters and buffers are frozen across execute() calls: the same
  // weight values are passed by the ET runtime on every decode step. Uploading
  // them to the device once and reusing the device tensor saves one H2D DMA per
  // weight per call.
  //
  // ``input_kinds[i]`` mirrors ParsedPayload::input_kinds[i]: "user_input",
  // "parameter", "buffer", or "constant". Copied from the payload header in
  // init() so execute() can decide — without touching the payload — which
  // delegate-input indices are safe to cache.
  //
  // ``weight_cache[i]``: the persistent device tensor for input i. Defined
  // (non-null storage) iff input_kinds[i] is "parameter", "buffer", or
  // "constant" AND the first execute() has already run. Initialised lazily on
  // the first execute() call so init() stays device-free with respect to
  // weights (it has no access to the actual weight values).
  //
  // Lifetime: owned by SpyreHandle from first execute() until destroy() calls
  // ~SpyreHandle(), which destructs the vector and drops each at::Tensor
  // refcount to zero, triggering SpyreAllocator::ReportAndDelete -> device
  // free. No explicit teardown needed.
  std::vector<std::string> input_kinds;  // parallel to delegate input index
  std::vector<at::Tensor>
      weight_cache;  // device tensors; undefined = not cached
  // Per-input state_out index (parallel to delegate input index). Copied from
  // ParsedPayload::input_state_out in init(). A value of -1 means "no output
  // mapping" (every input that is not kv_state); a non-negative value names the
  // delegate output index that execute() captures for this kv_state input.
  std::vector<std::int64_t>
      input_state_out;  // parallel to delegate input index

  // --- DelegateCall <-> payload ordinal mapping ---
  //
  // ExecuTorch's DelegateCall passes only the *ET-visible* args: the
  // non-kv_state inputs followed by the non-kv_state outputs. kv_state inputs
  // (BUFFER) and their BUFFER_MUTATION outputs are owned by the delegate and
  // never appear in args[]. The payload, however, enumerates every input and
  // output, so these tables translate payload ordinal -> args[] position (-1 =
  // not ET-visible).
  //   et_in_of[payload_input_idx]   : args[] index, or -1 for kv_state inputs
  //   et_out_of[payload_output_idx] : args[] index, or -1 for kv_state outputs
  // For a payload with no kv_state both are the identity (in: i, out: n_in +
  // i).
  std::vector<std::int64_t> et_in_of;
  std::vector<std::int64_t> et_out_of;
  // For a kv_state output o, the payload input index whose state_out is o
  // (-1 for every other output). Supplies shape/dtype when there is no ET arg.
  std::vector<std::int64_t> kv_out_src_input;
  size_t n_et_args = 0;  // expected args.size()
  // Per-input AOT shape / dtype string (parallel to payload input index).
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::string> input_dtypes;
  // Per-input device layout (parallel to delegate input index), copied from
  // ParsedPayload::input_layouts. present==false => no layout recorded; the
  // runtime keeps its default-layout allocation.
  std::vector<InputLayout> input_layouts;

  // --- Phase B: Delegate-owned KV state (kv_state) ---
  //
  // Inputs classified as "kv_state" are BUFFER + BUFFER_MUTATION delegate
  // inputs (ExecuTorch ``tag_mutated_buffer`` moves them entirely into the
  // delegate). The ET runtime never passes the actual cache bytes through
  // args[] on any call — the delegate owns the allocation entirely. The
  // runtime must:
  //
  //   First call:  Allocate the device tensor from the payload shape/dtype
  //                (and layout, when the payload records one) and zero-fill
  //                it on the device (spyre_fill_tensor 0.0). No H2D.
  //                spyre_empty_from_payload_layout only allocates; torch-spyre
  //                does not zero it, so the fill is required on that path.
  //   Every call:  Inject kv_state_cache[i] as dev_inputs[i] directly. No H2D.
  //   Post-sync:   Capture dev_outs[state_out[i]] into kv_state_cache[i]
  //                (ping-pong: the new KV slice is the next call's input).
  //   D2H:         Skip — the KV mutation output is never read by the host.
  //
  // kv_state_cache[i]: undefined until first execute(); thereafter holds
  // the device tensor from the previous execute()'s output[state_out[i]].
  // Shape and dtype come from the payload's input spec for index i.
  std::vector<at::Tensor>
      kv_state_cache;  // device tensors; undefined = not yet allocated
  // Set of output indices that are KV-state captures (ping-pong outputs).
  // Built once on first execute() from input_kinds + input_state_out.
  // D2H is skipped for these output indices.
  std::vector<bool> kv_out_skip_d2h;  // parallel to n_out; true = skip D2H
};

}  // namespace

class SpyreBackend final : public BackendInterface {
 public:
  ~SpyreBackend() override = default;

  bool is_available() const override { return true; }

  Result<DelegateHandle*> init(
      BackendInitContext& context, FreeableBuffer* processed,
      ArrayRef<CompileSpec> compile_specs) const override {
    (void)compile_specs;

    // Phase 9: assign this partition its instance id (always — the fetch_add is
    // trivial) and start the init() wall clock (only read when
    // EXECUTORCH_SPYRE_PROFILE is on).
    const bool prof = profile_enabled();
    const int inst = g_profile_delegate_seq.fetch_add(1);
    const auto t_init0 = ProfileClock::now();

    const auto* bytes = static_cast<const uint8_t*>(processed->data());
    const size_t total = processed->size();

    // Parse the [u32 header_len][header_json][tar] wire format with the
    // host-only parser (see SpyrePayload.h). It returns a status we map onto
    // the ET Error space; the actual device work stays here.
    ParsedPayload payload;
    std::string parse_err;
    switch (parse_payload(bytes, total, &payload, &parse_err)) {
      case PayloadStatus::kOk:
        break;
      case PayloadStatus::kTooSmall:
      case PayloadStatus::kHeaderOverrun:
      case PayloadStatus::kJsonParseError:
      case PayloadStatus::kV1Unsupported:
      case PayloadStatus::kInvalidSchema:
      case PayloadStatus::kMutableStateUnsupported:
        ET_LOG(Error, "Spyre delegate: %s", parse_err.c_str());
        return Error::InvalidProgram;
    }
    const size_t n_inputs = payload.n_inputs;
    const size_t n_outputs = payload.n_outputs;

    const uint8_t* tar_data = payload.tar_data;
    const size_t tar_size = payload.tar_size;

    // Extract the tar into a fresh temp dir. mkdtemp gives a private,
    // collision-free directory (HOME may be unwritable in the pod).
    char tmpl[] = "/tmp/spyre_code_XXXXXX";
    char* made = ::mkdtemp(tmpl);
    ET_CHECK_OR_RETURN_ERROR(made != nullptr, Internal,
                             "Spyre delegate: mkdtemp failed");
    const std::string tmpdir(made);

    const auto t_tar0 = ProfileClock::now();
    const std::string tar_path = tmpdir + "/blob.tar";
    {
      std::ofstream ofs(tar_path, std::ios::binary);
      ET_CHECK_OR_RETURN_ERROR(ofs.good(), AccessFailed,
                               "Spyre delegate: cannot open %s",
                               tar_path.c_str());
      ofs.write(reinterpret_cast<const char*>(tar_data),
                static_cast<std::streamsize>(tar_size));
    }
    const std::string cmd = "tar -xf '" + tar_path + "' -C '" + tmpdir + "'";
    const int rc = std::system(cmd.c_str());
    ET_CHECK_OR_RETURN_ERROR(rc == 0, Internal,
                             "Spyre delegate: tar extract failed (rc=%d)", rc);
    const double tar_ms = profile_ms_since(t_tar0);

    // Start the Spyre runtime (once) and prepare every kernel of every shape
    // bucket. The parser guarantees at least one bucket (a non-bucketed v2
    // payload synthesizes a single empty-prefix bucket) and at least one kernel
    // per bucket. We prepareKernel() each in the compiler's scheduler order —
    // the vector order preserved on the wire. Each kernel's SpyreCode bundle
    // lives at
    // ``<tmpdir>/<subdir_prefix>/<kernel.subdir>`` (the prefix is empty for a
    // non-bucketed payload, so the path collapses to the Phase-6 layout).
    ensure_runtime_started();
    const auto t_prep0 = ProfileClock::now();
    size_t n_kernels_prepared = 0;
    std::vector<PreparedBucket> buckets;
    buckets.reserve(payload.buckets.size());
    for (size_t b = 0; b < payload.buckets.size(); ++b) {
      const Bucket& bdesc = payload.buckets[b];
      PreparedBucket prepared;
      prepared.shape_key = bdesc.shape_key;
      prepared.pool_sticks = bdesc.pool_sticks;
      prepared.intermediates = bdesc.intermediates;
      prepared.output_shapes = bdesc.output_shapes;
      prepared.output_layouts = bdesc.output_layouts;
      prepared.output_dtypes = bdesc.output_dtypes;
      prepared.kernels.reserve(bdesc.kernels.size());
      const std::string prefix = bdesc.subdir_prefix.empty()
                                     ? std::string()
                                     : bdesc.subdir_prefix + "/";
      for (size_t k = 0; k < bdesc.kernels.size(); ++k) {
        const auto& kdesc = bdesc.kernels[k];
        const std::string spyrecode_dir = tmpdir + "/" + prefix + kdesc.subdir;
        std::unique_ptr<::spyre::JobPlan> plan;
        try {
          plan = ::spyre::prepareKernel(spyrecode_dir);
        } catch (const std::exception& e) {
          ET_LOG(
              Error,
              "Spyre delegate: prepareKernel failed for bucket %zu kernel %zu "
              "(%s): %s",
              b, k, spyrecode_dir.c_str(), e.what());
          return Error::Internal;
        }
        ET_CHECK_OR_RETURN_ERROR(plan != nullptr, Internal,
                                 "Spyre delegate: prepareKernel returned null "
                                 "for bucket %zu kernel %zu "
                                 "(%s)",
                                 b, k, spyrecode_dir.c_str());
        prepared.kernels.push_back(
            PreparedKernel{std::move(plan), kdesc.launch_layout});
        ++n_kernels_prepared;
      }
      ET_CHECK_OR_RETURN_ERROR(
          !prepared.kernels.empty(), InvalidProgram,
          "Spyre delegate: bucket %zu contained no kernels", b);
      buckets.push_back(std::move(prepared));
    }
    ET_CHECK_OR_RETURN_ERROR(!buckets.empty(), InvalidProgram,
                             "Spyre delegate: payload contained no buckets");
    const double prepare_ms = profile_ms_since(t_prep0);

    // Allocate the handle in the runtime arena; placement-new because
    // SpyreHandle has non-trivial members. destroy() runs the destructor
    // explicitly.
    auto* allocator = context.get_runtime_allocator();
    auto* handle = allocator->allocateInstance<SpyreHandle>();
    ET_CHECK_OR_RETURN_ERROR(handle != nullptr, MemoryAllocationFailed,
                             "Spyre delegate: could not allocate SpyreHandle");
    new (handle) SpyreHandle();
    handle->buckets = std::move(buckets);
    handle->dynamic_input_dims = std::move(payload.dynamic_input_dims);
    handle->n_inputs = n_inputs;
    handle->n_outputs = n_outputs;
    // Classify the launch DAG now: launchable iff EVERY bucket is launchable.
    // init() prepares all JobPlans regardless of this verdict — the gate is
    // enforced at execute() so an init failure always means prepareKernel
    // failed, not that the DAG shape is unrunnable.
    handle->launchability = classify_launch(payload, &handle->launch_reject);
    handle->tmpdir = tmpdir;
    handle->profile_id = inst;
    // Copy the per-input kind vector from the parsed payload. weight_cache is
    // left empty (size 0) here; execute() resizes it to n_inputs on first call
    // and populates each cacheable slot after the first H2D.
    handle->input_kinds = std::move(payload.input_kinds);
    handle->input_state_out = std::move(payload.input_state_out);
    handle->input_layouts = std::move(payload.input_layouts);
    handle->input_shapes = std::move(payload.input_shapes);
    handle->input_dtypes = std::move(payload.input_dtypes);

    // Build payload-ordinal -> ET-arg-ordinal maps. kv_state inputs and the
    // outputs they capture (state_out) are delegate-owned => -1.
    handle->kv_out_src_input.assign(n_outputs, -1);
    for (size_t i = 0; i < n_inputs; ++i) {
      if (i >= handle->input_kinds.size() ||
          handle->input_kinds[i] != "kv_state") {
        continue;
      }
      if (i >= handle->input_state_out.size()) continue;
      const std::int64_t so = handle->input_state_out[i];
      if (so >= 0 && static_cast<size_t>(so) < n_outputs) {
        handle->kv_out_src_input[static_cast<size_t>(so)] =
            static_cast<std::int64_t>(i);
      }
    }
    handle->et_in_of.assign(n_inputs, -1);
    handle->et_out_of.assign(n_outputs, -1);
    {
      std::int64_t next = 0;
      for (size_t i = 0; i < n_inputs; ++i) {
        const bool kv = i < handle->input_kinds.size() &&
                        handle->input_kinds[i] == "kv_state";
        if (!kv) handle->et_in_of[i] = next++;
      }
      for (size_t o = 0; o < n_outputs; ++o) {
        if (handle->kv_out_src_input[o] < 0) handle->et_out_of[o] = next++;
      }
      handle->n_et_args = static_cast<size_t>(next);
    }

    // The processed buffer is fully consumed (SpyreCode is on disk + in the
    // JobPlan); release it back to the runtime.
    processed->Free();

    if (prof) {
      // Per-partition init breakdown. cpython_init_ms (logged once, inside
      // ensure_runtime_started) is folded into this partition's init_total_ms
      // only for inst 0 — the partition that pays the one-time cost.
      ET_LOG(
          Info,
          "[SPYRE_PROFILE] delegate=%d init_total_ms=%.3f tar_extract_ms=%.3f "
          "prepare_kernels_ms=%.3f n_kernels=%zu n_buckets=%zu n_in=%zu "
          "n_out=%zu",
          inst, profile_ms_since(t_init0), tar_ms, prepare_ms,
          n_kernels_prepared, handle->buckets.size(), n_inputs, n_outputs);
    }

    return handle;
  }

  Error execute(BackendExecutionContext& context, DelegateHandle* handle_,
                Span<EValue*> args) const override {
    (void)context;
    auto* handle = static_cast<SpyreHandle*>(handle_);
    const bool prof = profile_enabled();  // Phase 9: per-phase execute() timing
    const size_t n_in = handle->n_inputs;
    const size_t n_out = handle->n_outputs;

    // DelegateCall passes only non-kv_state inputs + non-kv_state outputs.
    ET_CHECK_OR_RETURN_ERROR(
        args.size() == handle->n_et_args, InvalidArgument,
        "Spyre delegate: expected %zu args (non-kv_state inputs + non-kv_state "
        "outputs; payload n_in=%zu n_out=%zu), got %zu",
        handle->n_et_args, n_in, n_out, args.size());

    // Phase 7 Track 2: select the shape bucket by the runtime shape of the
    // dynamic input(s). A non-bucketed payload has exactly one bucket with an
    // empty shape_key / no dynamic_input_dims, so this always resolves to
    // bucket 0 and the Phase-2/3/6 path is unchanged. A v3 payload reads each
    // declared (input_index, axis) size and matches it against a bucket's
    // shape_key; no match is a hard error listing the observed shape and every
    // available bucket, since prepareKernel compiled only for the declared
    // sizes.
    std::vector<int64_t> observed(handle->dynamic_input_dims.size());
    for (std::size_t d = 0; d < handle->dynamic_input_dims.size(); ++d) {
      const auto& dim = handle->dynamic_input_dims[d];
      ET_CHECK_OR_RETURN_ERROR(
          dim.first < n_in, Internal,
          "Spyre delegate: dynamic input index %zu >= n_inputs %zu", dim.first,
          n_in);
      ET_CHECK_OR_RETURN_ERROR(handle->et_in_of[dim.first] >= 0, Internal,
                               "Spyre delegate: dynamic input %zu is "
                               "delegate-owned kv_state and has "
                               "no ExecuTorch arg",
                               dim.first);
      const Tensor& t =
          args[static_cast<size_t>(handle->et_in_of[dim.first])]->toTensor();
      ET_CHECK_OR_RETURN_ERROR(dim.second >= 0 && dim.second < t.dim(),
                               Internal,
                               "Spyre delegate: dynamic axis %lld out of range "
                               "for input %zu (dim %zd)",
                               static_cast<long long>(dim.second), dim.first,
                               static_cast<ssize_t>(t.dim()));
      observed[d] = static_cast<int64_t>(t.size(dim.second));
    }

    const PreparedBucket* bucket = nullptr;
    if (handle->dynamic_input_dims.empty()) {
      // Non-bucketed payload: exactly one bucket, always selected.
      bucket = &handle->buckets.front();
    } else {
      for (const auto& cand : handle->buckets) {
        if (cand.shape_key == observed) {
          bucket = &cand;
          break;
        }
      }
      if (bucket == nullptr) {
        std::string avail;
        for (std::size_t b = 0; b < handle->buckets.size(); ++b) {
          if (b) avail += ", ";
          avail += "[";
          const auto& key = handle->buckets[b].shape_key;
          for (std::size_t j = 0; j < key.size(); ++j) {
            if (j) avail += ",";
            avail += std::to_string(key[j]);
          }
          avail += "]";
        }
        ET_CHECK_OR_RETURN_ERROR(
            false, InvalidArgument,
            "Spyre delegate: no shape bucket matches observed dynamic dims %s; "
            "available buckets: %s",
            describe_dynamic_shape(handle->dynamic_input_dims, observed)
                .c_str(),
            avail.c_str());
      }
    }

    // Phase 6e: launch all N kernels of the selected bucket in wire (scheduler)
    // order. init() prepared every JobPlan and classified the DAG (launchable
    // iff every bucket is); here we only refuse a DAG that cannot be reproduced
    // on-device — a non-resident inter-kernel edge (host round-trip the payload
    // doesn't encode) or a dangling intermediate id. A single-kernel bucket is
    // always kOk, so the Phase-2/3 path never hits this gate.
    if (bucket->kernels.size() > 1) {
      ET_CHECK_OR_RETURN_ERROR(
          handle->launchability == Launchability::kOk, NotImplemented,
          "Spyre delegate: multi-kernel launch not runnable: %s",
          handle->launch_reject.c_str());
    }

    const auto spyre_dev = c10::Device(c10::DeviceType::PrivateUse1, 0);

    // The whole ATen<->Spyre bridge can throw c10::Error (a std::exception):
    // from_blob/.to() device copies, at::empty on the Spyre allocator, and the
    // launch itself. Wrap it all so the message surfaces via ET_LOG instead of
    // escaping execute() -> std::terminate (which Spyre's terminate handler
    // turns into a bare senlib backtrace with no what()).
    try {
      // Bridge the ET inputs/outputs to Spyre-device tensors, indexed by their
      // delegate position, then assemble the launch args in the JobPlan's
      // expected order (see SpyreHandle::launch_layout).
      //
      // Inputs: wrap the ET data ptr as a CPU tensor, copy to the Spyre device.
      //
      // Persistent-weight cache: parameters, buffers, and constants are frozen
      // across execute() calls. On the first call we upload them normally and
      // stash the resulting device tensor in handle->weight_cache[i]. On
      // subsequent calls we skip the H2D entirely and reuse the cached tensor.
      // User inputs (tokens, positions, masks) bypass the cache
      // unconditionally.
      //
      // weight_cache is resized to n_in on the first execute() that reaches
      // this point. Default-constructed at::Tensor is "undefined" (no storage),
      // which is the "not yet cached" sentinel.
      const auto t_h2d0 = ProfileClock::now();
      if (handle->weight_cache.empty()) {
        handle->weight_cache.resize(n_in);  // all undefined; populated below
      }
      if (handle->kv_state_cache.empty()) {
        handle->kv_state_cache.resize(
            n_in);  // all undefined; allocated on first use
      }
      // Build the kv_out_skip_d2h bitmap once on the first execute() call.
      // An output index is a KV-state capture iff some kv_state input maps to
      // it via state_out. These outputs are never read by the host; skipping
      // D2H avoids an unnecessary D2H DMA for every decode step.
      if (handle->kv_out_skip_d2h.empty() && n_out > 0) {
        handle->kv_out_skip_d2h.assign(n_out, false);
        for (size_t i = 0; i < n_in; ++i) {
          if (i >= handle->input_kinds.size() ||
              handle->input_kinds[i] != "kv_state")
            continue;
          if (i >= handle->input_state_out.size()) continue;
          const std::int64_t so = handle->input_state_out[i];
          if (so >= 0 && static_cast<size_t>(so) < n_out) {
            handle->kv_out_skip_d2h[static_cast<size_t>(so)] = true;
          }
        }
      }
      std::vector<at::Tensor> dev_inputs(n_in);
      size_t n_h2d =
          0;  // for SPYRE_PERSIST log: how many H2D transfers this call
      size_t n_hit = 0;  // for SPYRE_PERSIST log: how many cache hits this call
      for (size_t i = 0; i < n_in; ++i) {
        // Classify this input: is it a frozen weight safe to cache?
        const bool cacheable = i < handle->input_kinds.size() &&
                               (handle->input_kinds[i] == "parameter" ||
                                handle->input_kinds[i] == "buffer" ||
                                handle->input_kinds[i] == "constant");
        // Is it a delegate-owned KV state slot?
        const bool is_kv_state = i < handle->input_kinds.size() &&
                                 handle->input_kinds[i] == "kv_state";

        if (is_kv_state) {
          if (handle->kv_state_cache[i].defined()) {
            // Subsequent calls: inject the device tensor written by the
            // previous execute()'s KV scatter output. No H2D.
            dev_inputs[i] = handle->kv_state_cache[i];
            ++n_hit;
            if (prof)
              ET_LOG(Info,
                     "[SPYRE_KV_STATE] delegate=%d input=%zu INJECT "
                     "kv_ptr=%p",
                     handle->profile_id, i,
                     reinterpret_cast<void*>(
                         handle->kv_state_cache[i].data_ptr()));
          } else {
            // First call: allocate a zero-filled KV tensor on device.
            // ET never passes kv_state through args[]; the delegate owns the
            // allocation — derive shape/dtype from the payload input spec and
            // allocate on-device without copying host bytes (the initial
            // KV state is always zeros; H2D of zeros is wasteful).
            ET_CHECK_OR_RETURN_ERROR(i < handle->input_shapes.size() &&
                                         i < handle->input_dtypes.size() &&
                                         !handle->input_shapes[i].empty() &&
                                         !handle->input_dtypes[i].empty(),
                                     Internal,
                                     "Spyre delegate: kv_state input %zu has "
                                     "no shape/dtype in payload",
                                     i);
            const std::vector<int64_t> kv_shape(handle->input_shapes[i].begin(),
                                                handle->input_shapes[i].end());
            const auto kv_dtype = dtype_from_str(handle->input_dtypes[i]);
            if (i < handle->input_layouts.size() &&
                handle->input_layouts[i].present) {
              dev_inputs[i] = spyre_empty_from_payload_layout(
                  handle->input_layouts[i], kv_shape, kv_dtype, spyre_dev);
              // The allocation above is uninitialised: zero it on the device.
              // The fill is queued on the current stream, ahead of the kernel
              // launches below.
              ::spyre::spyre_fill_tensor(dev_inputs[i], 0.0);
            } else {
              dev_inputs[i] = at::zeros(
                  kv_shape,
                  at::TensorOptions().dtype(kv_dtype).device(spyre_dev));
            }
            handle->kv_state_cache[i] = dev_inputs[i];
            if (prof)
              ET_LOG(
                  Info,
                  "[SPYRE_KV_STATE] delegate=%d input=%zu ALLOC_ZERO "
                  "shape=[%s] dtype=%s kv_ptr=%p",
                  handle->profile_id, i,
                  [&]() -> std::string {
                    std::string s;
                    for (size_t d = 0; d < kv_shape.size(); ++d) {
                      if (d) s += ",";
                      s += std::to_string(kv_shape[d]);
                    }
                    return s;
                  }()
                               .c_str(),
                  c10::toString(kv_dtype),
                  reinterpret_cast<void*>(dev_inputs[i].data_ptr()));
          }
        } else if (cacheable && handle->weight_cache[i].defined()) {
          // Weight cache hit: reuse the device tensor from the previous
          // execute().
          dev_inputs[i] = handle->weight_cache[i];
          ++n_hit;
          if (prof) {
            ET_LOG(Info,
                   "[SPYRE_PERSIST] delegate=%d input=%zu kind=%s CACHE_HIT "
                   "data_ptr=%p",
                   handle->profile_id, i, handle->input_kinds[i].c_str(),
                   reinterpret_cast<void*>(handle->weight_cache[i].data_ptr()));
          }
        } else {
          // H2D upload: user input (every call), first call for a weight.
          ET_CHECK_OR_RETURN_ERROR(
              handle->et_in_of[i] >= 0, Internal,
              "Spyre delegate: input %zu (kind=%s) has no ExecuTorch arg", i,
              i < handle->input_kinds.size() ? handle->input_kinds[i].c_str()
                                             : "?");
          const Tensor& in =
              args[static_cast<size_t>(handle->et_in_of[i])]->toTensor();
          const auto opts = at::TensorOptions().dtype(to_c10(in.scalar_type()));
          at::Tensor cpu = at::from_blob(const_cast<void*>(in.const_data_ptr()),
                                         et_sizes(in), opts);
          dev_inputs[i] = cpu.to(spyre_dev);
          ++n_h2d;
          if (cacheable) {
            // First upload of this weight: store in cache for future calls.
            handle->weight_cache[i] = dev_inputs[i];
            if (prof) {
              ET_LOG(
                  Info,
                  "[SPYRE_PERSIST] delegate=%d input=%zu kind=%s CACHE_STORE "
                  "data_ptr=%p",
                  handle->profile_id, i, handle->input_kinds[i].c_str(),
                  reinterpret_cast<void*>(dev_inputs[i].data_ptr()));
            }
          }
        }
      }
      const double h2d_ms = profile_ms_since(t_h2d0);
      if (prof) {
        ET_LOG(Info,
               "[SPYRE_PERSIST] delegate=%d execute n_h2d=%zu n_hit=%zu "
               "h2d_ms=%.3f",
               handle->profile_id, n_h2d, n_hit, h2d_ms);
      }

      // Device-side allocation (outputs + shared pool + inter-kernel buffers).
      const auto t_alloc0 = ProfileClock::now();

      // Outputs: pre-allocate on the device; keep handles so we can copy back.
      // ExecuTorch's memory planner sizes each delegate output EValue at the
      // upper-bound trace shape (the largest bucket), so for a bucketed payload
      // the ET output is larger than the selected bucket's true output.
      // Allocate the device tensor at the selected bucket's concrete output
      // shape when the AOT header recorded one; otherwise fall back to the ET
      // shape (static v2 path). `out_shapes[i]` is the target we later resize
      // the ET output to.
      std::vector<at::Tensor> dev_outs(n_out);
      std::vector<std::vector<std::int64_t>> out_shapes(n_out);
      for (size_t i = 0; i < n_out; ++i) {
        // Resolve the ET output tensor via et_out_of[]. A delegate-owned
        // kv_state output (BUFFER_MUTATION) has no ET arg (-1): its dtype/shape
        // come from the kv_state input that captures it (kv_out_src_input[i]).
        const bool out_has_et_arg = handle->et_out_of[i] >= 0;
        const Tensor* out_ptr =
            out_has_et_arg
                ? &args[static_cast<size_t>(handle->et_out_of[i])]->toTensor()
                : nullptr;
        c10::ScalarType out_dtype = c10::kHalf;
        std::vector<std::int64_t> out_fallback_shape;
        if (out_has_et_arg) {
          out_dtype = to_c10(out_ptr->scalar_type());
          out_fallback_shape = et_sizes(*out_ptr);
        } else {
          const std::int64_t src = handle->kv_out_src_input[i];
          ET_CHECK_OR_RETURN_ERROR(
              src >= 0 &&
                  static_cast<size_t>(src) < handle->input_dtypes.size() &&
                  static_cast<size_t>(src) < handle->input_shapes.size() &&
                  !handle->input_dtypes[static_cast<size_t>(src)].empty(),
              Internal,
              "Spyre delegate: kv_state output %zu has no source input "
              "shape/dtype in payload",
              i);
          out_dtype =
              dtype_from_str(handle->input_dtypes[static_cast<size_t>(src)]);
          out_fallback_shape = handle->input_shapes[static_cast<size_t>(src)];
        }
        out_shapes[i] = (i < bucket->output_shapes.size() &&
                         !bucket->output_shapes[i].empty())
                            ? bucket->output_shapes[i]
                            : out_fallback_shape;
        // Dtype consistency: the AOT payload records the dtype the compiled
        // kernel actually produced ("outputs"[i]["dtype"]). If the ET EValue's
        // scalar_type() differs, the kernel will write the wrong bit-pattern
        // into a wrongly-typed device buffer and std::memcpy will misinterpret
        // it. An empty string means the field was absent (older payload) —
        // skip.
        if (out_has_et_arg && i < bucket->output_dtypes.size() &&
            !bucket->output_dtypes[i].empty()) {
          const c10::ScalarType aot_dtype =
              dtype_from_str(bucket->output_dtypes[i]);
          ET_CHECK_OR_RETURN_ERROR(
              out_dtype == aot_dtype, Internal,
              "Spyre delegate: output %zu dtype mismatch — ET EValue has %s "
              "but "
              "AOT payload recorded %s; the compiled kernel and the ExecuTorch "
              "memory plan disagree on output dtype.",
              i, c10::toString(out_dtype), bucket->output_dtypes[i].c_str());
        }
        const auto opts =
            at::TensorOptions().dtype(out_dtype).device(spyre_dev);
        if (i < bucket->output_layouts.size() &&
            bucket->output_layouts[i].present) {
          // The compiled kernel writes this output in the device layout
          // recorded at AOT (possibly non-default / tile-major). Allocate with
          // that exact layout so launch + the D2H copy-back interpret the
          // buffer the same way the kernel wrote it (at::empty would pick the
          // default layout).
          const OutputLayout& ol = bucket->output_layouts[i];
          dev_outs[i] = spyre_empty_from_payload_layout(ol, out_shapes[i],
                                                        out_dtype, spyre_dev);
          if (profile_enabled()) {
            ET_LOG(Info,
                   "[SPYRE_LAYOUT] output %zu allocated with recorded layout "
                   "(device_size rank %zu)",
                   i, ol.device_size.size());
          }
        } else {
          dev_outs[i] = at::empty(out_shapes[i], opts);
        }
      }

      // Allocate the shared scratch "pool" ONCE and reuse it across every
      // kernel's kPool slot: a 1-D uint8 device buffer of `pool_sticks`
      // 128-byte sticks (DeepTools intermediate device memory), laid out
      // exactly as inductor's wrapper does. pool_sticks==0 means no kernel
      // needs a pool; leave it undefined and no kPool slot fires.
      at::Tensor pool;
      if (bucket->pool_sticks > 0) {
        const int64_t sticks = bucket->pool_sticks;
        ::spyre::SpyreTensorLayout pool_layout(
            std::vector<int64_t>{sticks, 1, 1}, std::vector<int64_t>{1, 1, 1},
            DataFormats::SENINT8);
        pool = ::spyre::spyre_empty_with_layout(
            std::vector<int64_t>{sticks}, std::vector<int64_t>{1}, c10::kByte,
            pool_layout, spyre_dev);
      }

      // Allocate the id-indexed inter-kernel activation buffers on-device
      // (empty for a single-kernel payload). id == index in the intermediates
      // table, so a kIntermediate launch slot with buffer==id maps to
      // inter_bufs[id]. The producer kernel writes it and the consumer reads it
      // through the SAME tensor, so the activation never round-trips through
      // the host — the point of Phase 6e. The launchability gate above
      // guarantees every referenced id is in range and resident before we get
      // here.
      std::vector<at::Tensor> inter_bufs(bucket->intermediates.size());
      for (size_t m = 0; m < bucket->intermediates.size(); ++m) {
        const Intermediate& im = bucket->intermediates[m];
        const auto opts = at::TensorOptions()
                              .dtype(dtype_from_str(im.dtype))
                              .device(spyre_dev);
        if (im.has_init_value) {
          // Read-only constant operand (e.g. SDPA scale): materialize on host
          // and upload; at::empty would hand the kernel uninitialized memory.
          inter_bufs[m] =
              at::full(im.shape, im.init_value, opts.device(c10::kCPU))
                  .to(spyre_dev);
        } else {
          inter_bufs[m] = at::empty(im.shape, opts);
        }
      }
      const double alloc_ms = profile_ms_since(t_alloc0);
      const auto t_launch0 = ProfileClock::now();

      // Launch each kernel in wire (scheduler) order, assembling its args from
      // its own launch_layout. Intermediates thread producer->consumer through
      // inter_bufs; the pool and delegate I/O tensors are shared across
      // kernels.
      for (size_t ki = 0; ki < bucket->kernels.size(); ++ki) {
        const PreparedKernel& pk = bucket->kernels[ki];
        std::vector<at::Tensor> dev_args;
        const auto& layout = pk.launch_layout;
        dev_args.reserve(layout.size());
        for (const auto& le : layout) {
          switch (le.role) {
            case LaunchEntry::kInput:
              ET_CHECK_OR_RETURN_ERROR(
                  le.index < n_in, Internal,
                  "Spyre delegate: launch input index %zu >= n_inputs %zu",
                  le.index, n_in);
              dev_args.push_back(dev_inputs[le.index]);
              break;
            case LaunchEntry::kOutput:
              ET_CHECK_OR_RETURN_ERROR(
                  le.index < n_out, Internal,
                  "Spyre delegate: launch output index %zu >= n_outputs %zu",
                  le.index, n_out);
              dev_args.push_back(dev_outs[le.index]);
              break;
            case LaunchEntry::kPool:
              // The single shared pool, allocated once above. A kPool slot with
              // pool_sticks==0 would mean an unallocated pool tensor — the
              // parser lifts every kPool's sticks to pool_sticks, so this
              // holds.
              ET_CHECK_OR_RETURN_ERROR(
                  bucket->pool_sticks > 0, Internal,
                  "Spyre delegate: kernel %zu wants a pool but pool_sticks==0",
                  ki);
              dev_args.push_back(pool);
              break;
            case LaunchEntry::kIntermediate:
              // Inter-kernel activation edge: same device buffer for the
              // producing and consuming kernels (le.produces distinguishes the
              // roles at AOT; both map to the same tensor here).
              ET_CHECK_OR_RETURN_ERROR(
                  le.buffer >= 0 &&
                      static_cast<size_t>(le.buffer) < inter_bufs.size(),
                  Internal,
                  "Spyre delegate: kernel %zu intermediate buffer id %lld out "
                  "of "
                  "range (%zu intermediates)",
                  ki, static_cast<long long>(le.buffer), inter_bufs.size());
              dev_args.push_back(inter_bufs[static_cast<size_t>(le.buffer)]);
              break;
            default:
              ET_CHECK_OR_RETURN_ERROR(
                  false, Internal,
                  "Spyre delegate: unknown launch-layout role in kernel %zu",
                  ki);
          }
        }
        ::spyre::launchJobPlan(*pk.plan, dev_args);
      }
      // launchJobPlan only enqueues; the launch loop above is the async submit
      // cost. The real device compute is awaited at synchronizeDevice below, so
      // sync_ms — not launch_ms — is where the on-device time actually lands.
      const double launch_ms = profile_ms_since(t_launch0);

      // One device sync after the whole launch chain (kernels are enqueued in
      // order; the intermediates keep the data dependency on-device).
      const auto t_sync0 = ProfileClock::now();
      ::spyre::synchronizeDevice(spyre_dev);
      const double sync_ms = profile_ms_since(t_sync0);

      // --- Phase B: KV-state capture (ping-pong)
      // ------------------------------ All streams are quiesced after
      // synchronizeDevice(). For each kv_state input at delegate index i,
      // capture output[state_out[i]] into kv_state_cache[i]. The next call's
      // inject reads from this device tensor directly — no D2H, no host
      // involvement. The KV mutation output is never read back to the host (see
      // kv_out_skip_d2h below), so this is a pure device-resident ping-pong:
      // kernel writes output -> we store the handle -> next call uses it as
      // input. Shape/dtype must match the input (same buffer FQN, same
      // allocation shape); a mismatch is fatal.
      for (size_t i = 0; i < n_in; ++i) {
        if (i >= handle->input_kinds.size() ||
            handle->input_kinds[i] != "kv_state") {
          continue;
        }
        if (i >= handle->input_state_out.size()) continue;
        const std::int64_t so_idx = handle->input_state_out[i];
        if (so_idx < 0 || static_cast<size_t>(so_idx) >= n_out) {
          if (prof)
            ET_LOG(Error,
                   "[SPYRE_KV_STATE] delegate=%d input=%zu CAPTURE SKIP: "
                   "state_out=%lld out of range (n_out=%zu)",
                   handle->profile_id, i, static_cast<long long>(so_idx),
                   n_out);
          continue;
        }
        const size_t out_idx = static_cast<size_t>(so_idx);
        const at::Tensor& cap = dev_outs[out_idx];
        const at::Tensor& inj = dev_inputs[i];

        // Shape/dtype validation: the captured output must match the injected
        // input. A mismatch means state_out points to the wrong output.
        ET_CHECK_OR_RETURN_ERROR(
            cap.dtype() == inj.dtype(), Internal,
            "Spyre delegate: kv_state capture DTYPE MISMATCH for input %zu "
            "(out_idx=%zu): input dtype=%s output dtype=%s",
            i, out_idx, c10::toString(inj.dtype().toScalarType()),
            c10::toString(cap.dtype().toScalarType()));
        ET_CHECK_OR_RETURN_ERROR(
            cap.sizes() == inj.sizes(), Internal,
            "Spyre delegate: kv_state capture SHAPE MISMATCH for input %zu "
            "(out_idx=%zu): capture shape != inject shape",
            i, out_idx);

        handle->kv_state_cache[i] = cap;
        if (prof)
          ET_LOG(Info,
                 "[SPYRE_KV_STATE] delegate=%d input=%zu CAPTURE "
                 "out_ptr=%p (out_idx=%zu)",
                 handle->profile_id, i, reinterpret_cast<void*>(cap.data_ptr()),
                 out_idx);
      }
      // -----------------------------------------------------------------------

      // Copy device outputs back into the ET-provided output tensors.
      // KV-state mutation outputs are skipped (kv_out_skip_d2h[i] == true):
      // they are captured into kv_state_cache above and never read by the host.
      // Skipping avoids an unnecessary D2H DMA per KV cache slice per decode
      // step.
      const auto t_d2h0 = ProfileClock::now();
      //
      // ExecuTorch's memory planner sizes each delegate output EValue at the
      // upper-bound trace shape (the largest bucket) and marks it STATIC --
      // torch-spyre lowering drops the symbolic batch dim, so the .pte output
      // spec is concrete [max_batch, ...] with shape_dynamism==0. That means we
      // cannot resize the ET tensor down to a smaller bucket's shape (resize of
      // a static tensor aborts). Instead we write the selected bucket's result
      // to the FRONT of the statically-planned buffer: the batch dim is the
      // leading (outermost) dim, so a [b,...] result occupies the first
      // b*inner contiguous elements of the [max_b,...] buffer. Downstream
      // consumers slice to the true logical shape (which they derive from the
      // input, exactly as the coverage runner does from its manifest).
      for (size_t i = 0; i < n_out; ++i) {
        if (handle->et_out_of[i] < 0 ||
            (!handle->kv_out_skip_d2h.empty() && handle->kv_out_skip_d2h[i])) {
          if (prof)
            ET_LOG(Info,
                   "[SPYRE_KV_STATE] delegate=%d output=%zu SKIP_D2H (kv_state "
                   "capture)",
                   handle->profile_id, i);
          continue;
        }
        at::Tensor host = dev_outs[i].to(c10::kCPU).contiguous();
        Tensor out =
            args[static_cast<size_t>(handle->et_out_of[i])]->toTensor();
        ET_CHECK_OR_RETURN_ERROR(
            static_cast<size_t>(host.nbytes()) <= out.nbytes(), Internal,
            "Spyre delegate: output %zu bucket bytes %zu exceed planned %zu", i,
            static_cast<size_t>(host.nbytes()), out.nbytes());
        std::memcpy(out.mutable_data_ptr(), host.data_ptr(), host.nbytes());
      }
      if (prof) {
        const double d2h_ms = profile_ms_since(t_d2h0);
        ET_LOG(Info,
               "[SPYRE_PROFILE] delegate=%d execute h2d_ms=%.3f alloc_ms=%.3f "
               "launch_ms=%.3f sync_ms=%.3f d2h_ms=%.3f n_in=%zu n_out=%zu "
               "n_kernels=%zu pool_sticks=%lld",
               handle->profile_id, h2d_ms, alloc_ms, launch_ms, sync_ms, d2h_ms,
               n_in, n_out, bucket->kernels.size(),
               static_cast<long long>(bucket->pool_sticks));
      }
    } catch (const std::exception& e) {
      ET_LOG(Error, "Spyre delegate: execute() bridge failed: %s", e.what());
      return Error::Internal;
    }

    return Error::Ok;
  }

  void destroy(DelegateHandle* handle_) const override {
    if (handle_ == nullptr) {
      return;
    }
    auto* handle = static_cast<SpyreHandle*>(handle_);
    if (!handle->tmpdir.empty()) {
      const std::string cmd = "rm -rf '" + handle->tmpdir + "'";
      (void)std::system(cmd.c_str());
    }
    // Arena memory is reclaimed by the runtime; we only run the destructor to
    // release the JobPlans and std::string/vector members.
    handle->~SpyreHandle();
  }
};

namespace {
// Static registration: the global ctor runs at load time. The lib is
// whole-archive-linked (see CMakeLists.txt) so this is not stripped.
SpyreBackend g_spyre_backend;
Backend g_spyre_backend_entry{"SpyreBackend", &g_spyre_backend};
Error g_spyre_registered = register_backend(g_spyre_backend_entry);

// spyre2 alias (opt-in): the sibling spyre2 delegate (Triton->TTIR AOT
// front-end) tags its partitions with backend_id "Spyre2Backend" so
// ExecuTorch's AOT resolver (which matches backend_id == the Python class
// __name__) routes to Spyre2Backend.preprocess rather than spyre's
// SpyreBackend. Its device (Stage 2 SDSC) payload is byte-compatible with this
// runtime, so we register the SAME stateless interface under spyre2's name too.
// Guarded by SPYRE_REGISTER_SPYRE2_ALIAS: spyre's default build defines nothing
// and is byte-identical; spyre2's CMake defines it.
#ifdef SPYRE_REGISTER_SPYRE2_ALIAS
Backend g_spyre2_backend_entry{"Spyre2Backend", &g_spyre_backend};
Error g_spyre2_registered = register_backend(g_spyre2_backend_entry);
#endif
}  // namespace

}  // namespace spyre
}  // namespace backends
}  // namespace executorch
