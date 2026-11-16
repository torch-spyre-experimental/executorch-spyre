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
// Pure parser for the Spyre delegate payload wire format. Deliberately free of
// ExecuTorch, torch-spyre, ATen, and CPython dependencies so it can be unit
// tested off-device (see test/SpyrePayloadTest.cpp) — the C++ mirror of the
// Python-side ``_pack_payload`` round-trip tests in test/test_preprocess.py.
//
// Wire format (produced by spyre_backend.py::_pack_payload):
//   [u32 header_len, little-endian] [header_json utf-8] [tar bytes]
//
// Two header schema versions are understood, selected structurally (a "buckets"
// array means v3, a "kernels" array means v2):
//
//   v2 (multi-kernel, the Phase 6 payload; emitted for every static compile,
//   including a single kernel, which is one "kernels" entry):
//   a "kernels" array — each entry a {"subdir", "launch_layout"} — launched
//   sequentially, sharing one top-level "pool_sticks" scratch pool. Every
//   kernel must carry a non-empty "launch_layout". Inter-kernel
//   activation edges are named by an "intermediate" launch role that carries a
//   "buffer" id into the top-level "intermediates" table plus a "produces" flag
//   (true = this kernel writes the buffer, false = it reads it), so the runtime
//   reconstructs producer->consumer by walking each launch_layout in order.
//   See documentation for the schema details.
//
//   v3 (static bucketing, the Phase 7 Track 2 payload): a "buckets" array where
//   each entry is a complete v2 kernel-set (its own "pool_sticks"/"kernels"/
//   "intermediates") plus a "shape_key" (the concrete dynamic-dim sizes this
//   bucket was compiled for) and a "subdir_prefix" (the tar dir its SpyreCode
//   bundles live under, e.g. "bucket0"). A top-level "dynamic_input_dims" lists
//   the [input_index, axis] pairs execute() reads to select a bucket by runtime
//   shape. Spyre's prepareKernel compiles for a FIXED shape, so a dynamic
//   seq/batch dim is served by compiling the subgraph once per declared bucket.
//
// The v1 single-kernel schema ("spyrecode_subdir" + one top-level
// "launch_layout", no "kernels"/"buckets") is no longer supported: such a
// payload is rejected with PayloadStatus::kV1Unsupported and the .pte must be
// re-exported.
//
// Every accepted version populates ``ParsedPayload::buckets`` (a v2 payload
// gets one synthetic empty-prefix bucket), so callers iterate ``buckets``
// regardless of the on-wire version. This header extracts everything into POD
// structs; SpyreBackend.cpp maps the returned status onto ExecuTorch's Error
// enum and owns everything that actually touches the device (tar extraction,
// prepareKernel, the tensor bridge). Keeping the byte-level parsing here means
// a malformed-payload change is caught by a fast host-only gtest instead of
// only surfacing on a Spyre box.

#ifndef EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYREPAYLOAD_H_
#define EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYREPAYLOAD_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "SpyreLayout.h"

namespace executorch {
namespace backends {
namespace spyre {

// One slot in the JobPlan launch sequence (recovered at AOT preprocess; see
// spyre_backend.py's launch-layout capture). The compiled multi-op kernel does
// NOT launch in placeholder order: inductor prepends a scratch "pool" tensor
// and reorders the inputs. execute() replays this exact sequence.
//
// kIntermediate (v2 only) names an inter-kernel activation buffer: ``buffer``
// is an id into ParsedPayload::intermediates and ``produces`` says whether this
// kernel writes it (true) or reads it (false). This is how the launch DAG is
// threaded — producer and consumer kernels reference the same buffer id.
struct LaunchEntry {
  enum Role { kInput, kOutput, kPool, kIntermediate, kOther } role = kOther;
  std::size_t index = 0;     // role==kInput/kOutput: index into delegate I/O
  std::int64_t sticks = 0;   // role==kPool: scratch size in 128-byte sticks
  std::int64_t buffer = -1;  // role==kIntermediate: id into intermediates[]
  bool produces = false;  // role==kIntermediate: true=writes it, false=reads it
};

// One SpyreCode bundle (v2). Kernels launch in vector order, each with its own
// spyreCode subdir and launch layout, sharing the payload's single scratch
// pool.
struct Kernel {
  std::string subdir = "spyreCodeDir";
  std::vector<LaunchEntry> launch_layout;
};

// An inter-kernel activation buffer (v2). ``resident`` distinguishes a
// device-threaded edge from a host-round-tripped one. ``delegate_output`` marks
// an intermediate whose value is also a delegate output. Both are parsed now
// and acted on in Phase 6e.
struct Intermediate {
  std::int64_t id = 0;
  std::vector<std::int64_t> shape;
  std::string dtype;
  bool resident = false;
  bool delegate_output = false;
  // Read-only constant operand (e.g. the SDPA scale): when set, the runtime
  // fills the buffer with init_value instead of leaving it uninitialized.
  bool has_init_value = false;
  double init_value = 0.0;
};

// OutputLayout / InputLayout (the ``device_layout`` schema) and
// parse_device_layout / parse_i64_array live in SpyreLayout.h, the single
// ET-side layout boundary.

// One shape bucket (v3, Phase 7 Track 2). Spyre's prepareKernel compiles for a
// FIXED shape, so a dynamic seq/batch dim is served by compiling the subgraph
// once per declared bucket size and selecting at runtime by input shape. A
// bucket is a complete v2 kernel-set (its own scratch pool, kernels, and
// inter-kernel intermediates) plus the concrete ``shape_key`` the runtime
// matches against and the ``subdir_prefix`` its SpyreCode bundles live under in
// the tar (``bucket{b}/spyreCodeDir_{i}``). v2 payloads synthesize a single
// bucket with an empty prefix and empty shape_key so callers always iterate
// ``buckets`` uniformly.
struct Bucket {
  std::vector<std::int64_t>
      shape_key;              // concrete dynamic-dim sizes; empty=non-bucketed
  std::string subdir_prefix;  // "" for v2; "bucket0"/... for v3
  std::int64_t pool_sticks = 0;
  std::vector<Kernel> kernels;
  std::vector<Intermediate> intermediates;
  // Concrete output shape per delegate output for THIS bucket (v2/v3 header
  // "outputs"[j]["shape"]). ExecuTorch preallocates the delegate's output
  // EValue at the upper-bound trace shape (the largest bucket), so execute()
  // must resize each output to the selected bucket's true shape before copying
  // results back. Empty when the payload has no "outputs" record => fall back
  // to the ET shape.
  std::vector<std::vector<std::int64_t>> output_shapes;
  // Device layout per delegate output (header "outputs"[j]["device_layout"]).
  // Parallel to ``output_shapes``; an entry with present==false means "none".
  std::vector<OutputLayout> output_layouts;
  // AOT-recorded dtype per delegate output (header "outputs"[j]["dtype"]),
  // as the bare wire string ("float16", "float32", …). Parallel to
  // ``output_shapes``; empty string means the field was absent in an older
  // payload. execute() uses this to verify that the ET EValue's scalar_type()
  // agrees with the dtype the compiled kernel actually produced at AOT,
  // catching a fp32/fp16 mismatch before a silent memcpy-misinterpretation.
  std::vector<std::string> output_dtypes;
};

// Everything init() needs out of the payload. ``tar_data`` is a *view* into the
// caller's buffer (not owned): the source bytes must outlive this struct.
//
// The kernel sets live in ``buckets`` for both accepted schema versions; there
// are no top-level kernels / pool / intermediates fields.
struct ParsedPayload {
  std::size_t n_inputs = 0;
  std::size_t n_outputs = 0;
  const std::uint8_t* tar_data = nullptr;
  std::size_t tar_size = 0;

  int version = 1;

  // ``buckets`` is populated for EVERY accepted version: a v2 payload yields a
  // single bucket (empty prefix, empty shape_key) holding its kernels, pool and
  // intermediates, so execute() can always iterate ``buckets`` uniformly.
  // ``dynamic_input_dims`` names which delegate input dim(s) the runtime reads
  // to pick a bucket: each pair is (delegate_input_index, axis). Empty for
  // non-bucketed payloads.
  std::vector<Bucket> buckets;
  std::vector<std::pair<std::size_t, std::int64_t>> dynamic_input_dims;

  // Per-input kind classification, parallel to the delegate input index.
  // Populated from the top-level ``"inputs"[i]["kind"]`` field present in every
  // payload version (v2/v3). Values are the AOT strings:
  //   "user_input"    -- runtime activation (tokens, positions, masks, ...)
  //   "parameter"     -- lifted model weight (frozen across execute() calls)
  //   "buffer"        -- lifted model buffer (frozen across execute() calls)
  //   "constant"      -- lifted constant tensor (frozen across execute() calls)
  //   "kv_state"      -- delegate-owned KV cache (BUFFER + BUFFER_MUTATION)
  // Defaults to "user_input" for any index where the field is absent.
  // execute() uses this to decide which inputs are safe to cache as persistent
  // device-resident tensors. A "mutable_state" kind is retired and rejected by
  // parse_input_kinds (PayloadStatus::kMutableStateUnsupported).
  std::vector<std::string> input_kinds;

  // Per-input state_out index, parallel to the delegate input index.
  // Populated from the top-level ``"inputs"[i]["state_out"]`` field. A value
  // of -1 means "no mapping" (every input that is not kv_state, and any entry
  // that lacks the field). A non-negative value names the delegate output index
  // whose device tensor captures the updated value for this kv_state input
  // after each execute() call.
  //
  // Example (Granite KV partition):
  //   inputs[0] = kv_state  -> state_out=0  (index_put_1 is output[0])
  //   inputs[3] = kv_state  -> state_out=2  (index_put   is output[2])
  std::vector<std::int64_t> input_state_out;

  // Per-input device layout, parallel to the delegate input index. Populated
  // from the optional top-level ``"inputs"[i]["device_layout"]`` field. An
  // entry with present==false means "no layout recorded" (field absent).
  std::vector<InputLayout> input_layouts;

  // Per-input AOT shape / dtype string, parallel to the delegate input index
  // (header ``"inputs"[i]["shape"]`` / ``["dtype"]``). Needed for ``kv_state``
  // inputs, which ExecuTorch never passes through DelegateCall args: the
  // runtime allocates them from this metadata. Empty shape / empty dtype means
  // the field was absent.
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::string> input_dtypes;
};

// Parse outcome. Kept separate from ExecuTorch's Error so this header stays
// dependency-free; SpyreBackend.cpp translates each value to an Error code.
enum class PayloadStatus {
  kOk,
  kTooSmall,        // fewer than 4 bytes: no room for the length prefix
  kHeaderOverrun,   // header_len runs past the end of the buffer
  kJsonParseError,  // the header slice is not valid JSON
  kV1Unsupported,   // neither "kernels" nor "buckets": a v1 payload, re-export
  kInvalidSchema,   // a v2/v3 header that is structurally unusable
  kMutableStateUnsupported,  // an input of kind "mutable_state": re-export
};

// Parse one "launch_layout" JSON array into LaunchEntry structs. Shared by the
// v2 kernel and each v3 bucket kernel. An unrecognized role degrades to kOther
// rather than failing the parse (forward-compatible with roles added later).
inline std::vector<LaunchEntry> parse_launch_layout(const nlohmann::json& arr) {
  std::vector<LaunchEntry> layout;
  for (const auto& e : arr) {
    LaunchEntry le;
    const std::string role = e.value("role", std::string("other"));
    if (role == "input") {
      le.role = LaunchEntry::kInput;
      le.index = e.value("index", static_cast<std::size_t>(0));
    } else if (role == "output") {
      le.role = LaunchEntry::kOutput;
      le.index = e.value("index", static_cast<std::size_t>(0));
    } else if (role == "pool") {
      le.role = LaunchEntry::kPool;
      le.sticks = e.value("sticks", static_cast<std::int64_t>(0));
    } else if (role == "intermediate") {
      le.role = LaunchEntry::kIntermediate;
      le.buffer = e.value("buffer", static_cast<std::int64_t>(-1));
      le.produces = e.value("produces", false);
    } else {
      le.role = LaunchEntry::kOther;
    }
    layout.push_back(le);
  }
  return layout;
}

// Parse the top-level ``"inputs"`` array into ``ParsedPayload::input_kinds``,
// ``ParsedPayload::input_state_out`` and ``ParsedPayload::input_layouts``.
// Each entry is ``{"shape":[...], "dtype":"...", "kind":"...", "state_out":N,
// "device_layout":{...}}``. Only ``"kind"``, ``"state_out"`` and the optional
// ``"device_layout"`` are consumed here (shape/dtype are used only at AOT
// preprocess time). Missing or unrecognised kind values default to
// ``"user_input"``; missing ``"state_out"`` defaults to ``-1`` (no output
// mapping); missing ``"device_layout"`` yields a default (present==false)
// InputLayout. An input of kind ``"mutable_state"`` (a retired kind) stops the
// parse and returns ``PayloadStatus::kMutableStateUnsupported`` with a
// re-export message in ``err``. This function is called once in parse_payload
// after the structural dispatch (v2/v3) completes; the ``"inputs"`` key is
// at the top level in every payload version so it is read from ``header``
// directly.
inline PayloadStatus parse_input_kinds(const nlohmann::json& header,
                                       ParsedPayload* out, std::string* err) {
  if (!header.contains("inputs")) return PayloadStatus::kOk;
  for (const auto& inp : header["inputs"]) {
    const std::string kind = inp.value("kind", std::string("user_input"));
    if (kind == "mutable_state") {
      if (err) {
        *err =
            "Spyre payload has an input of kind \"mutable_state\", which is "
            "no longer supported; re-export the .pte with the current "
            "exporter";
      }
      return PayloadStatus::kMutableStateUnsupported;
    }
    out->input_kinds.push_back(kind);
    out->input_shapes.push_back(inp.contains("shape")
                                    ? parse_i64_array(inp["shape"])
                                    : std::vector<std::int64_t>{});
    out->input_dtypes.push_back(inp.value("dtype", std::string()));
    out->input_state_out.push_back(
        inp.value("state_out", static_cast<std::int64_t>(-1)));
    out->input_layouts.push_back(inp.contains("device_layout")
                                     ? parse_device_layout(inp["device_layout"])
                                     : InputLayout{});
  }
  return PayloadStatus::kOk;
}

// Parse the header's ``"outputs"`` array (per-output concrete shape + optional
// ``device_layout``) into ``bucket->output_shapes`` / ``output_layouts``.
// Shared by the v2/v3 kernel-set sub-schema.
inline void parse_outputs(const nlohmann::json& obj, Bucket* bucket) {
  if (!obj.contains("outputs")) return;
  for (const auto& o : obj["outputs"]) {
    std::vector<std::int64_t> shape;
    if (o.contains("shape")) shape = parse_i64_array(o["shape"]);
    bucket->output_shapes.push_back(std::move(shape));
    // Record the AOT dtype string ("float16", "float32", …). An absent or
    // empty field (older payloads) leaves an empty string — execute() treats
    // that as "dtype unknown, skip the consistency check" rather than failing.
    bucket->output_dtypes.push_back(o.value("dtype", std::string()));

    OutputLayout ol;
    if (o.contains("device_layout"))
      ol = parse_device_layout(o["device_layout"]);
    bucket->output_layouts.push_back(std::move(ol));
  }
}

// Parse the ``{"pool_sticks", "kernels", "intermediates"}`` sub-schema into a
// Bucket. Shared by the v2 top-level header and each v3 bucket entry (both
// carry the identical kernel-set shape) — the only structural difference
// between v2 and a v3 bucket is that a v3 bucket adds
// ``shape_key``/``subdir_prefix``, filled by the caller. ``shape_key`` and
// ``subdir_prefix`` are left at their defaults here.
//
// Every kernel must carry a non-empty ``launch_layout``: it is the only record
// of which tensor goes in which launch slot, and the runtime no longer guesses
// a naive inputs-then-outputs order. Returns false (and writes ``err``)
// otherwise.
inline bool parse_kernel_set(const nlohmann::json& obj, Bucket* bucket,
                             std::string* err = nullptr) {
  bucket->pool_sticks = obj.value("pool_sticks", static_cast<std::int64_t>(0));
  if (obj.contains("kernels")) {
    std::size_t ki = 0;
    for (const auto& k : obj["kernels"]) {
      Kernel kernel;
      kernel.subdir = k.value("subdir", std::string("spyreCodeDir"));
      if (!k.contains("launch_layout") || !k["launch_layout"].is_array() ||
          k["launch_layout"].empty()) {
        if (err) {
          *err = "Spyre payload kernel " + std::to_string(ki) + " (" +
                 kernel.subdir +
                 ") has a missing or empty \"launch_layout\"; re-export the "
                 ".pte with the current exporter";
        }
        return false;
      }
      kernel.launch_layout = parse_launch_layout(k["launch_layout"]);
      bucket->kernels.push_back(std::move(kernel));
      ++ki;
    }
  }
  if (obj.contains("intermediates")) {
    for (const auto& m : obj["intermediates"]) {
      Intermediate inter;
      inter.id = m.value("id", static_cast<std::int64_t>(0));
      if (m.contains("shape")) {
        for (const auto& s : m["shape"]) {
          inter.shape.push_back(s.get<std::int64_t>());
        }
      }
      inter.dtype = m.value("dtype", std::string());
      inter.resident = m.value("resident", false);
      inter.delegate_output = m.value("delegate_output", false);
      if (m.contains("init_value")) {
        inter.has_init_value = true;
        inter.init_value = m["init_value"].get<double>();
      }
      bucket->intermediates.push_back(std::move(inter));
    }
  }
  // Per-output concrete shapes (+ device layouts) for this bucket. The AOT side
  // records these from the compiled result
  // (spyre_backend._capture_one_compile), so bucket b's "outputs"[j]["shape"]
  // is the true shape at bucket b's dynamic-dim sizes.
  parse_outputs(obj, bucket);
  return true;
}

// Parse the wire format. On success fills ``*out`` and returns kOk; otherwise
// leaves ``*out`` unspecified, returns the failure reason, and (if ``err`` is
// non-null) writes a human-readable message. ``bytes``/``total`` must outlive
// ``*out`` because ``out->tar_data`` points into them.
inline PayloadStatus parse_payload(const std::uint8_t* bytes, std::size_t total,
                                   ParsedPayload* out,
                                   std::string* err = nullptr) {
  if (total < sizeof(std::uint32_t)) {
    if (err) *err = "payload too small (" + std::to_string(total) + " bytes)";
    return PayloadStatus::kTooSmall;
  }

  std::uint32_t header_len = 0;
  std::memcpy(&header_len, bytes, sizeof(std::uint32_t));
  const std::size_t header_off = sizeof(std::uint32_t);
  if (header_off + header_len > total) {
    if (err) {
      *err = "header_len " + std::to_string(header_len) +
             " overruns payload (" + std::to_string(total) + " bytes)";
    }
    return PayloadStatus::kHeaderOverrun;
  }

  nlohmann::json header;
  try {
    header = nlohmann::json::parse(bytes + header_off,
                                   bytes + header_off + header_len);
  } catch (const std::exception& e) {
    if (err) *err = std::string("header JSON parse failed: ") + e.what();
    return PayloadStatus::kJsonParseError;
  }

  ParsedPayload parsed;
  parsed.version = header.value("version", 1);
  parsed.n_inputs = header.value("n_inputs", static_cast<std::size_t>(0));
  parsed.n_outputs = header.value("n_outputs", static_cast<std::size_t>(0));

  // Dispatch on structure, not just the version int: a "buckets" array is the
  // v3 static-bucketing schema, a "kernels" array is the v2 multi-kernel
  // schema, and anything else is a retired v1 payload, which is rejected. This
  // keeps parsing tolerant if the version field and payload shape ever
  // disagree.
  if (header.contains("buckets")) {
    // ---- v3: one v2 kernel-set per shape bucket
    // ------------------------------ Which delegate input dim(s) select the
    // bucket. Each entry is a [input_index, axis] pair (see
    // spyre_backend._resolve_dynamic_dims).
    if (header.contains("dynamic_input_dims")) {
      for (const auto& d : header["dynamic_input_dims"]) {
        parsed.dynamic_input_dims.emplace_back(d.at(0).get<std::size_t>(),
                                               d.at(1).get<std::int64_t>());
      }
    }
    for (const auto& b : header["buckets"]) {
      Bucket bucket;
      if (b.contains("shape_key")) {
        for (const auto& s : b["shape_key"]) {
          bucket.shape_key.push_back(s.get<std::int64_t>());
        }
      }
      bucket.subdir_prefix = b.value("subdir_prefix", std::string());
      if (!parse_kernel_set(b, &bucket, err))
        return PayloadStatus::kInvalidSchema;
      parsed.buckets.push_back(std::move(bucket));
    }
    if (parsed.buckets.empty()) {
      if (err) *err = "Spyre payload has an empty \"buckets\" array";
      return PayloadStatus::kInvalidSchema;
    }
  } else if (header.contains("kernels")) {
    // ---- v2: N kernels sharing one scratch pool
    // ------------------------------
    Bucket bucket;  // one synthetic bucket, empty prefix/shape_key
    if (!parse_kernel_set(header, &bucket, err)) {
      return PayloadStatus::kInvalidSchema;
    }
    parsed.buckets.push_back(std::move(bucket));
  } else {
    // ---- v1: single kernel (Phase 3 payload), no longer supported
    // ------------ Reject rather than guess: a v1 header has neither "kernels"
    // nor "buckets", and misreading it as an empty v2 payload would launch
    // nothing.
    if (err) {
      *err =
          "Spyre payload is a v1 (single-kernel) payload, which is no longer "
          "supported; re-export the .pte with the current exporter";
    }
    return PayloadStatus::kV1Unsupported;
  }

  parsed.tar_data = bytes + header_off + header_len;
  parsed.tar_size = total - header_off - header_len;

  // Populate input_kinds from the top-level "inputs" array (present in every
  // payload version). Must be called after the version dispatch above because
  // it writes into ``parsed`` after all structural fields are settled.
  const PayloadStatus kinds_status = parse_input_kinds(header, &parsed, err);
  if (kinds_status != PayloadStatus::kOk) return kinds_status;

  *out = std::move(parsed);
  return PayloadStatus::kOk;
}

// Whether a parsed payload's launch DAG can actually be run by execute().
// Kept dependency-free and pure so it is host-testable (see
// SpyrePayloadTest.cpp).
enum class Launchability {
  kOk,               // launchable as-is (single kernel, or an all-resident DAG)
  kNonResidentEdge,  // an inter-kernel edge re-materializes on the host
  kMissingIntermediate  // a kIntermediate slot names a buffer id not in the
                        // table
};

// Decide whether one kernel-set (a bucket, or a v2 payload's single bucket)
// can be launched sequentially on-device.
//
// A single kernel is always launchable — it has no inter-kernel edges. For N>1
// kernels, every kIntermediate launch slot must (a) name a buffer id present in
// ``intermediates`` (id == index, so 0 <= buffer < size), and (b) reference an
// intermediate whose ``resident`` flag is true. A non-resident edge means the
// activation was re-materialized on the host between kernels; the payload does
// NOT encode that intervening host op, so execute() cannot reproduce it
// numerically and must reject it loudly rather than launch with an
// uninitialized device buffer.
//
// On a non-kOk verdict, if ``reason`` is non-null it receives a human-readable
// explanation naming the offending kernel index and buffer id.
inline Launchability classify_kernel_set(
    const std::vector<Kernel>& kernels,
    const std::vector<Intermediate>& intermediates,
    std::string* reason = nullptr) {
  if (kernels.size() <= 1) {
    return Launchability::kOk;
  }
  for (std::size_t ki = 0; ki < kernels.size(); ++ki) {
    for (const auto& le : kernels[ki].launch_layout) {
      if (le.role != LaunchEntry::kIntermediate) continue;
      if (le.buffer < 0 ||
          static_cast<std::size_t>(le.buffer) >= intermediates.size()) {
        if (reason) {
          *reason = "kernel " + std::to_string(ki) +
                    " references intermediate buffer id " +
                    std::to_string(le.buffer) +
                    " not in the intermediates table (" +
                    std::to_string(intermediates.size()) + " entries)";
        }
        return Launchability::kMissingIntermediate;
      }
      if (!intermediates[static_cast<std::size_t>(le.buffer)].resident) {
        if (reason) {
          *reason = "kernel " + std::to_string(ki) +
                    " consumes/produces non-resident intermediate buffer id " +
                    std::to_string(le.buffer) +
                    " (host round-trip edge, not reproducible on-device)";
        }
        return Launchability::kNonResidentEdge;
      }
    }
  }
  return Launchability::kOk;
}

// Decide whether ``p`` can be launched. A bucketed (v3) payload is launchable
// iff EVERY bucket is launchable — execute() picks the bucket at runtime by
// input shape, so any non-launchable bucket is a latent failure. v2 payloads
// carry exactly one synthesized bucket, so this reduces to the single-set
// check. On a non-kOk verdict the offending bucket index is prefixed to
// ``reason``.
inline Launchability classify_launch(const ParsedPayload& p,
                                     std::string* reason = nullptr) {
  for (std::size_t bi = 0; bi < p.buckets.size(); ++bi) {
    std::string set_reason;
    const Launchability v = classify_kernel_set(
        p.buckets[bi].kernels, p.buckets[bi].intermediates, &set_reason);
    if (v != Launchability::kOk) {
      if (reason) {
        // Only prefix the bucket index when there is more than one bucket, so
        // v2 diagnostics stay byte-identical to the pre-v3 message.
        *reason = (p.buckets.size() > 1)
                      ? ("bucket " + std::to_string(bi) + ": " + set_reason)
                      : set_reason;
      }
      return v;
    }
  }
  return Launchability::kOk;
}

}  // namespace spyre
}  // namespace backends
}  // namespace executorch

#endif  // EXECUTORCH_BACKENDS_SPYRE_RUNTIME_SPYREPAYLOAD_H_
