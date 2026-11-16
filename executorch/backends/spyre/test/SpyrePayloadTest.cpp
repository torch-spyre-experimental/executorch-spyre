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
// Host-only gtest for the delegate payload parser (runtime/SpyrePayload.h).
// This is the C++ counterpart to the Python ``_pack_payload`` round-trip tests
// in test/test_preprocess.py: the two must agree on the exact byte framing [u32
// header_len LE][header_json utf-8][tar bytes], or an AOT payload will not
// parse on device. No Spyre device, torch, or ExecuTorch runtime required.

#include <gtest/gtest.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "runtime/SpyrePayload.h"

namespace executorch {
namespace backends {
namespace spyre {
namespace {

// Build a payload the way spyre_backend.py::_pack_payload does:
// little-endian u32 header length, then the JSON, then the tar bytes.
std::vector<std::uint8_t> pack_payload(const nlohmann::json& header,
                                       const std::vector<std::uint8_t>& tar) {
  const std::string json = header.dump();
  const std::uint32_t len = static_cast<std::uint32_t>(json.size());
  std::vector<std::uint8_t> out;
  out.reserve(sizeof(len) + json.size() + tar.size());
  const auto* len_bytes = reinterpret_cast<const std::uint8_t*>(&len);
  out.insert(out.end(), len_bytes, len_bytes + sizeof(len));  // assumes LE host
  out.insert(out.end(), json.begin(), json.end());
  out.insert(out.end(), tar.begin(), tar.end());
  return out;
}

// A minimal v2 kernel entry. v1 headers (neither "kernels" nor "buckets") are
// rejected by the parser, so every test that wants a successful parse needs
// one.
nlohmann::json one_kernel(const std::string& subdir = "spyreCodeDir_0") {
  return nlohmann::json::array(
      {{{"subdir", subdir},
        {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}});
}

TEST(SpyrePayload, FramesHeaderThenTar) {
  const nlohmann::json header = {{"version", 2},
                                 {"backend", "SpyreBackend"},
                                 {"n_inputs", 3},
                                 {"kernels", one_kernel()}};
  const std::vector<std::uint8_t> tar = {0x00, 0x01, 0x02, 0xff};

  const auto bytes = pack_payload(header, tar);
  ParsedPayload out;
  std::string err;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kOk)
      << err;

  EXPECT_EQ(out.n_inputs, 3u);
  EXPECT_EQ(out.tar_size, tar.size());
  // tar_data is a view into the source buffer, positioned right after the JSON.
  EXPECT_EQ(std::memcmp(out.tar_data, tar.data(), tar.size()), 0);
}

TEST(SpyrePayload, DefaultsWhenFieldsAbsent) {
  // A minimal v2 header: the parser must fall back to sane defaults, not throw.
  const nlohmann::json header = {{"kernels", nlohmann::json::array()}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  EXPECT_EQ(out.n_inputs, 0u);
  EXPECT_EQ(out.n_outputs, 0u);
  ASSERT_EQ(out.buckets.size(), 1u);
  EXPECT_TRUE(out.buckets[0].kernels.empty());
  EXPECT_EQ(out.tar_size, 0u);
}

TEST(SpyrePayload, ReadsIoArityAndSubdir) {
  const nlohmann::json header = {
      {"n_inputs", 5}, {"n_outputs", 2}, {"kernels", one_kernel("customDir")}};
  const auto bytes = pack_payload(header, {0x2a});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  EXPECT_EQ(out.n_inputs, 5u);
  EXPECT_EQ(out.n_outputs, 2u);
  ASSERT_EQ(out.buckets.size(), 1u);
  ASSERT_EQ(out.buckets[0].kernels.size(), 1u);
  EXPECT_EQ(out.buckets[0].kernels[0].subdir, "customDir");
}

TEST(SpyrePayload, ParsesLaunchLayoutRoles) {
  // Mirror a Phase 3 layout: pool, then reordered inputs, then the output.
  const nlohmann::json header = {{"version", 2},
                                 {"n_inputs", 2},
                                 {"n_outputs", 1},
                                 {"kernels",
                                  {{{"subdir", "spyreCodeDir_0"},
                                    {"launch_layout",
                                     {{{"role", "pool"}, {"sticks", 64}},
                                      {{"role", "input"}, {"index", 1}},
                                      {{"role", "input"}, {"index", 0}},
                                      {{"role", "output"}, {"index", 0}},
                                      {{"role", "weird"}}}}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  ASSERT_EQ(out.buckets.size(), 1u);
  ASSERT_EQ(out.buckets[0].kernels.size(), 1u);
  const auto& layout = out.buckets[0].kernels[0].launch_layout;
  ASSERT_EQ(layout.size(), 5u);

  EXPECT_EQ(layout[0].role, LaunchEntry::kPool);
  EXPECT_EQ(layout[0].sticks, 64);

  EXPECT_EQ(layout[1].role, LaunchEntry::kInput);
  EXPECT_EQ(layout[1].index, 1u);

  EXPECT_EQ(layout[2].role, LaunchEntry::kInput);
  EXPECT_EQ(layout[2].index, 0u);

  EXPECT_EQ(layout[3].role, LaunchEntry::kOutput);
  EXPECT_EQ(layout[3].index, 0u);

  // An unrecognized role degrades to kOther rather than failing the parse.
  EXPECT_EQ(layout[4].role, LaunchEntry::kOther);
}

TEST(SpyrePayload, HeaderLenIsByteCountNotCharCount) {
  // A multi-byte utf-8 value: the length prefix counts bytes, so the parser's
  // JSON/tar split point must land exactly on the tar. Mirrors the Python
  // test_pack_payload_header_len_is_utf8_byte_count guard.
  const nlohmann::json header = {{"note", "café-µ"}, {"kernels", one_kernel()}};
  const std::vector<std::uint8_t> tar = {'t', 'a', 'i', 'l'};
  const auto bytes = pack_payload(header, tar);
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  ASSERT_EQ(out.tar_size, tar.size());
  EXPECT_EQ(std::memcmp(out.tar_data, tar.data(), tar.size()), 0);
}

TEST(SpyrePayload, RejectsPayloadTooSmall) {
  const std::vector<std::uint8_t> bytes = {0x01, 0x02};  // < 4 bytes
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kTooSmall);
  EXPECT_FALSE(err.empty());
}

TEST(SpyrePayload, RejectsHeaderLenOverrun) {
  // Claim a 1000-byte header in a 10-byte buffer.
  std::vector<std::uint8_t> bytes(10, 0);
  const std::uint32_t len = 1000;
  std::memcpy(bytes.data(), &len, sizeof(len));
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kHeaderOverrun);
  EXPECT_FALSE(err.empty());
}

TEST(SpyrePayload, RejectsMalformedJson) {
  // A valid length prefix pointing at bytes that are not JSON.
  const std::string garbage = "{not json";
  std::vector<std::uint8_t> bytes;
  const std::uint32_t len = static_cast<std::uint32_t>(garbage.size());
  const auto* len_bytes = reinterpret_cast<const std::uint8_t*>(&len);
  bytes.insert(bytes.end(), len_bytes, len_bytes + sizeof(len));
  bytes.insert(bytes.end(), garbage.begin(), garbage.end());
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kJsonParseError);
  EXPECT_FALSE(err.empty());
}

TEST(SpyrePayload, EmptyTarIsValid) {
  // header only, no tar bytes — tar_size should be exactly zero.
  const nlohmann::json header = {{"n_inputs", 1}, {"kernels", one_kernel()}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  EXPECT_EQ(out.tar_size, 0u);
}

// ---- v1 payloads are rejected ----------------------------------------------
// A header with neither "kernels" nor "buckets" is the retired v1 single-kernel
// schema. It must fail with a dedicated status rather than parse as an empty
// payload.

TEST(SpyrePayload, RejectsV1ShapedHeader) {
  const nlohmann::json header = {{"version", 1},
                                 {"n_inputs", 2},
                                 {"n_outputs", 1},
                                 {"spyrecode_subdir", "spyreCodeDir"},
                                 {"launch_layout",
                                  {{{"role", "pool"}, {"sticks", 64}},
                                   {{"role", "input"}, {"index", 0}},
                                   {{"role", "output"}, {"index", 0}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kV1Unsupported);
  EXPECT_NE(err.find("v1"), std::string::npos) << err;
  EXPECT_NE(err.find("re-export"), std::string::npos) << err;
}

TEST(SpyrePayload, RejectsHeaderWithNeitherKernelsNorBuckets) {
  const auto bytes = pack_payload(nlohmann::json::object(), {});
  ParsedPayload out;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kV1Unsupported);
}

// ---- mutable_state inputs are rejected -------------------------------------
// The "mutable_state" input kind is retired (kv_state replaces it). A payload
// that still carries one must fail with a dedicated status and a re-export
// message, in both the v2 and the v3 shape.

TEST(SpyrePayload, RejectsMutableStateInputV2) {
  const nlohmann::json header = {
      {"version", 2},
      {"n_inputs", 2},
      {"n_outputs", 1},
      {"kernels",
       {{{"subdir", "spyreCodeDir_0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
      {"inputs",
       {{{"kind", "user_input"}},
        {{"kind", "mutable_state"}, {"state_out", 0}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kMutableStateUnsupported);
  EXPECT_NE(err.find("mutable_state"), std::string::npos) << err;
  EXPECT_NE(err.find("re-export"), std::string::npos) << err;
}

TEST(SpyrePayload, RejectsMutableStateInputV3) {
  const nlohmann::json header = {
      {"version", 3},
      {"n_inputs", 1},
      {"n_outputs", 1},
      {"buckets",
       {{{"shape_key", {1}},
         {"subdir_prefix", "b0"},
         {"kernels",
          {{{"subdir", "spyreCodeDir_0"},
            {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}}}},
      {"inputs", {{{"kind", "mutable_state"}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kMutableStateUnsupported);
  EXPECT_NE(err.find("re-export"), std::string::npos) << err;
}

TEST(SpyrePayload, KvStateInputStillParses) {
  const nlohmann::json header = {
      {"version", 2},
      {"n_inputs", 1},
      {"n_outputs", 1},
      {"kernels",
       {{{"subdir", "spyreCodeDir_0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
      {"inputs", {{{"kind", "kv_state"}, {"state_out", 0}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  ASSERT_EQ(out.input_kinds.size(), 1u);
  EXPECT_EQ(out.input_kinds[0], "kv_state");
  EXPECT_EQ(out.input_state_out[0], 0);
}

TEST(SpyrePayload, RejectsEmptyBucketsArray) {
  const nlohmann::json header = {{"version", 3},
                                 {"buckets", nlohmann::json::array()}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kInvalidSchema);
  EXPECT_FALSE(err.empty());
}

TEST(SpyrePayload, RejectsKernelWithMissingLaunchLayout) {
  const nlohmann::json header = {{"version", 2},
                                 {"kernels", {{{"subdir", "spyreCodeDir_0"}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kInvalidSchema);
  EXPECT_NE(err.find("launch_layout"), std::string::npos) << err;
}

TEST(SpyrePayload, RejectsKernelWithEmptyLaunchLayout) {
  const nlohmann::json header = {
      {"version", 2},
      {"kernels",
       {{{"subdir", "spyreCodeDir_0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}},
        {{"subdir", "spyreCodeDir_1"},
         {"launch_layout", nlohmann::json::array()}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  std::string err;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kInvalidSchema);
  EXPECT_NE(err.find("kernel 1"), std::string::npos) << err;
}

TEST(SpyrePayload, RejectsV3BucketKernelWithoutLaunchLayout) {
  const nlohmann::json header = {
      {"version", 3},
      {"buckets",
       {{{"shape_key", {128}},
         {"subdir_prefix", "bucket0"},
         {"kernels", {{{"subdir", "spyreCodeDir_0"}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  EXPECT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kInvalidSchema);
}

// ---- v2 wire format (Phase 6b) --------------------------------------------
// The parser accepts v2 multi-kernel payloads and still fills the legacy
// single-kernel fields from kernels[0], while a v1 payload is mirrored into a
// one-element kernels vector so callers iterate ``kernels`` regardless of
// version.

TEST(SpyrePayload, ParsesV2TwoKernels) {
  // Two kernels launched sequentially, sharing one top-level scratch pool.
  const nlohmann::json header = {
      {"version", 2},
      {"n_inputs", 5},
      {"n_outputs", 1},
      {"pool_sticks", 2},
      {"kernels",
       {{{"subdir", "spyreCodeDir_0"},
         {"launch_layout",
          {{{"role", "pool"}},
           {{"role", "input"}, {"index", 0}},
           {{"role", "intermediate"}, {"buffer", 0}, {"produces", true}}}}},
        {{"subdir", "spyreCodeDir_1"},
         {"launch_layout",
          {{{"role", "pool"}},
           {{"role", "intermediate"}, {"buffer", 0}, {"produces", false}},
           {{"role", "output"}, {"index", 0}}}}}}}};
  const auto bytes = pack_payload(header, {0x01, 0x02});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  EXPECT_EQ(out.version, 2);
  EXPECT_EQ(out.buckets[0].pool_sticks, 2);
  ASSERT_EQ(out.buckets[0].kernels.size(), 2u);
  EXPECT_EQ(out.buckets[0].kernels[0].subdir, "spyreCodeDir_0");
  EXPECT_EQ(out.buckets[0].kernels[1].subdir, "spyreCodeDir_1");
  ASSERT_EQ(out.buckets[0].kernels[0].launch_layout.size(), 3u);
  ASSERT_EQ(out.buckets[0].kernels[1].launch_layout.size(), 3u);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[0].role,
            LaunchEntry::kPool);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[1].role,
            LaunchEntry::kInput);
  EXPECT_EQ(out.buckets[0].kernels[1].launch_layout[2].role,
            LaunchEntry::kOutput);
}

TEST(SpyrePayload, ParsesV2SingleKernelNoPoolNoIntermediates) {
  // What the AOT now emits for every single-kernel static compile: one
  // kernels[] entry, no pool, no intermediates. It must yield exactly one
  // bucket holding one kernel, and stay launchable.
  const nlohmann::json header = {{"version", 2},
                                 {"n_inputs", 2},
                                 {"n_outputs", 1},
                                 {"pool_sticks", 0},
                                 {"intermediates", nlohmann::json::array()},
                                 {"kernels",
                                  {{{"subdir", "spyreCodeDir_0"},
                                    {"launch_layout",
                                     {{{"role", "input"}, {"index", 0}},
                                      {{"role", "input"}, {"index", 1}},
                                      {{"role", "output"}, {"index", 0}}}}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  EXPECT_EQ(out.version, 2);
  EXPECT_EQ(out.buckets[0].pool_sticks, 0);
  EXPECT_TRUE(out.buckets[0].intermediates.empty());
  ASSERT_EQ(out.buckets[0].kernels.size(), 1u);
  EXPECT_EQ(out.buckets[0].kernels[0].subdir, "spyreCodeDir_0");
  ASSERT_EQ(out.buckets[0].kernels[0].launch_layout.size(), 3u);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[0].role,
            LaunchEntry::kInput);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[0].index, 0);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[1].index, 1);
  EXPECT_EQ(out.buckets[0].kernels[0].launch_layout[2].role,
            LaunchEntry::kOutput);
  ASSERT_EQ(out.buckets.size(), 1u);
  EXPECT_EQ(out.buckets[0].kernels.size(), 1u);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kOk) << reason;
}

TEST(SpyrePayload, ParsesIntermediateRoleProducerConsumer) {
  // The same buffer id is produced by kernel 0 and consumed by kernel 1 — this
  // is the launch-DAG edge (the fix for Phase 6a's mislabeling of a consumed
  // intermediate as an output).
  const nlohmann::json header = {
      {"version", 2},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 0}, {"produces", true}}}}},
        {{"subdir", "k1"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 0}, {"produces", false}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  ASSERT_EQ(out.buckets[0].kernels.size(), 2u);
  const auto& producer = out.buckets[0].kernels[0].launch_layout.at(0);
  EXPECT_EQ(producer.role, LaunchEntry::kIntermediate);
  EXPECT_EQ(producer.buffer, 0);
  EXPECT_TRUE(producer.produces);

  const auto& consumer = out.buckets[0].kernels[1].launch_layout.at(0);
  EXPECT_EQ(consumer.role, LaunchEntry::kIntermediate);
  EXPECT_EQ(consumer.buffer, 0);
  EXPECT_FALSE(consumer.produces);
}

TEST(SpyrePayload, ParsesIntermediatesTable) {
  // id/shape/dtype/resident/delegate_output round-trip, incl. a resident:false
  // host-round-trip edge and a delegate_output one.
  const nlohmann::json header = {
      {"version", 2},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
      {"intermediates",
       {{{"id", 0},
         {"shape", {64, 128}},
         {"dtype", "float16"},
         {"resident", false},
         {"delegate_output", false}},
        {{"id", 1},
         {"shape", {32}},
         {"dtype", "float32"},
         {"resident", true},
         {"delegate_output", true}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  ASSERT_EQ(out.buckets[0].intermediates.size(), 2u);
  const auto& a = out.buckets[0].intermediates[0];
  EXPECT_EQ(a.id, 0);
  EXPECT_EQ(a.shape, (std::vector<std::int64_t>{64, 128}));
  EXPECT_EQ(a.dtype, "float16");
  EXPECT_FALSE(a.resident);
  EXPECT_FALSE(a.delegate_output);

  const auto& b = out.buckets[0].intermediates[1];
  EXPECT_EQ(b.id, 1);
  EXPECT_EQ(b.shape, (std::vector<std::int64_t>{32}));
  EXPECT_EQ(b.dtype, "float32");
  EXPECT_TRUE(b.resident);
  EXPECT_TRUE(b.delegate_output);
}

TEST(SpyrePayload, V2DefaultsWhenIntermediatesAbsent) {
  // A v2 payload with kernels but no intermediates table parses cleanly to an
  // empty intermediates vector — no throw.
  const nlohmann::json header = {
      {"version", 2},
      {"pool_sticks", 4},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}},
        {{"subdir", "k1"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  EXPECT_EQ(out.version, 2);
  EXPECT_EQ(out.buckets[0].pool_sticks, 4);
  EXPECT_EQ(out.buckets[0].kernels.size(), 2u);
  EXPECT_TRUE(out.buckets[0].intermediates.empty());
}

TEST(SpyrePayload, V2SynthesizesSingleEmptyPrefixBucket) {
  // A v2 payload yields exactly one bucket (empty prefix and shape_key) holding
  // all of its kernels in wire order, so callers iterate ``buckets`` uniformly.
  const nlohmann::json header = {
      {"version", 2},
      {"pool_sticks", 2},
      {"kernels",
       {{{"subdir", "firstDir"},
         {"launch_layout",
          {{{"role", "pool"}},
           {{"role", "input"}, {"index", 0}},
           {{"role", "output"}, {"index", 0}}}}},
        {{"subdir", "secondDir"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  EXPECT_EQ(out.version, 2);
  ASSERT_EQ(out.buckets.size(), 1u);
  const Bucket& b = out.buckets[0];
  EXPECT_TRUE(b.subdir_prefix.empty());
  EXPECT_TRUE(b.shape_key.empty());
  EXPECT_EQ(b.pool_sticks, 2);
  ASSERT_EQ(b.kernels.size(), 2u);
  EXPECT_EQ(b.kernels[0].subdir, "firstDir");
  EXPECT_EQ(b.kernels[1].subdir, "secondDir");
  ASSERT_EQ(b.kernels[0].launch_layout.size(), 3u);
  EXPECT_EQ(b.kernels[0].launch_layout[0].role, LaunchEntry::kPool);
  EXPECT_EQ(b.kernels[0].launch_layout[1].role, LaunchEntry::kInput);
  EXPECT_EQ(b.kernels[0].launch_layout[2].role, LaunchEntry::kOutput);
}

// ---- launchability gate (Phase 6e) ----------------------------------------
// classify_launch decides whether execute() can actually run a parsed DAG:
// single kernel is always launchable; an N>1 DAG is launchable only if every
// inter-kernel edge names a present intermediate (id==index) that is resident
// (a device handoff) rather than a host-round-trip re-materialization.

TEST(SpyrePayload, ClassifyLaunchSingleKernelIsOk) {
  // A single-kernel v2 payload has no inter-kernel edges — always launchable.
  const nlohmann::json header = {{"version", 2},
                                 {"n_inputs", 2},
                                 {"n_outputs", 1},
                                 {"pool_sticks", 64},
                                 {"kernels",
                                  {{{"subdir", "spyreCodeDir_0"},
                                    {"launch_layout",
                                     {{{"role", "pool"}, {"sticks", 64}},
                                      {{"role", "input"}, {"index", 0}},
                                      {{"role", "output"}, {"index", 0}}}}}}}};
  const auto bytes = pack_payload(header, {0x01});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  ASSERT_EQ(out.buckets.size(), 1u);
  ASSERT_EQ(out.buckets[0].kernels.size(), 1u);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kOk) << reason;
}

TEST(SpyrePayload, ClassifyLaunchResidentDagIsOk) {
  // Two kernels threading one resident:true intermediate producer->consumer —
  // a genuine device handoff, so the DAG is launchable.
  const nlohmann::json header = {
      {"version", 2},
      {"pool_sticks", 2},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout",
          {{{"role", "input"}, {"index", 0}},
           {{"role", "intermediate"}, {"buffer", 0}, {"produces", true}}}}},
        {{"subdir", "k1"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 0}, {"produces", false}},
           {{"role", "output"}, {"index", 0}}}}}}},
      {"intermediates",
       {{{"id", 0},
         {"shape", {64, 128}},
         {"dtype", "float16"},
         {"resident", true},
         {"delegate_output", false}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kOk) << reason;
}

TEST(SpyrePayload, ClassifyLaunchNonResidentEdgeIsRejected) {
  // A non-resident inter-kernel edge: execute() cannot reproduce it.
  const nlohmann::json header = {
      {"version", 2},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 0}, {"produces", true}}}}},
        {{"subdir", "k1"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 0}, {"produces", false}}}}}}},
      {"intermediates",
       {{{"id", 0},
         {"shape", {64, 128}},
         {"dtype", "float16"},
         {"resident", false},
         {"delegate_output", false}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kNonResidentEdge);
  EXPECT_FALSE(reason.empty());
}

TEST(SpyrePayload, ClassifyLaunchMissingIntermediateIsRejected) {
  // A kIntermediate slot names buffer id 3, but the intermediates table has
  // only one entry (id 0) — out of range, so the DAG is not launchable.
  const nlohmann::json header = {
      {"version", 2},
      {"kernels",
       {{{"subdir", "k0"},
         {"launch_layout",
          {{{"role", "intermediate"}, {"buffer", 3}, {"produces", true}}}}},
        {{"subdir", "k1"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
      {"intermediates",
       {{{"id", 0},
         {"shape", {8}},
         {"dtype", "float16"},
         {"resident", true},
         {"delegate_output", false}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kMissingIntermediate);
  EXPECT_FALSE(reason.empty());
}

// ---- v3 wire format (Phase 7 Track 2, static bucketing) --------------------
// A "buckets" array carries one complete v2 kernel-set per compiled shape. The
// parser fills a ``buckets`` vector, records ``dynamic_input_dims``, and
// mirrors bucket[0] into the legacy top-level fields. v1/v2 payloads synthesize
// exactly one bucket so execute() can always iterate ``buckets`` uniformly.

TEST(SpyrePayload, ParsesV3TwoBuckets) {
  // Two buckets (M=128 and M=256), each a single-kernel v2 set with its own
  // subdir_prefix. A top-level dynamic_input_dims names input 1, axis 0.
  const nlohmann::json header = {
      {"version", 3},
      {"n_inputs", 2},
      {"n_outputs", 1},
      {"dynamic_input_dims", {{1, 0}}},
      {"buckets",
       {{{"shape_key", {128}},
         {"subdir_prefix", "bucket0"},
         {"pool_sticks", 2},
         {"kernels",
          {{{"subdir", "spyreCodeDir_0"},
            {"launch_layout",
             {{{"role", "pool"}},
              {{"role", "input"}, {"index", 1}},
              {{"role", "output"}, {"index", 0}}}}}}}},
        {{"shape_key", {256}},
         {"subdir_prefix", "bucket1"},
         {"pool_sticks", 4},
         {"kernels",
          {{{"subdir", "spyreCodeDir_0"},
            {"launch_layout",
             {{{"role", "pool"}},
              {{"role", "input"}, {"index", 1}},
              {{"role", "output"}, {"index", 0}}}}}}}}}}};
  const auto bytes = pack_payload(header, {0x01, 0x02});
  ParsedPayload out;
  std::string err;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out, &err),
            PayloadStatus::kOk)
      << err;

  EXPECT_EQ(out.version, 3);
  ASSERT_EQ(out.buckets.size(), 2u);

  EXPECT_EQ(out.buckets[0].shape_key, (std::vector<std::int64_t>{128}));
  EXPECT_EQ(out.buckets[0].subdir_prefix, "bucket0");
  EXPECT_EQ(out.buckets[0].pool_sticks, 2);
  ASSERT_EQ(out.buckets[0].kernels.size(), 1u);
  EXPECT_EQ(out.buckets[0].kernels[0].subdir, "spyreCodeDir_0");

  EXPECT_EQ(out.buckets[1].shape_key, (std::vector<std::int64_t>{256}));
  EXPECT_EQ(out.buckets[1].subdir_prefix, "bucket1");
  EXPECT_EQ(out.buckets[1].pool_sticks, 4);

  ASSERT_EQ(out.dynamic_input_dims.size(), 1u);
  EXPECT_EQ(out.dynamic_input_dims[0].first, 1u);
  EXPECT_EQ(out.dynamic_input_dims[0].second, 0);

  // tar_data still splits right after the JSON.
  ASSERT_EQ(out.tar_size, 2u);
}

TEST(SpyrePayload, V3BucketsKeepTheirOwnKernelSets) {
  // Each bucket carries its own pool, kernels and intermediates.
  const nlohmann::json header = {
      {"version", 3},
      {"buckets",
       {{{"shape_key", {128}},
         {"subdir_prefix", "bucket0"},
         {"pool_sticks", 7},
         {"kernels",
          {{{"subdir", "firstDir"},
            {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
         {"intermediates",
          {{{"id", 0}, {"resident", true}, {"dtype", "float16"}}}}},
        {{"shape_key", {256}},
         {"subdir_prefix", "bucket1"},
         {"pool_sticks", 9},
         {"kernels",
          {{{"subdir", "otherDir"},
            {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);

  ASSERT_EQ(out.buckets.size(), 2u);
  const Bucket& b0 = out.buckets[0];
  EXPECT_EQ(b0.pool_sticks, 7);
  ASSERT_EQ(b0.kernels.size(), 1u);
  EXPECT_EQ(b0.kernels[0].subdir, "firstDir");
  ASSERT_EQ(b0.intermediates.size(), 1u);
  EXPECT_TRUE(b0.intermediates[0].resident);
  const Bucket& b1 = out.buckets[1];
  EXPECT_EQ(b1.pool_sticks, 9);
  ASSERT_EQ(b1.kernels.size(), 1u);
  EXPECT_EQ(b1.kernels[0].subdir, "otherDir");
  EXPECT_TRUE(b1.intermediates.empty());
}

TEST(SpyrePayload, ClassifyLaunchV3AllBucketsLaunchable) {
  // Both buckets are single-kernel — each trivially launchable, so the whole
  // bucketed payload is launchable.
  const nlohmann::json header = {
      {"version", 3},
      {"dynamic_input_dims", {{0, 0}}},
      {"buckets",
       {{{"shape_key", {128}},
         {"subdir_prefix", "bucket0"},
         {"kernels",
          {{{"subdir", "spyreCodeDir_0"},
            {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}},
        {{"shape_key", {256}},
         {"subdir_prefix", "bucket1"},
         {"kernels",
          {{{"subdir", "spyreCodeDir_0"},
            {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kOk) << reason;
}

TEST(SpyrePayload, ClassifyLaunchV3RejectsIfAnyBucketNonLaunchable) {
  // Bucket 0 is a clean resident DAG, but bucket 1 threads a non-resident edge.
  // Since execute() may select bucket 1 at runtime, the payload is rejected and
  // the reason names the offending bucket index.
  const nlohmann::json good_kernels = {
      {{"subdir", "k0"},
       {"launch_layout",
        {{{"role", "intermediate"}, {"buffer", 0}, {"produces", true}}}}},
      {{"subdir", "k1"},
       {"launch_layout",
        {{{"role", "intermediate"}, {"buffer", 0}, {"produces", false}}}}}};
  const nlohmann::json header = {
      {"version", 3},
      {"dynamic_input_dims", {{0, 0}}},
      {"buckets",
       {{{"shape_key", {128}},
         {"subdir_prefix", "bucket0"},
         {"kernels", good_kernels},
         {"intermediates",
          {{{"id", 0}, {"resident", true}, {"dtype", "float16"}}}}},
        {{"shape_key", {256}},
         {"subdir_prefix", "bucket1"},
         {"kernels", good_kernels},
         {"intermediates",
          {{{"id", 0}, {"resident", false}, {"dtype", "float16"}}}}}}}};
  const auto bytes = pack_payload(header, {});
  ParsedPayload out;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &out),
            PayloadStatus::kOk);
  std::string reason;
  EXPECT_EQ(classify_launch(out, &reason), Launchability::kNonResidentEdge);
  EXPECT_NE(reason.find("bucket 1"), std::string::npos) << reason;
}

}  // namespace
}  // namespace spyre
}  // namespace backends
}  // namespace executorch
