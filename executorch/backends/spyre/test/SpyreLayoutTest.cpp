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
// Host-only gtest for the ET-side layout schema (runtime/SpyreLayout.h). C++
// counterpart of test/test_spyre_layout.py: kGoldenLayoutJson below is the
// exact string that test pins as the Python serializer's output, so both
// languages are checked against the same wire bytes. No Spyre device, torch, or
// ExecuTorch.

#include <gtest/gtest.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "runtime/SpyreLayout.h"
#include "runtime/SpyrePayload.h"

namespace executorch {
namespace backends {
namespace spyre {
namespace {

// MUST stay identical to GOLDEN_JSON in test_spyre_layout.py.
const char* const kGoldenLayoutJson =
    R"({"device_size": [2, 64, 64], "stride_map": [64, 1, 4096], )"
    R"("device_dtype": 3, "element_arrangement": 1, )"
    R"("valid_elements": [[[0, 1], 100], [[2], 7]], "strides": [128, 64, 1]})";

TEST(SpyreLayout, SchemaFieldNamesAndOrderAreFrozen) {
  const std::vector<std::string> fields(std::begin(kLayoutFields),
                                        std::end(kLayoutFields));
  EXPECT_EQ(fields, (std::vector<std::string>{
                        "device_size", "stride_map", "device_dtype",
                        "element_arrangement", "valid_elements", "strides"}));
}

TEST(SpyreLayout, ParsesGoldenPythonOutput) {
  const OutputLayout ol =
      parse_device_layout(nlohmann::json::parse(kGoldenLayoutJson));
  EXPECT_TRUE(ol.present);
  EXPECT_EQ(ol.device_size, (std::vector<std::int64_t>{2, 64, 64}));
  EXPECT_EQ(ol.stride_map, (std::vector<std::int64_t>{64, 1, 4096}));
  EXPECT_EQ(ol.device_dtype, 3);
  EXPECT_EQ(ol.element_arrangement, 1);
  ASSERT_EQ(ol.valid_elements.size(), 2u);
  EXPECT_EQ(ol.valid_elements[0].first, (std::vector<std::int64_t>{0, 1}));
  EXPECT_EQ(ol.valid_elements[0].second, 100);
  EXPECT_EQ(ol.valid_elements[1].first, (std::vector<std::int64_t>{2}));
  EXPECT_EQ(ol.valid_elements[1].second, 7);
  EXPECT_EQ(ol.strides, (std::vector<std::int64_t>{128, 64, 1}));
}

TEST(SpyreLayout, SchemaRoundTripsJsonToStructToJson) {
  const nlohmann::json in = nlohmann::json::parse(kGoldenLayoutJson);
  const nlohmann::json out = layout_to_json(parse_device_layout(in));
  EXPECT_EQ(out, in);
  // Every schema field is emitted, nothing else.
  EXPECT_EQ(out.size(), std::size(kLayoutFields));
  for (const char* f : kLayoutFields) EXPECT_TRUE(out.contains(f)) << f;
}

TEST(SpyreLayout, StructRoundTripsStructToJsonToStruct) {
  OutputLayout a;
  a.present = true;
  a.device_size = {4, 8};
  a.stride_map = {8, 1};
  a.device_dtype = 2;
  a.element_arrangement = 0;
  a.valid_elements = {{{0}, 3}, {{1, 2}, 5}};
  a.strides = {8, 1};
  const OutputLayout b = parse_device_layout(layout_to_json(a));
  EXPECT_TRUE(b.present);
  EXPECT_EQ(b.device_size, a.device_size);
  EXPECT_EQ(b.stride_map, a.stride_map);
  EXPECT_EQ(b.device_dtype, a.device_dtype);
  EXPECT_EQ(b.element_arrangement, a.element_arrangement);
  EXPECT_EQ(b.valid_elements, a.valid_elements);
  EXPECT_EQ(b.strides, a.strides);
}

TEST(SpyreLayout, OptionalFieldsDefaultWhenAbsent) {
  // Only device_size / stride_map are mandatory (dl.at); the rest default.
  const OutputLayout ol = parse_device_layout(
      nlohmann::json::parse(R"({"device_size":[1],"stride_map":[1]})"));
  EXPECT_TRUE(ol.present);
  EXPECT_EQ(ol.device_dtype, 0);
  EXPECT_EQ(ol.element_arrangement, 0);
  EXPECT_TRUE(ol.valid_elements.empty());
  EXPECT_TRUE(ol.strides.empty());
}

TEST(SpyreLayout, MissingMandatoryFieldThrows) {
  EXPECT_ANY_THROW(
      parse_device_layout(nlohmann::json::parse(R"({"stride_map":[1]})")));
}

TEST(SpyreLayout, DefaultLayoutIsNotPresent) {
  EXPECT_FALSE(OutputLayout{}.present);
  EXPECT_FALSE(InputLayout{}.present);
}

TEST(SpyreLayout, ResolveHostStrides) {
  OutputLayout ol;
  // rank matches -> payload strides win
  ol.strides = {128, 16, 1};
  EXPECT_EQ(resolve_host_strides(ol, {4, 8, 16}),
            (std::vector<std::int64_t>{128, 16, 1}));
  // absent -> row-major
  ol.strides = {};
  EXPECT_EQ(resolve_host_strides(ol, {4, 8, 16}),
            (std::vector<std::int64_t>{128, 16, 1}));
  // rank mismatch -> row-major
  ol.strides = {1, 1};
  EXPECT_EQ(resolve_host_strides(ol, {4, 8, 16}),
            (std::vector<std::int64_t>{128, 16, 1}));
  // zero dim clamps to 1
  ol.strides = {};
  EXPECT_EQ(resolve_host_strides(ol, {2, 0, 3}),
            (std::vector<std::int64_t>{3, 3, 1}));
  EXPECT_TRUE(resolve_host_strides(ol, {}).empty());
  EXPECT_EQ(resolve_host_strides(ol, {5}), (std::vector<std::int64_t>{1}));
}

// The payload parser must consume the layout through the shared schema header
// exactly as before (input + output positions).
TEST(SpyreLayout, PayloadParserUsesSharedSchemaForInputsAndOutputs) {
  nlohmann::json layout = nlohmann::json::parse(kGoldenLayoutJson);
  nlohmann::json header = {
      {"version", 2},
      {"n_inputs", 1},
      {"n_outputs", 1},
      {"kernels",
       {{{"subdir", "spyreCodeDir_0"},
         {"launch_layout", {{{"role", "output"}, {"index", 0}}}}}}},
      {"inputs", {{{"kind", "kv_state"}, {"device_layout", layout}}}},
      {"outputs", {{{"shape", {1}}, {"device_layout", layout}}}}};
  const std::string json = header.dump();
  const std::uint32_t len = static_cast<std::uint32_t>(json.size());
  std::vector<std::uint8_t> bytes(sizeof(len));
  std::memcpy(bytes.data(), &len, sizeof(len));
  bytes.insert(bytes.end(), json.begin(), json.end());

  ParsedPayload p;
  ASSERT_EQ(parse_payload(bytes.data(), bytes.size(), &p), PayloadStatus::kOk);
  ASSERT_EQ(p.input_layouts.size(), 1u);
  EXPECT_TRUE(p.input_layouts[0].present);
  EXPECT_EQ(layout_to_json(p.input_layouts[0]), layout);
  ASSERT_EQ(p.buckets.size(), 1u);
  ASSERT_EQ(p.buckets[0].output_layouts.size(), 1u);
  EXPECT_EQ(layout_to_json(p.buckets[0].output_layouts[0]), layout);
}

}  // namespace
}  // namespace spyre
}  // namespace backends
}  // namespace executorch
