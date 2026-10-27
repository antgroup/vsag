// Copyright 2024-present the vsag project
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

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "datacell/flatten_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "simd/fp16_simd.h"
#include "vsag/vsag.h"

namespace vsag {

TEST_CASE("HGraph half input reconstructs BF16 base codes without narrowing",
          "[ft][half_storage_preservation]") {
    constexpr int64_t dim = 32;
    IndexCommonParam common;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common.dim_ = dim;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.data_type_ = DataTypes::DATA_TYPE_FP16;
    const auto params = JsonType::Parse(
        R"({"base_quantization_type":"bf16","use_reorder":false,"store_raw_vector":false})");
    HGraph index(HGraph::CheckAndMappingExternalParam(params, common), common);
    std::vector<uint16_t> values(dim, 0);
    values[0] = generic::FloatToFP16(65504.0F);
    int64_t label = 22;
    auto base = Dataset::Make()->Owner(false)->Dim(dim)->NumElements(1)->Ids(&label);
    base->Float16Vectors(values.data());
    REQUIRE(index.Build(base).empty());

    // BF16-only HGraph does not advertise public raw-vector export. Exercise the
    // inner reconstruction API directly, with no precise/raw storage to mask the base codes.
    // A single inserted vector has inner id 0, independently of its external label.
    std::vector<float> reconstructed(dim, 0);
    index.GetVectorByInnerId(0, reconstructed.data());
    REQUIRE(std::isfinite(reconstructed[0]));
    REQUIRE(reconstructed[0] == 65536.0F);
    for (int64_t d = 1; d < dim; ++d) {
        REQUIRE(reconstructed[d] == 0.0F);
    }
}

TEST_CASE("BruteForce half input exports BF16 reconstruction before and after removal",
          "[ft][half_storage_preservation]") {
    constexpr int64_t dim = 32;
    const auto params = R"({"dtype":"float16","metric_type":"l2","dim":32,"index_param":{
        "base_quantization_type":"bf16"}})";
    auto made = Factory::CreateIndex("brute_force", params);
    REQUIRE(made.has_value());
    auto index = made.value();
    std::vector<uint16_t> values(2 * dim, 0);
    values[dim] = generic::FloatToFP16(65504.0F);
    int64_t labels[] = {11, 22};
    auto base = Dataset::Make()->Owner(false)->Dim(dim)->NumElements(2)->Ids(labels);
    base->Float16Vectors(values.data());
    REQUIRE(index->Build(base).has_value());
    // BruteForce exposes stored reconstructions through GetDataByIds;
    // GetRawVectorByIds is feature-gated to FP32 storage.
    auto exported = index->GetDataByIds(labels + 1, 1);
    REQUIRE(exported.has_value());
    REQUIRE(exported.value()->GetFloat32Vectors() != nullptr);
    REQUIRE(std::isfinite(exported.value()->GetFloat32Vectors()[0]));
    REQUIRE(exported.value()->GetFloat32Vectors()[0] == 65536.0F);

    auto query = Dataset::Make()->Owner(false)->Dim(dim)->NumElements(1);
    query->Float16Vectors(values.data() + dim);
    auto before = index->KnnSearch(query, 2, "{}");
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetIds()[0] == 22);
    const auto distance = before.value()->GetDistances()[0];
    REQUIRE(std::isfinite(distance));
    // The BF16 computer quantizes the query to the same stored code.
    REQUIRE(distance == 0.0F);
    auto removed = index->Remove(11, RemoveMode::FORCE_REMOVE);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() == 1);
    exported = index->GetDataByIds(labels + 1, 1);
    REQUIRE(exported.has_value());
    REQUIRE(exported.value()->GetFloat32Vectors() != nullptr);
    REQUIRE(exported.value()->GetFloat32Vectors()[0] == 65536.0F);
    auto after = index->KnnSearch(query, 1, "{}");
    REQUIRE(after.has_value());
    REQUIRE(after.value()->GetIds()[0] == 22);
    REQUIRE(after.value()->GetDistances()[0] == distance);
}

TEST_CASE("Half storage move preserves encoded bytes and native decode contract",
          "[ft][half_storage_preservation]") {
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(JsonType::Parse(R"({"io_params":{"type":"memory_io"},
                                      "quantization_params":{"type":"bf16"}})"));
    IndexCommonParam common;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common.dim_ = 1;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    common.data_type_ = DataTypes::DATA_TYPE_FP16;
    auto cell = FlattenInterface::MakeInstance(param, common);
    uint16_t input[] = {0, generic::FloatToFP16(65504.0F)};
    cell->Train(input, 2);
    cell->BatchInsertVector(input, 2);
    std::vector<uint8_t> before(cell->code_size_);
    std::vector<uint8_t> after(cell->code_size_);
    REQUIRE(cell->GetCodesById(1, before.data()));
    float decoded = 0;
    REQUIRE(cell->DecodeToFloat(before.data(), &decoded));
    REQUIRE(decoded == 65536.0F);
    alignas(float) uint16_t native = 0;
    REQUIRE(cell->Decode(before.data(), reinterpret_cast<float*>(&native)));
    REQUIRE(std::isinf(generic::FP16ToFloat(native)));
    cell->Move(1, 0);
    REQUIRE(cell->GetCodesById(0, after.data()));
    REQUIRE(before == after);
}

TEST_CASE("BruteForce rejects unsupported sparse and int8 input at creation",
          "[ft][half_storage_preservation]") {
    for (const auto& dtype : {std::string("sparse"), std::string("int8")}) {
        CAPTURE(dtype);
        const auto params =
            "{\"dtype\":\"" + dtype + "\",\"metric_type\":\"ip\",\"dim\":32,\"index_param\":{}}";
        auto made = Factory::CreateIndex("brute_force", params);
        REQUIRE_FALSE(made.has_value());
        REQUIRE(made.error().type == ErrorType::INVALID_ARGUMENT);
    }
}

}  // namespace vsag
