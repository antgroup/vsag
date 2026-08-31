
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

#include "algorithm/pyramid.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"

namespace {

constexpr int64_t PYRAMID_RAW_VECTOR_TEST_DIM = 4;

std::shared_ptr<vsag::Index>
MakePyramidRawVectorTestIndex(const std::string& graph_type,
                              bool use_reorder,
                              vsag::MetricType metric,
                              const std::string& base_quantization_type = "fp32",
                              const std::string& precise_quantization_type = "fp32",
                              bool store_raw_vector = false) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = PYRAMID_RAW_VECTOR_TEST_DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = metric;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "precise_quantization_type": "fp32",
        "use_reorder": false,
        "max_degree": 4,
        "ef_construction": 8,
        "graph_type": "nsw",
        "no_build_levels": [0, 1],
        "index_min_size": 0,
        "store_raw_vector": false
    })");
    external_param[vsag::PYRAMID_GRAPH_TYPE].SetString(graph_type);
    external_param[vsag::PYRAMID_USE_REORDER].SetBool(use_reorder);
    external_param[vsag::PYRAMID_BASE_QUANTIZATION_TYPE].SetString(base_quantization_type);
    external_param[vsag::PYRAMID_PRECISE_QUANTIZATION_TYPE].SetString(precise_quantization_type);
    external_param[vsag::STORE_RAW_VECTOR].SetBool(store_raw_vector);
    return std::make_shared<vsag::IndexImpl<vsag::Pyramid>>(external_param, common_param);
}

vsag::DatasetPtr
MakePyramidRawVectorTestDataset(const float* vectors,
                                const int64_t* ids,
                                const std::string* paths,
                                int64_t count) {
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(PYRAMID_RAW_VECTOR_TEST_DIM)
        ->Float32Vectors(vectors)
        ->Ids(ids)
        ->Paths(paths)
        ->Owner(false);
}

void
RequireRawVectors(const vsag::IndexPtr& index,
                  const int64_t* ids,
                  int64_t count,
                  const float* expected) {
    auto result = index->GetRawVectorByIds(ids, count);
    REQUIRE(result.has_value());
    REQUIRE(std::equal(result.value()->GetFloat32Vectors(),
                       result.value()->GetFloat32Vectors() + count * PYRAMID_RAW_VECTOR_TEST_DIM,
                       expected));
}

void
RequireStoredRawVectors(const vsag::IndexPtr& index,
                        const int64_t* ids,
                        int64_t count,
                        const float* expected) {
    auto data_result = index->GetDataByIds(ids, count);
    REQUIRE(data_result.has_value());
    REQUIRE(std::equal(data_result.value()->GetIds(), data_result.value()->GetIds() + count, ids));
    REQUIRE(
        std::equal(data_result.value()->GetFloat32Vectors(),
                   data_result.value()->GetFloat32Vectors() + count * PYRAMID_RAW_VECTOR_TEST_DIM,
                   expected));

    auto flagged_result =
        index->GetDataByIdsWithFlag(ids, count, DATA_FLAG_ID | DATA_FLAG_FLOAT32_VECTOR);
    REQUIRE(flagged_result.has_value());
    REQUIRE(std::equal(
        flagged_result.value()->GetFloat32Vectors(),
        flagged_result.value()->GetFloat32Vectors() + count * PYRAMID_RAW_VECTOR_TEST_DIM,
        expected));
}

}  // namespace

#include "impl/allocator/safe_allocator.h"
#include "vsag_exception.h"

TEST_CASE("Split function tests", "[ut][pyramid]") {
    SECTION("Empty input string") {
        auto result = vsag::split("", ',');
        REQUIRE(result.empty());
    }

    SECTION("No delimiters in string") {
        auto result = vsag::split("hello", ',');
        REQUIRE(result == std::vector<std::string>{"hello"});
    }

    SECTION("Delimiter at start") {
        auto result = vsag::split(",hello,world", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Delimiter at end") {
        auto result = vsag::split("hello,world,", ',');
        REQUIRE(result == std::vector<std::string>{"hello", "world"});
    }

    SECTION("Multiple consecutive delimiters") {
        auto result = vsag::split("a,,b,,,c", ',');
        REQUIRE(result == std::vector<std::string>{"a", "b", "c"});
    }

    SECTION("Normal split with multiple tokens") {
        auto result = vsag::split("one,two,three", ',');
        REQUIRE(result == std::vector<std::string>{"one", "two", "three"});
    }

    SECTION("All delimiters") {
        auto result = vsag::split(",,,", ',');
        REQUIRE(result.empty());
    }

    SECTION("Mixed delimiters and spaces") {
        auto result = vsag::split("  , hello,  world  ", ',');
        REQUIRE(result == std::vector<std::string>{"  ", " hello", "  world  "});
    }
}

TEST_CASE("Pyramid ExportModel rejects inconsistent reorder configuration",
          "[ut][Pyramid][ExportModel]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 4;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();

    auto external_param = vsag::JsonType::Parse(R"({"use_reorder": false})");
    auto index_param = vsag::Pyramid::CheckAndMappingExternalParam(external_param, common_param);
    auto index = std::make_shared<vsag::Pyramid>(index_param, common_param);
    REQUIRE_FALSE(index->use_reorder_);

    index->use_reorder_ = true;
    try {
        index->ExportModel(common_param);
        FAIL("ExportModel should reject a mismatched reorder configuration");
    } catch (const vsag::VsagException& exception) {
        REQUIRE(exception.error_.type == vsag::ErrorType::INTERNAL_ERROR);
        REQUIRE(std::string(exception.what()) ==
                "Export model's pyramid reorder config mismatched");
    }
}

TEST_CASE("Pyramid raw-vector capability does not imply stored raw data",
          "[ut][pyramid][raw_vector]") {
    const auto graph_type = GENERATE(std::string("nsw"), std::string("odescent"));
    const auto use_reorder = GENERATE(false, true);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    CAPTURE(graph_type, use_reorder, metric);

    const auto base_quantization_type = use_reorder ? "sq8" : "fp32";
    auto index =
        MakePyramidRawVectorTestIndex(graph_type, use_reorder, metric, base_quantization_type);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));

    constexpr int64_t count = 3;
    std::array<float, count* PYRAMID_RAW_VECTOR_TEST_DIM> vectors = {
        0.125F, 0.25F, 0.5F, 1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    std::array<int64_t, count> ids = {10, 42, 1001};
    std::array<std::string, count> paths = {"tenant/a", "tenant/b", "tenant/c"};
    auto dataset = MakePyramidRawVectorTestDataset(vectors.data(), ids.data(), paths.data(), count);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::array<int64_t, 2> requested_ids = {1001, 10};
    std::array<float, 2 * PYRAMID_RAW_VECTOR_TEST_DIM> requested_vectors = {
        4.0F, 3.0F, 2.0F, 1.0F, 0.125F, 0.25F, 0.5F, 1.0F};
    RequireRawVectors(index, requested_ids.data(), requested_ids.size(), requested_vectors.data());

    auto data_result = index->GetDataByIds(requested_ids.data(), requested_ids.size());
    REQUIRE(data_result.has_value());
    REQUIRE(std::equal(data_result.value()->GetIds(),
                       data_result.value()->GetIds() + requested_ids.size(),
                       requested_ids.begin()));
    REQUIRE(data_result.value()->GetFloat32Vectors() == nullptr);
    auto unavailable_raw = index->GetDataByIdsWithFlag(
        requested_ids.data(), requested_ids.size(), DATA_FLAG_ID | DATA_FLAG_FLOAT32_VECTOR);
    REQUIRE_FALSE(unavailable_raw.has_value());
    REQUIRE(unavailable_raw.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(unavailable_raw.error().message == "has_raw_vector_ is false");

    std::array<float, PYRAMID_RAW_VECTOR_TEST_DIM> added_vector = {8.0F, 6.0F, 4.0F, 2.0F};
    std::array<int64_t, 1> added_id = {77};
    std::array<std::string, 1> added_path = {"tenant/d"};
    auto add_result = index->Add(MakePyramidRawVectorTestDataset(
        added_vector.data(), added_id.data(), added_path.data(), added_id.size()));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    RequireRawVectors(index, added_id.data(), added_id.size(), added_vector.data());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored =
        MakePyramidRawVectorTestIndex(graph_type, use_reorder, metric, base_quantization_type);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    RequireRawVectors(restored, added_id.data(), added_id.size(), added_vector.data());
    auto restored_data = restored->GetDataByIds(added_id.data(), added_id.size());
    REQUIRE(restored_data.has_value());
    REQUIRE(restored_data.value()->GetFloat32Vectors() == nullptr);
}

TEST_CASE("Pyramid stores exact raw vectors when requested", "[ut][pyramid][raw_vector]") {
    const auto graph_type = GENERATE(std::string("nsw"), std::string("odescent"));
    const auto use_reorder = GENERATE(false, true);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_COSINE);
    const auto base_quantization_type = GENERATE(std::string("fp32"), std::string("sq8"));
    CAPTURE(graph_type, use_reorder, metric, base_quantization_type);

    auto index = MakePyramidRawVectorTestIndex(
        graph_type, use_reorder, metric, base_quantization_type, "fp32", true);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));

    constexpr int64_t count = 3;
    std::array<float, count* PYRAMID_RAW_VECTOR_TEST_DIM> vectors = {
        0.125F, 0.25F, 0.5F, 1.0F, 1.0F, 2.0F, 3.0F, 4.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    std::array<int64_t, count> ids = {10, 42, 1001};
    std::array<std::string, count> paths = {"tenant/a", "tenant/b", "tenant/c"};
    auto dataset = MakePyramidRawVectorTestDataset(vectors.data(), ids.data(), paths.data(), count);
    auto build_result = index->Build(dataset);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::array<int64_t, 2> requested_ids = {1001, 10};
    std::array<float, 2 * PYRAMID_RAW_VECTOR_TEST_DIM> requested_vectors = {
        4.0F, 3.0F, 2.0F, 1.0F, 0.125F, 0.25F, 0.5F, 1.0F};
    RequireRawVectors(index, requested_ids.data(), requested_ids.size(), requested_vectors.data());
    RequireStoredRawVectors(
        index, requested_ids.data(), requested_ids.size(), requested_vectors.data());

    std::array<float, PYRAMID_RAW_VECTOR_TEST_DIM> added_vector = {8.0F, 6.0F, 4.0F, 2.0F};
    std::array<int64_t, 1> added_id = {77};
    std::array<std::string, 1> added_path = {"tenant/d"};
    auto add_result = index->Add(MakePyramidRawVectorTestDataset(
        added_vector.data(), added_id.data(), added_path.data(), added_id.size()));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    RequireRawVectors(index, added_id.data(), added_id.size(), added_vector.data());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());
    auto restored = MakePyramidRawVectorTestIndex(
        graph_type, use_reorder, metric, base_quantization_type, "fp32", true);
    REQUIRE(restored->Deserialize(binary_set.value()).has_value());
    RequireRawVectors(
        restored, requested_ids.data(), requested_ids.size(), requested_vectors.data());
    RequireStoredRawVectors(
        restored, requested_ids.data(), requested_ids.size(), requested_vectors.data());

    const int64_t missing_id = -1;
    REQUIRE_FALSE(index->GetRawVectorByIds(&missing_id, 1).has_value());
}

TEST_CASE("Pyramid does not advertise unavailable raw vectors", "[ut][pyramid][raw_vector]") {
    const auto unsupported_index = GENERATE_COPY(
        MakePyramidRawVectorTestIndex("nsw", false, vsag::MetricType::METRIC_TYPE_L2SQR, "sq8"),
        MakePyramidRawVectorTestIndex("nsw", false, vsag::MetricType::METRIC_TYPE_COSINE, "fp32"));

    REQUIRE_FALSE(
        unsupported_index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS));
    const int64_t id = 1;
    REQUIRE_FALSE(unsupported_index->GetRawVectorByIds(&id, 1).has_value());
}

TEST_CASE("Pyramid ODescent build registers non-contiguous external ids",
          "[ut][pyramid][raw_vector]") {
    auto index = MakePyramidRawVectorTestIndex(
        "odescent", false, vsag::MetricType::METRIC_TYPE_L2SQR, "sq8", "fp32", true);
    std::array<float, 2 * PYRAMID_RAW_VECTOR_TEST_DIM> vectors = {
        0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F};
    std::array<int64_t, 2> ids = {42, 1001};
    std::array<std::string, 2> paths = {"tenant/a", "tenant/b"};

    auto build_result = index->Build(
        MakePyramidRawVectorTestDataset(vectors.data(), ids.data(), paths.data(), ids.size()));
    REQUIRE(build_result.has_value());
    RequireRawVectors(index, ids.data(), ids.size(), vectors.data());
}
