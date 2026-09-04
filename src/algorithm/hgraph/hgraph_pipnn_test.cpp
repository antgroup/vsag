// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "unittest.h"

namespace {

vsag::IndexCommonParam
MakePiPNNCommonParam(int64_t dimensions) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dimensions;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

vsag::JsonType
MakePiPNNHGraphParam() {
    return vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "base_io_type": "block_memory_io",
        "graph_io_type": "block_memory_io",
        "graph_storage_type": "flat",
        "graph_type": "pipnn",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1
    })");
}

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
MakePiPNNIndex(const vsag::JsonType& parameter, const vsag::IndexCommonParam& common_param) {
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(parameter, common_param);
}

vsag::DatasetPtr
MakeDataset(std::vector<float>& vectors,
            std::vector<int64_t>& labels,
            int64_t dimensions,
            int64_t count) {
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(dimensions)
        ->Ids(labels.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

}  // namespace

TEST_CASE("HGraph PiPNN builds and searches", "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 8;
    constexpr int64_t count = 48;
    auto common_param = MakePiPNNCommonParam(dimensions);
    auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), common_param);

    std::vector<float> vectors(static_cast<uint64_t>(count * dimensions));
    std::vector<int64_t> labels(static_cast<uint64_t>(count));
    for (int64_t point = 0; point < count; ++point) {
        labels[point] = 1000 + point * 7;
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] =
                static_cast<float>((point * 13 + dim * 5) % 37) + point * 0.001F;
        }
    }
    auto base = MakeDataset(vectors, labels, dimensions, count);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == count);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dimensions)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    auto search_result = index->KnnSearch(query, 3, R"({"hgraph": {"ef_search": 64}})");
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 3);
    for (int64_t result = 0; result < search_result.value()->GetDim(); ++result) {
        REQUIRE(std::find(labels.begin(), labels.end(), search_result.value()->GetIds()[result]) !=
                labels.end());
    }
}

TEST_CASE("HGraph PiPNN keeps configured ODescent routing parameters", "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 4;
    constexpr int64_t count = 32;
    auto common_param = MakePiPNNCommonParam(dimensions);
    auto parameter = MakePiPNNHGraphParam();
    parameter["max_degree"].SetInt(4);
    parameter["build_thread_count"].SetInt(4);
    parameter["build_block_size"].SetInt(73);
    auto mapped = vsag::HGraph::CheckAndMappingExternalParam(parameter, common_param);
    auto typed = std::dynamic_pointer_cast<vsag::HGraphParameter>(mapped);
    REQUIRE(typed != nullptr);
    REQUIRE(typed->odescent_param != nullptr);

    auto inner = std::make_shared<vsag::HGraph>(mapped, common_param);
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(inner, common_param);
    std::vector<float> vectors(static_cast<uint64_t>(count * dimensions));
    std::vector<int64_t> labels(static_cast<uint64_t>(count));
    for (int64_t point = 0; point < count; ++point) {
        labels[point] = point;
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] = static_cast<float>(point + dim);
        }
    }

    REQUIRE(index->Build(MakeDataset(vectors, labels, dimensions, count)).has_value());
    REQUIRE(typed->odescent_param->max_degree == 4);
    REQUIRE(typed->odescent_param->block_size == 73);
}

TEST_CASE("HGraph PiPNN keeps duplicate-label and entry-point semantics", "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 4;
    auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), MakePiPNNCommonParam(dimensions));
    std::vector<float> vectors = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
    std::vector<int64_t> labels = {10, 10, 30};
    auto base = MakeDataset(vectors, labels, dimensions, 3);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value() == std::vector<int64_t>{10});
    REQUIRE(index->GetNumElements() == 2);

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dimensions)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    auto search_result = index->KnnSearch(query, 1, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 1);
}

TEST_CASE("HGraph validates PiPNN graph type and data boundary", "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 4;
    auto common_param = MakePiPNNCommonParam(dimensions);

    SECTION("unknown graph type") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["graph_type"].SetString("unknown");
        REQUIRE_THROWS(MakePiPNNIndex(parameter, common_param));
    }
    SECTION("non L2 metric") {
        common_param.metric_ = vsag::MetricType::METRIC_TYPE_IP;
        REQUIRE_THROWS(MakePiPNNIndex(MakePiPNNHGraphParam(), common_param));
    }
    SECTION("deduplicated storage") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["support_duplicate"].SetBool(true);
        parameter["deduplicate_storage"].SetBool(true);
        REQUIRE_THROWS(MakePiPNNIndex(parameter, common_param));
    }
    SECTION("non-fp32 base codes") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["base_quantization_type"].SetString("sq8");
        REQUIRE_NOTHROW(MakePiPNNIndex(parameter, common_param));
    }
    SECTION("reorder") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["use_reorder"].SetBool(true);
        parameter["precise_quantization_type"].SetString("fp32");
        REQUIRE_NOTHROW(MakePiPNNIndex(parameter, common_param));
    }
    SECTION("extra info configuration") {
        common_param.extra_info_size_ = 4;
        REQUIRE_NOTHROW(MakePiPNNIndex(MakePiPNNHGraphParam(), common_param));
    }
    SECTION("missing extra info payload") {
        constexpr int64_t count = 4;
        common_param.extra_info_size_ = 4;
        auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), common_param);
        std::vector<float> vectors(static_cast<uint64_t>(count * dimensions), 1.0F);
        std::vector<int64_t> labels = {10, 20, 30, 40};

        auto result = index->Build(MakeDataset(vectors, labels, dimensions, count));
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(index->GetNumElements() == 0);
    }
    SECTION("mismatched extra info size") {
        constexpr int64_t count = 4;
        common_param.extra_info_size_ = 4;
        auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), common_param);
        std::vector<float> vectors(static_cast<uint64_t>(count * dimensions), 1.0F);
        std::vector<int64_t> labels = {10, 20, 30, 40};
        std::vector<char> extra_infos(static_cast<uint64_t>(count * 2), 'x');
        auto base = MakeDataset(vectors, labels, dimensions, count)
                        ->ExtraInfoSize(2)
                        ->ExtraInfos(extra_infos.data());

        auto result = index->Build(base);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(index->GetNumElements() == 0);
    }
    SECTION("attribute filtering") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["use_attribute_filter"].SetBool(true);
        REQUIRE_NOTHROW(MakePiPNNIndex(parameter, common_param));
    }
}

TEST_CASE("HGraph PiPNN supports RaBitQ with SQ8 reorder, add, and serialization",
          "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 8;
    constexpr int64_t count = 32;
    auto common_param = MakePiPNNCommonParam(dimensions);
    auto parameter = MakePiPNNHGraphParam();
    parameter["base_quantization_type"].SetString("rabitq");
    parameter["use_reorder"].SetBool(true);
    parameter["precise_quantization_type"].SetString("sq8");
    parameter["rabitq_bits_per_dim_base"].SetUint64(1);
    auto index = MakePiPNNIndex(parameter, common_param);

    std::vector<float> vectors(static_cast<uint64_t>((count + 1) * dimensions));
    std::vector<int64_t> labels(static_cast<uint64_t>(count + 1));
    for (int64_t point = 0; point <= count; ++point) {
        labels[point] = 1000 + point;
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] =
                static_cast<float>((point * 13 + dim * 5) % 71) + point * 0.001F;
        }
    }

    REQUIRE(index->Build(MakeDataset(vectors, labels, dimensions, count)).has_value());
    auto added = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dimensions)
                     ->Ids(labels.data() + count)
                     ->Float32Vectors(vectors.data() + count * dimensions)
                     ->Owner(false);
    REQUIRE(index->Add(added).has_value());
    REQUIRE(index->GetNumElements() == count + 1);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(parameter, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dimensions)
                     ->Float32Vectors(vectors.data() + count * dimensions)
                     ->Owner(false);
    auto result = restored->KnnSearch(query, 1, R"({"hgraph": {"ef_search": 64}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == labels[count]);
}

TEST_CASE("HGraph PiPNN preserves extra info during batch build", "[ut][pipnn][hgraph]") {
    constexpr int64_t dimensions = 4;
    constexpr int64_t count = 12;
    constexpr int64_t extra_info_size = 2;
    auto common_param = MakePiPNNCommonParam(dimensions);
    common_param.extra_info_size_ = extra_info_size;
    auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), common_param);

    std::vector<float> vectors(static_cast<uint64_t>(count * dimensions));
    std::vector<int64_t> labels(static_cast<uint64_t>(count));
    std::vector<char> extra_infos(static_cast<uint64_t>(count * extra_info_size));
    for (int64_t point = 0; point < count; ++point) {
        labels[point] = 2000 + point;
        extra_infos[point * extra_info_size] = static_cast<char>('a' + point);
        extra_infos[point * extra_info_size + 1] = static_cast<char>('A' + point);
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] = static_cast<float>(point * dimensions + dim);
        }
    }
    auto base = MakeDataset(vectors, labels, dimensions, count)
                    ->ExtraInfoSize(extra_info_size)
                    ->ExtraInfos(extra_infos.data());
    REQUIRE(index->Build(base).has_value());

    std::array<char, extra_info_size> fetched{};
    REQUIRE(index->GetExtraInfoByIds(labels.data() + 5, 1, fetched.data()).has_value());
    REQUIRE(std::equal(fetched.begin(), fetched.end(), extra_infos.begin() + 5 * extra_info_size));
}
