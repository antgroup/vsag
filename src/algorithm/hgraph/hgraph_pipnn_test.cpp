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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "unittest.h"
#include "vsag_exception.h"

namespace vsag {

// Test-only access, following the existing HGraphCodeWriteTestAccess convention.
class HGraphFusedInputTestAccess {
public:
    static FlattenInterfacePtr
    BaseCodes(const HGraph& graph) {
        return graph.basic_flatten_codes_;
    }

    static const float*
    Convert(const HGraph& graph, const void* data, uint64_t count, Vector<float>& scratch) {
        return graph.fused_float_input(data, count, scratch);
    }

    static std::vector<int64_t>
    NeighborLabels(const HGraph& graph, InnerIdType id) {
        Vector<InnerIdType> neighbors(graph.allocator_);
        graph.bottom_graph_->GetNeighbors(id, neighbors);
        std::vector<int64_t> labels;
        for (const auto neighbor : neighbors) {
            if (neighbor != id) {
                labels.push_back(graph.label_table_->GetLabelById(neighbor));
            }
        }
        return labels;
    }
};

}  // namespace vsag

namespace {

// Test input encoding must match IEEE binary16 round-to-nearest, ties-to-even.
// The legacy generic encoder truncates normal values and mishandles tiny values.
uint16_t
EncodeFP16Input(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    const uint32_t exponent = (bits >> 23) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 255) {
        return static_cast<uint16_t>(sign | 0x7C00U |
                                     (mantissa == 0 ? 0U : ((mantissa >> 13) | 0x0200U)));
    }
    if (exponent >= 143) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (exponent < 102) {
        return sign;
    }
    const uint32_t significand = mantissa | 0x800000U;
    const uint32_t shift = exponent < 113 ? 126 - exponent : 13;
    uint32_t rounded = significand >> shift;
    const uint32_t remainder = significand & ((uint32_t{1} << shift) - 1);
    const uint32_t halfway = uint32_t{1} << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) {
        ++rounded;
    }
    const uint32_t magnitude = exponent < 113 ? rounded : ((exponent - 113) << 10) + rounded;
    return static_cast<uint16_t>(sign | magnitude);
}

vsag::IndexCommonParam
MakePiPNNCommonParam(int64_t dimensions,
                     vsag::MetricType metric = vsag::MetricType::METRIC_TYPE_L2SQR) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dimensions;
    common_param.metric_ = metric;
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

TEST_CASE("HGraph test input FP16 encoding preserves IEEE boundaries", "[ut][hgraph][half_input]") {
    CHECK(EncodeFP16Input(0.0F) == 0x0000);
    CHECK(EncodeFP16Input(-0.0F) == 0x8000);
    CHECK(EncodeFP16Input(0x1p-24F) == 0x0001);
    CHECK(EncodeFP16Input(-0x1p-24F) == 0x8001);
    CHECK(EncodeFP16Input(0x1p-25F) == 0x0000);
    CHECK(EncodeFP16Input(-0x1p-25F) == 0x8000);
    CHECK(EncodeFP16Input(0x1.8p-24F) == 0x0002);
    CHECK(EncodeFP16Input(0x1.ffcp-15F) == 0x0400);
    CHECK(EncodeFP16Input(1.0F) == 0x3C00);
    CHECK(EncodeFP16Input(0x1.002p0F) == 0x3C00);
    CHECK(EncodeFP16Input(0x1.006p0F) == 0x3C02);
    CHECK(EncodeFP16Input(65504.0F) == 0x7BFF);
}

TEST_CASE("HGraph PiPNN builds and searches with all supported metrics",
          "[ut][pipnn][hgraph][metric]") {
    constexpr int64_t dimensions = 8;
    constexpr int64_t count = 48;
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    auto common_param = MakePiPNNCommonParam(dimensions, metric);
    auto index = MakePiPNNIndex(MakePiPNNHGraphParam(), common_param);

    std::vector<float> vectors(static_cast<uint64_t>(count * dimensions));
    std::vector<int64_t> labels(static_cast<uint64_t>(count));
    for (int64_t point = 0; point < count; ++point) {
        labels[point] = 1000 + point * 7;
        for (int64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] =
                std::sin(static_cast<float>(point * 13 + dim * 5)) + point * 0.001F;
        }
        if (metric != vsag::MetricType::METRIC_TYPE_L2SQR) {
            float norm = 0.0F;
            for (int64_t dim = 0; dim < dimensions; ++dim) {
                const float value = vectors[point * dimensions + dim];
                norm += value * value;
            }
            norm = std::sqrt(norm);
            for (int64_t dim = 0; dim < dimensions; ++dim) {
                vectors[point * dimensions + dim] /= norm;
            }
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
    REQUIRE(search_result.value()->GetIds()[0] == labels[0]);
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

TEST_CASE("HGraph exposes PiPNN build parameters", "[ut][pipnn][hgraph][parameter]") {
    auto parameter = MakePiPNNHGraphParam();
    parameter[vsag::PIPNN_PARAMETER_MAX_LEAF_SIZE].SetUint64(256);
    parameter[vsag::PIPNN_PARAMETER_MIN_LEAF_SIZE].SetUint64(32);
    parameter[vsag::PIPNN_PARAMETER_LEADER_SAMPLE_RATE].SetFloat(0.02F);
    *parameter[vsag::PIPNN_PARAMETER_FANOUT].GetInnerJson() = {5, 3, 1};
    parameter[vsag::PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT].SetUint64(6);
    parameter[vsag::PIPNN_PARAMETER_HASH_PLANE_COUNT].SetUint64(4);
    parameter[vsag::PIPNN_PARAMETER_RESERVOIR_SIZE].SetUint64(12);

    auto mapped = vsag::HGraph::CheckAndMappingExternalParam(parameter, MakePiPNNCommonParam(8));
    auto typed = std::dynamic_pointer_cast<vsag::HGraphParameter>(mapped);
    REQUIRE(typed != nullptr);
    REQUIRE(typed->pipnn_param.max_leaf_size == 256);
    REQUIRE(typed->pipnn_param.min_leaf_size == 32);
    REQUIRE(typed->pipnn_param.leader_sample_rate == 0.02F);
    REQUIRE(typed->pipnn_param.fanout == std::vector<uint64_t>{5, 3, 1});
    REQUIRE(typed->pipnn_param.leaf_neighbor_count == 6);
    REQUIRE(typed->pipnn_param.hash_plane_count == 4);
    REQUIRE(typed->pipnn_param.reservoir_size == 12);

    const auto serialized = typed->ToJson();
    const auto graph_json = serialized[vsag::GRAPH_KEY];
    REQUIRE(graph_json[vsag::PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT].GetUint64() == 6);
    REQUIRE(graph_json[vsag::PIPNN_PARAMETER_HASH_PLANE_COUNT].GetUint64() == 4);

    parameter[vsag::PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT].SetUint64(0);
    REQUIRE_THROWS(vsag::HGraph::CheckAndMappingExternalParam(parameter, MakePiPNNCommonParam(8)));
    parameter[vsag::PIPNN_PARAMETER_LEAF_NEIGHBOR_COUNT].SetInt(-1);
    REQUIRE_THROWS(vsag::HGraph::CheckAndMappingExternalParam(parameter, MakePiPNNCommonParam(8)));
}

TEST_CASE("HGraph PiPNN adaptive policy validates and persists", "[ut][pipnn_adaptive][hgraph]") {
    const auto common = MakePiPNNCommonParam(8);
    auto parameter = MakePiPNNHGraphParam();
    parameter["alpha"].SetFloat(1.2F);
    parameter["pipnn_adaptive_pruning"].SetBool(true);
    parameter["pipnn_adaptive_pruning_adjust_step"].SetFloat(0.04F);
    auto mapped = std::dynamic_pointer_cast<vsag::HGraphParameter>(
        vsag::HGraph::CheckAndMappingExternalParam(parameter, common));
    REQUIRE(mapped->pipnn_param.adaptive_pruning);
    CHECK(mapped->pipnn_param.alpha == 1.2F);
    CHECK(mapped->pipnn_param.adaptive_pruning_adjust_step == 0.04F);
    CHECK(mapped->odescent_param->alpha == 1.2F);
    auto restored = std::make_shared<vsag::HGraphParameter>();
    restored->FromJson(mapped->ToJson());
    REQUIRE(mapped->CheckCompatibility(restored));
    restored->pipnn_param.adaptive_pruning_adjust_step = 0.06F;
    CHECK_FALSE(mapped->CheckCompatibility(restored));
    restored->FromJson(mapped->ToJson());
    restored->alpha = 1.1F;
    CHECK_FALSE(mapped->CheckCompatibility(restored));
    restored->FromJson(mapped->ToJson());
    restored->pipnn_param.adaptive_pruning = false;
    CHECK_FALSE(mapped->CheckCompatibility(restored));

    auto legacy_json = mapped->ToJson();
    legacy_json[vsag::GRAPH_KEY].GetInnerJson()->erase("pipnn_adaptive_pruning");
    legacy_json[vsag::GRAPH_KEY].GetInnerJson()->erase("pipnn_adaptive_pruning_adjust_step");
    restored->FromJson(legacy_json);
    CHECK_FALSE(restored->pipnn_param.adaptive_pruning);
    CHECK(restored->pipnn_param.adaptive_pruning_adjust_step == 0.06F);

    for (const auto metric :
         {vsag::MetricType::METRIC_TYPE_IP, vsag::MetricType::METRIC_TYPE_COSINE}) {
        CHECK_THROWS(MakePiPNNIndex(parameter, MakePiPNNCommonParam(8, metric)));
    }
    for (const auto* graph_type : {"nsw", "odescent"}) {
        auto wrong_graph = vsag::JsonType::Parse(parameter.Dump());
        wrong_graph["graph_type"].SetString(graph_type);
        CHECK_THROWS(vsag::HGraph::CheckAndMappingExternalParam(wrong_graph, common));
    }
    parameter["pipnn_adaptive_pruning_adjust_step"].SetFloat(0.6F);
    CHECK_THROWS(vsag::HGraph::CheckAndMappingExternalParam(parameter, common));
}

TEST_CASE("HGraph PiPNN adaptive build reload and search", "[ut][pipnn_adaptive][hgraph]") {
    const int threads = GENERATE(1, 4);
    const auto common = MakePiPNNCommonParam(8);
    auto parameter = MakePiPNNHGraphParam();
    parameter["build_thread_count"].SetInt(threads);
    parameter["alpha"].SetFloat(1.2F);
    parameter["pipnn_adaptive_pruning"].SetBool(true);
    parameter["pipnn_max_leaf_size"].SetUint64(24);
    parameter["pipnn_min_leaf_size"].SetUint64(8);
    auto index = MakePiPNNIndex(parameter, common);
    std::vector<float> vectors(128 * 8);
    std::vector<int64_t> labels(128);
    for (int64_t point = 0; point < 128; ++point) {
        labels[point] = 1000 + point;
        for (int64_t dim = 0; dim < 8; ++dim) {
            vectors[point * 8 + dim] =
                std::sin(static_cast<float>(point * 13 + dim * 5)) + point * 0.001F;
        }
    }
    REQUIRE(index->Build(MakeDataset(vectors, labels, 8, 128)).has_value());
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->GetNumElements() == 128);
    for (int64_t point : {0, 31, 95}) {
        auto query = vsag::Dataset::Make()
                         ->NumElements(1)
                         ->Dim(8)
                         ->Float32Vectors(vectors.data() + point * 8)
                         ->Owner(false);
        auto before = index->KnnSearch(query, 10, R"({"hgraph":{"ef_search":128}})");
        auto after = restored->KnnSearch(query, 10, R"({"hgraph":{"ef_search":128}})");
        REQUIRE(before.has_value());
        REQUIRE(after.has_value());
        REQUIRE(before.value()->GetDim() == 10);
        REQUIRE(after.value()->GetDim() == 10);
        // Approximate search may miss a self neighbor; persistence must preserve every result.
        for (int64_t k = 0; k < 10; ++k) {
            REQUIRE(before.value()->GetIds()[k] == after.value()->GetIds()[k]);
            CHECK(before.value()->GetDistances()[k] == after.value()->GetDistances()[k]);
        }
    }
    SECTION("disabled policy cannot load an adaptive index") {
        parameter["pipnn_adaptive_pruning"].SetBool(false);
    }
    SECTION("different base alpha cannot load an adaptive index") {
        parameter["alpha"].SetFloat(1.3F);
    }
    SECTION("different adjustment step cannot load an adaptive index") {
        parameter["pipnn_adaptive_pruning_adjust_step"].SetFloat(0.04F);
    }
    auto incompatible = MakePiPNNIndex(parameter, common);
    CHECK_FALSE(incompatible->Deserialize(binary.value()).has_value());
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
    SECTION("inner product metric") {
        common_param.metric_ = vsag::MetricType::METRIC_TYPE_IP;
        REQUIRE_NOTHROW(MakePiPNNIndex(MakePiPNNHGraphParam(), common_param));
    }
    SECTION("cosine metric") {
        common_param.metric_ = vsag::MetricType::METRIC_TYPE_COSINE;
        REQUIRE_NOTHROW(MakePiPNNIndex(MakePiPNNHGraphParam(), common_param));
    }
    SECTION("deduplicated storage") {
        auto parameter = MakePiPNNHGraphParam();
        parameter["support_duplicate"].SetBool(true);
        parameter["deduplicate_storage"].SetBool(true);
        REQUIRE_NOTHROW(MakePiPNNIndex(parameter, common_param));
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

namespace {

void
CheckHalfPiPNNQuantizer(bool bf16, vsag::MetricType metric, const std::string& configuration) {
    CAPTURE(bf16, metric, configuration);
    constexpr int64_t count = 544;
    constexpr int64_t dim = 32;
    std::vector<float> decoded(count * dim);
    std::vector<uint16_t> halves(count * dim);
    std::vector<int64_t> ids(count);
    std::mt19937 rng(20261104);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 1000 + row * 7;
        float norm = 0;
        for (int64_t d = 0; d < dim; ++d) {
            // Independent unit directions have a large nearest-neighbor margin.
            // Avoid zero/subnormal inputs to the legacy generic FP16 converter.
            const auto sample = static_cast<int>(rng() % 2000) - 1000;
            const float value = (sample < 0 ? sample : sample + 1) / 1001.0F;
            decoded[row * dim + d] = value;
            norm += value * value;
        }
        norm = std::sqrt(norm);
        for (int64_t d = 0; d < dim; ++d) {
            const float value = decoded[row * dim + d] / norm;
            auto& bits = halves[row * dim + d];
            bits = bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
            decoded[row * dim + d] =
                bf16 ? vsag::generic::BF16ToFloat(bits) : vsag::generic::FP16ToFloat(bits);
        }
    }
    const auto dataset = [&](int64_t row, int64_t size, bool half) {
        auto data =
            vsag::Dataset::Make()->NumElements(size)->Dim(dim)->Ids(ids.data() + row)->Owner(false);
        return half ? data->Float16Vectors(halves.data() + row * dim)
                    : data->Float32Vectors(decoded.data() + row * dim);
    };
    const auto common = [&](bool half) {
        auto result = MakePiPNNCommonParam(dim, metric);
        if (half) {
            result.data_type_ =
                bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
        }
        return result;
    };
    auto parameter = MakePiPNNHGraphParam();
    parameter["max_degree"].SetInt(16);
    parameter["ef_construction"].SetInt(64);
    parameter["train_sample_count"].SetInt(512);
    parameter["base_pq_dim"].SetInt(8);
    parameter["use_reorder"].SetBool(true);
    parameter["precise_quantization_type"].SetString("fp32");
    const bool split = configuration.find("split3+5") != std::string::npos;
    const bool fused = configuration == "fused split3+5";
    const bool transform = configuration.find(',') != std::string::npos;
    parameter["base_quantization_type"].SetString(transform ? "tq"
                                                            : (split ? "rabitq" : configuration));
    if (transform) {
        parameter["tq_chain"].SetString(split ? "mrle, rabitq" : configuration);
        parameter["rabitq_pca_dim"].SetInt(configuration.find("pca") != std::string::npos ? 16 : 0);
        parameter["mrle_dim"].SetInt(dim);
    }
    if (split) {
        parameter["precise_quantization_type"].SetString("rabitq");
        parameter["rabitq_bits_per_dim_base"].SetInt(3);
        parameter["rabitq_bits_per_dim_precise"].SetInt(5);
        parameter["rabitq_bits_per_dim_query"].SetInt(32);
    }
    if (fused) {
        // Existing FP32 fused configuration; cosine is deliberately unsupported.
        parameter["base_io_type"].SetString("memory_io");
        parameter["base_supplement_io_type"].SetString("memory_io");
        parameter["graph_io_type"].SetString("memory_io");
        parameter["reorder_source"].SetString("base");
        parameter["rabitq_use_fht"].SetBool(true);
        parameter["use_mci"].SetBool(false);
        parameter["store_raw_vector"].SetBool(false);
        parameter["rabitq_fused_datacell"].SetBool(true);
    }
    auto half_index = MakePiPNNIndex(parameter, common(true));
    auto control = MakePiPNNIndex(parameter, common(false));
    for (const bool half : {true, false}) {
        auto target = half ? half_index : control;
        auto built = target->Build(dataset(0, count, half));
        REQUIRE(built.has_value());
        REQUIRE(built.value().empty());
        REQUIRE(target->GetNumElements() == count);
    }
    const auto check = [&](const auto& target, bool half) {
        for (const int64_t row : {int64_t{0}, int64_t{257}, count - 1}) {
            CAPTURE(row, half);
            // A tightly sized query catches accidental FP32 reads of half input.
            auto query = dataset(row, 1, half);
            std::vector<uint16_t> tight_half;
            if (half) {
                tight_half.assign(halves.begin() + row * dim, halves.begin() + (row + 1) * dim);
                query->Float16Vectors(tight_half.data());
            }
            auto result = target->KnnSearch(query, 1, R"({"hgraph":{"ef_search":1024}})");
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 1);
            REQUIRE(std::isfinite(result.value()->GetDistances()[0]));
            // Lossy distances need not be exact or nonnegative (IP).
            REQUIRE(result.value()->GetIds()[0] == ids[row]);
        }
    };
    check(half_index, true);
    check(control, false);  // Same quantizer and identical decoded coordinates.
    for (const bool half : {true, false}) {
        auto binary = (half ? half_index : control)->Serialize();
        REQUIRE(binary.has_value());
        auto restored = MakePiPNNIndex(parameter, common(half));
        REQUIRE(restored->Deserialize(binary.value()).has_value());
        REQUIRE(restored->GetNumElements() == count);
        check(restored, half);
    }
}

}  // namespace

TEST_CASE("HGraph PiPNN sampled half input covers FP32 quantizers and split3+5",
          "[ut][pipnn][hgraph][half_input][quantizer_matrix]") {
    const bool bf16 = GENERATE(false, true);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    const std::string configuration = GENERATE("fp32",
                                               "fp16",
                                               "bf16",
                                               "sq8",
                                               "sq4",
                                               "sq8_uniform",
                                               "sq4_uniform",
                                               "sq8_local",
                                               "sq4_local",
                                               "pq",
                                               "pqfs",
                                               "rabitq",
                                               "rom, fp32",
                                               "rom, fp16",
                                               "rom, bf16",
                                               "rom, sq8",
                                               "rom, sq4",
                                               "rom, sq8_uniform",
                                               "rom, sq4_uniform",
                                               "rom, pq",
                                               "rom, pqfs",
                                               "rom, rabitq",
                                               "pca, rom, sq8_uniform",
                                               "fht, sq8_uniform",
                                               "mrle, fp32",
                                               "split3+5",
                                               "mrle, split3+5");
    CheckHalfPiPNNQuantizer(bf16, metric, configuration);
}

TEST_CASE("HGraph PiPNN fused split3+5 half input build search and reload",
          "[ut][pipnn][hgraph][half_input][rabitq_split][fused][quantizer_matrix]") {
    const bool bf16 = GENERATE(false, true);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    CheckHalfPiPNNQuantizer(bf16, metric, "fused split3+5");
}

// HalfFusedSafety: independent non-power-of-two and non-finite coverage.
namespace {

vsag::JsonType
MakeHalfFusedSafetyParam(int base_bits) {
    auto parameter = MakePiPNNHGraphParam();
    parameter["base_quantization_type"].SetString("rabitq");
    parameter["precise_quantization_type"].SetString("rabitq");
    parameter["rabitq_bits_per_dim_base"].SetInt(base_bits);
    parameter["rabitq_bits_per_dim_precise"].SetInt(8 - base_bits);
    parameter["rabitq_bits_per_dim_query"].SetInt(32);
    parameter["use_reorder"].SetBool(true);
    parameter["base_io_type"].SetString("memory_io");
    parameter["base_supplement_io_type"].SetString("memory_io");
    parameter["graph_io_type"].SetString("memory_io");
    parameter["reorder_source"].SetString("base");
    parameter["rabitq_use_fht"].SetBool(true);
    parameter["use_mci"].SetBool(false);
    parameter["store_raw_vector"].SetBool(false);
    parameter["rabitq_fused_datacell"].SetBool(true);
    return parameter;
}

std::vector<uint16_t>
MakeHalfFusedSafetyVectors(bool bf16, int64_t dim, int64_t count) {
    std::vector<uint16_t> halves(count * dim);
    std::mt19937 rng(20261105);
    std::vector<float> row(dim);
    for (int64_t point = 0; point < count; ++point) {
        float norm = 0.0F;
        for (int64_t d = 0; d < dim; ++d) {
            const auto sample = static_cast<int>(rng() % 2000) - 1000;
            row[d] = (sample < 0 ? sample : sample + 1) / 1001.0F;
            norm += row[d] * row[d];
        }
        norm = std::sqrt(norm);
        for (int64_t d = 0; d < dim; ++d) {
            const float value = row[d] / norm;
            halves[point * dim + d] =
                bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
        }
    }
    return halves;
}

}  // namespace

TEST_CASE("HGraph PiPNN fused half input handles non-power-of-two dimensions",
          "[ut][pipnn][hgraph][half_input][quantizer_matrix][HalfFusedSafety][fused]") {
    // Exercise a non-power-of-two dimension without padding either source buffer.
    // A direct half-to-float reinterpretation is not safe.
    constexpr int64_t dim = 33;
    constexpr int64_t count = 64;
    const bool bf16 = GENERATE(false, true);
    const int base_bits = GENERATE(1, 3);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    CAPTURE(bf16, base_bits, metric);
    auto common = MakePiPNNCommonParam(dim, metric);
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    auto parameter = MakeHalfFusedSafetyParam(base_bits);
    auto index = MakePiPNNIndex(parameter, common);
    auto halves = MakeHalfFusedSafetyVectors(bf16, dim, count);
    std::vector<int64_t> ids(count);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 5000 + row * 7;
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float16Vectors(halves.data())
                    ->Owner(false);
    auto built = index->Build(base);
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    REQUIRE(index->GetNumElements() == count);
    const auto check = [&](const auto& target) {
        for (const int64_t row : {int64_t{0}, count - 1}) {
            CAPTURE(row);
            // A separately allocated exact-size query exercises the public boundary.
            std::vector<uint16_t> query_halves(halves.begin() + row * dim,
                                               halves.begin() + (row + 1) * dim);
            auto query = vsag::Dataset::Make()
                             ->NumElements(1)
                             ->Dim(dim)
                             ->Float16Vectors(query_halves.data())
                             ->Owner(false);
            auto result = target->KnnSearch(query, 1, R"({"hgraph":{"ef_search":128}})");
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetDim() == 1);
            REQUIRE(std::isfinite(result.value()->GetDistances()[0]));
            REQUIRE(result.value()->GetIds()[0] == ids[row]);
        }
    };
    check(index);
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->GetNumElements() == count);
    check(restored);
}

TEST_CASE("HGraph fused half input default UpdateVector preserves connectivity and reload",
          "[ut][pipnn][hgraph][half_input][rabitq_split][fused][update_vector]") {
    constexpr int64_t dim = 33;
    constexpr int64_t count = 64;
    const bool bf16 = GENERATE(false, true);
    const int base_bits = GENERATE(1, 3);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    CAPTURE(bf16, base_bits, metric);
    auto common = MakePiPNNCommonParam(dim, metric);
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    const auto parameter = MakeHalfFusedSafetyParam(base_bits);
    auto index = MakePiPNNIndex(parameter, common);
    auto halves = MakeHalfFusedSafetyVectors(bf16, dim, count);
    std::vector<int64_t> ids(count);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 9000 + row * 7;
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(ids.data())
                    ->Float16Vectors(halves.data())
                    ->Owner(false);
    auto built = index->Build(base);
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    const auto graph = std::static_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    const auto checksum = graph->GraphChecksum();
    const auto neighbors = vsag::HGraphFusedInputTestAccess::NeighborLabels(*graph, 0);
    REQUIRE_FALSE(neighbors.empty());
    const auto neighbor_row = (neighbors.front() - ids.front()) / 7;
    REQUIRE(neighbor_row > 0);
    REQUIRE(neighbor_row < count);
    std::vector<uint16_t> updated(dim);
    for (int64_t d = 0; d < dim; ++d) {
        const auto value =
            bf16 ? vsag::generic::BF16ToFloat(halves[d]) : vsag::generic::FP16ToFloat(halves[d]);
        updated[d] =
            bf16 ? vsag::generic::FloatToBF16(value * 1.125F) : EncodeFP16Input(value * 1.125F);
    }
    auto update = vsag::Dataset::Make()
                      ->NumElements(1)
                      ->Dim(dim)
                      ->Float16Vectors(updated.data())
                      ->Owner(false);
    auto neighbor_query = vsag::Dataset::Make()
                              ->NumElements(1)
                              ->Dim(dim)
                              ->Float16Vectors(halves.data() + neighbor_row * dim)
                              ->Owner(false);
    const auto before = index->CalcDistanceById(update, ids.front());
    REQUIRE(before.has_value());
    // Omit force_update: both native self-distance and neighbor-distance checks must run.
    auto accepted = index->UpdateVector(ids.front(), update);
    REQUIRE(accepted.has_value());
    REQUIRE(accepted.value());
    auto after = index->CalcDistanceById(update, ids.front());
    REQUIRE(after.has_value());
    REQUIRE(after.value() < before.value());
    const auto check = [&](const auto& target) {
        auto inner = std::static_pointer_cast<vsag::HGraph>(target->GetInnerIndex());
        REQUIRE(inner->GraphChecksum() == checksum);
        REQUIRE(target->GetNumElements() == count);
        auto persisted_distance = target->CalcDistanceById(update, ids.front());
        REQUIRE(persisted_distance.has_value());
        REQUIRE(std::abs(persisted_distance.value() - after.value()) < 1e-5F);
        // Moving a node onto an actual connected neighbor must be rejected, not forced.
        auto self_distance = target->CalcDistanceById(neighbor_query, ids.front());
        auto neighbor_distance = target->CalcDistanceById(neighbor_query, neighbors.front());
        REQUIRE(self_distance.has_value());
        REQUIRE(neighbor_distance.has_value());
        REQUIRE(neighbor_distance.value() < self_distance.value());
        auto rejected = target->UpdateVector(ids.front(), neighbor_query);
        REQUIRE(rejected.has_value());
        REQUIRE_FALSE(rejected.value());
        auto unchanged = target->CalcDistanceById(update, ids.front());
        REQUIRE(unchanged.has_value());
        REQUIRE(std::abs(unchanged.value() - after.value()) < 1e-5F);
        REQUIRE(inner->GraphChecksum() == checksum);
        auto result = target->KnnSearch(update, 1, R"({"hgraph":{"ef_search":128}})");
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 1);
        REQUIRE(result.value()->GetIds()[0] == ids.front());
        REQUIRE(std::isfinite(result.value()->GetDistances()[0]));
    };
    check(index);
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    check(restored);
    auto accepted_after_reload = restored->UpdateVector(ids.front(), update);
    REQUIRE(accepted_after_reload.has_value());
    REQUIRE(accepted_after_reload.value());
    check(restored);
}

TEST_CASE("HGraph PiPNN fused half input rejects non-finite build and tight queries",
          "[ut][pipnn][hgraph][half_input][quantizer_matrix][HalfFusedSafety][fused]") {
    constexpr int64_t dim = 33;
    constexpr int64_t count = 64;
    const bool bf16 = GENERATE(false, true);
    const int base_bits = GENERATE(1, 3);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    CAPTURE(bf16, base_bits, metric);
    auto common = MakePiPNNCommonParam(dim, metric);
    common.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    auto parameter = MakeHalfFusedSafetyParam(base_bits);
    auto halves = MakeHalfFusedSafetyVectors(bf16, dim, count);
    std::vector<int64_t> ids(count);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 7000 + row;
    }
    const auto base_dataset = [&](std::vector<uint16_t>& values) {
        return vsag::Dataset::Make()
            ->NumElements(count)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float16Vectors(values.data())
            ->Owner(false);
    };
    auto index = MakePiPNNIndex(parameter, common);
    auto built = index->Build(base_dataset(halves));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    // Encode IEEE quiet NaN, +Inf, -Inf directly: the conversion helpers are
    // intended for finite input and must not mask the actual boundary payload.
    const std::array<uint16_t, 3> invalid_bits =
        bf16 ? std::array<uint16_t, 3>{0x7FC0, 0x7F80, 0xFF80}
             : std::array<uint16_t, 3>{0x7E00, 0x7C00, 0xFC00};
    for (const uint16_t bits : invalid_bits) {
        CAPTURE(bits);
        auto invalid_base = halves;
        invalid_base.back() = bits;  // Last real coordinate, not an FHT padding slot.
        auto rejected_index = MakePiPNNIndex(parameter, common);
        auto rejected_build = rejected_index->Build(base_dataset(invalid_base));
        REQUIRE_FALSE(rejected_build.has_value());
        REQUIRE(rejected_build.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(rejected_index->GetNumElements() == 0);
        for (const int64_t coordinate : {int64_t{0}, dim - 1}) {
            CAPTURE(coordinate);
            std::vector<uint16_t> query_halves(halves.begin(), halves.begin() + dim);
            query_halves[coordinate] = bits;
            auto query = vsag::Dataset::Make()
                             ->NumElements(1)
                             ->Dim(dim)
                             ->Float16Vectors(query_halves.data())
                             ->Owner(false);
            auto result = index->KnnSearch(
                query, 1, R"({"hgraph":{"ef_search":128,"rabitq_one_bit_search":true}})");
            REQUIRE_FALSE(result.has_value());
            REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
            REQUIRE(index->GetNumElements() == count);
        }
    }
}

// OrdinaryRaBitQTypedFallback: cover the non-fused, no-reorder SQ8 fallback.
// Half-input adapters report QUANTIZATION_ADAPTER_rabitq rather than rabitq.

TEST_CASE("HGraph ordinary RaBitQ typed input build add and reload",
          "[ut][hgraph][rabitq][half_input][OrdinaryRaBitQTypedFallback]") {
    constexpr int64_t dim = 64;
    constexpr int64_t build_count = 64;
    constexpr int64_t add_count = 8;
    constexpr int64_t total_count = build_count + add_count;
    constexpr int64_t k = 3;
    const auto input_type = GENERATE(vsag::DataTypes::DATA_TYPE_FLOAT,
                                     vsag::DataTypes::DATA_TYPE_FP16,
                                     vsag::DataTypes::DATA_TYPE_BF16);
    const std::string graph_type = GENERATE("odescent", "nsw");
    CAPTURE(input_type, graph_type);

    // Keep the shared safe allocator alive until both indexes/results are gone.
    auto common = MakePiPNNCommonParam(dim);
    common.data_type_ = input_type;
    auto parameter = MakePiPNNHGraphParam();
    parameter["graph_type"].SetString(graph_type);
    parameter["base_quantization_type"].SetString("rabitq");
    parameter["rabitq_bits_per_dim_base"].SetInt(1);
    parameter["rabitq_bits_per_dim_query"].SetInt(32);
    parameter["use_reorder"].SetBool(false);
    // Existing FP32 contract: incremental temporary-SQ8 rebuild needs raw vectors.
    parameter["store_raw_vector"].SetBool(true);
    parameter["rabitq_fused_datacell"].SetBool(false);
    parameter["use_mci"].SetBool(false);
    auto index = MakePiPNNIndex(parameter, common);

    // Stable, distinct, nonzero coordinates; no legacy FP16 zero encoder.
    std::vector<float> floats(total_count * dim);
    std::vector<uint16_t> halves(total_count * dim);
    std::vector<int64_t> ids(total_count);
    std::mt19937 rng(20261105);
    for (int64_t row = 0; row < total_count; ++row) {
        ids[row] = 9000 + row * 7;
        for (int64_t d = 0; d < dim; ++d) {
            const int sample = static_cast<int>(rng() % 2000) - 1000;
            const float value = (sample < 0 ? sample : sample + 1) / 1001.0F;
            floats[row * dim + d] = value;
            if (input_type == vsag::DataTypes::DATA_TYPE_BF16) {
                halves[row * dim + d] = vsag::generic::FloatToBF16(value);
            } else if (input_type == vsag::DataTypes::DATA_TYPE_FP16) {
                halves[row * dim + d] = EncodeFP16Input(value);
            }
        }
    }
    const auto dataset = [&](int64_t start, int64_t count) {
        auto result = vsag::Dataset::Make()
                          ->NumElements(count)
                          ->Dim(dim)
                          ->Ids(ids.data() + start)
                          ->Owner(false);
        if (input_type == vsag::DataTypes::DATA_TYPE_FLOAT) {
            result->Float32Vectors(floats.data() + start * dim);
        } else {
            result->Float16Vectors(halves.data() + start * dim);
        }
        return result;
    };

    const auto built = index->Build(dataset(0, build_count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    REQUIRE(index->GetNumElements() == build_count);
    // More than one row is essential: Add also has a temporary-build-data predicate.
    const auto added = index->Add(dataset(build_count, add_count));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());
    REQUIRE(index->GetNumElements() == total_count);

    const auto check_search = [&](const auto& target) {
        for (const int64_t row : {int64_t{0}, build_count - 1, build_count, total_count - 1}) {
            CAPTURE(row);
            // Exact-size typed buffers make accidental FP32 half-input reads visible
            // to sanitizers, without depending on the surrounding base allocation.
            std::vector<float> query_float;
            std::vector<uint16_t> query_half;
            auto query = vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Owner(false);
            if (input_type == vsag::DataTypes::DATA_TYPE_FLOAT) {
                query_float.assign(floats.begin() + row * dim, floats.begin() + (row + 1) * dim);
                query->Float32Vectors(query_float.data());
            } else {
                query_half.assign(halves.begin() + row * dim, halves.begin() + (row + 1) * dim);
                query->Float16Vectors(query_half.data());
            }
            const auto found = target->KnnSearch(query, k, R"({"hgraph":{"ef_search":128}})");
            REQUIRE(found.has_value());
            REQUIRE(found.value() != nullptr);
            REQUIRE(found.value()->GetNumElements() == 1);
            REQUIRE(found.value()->GetDim() == k);
            REQUIRE(found.value()->GetIds() != nullptr);
            REQUIRE(found.value()->GetDistances() != nullptr);
            for (int64_t rank = 0; rank < k; ++rank) {
                const auto label = found.value()->GetIds()[rank];
                REQUIRE(std::find(ids.begin(), ids.end(), label) != ids.end());
                REQUIRE(std::isfinite(found.value()->GetDistances()[rank]));
                for (int64_t previous = 0; previous < rank; ++previous) {
                    REQUIRE(found.value()->GetIds()[previous] != label);
                }
            }
            // RaBitQ 1-bit is approximate: deliberately do not assert self top-1.
        }
    };
    check_search(index);
    const auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(parameter, common);
    const auto loaded = restored->Deserialize(binary.value());
    REQUIRE(loaded.has_value());
    REQUIRE(restored->GetNumElements() == total_count);
    check_search(restored);
}

TEST_CASE("HGraph typed no-train terminal feature and range parity",
          "[ut][pipnn][hgraph][TypedNoTrainFeatureParity]") {
    constexpr int64_t dim = 8;
    constexpr int64_t count = 16;
    constexpr float radius = 1024.0F;
    const auto input_type = GENERATE(vsag::DataTypes::DATA_TYPE_FLOAT,
                                     vsag::DataTypes::DATA_TYPE_FP16,
                                     vsag::DataTypes::DATA_TYPE_BF16);
    const std::string codec = GENERATE("fp32", "fp16", "bf16", "sq8_local", "sq4_local");
    CAPTURE(input_type, codec);
    auto common = MakePiPNNCommonParam(dim);
    common.data_type_ = input_type;
    auto native_common = common;
    native_common.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    auto parameter = MakePiPNNHGraphParam();
    parameter["base_quantization_type"].SetString(codec);
    parameter["use_reorder"].SetBool(false);
    auto typed = MakePiPNNIndex(parameter, common);
    auto native = MakePiPNNIndex(parameter, native_common);
    const auto check_features = [&]() {
        for (const auto feature : {vsag::IndexFeature::NEED_TRAIN,
                                   vsag::IndexFeature::SUPPORT_RANGE_SEARCH,
                                   vsag::IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER}) {
            CAPTURE(feature);
            CHECK(typed->CheckFeature(feature) == native->CheckFeature(feature));
        }
        REQUIRE_FALSE(native->CheckFeature(vsag::IndexFeature::NEED_TRAIN));
        REQUIRE(native->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH));
        REQUIRE(native->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER));
        REQUIRE_FALSE(typed->CheckFeature(vsag::IndexFeature::NEED_TRAIN));
        REQUIRE(typed->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH));
        REQUIRE(typed->CheckFeature(vsag::IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER));
    };
    check_features();

    // Nonzero binary-exact input avoids legacy FP16 zero/subnormal conversion.
    // FLOAT control receives identical decoded coordinates.
    std::vector<float> decoded(count * dim);
    std::vector<uint16_t> halves(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 6000 + row * 7;
        for (int64_t d = 0; d < dim; ++d) {
            const auto offset = row * dim + d;
            const float value = 1.0F + static_cast<float>(offset) / 64.0F;
            decoded[offset] = value;
            if (input_type == vsag::DataTypes::DATA_TYPE_FP16) {
                halves[offset] = vsag::generic::FloatToFP16(value);
                decoded[offset] = vsag::generic::FP16ToFloat(halves[offset]);
            } else if (input_type == vsag::DataTypes::DATA_TYPE_BF16) {
                halves[offset] = vsag::generic::FloatToBF16(value);
                decoded[offset] = vsag::generic::BF16ToFloat(halves[offset]);
            }
        }
    }
    const bool half_input = input_type != vsag::DataTypes::DATA_TYPE_FLOAT;
    const auto dataset = [&](bool half) {
        auto data =
            vsag::Dataset::Make()->NumElements(count)->Dim(dim)->Ids(ids.data())->Owner(false);
        return half ? data->Float16Vectors(halves.data()) : data->Float32Vectors(decoded.data());
    };
    auto built = typed->Build(dataset(half_input));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    auto native_built = native->Build(dataset(false));
    REQUIRE(native_built.has_value());
    REQUIRE(native_built.value().empty());
    REQUIRE(typed->GetNumElements() == count);
    REQUIRE(native->GetNumElements() == count);
    check_features();

    // Radius contains every point; unlimited results and ef_search > count.
    const std::string search_parameters = R"({"hgraph":{"ef_search":64}})";
    for (const bool use_typed : {false, true}) {
        CAPTURE(use_typed);
        auto target = use_typed ? typed : native;
        // Exact-size typed query buffers catch accidental FP32 reads under ASan.
        std::vector<float> query_float(decoded.begin(), decoded.begin() + dim);
        std::vector<uint16_t> query_half(halves.begin(), halves.begin() + dim);
        auto query = vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Owner(false);
        if (use_typed and half_input) {
            query->Float16Vectors(query_half.data());
        } else {
            query->Float32Vectors(query_float.data());
        }
        const auto check_result = [&](const auto& result, bool filtered) {
            REQUIRE(result.has_value());
            REQUIRE(result.value()->GetNumElements() == 1);
            REQUIRE(result.value()->GetDim() == (filtered ? count / 2 : count));
            std::vector<int64_t> found;
            for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
                const auto id = result.value()->GetIds()[i];
                const float distance = result.value()->GetDistances()[i];
                REQUIRE(std::find(ids.begin(), ids.end(), id) != ids.end());
                REQUIRE(std::isfinite(distance));
                REQUIRE(distance >= 0.0F);
                REQUIRE(distance <= radius);
                if (filtered) {
                    REQUIRE((id - ids[0]) % 14 == 0);
                }
                found.push_back(id);
            }
            std::sort(found.begin(), found.end());
            REQUIRE(std::adjacent_find(found.begin(), found.end()) == found.end());
        };
        check_result(target->RangeSearch(query, radius, search_parameters), false);
        // The std::function overload is a blacklist: true means exclude.
        const std::function<bool(int64_t)> exclude_odd_rows = [&](int64_t id) {
            return ((id - ids[0]) / 7) % 2 != 0;
        };
        check_result(target->RangeSearch(query, radius, search_parameters, exclude_odd_rows), true);
    }
}

TEST_CASE("HGraph fused input converts finite half values without changing float input",
          "[ut][pipnn][hgraph][half_input][fused_bounds]") {
    constexpr int64_t dim = 3;
    const auto type = GENERATE(vsag::DataTypes::DATA_TYPE_FP16,
                               vsag::DataTypes::DATA_TYPE_BF16,
                               vsag::DataTypes::DATA_TYPE_FLOAT);
    CAPTURE(type);
    auto common = MakePiPNNCommonParam(dim);
    common.data_type_ = type;
    vsag::HGraph graph(vsag::HGraph::CheckAndMappingExternalParam(MakePiPNNHGraphParam(), common),
                       common);
    vsag::Vector<float> scratch(common.allocator_.get());
    scratch.assign(4, 42.0F);
    // Literal encodings/expected values avoid testing a converter against itself.
    // Includes signed zero and the minimum positive half subnormal.
    const std::array<uint16_t, 6> fp16{0x0000, 0x8000, 0x3c00, 0xc000, 0x3800, 0x0001};
    const std::array<uint16_t, 6> bf16{0x0000, 0x8000, 0x3f80, 0xc000, 0x3f00, 0x0001};
    const float minimum = type == vsag::DataTypes::DATA_TYPE_BF16 ? 0x1p-133F : 0x1p-24F;
    const std::array<float, 6> expected{0.0F, -0.0F, 1.0F, -2.0F, 0.5F, minimum};
    const bool passthrough = type == vsag::DataTypes::DATA_TYPE_FLOAT;
    const void* input =
        passthrough ? static_cast<const void*>(expected.data())
                    : static_cast<const void*>(
                          type == vsag::DataTypes::DATA_TYPE_FP16 ? fp16.data() : bf16.data());
    const auto* result = vsag::HGraphFusedInputTestAccess::Convert(graph, input, 2, scratch);
    REQUIRE(result != nullptr);
    if (passthrough) {
        REQUIRE(result == expected.data());
        REQUIRE(scratch.size() == 4);
        REQUIRE(std::all_of(scratch.begin(), scratch.end(), [](float v) { return v == 42.0F; }));
    } else {
        REQUIRE(result == scratch.data());
        REQUIRE(scratch.size() == expected.size());
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i);
        REQUIRE(std::isfinite(result[i]));
        REQUIRE(result[i] == expected[i]);
        REQUIRE(std::signbit(result[i]) == std::signbit(expected[i]));
    }

    // count==0 permits nullptr: never dereference or form nullptr+0 in the test.
    // A vector's data() after resize(0) need not be nullptr.
    result = vsag::HGraphFusedInputTestAccess::Convert(graph, nullptr, 0, scratch);
    if (passthrough) {
        REQUIRE(result == nullptr);
        REQUIRE(scratch.size() == 4);
    } else {
        REQUIRE(scratch.empty());
        REQUIRE(result == scratch.data());
    }
}

TEST_CASE("HGraph fused half input rejects invalid spans before resizing or reading",
          "[ut][pipnn][hgraph][half_input][fused_bounds]") {
    constexpr int64_t dim = 3;
    const auto type = GENERATE(vsag::DataTypes::DATA_TYPE_FP16, vsag::DataTypes::DATA_TYPE_BF16);
    CAPTURE(type);
    auto common = MakePiPNNCommonParam(dim);
    common.data_type_ = type;
    vsag::HGraph graph(vsag::HGraph::CheckAndMappingExternalParam(MakePiPNNHGraphParam(), common),
                       common);
    vsag::Vector<float> scratch(common.allocator_.get());
    scratch.assign(3, 42.0F);
    const auto* previous_data = scratch.data();
    const auto previous_capacity = scratch.capacity();
    // Only one valid source element: accepted huge counts must never reach a read.
    const uint16_t dummy = 0x3c00;
    const auto reject = [&](const void* data, uint64_t count, const std::string& message) {
        CAPTURE(count, message);
        bool caught = false;
        try {
            (void)vsag::HGraphFusedInputTestAccess::Convert(graph, data, count, scratch);
        } catch (const vsag::VsagException& error) {
            caught = true;
            REQUIRE(error.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
            REQUIRE(std::string(error.what()).find(message) != std::string::npos);
        }
        // bad_alloc/length_error must not masquerade as an argument guard.
        REQUIRE(caught);
        REQUIRE(scratch.data() == previous_data);
        REQUIRE(scratch.capacity() == previous_capacity);
        REQUIRE(scratch.size() == 3);
        REQUIRE(std::all_of(scratch.begin(), scratch.end(), [](float v) { return v == 42.0F; }));
    };
    SECTION("nonnull count requires a source") {
        reject(nullptr, 1, "fused HGraph half input must not be null");
    }
    SECTION("count times dimension overflow") {
        const auto count = std::numeric_limits<uint64_t>::max() / uint64_t{dim} + 1;
        reject(&dummy, count, "fused HGraph input element count overflows");
    }
    SECTION("converted float byte span overflow without element overflow") {
        const auto count =
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float) / dim) + 1;
        REQUIRE(count <= std::numeric_limits<uint64_t>::max() / uint64_t{dim});
        REQUIRE(count * uint64_t{dim} > std::numeric_limits<size_t>::max() / sizeof(float));
        reject(&dummy, count, "fused HGraph converted input size overflows");
    }
}

TEST_CASE("HGraph Tune trains transforms on packed native rows",
          "[ut][pipnn][hgraph][half_input][tune_packing]") {
    const auto type = GENERATE(vsag::DataTypes::DATA_TYPE_FLOAT,
                               vsag::DataTypes::DATA_TYPE_FP16,
                               vsag::DataTypes::DATA_TYPE_BF16);
    const int64_t dim = GENERATE(7, 8);
    const std::string chain = GENERATE("pca,fp32", "mrle,pca,fp32");
    constexpr int64_t count = 40;
    CAPTURE(type, dim, chain);
    auto common = MakePiPNNCommonParam(dim);
    common.data_type_ = type;
    auto parameter = MakePiPNNHGraphParam();
    parameter["store_raw_vector"].SetBool(true);
    std::vector<float> decoded(count * dim);
    std::vector<uint16_t> halves(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t row = 0; row < count; ++row) {
        ids[row] = 7000 + row;
        for (int64_t d = 0; d < dim; ++d) {
            const auto offset = row * dim + d;
            const float value =
                1.0F + static_cast<float>((row * 17 + d * 11 + row * d * 3) % 127) / 32.0F;
            halves[offset] = type == vsag::DataTypes::DATA_TYPE_BF16
                                 ? vsag::generic::FloatToBF16(value)
                                 : EncodeFP16Input(value);
            decoded[offset] = type == vsag::DataTypes::DATA_TYPE_BF16
                                  ? vsag::generic::BF16ToFloat(halves[offset])
                                  : vsag::generic::FP16ToFloat(halves[offset]);
        }
    }
    const bool half = type != vsag::DataTypes::DATA_TYPE_FLOAT;
    auto data = vsag::Dataset::Make()->NumElements(count)->Dim(dim)->Ids(ids.data())->Owner(false);
    if (half) {
        data->Float16Vectors(halves.data());
    } else {
        data->Float32Vectors(decoded.data());
    }
    auto index = MakePiPNNIndex(parameter, common);
    auto built = index->Build(data);
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    auto tuned_parameter = parameter;
    tuned_parameter["base_quantization_type"].SetString("tq");
    tuned_parameter["tq_chain"].SetString(chain);
    tuned_parameter["mrle_dim"].SetInt(dim);
    vsag::JsonType tune;
    tune["index_param"].SetJson(tuned_parameter);
    auto tuned = index->Tune(tune.Dump());
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());

    // Independent Train gets contiguous native input, never Tune's reconstruction buffer.
    // Compare encoded vectors, not just distances: full-dimensional PCA preserves distances
    // even when trained on the wrong samples, hiding this packing regression in recall tests.
    auto mapped = std::dynamic_pointer_cast<vsag::HGraphParameter>(
        vsag::HGraph::CheckAndMappingExternalParam(tuned_parameter, common));
    auto reference = vsag::FlattenInterface::MakeInstance(mapped->base_codes_param, common);
    const void* training =
        half ? static_cast<const void*>(halves.data()) : static_cast<const void*>(decoded.data());
    reference->Train(training, count);
    for (int64_t row = 0; row < count; ++row) {
        reference->InsertVector(half ? static_cast<const void*>(halves.data() + row * dim)
                                     : static_cast<const void*>(decoded.data() + row * dim),
                                row);
    }
    const auto check = [&](const auto& target) {
        auto graph = std::dynamic_pointer_cast<vsag::HGraph>(target->GetInnerIndex());
        auto actual = vsag::HGraphFusedInputTestAccess::BaseCodes(*graph);
        REQUIRE(actual->GetQuantizerCodeSize() == reference->GetQuantizerCodeSize());
        for (int64_t row = 0; row < count; ++row) {
            CAPTURE(row);
            auto expected_code = reference->AcquireCodesById(row);
            auto actual_code = actual->AcquireCodesById(row);
            REQUIRE(expected_code);
            REQUIRE(actual_code);
            CHECK(std::memcmp(expected_code.Data(), actual_code.Data(), dim * sizeof(float)) == 0);
        }
        auto query = vsag::Dataset::Make()->NumElements(1)->Dim(dim)->Owner(false);
        if (half) {
            query->Float16Vectors(halves.data());
        } else {
            query->Float32Vectors(decoded.data());
        }
        auto result = target->KnnSearch(query, 1, R"({"hgraph":{"ef_search":64}})");
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 1);
        CHECK(result.value()->GetIds()[0] == ids[0]);
        CHECK(std::isfinite(result.value()->GetDistances()[0]));
    };
    check(index);
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakePiPNNIndex(tuned_parameter, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    check(restored);
}
