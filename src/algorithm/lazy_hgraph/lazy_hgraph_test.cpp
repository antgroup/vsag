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

#include "lazy_hgraph.h"

#include <atomic>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/factory.h"
#include "vsag/index_features.h"

namespace {

constexpr int64_t DIM = 8;

vsag::IndexCommonParam
MakeCommonParam(uint64_t extra_info_size = 0) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = DIM;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.extra_info_size_ = extra_info_size;
    return common_param;
}

vsag::JsonType
MakeLazyParam(uint64_t threshold,
              const std::string& base_quantization_type = "fp32",
              bool support_force_remove = false) {
    auto param = vsag::JsonType::Parse(R"({
        "transition_threshold": 4,
        "hgraph": {
            "base_quantization_type": "fp32",
            "max_degree": 4,
            "ef_construction": 8,
            "build_thread_count": 1
        }
    })");
    param["transition_threshold"].SetUint64(threshold);
    param["hgraph"]["base_quantization_type"].SetString(base_quantization_type);
    param["hgraph"]["support_force_remove"].SetBool(support_force_remove);
    return param;
}

vsag::DatasetPtr
MakeDataset(int64_t count,
            int64_t first_id,
            std::vector<float>& vectors,
            std::vector<int64_t>& ids) {
    vectors.resize(count * DIM);
    ids.resize(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = first_id + i;
        for (int64_t j = 0; j < DIM; ++j) {
            vectors[i * DIM + j] = static_cast<float>((first_id + i) * 0.1 + j);
        }
    }
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

void
AttachExtraInfos(const vsag::DatasetPtr& dataset,
                 const std::vector<char>& tags,
                 std::vector<char>& extra_infos,
                 uint64_t extra_info_size) {
    extra_infos.assign(static_cast<uint64_t>(dataset->GetNumElements()) * extra_info_size, '\0');
    for (int64_t i = 0; i < dataset->GetNumElements(); ++i) {
        extra_infos[static_cast<uint64_t>(i) * extra_info_size] = tags[static_cast<uint64_t>(i)];
    }
    dataset->ExtraInfos(extra_infos.data())->ExtraInfoSize(static_cast<int64_t>(extra_info_size));
}

vsag::DatasetPtr
MakeQuery(const std::vector<float>& vectors, int64_t row) {
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(DIM)
        ->Float32Vectors(vectors.data() + row * DIM)
        ->Owner(false);
}

std::shared_ptr<vsag::LazyHGraph>
MakeLazyIndex(uint64_t threshold,
              const std::string& base_quantization_type = "fp32",
              bool support_force_remove = false,
              uint64_t extra_info_size = 0) {
    auto common_param = MakeCommonParam(extra_info_size);
    auto param = vsag::LazyHGraph::CheckAndMappingExternalParam(
        MakeLazyParam(threshold, base_quantization_type, support_force_remove), common_param);
    auto index = std::make_shared<vsag::LazyHGraph>(param, common_param);
    index->InitFeatures();
    return index;
}

class TagExtraInfoFilter : public vsag::Filter {
public:
    explicit TagExtraInfoFilter(char tag) : tag_(tag) {
    }

    bool
    CheckValid(int64_t) const override {
        return false;
    }

    bool
    CheckValid(const char* data) const override {
        return data != nullptr and data[0] == tag_;
    }

private:
    char tag_;
};

std::string
MakeFactoryParam(uint64_t threshold) {
    auto root = vsag::JsonType::Parse(R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "lazy_hgraph": {
            "transition_threshold": 4,
            "hgraph": {
                "base_quantization_type": "fp32",
                "max_degree": 4,
                "ef_construction": 8,
                "build_thread_count": 1
            }
        }
    })");
    root["lazy_hgraph"]["transition_threshold"].SetUint64(threshold);
    return root.Dump();
}

float
L2(const float* lhs, const float* rhs) {
    float distance = 0.0F;
    for (int64_t i = 0; i < DIM; ++i) {
        auto diff = lhs[i] - rhs[i];
        distance += diff * diff;
    }
    return distance;
}

vsag::DatasetPtr
MakeSingleVector(std::vector<float>& vector, float first_value) {
    vector.resize(DIM);
    for (int64_t i = 0; i < DIM; ++i) {
        vector[i] = first_value + static_cast<float>(i);
    }
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(DIM)
        ->Float32Vectors(vector.data())
        ->Owner(false);
}

void
RequireStoredVector(const std::shared_ptr<vsag::LazyHGraph>& index,
                    int64_t id,
                    const std::vector<float>& expected) {
    auto fetched = index->GetVectorByIds(&id, 1, nullptr);
    REQUIRE(fetched->GetNumElements() == 1);
    const auto* fetched_vectors = fetched->GetFloat32Vectors();
    for (int64_t i = 0; i < DIM; ++i) {
        REQUIRE(fetched_vectors[i] == expected[static_cast<uint64_t>(i)]);
    }
}

}  // namespace

TEST_CASE("LazyHGraph stays flat before threshold and searches exactly", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 100, vectors, ids);

    auto failed_ids = index->Add(data);
    REQUIRE(failed_ids.empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto result =
        index->KnnSearch(MakeQuery(vectors, 1), 1, R"({"hgraph":{"ef_search":40}})", nullptr);
    REQUIRE(result->GetIds()[0] == 101);
    REQUIRE(result->GetDistances()[0] == 0.0F);

    auto range = index->RangeSearch(MakeQuery(vectors, 2), 0.0F, "{}", nullptr);
    REQUIRE(range->GetNumElements() == 1);
    REQUIRE(range->GetIds()[0] == 102);
}

TEST_CASE("LazyHGraph supports extra info filter in flat phase", "[ut][lazy_hgraph]") {
    constexpr uint64_t extra_info_size = 4;
    auto index = MakeLazyIndex(10, "fp32", false, extra_info_size);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER));
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID));

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 100, vectors, ids);
    std::vector<char> extra_infos;
    AttachExtraInfos(data, {'A', 'A', 'B'}, extra_infos, extra_info_size);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto filter = std::make_shared<TagExtraInfoFilter>('B');
    auto result = index->KnnSearch(MakeQuery(vectors, 1),
                                   1,
                                   R"({"hgraph":{"ef_search":40,"use_extra_info_filter":true}})",
                                   filter);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 102);
    REQUIRE(result->GetExtraInfoSize() == static_cast<int64_t>(extra_info_size));
    REQUIRE(result->GetExtraInfos()[0] == 'B');
}

TEST_CASE("LazyHGraph migrates extra info filter data to graph phase", "[ut][lazy_hgraph]") {
    constexpr uint64_t extra_info_size = 4;
    auto index = MakeLazyIndex(4, "fp32", false, extra_info_size);

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 200, vectors, ids);
    std::vector<char> extra_infos;
    AttachExtraInfos(data, {'A', 'B', 'A'}, extra_infos, extra_info_size);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 300, more_vectors, more_ids);
    std::vector<char> more_extra_infos;
    AttachExtraInfos(more, {'A'}, more_extra_infos, extra_info_size);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);

    auto filter = std::make_shared<TagExtraInfoFilter>('B');
    auto result = index->KnnSearch(MakeQuery(vectors, 0),
                                   1,
                                   R"({"hgraph":{"ef_search":40,"use_extra_info_filter":true}})",
                                   filter);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 201);

    std::vector<char> fetched(extra_info_size);
    index->GetExtraInfoByIds(&ids[1], 1, fetched.data());
    REQUIRE(fetched[0] == 'B');
}

TEST_CASE("LazyHGraph keeps flat extra info filter after serialization", "[ut][lazy_hgraph]") {
    constexpr uint64_t extra_info_size = 4;
    auto index = MakeLazyIndex(10, "fp32", false, extra_info_size);

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 350, vectors, ids);
    std::vector<char> extra_infos;
    AttachExtraInfos(data, {'A', 'B', 'A'}, extra_infos, extra_info_size);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto binary = static_cast<vsag::InnerIndexInterface*>(index.get())->Serialize();
    auto restored = MakeLazyIndex(10, "fp32", false, extra_info_size);
    static_cast<vsag::InnerIndexInterface*>(restored.get())->Deserialize(binary);
    REQUIRE(restored->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto filter = std::make_shared<TagExtraInfoFilter>('B');
    auto result = restored->KnnSearch(MakeQuery(vectors, 0),
                                      1,
                                      R"({"hgraph":{"ef_search":40,"use_extra_info_filter":true}})",
                                      filter);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 351);
}

TEST_CASE("LazyHGraph transitions to graph and accepts more data", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 200, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 300, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetNumElements() == 5);

    auto result =
        index->KnnSearch(MakeQuery(more_vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", nullptr);
    REQUIRE(result->GetIds()[0] == 300);
}

TEST_CASE("LazyHGraph updates vectors in flat phase", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(10);
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT));

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 320, vectors, ids);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    std::vector<float> updated_vector;
    auto update = MakeSingleVector(updated_vector, 100.0F);
    REQUIRE(index->UpdateVector(321, update));

    RequireStoredVector(index, 321, updated_vector);
    REQUIRE(index->CalcDistanceById(updated_vector.data(), 321) == 0.0F);
    auto result = index->KnnSearch(update, 1, "{}", nullptr);
    REQUIRE(result->GetIds()[0] == 321);
}

TEST_CASE("LazyHGraph migrates flat vector updates to graph phase", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 330, vectors, ids);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    std::vector<float> updated_vector;
    auto update = MakeSingleVector(updated_vector, 200.0F);
    REQUIRE(index->UpdateVector(331, update));

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 340, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);

    RequireStoredVector(index, 331, updated_vector);
    REQUIRE(index->CalcDistanceById(updated_vector.data(), 331) == 0.0F);
}

TEST_CASE("LazyHGraph updates vectors in graph phase", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 360, vectors, ids);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);

    std::vector<float> updated_vector;
    auto update = MakeSingleVector(updated_vector, 300.0F);
    REQUIRE(index->UpdateVector(362, update, true));

    RequireStoredVector(index, 362, updated_vector);
    REQUIRE(index->CalcDistanceById(updated_vector.data(), 362) == 0.0F);
}

TEST_CASE("LazyHGraph filters removed flat ids during transition", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(3);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(2, 400, vectors, ids);
    REQUIRE(index->Add(data).empty());
    REQUIRE(index->Remove({401}) == 1);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(2, 500, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE_FALSE(index->CheckIdExist(401));
}

TEST_CASE("LazyHGraph flat remove always force removes", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(10);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 450, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    REQUIRE(index->Remove({451}, vsag::RemoveMode::MARK_REMOVE) == 1);
    REQUIRE_FALSE(index->CheckIdExist(451));
    REQUIRE(index->GetNumElements() == 2);
    REQUIRE(index->GetNumberRemoved() == 0);
}

TEST_CASE("LazyHGraph threshold one transitions immediately", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(1);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(1, 600, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
}

TEST_CASE("LazyHGraph factory and flat serialization round trip", "[ut][lazy_hgraph]") {
    auto index = vsag::Factory::CreateIndex("lazy_hgraph", MakeFactoryParam(10));
    REQUIRE(index.has_value());
    REQUIRE(index.value()->GetIndexType() == vsag::IndexType::LAZY_HGRAPH);

    auto invalid_param = vsag::JsonType::Parse(MakeFactoryParam(10));
    invalid_param["lazy_hgraph"]["transition_threshold"].SetJson(vsag::JsonType::Parse("-1"));
    REQUIRE_FALSE(vsag::Factory::CreateIndex("lazy_hgraph", invalid_param.Dump()).has_value());

    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 700, vectors, ids);
    REQUIRE(index.value()->Add(data).has_value());

    auto binary = index.value()->Serialize();
    REQUIRE(binary.has_value());

    auto restored = vsag::Factory::CreateIndex("lazy_hgraph", MakeFactoryParam(10));
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(binary.value()).has_value());

    auto result = restored.value()->KnnSearch(
        MakeQuery(vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", vsag::FilterPtr{});
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == 700);
}

TEST_CASE("LazyHGraph rejects flat quantization parameters", "[ut][lazy_hgraph]") {
    auto common_param = MakeCommonParam();
    auto param = MakeLazyParam(4);
    param["flat_quantization_type"].SetString("sq8");

    REQUIRE_THROWS(vsag::LazyHGraph::CheckAndMappingExternalParam(param, common_param));

    auto flat_param = MakeLazyParam(4);
    flat_param["flat"].SetJson(vsag::JsonType::Parse(R"({"base_quantization_type":"sq8"})"));
    REQUIRE_THROWS(vsag::LazyHGraph::CheckAndMappingExternalParam(flat_param, common_param));
}

TEST_CASE("LazyHGraph rejects invalid parameter types", "[ut][lazy_hgraph]") {
    auto common_param = MakeCommonParam();

    auto negative_threshold = MakeLazyParam(4);
    negative_threshold["transition_threshold"].SetJson(vsag::JsonType::Parse("-1"));
    REQUIRE_THROWS(
        vsag::LazyHGraph::CheckAndMappingExternalParam(negative_threshold, common_param));

    auto graph_param =
        vsag::HGraph::CheckAndMappingExternalParam(MakeLazyParam(4)["hgraph"], common_param);
    REQUIRE_THROWS(vsag::LazyHGraph(graph_param, common_param));

    auto large_threshold = MakeLazyParam(1ULL << 40U);
    auto lazy_param = vsag::LazyHGraph::CheckAndMappingExternalParam(large_threshold, common_param);
    REQUIRE(lazy_param->ToJson()["transition_threshold"].GetUint64() == (1ULL << 40U));
}

TEST_CASE("LazyHGraph build chooses flat or graph by threshold", "[ut][lazy_hgraph]") {
    auto small_index = MakeLazyIndex(4);
    std::vector<float> small_vectors;
    std::vector<int64_t> small_ids;
    auto small = MakeDataset(3, 800, small_vectors, small_ids);

    REQUIRE(small_index->Build(small).empty());
    REQUIRE(small_index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);
    REQUIRE(small_index->KnnSearch(MakeQuery(small_vectors, 0), 1, "{}", nullptr)->GetIds()[0] ==
            800);

    auto large_index = MakeLazyIndex(4);
    std::vector<float> large_vectors;
    std::vector<int64_t> large_ids;
    auto large = MakeDataset(4, 900, large_vectors, large_ids);

    REQUIRE(large_index->Build(large).empty());
    REQUIRE(large_index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(
        large_index
            ->KnnSearch(MakeQuery(large_vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", nullptr)
            ->GetIds()[0] == 900);
}

TEST_CASE("LazyHGraph serializes empty flat and graph phases", "[ut][lazy_hgraph]") {
    auto empty = MakeLazyIndex(4);
    auto empty_binary = static_cast<vsag::InnerIndexInterface*>(empty.get())->Serialize();

    auto restored_empty = MakeLazyIndex(4);
    static_cast<vsag::InnerIndexInterface*>(restored_empty.get())->Deserialize(empty_binary);
    REQUIRE(restored_empty->GetPhase() == vsag::LazyHGraph::Phase::FLAT);
    REQUIRE(restored_empty->GetNumElements() == 0);

    auto flat = MakeLazyIndex(4);
    std::vector<float> flat_vectors;
    std::vector<int64_t> flat_ids;
    auto flat_data = MakeDataset(3, 950, flat_vectors, flat_ids);
    REQUIRE(flat->Add(flat_data).empty());
    REQUIRE(flat->GetPhase() == vsag::LazyHGraph::Phase::FLAT);

    auto flat_binary = static_cast<vsag::InnerIndexInterface*>(flat.get())->Serialize();
    auto restored_flat = MakeLazyIndex(4);
    static_cast<vsag::InnerIndexInterface*>(restored_flat.get())->Deserialize(flat_binary);
    REQUIRE(restored_flat->GetPhase() == vsag::LazyHGraph::Phase::FLAT);
    REQUIRE(
        restored_flat
            ->KnnSearch(MakeQuery(flat_vectors, 2), 1, R"({"hgraph":{"ef_search":40}})", nullptr)
            ->GetIds()[0] == 952);

    auto graph = MakeLazyIndex(2);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(2, 1000, vectors, ids);
    REQUIRE(graph->Add(data).empty());
    REQUIRE(graph->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);

    auto graph_binary = static_cast<vsag::InnerIndexInterface*>(graph.get())->Serialize();
    auto restored_graph = MakeLazyIndex(2);
    static_cast<vsag::InnerIndexInterface*>(restored_graph.get())->Deserialize(graph_binary);
    REQUIRE(restored_graph->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(restored_graph
                ->KnnSearch(MakeQuery(vectors, 1), 1, R"({"hgraph":{"ef_search":40}})", nullptr)
                ->GetIds()[0] == 1001);
}

TEST_CASE("LazyHGraph rejects invalid serialized phase", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    auto binary = static_cast<vsag::InnerIndexInterface*>(index.get())->Serialize();
    auto payload = binary.Get(vsag::INDEX_LAZY_HGRAPH);
    REQUIRE(payload.size > 16);
    payload.data.get()[16] = 2;

    auto restored = MakeLazyIndex(4);
    REQUIRE_THROWS(static_cast<vsag::InnerIndexInterface*>(restored.get())->Deserialize(binary));
}

TEST_CASE("LazyHGraph removes ids in both phases", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 1100, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);
    REQUIRE(index->Remove({1101}) == 1);
    REQUIRE_FALSE(index->CheckIdExist(1101));
    REQUIRE(index->GetNumberRemoved() == 0);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(2, 1200, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->Remove({1200}) == 1);
    REQUIRE_FALSE(index->CheckIdExist(1200));
    REQUIRE(index->GetNumberRemoved() == 1);
}

TEST_CASE("LazyHGraph preserves fp32 vectors during transition", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 1300, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);

    auto fetched = index->GetVectorByIds(ids.data(), ids.size(), nullptr);
    REQUIRE(fetched->GetNumElements() == ids.size());
    const auto* fetched_vectors = fetched->GetFloat32Vectors();
    for (int64_t i = 0; i < data->GetNumElements() * DIM; ++i) {
        REQUIRE(fetched_vectors[i] == vectors[i]);
    }
}

TEST_CASE("LazyHGraph graph search matches direct HGraph baseline", "[ut][lazy_hgraph]") {
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 1400, vectors, ids);
    auto query = MakeQuery(vectors, 2);
    auto search_param = R"({"hgraph":{"ef_search":40}})";

    auto lazy = MakeLazyIndex(4);
    REQUIRE(lazy->Add(data).empty());
    auto lazy_result = lazy->KnnSearch(query, 2, search_param, nullptr);

    auto common_param = MakeCommonParam();
    auto graph_param =
        vsag::HGraph::CheckAndMappingExternalParam(MakeLazyParam(4)["hgraph"], common_param);
    auto hgraph = std::make_shared<vsag::HGraph>(graph_param, common_param);
    hgraph->InitFeatures();
    REQUIRE(hgraph->Build(data).empty());
    auto hgraph_result = hgraph->KnnSearch(query, 2, search_param, nullptr);

    REQUIRE(lazy_result->GetDim() == hgraph_result->GetDim());
    for (int64_t i = 0; i < lazy_result->GetDim(); ++i) {
        REQUIRE(lazy_result->GetIds()[i] == hgraph_result->GetIds()[i]);
        REQUIRE(lazy_result->GetDistances()[i] == hgraph_result->GetDistances()[i]);
    }
}

TEST_CASE("LazyHGraph keeps flat fp32 while graph quantization is configurable",
          "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4, "sq8");
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 1500, vectors, ids);

    REQUIRE(index->Add(data).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::FLAT);
    REQUIRE(index->CalcDistanceById(vectors.data(), 1500) == 0.0F);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 1600, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->CheckIdExist(1600));
}

TEST_CASE("LazyHGraph calculates distance by id in flat and graph phases", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(4);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(3, 1700, vectors, ids);

    REQUIRE(index->Add(data).empty());
    auto expected = L2(vectors.data(), vectors.data() + DIM);
    REQUIRE(index->CalcDistanceById(vectors.data(), 1701) == expected);

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(1, 1800, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->CalcDistanceById(more_vectors.data(), 1800) == 0.0F);
}

TEST_CASE("LazyHGraph supports concurrent search during transition", "[ut][lazy_hgraph]") {
    auto index = MakeLazyIndex(8);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    auto data = MakeDataset(4, 1900, vectors, ids);
    REQUIRE(index->Add(data).empty());

    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::thread search_thread([&]() {
        while (not stop.load(std::memory_order_acquire)) {
            try {
                auto result = index->KnnSearch(
                    MakeQuery(vectors, 0), 1, R"({"hgraph":{"ef_search":40}})", nullptr);
                if (result == nullptr or result->GetIds()[0] != 1900) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
            } catch (...) {
                failed.store(true, std::memory_order_release);
                return;
            }
            std::this_thread::yield();
        }
    });

    std::vector<float> more_vectors;
    std::vector<int64_t> more_ids;
    auto more = MakeDataset(4, 2000, more_vectors, more_ids);
    REQUIRE(index->Add(more).empty());
    stop.store(true, std::memory_order_release);
    search_thread.join();

    REQUIRE_FALSE(failed.load(std::memory_order_acquire));
    REQUIRE(index->GetPhase() == vsag::LazyHGraph::Phase::GRAPH);
    REQUIRE(index->CheckIdExist(2000));
}
