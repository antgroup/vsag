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

#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "sindi.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

using namespace vsag;

namespace {

struct SmallDmqDataset {
    explicit SmallDmqDataset(uint32_t term_id_offset) {
        ids0 = {term_id_offset + 1, term_id_offset + 4, term_id_offset + 9};
        ids1 = {term_id_offset + 1, term_id_offset + 2, term_id_offset + 4};
        ids2 = {term_id_offset + 5, term_id_offset + 9};

        sparse_vectors[0].len_ = ids0.size();
        sparse_vectors[0].ids_ = ids0.data();
        sparse_vectors[0].vals_ = vals0.data();
        sparse_vectors[2].len_ = ids1.size();
        sparse_vectors[2].ids_ = ids1.data();
        sparse_vectors[2].vals_ = vals1.data();
        sparse_vectors[3].len_ = ids2.size();
        sparse_vectors[3].ids_ = ids2.data();
        sparse_vectors[3].vals_ = vals2.data();
    }

    DatasetPtr
    Base() {
        return Dataset::Make()
            ->NumElements(sparse_vectors.size())
            ->SparseVectors(sparse_vectors.data())
            ->Ids(labels.data())
            ->Owner(false);
    }

    DatasetPtr
    Query() {
        return Dataset::Make()->NumElements(1)->SparseVectors(sparse_vectors.data())->Owner(false);
    }

    std::array<int64_t, 4> labels{10, 40, 20, 30};
    std::array<uint32_t, 3> ids0{};
    std::array<float, 3> vals0{0.0F, 0.5F, 1.0F};
    std::array<uint32_t, 3> ids1{};
    std::array<float, 3> vals1{1.0F, 0.25F, 0.5F};
    std::array<uint32_t, 2> ids2{};
    std::array<float, 2> vals2{0.25F, 1.0F};
    std::array<SparseVector, 4> sparse_vectors{};
};

std::shared_ptr<SINDIParameter>
CreateDmqParameter(bool immutable, bool remap_term_ids) {
    auto param_json = JsonType::Parse(R"({
        "use_reorder": true,
        "rerank_type": "dmq8",
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 16,
        "avg_doc_term_length": 3,
        "remap_term_ids": false,
        "immutable": false
    })");
    param_json["immutable"].SetBool(immutable);
    param_json["remap_term_ids"].SetBool(remap_term_ids);
    auto index_param = std::make_shared<SINDIParameter>();
    index_param->FromJson(param_json);
    return index_param;
}

constexpr auto kDmqSearchParameters = R"({
    "sindi": {
        "query_prune_ratio": 0.0,
        "term_prune_ratio": 0.0,
        "n_candidate": 3,
        "use_term_lists_heap_insert": false
    }
})";

void
RequireSameResults(const DatasetPtr& expected, const DatasetPtr& actual) {
    REQUIRE(actual->GetDim() == expected->GetDim());
    for (int64_t i = 0; i < expected->GetDim(); ++i) {
        REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
        REQUIRE(std::abs(actual->GetDistances()[i] - expected->GetDistances()[i]) < 1e-6F);
    }
}

}  // namespace

TEST_CASE("SINDI DMQ Rerank DataCell Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    std::vector<int64_t> ids = {10, 20, 30};
    uint32_t ids0[] = {1, 4, 9};
    float vals0[] = {0.0F, 0.5F, 1.0F};
    uint32_t ids1[] = {1, 2, 4};
    float vals1[] = {1.0F, 0.25F, 0.5F};
    uint32_t ids2[] = {5, 9};
    float vals2[] = {0.25F, 1.0F};
    SparseVector sparse_vectors[3];
    sparse_vectors[0].len_ = 3;
    sparse_vectors[0].ids_ = ids0;
    sparse_vectors[0].vals_ = vals0;
    sparse_vectors[1].len_ = 3;
    sparse_vectors[1].ids_ = ids1;
    sparse_vectors[1].vals_ = vals1;
    sparse_vectors[2].len_ = 2;
    sparse_vectors[2].ids_ = ids2;
    sparse_vectors[2].vals_ = vals2;

    auto base = vsag::Dataset::Make();
    base->NumElements(3)->SparseVectors(sparse_vectors)->Ids(ids.data())->Owner(false);

    auto param_json = vsag::JsonType::Parse(R"({
        "use_reorder": true,
        "rerank_type": "dmq8",
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 10,
        "avg_doc_term_length": 3
    })");
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.empty());

    auto memory_detail = index->GetMemoryUsageDetail();
    REQUIRE(memory_detail.count("rerank_backend") == 1);
    REQUIRE(memory_detail.at("rerank_backend") > 0);

    test_serializion(*index, *another_index);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(sparse_vectors)->Owner(false);

    std::string search_param_str = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 3,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = index->KnnSearch(query, 1, search_param_str, nullptr);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 10);

    float self_distance = index->CalcDistanceById(query, 10, true);
    float serialized_self_distance = another_index->CalcDistanceById(query, 10, true);
    REQUIRE(std::abs(self_distance - serialized_self_distance) < 1e-6F);
    REQUIRE(index->CalcDistanceById(query, 999, true) == -1.0F);

    int64_t distance_ids[] = {10, 999};
    auto distances = index->CalDistanceById(query, distance_ids, 2, true);
    REQUIRE(std::abs(distances->GetDistances()[0] - self_distance) < 1e-6F);
    REQUIRE(distances->GetDistances()[1] == -1.0F);

    SparseVector decoded;
    index->GetSparseVectorByInnerId(0, &decoded, allocator.get());
    REQUIRE(decoded.len_ == sparse_vectors[0].len_);
    for (uint32_t i = 0; i < decoded.len_; ++i) {
        REQUIRE(decoded.ids_[i] == sparse_vectors[0].ids_[i]);
    }
    allocator->Deallocate(decoded.ids_);
    allocator->Deallocate(decoded.vals_);
}

TEST_CASE("SINDI DMQ Rerank Large Term ID Fallback Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    std::vector<int64_t> ids = {10, 20};
    uint32_t ids0[] = {1'100'001, 1'100'009};
    float vals0[] = {0.0F, 1.0F};
    uint32_t ids1[] = {1'100'001, 1'100'010};
    float vals1[] = {0.5F, 0.25F};
    SparseVector sparse_vectors[2];
    sparse_vectors[0].len_ = 2;
    sparse_vectors[0].ids_ = ids0;
    sparse_vectors[0].vals_ = vals0;
    sparse_vectors[1].len_ = 2;
    sparse_vectors[1].ids_ = ids1;
    sparse_vectors[1].vals_ = vals1;

    auto base = vsag::Dataset::Make();
    base->NumElements(2)->SparseVectors(sparse_vectors)->Ids(ids.data())->Owner(false);

    auto param_json = vsag::JsonType::Parse(R"({
        "use_reorder": true,
        "rerank_type": "dmq8",
        "use_quantization": false,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 1200000,
        "avg_doc_term_length": 2
    })");
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.empty());
    test_serializion(*index, *another_index);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(sparse_vectors)->Owner(false);
    auto distance = index->CalcDistanceById(query, 10, true);
    auto serialized_distance = another_index->CalcDistanceById(query, 10, true);
    REQUIRE(std::isfinite(distance));
    REQUIRE(std::abs(distance - serialized_distance) < 1e-6F);
}

TEST_CASE("SINDI DMQ Rerank with Quantized Posting Test", "[ut][SINDI]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    std::vector<int64_t> ids = {10, 20, 30};
    uint32_t ids0[] = {1, 4, 9};
    float vals0[] = {0.0F, 0.5F, 1.0F};
    uint32_t ids1[] = {1, 2, 4};
    float vals1[] = {1.0F, 0.25F, 0.5F};
    uint32_t ids2[] = {5, 9};
    float vals2[] = {0.25F, 1.0F};
    SparseVector sparse_vectors[3];
    sparse_vectors[0].len_ = 3;
    sparse_vectors[0].ids_ = ids0;
    sparse_vectors[0].vals_ = vals0;
    sparse_vectors[1].len_ = 3;
    sparse_vectors[1].ids_ = ids1;
    sparse_vectors[1].vals_ = vals1;
    sparse_vectors[2].len_ = 2;
    sparse_vectors[2].ids_ = ids2;
    sparse_vectors[2].vals_ = vals2;

    auto base = vsag::Dataset::Make();
    base->NumElements(3)->SparseVectors(sparse_vectors)->Ids(ids.data())->Owner(false);

    auto param_json = vsag::JsonType::Parse(R"({
        "use_reorder": true,
        "rerank_type": "dmq8",
        "use_quantization": true,
        "doc_prune_ratio": 0.0,
        "window_size": 10000,
        "term_id_limit": 10,
        "avg_doc_term_length": 3
    })");
    auto index_param = std::make_shared<vsag::SINDIParameter>();
    index_param->FromJson(param_json);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto another_index = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(base);
    REQUIRE(build_res.empty());
    test_serializion(*index, *another_index);

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->SparseVectors(sparse_vectors)->Owner(false);
    std::string search_param_str = R"({
        "sindi": {
            "query_prune_ratio": 0.0,
            "term_prune_ratio": 0.0,
            "n_candidate": 3,
            "use_term_lists_heap_insert": false
        }
    })";
    auto result = index->KnnSearch(query, 1, search_param_str, nullptr);
    auto another_result = another_index->KnnSearch(query, 1, search_param_str, nullptr);
    REQUIRE(result->GetDim() == 1);
    REQUIRE(result->GetIds()[0] == 10);
    REQUIRE(result->GetIds()[0] == another_result->GetIds()[0]);
}

TEST_CASE("SINDI Immutable DMQ Build Search Serialization and Remap Test", "[ut][SINDI]") {
    const bool remap_term_ids = GENERATE(false, true);
    const uint32_t term_id_offset = remap_term_ids ? 1'100'000 : 0;
    SmallDmqDataset data(term_id_offset);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto index_param = CreateDmqParameter(true, remap_term_ids);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto restored = std::make_unique<SINDI>(index_param, common_param);

    auto build_res = index->Build(data.Base());
    REQUIRE(build_res.size() == 1);
    REQUIRE(build_res[0] == data.labels[1]);
    REQUIRE(index->GetNumElements() == 3);

    auto query = data.Query();
    auto knn_result = index->KnnSearch(query, 3, kDmqSearchParameters, nullptr);
    REQUIRE(knn_result->GetDim() == 3);
    REQUIRE(knn_result->GetIds()[0] == data.labels[0]);

    auto range_result = index->RangeSearch(query, 2.0F, kDmqSearchParameters, nullptr);
    REQUIRE(range_result->GetDim() == 3);

    test_serializion(*index, *restored);
    REQUIRE(restored->GetNumElements() == 3);
    auto restored_knn_result = restored->KnnSearch(query, 3, kDmqSearchParameters, nullptr);
    auto restored_range_result = restored->RangeSearch(query, 2.0F, kDmqSearchParameters, nullptr);
    RequireSameResults(knn_result, restored_knn_result);
    RequireSameResults(range_result, restored_range_result);
}

TEST_CASE("SINDI DMQ Streaming Serialization Test", "[ut][SINDI][streaming]") {
    SmallDmqDataset data(1'100'000);

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.metric_ = MetricType::METRIC_TYPE_IP;

    auto index_param = CreateDmqParameter(false, true);
    auto index = std::make_unique<SINDI>(index_param, common_param);
    auto build_res = index->Build(data.Base());
    REQUIRE(build_res.size() == 1);
    REQUIRE(build_res[0] == data.labels[1]);
    REQUIRE_THROWS_AS(index->Add(data.Base()), VsagException);
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE_THROWS_AS(index->UpdateVector(data.labels[0], data.Query()), VsagException);

    std::stringstream stream;
    REQUIRE_NOTHROW(index->SerializeStreaming(stream));
    const auto bytes = stream.str();

    auto restored = std::make_unique<SINDI>(index_param, common_param);
    std::stringstream deserialize_stream(bytes);
    REQUIRE_NOTHROW(restored->DeserializeStreaming(deserialize_stream));

    std::stringstream load_stream(bytes);
    auto loaded = Index::Load(load_stream, "{}");
    REQUIRE(loaded.has_value());

    auto query = data.Query();
    auto expected = index->KnnSearch(query, 3, kDmqSearchParameters, nullptr);
    auto restored_result = restored->KnnSearch(query, 3, kDmqSearchParameters, nullptr);
    auto loaded_result = loaded.value()->KnnSearch(query, 3, kDmqSearchParameters).value();
    RequireSameResults(expected, restored_result);
    RequireSameResults(expected, loaded_result);
}
