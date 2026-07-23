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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <initializer_list>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "unittest.h"
#include "vsag/options.h"

namespace {

vsag::DatasetPtr
MakeFloatDataset(std::vector<float>& vectors,
                 std::vector<int64_t>& ids,
                 int64_t dim,
                 int64_t count) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    return dataset;
}

vsag::DatasetPtr
MakeFloatDatasetWithSourceIds(std::vector<float>& vectors,
                              std::vector<int64_t>& ids,
                              std::vector<std::string>& source_ids,
                              int64_t dim,
                              int64_t count) {
    auto dataset = MakeFloatDataset(vectors, ids, dim, count);
    dataset->SourceID(source_ids.data());
    return dataset;
}

vsag::DatasetPtr
MakeFloatQuery(std::vector<float>& vector, int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(1)->Dim(dim)->Float32Vectors(vector.data())->Owner(false);
    return dataset;
}

class BlockSizeLimitGuard {
public:
    explicit BlockSizeLimitGuard(uint64_t block_size_limit)
        : old_block_size_limit_(vsag::Options::Instance().block_size_limit()) {
        vsag::Options::Instance().set_block_size_limit(block_size_limit);
    }

    ~BlockSizeLimitGuard() {
        vsag::Options::Instance().set_block_size_limit(old_block_size_limit_);
    }

private:
    uint64_t old_block_size_limit_;
};

vsag::IndexCommonParam
MakeCommonParam(int64_t dim, uint64_t thread_count = 0) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    if (thread_count > 0) {
        common_param.thread_pool_ = vsag::SafeThreadPool::FactoryDefaultThreadPool();
        common_param.thread_pool_->SetPoolSize(thread_count);
    }
    return common_param;
}

vsag::JsonType
MakeFp32HGraphJson(bool deduplicate_storage = true, int64_t build_thread_count = 1) {
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true
    })");
    hgraph_json["build_thread_count"].SetInt(build_thread_count);
    hgraph_json["deduplicate_storage"].SetBool(deduplicate_storage);
    return hgraph_json;
}

vsag::JsonType
MakeFp32ReorderHGraphJson() {
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["use_reorder"].SetBool(true);
    hgraph_json["precise_quantization_type"].SetString("fp32");
    return hgraph_json;
}

vsag::JsonType
MakeRabitQRawVectorHGraphJson() {
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["base_quantization_type"].SetString("rabitq");
    hgraph_json["rabitq_bits_per_dim_base"].SetInt(1);
    hgraph_json["rabitq_bits_per_dim_query"].SetInt(32);
    hgraph_json["store_raw_vector"].SetBool(true);
    return hgraph_json;
}

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
MakeHGraphIndex(const vsag::JsonType& hgraph_json, const vsag::IndexCommonParam& common_param) {
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);
}

bool
ResultContainsId(const vsag::DatasetPtr& result, int64_t id) {
    const auto* ids = result->GetIds();
    return std::find(ids, ids + result->GetDim(), id) != ids + result->GetDim();
}

void
RequireRangeContains(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index,
                     const vsag::DatasetPtr& query,
                     std::initializer_list<int64_t> expected_ids,
                     int64_t expected_count = -1,
                     float radius = 0.01F,
                     const std::string& parameters = R"({"hgraph": {"ef_search": 32}})") {
    auto search_result = index->RangeSearch(query, radius, parameters);
    REQUIRE(search_result.has_value());
    if (expected_count >= 0) {
        REQUIRE(search_result.value()->GetDim() == expected_count);
    }
    for (auto expected_id : expected_ids) {
        REQUIRE(ResultContainsId(search_result.value(), expected_id));
    }
}

const std::string kBruteForceSearchParams =
    R"({"hgraph": {"ef_search": 32, "brute_force_threshold": 1.0}})";

}  // namespace

TEST_CASE("HGraph exact duplicate fallback supports every dense data type",
          "[ut][hgraph][duplicate][data_type]") {
    constexpr int64_t dim = 8;
    auto hgraph_json = MakeFp32HGraphJson();
    std::vector<int64_t> base_ids = {10, 20};
    std::vector<int64_t> duplicate_id = {30};

    SECTION("int8") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_INT8;
        hgraph_json["base_quantization_type"].SetString("int8");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<int8_t> base_vectors = {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8};
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Int8Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Int8Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }

    SECTION("fp16") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FP16;
        hgraph_json["base_quantization_type"].SetString("fp16");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<uint16_t> base_vectors(static_cast<uint64_t>(dim * 2));
        for (int64_t i = 0; i < dim; ++i) {
            base_vectors[dim + i] = vsag::generic::FloatToFP16(static_cast<float>(i + 1));
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Float16Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Float16Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }

    SECTION("bf16") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_BF16;
        hgraph_json["base_quantization_type"].SetString("bf16");
        auto index = MakeHGraphIndex(hgraph_json, common_param);
        std::vector<uint16_t> base_vectors(static_cast<uint64_t>(dim * 2));
        for (int64_t i = 0; i < dim; ++i) {
            base_vectors[dim + i] = vsag::generic::FloatToBF16(static_cast<float>(i + 1));
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(2)
            ->Dim(dim)
            ->Ids(base_ids.data())
            ->Float16Vectors(base_vectors.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());

        auto duplicate = vsag::Dataset::Make();
        duplicate->NumElements(1)
            ->Dim(dim)
            ->Ids(duplicate_id.data())
            ->Float16Vectors(base_vectors.data() + dim)
            ->Owner(false);
        REQUIRE(index->Add(duplicate).has_value());
        RequireRangeContains(index, duplicate, {20, 30}, 2);
    }
}

TEST_CASE("HGraph deduplicate_storage rejects non-dense vector representations",
          "[ut][hgraph][duplicate][data_type][config]") {
    constexpr int64_t dim = 16;
    auto hgraph_json = MakeFp32HGraphJson();

    SECTION("sparse") {
        auto common_param = MakeCommonParam(dim);
        common_param.data_type_ = vsag::DataTypes::DATA_TYPE_SPARSE;
        common_param.metric_ = vsag::MetricType::METRIC_TYPE_IP;
        hgraph_json["base_quantization_type"].SetString("sparse");
        REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
    }

    SECTION("multi-vector") {
        auto common_param = MakeCommonParam(dim);
        common_param.repr_ = vsag::RecordRepr::MULTI_VECTOR;
        REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
    }
}

TEST_CASE("HGraph Add exact duplicate keeps duplicate label searchable",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = allocator;

    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    std::vector<float> duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    auto add_result = index->Add(duplicate);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->CheckIdExist(duplicate_id[0]));

    auto search_result = index->RangeSearch(duplicate, 0.01F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 2);

    const auto* result_ids = search_result.value()->GetIds();
    std::vector<int64_t> returned_ids(result_ids, result_ids + search_result.value()->GetDim());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), base_ids[1]) != returned_ids.end());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), duplicate_id[0]) !=
            returned_ids.end());

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);
    auto deserialize_result = restored->Deserialize(binary.value());
    REQUIRE(deserialize_result.has_value());
    REQUIRE(restored->CheckIdExist(duplicate_id[0]));

    auto restored_search_result =
        restored->RangeSearch(duplicate, 0.01F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(restored_search_result.has_value());
    REQUIRE(restored_search_result.value()->GetDim() == 2);

    const auto* restored_result_ids = restored_search_result.value()->GetIds();
    std::vector<int64_t> restored_returned_ids(
        restored_result_ids, restored_result_ids + restored_search_result.value()->GetDim());
    REQUIRE(std::find(restored_returned_ids.begin(), restored_returned_ids.end(), base_ids[1]) !=
            restored_returned_ids.end());
    REQUIRE(std::find(restored_returned_ids.begin(),
                      restored_returned_ids.end(),
                      duplicate_id[0]) != restored_returned_ids.end());
}

TEST_CASE("HGraph RabitQ Build exact duplicate with temporary read codes",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 960;
    constexpr int64_t count = 3;
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();

    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = allocator;

    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "support_duplicate": true,
        "deduplicate_storage": true,
        "rabitq_bits_per_dim_base": 1,
        "rabitq_bits_per_dim_query": 32
    })");
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    std::vector<float> vectors(static_cast<uint64_t>(dim * count));
    for (int64_t d = 0; d < dim; ++d) {
        vectors[d] = static_cast<float>(d % 17) * 0.01F;
        vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
        vectors[2 * dim + d] = vectors[dim + d];
    }
    std::vector<int64_t> ids = {10, 20, 30};
    auto base = MakeFloatDataset(vectors, ids, dim, count);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->CheckIdExist(ids[2]));

    std::vector<float> query(vectors.begin() + dim, vectors.begin() + 2 * dim);
    std::vector<int64_t> query_id = {100};
    auto query_dataset = MakeFloatDataset(query, query_id, dim, 1);
    auto search_result =
        index->RangeSearch(query_dataset, 1000000.0F, R"({"hgraph": {"ef_search": 16}})");
    REQUIRE(search_result.has_value());

    const auto* result_ids = search_result.value()->GetIds();
    std::vector<int64_t> returned_ids(result_ids, result_ids + search_result.value()->GetDim());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), ids[1]) != returned_ids.end());
    REQUIRE(std::find(returned_ids.begin(), returned_ids.end(), ids[2]) != returned_ids.end());
}

TEST_CASE("HGraph fp32 Build exact duplicate with deduplicate_storage",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.0F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30, 40};
    auto base = MakeFloatDataset(vectors, ids, dim, 4);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> query_vector = {1.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage uses representative vector for approximate duplicates",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["duplicate_distance_threshold"].SetFloat(0.0001F);
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.005F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30, 40};
    auto base = MakeFloatDataset(vectors, ids, dim, 4);

    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());
    REQUIRE(index->GetNumElements() == 4);

    std::vector<float> query_vector = {1.005F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);

    auto representative_distance = index->CalcDistanceById(query_vector.data(), ids[1]);
    auto approximate_duplicate_distance = index->CalcDistanceById(query_vector.data(), ids[2]);
    REQUIRE(representative_distance.has_value());
    REQUIRE(approximate_duplicate_distance.has_value());
    REQUIRE(approximate_duplicate_distance.value() == representative_distance.value());
}

TEST_CASE("HGraph deduplicate_storage keeps physical flatten capacity slot-based",
          "[ut][hgraph][duplicate][serialize][memory]") {
    constexpr int64_t dim = 16;
    constexpr int64_t count = 5000;
    BlockSizeLimitGuard block_size_limit_guard(256UL * 1024);
    auto common_param = MakeCommonParam(dim);

    std::vector<float> vectors(static_cast<uint64_t>(dim * count), 1.0F);
    std::vector<int64_t> ids(static_cast<uint64_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        ids[static_cast<uint64_t>(i)] = i;
    }
    auto base = MakeFloatDataset(vectors, ids, dim, count);

    auto dedup_index = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto dedup_build = dedup_index->Build(base);
    REQUIRE(dedup_build.has_value());
    REQUIRE(dedup_build.value().empty());
    auto dedup_hgraph = std::dynamic_pointer_cast<vsag::HGraph>(dedup_index->GetInnerIndex());
    REQUIRE(dedup_hgraph != nullptr);
    REQUIRE(dedup_hgraph->GetCodeStorageCounts() ==
            std::pair<vsag::InnerIdType, vsag::CodeSlotIdType>{count, 1});

    auto old_storage_index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);
    auto old_storage_build = old_storage_index->Build(base);
    REQUIRE(old_storage_build.has_value());
    REQUIRE(old_storage_build.value().empty());

    auto dedup_memory = dedup_index->GetMemoryUsageDetail();
    auto old_storage_memory = old_storage_index->GetMemoryUsageDetail();
    REQUIRE(dedup_memory.at("basic_flatten_codes") < old_storage_memory.at("basic_flatten_codes"));
    REQUIRE(dedup_memory.count("code_slot_map") == 1);
    uint64_t dedup_total = 0;
    for (const auto& item : dedup_memory) {
        dedup_total += item.second;
    }
    uint64_t old_storage_total = 0;
    for (const auto& item : old_storage_memory) {
        old_storage_total += item.second;
    }
    REQUIRE(dedup_total < old_storage_total);
}

TEST_CASE("HGraph support_duplicate without deduplicate_storage keeps old storage semantics",
          "[ut][hgraph][duplicate][regression][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson(false);
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());
    REQUIRE(index->GetNumElements() == 3);
    RequireRangeContains(index, duplicate, {20, 30}, 2);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->GetNumElements() == 3);
    RequireRangeContains(restored, duplicate, {20, 30}, 2);

    auto dedup_index = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto wrong_deserialize = dedup_index->Deserialize(binary.value());
    REQUIRE_FALSE(wrong_deserialize.has_value());
    REQUIRE(wrong_deserialize.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("HGraph deduplicate_storage Add unique then duplicate of added point",
          "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 1);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> unique_vector = {2.0F, 0.0F};
    std::vector<int64_t> unique_id = {20};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(index->Add(unique).has_value());

    std::vector<float> duplicate_vector = {2.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());
    REQUIRE(index->GetNumElements() == 3);

    auto query = MakeFloatQuery(unique_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage handles duplicate chains", "[ut][hgraph][duplicate][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 1);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector_1 = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id_1 = {20};
    auto duplicate_1 = MakeFloatDataset(duplicate_vector_1, duplicate_id_1, dim, 1);
    REQUIRE(index->Add(duplicate_1).has_value());

    std::vector<float> duplicate_vector_2 = {1.0F, 0.0F};
    std::vector<int64_t> duplicate_id_2 = {30};
    auto duplicate_2 = MakeFloatDataset(duplicate_vector_2, duplicate_id_2, dim, 1);
    REQUIRE(index->Add(duplicate_2).has_value());
    REQUIRE(index->GetNumElements() == 3);

    auto query = MakeFloatQuery(base_vectors, dim);
    RequireRangeContains(index, query, {10, 20, 30}, 3, 0.01F, kBruteForceSearchParams);
    for (auto id : {10, 20, 30}) {
        auto distance = index->CalcDistanceById(base_vectors.data(), id);
        REQUIRE(distance.has_value());
        REQUIRE(distance.value() == 0.0F);
    }
}

TEST_CASE("HGraph deduplicate_storage skips duplicate external labels without binding storage",
          "[ut][hgraph][duplicate][label][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> add_vectors = {
        3.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> add_ids = {20, 30};
    auto add = MakeFloatDataset(add_vectors, add_ids, dim, 2);
    auto add_result = index->Add(add);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value() == std::vector<int64_t>{20});
    REQUIRE(index->GetNumElements() == 3);
    REQUIRE_FALSE(index->CheckIdExist(40));

    std::vector<float> query_vector = {1.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2);
}

TEST_CASE("HGraph deduplicate_storage Build skips duplicate external labels",
          "[ut][hgraph][duplicate][label][build]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        2.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 3);
    auto build_result = index->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value() == std::vector<int64_t>{10});
    REQUIRE(index->GetNumElements() == 2);
    REQUIRE(index->CheckIdExist(10));
    REQUIRE(index->CheckIdExist(20));
}

TEST_CASE("HGraph deduplicate_storage deserialize keeps map extensible",
          "[ut][hgraph][duplicate][serialize][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> first_duplicate_vector = {1.0F, 0.0F};
    std::vector<int64_t> first_duplicate_id = {30};
    auto first_duplicate = MakeFloatDataset(first_duplicate_vector, first_duplicate_id, dim, 1);
    REQUIRE(index->Add(first_duplicate).has_value());

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());

    auto wrong_config = MakeFp32HGraphJson(false);
    auto wrong_index = MakeHGraphIndex(wrong_config, common_param);
    auto wrong_deserialize = wrong_index->Deserialize(binary.value());
    REQUIRE_FALSE(wrong_deserialize.has_value());
    REQUIRE(wrong_deserialize.error().type == vsag::ErrorType::INVALID_ARGUMENT);

    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    auto deserialize_result = restored->Deserialize(binary.value());
    REQUIRE(deserialize_result.has_value());
    REQUIRE(restored->GetNumElements() == 3);

    std::vector<float> unique_vector = {2.0F, 0.0F};
    std::vector<int64_t> unique_id = {40};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(restored->Add(unique).has_value());

    std::vector<float> second_duplicate_vector = {2.0F, 0.0F};
    std::vector<int64_t> second_duplicate_id = {50};
    auto second_duplicate = MakeFloatDataset(second_duplicate_vector, second_duplicate_id, dim, 1);
    REQUIRE(restored->Add(second_duplicate).has_value());
    REQUIRE(restored->GetNumElements() == 5);

    std::vector<float> base_duplicate_query_vector = {1.0F, 0.0F};
    auto base_duplicate_query = MakeFloatQuery(base_duplicate_query_vector, dim);
    RequireRangeContains(restored, base_duplicate_query, {20, 30}, 2);

    std::vector<float> added_duplicate_query_vector = {2.0F, 0.0F};
    auto added_duplicate_query = MakeFloatQuery(added_duplicate_query_vector, dim);
    RequireRangeContains(restored, added_duplicate_query, {40, 50}, 2);
}

TEST_CASE("HGraph deduplicate_storage rejects v0.14 serialization",
          "[ut][hgraph][duplicate][serialize]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    common_param.use_old_serial_format_ = true;
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    auto binary = index->Serialize();
    REQUIRE_FALSE(binary.has_value());
    REQUIRE(binary.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(binary.error().message.find("v0.14") != std::string::npos);
}

TEST_CASE("HGraph deduplicate_storage supports precise reorder code path",
          "[ut][hgraph][duplicate][reorder][add]") {
    constexpr int64_t dim = 4;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32ReorderHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        0.0F,
        0.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector = {1.0F, 0.0F, 0.0F, 0.0F};
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());

    auto query = MakeFloatQuery(duplicate_vector, dim);
    RequireRangeContains(index, query, {20, 30}, 2, 0.01F, kBruteForceSearchParams);

    auto knn = index->KnnSearch(
        query, 2, R"({"hgraph": {"ef_search": 16, "brute_force_threshold": 1.0}})");
    REQUIRE(knn.has_value());
    REQUIRE(knn.value()->GetDim() == 2);
    REQUIRE(ResultContainsId(knn.value(), 20));
    REQUIRE(ResultContainsId(knn.value(), 30));
}

TEST_CASE("HGraph deduplicate_storage supports RabitQ raw-vector graph read path",
          "[ut][hgraph][duplicate][raw_vector][add]") {
    constexpr int64_t dim = 960;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeRabitQRawVectorHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors(dim * 3);
    for (int64_t d = 0; d < dim; ++d) {
        base_vectors[d] = static_cast<float>(d % 17) * 0.01F;
        base_vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
        base_vectors[2 * dim + d] = 2.0F + static_cast<float>(d % 11) * 0.01F;
    }
    std::vector<int64_t> base_ids = {10, 20, 30};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 3);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector(base_vectors.begin() + dim, base_vectors.begin() + dim * 2);
    std::vector<int64_t> duplicate_id = {40};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    auto duplicate_add_result = index->Add(duplicate);
    REQUIRE(duplicate_add_result.has_value());
    REQUIRE(duplicate_add_result.value().empty());

    std::vector<float> unique_vector(dim);
    for (int64_t d = 0; d < dim; ++d) {
        unique_vector[d] = 10.0F + static_cast<float>(d % 19) * 0.02F;
    }
    std::vector<int64_t> unique_id = {50};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    auto unique_add_result = index->Add(unique);
    REQUIRE(unique_add_result.has_value());
    REQUIRE(unique_add_result.value().empty());
    REQUIRE(index->CheckIdExist(unique_id[0]));
    auto unique_distance = index->CalcDistanceById(unique_vector.data(), unique_id[0]);
    REQUIRE(unique_distance.has_value());
    REQUIRE(unique_distance.value() == 0.0F);

    auto duplicate_query = MakeFloatQuery(duplicate_vector, dim);
    RequireRangeContains(index, duplicate_query, {20, 40}, 2, 0.01F, kBruteForceSearchParams);

    auto unique_query = MakeFloatQuery(unique_vector, dim);
    RequireRangeContains(index, unique_query, {50}, 1, 0.01F, kBruteForceSearchParams);
}

TEST_CASE("HGraph deduplicate_storage Tune keeps physical slots isolated",
          "[ut][hgraph][duplicate][tune]") {
    constexpr int64_t dim = 960;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeRabitQRawVectorHGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors(dim * 2);
    for (int64_t d = 0; d < dim; ++d) {
        base_vectors[d] = static_cast<float>(d % 17) * 0.01F;
        base_vectors[dim + d] = 1.0F + static_cast<float>(d % 13) * 0.01F;
    }
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> duplicate_vector(base_vectors.begin() + dim, base_vectors.end());
    std::vector<int64_t> duplicate_id = {30};
    auto duplicate = MakeFloatDataset(duplicate_vector, duplicate_id, dim, 1);
    REQUIRE(index->Add(duplicate).has_value());

    auto tune_result = index->Tune(R"({
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "support_duplicate": true,
            "deduplicate_storage": true,
            "store_raw_vector": true
        }
    })");
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value());

    std::vector<float> unique_vector(dim);
    for (int64_t d = 0; d < dim; ++d) {
        unique_vector[d] = 10.0F + static_cast<float>(d % 19) * 0.02F;
    }
    std::vector<int64_t> unique_id = {40};
    auto unique = MakeFloatDataset(unique_vector, unique_id, dim, 1);
    REQUIRE(index->Add(unique).has_value());

    auto duplicate_base_distance =
        index->CalcDistanceById(duplicate_vector.data(), duplicate_id[0], false);
    REQUIRE(duplicate_base_distance.has_value());
    REQUIRE(duplicate_base_distance.value() == 0.0F);
}

TEST_CASE("HGraph deduplicate_storage rejects Merge", "[ut][hgraph][duplicate][merge]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();

    auto target = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE_FALSE(target->CheckFeature(vsag::IndexFeature::SUPPORT_MERGE_INDEX));
    std::vector<float> target_vectors = {0.0F, 0.0F};
    std::vector<int64_t> target_ids = {10};
    auto target_data = MakeFloatDataset(target_vectors, target_ids, dim, 1);
    REQUIRE(target->Build(target_data).has_value());

    auto source = MakeHGraphIndex(hgraph_json, common_param);
    std::vector<float> source_vectors = {4.0F, 0.0F};
    std::vector<int64_t> source_ids = {20};
    auto source_data = MakeFloatDataset(source_vectors, source_ids, dim, 1);
    REQUIRE(source->Build(source_data).has_value());

    vsag::IdMapFunction id_map = [](int64_t id) -> std::tuple<bool, int64_t> {
        return std::make_tuple(true, id);
    };
    auto merge_result = target->Merge({vsag::MergeUnit{source, id_map}});
    REQUIRE_FALSE(merge_result.has_value());
    REQUIRE(merge_result.error().message.find("deduplicate_storage") != std::string::npos);
    REQUIRE(target->GetNumElements() == 1);
}

TEST_CASE("HGraph deduplicate_storage ExportModel keeps an empty reusable model",
          "[ut][hgraph][duplicate][export_model]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);
    REQUIRE(index->Build(base).has_value());

    auto model_result = index->ExportModel();
    REQUIRE(model_result.has_value());
    auto model = model_result.value();
    REQUIRE(model->GetNumElements() == 0);

    std::vector<float> model_vectors = {
        3.0F,
        0.0F,
        3.0F,
        0.0F,
    };
    std::vector<int64_t> model_ids = {30, 40};
    auto model_data = MakeFloatDataset(model_vectors, model_ids, dim, 2);
    REQUIRE(model->Build(model_data).has_value());

    std::vector<float> query_vector = {3.0F, 0.0F};
    auto query = MakeFloatQuery(query_vector, dim);
    auto search_result = model->RangeSearch(query, 0.01F, kBruteForceSearchParams);
    REQUIRE(search_result.has_value());
    REQUIRE(search_result.value()->GetDim() == 2);
    REQUIRE(ResultContainsId(search_result.value(), 30));
    REQUIRE(ResultContainsId(search_result.value(), 40));
}

TEST_CASE("HGraph deduplicate_storage rejects UpdateVector for a shared duplicate group",
          "[ut][hgraph][duplicate][update]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors = {
        0.0F,
        0.0F,
        1.0F,
        0.0F,
        1.0F,
        0.0F,
    };
    std::vector<int64_t> ids = {10, 20, 30};
    auto base = MakeFloatDataset(vectors, ids, dim, 3);
    REQUIRE(index->Build(base).has_value());

    std::vector<float> updated_vector = {3.0F, 0.0F};
    std::vector<int64_t> update_id = {30};
    auto update_data = MakeFloatDataset(updated_vector, update_id, dim, 1);
    auto update_result = index->UpdateVector(update_id[0], update_data, true);
    REQUIRE_FALSE(update_result.has_value());
    REQUIRE(update_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    std::vector<float> original_duplicate_vector = {1.0F, 0.0F};
    auto representative_distance =
        index->CalcDistanceById(original_duplicate_vector.data(), 20, false);
    REQUIRE(representative_distance.has_value());
    REQUIRE(representative_distance.value() == 0.0F);

    auto duplicate_distance = index->CalcDistanceById(original_duplicate_vector.data(), 30, false);
    REQUIRE(duplicate_distance.has_value());
    REQUIRE(duplicate_distance.value() == 0.0F);

    std::vector<float> unique_update_vector = {4.0F, 0.0F};
    std::vector<int64_t> unique_update_id = {10};
    auto unique_update_data = MakeFloatDataset(unique_update_vector, unique_update_id, dim, 1);
    auto unique_update_result = index->UpdateVector(unique_update_id[0], unique_update_data, true);
    REQUIRE(unique_update_result.has_value());
    REQUIRE(unique_update_result.value());
    auto unique_distance =
        index->CalcDistanceById(unique_update_vector.data(), unique_update_id[0], false);
    REQUIRE(unique_distance.has_value());
    REQUIRE(unique_distance.value() == 0.0F);
}

TEST_CASE("HGraph deduplicate_storage batch Add supports internal parallel add",
          "[ut][hgraph][duplicate][parallel][add]") {
    constexpr int64_t dim = 4;
    constexpr uint64_t build_threads = 4;
    auto common_param = MakeCommonParam(dim, build_threads);
    auto hgraph_json = MakeFp32HGraphJson(true, static_cast<int64_t>(build_threads));
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> seed_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        100.0F,
        0.0F,
        0.0F,
        0.0F,
        200.0F,
        0.0F,
        0.0F,
        0.0F,
        300.0F,
        0.0F,
        0.0F,
        0.0F,
    };
    std::vector<int64_t> seed_ids = {10, 20, 30, 40};
    auto seed = MakeFloatDataset(seed_vectors, seed_ids, dim, 4);
    REQUIRE(index->Build(seed).has_value());

    constexpr int64_t add_count = 8;
    std::vector<float> add_vectors(dim * add_count);
    std::vector<int64_t> add_ids(add_count);
    for (int64_t i = 0; i < add_count; ++i) {
        add_ids[i] = 100 + i;
        if (i < 4) {
            std::copy(seed_vectors.begin() + dim,
                      seed_vectors.begin() + dim * 2,
                      add_vectors.begin() + i * dim);
            continue;
        }
        for (int64_t d = 0; d < dim; ++d) {
            add_vectors[i * dim + d] = static_cast<float>(1000 + i * 100 + d);
        }
    }

    auto batch = MakeFloatDataset(add_vectors, add_ids, dim, add_count);
    auto add_result = index->Add(batch);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->GetNumElements() == 12);
    for (auto id : add_ids) {
        REQUIRE(index->CheckIdExist(id));
    }

    std::vector<float> duplicate_query_vector(seed_vectors.begin() + dim,
                                              seed_vectors.begin() + dim * 2);
    auto duplicate_query = MakeFloatQuery(duplicate_query_vector, dim);
    RequireRangeContains(index, duplicate_query, {20, 100, 101, 102, 103}, 5);

    std::vector<float> unique_query_vector(add_vectors.begin() + dim * 4,
                                           add_vectors.begin() + dim * 5);
    auto unique_query = MakeFloatQuery(unique_query_vector, dim);
    RequireRangeContains(index, unique_query, {104}, 1);
}

TEST_CASE("HGraph deduplicate_storage physical growth is single-flight",
          "[ut][hgraph][duplicate][parallel][resize]") {
    constexpr int64_t dim = 128;
    constexpr int64_t base_count = 500;
    constexpr uint64_t search_threads = 8;
    constexpr uint64_t resize_threads = 32;
    constexpr vsag::InnerIdType required_capacity = 513;
    BlockSizeLimitGuard block_size_limit_guard(256UL * 1024);

    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors(dim * base_count, 0.0F);
    std::vector<int64_t> ids(base_count);
    for (int64_t i = 0; i < base_count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
        ids[i] = i;
    }
    auto base = MakeFloatDataset(vectors, ids, dim, base_count);
    REQUIRE(index->Build(base).has_value());

    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);

    std::vector<float> query_vector(dim, 0.0F);
    auto query = MakeFloatQuery(query_vector, dim);
    std::atomic<bool> keep_searching{true};
    std::atomic<uint64_t> searchers_started{0};
    std::vector<std::future<bool>> searchers;
    searchers.reserve(search_threads);
    for (uint64_t i = 0; i < search_threads; ++i) {
        searchers.emplace_back(std::async(std::launch::async, [&]() {
            searchers_started.fetch_add(1, std::memory_order_release);
            while (keep_searching.load(std::memory_order_acquire)) {
                auto result = index->KnnSearch(query, 10, R"({"hgraph": {"ef_search": 128}})");
                if (not result.has_value()) {
                    return false;
                }
            }
            return true;
        }));
    }
    while (searchers_started.load(std::memory_order_acquire) < search_threads) {
        std::this_thread::yield();
    }

    std::atomic<uint64_t> resizers_ready{0};
    std::atomic<bool> start_resizing{false};
    std::vector<std::future<void>> resizers;
    resizers.reserve(resize_threads);
    for (uint64_t i = 0; i < resize_threads; ++i) {
        resizers.emplace_back(std::async(std::launch::async, [&]() {
            resizers_ready.fetch_add(1, std::memory_order_release);
            while (not start_resizing.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            hgraph->ensure_physical_code_capacity(required_capacity);
        }));
    }
    while (resizers_ready.load(std::memory_order_acquire) < resize_threads) {
        std::this_thread::yield();
    }
    start_resizing.store(true, std::memory_order_release);

    bool all_resizers_finished = true;
    for (auto& resizer : resizers) {
        if (resizer.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
            all_resizers_finished = false;
            break;
        }
    }
    keep_searching.store(false, std::memory_order_release);
    for (auto& searcher : searchers) {
        REQUIRE(searcher.get());
    }
    for (auto& resizer : resizers) {
        resizer.get();
    }
    REQUIRE(all_resizers_finished);
}

TEST_CASE("HGraph deduplicate_storage concurrent Add keeps visible duplicates searchable",
          "[ut][hgraph][duplicate][concurrent][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> seed_vectors = {
        0.0F,
        0.0F,
        100.0F,
        0.0F,
    };
    std::vector<int64_t> seed_ids = {10, 20};
    auto seed = MakeFloatDataset(seed_vectors, seed_ids, dim, 2);
    REQUIRE(index->Build(seed).has_value());

    constexpr int64_t add_count = 10;
    std::vector<std::future<bool>> futures;
    futures.reserve(add_count);
    for (int64_t i = 0; i < add_count; ++i) {
        futures.emplace_back(std::async(std::launch::async, [index, i, dim]() {
            std::vector<float> vector(dim);
            if (i % 2 == 0) {
                vector[0] = 0.0F;
                vector[1] = 0.0F;
            } else {
                vector[0] = static_cast<float>(1000 + i * 100);
                vector[1] = 0.0F;
            }
            std::vector<int64_t> id = {1000 + i};
            auto dataset = MakeFloatDataset(vector, id, dim, 1);
            return index->Add(dataset).has_value();
        }));
    }
    for (auto& future : futures) {
        REQUIRE(future.get());
    }

    REQUIRE(index->GetNumElements() == 12);
    for (int64_t i = 0; i < add_count; ++i) {
        REQUIRE(index->CheckIdExist(1000 + i));
    }

    std::vector<float> duplicate_query_vector = {0.0F, 0.0F};
    auto duplicate_query = MakeFloatQuery(duplicate_query_vector, dim);
    RequireRangeContains(index,
                         duplicate_query,
                         {10, 1000, 1002, 1004, 1006, 1008},
                         6,
                         0.01F,
                         kBruteForceSearchParams);

    std::vector<float> unique_query_vector = {1100.0F, 0.0F};
    auto unique_query = MakeFloatQuery(unique_query_vector, dim);
    RequireRangeContains(index, unique_query, {1001}, 1, 0.01F, kBruteForceSearchParams);
}

TEST_CASE("HGraph deduplicate_storage rejects cache accelerated Build",
          "[ut][hgraph][duplicate][cache]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 4;
    auto common_param = MakeCommonParam(dim);

    std::vector<float> vectors(dim * count);
    for (int64_t i = 0; i < count; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>(i * 10 + d);
        }
    }
    std::vector<int64_t> ids = {10, 20, 30, 40};
    std::vector<std::string> source_ids = {"sid_10", "sid_20", "sid_30", "sid_40"};
    auto base = MakeFloatDatasetWithSourceIds(vectors, ids, source_ids, dim, count);

    auto cache_source_json = MakeFp32HGraphJson(false);
    auto cache_source = MakeHGraphIndex(cache_source_json, common_param);
    REQUIRE(cache_source->Build(base).has_value());

    std::stringstream cache_stream;
    auto export_result = cache_source->ExportCache(cache_stream);
    REQUIRE(export_result.has_value());
    cache_stream.seekg(0);

    auto warmed = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    auto import_result = warmed->ImportCache(cache_stream);
    REQUIRE(import_result.has_value());

    auto build_result = warmed->Build(base);
    REQUIRE_FALSE(build_result.has_value());
    REQUIRE(build_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(build_result.error().message.find("deduplicate_storage") != std::string::npos);
}

TEST_CASE("HGraph deduplicate_storage rejects unsupported graph type",
          "[ut][hgraph][duplicate][config]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["graph_type"].SetString("odescent");

    REQUIRE_THROWS(MakeHGraphIndex(hgraph_json, common_param));
}
