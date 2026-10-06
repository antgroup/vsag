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
#include <cmath>
#include <cstring>
#include <functional>
#include <future>
#include <initializer_list>
#include <new>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "hgraph.h"
#include "hgraph_fast_build.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io.h"
#include "io/memory_io/memory_io_parameter.h"
#include "quantization/fp32_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/serialization.h"
#include "storage/streaming_serialization_test_utils.h"
#include "unittest.h"
#include "vsag/deserialize_reader.h"
#include "vsag/options.h"
#include "vsag/serialize_writer.h"

namespace vsag {

// Fault injection for partial persistent-code writes, without changing the public API.
class HGraphCodeWriteTestAccess {
public:
    static void
    SetPending(HGraph& graph, InnerIdType id, bool base, bool precise, bool raw) {
        graph.pending_code_writes_[id] = (base ? HGraph::BASE_CODE : 0) |
                                         (precise ? HGraph::PRECISE_CODE : 0) |
                                         (raw ? HGraph::RAW_CODE : 0);
    }

    static uint8_t
    Pending(const HGraph& graph, InnerIdType id) {
        const auto it = graph.pending_code_writes_.find(id);
        return it == graph.pending_code_writes_.end() ? 0 : it->second;
    }

    static uint64_t
    PendingCount(const HGraph& graph) {
        return graph.pending_code_writes_.size();
    }

    static InnerIdType
    InnerId(const HGraph& graph, int64_t label) {
        return graph.label_table_->GetIdByLabel(label);
    }

    static void
    SetPrecise(HGraph& graph, const FlattenInterfacePtr& codes) {
        graph.high_precise_codes_ = codes;
    }

    static bool
    CodesLocked(HGraph& graph) {
        if (graph.persistent_codes_mutex_.try_lock_shared()) {
            graph.persistent_codes_mutex_.unlock_shared();
            return false;
        }
        return true;
    }

    static void
    Move(HGraph& graph, InnerIdType from, InnerIdType to) {
        graph.move_id(from, to);
    }
};

}  // namespace vsag

namespace {

class ArmableRejectingThreadPool final : public vsag::ThreadPool {
public:
    void
    WaitUntilEmpty() override {
    }

    void
    SetQueueSizeLimit(uint64_t) override {
    }

    void
    SetPoolSize(uint64_t) override {
    }

    std::future<void>
    Enqueue(std::function<void(void)> task) override {
        if (reject_submissions_.load(std::memory_order_acquire)) {
            throw std::bad_alloc();
        }
        task();
        return {};
    }

    void
    SetRejectSubmissions(bool reject) {
        reject_submissions_.store(reject, std::memory_order_release);
    }

private:
    std::atomic<bool> reject_submissions_{false};
};

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

TEST_CASE("HGraph UpdateId preserves equal-label no-op", "[ut][hgraph][update]") {
    auto common_param = MakeCommonParam(8);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);

    auto result = index->UpdateId(404, 404);
    REQUIRE(result.has_value());
    REQUIRE(result.value());
}

TEST_CASE("HGraph non-fused SetImmutable does not require a fused searcher",
          "[ut][hgraph][immutable]") {
    constexpr int64_t dim = 8;
    auto common_param = MakeCommonParam(dim);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(false), common_param);
    std::vector<float> vectors(static_cast<uint64_t>(dim) * 2, 0.0F);
    vectors[dim] = 1.0F;
    std::vector<int64_t> ids{10, 20};
    auto base = MakeFloatDataset(vectors, ids, dim, 2);

    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->SetImmutable().has_value());

    auto query = MakeFloatQuery(vectors, dim);
    auto result = index->KnnSearch(query, 1, kBruteForceSearchParams);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetIds()[0] == ids[0]);
}

TEST_CASE("HGraph Tune requires a complete source after Add",
          "[ut][hgraph][add][hgraph_tune_incomplete_source]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 2;
    const auto source = GENERATE(std::string("raw"), std::string("precise"), std::string("base"));
    const bool reject_add = GENERATE(true, false);
    const bool dedup = GENERATE(false, true);
    const bool add_after_failure = GENERATE(false, true);
    CAPTURE(source, reject_add, dedup, add_after_failure);

    auto rejecting_pool = std::make_shared<ArmableRejectingThreadPool>();
    auto common_param = MakeCommonParam(dim);
    common_param.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(rejecting_pool);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "sq8",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "store_raw_vector": true
    })");
    if (source == "precise") {
        hgraph_json["store_raw_vector"].SetBool(false);
        hgraph_json["use_reorder"].SetBool(true);
        hgraph_json["precise_quantization_type"].SetString("fp32");
    } else if (source == "base") {
        hgraph_json["store_raw_vector"].SetBool(false);
        hgraph_json["base_quantization_type"].SetString("fp32");
    }
    hgraph_json["support_duplicate"].SetBool(dedup);
    hgraph_json["deduplicate_storage"].SetBool(dedup);
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> base_vectors = {
        0.0F,
        0.0F,
        0.0F,
        0.0F,
        1.0F,
        1.0F,
        1.0F,
        1.0F,
    };
    std::vector<int64_t> base_ids = {10, 20};
    auto base = MakeFloatDataset(base_vectors, base_ids, dim, base_count);
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->GetNumElements() == base_count);

    // Binary coordinates are represented exactly by both SQ8 and BF16.
    std::vector<float> add_vectors = {1.0F, 0.0F, 0.0F, 0.0F};
    std::vector<int64_t> add_ids = {30};
    auto add = MakeFloatDataset(add_vectors, add_ids, dim, 1);

    rejecting_pool->SetRejectSubmissions(reject_add);
    auto add_result = index->Add(add);
    rejecting_pool->SetRejectSubmissions(false);

    if (reject_add) {
        REQUIRE_FALSE(add_result.has_value());
        REQUIRE(add_result.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);
        // The label is reserved before enqueue, but its vector and links were never written.
        REQUIRE(index->GetNumElements() == base_count);
    } else {
        REQUIRE(add_result.has_value());
        REQUIRE(add_result.value().empty());
        REQUIRE(index->GetNumElements() == base_count + 1);
    }
    REQUIRE(index->CheckIdExist(add_ids[0]));

    if (add_after_failure) {
        std::vector<float> later_vectors = {1.0F, 1.0F, 0.0F, 0.0F};
        std::vector<int64_t> later_ids{40};
        auto later = index->Add(MakeFloatDataset(later_vectors, later_ids, dim, 1));
        REQUIRE(later.has_value());
        REQUIRE(later.value().empty());
    }
    const auto count_before_tune = index->GetNumElements();

    auto query = MakeFloatQuery(base_vectors, dim);
    const std::string search_params =
        add_after_failure ? R"({"hgraph":{"ef_search":32}})" : kBruteForceSearchParams;
    auto before = index->KnnSearch(query, base_count + 2, search_params);
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetDim() ==
            base_count + (reject_add ? 0 : 1) + (add_after_failure ? 1 : 0));

    auto tune_result = index->Tune(R"({
        "index_param": {
            "base_quantization_type": "bf16",
            "max_degree": 8,
            "ef_construction": 32
        }
    })",
                                   true);
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value() == not reject_add);
    REQUIRE(index->GetNumElements() == count_before_tune);
    auto after = index->KnnSearch(query, base_count + 2, search_params);
    REQUIRE(after.has_value());
    REQUIRE(after.value()->GetDim() == before.value()->GetDim());
    for (int64_t i = 0; i < before.value()->GetDim(); ++i) {
        REQUIRE(after.value()->GetIds()[i] == before.value()->GetIds()[i]);
        REQUIRE(after.value()->GetDistances()[i] == before.value()->GetDistances()[i]);
    }
    if (reject_add) {
        // Rejection must not disable future tuning, even when explicitly requested above.
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_TUNE));
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        auto restored = MakeHGraphIndex(hgraph_json, common_param);
        auto restored_result = restored->Deserialize(binary.value());
        if (dedup) {
            // Deduplicated deserialization already rejects an unbound logical ID.
            REQUIRE_FALSE(restored_result.has_value());
            return;
        }
        REQUIRE(restored_result.has_value());
        auto restored_tune =
            restored->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})", true);
        REQUIRE(restored_tune.has_value());
        REQUIRE_FALSE(restored_tune.value());
        REQUIRE(restored->CheckFeature(vsag::IndexFeature::SUPPORT_TUNE));
    }
}

TEST_CASE("HGraph restore preserves unfinished reservations before a later Add",
          "[ut][hgraph][add][hgraph_tune_incomplete_source][hgraph_restore_reservation]") {
    const auto restore_mode =
        GENERATE(std::string("ordinary"), std::string("streaming"), std::string("parallel"));
    CAPTURE(restore_mode);
    auto pool = std::make_shared<ArmableRejectingThreadPool>();
    auto common = MakeCommonParam(4);
    common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    auto parameters = MakeFp32HGraphJson(false);
    auto index = MakeHGraphIndex(parameters, common);
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> labels{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, labels, 4, 2)).has_value());
    std::vector<float> missing{1, 0, 0, 0};
    std::vector<int64_t> failed_label{30};
    pool->SetRejectSubmissions(true);
    const auto failed = index->Add(MakeFloatDataset(missing, failed_label, 4, 1));
    pool->SetRejectSubmissions(false);
    REQUIRE_FALSE(failed.has_value());
    REQUIRE(failed.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY);

    auto restored = MakeHGraphIndex(parameters, MakeCommonParam(4));
    if (restore_mode == "streaming") {
        std::stringstream snapshot;
        REQUIRE(index->SerializeStreaming(snapshot).has_value());
        REQUIRE(restored->DeserializeStreaming(snapshot).has_value());
    } else if (restore_mode == "parallel") {
        class MemorySnapshot : public vsag::SerializeWriter, public vsag::DeserializeReader {
        public:
            void
            Write(const char* data, uint64_t size) override {
                bytes.append(data, size);
            }
            [[nodiscard]] uint64_t
            Size() const override {
                return bytes.size();
            }
            void
            Read(uint64_t offset, uint64_t size, void* dest) override {
                if (size > bytes.size() or offset > bytes.size() - size) {
                    throw std::out_of_range("snapshot read");
                }
                std::memcpy(dest, bytes.data() + offset, size);
            }
            std::string bytes;
        } snapshot;
        REQUIRE(index->Serialize(snapshot, 4096).has_value());
        REQUIRE(restored->ParallelDeserialize(snapshot).has_value());
    } else {
        auto binary = index->Serialize();
        REQUIRE(binary.has_value());
        REQUIRE(restored->Deserialize(binary.value()).has_value());
    }
    // Restoring reservations must not make the unwritten tail searchable.
    REQUIRE(restored->GetNumElements() == 2);
    auto before = restored->KnnSearch(MakeFloatQuery(vectors, 4), 4, kBruteForceSearchParams);
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetDim() == 2);
    REQUIRE_FALSE(ResultContainsId(before.value(), 30));

    std::vector<float> later{1, 1, 0, 0};
    std::vector<int64_t> later_label{40};
    auto added = restored->Add(MakeFloatDataset(later, later_label, 4, 1));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());
    auto graph = std::static_pointer_cast<vsag::HGraph>(restored->GetInnerIndex());
    REQUIRE(vsag::HGraphCodeWriteTestAccess::InnerId(*graph, 30) == 2);
    REQUIRE(vsag::HGraphCodeWriteTestAccess::InnerId(*graph, 40) == 3);
    REQUIRE(vsag::HGraphCodeWriteTestAccess::Pending(*graph, 2) != 0);
    REQUIRE(vsag::HGraphCodeWriteTestAccess::Pending(*graph, 3) == 0);
    const auto tuned = restored->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})", true);
    REQUIRE(tuned.has_value());
    REQUIRE_FALSE(tuned.value());
    REQUIRE(restored->CheckFeature(vsag::IndexFeature::SUPPORT_TUNE));
}

TEST_CASE("HGraph Tune accepts complete shared code slots",
          "[ut][hgraph][add][hgraph_tune_incomplete_source]") {
    constexpr int64_t dim = 4;
    auto common_param = MakeCommonParam(dim);
    auto index = MakeHGraphIndex(MakeFp32HGraphJson(), common_param);
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> ids{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, dim, 2)).has_value());
    std::vector<float> duplicate(dim, 1.0F);
    std::vector<int64_t> duplicate_ids{30};
    auto added = index->Add(MakeFloatDataset(duplicate, duplicate_ids, dim, 1));
    REQUIRE(added.has_value());
    REQUIRE(added.value().empty());

    // Three logical IDs share two physical slots; physical count is not logical coverage.
    auto tuned = index->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})", true);
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());
    REQUIRE(index->GetNumElements() == 3);
    RequireRangeContains(index, MakeFloatQuery(duplicate, dim), {20, 30}, 2);
}

TEST_CASE("HGraph Tune coverage follows code compaction",
          "[ut][hgraph][add][hgraph_tune_incomplete_source]") {
    constexpr int64_t dim = 4;
    const int64_t removed_id = GENERATE(10, 30);
    const bool complete_tail = GENERATE(false, true);
    CAPTURE(removed_id, complete_tail);
    auto pool = std::make_shared<ArmableRejectingThreadPool>();
    auto common_param = MakeCommonParam(dim);
    common_param.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    auto parameters = MakeFp32HGraphJson(false);
    parameters["support_force_remove"].SetBool(true);
    auto index = MakeHGraphIndex(parameters, common_param);
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> ids{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, dim, 2)).has_value());
    std::vector<float> added_vector{1, 0, 0, 0};
    std::vector<int64_t> added_id{30};
    pool->SetRejectSubmissions(true);
    REQUIRE_FALSE(index->Add(MakeFloatDataset(added_vector, added_id, dim, 1)).has_value());
    pool->SetRejectSubmissions(false);
    if (complete_tail) {
        added_id[0] = 40;
        REQUIRE(index->Add(MakeFloatDataset(added_vector, added_id, dim, 1)).has_value());
    }
    auto removed = index->Remove({removed_id}, vsag::RemoveMode::FORCE_REMOVE);
    REQUIRE(removed.has_value());
    REQUIRE(removed.value() == 1);
    added_id[0] = 50;
    REQUIRE(index->Add(MakeFloatDataset(added_vector, added_id, dim, 1)).has_value());
    auto tuned = index->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})");
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value() == (removed_id == 30));
}

TEST_CASE("HGraph Tune tracks the retained raw storage after an alias changes",
          "[ut][hgraph][hgraph_tune_incomplete_source][hgraph_tune_review]") {
    const bool drop_precise = GENERATE(false, true);
    auto parameters = MakeFp32ReorderHGraphJson();
    parameters["deduplicate_storage"].SetBool(false);
    parameters["store_raw_vector"].SetBool(true);
    auto index = MakeHGraphIndex(parameters, MakeCommonParam(4));
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> ids{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, 4, 2)).has_value());
    auto graph = std::static_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    // Model an interrupted base write: raw aliases base, but precise is complete.
    vsag::HGraphCodeWriteTestAccess::SetPending(*graph, 1, true, false, true);
    const auto tuned = index->Tune(
        drop_precise ? R"({"index_param":{"base_quantization_type":"bf16"}})"
                     : R"({"index_param":{"base_quantization_type":"bf16","use_reorder":true,
                                "precise_quantization_type":"fp32"}})");
    REQUIRE(tuned.has_value());
    REQUIRE(tuned.value());
    // Keeping precise rebinds raw to complete precise storage. Dropping precise instead
    // retains the incomplete old base as raw; rebuilding base cannot repair that old object.
    REQUIRE(vsag::HGraphCodeWriteTestAccess::Pending(*graph, 1) == (drop_precise ? 4 : 0));
    const auto again = index->Tune(R"({"index_param":{"base_quantization_type":"fp32"}})");
    REQUIRE(again.has_value());
    REQUIRE(again.value() == not drop_precise);
}

TEST_CASE("HGraph code compaction preserves coverage when a later move throws",
          "[ut][hgraph][hgraph_tune_incomplete_source][hgraph_tune_review]") {
    using Cell = vsag::FlattenDataCell<vsag::FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                       vsag::FixedLayout<vsag::MemoryIO>>;
    class ThrowingMoveCell : public Cell {
    public:
        using Cell::Cell;
        void
        Move(vsag::InnerIdType /*from*/, vsag::InnerIdType /*to*/) override {
            throw std::bad_alloc();
        }
    };
    const bool incomplete_source = GENERATE(false, true);
    auto common = MakeCommonParam(4);
    auto parameters = MakeFp32ReorderHGraphJson();
    parameters["deduplicate_storage"].SetBool(false);
    parameters["store_raw_vector"].SetBool(true);
    auto index = MakeHGraphIndex(parameters, common);
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> ids{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, 4, 2)).has_value());
    auto graph = std::static_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    vsag::HGraphCodeWriteTestAccess::SetPending(*graph, 0, true, false, true);
    vsag::HGraphCodeWriteTestAccess::SetPending(
        *graph, 1, incomplete_source, false, incomplete_source);
    auto failing =
        std::make_shared<ThrowingMoveCell>(std::make_shared<vsag::FP32QuantizerParameter>(),
                                           std::make_shared<vsag::MemoryIOParameter>(),
                                           common);
    vsag::HGraphCodeWriteTestAccess::SetPrecise(*graph, failing);
    REQUIRE_THROWS_AS(vsag::HGraphCodeWriteTestAccess::Move(*graph, 1, 0), std::bad_alloc);
    // Base (and its raw alias) moved successfully, so they inherit source coverage even
    // though precise Move failed. Only precise is conservatively unknown after its exception.
    REQUIRE(vsag::HGraphCodeWriteTestAccess::Pending(*graph, 0) == (incomplete_source ? 7 : 2));
    REQUIRE(vsag::HGraphCodeWriteTestAccess::Pending(*graph, 1) == (incomplete_source ? 5 : 0));
}

TEST_CASE("HGraph Tune and brute force search finish concurrently",
          "[ut][hgraph][hgraph_tune_review]") {
    auto parameters = MakeFp32HGraphJson(false);
    parameters["store_raw_vector"].SetBool(true);
    auto index = MakeHGraphIndex(parameters, MakeCommonParam(4));
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> ids{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, ids, 4, 2)).has_value());
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(4)->Float32Vectors(new float[4]{0, 0, 0, 0})->Owner(true);
    class PausingFilter : public vsag::Filter {
    public:
        bool
        CheckValid(int64_t /*id*/) const override {
            return true;
        }
        float
        ValidRatio() const override {
            entered.store(true);
            while (not released.load()) {
                std::this_thread::yield();
            }
            return 1.0F;
        }
        mutable std::atomic<bool> entered{false};
        std::atomic<bool> released{false};
    };
    auto filter = std::make_shared<PausingFilter>();
    auto start = std::make_shared<std::atomic<bool>>(false);
    std::packaged_task<bool()> search([index, query, filter] {
        for (uint64_t i = 0; i < 1000; ++i) {
            auto result = index->KnnSearch(query, 2, kBruteForceSearchParams, filter);
            if (not result.has_value() or result.value()->GetDim() != 2 or
                result.value()->GetIds()[0] != 10 or result.value()->GetDistances()[0] != 0) {
                return false;
            }
        }
        return true;
    });
    std::packaged_task<bool()> tune([index, start] {
        while (not start->load()) {
            std::this_thread::yield();
        }
        for (uint64_t i = 0; i < 50; ++i) {
            const auto result =
                index->Tune(i % 2 == 0 ? R"({"index_param":{"base_quantization_type":"bf16"}})"
                                       : R"({"index_param":{"base_quantization_type":"fp32"}})");
            if (not result.has_value() or not result.value()) {
                return false;
            }
        }
        return true;
    });
    auto searched = search.get_future();
    auto tuned = tune.get_future();
    std::thread search_thread(std::move(search));
    std::thread tune_thread(std::move(tune));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (not filter->entered.load() and std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    // ValidRatio runs under the search's global read lock, before its codes read lock.
    // Let Tune reach the swap while that global lock is held. The old lock ordering
    // takes codes exclusively here, reproducing the cycle when the filter is released.
    start->store(true);
    auto graph = std::static_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    const auto swap_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (not vsag::HGraphCodeWriteTestAccess::CodesLocked(*graph) and
           std::chrono::steady_clock::now() < swap_deadline) {
        std::this_thread::yield();
    }
    filter->released.store(true);
    const bool search_ready = searched.wait_until(deadline) == std::future_status::ready;
    const bool tune_ready = tuned.wait_until(deadline) == std::future_status::ready;
    // Detached workers own all their state on failure; future destruction cannot hang the test.
    if (search_ready) {
        search_thread.join();
    } else {
        search_thread.detach();
    }
    if (tune_ready) {
        tune_thread.join();
    } else {
        tune_thread.detach();
    }
    REQUIRE(search_ready);
    REQUIRE(tune_ready);
    REQUIRE(searched.get());
    REQUIRE(tuned.get());
}

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

TEST_CASE("HGraph fused RaBitQ rejects v0.14 serialization",
          "[ut][hgraph][rabitq_split][fused][serialize]") {
    constexpr int64_t dim = 64;
    auto common_param = MakeCommonParam(dim);
    common_param.use_old_serial_format_ = true;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 2,
        "rabitq_bits_per_dim_precise": 6,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "nsw",
        "max_degree": 8,
        "ef_construction": 32,
        "use_reorder": true,
        "reorder_source": "base",
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    auto binary = index->Serialize();
    REQUIRE_FALSE(binary.has_value());
    REQUIRE(binary.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(binary.error().message.find("v0.14") != std::string::npos);
}

TEST_CASE("HGraph fused RaBitQ GetStats decodes vectors from node records",
          "[ut][hgraph][rabitq_split][fused][stats]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "rabitq_use_fht": true,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "nsw",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "use_reorder": true,
        "reorder_source": "base",
        "store_raw_vector": false,
        "use_mci": false,
        "support_duplicate": true,
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    std::vector<float> vectors(static_cast<uint64_t>(count) * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[static_cast<uint64_t>(i) * dim + d] =
                0.1F + static_cast<float>((i * 17 + d * 13) % 79) / 100.0F;
        }
    }
    std::copy_n(vectors.data(), dim, vectors.data() + static_cast<uint64_t>(count - 1) * dim);
    auto base = MakeFloatDataset(vectors, ids, dim, count);
    REQUIRE(index->Build(base).has_value());

    std::string stats;
    REQUIRE_NOTHROW(stats = index->GetStats());
    REQUIRE_FALSE(stats.empty());
    const auto parsed_stats = vsag::JsonType::Parse(stats);
    REQUIRE(parsed_stats["duplicate_ratio"].GetFloat() > 0.0F);

    const std::vector<int64_t> fetch_ids = {ids.front(), ids.back()};
    auto fetched = index->GetInnerIndex()->GetVectorByIds(
        fetch_ids.data(), static_cast<int64_t>(fetch_ids.size()), nullptr);
    REQUIRE(fetched != nullptr);
    REQUIRE(fetched->GetNumElements() == static_cast<int64_t>(fetch_ids.size()));
    REQUIRE(fetched->GetDim() == dim);
    const auto* fetched_vectors = fetched->GetFloat32Vectors();
    REQUIRE(fetched_vectors != nullptr);
    for (uint64_t i = 0; i < static_cast<uint64_t>(fetch_ids.size()) * dim; ++i) {
        REQUIRE(std::isfinite(fetched_vectors[i]));
    }
}

TEST_CASE("HGraph fused RaBitQ remains mutable after fast build and deserialize",
          "[ut][hgraph][rabitq_split][fused][lifecycle]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    constexpr int64_t extra_info_size = 4;
    auto common_param = MakeCommonParam(dim);
    common_param.extra_info_size_ = extra_info_size;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "rabitq",
        "precise_quantization_type": "rabitq",
        "base_io_type": "memory_io",
        "base_supplement_io_type": "memory_io",
        "rabitq_bits_per_dim_base": 2,
        "rabitq_bits_per_dim_precise": 6,
        "graph_io_type": "memory_io",
        "graph_storage_type": "flat",
        "graph_type": "odescent",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "use_reorder": true,
        "reorder_source": "base",
        "use_attribute_filter": true,
        "rabitq_fused_datacell": true
    })");
    auto index = MakeHGraphIndex(hgraph_json, common_param);

    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_ADD_AFTER_BUILD));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_MERGE_INDEX));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_CLONE));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_EXPORT_MODEL));
    REQUIRE_FALSE(index->CheckFeature(vsag::IndexFeature::SUPPORT_TUNE));
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT));
    REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT));

    std::vector<float> vectors(static_cast<uint64_t>(count) * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i + 100;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[static_cast<uint64_t>(i) * dim + d] =
                static_cast<float>((i * 17 + d * 13) % 101) / 101.0F;
        }
    }
    auto base = MakeFloatDataset(vectors, ids, dim, count);
    std::vector<char> base_extra_infos(count * extra_info_size, 'a');
    base->ExtraInfos(base_extra_infos.data())->ExtraInfoSize(extra_info_size);
    std::vector<vsag::AttributeValue<int64_t>> base_attributes(count);
    std::vector<vsag::AttributeSet> base_attribute_sets(count);
    for (int64_t i = 0; i < count; ++i) {
        base_attributes[static_cast<uint64_t>(i)].name_ = "tenant";
        base_attributes[static_cast<uint64_t>(i)].GetValue().push_back(1);
        base_attribute_sets[static_cast<uint64_t>(i)].attrs_.push_back(
            &base_attributes[static_cast<uint64_t>(i)]);
    }
    base->AttributeSets(base_attribute_sets.data());
    REQUIRE(index->Build(base).has_value());

    std::vector<float> add_vector(dim, 2.0F);
    std::vector<int64_t> add_id{1000};
    auto add = MakeFloatDataset(add_vector, add_id, dim, 1);
    std::vector<char> add_extra_info(extra_info_size, 'b');
    add->ExtraInfos(add_extra_info.data())->ExtraInfoSize(extra_info_size);
    vsag::AttributeValue<int64_t> add_attribute;
    add_attribute.name_ = "tenant";
    add_attribute.GetValue().push_back(1);
    vsag::AttributeSet add_attribute_set;
    add_attribute_set.attrs_.push_back(&add_attribute);
    add->AttributeSets(&add_attribute_set);
    auto add_result = index->Add(add);
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());
    REQUIRE(index->CheckIdExist(add_id.front()));

    std::vector<float> updated_vector(dim, 3.0F);
    std::vector<int64_t> updated_id{ids.front()};
    auto update = MakeFloatDataset(updated_vector, updated_id, dim, 1);
    auto update_result = index->UpdateVector(updated_id.front(), update, true);
    REQUIRE(update_result.has_value());
    REQUIRE(update_result.value());

    constexpr int64_t renamed_id = 2000;
    auto update_id_result = index->UpdateId(updated_id.front(), renamed_id);
    REQUIRE(update_id_result.has_value());
    REQUIRE(update_id_result.value());
    REQUIRE_FALSE(index->CheckIdExist(updated_id.front()));
    REQUIRE(index->CheckIdExist(renamed_id));

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->CheckIdExist(renamed_id));

    std::vector<float> restored_add_vector(dim, 4.0F);
    std::vector<int64_t> restored_add_id{3000};
    auto restored_add = MakeFloatDataset(restored_add_vector, restored_add_id, dim, 1);
    std::vector<char> restored_add_extra_info(extra_info_size, 'c');
    restored_add->ExtraInfos(restored_add_extra_info.data())->ExtraInfoSize(extra_info_size);
    vsag::AttributeValue<int64_t> restored_add_attribute;
    restored_add_attribute.name_ = "tenant";
    restored_add_attribute.GetValue().push_back(1);
    vsag::AttributeSet restored_add_attribute_set;
    restored_add_attribute_set.attrs_.push_back(&restored_add_attribute);
    restored_add->AttributeSets(&restored_add_attribute_set);
    REQUIRE(restored->Add(restored_add).has_value());
    REQUIRE(restored->CheckIdExist(restored_add_id.front()));

    vsag::AttributeValue<int64_t> updated_attribute;
    updated_attribute.name_ = "tenant";
    updated_attribute.GetValue().push_back(2);
    vsag::AttributeSet updated_attribute_set;
    updated_attribute_set.attrs_.push_back(&updated_attribute);
    REQUIRE(restored->UpdateAttribute(renamed_id, updated_attribute_set).has_value());

    std::vector<char> updated_extra_info(extra_info_size, 'd');
    auto extra_info_update = vsag::Dataset::Make();
    extra_info_update->NumElements(1)
        ->Ids(&renamed_id)
        ->ExtraInfos(updated_extra_info.data())
        ->ExtraInfoSize(extra_info_size)
        ->Owner(false);
    auto extra_info_result = restored->UpdateExtraInfo(extra_info_update);
    REQUIRE(extra_info_result.has_value());
    REQUIRE(extra_info_result.value());
    std::vector<char> fetched_extra_info(extra_info_size);
    REQUIRE(restored->GetExtraInfoByIds(&renamed_id, 1, fetched_extra_info.data()).has_value());
    REQUIRE(fetched_extra_info == updated_extra_info);

    const std::vector<int64_t> remove_ids{add_id.front()};
    auto remove_result = restored->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
    REQUIRE(remove_result.has_value());
    REQUIRE(remove_result.value() == 1);
    REQUIRE_FALSE(restored->CheckIdExist(add_id.front()));

    auto require_unsupported = [](const auto& result) {
        REQUIRE_FALSE(result.has_value());
        CAPTURE(result.error().message);
        REQUIRE(result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    };
    auto train_result = index->Train(base);
    require_unsupported(train_result);
    REQUIRE(train_result.error().message == "fused RaBitQ HGraph does not support Train");
    require_unsupported(index->Remove(remove_ids, vsag::RemoveMode::FORCE_REMOVE));
    require_unsupported(index->Tune("{}"));
    require_unsupported(index->Merge({}));
    require_unsupported(index->Clone());
    require_unsupported(index->ExportModel());

    std::stringstream cache;
    require_unsupported(index->ExportCache(cache));
    require_unsupported(index->ImportCache(cache));

    REQUIRE(restored->SetImmutable().has_value());
    require_unsupported(restored->Add(restored_add));
    require_unsupported(restored->UpdateVector(renamed_id, update, true));
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
    hgraph_json["graph_type"].SetString(GENERATE("nsw", "pipnn"));
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

TEST_CASE("HGraph SAQ index survives serialization round trip", "[ut][hgraph][SAQ]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    constexpr int64_t k = 10;
    auto common_param = MakeCommonParam(dim);
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_COSINE;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "saq",
        "saq_avg_bits": 2,
        "saq_segment_count": 1,
        "saq_random_rotation": true,
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1,
        "use_reorder": false
    })");

    auto vectors = fixtures::generate_vectors(count, dim, false, 229);
    std::fill(vectors.begin(), vectors.begin() + dim, 0.0F);
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 1000);
    auto base = MakeFloatDataset(vectors, ids, dim, count);
    std::vector<float> query_vector(vectors.begin() + dim, vectors.begin() + 2 * dim);
    auto query = MakeFloatQuery(query_vector, dim);
    auto index = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(index->Build(base).has_value());

    const std::string search_parameters = R"({"hgraph": {"ef_search": 32}})";
    auto before = index->KnnSearch(query, k, search_parameters);
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetDim() == k);

    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(hgraph_json, common_param);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    REQUIRE(restored->GetNumElements() == count);
    auto after = restored->KnnSearch(query, k, search_parameters);
    REQUIRE(after.has_value());
    REQUIRE(after.value()->GetDim() == before.value()->GetDim());

    for (int64_t i = 0; i < k; ++i) {
        REQUIRE(after.value()->GetIds()[i] == before.value()->GetIds()[i]);
        REQUIRE(std::isfinite(before.value()->GetDistances()[i]));
        REQUIRE(std::isfinite(after.value()->GetDistances()[i]));
        REQUIRE(std::abs(after.value()->GetDistances()[i] - before.value()->GetDistances()[i]) <=
                1e-6F);
    }
}

TEST_CASE("HGraph deduplicate_storage concurrent Add keeps visible duplicates searchable",
          "[ut][hgraph][duplicate][concurrent][add]") {
    constexpr int64_t dim = 2;
    auto common_param = MakeCommonParam(dim);
    auto hgraph_json = MakeFp32HGraphJson();
    hgraph_json["graph_type"].SetString(GENERATE("nsw", "pipnn"));
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

TEST_CASE("HGraph optimized build accepts MRLE RaBitQ split with raw-vector storage",
          "[ut][hgraph][optimized_build][MRLE]") {
    constexpr int64_t dim = 64;
    auto hgraph_json = vsag::JsonType::Parse(R"({
        "base_quantization_type": "tq",
        "tq_chain": "mrle, rabitq",
        "mrle_dim": 32,
        "precise_quantization_type": "rabitq",
        "rabitq_bits_per_dim_base": 3,
        "rabitq_bits_per_dim_precise": 5,
        "use_reorder": true,
        "build_by_base": false,
        "max_degree": 8,
        "ef_construction": 32
    })");
    auto index = MakeHGraphIndex(hgraph_json, MakeCommonParam(dim, 2));
    auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(hgraph != nullptr);

    vsag::HGraphOptimizedBuildSession session(*hgraph);
    REQUIRE(session.Active());
}

TEST_CASE("HGraph adaptive pruning build reload and add", "[ut][hgraph][adaptive_pruning]") {
    const int64_t threads = GENERATE(1, 4);
    auto common = MakeCommonParam(2, threads);
    auto json = MakeFp32HGraphJson(false, threads);
    json["alpha"].SetFloat(1.06F);
    json["adaptive_pruning"].SetBool(true);
    json["adaptive_pruning_adjust_step"].SetFloat(0.06F);
    const bool reverse = GENERATE(false, true);
    json["adaptive_pruning_apply_to_reverse"].SetBool(reverse);
    auto index = MakeHGraphIndex(json, common);
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    for (int64_t i = 0; i < 128; ++i) {
        vectors.push_back(static_cast<float>(i % 16));
        vectors.push_back(static_cast<float>(i / 16));
        ids.push_back(i);
    }
    auto build = index->Build(MakeFloatDataset(vectors, ids, 2, ids.size()));
    REQUIRE(build.has_value());
    REQUIRE(build.value().empty());
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(json, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    std::vector<float> query = {3.0F, 3.0F};
    auto found =
        restored->KnnSearch(MakeFloatQuery(query, 2), 1, R"({"hgraph":{"ef_search":128}})");
    REQUIRE(found.has_value());
    REQUIRE(found.value()->GetDim() == 1);
    CHECK(found.value()->GetIds()[0] == 51);
    std::vector<float> added = {3.25F, 3.25F};
    std::vector<int64_t> added_ids = {128};
    auto add = restored->Add(MakeFloatDataset(added, added_ids, 2, 1));
    REQUIRE(add.has_value());
    REQUIRE(add.value().empty());
    found = restored->KnnSearch(MakeFloatQuery(added, 2), 1, R"({"hgraph":{"ef_search":128}})");
    REQUIRE(found.has_value());
    REQUIRE(found.value()->GetDim() == 1);
    CHECK(found.value()->GetIds()[0] == 128);
    auto incompatible_json = MakeFp32HGraphJson(false, threads);
    auto incompatible = MakeHGraphIndex(incompatible_json, common);
    CHECK_FALSE(incompatible->Deserialize(binary.value()).has_value());
    auto changed = vsag::JsonType::Parse(json.Dump());
    changed["adaptive_pruning_adjust_step"].SetFloat(0.03F);
    CHECK_FALSE(MakeHGraphIndex(changed, common)->Deserialize(binary.value()).has_value());
    auto wrong_scope = vsag::JsonType::Parse(json.Dump());
    wrong_scope["adaptive_pruning_apply_to_reverse"].SetBool(not reverse);
    CHECK_FALSE(MakeHGraphIndex(wrong_scope, common)->Deserialize(binary.value()).has_value());
    json["alpha"].SetFloat(1.12F);
    CHECK_FALSE(MakeHGraphIndex(json, common)->Deserialize(binary.value()).has_value());
}

TEST_CASE("HGraph adaptive pruning unsupported contexts", "[ut][hgraph][adaptive_pruning]") {
    auto common = MakeCommonParam(2);
    auto json = MakeFp32HGraphJson(false);
    json["adaptive_pruning"].SetBool(true);
    SECTION("ODescent") {
        json["graph_type"].SetString("odescent");
        CHECK_THROWS(MakeHGraphIndex(json, common));
    }
    SECTION("Inner product") {
        common.metric_ = vsag::MetricType::METRIC_TYPE_IP;
        CHECK_THROWS(MakeHGraphIndex(json, common));
    }
    SECTION("Unsupported scope") {
        json["adaptive_pruning_apply_to_upper"].SetBool(true);
        CHECK_THROWS(MakeHGraphIndex(json, common));
    }
    SECTION("Invalid threshold") {
        json["adaptive_pruning_adjust_step"].SetFloat(1.0F);
        CHECK_THROWS(MakeHGraphIndex(json, common));
    }
}

TEST_CASE("HGraph adaptive pruning rejects imported build caches",
          "[ut][hgraph][adaptive_pruning][cache]") {
    auto common = MakeCommonParam(2);
    std::vector<float> vectors = {0, 0, 1, 0, 0, 1, 1, 1};
    std::vector<int64_t> ids = {10, 20, 30, 40};
    std::vector<std::string> source_ids = {"a", "b", "c", "d"};
    auto data = MakeFloatDatasetWithSourceIds(vectors, ids, source_ids, 2, 4);
    auto json = MakeFp32HGraphJson(false);
    auto source = MakeHGraphIndex(json, common);
    REQUIRE(source->Build(data).has_value());
    std::stringstream cache;
    REQUIRE(source->ExportCache(cache).has_value());
    cache.seekg(0);
    json["adaptive_pruning"].SetBool(true);
    json["adaptive_pruning_apply_to_reverse"].SetBool(GENERATE(false, true));
    auto target = MakeHGraphIndex(json, common);
    REQUIRE(target->ImportCache(cache).has_value());
    const auto result = target->Build(data);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    CHECK(result.error().message.find("adaptive_pruning") != std::string::npos);
}

TEST_CASE("HGraph adaptive pruning RaBitQ split and fused lifecycle",
          "[ut][hgraph][adaptive_pruning][fused]") {
    const bool fused = GENERATE(false, true);
    const int64_t threads = GENERATE(1, 4);
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    auto common = MakeCommonParam(dim, threads);
    auto json = vsag::JsonType::Parse(R"({
        "base_quantization_type":"rabitq", "precise_quantization_type":"rabitq",
        "base_io_type":"memory_io", "base_supplement_io_type":"memory_io",
        "rabitq_bits_per_dim_base":2, "rabitq_bits_per_dim_precise":6,
        "graph_io_type":"memory_io", "graph_storage_type":"flat", "graph_type":"nsw",
        "max_degree":8, "ef_construction":32, "use_reorder":true, "reorder_source":"base",
        "alpha":1.06, "adaptive_pruning":true
    })");
    json["rabitq_fused_datacell"].SetBool(fused);
    // Split incremental Add already requires raw vectors for its temporary SQ8 build store.
    json["store_raw_vector"].SetBool(not fused);
    json["build_thread_count"].SetInt(threads);
    json["adaptive_pruning_apply_to_reverse"].SetBool(GENERATE(false, true));
    auto index = MakeHGraphIndex(json, common);
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>((i * 17 + d * 13) % 101) / 101.0F;
        }
    }
    auto built = index->Build(MakeFloatDataset(vectors, ids, dim, count));
    REQUIRE(built.has_value());
    REQUIRE(built.value().empty());
    auto binary = index->Serialize();
    REQUIRE(binary.has_value());
    auto restored = MakeHGraphIndex(json, common);
    REQUIRE(restored->Deserialize(binary.value()).has_value());
    std::vector<float> added(dim, 2.0F);
    std::vector<int64_t> added_ids = {count};
    auto result = restored->Add(MakeFloatDataset(added, added_ids, dim, 1));
    REQUIRE(result.has_value());
    REQUIRE(result.value().empty());
    auto found =
        restored->KnnSearch(MakeFloatQuery(added, dim), 1, R"({"hgraph":{"ef_search":64}})");
    REQUIRE(found.has_value());
    REQUIRE(found.value()->GetDim() == 1);
    CHECK(found.value()->GetIds()[0] == count);
}

TEST_CASE("HGraph fused Add preencoding preserves row mapping and rejects failed batches",
          "[ut][hgraph][fused][preencode]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    const auto threads = GENERATE(0U, 4U);
    const auto bits = GENERATE(1, 3);
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    auto common = MakeCommonParam(dim, threads);
    common.metric_ = metric;
    auto param = vsag::JsonType::Parse(R"({
        "base_quantization_type":"rabitq", "precise_quantization_type":"rabitq",
        "base_io_type":"memory_io", "base_supplement_io_type":"memory_io",
        "graph_io_type":"memory_io", "graph_storage_type":"flat", "graph_type":"nsw",
        "rabitq_use_fht":true, "rabitq_fused_datacell":true,
        "use_reorder":true, "reorder_source":"base", "store_raw_vector":false,
        "use_mci":false, "max_degree":8, "ef_construction":32
    })");
    param["build_thread_count"].SetInt(threads == 0 ? 1 : threads);
    param["rabitq_bits_per_dim_base"].SetInt(bits);
    param["rabitq_bits_per_dim_precise"].SetInt(8 - bits);
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = 0.01F * static_cast<float>((i * 17 + d * 13) % 97 - 48);
        }
    }
    auto dataset = MakeFloatDataset(vectors, ids, dim, count);
    auto index = MakeHGraphIndex(param, common);
    auto seed = MakeFloatDataset(vectors, ids, dim, 16);
    REQUIRE(index->Add(seed).has_value());
    auto original = index->CalcDistanceById(vectors.data(), ids.front());
    REQUIRE(original.has_value());

    // A finite input can still overflow encoding. Other worker ranges contain valid rows.
    auto invalid = vectors;
    std::fill_n(invalid.data() + 53 * dim, dim, std::numeric_limits<float>::max());
    auto failed = index->Add(MakeFloatDataset(invalid, ids, dim, count));
    REQUIRE_FALSE(failed.has_value());
    REQUIRE(failed.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    REQUIRE(index->GetNumElements() == 16);
    for (int64_t i = 16; i < count; ++i) {
        REQUIRE_FALSE(index->CheckIdExist(ids[i]));
    }
    REQUIRE(index->CalcDistanceById(vectors.data(), ids.front()).value() == original.value());

    // Retrying the same labels must work; skipped input row 0 must not shift cached codes.
    auto added = index->Add(dataset);
    REQUIRE(added.has_value());
    REQUIRE(added.value() == std::vector<int64_t>(ids.begin(), ids.begin() + 16));
    REQUIRE(index->GetNumElements() == count);
    auto serial_common = MakeCommonParam(dim);
    serial_common.metric_ = metric;
    param["build_thread_count"].SetInt(1);
    auto serial = MakeHGraphIndex(param, serial_common);
    REQUIRE(serial->Add(seed).has_value());
    for (int64_t i = 16; i < count; ++i) {
        auto one = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(dim)
                       ->Ids(ids.data() + i)
                       ->Float32Vectors(vectors.data() + i * dim)
                       ->Owner(false);
        REQUIRE(serial->Add(one).has_value());
    }
    auto decoded = index->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    auto expected = serial->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    REQUIRE(decoded->GetNumElements() == count);
    for (int64_t i = 0; i < count * dim; ++i) {
        REQUIRE(decoded->GetFloat32Vectors()[i] == expected->GetFloat32Vectors()[i]);
    }
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = MakeHGraphIndex(param, serial_common);
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    REQUIRE(restored->GetNumElements() == count);
    auto restored_vectors = restored->GetInnerIndex()->GetVectorByIds(ids.data(), count, nullptr);
    for (int64_t i = 0; i < count * dim; ++i) {
        REQUIRE(restored_vectors->GetFloat32Vectors()[i] == expected->GetFloat32Vectors()[i]);
    }
}

TEST_CASE("HGraph Merge rejects unfinished inputs before copying any unit",
          "[ut][hgraph][hgraph_review_lifecycle]") {
    const bool fail_add = GENERATE(false, true);
    const bool incomplete_destination = GENERATE(false, true);
    auto pool = std::make_shared<ArmableRejectingThreadPool>();
    auto common = MakeCommonParam(4);
    common.thread_pool_ = std::make_shared<vsag::SafeThreadPool>(pool);
    auto param = MakeFp32HGraphJson(false);
    auto source = MakeHGraphIndex(param, common);
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> labels{10, 20};
    REQUIRE(source->Build(MakeFloatDataset(vectors, labels, 4, 2)).has_value());
    std::vector<float> missing{1, 0, 0, 0};
    std::vector<int64_t> missing_label{30};
    if (fail_add) {
        pool->SetRejectSubmissions(true);
        REQUIRE_FALSE(source->Add(MakeFloatDataset(missing, missing_label, 4, 1)).has_value());
        pool->SetRejectSubmissions(false);
    }
    std::vector<int64_t> later_label{40};
    REQUIRE(source->Add(MakeFloatDataset(missing, later_label, 4, 1)).has_value());
    auto destination = MakeHGraphIndex(param, MakeCommonParam(4));
    auto complete = MakeHGraphIndex(param, MakeCommonParam(4));
    std::vector<int64_t> complete_labels{50, 60};
    REQUIRE(complete->Build(MakeFloatDataset(vectors, complete_labels, 4, 2)).has_value());
    vsag::IdMapFunction identity = [](int64_t id) { return std::make_tuple(true, id); };
    if (incomplete_destination) {
        std::swap(source, destination);
    }
    const auto before = destination->GetNumElements();
    auto result = incomplete_destination
                      ? destination->Merge({{complete, identity}})
                      : destination->Merge({{complete, identity}, {source, identity}});
    if (fail_add) {
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(destination->GetNumElements() == before);
        REQUIRE_FALSE(destination->CheckIdExist(50));
        auto damaged = incomplete_destination ? destination : source;
        auto tuned = damaged->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})", true);
        REQUIRE(tuned.has_value());
        REQUIRE_FALSE(tuned.value());
    } else {
        REQUIRE(result.has_value());
        REQUIRE(destination->GetNumElements() == 5);
        auto tuned =
            destination->Tune(R"({"index_param":{"base_quantization_type":"bf16"}})", true);
        REQUIRE(tuned.has_value());
        REQUIRE(tuned.value());
    }
}

TEST_CASE("HGraph restore validates reservation metadata and accepts legacy counts",
          "[ut][hgraph][hgraph_review_lifecycle]") {
    const auto mode =
        GENERATE(std::string("ordinary"), std::string("streaming"), std::string("parallel"));
    const auto mutation =
        GENERATE(std::string("capacity"), std::string("overflow"), std::string("legacy"));
    CAPTURE(mode, mutation);
    auto param = MakeFp32HGraphJson(false);
    auto index = MakeHGraphIndex(param, MakeCommonParam(4));
    std::vector<float> vectors{0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int64_t> labels{10, 20};
    REQUIRE(index->Build(MakeFloatDataset(vectors, labels, 4, 2)).has_value());
    class Snapshot : public vsag::SerializeWriter, public vsag::DeserializeReader {
    public:
        void
        Write(const char* data, uint64_t size) override {
            bytes.append(data, size);
        }
        [[nodiscard]] uint64_t
        Size() const override {
            return bytes.size();
        }
        void
        Read(uint64_t offset, uint64_t size, void* dest) override {
            if (size > bytes.size() or offset > bytes.size() - size) {
                throw std::out_of_range("snapshot read");
            }
            std::memcpy(dest, bytes.data() + offset, size);
        }
        std::string bytes;
    } snapshot;
    auto mutate = [&](const vsag::MetadataPtr& metadata) {
        auto info = metadata->Get(vsag::BASIC_INFO);
        if (mutation == "legacy") {
            info.GetInnerJson()->erase("total_count");
        } else {
            const uint64_t count =
                mutation == "capacity"
                    ? info["max_capacity"].GetUint64() + 1
                    : static_cast<uint64_t>(std::numeric_limits<vsag::InnerIdType>::max()) + 1;
            info["total_count"].SetUint64(count);
        }
        metadata->Set(vsag::BASIC_INFO, info);
    };
    if (mode == "streaming") {
        std::stringstream stream;
        REQUIRE(index->SerializeStreaming(stream).has_value());
        snapshot.bytes = stream.str();
        vsag::IOStreamReader reader(stream);
        auto metadata = vsag::StreamHeader::Read(reader);
        mutate(metadata);
        std::ostringstream header;
        vsag::IOStreamWriter writer(header);
        vsag::StreamHeader::Write(writer, metadata);
        snapshot.bytes =
            header.str() + snapshot.bytes.substr(vsag::test::StreamingBodyOffset(snapshot.bytes));
    } else {
        if (mode == "parallel") {
            REQUIRE(index->Serialize(snapshot, 4096).has_value());
        } else {
            std::stringstream stream;
            REQUIRE(index->Serialize(stream).has_value());
            snapshot.bytes = stream.str();
        }
        vsag::ReadFuncStreamReader reader(
            [&](uint64_t offset, uint64_t size, void* dest) { snapshot.Read(offset, size, dest); },
            0,
            snapshot.bytes.size());
        auto footer = vsag::Footer::Parse(reader);
        REQUIRE(footer != nullptr);
        mutate(footer->GetMetadata());
        std::ostringstream output;
        vsag::IOStreamWriter writer(output);
        footer->Write(writer);
        snapshot.bytes =
            snapshot.bytes.substr(0, snapshot.bytes.size() - footer->Length()) + output.str();
    }
    auto restored = MakeHGraphIndex(param, MakeCommonParam(4));
    std::stringstream input(snapshot.bytes);
    auto result = mode == "streaming"  ? restored->DeserializeStreaming(input)
                  : mode == "parallel" ? restored->ParallelDeserialize(snapshot)
                                       : restored->Deserialize(input);
    if (mutation == "legacy") {
        REQUIRE(result.has_value());
        REQUIRE(restored->GetNumElements() == 2);
        std::vector<int64_t> added_label{30};
        REQUIRE(restored->Add(MakeFloatDataset(vectors, added_label, 4, 1)).has_value());
        auto graph = std::static_pointer_cast<vsag::HGraph>(restored->GetInnerIndex());
        REQUIRE(vsag::HGraphCodeWriteTestAccess::InnerId(*graph, 30) == 2);
    } else {
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_BINARY);
    }
}

TEST_CASE("HGraph optimized fused build omits Tune pending metadata",
          "[ut][hgraph][hgraph_review_lifecycle]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    auto param = vsag::JsonType::Parse(R"({
        "base_quantization_type":"rabitq", "precise_quantization_type":"rabitq",
        "base_io_type":"memory_io", "base_supplement_io_type":"memory_io",
        "graph_io_type":"memory_io", "graph_storage_type":"flat", "graph_type":"nsw",
        "rabitq_use_fht":true, "rabitq_fused_datacell":true,
        "rabitq_bits_per_dim_base":2, "rabitq_bits_per_dim_precise":6,
        "use_reorder":true, "reorder_source":"base", "store_raw_vector":false,
        "max_degree":8, "ef_construction":32, "build_thread_count":1
    })");
    auto index = MakeHGraphIndex(param, MakeCommonParam(dim));
    std::vector<float> vectors(count * dim);
    std::vector<int64_t> labels(count);
    for (int64_t i = 0; i < count; ++i) {
        labels[i] = i;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>((i * 17 + d * 13) % 97) / 97.0F;
        }
    }
    REQUIRE(index->Build(MakeFloatDataset(vectors, labels, dim, count)).has_value());
    auto graph = std::static_pointer_cast<vsag::HGraph>(index->GetInnerIndex());
    REQUIRE(vsag::HGraphCodeWriteTestAccess::PendingCount(*graph) == 0);
    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    vsag::IOStreamReader reader(stream);
    auto metadata = vsag::StreamHeader::Read(reader);
    REQUIRE_FALSE(metadata->Get(vsag::BASIC_INFO).Contains("pending_code_writes"));
    stream.seekg(0);
    auto restored = MakeHGraphIndex(param, MakeCommonParam(dim));
    REQUIRE(restored->DeserializeStreaming(stream).has_value());
    REQUIRE(restored->GetNumElements() == count);
    REQUIRE(
        restored->KnnSearch(MakeFloatQuery(vectors, dim), 1, kBruteForceSearchParams).has_value());
}
