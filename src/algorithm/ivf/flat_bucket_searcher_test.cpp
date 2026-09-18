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

#include "flat_bucket_searcher.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/reasoning/search_reasoning.h"
#include "rabitq_split_bucket_searcher.h"
#include "unittest.h"
#include "vsag/filter.h"

namespace vsag {
namespace {

bool
IsNaNBits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) == 0x7F800000U and (bits & 0x007FFFFFU) != 0U;
}

class EvenIdFilter final : public Filter {
public:
    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return id % 2 == 0;
    }
};

class StaticDistanceBucket final : public BucketInterface {
public:
    static constexpr uint64_t kSourceVersion = 37;

    explicit StaticDistanceBucket(std::vector<float> distances,
                                  std::vector<float> filter_inner_products = {})
        : distances_(std::move(distances)),
          filter_inner_products_(std::move(filter_inner_products)),
          ids_(distances_.size()) {
        for (uint64_t i = 0; i < ids_.size(); ++i) {
            ids_[i] = static_cast<InnerIdType>(i);
        }
    }

    void
    AppendOnNextCandidateScan() {
        append_on_next_candidate_scan_ = true;
    }

    void
    ShrinkOnNextCandidateScan(InnerIdType size) {
        candidate_scan_size_override_ = size;
    }

    void
    AppendOnNextNormalScan() {
        append_on_next_normal_scan_ = true;
    }

    void
    ShrinkOnNextNormalScan(InnerIdType size) {
        normal_scan_size_override_ = size;
    }

    void
    ScanBucketById(float* result_dists,
                   const ComputerInterfacePtr& computer,
                   const BucketIdType& bucket_id,
                   QueryContext* ctx = nullptr,
                   InnerIdType* scanned_inner_ids = nullptr,
                   InnerIdType max_scan_size = std::numeric_limits<InnerIdType>::max(),
                   InnerIdType* scanned_size = nullptr) override {
        (void)computer;
        (void)bucket_id;
        (void)ctx;
        if (normal_scan_size_override_ != std::numeric_limits<InnerIdType>::max()) {
            distances_.resize(normal_scan_size_override_);
            filter_inner_products_.resize(normal_scan_size_override_);
            ids_.resize(normal_scan_size_override_);
            normal_scan_size_override_ = std::numeric_limits<InnerIdType>::max();
        }
        if (append_on_next_normal_scan_) {
            const auto new_id = static_cast<InnerIdType>(ids_.size());
            distances_.push_back(-1000.0F);
            if (not filter_inner_products_.empty()) {
                filter_inner_products_.push_back(2000.0F + static_cast<float>(new_id));
            }
            ids_.push_back(new_id);
            append_on_next_normal_scan_ = false;
        }
        const auto scan_size = std::min<uint64_t>(distances_.size(), max_scan_size);
        if (scanned_size != nullptr) {
            *scanned_size = static_cast<InnerIdType>(scan_size);
        }
        std::copy_n(distances_.begin(), scan_size, result_dists);
        if (scanned_inner_ids != nullptr) {
            std::copy_n(ids_.begin(), scan_size, scanned_inner_ids);
        }
    }

    uint64_t
    ScanBucketWithFilterInnerProduct(
        float* result_dists,
        float* filter_inner_products,
        const ComputerInterfacePtr& computer,
        const BucketIdType& bucket_id,
        QueryContext* ctx = nullptr,
        InnerIdType* scanned_inner_ids = nullptr,
        InnerIdType max_scan_size = std::numeric_limits<InnerIdType>::max(),
        InnerIdType* scanned_size = nullptr) override {
        (void)computer;
        (void)bucket_id;
        (void)ctx;
        if (candidate_scan_size_override_ != std::numeric_limits<InnerIdType>::max()) {
            distances_.resize(candidate_scan_size_override_);
            filter_inner_products_.resize(candidate_scan_size_override_);
            ids_.resize(candidate_scan_size_override_);
            candidate_scan_size_override_ = std::numeric_limits<InnerIdType>::max();
        }
        if (append_on_next_candidate_scan_) {
            const auto new_id = static_cast<InnerIdType>(ids_.size());
            distances_.push_back(-1000.0F);
            filter_inner_products_.push_back(2000.0F + static_cast<float>(new_id));
            ids_.push_back(new_id);
            append_on_next_candidate_scan_ = false;
        }
        const auto scan_size = std::min<uint64_t>(distances_.size(), max_scan_size);
        if (scanned_size != nullptr) {
            *scanned_size = static_cast<InnerIdType>(scan_size);
        }
        std::copy_n(distances_.begin(), scan_size, result_dists);
        std::copy_n(filter_inner_products_.begin(), scan_size, filter_inner_products);
        if (scanned_inner_ids != nullptr) {
            std::copy_n(ids_.begin(), scan_size, scanned_inner_ids);
        }
        return kSourceVersion;
    }

    [[nodiscard]] bool
    SupportSplitCodeStorage() const override {
        return filter_inner_products_.size() == distances_.size();
    }

    float
    QueryOneById(const ComputerInterfacePtr& computer,
                 const BucketIdType& bucket_id,
                 const InnerIdType& offset_id) override {
        (void)computer;
        (void)bucket_id;
        return distances_.at(static_cast<uint64_t>(offset_id));
    }

    float
    ComputePairVectors(BucketIdType bucket_id, InnerIdType id1, InnerIdType id2) override {
        (void)bucket_id;
        (void)id1;
        (void)id2;
        return 0.0F;
    }

    ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        (void)query;
        return nullptr;
    }

    void
    Train(const void* data, uint64_t count) override {
        (void)data;
        (void)count;
    }

    InnerIdType
    InsertVector(const void* vector, BucketIdType bucket_id, InnerIdType inner_id) override {
        (void)vector;
        (void)bucket_id;
        (void)inner_id;
        return 0;
    }

    void
    BatchInsertVector(const void* vectors,
                      const BucketIdType* bucket_ids,
                      const InnerIdType* inner_ids,
                      InnerIdType count,
                      InnerIdType* out_offsets) override {
        (void)vectors;
        (void)bucket_ids;
        (void)inner_ids;
        (void)count;
        (void)out_offsets;
    }

    InnerIdType*
    GetInnerIds(BucketIdType bucket_id) override {
        (void)bucket_id;
        return ids_.data();
    }

    void
    Prefetch(BucketIdType bucket_id, InnerIdType offset_id) override {
        (void)bucket_id;
        (void)offset_id;
    }

    void
    GetCodesById(BucketIdType bucket_id, InnerIdType offset_id, uint8_t* data) const override {
        (void)bucket_id;
        (void)offset_id;
        (void)data;
    }

    [[nodiscard]] std::string
    GetQuantizerName() override {
        return "static-distance-test";
    }

    [[nodiscard]] MetricType
    GetMetricType() override {
        return MetricType::METRIC_TYPE_L2SQR;
    }

    [[nodiscard]] InnerIdType
    GetBucketSize(BucketIdType bucket_id) override {
        (void)bucket_id;
        return static_cast<InnerIdType>(distances_.size());
    }

    void
    ExportModel(const BucketInterfacePtr& other) const override {
        (void)other;
    }

    void
    MergeOther(const BucketInterfacePtr& other, InnerIdType bias) override {
        (void)other;
        (void)bias;
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const override {
        return distances_.size() * sizeof(float) + ids_.size() * sizeof(InnerIdType);
    }

private:
    std::vector<float> distances_;
    std::vector<float> filter_inner_products_;
    std::vector<InnerIdType> ids_;
    bool append_on_next_candidate_scan_{false};
    bool append_on_next_normal_scan_{false};
    InnerIdType candidate_scan_size_override_{std::numeric_limits<InnerIdType>::max()};
    InnerIdType normal_scan_size_override_{std::numeric_limits<InnerIdType>::max()};
};

using DistanceRecord = DistanceHeap::DistanceRecord;

std::vector<DistanceRecord>
RunFlatSearch(const std::vector<float>& distances,
              InnerSearchParam param,
              int64_t topk,
              std::vector<float>* auxiliary_values = nullptr,
              const std::vector<float>& filter_inner_products = {},
              ReasoningContext* reasoning_ctx = nullptr,
              std::vector<DistanceHeap::AuxiliaryDistanceRecord>* auxiliary_records = nullptr,
              bool append_during_scan = false,
              InnerIdType scan_size_override = std::numeric_limits<InnerIdType>::max()) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto bucket = std::make_shared<StaticDistanceBucket>(distances, filter_inner_products);
    const bool collect_candidate_filter_inner_products =
        param.search_mode == KNN_SEARCH and param.enable_reorder and
        bucket->SupportSplitCodeStorage() and
        (auxiliary_values != nullptr or auxiliary_records != nullptr);
    if (append_during_scan) {
        if (collect_candidate_filter_inner_products) {
            bucket->AppendOnNextCandidateScan();
        } else {
            bucket->AppendOnNextNormalScan();
        }
    }
    if (scan_size_override != std::numeric_limits<InnerIdType>::max()) {
        if (collect_candidate_filter_inner_products) {
            bucket->ShrinkOnNextCandidateScan(scan_size_override);
        } else {
            bucket->ShrinkOnNextNormalScan(scan_size_override);
        }
    }
    DistHeapPtr heap = nullptr;
    if (auxiliary_values == nullptr and auxiliary_records == nullptr) {
        heap = DistanceHeap::MakeInstanceBySize<true, false>(allocator.get(), topk);
    } else {
        heap = DistanceHeap::MakeInstanceBySizeWithAuxiliary<true, false>(allocator.get(), topk);
    }
    Vector<float> scratch(allocator.get());
    Vector<InnerIdType> scanned_inner_ids(allocator.get());
    const bool use_split_searcher = auxiliary_values != nullptr or auxiliary_records != nullptr or
                                    not filter_inner_products.empty();
    IVFBucketSearcherPtr searcher;
    if (use_split_searcher) {
        searcher = std::make_shared<RaBitQSplitBucketSearcher>();
    } else {
        searcher = std::make_shared<FlatBucketSearcher>();
    }
    searcher->Search(
        0, bucket, nullptr, param, -1, topk, 1, heap, scratch, scanned_inner_ids, reasoning_ctx);

    std::vector<DistanceRecord> result;
    if (not heap->Empty()) {
        result.assign(heap->GetData(), heap->GetData() + heap->Size());
    }
    if (auxiliary_values != nullptr or auxiliary_records != nullptr) {
        const auto* auxiliary_data = heap->GetDataWithAuxiliary();
        if (auxiliary_values != nullptr) {
            auxiliary_values->clear();
        }
        if (auxiliary_records != nullptr) {
            auxiliary_records->clear();
        }
        for (uint64_t i = 0; i < heap->Size(); ++i) {
            if (auxiliary_values != nullptr) {
                auxiliary_values->emplace_back(auxiliary_data[i].auxiliary);
            }
            if (auxiliary_records != nullptr) {
                auxiliary_records->emplace_back(auxiliary_data[i]);
            }
        }
    }
    return result;
}

std::vector<InnerIdType>
SortedIds(const std::vector<DistanceRecord>& records) {
    std::vector<InnerIdType> result;
    result.reserve(records.size());
    for (const auto& record : records) {
        result.emplace_back(record.second);
    }
    std::sort(result.begin(), result.end());
    return result;
}

void
RequireAuxiliaryMatchesHeap(const std::vector<DistanceRecord>& records,
                            const std::vector<float>& auxiliary_values,
                            const std::vector<float>& filter_inner_products) {
    REQUIRE(auxiliary_values.size() == records.size());
    for (uint64_t i = 0; i < records.size(); ++i) {
        REQUIRE(auxiliary_values[i] ==
                filter_inner_products[static_cast<uint64_t>(records[i].second)]);
    }
}

}  // namespace

TEST_CASE("IVF FlatBucketSearcher normal scans remain bounded",
          "[ut][ivf][flat_bucket_searcher][bounded]") {
    std::vector<float> distances(37);
    for (uint64_t i = 0; i < distances.size(); ++i) {
        distances[i] = static_cast<float>(distances.size() - i);
    }

    SECTION("KNN scan bounds a concurrent append and snapshots ids") {
        InnerSearchParam param;
        param.search_mode = KNN_SEARCH;
        param.enable_reorder = true;
        const auto records =
            RunFlatSearch(distances, param, 64, nullptr, {}, nullptr, nullptr, true);

        REQUIRE(records.size() == distances.size());
        REQUIRE(std::none_of(records.begin(), records.end(), [&](const auto& record) {
            return record.second == distances.size();
        }));
    }

    SECTION("disabled-reorder KNN scan remains bounded across reallocation") {
        InnerSearchParam param;
        param.search_mode = KNN_SEARCH;
        param.enable_reorder = false;
        param.distance_threshold = 100.0F;
        const auto records =
            RunFlatSearch(distances, param, 64, nullptr, {}, nullptr, nullptr, true);

        REQUIRE(records.size() == distances.size());
        REQUIRE(std::none_of(records.begin(), records.end(), [&](const auto& record) {
            return record.second == distances.size();
        }));
    }

    SECTION("range scan bounds a concurrent append and snapshots ids") {
        InnerSearchParam param;
        param.search_mode = RANGE_SEARCH;
        param.radius = 100.0F;
        const auto records =
            RunFlatSearch(distances, param, 64, nullptr, {}, nullptr, nullptr, true);

        REQUIRE(records.size() == distances.size());
        REQUIRE(std::none_of(records.begin(), records.end(), [&](const auto& record) {
            return record.second == distances.size();
        }));
    }

    SECTION("normal scan consumes only the actual prefix after shrink") {
        InnerSearchParam param;
        param.search_mode = KNN_SEARCH;
        const auto records =
            RunFlatSearch(distances, param, 64, nullptr, {}, nullptr, nullptr, false, 5);

        REQUIRE(records.size() == 5);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{0, 1, 2, 3, 4});
    }

    SECTION("empty tail after shrink is not consumed") {
        InnerSearchParam param;
        param.search_mode = RANGE_SEARCH;
        param.radius = 100.0F;
        const auto records =
            RunFlatSearch(distances, param, 64, nullptr, {}, nullptr, nullptr, false, 0);

        REQUIRE(records.empty());
    }

    SECTION("empty normal scan remains bounded when the first lane is appended") {
        InnerSearchParam param;
        param.search_mode = KNN_SEARCH;
        const auto records = RunFlatSearch({}, param, 5, nullptr, {}, nullptr, nullptr, true);

        REQUIRE(records.empty());
    }
}

TEST_CASE("IVF RaBitQSplitBucketSearcher tracks KNN survivors",
          "[ut][ivf][rabitq_split_bucket_searcher][auxiliary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<float> auxiliary_values;
    InnerSearchParam param;
    param.search_mode = KNN_SEARCH;
    param.enable_reorder = true;

    std::vector<float> distances(37);
    std::vector<float> filter_inner_products(37);
    for (uint64_t i = 0; i < distances.size(); ++i) {
        distances[i] = static_cast<float>(distances.size() - i);
        filter_inner_products[i] = 1000.0F + static_cast<float>(i);
    }

    SECTION("fast path retains payloads and source provenance for tail survivors") {
        std::vector<DistanceHeap::AuxiliaryDistanceRecord> auxiliary_records;
        const auto records = RunFlatSearch(distances,
                                           param,
                                           5,
                                           &auxiliary_values,
                                           filter_inner_products,
                                           nullptr,
                                           &auxiliary_records);

        RequireAuxiliaryMatchesHeap(records, auxiliary_values, filter_inner_products);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{32, 33, 34, 35, 36});
        REQUIRE(auxiliary_records.size() == records.size());
        for (const auto& record : auxiliary_records) {
            REQUIRE(record.source_bucket_id == 0);
            REQUIRE(record.source_offset_id == record.record.second);
            REQUIRE(record.source_version == StaticDistanceBucket::kSourceVersion);
        }
    }

    SECTION("candidate scan bounds append to the allocated prefix and snapshots ids") {
        std::vector<DistanceHeap::AuxiliaryDistanceRecord> auxiliary_records;
        const auto records = RunFlatSearch(distances,
                                           param,
                                           64,
                                           &auxiliary_values,
                                           filter_inner_products,
                                           nullptr,
                                           &auxiliary_records,
                                           true);

        REQUIRE(records.size() == distances.size());
        REQUIRE(auxiliary_records.size() == distances.size());
        REQUIRE(std::none_of(records.begin(), records.end(), [&](const auto& record) {
            return record.second == distances.size();
        }));
        for (const auto& record : auxiliary_records) {
            REQUIRE(record.source_offset_id == record.record.second);
            REQUIRE(record.source_version == StaticDistanceBucket::kSourceVersion);
        }
    }

    SECTION("candidate scan consumes only the actual locked prefix after shrink") {
        std::vector<DistanceHeap::AuxiliaryDistanceRecord> auxiliary_records;
        const auto records = RunFlatSearch(distances,
                                           param,
                                           64,
                                           &auxiliary_values,
                                           filter_inner_products,
                                           nullptr,
                                           &auxiliary_records,
                                           false,
                                           5);

        REQUIRE(records.size() == 5);
        REQUIRE(auxiliary_records.size() == 5);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{0, 1, 2, 3, 4});
    }

    SECTION("empty candidate scan remains bounded when the first lane is appended") {
        const auto records =
            RunFlatSearch({}, param, 5, &auxiliary_values, {}, nullptr, nullptr, true);

        REQUIRE(records.empty());
        REQUIRE(auxiliary_values.empty());
    }

    SECTION("filter scalar path keeps payloads for valid survivors") {
        param.is_inner_id_allowed = std::make_shared<EvenIdFilter>();
        const auto records =
            RunFlatSearch(distances, param, 4, &auxiliary_values, filter_inner_products);

        RequireAuxiliaryMatchesHeap(records, auxiliary_values, filter_inner_products);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{30, 32, 34, 36});
    }

    SECTION("reasoning scalar path keeps payloads synchronized") {
        ReasoningContext reasoning_ctx(allocator.get());
        const auto records = RunFlatSearch(
            distances, param, 3, &auxiliary_values, filter_inner_products, &reasoning_ctx);

        RequireAuxiliaryMatchesHeap(records, auxiliary_values, filter_inner_products);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{34, 35, 36});
    }

    SECTION("reorder keeps payloads for approximate candidates above threshold") {
        param.distance_threshold = 0.5F;
        const auto records =
            RunFlatSearch(distances, param, 3, &auxiliary_values, filter_inner_products);

        RequireAuxiliaryMatchesHeap(records, auxiliary_values, filter_inner_products);
        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{34, 35, 36});
        for (const auto& record : records) {
            REQUIRE(record.first > param.distance_threshold.value());
        }
    }

    SECTION("disabled reorder leaves auxiliary values missing") {
        param.enable_reorder = false;
        param.distance_threshold = 5.0F;
        std::fill(distances.begin(), distances.end(), 10.0F);
        distances[0] = 1.0F;
        distances[1] = 2.0F;
        distances[2] = 3.0F;
        const auto records =
            RunFlatSearch(distances, param, 3, &auxiliary_values, filter_inner_products);

        REQUIRE(SortedIds(records) == std::vector<InnerIdType>{0, 1, 2});
        REQUIRE(std::all_of(auxiliary_values.begin(), auxiliary_values.end(), [](float value) {
            return IsNaNBits(value);
        }));
    }

    SECTION("zero topk removes every transient auxiliary entry") {
        const auto records =
            RunFlatSearch(distances, param, 0, &auxiliary_values, filter_inner_products);

        REQUIRE(records.empty());
        REQUIRE(auxiliary_values.empty());
    }

    SECTION("empty split bucket does not require auxiliary scratch") {
        const auto records = RunFlatSearch({}, param, 5, &auxiliary_values, {});

        REQUIRE(records.empty());
        REQUIRE(auxiliary_values.empty());
    }
}

}  // namespace vsag
