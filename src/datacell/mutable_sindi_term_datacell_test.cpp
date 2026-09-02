
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

#include "mutable_sindi_term_datacell.h"

#include <array>
#include <limits>
#include <set>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

namespace {

class ModuloFilter final : public Filter {
public:
    explicit ModuloFilter(uint32_t divisor) : divisor_(divisor) {
    }

    bool
    CheckValid(int64_t id) const override {
        return id % divisor_ == 0;
    }

private:
    uint32_t divisor_;
};

uint64_t
QueryFirstWindow(const MutableSindiTermDataCellPtr& data_cell,
                 float* dists,
                 const SparseTermComputerPtr& computer,
                 Allocator* allocator) {
    SindiQueryContext query_context(allocator);
    data_cell->QueryWindow(dists, 0, computer, false, query_context);
    return query_context.evaluation_tracker.Count();
}

template <InnerSearchMode mode, InnerSearchType type>
void
RequireHeapEquivalence(const MutableSindiTermDataCellPtr& data_cell,
                       const SparseTermComputerPtr& computer,
                       uint32_t posting_count,
                       const FilterPtr& filter,
                       Allocator* allocator) {
    std::vector<float> term_list_distances(posting_count, 0.0F);
    std::vector<float> full_scan_distances(posting_count, 0.0F);
    QueryFirstWindow(data_cell, term_list_distances.data(), computer, allocator);
    QueryFirstWindow(data_cell, full_scan_distances.data(), computer, allocator);

    InnerSearchParam search_param;
    search_param.ef = 20;
    search_param.radius = 0.99F;
    search_param.is_inner_id_allowed = filter;
    MaxHeap term_list_heap(allocator);
    MaxHeap full_scan_heap(allocator);
    REQUIRE_FALSE(data_cell->InsertHeapByTermLists<mode, type>(
        term_list_distances.data(), computer, term_list_heap, search_param, 0));
    REQUIRE_FALSE(data_cell->InsertHeapByDists<mode, type>(
        full_scan_distances.data(), full_scan_distances.size(), full_scan_heap, search_param, 0));

    std::vector<std::pair<float, InnerIdType>> term_list_results;
    std::vector<std::pair<float, InnerIdType>> full_scan_results;
    while (!term_list_heap.empty()) {
        term_list_results.push_back(term_list_heap.top());
        term_list_heap.pop();
    }
    while (!full_scan_heap.empty()) {
        full_scan_results.push_back(full_scan_heap.top());
        full_scan_heap.pop();
    }
    const auto by_distance_and_id = [](const auto& left, const auto& right) {
        return left.first == right.first ? left.second < right.second : left.first < right.first;
    };
    std::sort(term_list_results.begin(), term_list_results.end(), by_distance_and_id);
    std::sort(full_scan_results.begin(), full_scan_results.end(), by_distance_and_id);
    REQUIRE(term_list_results.size() == full_scan_results.size());
    for (uint32_t i = 0; i < term_list_results.size(); ++i) {
        REQUIRE(term_list_results[i].first == full_scan_results[i].first);
        const auto distance = term_list_results[i].first;
        const bool unique_distance =
            (i == 0 || term_list_results[i - 1].first != distance) &&
            (i + 1 == term_list_results.size() || term_list_results[i + 1].first != distance);
        if (unique_distance) {
            REQUIRE(term_list_results[i].second == full_scan_results[i].second);
        }
    }
}

}  // namespace

TEST_CASE("MutableSindiTermDataCell uses caller document id coordinates",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell =
        std::make_shared<MutableSindiTermDataCell>(16,
                                                   4,
                                                   allocator.get(),
                                                   SparseValueQuantizationType::FP32,
                                                   std::make_shared<QuantizationParams>());

    uint32_t term_ids[] = {3};
    float term_values[] = {1.0F};
    SparseVector vector{1, term_ids, term_values};
    data_cell->InsertVector(vector, 1);

    REQUIRE(data_cell->GetWindowCount() == 1);
    REQUIRE(data_cell->GetWindow(0).term_sizes_[3] == 1);
    REQUIRE(data_cell->GetWindow(0).term_ids_[3]->front() == 1);
    REQUIRE(data_cell->total_count_ == 2);

    data_cell->InsertVector(vector, 5);
    REQUIRE(data_cell->GetWindowCount() == 2);
    REQUIRE(data_cell->GetWindow(1).term_ids_[3]->front() == 1);
}

TEST_CASE("MutableSindiTermDataCell inserts every provided term",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(
        2000, 100, allocator.get(), SparseValueQuantizationType::FP32, nullptr);

    uint32_t term_ids[] = {1000, 3};
    float term_values[] = {0.01F, 1.0F};
    SparseVector vector{2, term_ids, term_values};
    data_cell->InsertVector(vector, 0);

    const auto& window = data_cell->GetWindow(0);
    REQUIRE(window.term_capacity_ == 1001);
    REQUIRE(window.term_sizes_[1000] == 1);
    REQUIRE(window.term_sizes_[3] == 1);
}

TEST_CASE("MutableSindiTermDataCell sorts postings by stored value",
          "[ut][MutableSindiTermDataCell]") {
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    DYNAMIC_SECTION("quantization=" << static_cast<int>(quantization)) {
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto quantization_params = std::make_shared<QuantizationParams>();
        quantization_params->min_val = 0.0F;
        quantization_params->max_val = 4.0F;
        quantization_params->diff = 4.0F;
        MutableSindiTermDataCell data_cell(
            16, 4, allocator.get(), quantization, quantization_params);

        uint32_t term = 3;
        std::array<float, 4> values = {1.0F, 3.0F, 2.0F, 3.0F};
        for (uint32_t document = 0; document < values.size(); ++document) {
            SparseVector vector{1, &term, values.data() + document};
            data_cell.InsertVector(vector, document);
        }

        REQUIRE_FALSE(data_cell.GetWindow(0).postings_sorted_);
        data_cell.SortByValue(0);
        REQUIRE(data_cell.GetWindow(0).postings_sorted_);
        data_cell.SortByValue(0);
        REQUIRE(data_cell.GetWindow(0).postings_sorted_);
        const auto& window = data_cell.GetWindow(0);
        const std::array<uint16_t, 4> expected_ids = {1, 3, 2, 0};
        REQUIRE(
            std::equal(expected_ids.begin(), expected_ids.end(), window.term_ids_[term]->begin()));

        const auto value_code_size = sindi_datacell_utils::GetValueCodeSize(quantization);
        float previous = std::numeric_limits<float>::infinity();
        for (uint32_t posting = 0; posting < values.size(); ++posting) {
            const auto decoded = sindi_datacell_utils::DecodeValue(
                window.term_datas_[term]->data() + static_cast<uint64_t>(posting) * value_code_size,
                quantization,
                quantization_params.get());
            REQUIRE(decoded <= previous);
            previous = decoded;
        }
    }
}

TEST_CASE("MutableSindiTermDataCell prunes incrementally appended postings",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto quantization_params = std::make_shared<QuantizationParams>();
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(
        16, 8, allocator.get(), SparseValueQuantizationType::FP32, quantization_params);

    uint32_t term = 3;
    std::array<float, 4> values = {1.0F, 3.0F, 2.0F, 4.0F};
    for (uint32_t document = 0; document < 3; ++document) {
        SparseVector vector{1, &term, values.data() + document};
        data_cell->InsertVector(vector, document);
    }
    data_cell->SortByValue(0);

    SparseVector appended{1, &term, values.data() + 3};
    data_cell->InsertVector(appended, 3);
    REQUIRE(data_cell->GetWindow(0).term_sizes_[term] == 4);
    REQUIRE_FALSE(data_cell->GetWindow(0).postings_sorted_);

    float query_value = 1.0F;
    SparseVector query{1, &term, &query_value};
    SINDISearchParameter search_parameter;
    search_parameter.term_retain_threshold = 1;
    auto computer = std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
    std::array<float, 4> sorted_distances{};
    QueryFirstWindow(data_cell, sorted_distances.data(), computer, allocator.get());
    REQUIRE(sorted_distances[3] == -4.0F);
    REQUIRE(sorted_distances[0] == 0.0F);
    REQUIRE(sorted_distances[1] == 0.0F);
    REQUIRE(sorted_distances[2] == 0.0F);
}

TEST_CASE("MutableSindiTermDataCell two-run search matches eager normalization",
          "[ut][MutableSindiTermDataCell]") {
    const auto quantization = GENERATE(SparseValueQuantizationType::FP32,
                                       SparseValueQuantizationType::FP16,
                                       SparseValueQuantizationType::SQ8);
    DYNAMIC_SECTION("quantization=" << static_cast<int>(quantization)) {
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto quantization_params = std::make_shared<QuantizationParams>();
        quantization_params->min_val = 0.0F;
        quantization_params->max_val = 1024.0F;
        quantization_params->diff = 1024.0F;
        uint32_t term = 9;
        constexpr uint32_t prefix_count = 300;
        constexpr uint32_t posting_count = 340;
        auto split = std::make_shared<MutableSindiTermDataCell>(
            16, 512, allocator.get(), quantization, quantization_params);
        auto eager = std::make_shared<MutableSindiTermDataCell>(
            16, 512, allocator.get(), quantization, quantization_params);

        std::array<float, posting_count> values{};
        for (uint32_t id = 0; id < posting_count; ++id) {
            values[id] = static_cast<float>((id * 43U) % 401U + 25U);
            if (id == prefix_count) {
                values[id] = 1000.0F;
            }
            SparseVector vector{1, &term, values.data() + id};
            split->InsertVector(vector, id);
            eager->InsertVector(vector, id);
            if (id + 1 == prefix_count) {
                split->SortByValue(0);
            }
        }
        REQUIRE(split->FinalizeInsertBatch(0));
        eager->SortByValue(0);
        REQUIRE(split->GetWindow(0).dirty_posting_prefixes_.at(term) == prefix_count);

        const std::array<uint32_t, 6> retained_counts{1, 31, 32, 255, 256, posting_count};
        for (const auto retained_count : retained_counts) {
            CAPTURE(quantization, retained_count);
            float query_value = 1.0F;
            SparseVector query{1, &term, &query_value};
            SINDISearchParameter search_parameter;
            search_parameter.term_retain_threshold = retained_count;
            auto split_computer =
                std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
            auto eager_computer =
                std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
            std::array<float, posting_count> split_distances{};
            std::array<float, posting_count> eager_distances{};
            REQUIRE(
                QueryFirstWindow(split, split_distances.data(), split_computer, allocator.get()) ==
                retained_count);
            REQUIRE(
                QueryFirstWindow(eager, eager_distances.data(), eager_computer, allocator.get()) ==
                retained_count);
            REQUIRE(split_distances == eager_distances);
        }

        float query_value = 1.0F;
        SparseVector query{1, &term, &query_value};
        SINDISearchParameter search_parameter;
        search_parameter.term_retain_threshold = 256;
        auto computer =
            std::make_shared<SparseTermComputer>(query, search_parameter, allocator.get());
        RequireHeapEquivalence<InnerSearchMode::KNN_SEARCH, InnerSearchType::PURE>(
            split, computer, posting_count, nullptr, allocator.get());
        RequireHeapEquivalence<InnerSearchMode::RANGE_SEARCH, InnerSearchType::PURE>(
            split, computer, posting_count, nullptr, allocator.get());
        auto filter = std::make_shared<ModuloFilter>(3);
        RequireHeapEquivalence<InnerSearchMode::KNN_SEARCH, InnerSearchType::WITH_FILTER>(
            split, computer, posting_count, filter, allocator.get());
        RequireHeapEquivalence<InnerSearchMode::RANGE_SEARCH, InnerSearchType::WITH_FILTER>(
            split, computer, posting_count, filter, allocator.get());

        REQUIRE(split->NormalizeDirtyPostings(0));
        REQUIRE_FALSE(split->NormalizeDirtyPostings(0));
        REQUIRE(*split->GetWindow(0).term_ids_[term] == *eager->GetWindow(0).term_ids_[term]);
        REQUIRE(*split->GetWindow(0).term_datas_[term] == *eager->GetWindow(0).term_datas_[term]);
        REQUIRE(split->normalization_ids_scratch_.capacity() == 0);
        REQUIRE(split->normalization_data_scratch_.capacity() == 0);
    }
}

TEST_CASE("MutableSindiTermDataCell bounds dirty posting runs", "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(
        16, 1024, allocator.get(), SparseValueQuantizationType::FP32, nullptr);
    uint32_t term = 7;
    std::array<float, 556> values{};
    for (uint32_t id = 0; id < 300; ++id) {
        values[id] = static_cast<float>((id * 37U) % 113U + 1U);
        SparseVector vector{1, &term, values.data() + id};
        data_cell->InsertVector(vector, id);
    }
    REQUIRE(data_cell->FinalizeInsertBatch(0));
    REQUIRE(data_cell->GetWindow(0).dirty_posting_prefixes_.empty());

    for (uint32_t id = 300; id < 555; ++id) {
        values[id] = static_cast<float>((id * 37U) % 113U + 1U);
        SparseVector vector{1, &term, values.data() + id};
        data_cell->InsertVector(vector, id);
        REQUIRE(data_cell->FinalizeInsertBatch(0));
    }
    REQUIRE(data_cell->GetWindow(0).dirty_posting_prefixes_.at(term) == 300);

    values[555] = 1000.0F;
    SparseVector vector{1, &term, values.data() + 555};
    data_cell->InsertVector(vector, 555);
    REQUIRE(data_cell->FinalizeInsertBatch(0));
    REQUIRE(data_cell->GetWindow(0).dirty_posting_prefixes_.empty());
    REQUIRE(data_cell->GetWindow(0).postings_sorted_);

    values[100] = 2000.0F;
    SparseVector appended{1, &term, values.data() + 100};
    data_cell->InsertVector(appended, 556);
    REQUIRE(data_cell->FinalizeInsertBatch(0));
    REQUIRE_FALSE(data_cell->GetWindow(0).dirty_posting_prefixes_.empty());
    data_cell->Compact();
    REQUIRE(data_cell->GetWindow(0).dirty_posting_prefixes_.empty());
    REQUIRE(data_cell->GetWindow(0).postings_sorted_);
    REQUIRE(data_cell->normalization_ids_scratch_.capacity() == 0);
    REQUIRE(data_cell->normalization_data_scratch_.capacity() == 0);
}

TEST_CASE("MutableSindiTermDataCell normalizes legacy posting order on deserialize",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MutableSindiTermDataCell source(
        16, 8, allocator.get(), SparseValueQuantizationType::FP32, nullptr);
    uint32_t term = 3;
    std::array<float, 2> values = {1.0F, 4.0F};
    for (uint32_t document = 0; document < values.size(); ++document) {
        SparseVector vector{1, &term, values.data() + document};
        source.InsertVector(vector, document);
    }
    std::stringstream stream;
    IOStreamWriter writer(stream);
    source.SerializeWindows(writer);

    MutableSindiTermDataCell restored(
        16, 8, allocator.get(), SparseValueQuantizationType::FP32, nullptr);
    IOStreamReader reader(stream);
    restored.DeserializeWindows(reader, 1);
    REQUIRE(*restored.GetWindow(0).term_ids_[term] == Vector<uint16_t>({1, 0}, allocator.get()));
}

TEST_CASE("MutableSindiTermDataCell trusts versioned posting order on deserialize",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MutableSindiTermDataCell source(
        16, 8, allocator.get(), SparseValueQuantizationType::FP32, nullptr);
    uint32_t term = 3;
    std::array<float, 2> values = {1.0F, 4.0F};
    for (uint32_t document = 0; document < values.size(); ++document) {
        SparseVector vector{1, &term, values.data() + document};
        source.InsertVector(vector, document);
    }
    std::stringstream stream;
    IOStreamWriter writer(stream);
    source.SerializeWindows(writer);

    MutableSindiTermDataCell restored(
        16, 8, allocator.get(), SparseValueQuantizationType::FP32, nullptr);
    IOStreamReader reader(stream);
    restored.DeserializeWindows(reader, 1, true);
    REQUIRE(*restored.GetWindow(0).term_ids_[term] == Vector<uint16_t>({0, 1}, allocator.get()));
}

TEST_CASE("MutableSindiTermDataCell Basic Test", "[ut][MutableSindiTermDataCell]") {
    // prepare data
    auto count_base = 10;
    auto len_base = 10;
    std::vector<SparseVector> sparse_vectors(count_base);
    for (int i = 0; i < count_base; i++) {
        sparse_vectors[i].len_ = len_base;
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].len_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].len_];
        // base[0] = [0:0, 1:1, 2:2, ..., 9:9] = after_prune = [7:7, 8:8, 9:9]
        // base[1] = [1:1, 2:2, 3:3, ..., 10:10] = after_prune = [7:7, 8:8, 9:9, 10:10]
        // base[2] = [2:2, 3:3, 4:4, ..., 11:11] = after_prune = [8:8, 9:9, 10:10, 11:11]
        // base[3] = [3:3, 4:4, 5:5, ..., 12:12] = after_prune = [9:9, 10:10, 11:11, 12:12]
        // base[4] = [4:4, 5:5, 6:6, ..., 13:13] = after_prune = [10:10, 11:11, 12:12, 13:13]
        // base[5] = [5:5, 6:6, 7:7, ..., 14:14] = after_prune = [11:11, 12:12, 13:13, 14:14]
        // base[6] = [6:6, 7:7, 8:8, ..., 15:15] = after_prune = [12:12, 13:13, 14:14, 15:15]
        // base[7] = [7:7, 8:8, 9:9, ..., 16:16] = after_prune = [13:13, 14:14, 15:15, 16:16]
        // base[8] = [8:8, 9:9, 10:10, ..., 17:17]
        // after_prune = [13:13, 14:14, 15:15, 16:16, 17:17]
        // base[9] = [9:9, 10:10, 11:11, ..., 18:18]
        // after_prune = [14:14, 15:15, 16:16, 17:17, 18:18]
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            sparse_vectors[i].ids_[d] = i + d;
            sparse_vectors[i].vals_[d] = i + d;
        }
    }

    // query: [0:1, 1:1, 2:1 .... 18:1]
    // dis(q, b0) = 9 + 8 + 7 = 24
    // dis(q, b1) = 10 + 9 + 8 + 7 = 34
    // ...
    // dis(q, b9) = 80
    SparseVector query_sv;
    query_sv.len_ = 19;
    query_sv.ids_ = new uint32_t[query_sv.len_];
    query_sv.vals_ = new float[query_sv.len_];
    for (int d = 0; d < query_sv.len_; d++) {
        query_sv.ids_[d] = d;
        query_sv.vals_[d] = 1;
    }

    // prepare data_cell
    float query_prune_ratio = 0.0;
    float term_prune_ratio = 0.0;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // disable quantization for this basic test
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                                DEFAULT_WINDOW_SIZE,
                                                                allocator.get(),
                                                                SparseValueQuantizationType::FP32,
                                                                q_params);
    // test factory computer
    SINDISearchParameter search_params;
    search_params.term_prune_ratio = term_prune_ratio;
    search_params.query_prune_ratio = query_prune_ratio;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());
    REQUIRE(computer->pruned_len_ == (1.0F - query_prune_ratio) * query_sv.len_);

    // test insert
    auto exp_id_size = 19;
    std::vector<uint32_t> exp_size = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    for (auto i = 0; i < count_base; i++) {
        data_cell->InsertVector(sparse_vectors[i], i);
    }
    REQUIRE(data_cell->GetWindow(0).term_capacity_ >= exp_id_size);
    REQUIRE(data_cell->GetWindow(0).term_ids_.size() == data_cell->GetWindow(0).term_capacity_);
    REQUIRE(data_cell->GetWindow(0).term_datas_.size() == data_cell->GetWindow(0).term_capacity_);
    for (auto i = 0; i < exp_id_size; i++) {
        if (exp_size[i] == 0) {
            REQUIRE(data_cell->GetWindow(0).term_ids_[i] == nullptr);
            REQUIRE(data_cell->GetWindow(0).term_datas_[i] == nullptr);
        } else {
            REQUIRE(data_cell->GetWindow(0).term_ids_[i]->size() ==
                    data_cell->GetWindow(0).term_sizes_[i]);
            REQUIRE(data_cell->GetWindow(0).term_ids_[i]->size() == exp_size[i]);
            REQUIRE(data_cell->GetWindow(0).term_datas_[i]->size() == exp_size[i] * sizeof(float));
        }
    }
    for (auto i = exp_id_size; i < data_cell->GetWindow(0).term_capacity_; i++) {
        REQUIRE(data_cell->GetWindow(0).term_ids_[i] == nullptr);
        REQUIRE(data_cell->GetWindow(0).term_datas_[i] == nullptr);
    }

    // Calculate expected distances programmatically to match the test logic
    std::vector<float> exp_dists(count_base, 0.0f);
    for (int i = 0; i < count_base; ++i) {
        const auto& vec = sparse_vectors[i];
        float total_dist = 0.0f;
        for (uint32_t term = 0; term < vec.len_; ++term) {
            float val = vec.vals_[term];
            float query_val = -1.0f;  // The computer uses -1.0 as query value
            total_dist += query_val * val;
        }
        exp_dists[i] = total_dist;
    }

    SECTION("test query") {
        std::vector<float> dists(count_base, 0);
        REQUIRE(QueryFirstWindow(data_cell, dists.data(), computer, allocator.get()) == count_base);
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] - exp_dists[i]) < 1e-3);
        }
        std::fill(dists.begin(), dists.end(), 0.0F);
        REQUIRE(QueryFirstWindow(data_cell, dists.data(), computer, allocator.get()) == count_base);
    }

    SECTION("test insert heap in knn search") {
        auto topk = 5;
        auto pos = count_base - topk;
        InnerSearchParam inner_param;
        inner_param.ef = topk;
        MaxHeap heap(allocator.get());
        std::vector<float> dists(count_base, 0);
        QueryFirstWindow(data_cell, dists.data(), computer, allocator.get());

        data_cell->InsertHeapByTermLists<KNN_SEARCH, PURE>(
            dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == topk);

        // Extract results from InsertHeapByTermLists
        std::vector<std::pair<float, int64_t>> results_by_term_lists;
        while (!heap.empty()) {
            results_by_term_lists.push_back(heap.top());
            heap.pop();
        }

        for (auto i = 0; i < topk; i++) {
            auto exp_id = pos + i;
            REQUIRE(results_by_term_lists[i].second == exp_id);
            REQUIRE(std::abs(results_by_term_lists[i].first - exp_dists[exp_id]) < 1e-3);
        }

        std::vector<float> dists2(count_base, 0);
        QueryFirstWindow(data_cell, dists2.data(), computer, allocator.get());
        MaxHeap heap2(allocator.get());
        data_cell->InsertHeapByDists<KNN_SEARCH, PURE>(
            dists2.data(), dists2.size(), heap2, inner_param, 0);

        // Extract results from InsertHeapByDists
        std::vector<std::pair<float, int64_t>> results_by_dists;
        while (!heap2.empty()) {
            results_by_dists.push_back(heap2.top());
            heap2.pop();
        }
        // Compare results from both methods
        REQUIRE(results_by_term_lists.size() == results_by_dists.size());
        for (size_t i = 0; i < results_by_term_lists.size(); i++) {
            REQUIRE(results_by_term_lists[i].second == results_by_dists[i].second);
            REQUIRE(std::abs(results_by_term_lists[i].first - results_by_dists[i].first) < 1e-3);
        }
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] - 0) < 1e-3);
        }
    }

    SECTION("test insert heap in range search") {
        auto range_topk = 3;
        auto pos = count_base - range_topk - 1;  // note that we retrieval dist < dists[pos]
        InnerSearchParam inner_param;
        std::vector<float> dists(count_base, 0);
        QueryFirstWindow(data_cell, dists.data(), computer, allocator.get());
        inner_param.radius = dists[pos];
        MaxHeap heap(allocator.get());

        data_cell->InsertHeapByTermLists<RANGE_SEARCH, PURE>(
            dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == range_topk);

        // Extract results from InsertHeapByTermLists
        std::vector<std::pair<float, int64_t>> results_by_term_lists;
        while (!heap.empty()) {
            results_by_term_lists.push_back(heap.top());
            heap.pop();
        }
        for (auto i = 0; i < range_topk; i++) {
            auto exp_id = pos + i + 1;
            REQUIRE(results_by_term_lists[i].second == exp_id);
            REQUIRE(std::abs(results_by_term_lists[i].first - exp_dists[exp_id]) < 1e-3);
        }

        std::vector<float> dists2(count_base, 0);
        QueryFirstWindow(data_cell, dists2.data(), computer, allocator.get());
        MaxHeap heap2(allocator.get());
        data_cell->InsertHeapByDists<RANGE_SEARCH, PURE>(
            dists2.data(), dists2.size(), heap2, inner_param, 0);

        // Extract results from InsertHeapByDists
        std::vector<std::pair<float, int64_t>> results_by_dists;
        while (!heap2.empty()) {
            results_by_dists.push_back(heap2.top());
            heap2.pop();
        }

        // Compare results from both methods
        REQUIRE(results_by_term_lists.size() == results_by_dists.size());
        for (size_t i = 0; i < results_by_term_lists.size(); i++) {
            REQUIRE(results_by_term_lists[i].second == results_by_dists[i].second);
            REQUIRE(std::abs(results_by_term_lists[i].first - results_by_dists[i].first) < 1e-3);
        }
        for (auto i = 0; i < range_topk; i++) {
            REQUIRE(std::abs(dists[i] - 0) < 1e-3);
        }
    }

    SECTION("test zero distance is ignored in range search") {
        InnerSearchParam inner_param;
        inner_param.radius = 1.0F;
        std::vector<float> dists(count_base, 0.0F);
        dists[2] = -0.25F;
        dists[7] = -0.5F;
        MaxHeap heap(allocator.get());

        data_cell->InsertHeapByDists<RANGE_SEARCH, PURE>(
            dists.data(), dists.size(), heap, inner_param, 0);

        REQUIRE(heap.size() == 2);
        std::set<int64_t> result_ids;
        while (not heap.empty()) {
            result_ids.insert(heap.top().second);
            heap.pop();
        }
        REQUIRE(result_ids == std::set<int64_t>{2, 7});
    }
    // clean
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    delete[] query_sv.ids_;
    delete[] query_sv.vals_;
}

TEST_CASE("MutableSindiTermDataCell Encode/Decode Test", "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // Prepare data
    std::vector<uint32_t> ids = {10, 20, 30};
    std::vector<float> vals = {1.1f, 2.2f, 3.3f};
    SparseVector sv;
    sv.len_ = ids.size();
    sv.ids_ = ids.data();
    sv.vals_ = vals.data();

    float min_val = 1.1f;
    float max_val = 3.3f;

    // Prepare datacell
    auto q_params = std::make_shared<QuantizationParams>();
    q_params->min_val = min_val;
    q_params->max_val = max_val;
    q_params->diff = max_val - min_val;
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                                DEFAULT_WINDOW_SIZE,
                                                                allocator.get(),
                                                                SparseValueQuantizationType::SQ8,
                                                                q_params);

    // Insert vector (tests Encode)
    uint16_t base_id = 5;
    data_cell->InsertVector(sv, base_id);

    // Get vector (tests Decode)
    SparseVector retrieved_sv;
    data_cell->GetSparseVector(base_id, &retrieved_sv, allocator.get());

    REQUIRE(retrieved_sv.len_ == sv.len_);

    // Verify results
    std::map<uint32_t, float> retrieved_map;
    for (size_t i = 0; i < retrieved_sv.len_; ++i) {
        retrieved_map[retrieved_sv.ids_[i]] = retrieved_sv.vals_[i];
    }

    float tolerance = 0.1f;

    for (size_t i = 0; i < sv.len_; ++i) {
        REQUIRE(retrieved_map.count(sv.ids_[i]));
        REQUIRE(std::abs(retrieved_map[sv.ids_[i]] - sv.vals_[i]) < tolerance);
    }

    allocator->Deallocate(retrieved_sv.ids_);
    allocator->Deallocate(retrieved_sv.vals_);
}

TEST_CASE("MutableSindiTermDataCell FP16 Roundtrip Test", "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    std::vector<uint32_t> ids = {1, 3, 7};
    std::vector<float> vals = {1.5F, 2.25F, 3.75F};
    SparseVector sv;
    sv.len_ = ids.size();
    sv.ids_ = ids.data();
    sv.vals_ = vals.data();

    auto data_cell = std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                                DEFAULT_WINDOW_SIZE,
                                                                allocator.get(),
                                                                SparseValueQuantizationType::FP16,
                                                                nullptr);
    uint16_t base_id = 2;
    data_cell->InsertVector(sv, base_id);

    std::vector<float> query_vals(ids.size(), 1.0F);
    SparseVector query_sv;
    query_sv.len_ = ids.size();
    query_sv.ids_ = ids.data();
    query_sv.vals_ = query_vals.data();

    SINDISearchParameter search_params;
    search_params.term_prune_ratio = 0.0F;
    search_params.query_prune_ratio = 0.0F;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());

    std::vector<float> dists(4, 0.0F);
    QueryFirstWindow(data_cell, dists.data(), computer, allocator.get());
    REQUIRE(std::abs(dists[base_id] + 7.5F) < 1e-3F);

    REQUIRE(std::abs(data_cell->CalcDistanceByInnerId(computer, base_id) - (1.0F - 7.5F)) < 1e-3F);

    SparseVector retrieved_sv;
    data_cell->GetSparseVector(base_id, &retrieved_sv, allocator.get());
    REQUIRE(retrieved_sv.len_ == sv.len_);
    for (uint32_t i = 0; i < retrieved_sv.len_; ++i) {
        REQUIRE(retrieved_sv.ids_[i] == sv.ids_[i]);
        REQUIRE(std::abs(retrieved_sv.vals_[i] - sv.vals_[i]) < 1e-3F);
    }

    allocator->Deallocate(retrieved_sv.ids_);
    allocator->Deallocate(retrieved_sv.vals_);
}

TEST_CASE("MutableSindiTermDataCell Compact Memory Usage", "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                                DEFAULT_WINDOW_SIZE,
                                                                allocator.get(),
                                                                SparseValueQuantizationType::FP32,
                                                                nullptr);

    std::vector<uint32_t> ids = {1, 3, 5};
    std::vector<float> vals = {1.0F, 2.0F, 3.0F};
    SparseVector sparse_vector;
    sparse_vector.len_ = static_cast<uint32_t>(ids.size());
    sparse_vector.ids_ = ids.data();
    sparse_vector.vals_ = vals.data();
    data_cell->InsertVector(sparse_vector, 0);

    data_cell->GetWindow(0).term_ids_[1]->reserve(32);
    data_cell->GetWindow(0).term_datas_[1]->reserve(128);
    auto compact_capacity = data_cell->GetWindow(0).term_capacity_;
    auto reserved_id_capacity = data_cell->GetWindow(0).term_ids_[1]->capacity();
    auto reserved_data_capacity = data_cell->GetWindow(0).term_datas_[1]->capacity();
    data_cell->ResizeTermList(128);

    auto memory_with_reserved_capacity = data_cell->GetMemoryUsage();
    REQUIRE(data_cell->GetWindow(0).term_capacity_ >= 128);

    data_cell->Compact();

    REQUIRE(data_cell->GetWindow(0).term_capacity_ == compact_capacity);
    REQUIRE(data_cell->GetWindow(0).term_ids_[1]->capacity() <= reserved_id_capacity);
    REQUIRE(data_cell->GetWindow(0).term_datas_[1]->capacity() <= reserved_data_capacity);
    REQUIRE(data_cell->GetMemoryUsage() < memory_with_reserved_capacity);
}

TEST_CASE("MutableSindiTermDataCell Deserialize Compacts Capacity",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell = std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                                DEFAULT_WINDOW_SIZE,
                                                                allocator.get(),
                                                                SparseValueQuantizationType::FP32,
                                                                nullptr);

    std::vector<uint32_t> ids = {1, 3, 5};
    std::vector<float> vals = {1.0F, 2.0F, 3.0F};
    SparseVector sparse_vector;
    sparse_vector.len_ = static_cast<uint32_t>(ids.size());
    sparse_vector.ids_ = ids.data();
    sparse_vector.vals_ = vals.data();
    data_cell->InsertVector(sparse_vector, 0);
    data_cell->ResizeTermList(128);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    data_cell->SerializeWindows(writer);

    MutableSindiTermDataCell restored(DEFAULT_TERM_ID_LIMIT,
                                      DEFAULT_WINDOW_SIZE,
                                      allocator.get(),
                                      SparseValueQuantizationType::FP32,
                                      nullptr);
    IOStreamReader reader(ss);
    restored.DeserializeWindows(reader, 1);

    REQUIRE(restored.GetWindow(0).term_capacity_ == 6);
    REQUIRE(restored.GetMemoryUsage() < data_cell->GetMemoryUsage());
}

TEST_CASE("MutableSindiTermDataCell Deserialize Clears Stale Posting Lists",
          "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto make_sparse_vector = [](std::vector<uint32_t>& ids, std::vector<float>& vals) {
        SparseVector sparse_vector;
        sparse_vector.len_ = static_cast<uint32_t>(ids.size());
        sparse_vector.ids_ = ids.data();
        sparse_vector.vals_ = vals.data();
        return sparse_vector;
    };

    MutableSindiTermDataCell stale(DEFAULT_TERM_ID_LIMIT,
                                   DEFAULT_WINDOW_SIZE,
                                   allocator.get(),
                                   SparseValueQuantizationType::FP32,
                                   nullptr);
    std::vector<uint32_t> stale_ids = {9};
    std::vector<float> stale_vals = {9.0F};
    auto stale_vector = make_sparse_vector(stale_ids, stale_vals);
    stale.InsertVector(stale_vector, 42);

    MutableSindiTermDataCell source(DEFAULT_TERM_ID_LIMIT,
                                    DEFAULT_WINDOW_SIZE,
                                    allocator.get(),
                                    SparseValueQuantizationType::FP32,
                                    nullptr);
    std::vector<uint32_t> ids = {1};
    std::vector<float> vals = {1.0F};
    auto sparse_vector = make_sparse_vector(ids, vals);
    source.InsertVector(sparse_vector, 0);

    std::stringstream ss;
    IOStreamWriter writer(ss);
    source.SerializeWindows(writer);

    IOStreamReader reader(ss);
    stale.DeserializeWindows(reader, 1);

    REQUIRE(stale.total_count_ == 1);
    REQUIRE(stale.GetWindow(0).term_capacity_ == 2);
    REQUIRE(stale.GetWindow(0).term_ids_.size() == 2);
}

TEST_CASE("MutableSindiTermDataCell Last Term Test", "[ut][MutableSindiTermDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    auto make_sv = [](const std::vector<uint32_t>& ids, const std::vector<float>& vals) {
        vsag::SparseVector sv;
        sv.len_ = static_cast<uint32_t>(ids.size());
        sv.ids_ = const_cast<uint32_t*>(ids.data());
        sv.vals_ = const_cast<float*>(vals.data());
        return sv;
    };

    std::vector<int64_t> ids = {0, 1};

    {
        std::vector<uint32_t> ids0 = {1, 2};
        std::vector<float> vals0 = {0.1f, 0.0f};
        std::vector<uint32_t> ids1 = {1};
        std::vector<float> vals1 = {0.1f};

        auto sv0 = make_sv(ids0, vals0);
        auto sv1 = make_sv(ids1, vals1);

        auto q_params = std::make_shared<QuantizationParams>();
        q_params->min_val = 0.0f;
        q_params->max_val = 0.1f;
        q_params->diff = q_params->max_val - q_params->min_val;
        auto data_cell =
            std::make_shared<MutableSindiTermDataCell>(DEFAULT_TERM_ID_LIMIT,
                                                       DEFAULT_WINDOW_SIZE,
                                                       allocator.get(),
                                                       SparseValueQuantizationType::FP32,
                                                       q_params);
        data_cell->InsertVector(sv0, ids[0]);
        data_cell->InsertVector(sv1, ids[1]);

        std::vector<uint32_t> q_ids = {1, 4};
        std::vector<float> q_vals = {1.0f, 1.0f};
        auto sv_query = make_sv(q_ids, q_vals);

        SINDISearchParameter search_params;
        search_params.term_prune_ratio = 0;
        search_params.query_prune_ratio = 0;
        auto computer =
            std::make_shared<SparseTermComputer>(sv_query, search_params, allocator.get());

        std::vector<float> dists(2, 0);
        QueryFirstWindow(data_cell, dists.data(), computer, allocator.get());
        REQUIRE(std::abs(dists[0] - (-0.1f)) < 1e-2f);
        REQUIRE(std::abs(dists[1] - (-0.1f)) < 1e-2f);
    }
}
