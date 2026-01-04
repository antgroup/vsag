
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

#include "sparse_term_datacell.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("SparseTermDatacell Basic Test", "[ut][SparseTermDatacell]") {
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
        // base[8] = [8:8, 9:9, 10:10, ..., 17:17] = after_prune = [13:13, 14:14, 15:15, 16:16, 17:17]
        // base[9] = [9:9, 10:10, 11:11, ..., 18:18] = after_prune = [14:14, 15:15, 16:16, 17:17, 18:18]
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
    float doc_retain_ratio = 0.5;
    float term_prune_ratio = 0.0;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    // disable quantization for this basic test
    std::shared_ptr<QuantizationParams> q_params = nullptr;
    auto data_cell = std::make_shared<SparseTermDataCell>(
        doc_retain_ratio, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);
    REQUIRE(std::abs(data_cell->doc_retain_ratio_ - doc_retain_ratio) < 1e-3);

    // test factory computer
    SINDISearchParameter search_params;
    search_params.term_prune_ratio = term_prune_ratio;
    search_params.query_prune_ratio = query_prune_ratio;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());
    REQUIRE(computer->pruned_len_ == (1.0F - query_prune_ratio) * query_sv.len_);

    // test insert
    auto exp_id_size = 19;
    std::vector<uint32_t> exp_size = {0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 4, 4, 4, 5, 5, 4, 3, 2, 1};
    for (auto i = 0; i < count_base; i++) {
        data_cell->InsertVector(sparse_vectors[i], i);
    }
    REQUIRE(data_cell->term_capacity_ == exp_id_size);
    REQUIRE(data_cell->term_ids_.size() == exp_id_size);
    REQUIRE(data_cell->term_datas_.size() == exp_id_size);
    for (auto i = 0; i < data_cell->term_capacity_; i++) {
        if (exp_size[i] == 0) {
            REQUIRE(data_cell->term_ids_[i] == nullptr);
            REQUIRE(data_cell->term_datas_[i] == nullptr);
        } else {
            REQUIRE(data_cell->term_ids_[i]->size() == data_cell->term_sizes_[i]);
            REQUIRE(data_cell->term_ids_[i]->size() == exp_size[i]);
            REQUIRE(data_cell->term_datas_[i]->size() == exp_size[i] * sizeof(float));
        }
    }

    // Calculate expected distances programmatically to match the test logic
    std::vector<float> exp_dists(count_base, 0.0f);
    for (int i = 0; i < count_base; ++i) {
        // 1. Get the original vector and sort it
        const auto& vec = sparse_vectors[i];
        Vector<std::pair<uint32_t, float>> sorted_base(allocator.get());
        sort_sparse_vector(vec, sorted_base);

        // 2. Call the actual DocPrune function
        data_cell->DocPrune(sorted_base);

        // 3. Simulate quantization and inner product calculation
        float total_dist = 0.0f;
        for (const auto& pair : sorted_base) {
            float val = pair.second;
            float query_val = -1.0f;  // The computer uses -1.0 as query value
            total_dist += query_val * val;
        }
        exp_dists[i] = total_dist;
    }

    SECTION("test query") {
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] - exp_dists[i]) < 1e-3);
        }
    }

    SECTION("test insert heap in knn search") {
        auto topk = 5;
        auto pos = count_base - topk;
        InnerSearchParam inner_param;
        inner_param.ef = topk;
        MaxHeap heap(allocator.get());
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);

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
        data_cell->Query(dists2.data(), computer);
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
        data_cell->Query(dists.data(), computer);
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
        data_cell->Query(dists2.data(), computer);
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
    // clean
    for (auto& item : sparse_vectors) {
        delete[] item.vals_;
        delete[] item.ids_;
    }
    delete[] query_sv.ids_;
    delete[] query_sv.vals_;
}

TEST_CASE("SparseTermDatacell Encode/Decode Test", "[ut][SparseTermDatacell]") {
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
    auto data_cell = std::make_shared<SparseTermDataCell>(
        1.0f, DEFAULT_TERM_ID_LIMIT, allocator.get(), true, q_params);

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

TEST_CASE("SparseTermDatacell Last Term Test", "[ut][SparseTermDatacell]") {
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
        auto data_cell = std::make_shared<SparseTermDataCell>(
            1, DEFAULT_TERM_ID_LIMIT, allocator.get(), false, q_params);
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
        data_cell->Query(dists.data(), computer);
        REQUIRE(std::abs(dists[0] - (-0.1f)) < 1e-2f);
        REQUIRE(std::abs(dists[1] - (-0.1f)) < 1e-2f);
    }
}