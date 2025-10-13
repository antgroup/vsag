
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
        // base[0] = [0:0, 1:1, 2:2, ..., 9:9] = after_prune = [5:5, 6:6, 7:7, 8:8, 9:9]
        // base[1] = [1:1, 2:2, 3:3, ..., 10:10] = after_prune = [6:6, 7:7, 8:8, 9:9, 10:10]
        // ...
        // base[8] = [8:8, 9:9, 10:10, ..., 17:17] = after_prune = [13:13, 14:14, 15:15, 16:16, 17:17]
        // base[9] = [9:9, 10:10, ..., 18:18] = after_prune = [14:14, 15:15, 16:16, 17:17, 18:18]
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            sparse_vectors[i].ids_[d] = i + d;
            sparse_vectors[i].vals_[d] = i + d;
        }
    }

    // query: [0:1, 1:1, 2:1 .... 18:1]
    // dis(q, b0) = 9 + 8 + 7 + 6 + 5 = 35
    // dis(q, b1) = 10 + 9 + 8 + 7 + 6 = 40
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
    float doc_prune_ratio = 0.5;
    float term_prune_ratio = 0.0;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto data_cell = std::make_shared<SparseTermDataCell>(
        doc_prune_ratio, DEFAULT_TERM_ID_LIMIT, allocator.get());
    REQUIRE(std::abs(data_cell->doc_prune_ratio_ - doc_prune_ratio) < 1e-3);

    // test factory computer
    SINDISearchParameter search_params;
    search_params.term_prune_ratio = term_prune_ratio;
    search_params.query_prune_ratio = query_prune_ratio;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());
    REQUIRE(computer->pruned_len_ == (1.0F - query_prune_ratio) * query_sv.len_);

    // test insert
    auto exp_id_size = 19;
    std::vector<uint32_t> exp_size = {0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1};
    for (auto i = 0; i < count_base; i++) {
        data_cell->InsertVector(sparse_vectors[i], i);
    }
    REQUIRE(data_cell->term_capacity_ == exp_id_size);
    REQUIRE(data_cell->term_ids_.size() == exp_id_size);
    REQUIRE(data_cell->term_datas_.size() == exp_id_size);
    for (auto i = 0; i < data_cell->term_capacity_; i++) {
        REQUIRE(data_cell->term_ids_[i].size() == data_cell->term_sizes_[i]);
        REQUIRE(data_cell->term_ids_[i].size() == exp_size[i]);
        REQUIRE(data_cell->term_datas_[i].size() == exp_size[i]);
    }

    std::vector<float> exp_dists = {35, 40, 45, 50, 55, 60, 65, 70, 75, 80};
    SECTION("test query") {
        std::vector<float> dists(count_base, 0);
        data_cell->Query(dists.data(), computer);
        for (auto i = 0; i < dists.size(); i++) {
            REQUIRE(std::abs(dists[i] + exp_dists[i]) < 1e-3);
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

        data_cell->InsertHeap<KNN_SEARCH, PURE>(dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == topk);
        for (auto i = 0; i < topk; i++) {
            auto cur_top = heap.top();
            auto exp_id = pos + i;
            REQUIRE(cur_top.second == exp_id);
            REQUIRE(std::abs(cur_top.first + exp_dists[exp_id]) < 1e-3);
            heap.pop();
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

        data_cell->InsertHeap<RANGE_SEARCH, PURE>(dists.data(), computer, heap, inner_param, 0);
        REQUIRE(heap.size() == range_topk);
        for (auto i = 0; i < range_topk; i++) {
            auto cur_top = heap.top();
            auto exp_id = pos + i + 1;
            REQUIRE(cur_top.second == exp_id);
            REQUIRE(std::abs(cur_top.first + exp_dists[exp_id]) < 1e-3);
            heap.pop();
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
