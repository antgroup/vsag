
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

#pragma once

#include "algorithm/sparse_term_index/sparse_term_index_parameters.h"
#include "io/basic_io.h"
#include "io/memory_block_io.h"
#include "quantization/sparse_term_computer.h"
#include "vsag/dataset.h"

namespace vsag {
class SparseTermDataCell {
public:
    SparseTermDataCell() = default;

    SparseTermDataCell(float query_prune_ratio,
                       float doc_prune_ratio,
                       float term_prune_ratio,
                       Allocator* allocator)
        : doc_prune_ratio_(doc_prune_ratio),
          term_prune_ratio_(term_prune_ratio),
          query_prune_ratio_(query_prune_ratio),
          allocator_(allocator),
          term_ids_(0, Vector<uint32_t>(allocator), allocator),
          term_datas_(0, Vector<float>(allocator), allocator),
          term_sizes_(allocator),
          term_pruned_sizes_(allocator) {
    }

    std::shared_ptr<SparseTermDataCell>
    Clone() {
        std::shared_ptr<SparseTermDataCell> ret =
            std::make_shared<SparseTermDataCell>(this->query_prune_ratio_,
                                                 this->doc_prune_ratio_,
                                                 this->term_prune_ratio_,
                                                 this->allocator_);
        return ret;
    }

    void
    Query(float* global_dists,
          const SparseTermComputerPtr& computer,
          const bool only_collect_id = false) {
        while (computer->HasNextTerm()) {
            auto term = computer->NextTerm();
            computer->ScanForAccumulate(term,
                                        term_ids_[term].data(),
                                        term_datas_[term].data(),
                                        term_pruned_sizes_[term],
                                        global_dists);
        }
        computer->ResetTerm();
    }

    void
    InsertHeapPreFill(float* dists,
                      const SparseTermComputerPtr& computer,
                      MaxHeap& heap,
                      uint32_t n_candidate) {
        uint32_t id = 0;
        while (computer->HasNextTerm()) {
            auto term = computer->NextTerm();

            for (auto i = 0; i < term_pruned_sizes_[term]; i++) {
                id = term_ids_[term][i];
                if (dists[id] >= 0) [[likely]] {
                    continue;
                }
                heap.emplace(dists[id], id);
                if (heap.size() > n_candidate) [[likely]] {
                    heap.pop();
                }
                dists[id] = 0;
            }
        }
        computer->ResetTerm();
    }

    void
    InsertHeapFull(float* dists,
                   const SparseTermComputerPtr& computer,
                   MaxHeap& heap,
                   uint32_t offset_id) {
        uint32_t id = 0;
        float cur_heap_top = heap.top().first;
        while (computer->HasNextTerm()) {
            auto term = computer->NextTerm();

            for (auto i = 0; i < term_pruned_sizes_[term]; i++) {
                id = term_ids_[term][i];

                if (dists[id] >= cur_heap_top) [[likely]] {
                    dists[id] = 0;
                    continue;
                } else {
                    heap.emplace(dists[id], id + offset_id);
                    heap.pop();
                    cur_heap_top = heap.top().first;
                    dists[id] = 0;
                }
            }
        }
        computer->ResetTerm();
    }

    SparseTermComputerPtr
    FactoryComputer(const SparseVector& sparse_query) {
        auto computer = std::make_shared<SparseTermComputer>(query_prune_ratio_, allocator_);
        computer->SetQuery(sparse_query);
        return computer;
    }

    void
    TermPrune() {
        // use this function after all vectors have been inserted
        for (auto i = 0; i < term_sizes_.size(); i++) {
            if (term_sizes_[i] <= 1) {
                continue;
            }

            term_pruned_sizes_[i] = term_sizes_[i] * term_prune_ratio_;
        }
    }

    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) {
        // use this function when inserting
        if (sorted_base.size() <= 1) {
            return;
        }
        uint32_t pruned_doc_len = sorted_base.size() * doc_prune_ratio_;
        sorted_base.resize(pruned_doc_len);
    }

    void
    InsertVector(const SparseVector& sparse_base, uint32_t base_id) {
        // resize term
        uint32_t max_term_id = 0;
        for (auto i = 0; i < sparse_base.len_; i++) {
            auto term_id = sparse_base.ids_[i];
            max_term_id = std::max(max_term_id, term_id);
        }
        ResizeTermList(max_term_id + 1);

        Vector<std::pair<uint32_t, float>> sorted_base(allocator_);
        sort_sparse_vector(sparse_base, sorted_base);

        // doc prune
        DocPrune(sorted_base);

        // insert vector
        for (auto i = 0; i < sorted_base.size(); i++) {
            auto term = sorted_base[i].first;
            auto val = sorted_base[i].second;
            term_ids_[term].push_back(base_id);
            term_datas_[term].push_back(val);
            term_sizes_[term] += 1;
        }
    }

    void
    ResizeTermList(InnerIdType new_term_capacity) {
        if (new_term_capacity <= this->term_capacity_) {
            return;
        }
        this->term_capacity_ = new_term_capacity;
        term_ids_.resize(term_capacity_, Vector<uint32_t>(allocator_));
        term_datas_.resize(term_capacity_, Vector<float>(allocator_));
        term_sizes_.resize(term_capacity_, 0);
        term_pruned_sizes_.resize(term_capacity_, 0);
    }

private:
    float query_prune_ratio_{0};

    float doc_prune_ratio_{0};

    float term_prune_ratio_{0};

    uint32_t term_capacity_{0};

    Vector<Vector<uint32_t>> term_ids_;

    Vector<Vector<float>> term_datas_;

    Vector<uint32_t> term_sizes_;

    Vector<uint32_t> term_pruned_sizes_;

    Allocator* const allocator_{nullptr};
};

using SparseTermDataCellPtr = std::shared_ptr<SparseTermDataCell>;

}  // namespace vsag
