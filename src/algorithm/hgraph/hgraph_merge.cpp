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
#include <mutex>
#include <vector>

#include "hgraph.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "index/index_impl.h"

namespace vsag {

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    // Dispatch to the configured merge strategy.
    //   - "odescent" (default): original rebuild-based strategy that truncates
    //     neighbors and rebuilds the bottom graph with ODescent. Kept as the
    //     default to preserve backward-compatible behavior.
    //   - "two_stage": two-stage forward-search + backward-connect algorithm
    //     (Jin et al., SIGMOD 2026). Much faster and out-of-place friendly;
    //     opt in via the merge_strategy config.
    if (merge_strategy_ == MERGE_STRATEGY_VALUE_ODESCENT) {
        this->merge_with_odescent(merge_units);
    } else {
        this->merge_with_two_stage(merge_units);
    }
}

void
HGraph::merge_with_two_stage(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }

    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());

        if (other_index->total_count_ == 0) {
            continue;
        }

        uint64_t prev_total_count = this->total_count_;
        uint64_t other_count = other_index->total_count_;

        // Merge data: flatten codes, label table, high precise codes
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, prev_total_count);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (has_precise_reorder()) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, prev_total_count);
        }

        // Merge graph structures (copy existing edges with bias)
        bottom_graph_->MergeOther(other_index->bottom_graph_, prev_total_count);
        while (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (uint64_t j = 0; j < std::min(other_index->route_graphs_.size(), route_graphs_.size());
             ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], prev_total_count);
        }
        this->total_count_ += other_count;

        // Handle empty self case
        if (prev_total_count == 0) {
            this->entry_point_id_ = other_index->entry_point_id_ + prev_total_count;
            continue;
        }

        // ====== HNSW-Merger Algorithm (Jin et al., SIGMOD 2026) ======
        //
        // Determine I1 (smaller) and I2 (larger) for forward search direction.
        // After MergeOther, the merged graph has two disconnected components:
        //   - "self" nodes: IDs in [0, prev_total_count)
        //   - "other" nodes: IDs in [prev_total_count, total_count)
        //
        // I1 = smaller index, I2 = larger index.
        // Forward search: each node in I1 searches I2 for top-lambda neighbors.

        uint64_t self_count = prev_total_count;
        bool self_is_smaller = (self_count <= other_count);

        uint64_t n1, n2;
        InnerIdType i1_start, i1_end;
        InnerIdType i2_start, i2_end;
        InnerIdType i2_entry_point;

        if (self_is_smaller) {
            // I1 = self [0, prev_total_count), I2 = other [prev_total_count, total)
            n1 = self_count;
            n2 = other_count;
            i1_start = 0;
            i1_end = static_cast<InnerIdType>(self_count);
            i2_start = static_cast<InnerIdType>(prev_total_count);
            i2_end = static_cast<InnerIdType>(this->total_count_);
            i2_entry_point = other_index->entry_point_id_ + prev_total_count;
        } else {
            // I1 = other [prev_total_count, total), I2 = self [0, prev_total_count)
            n1 = other_count;
            n2 = self_count;
            i1_start = static_cast<InnerIdType>(prev_total_count);
            i1_end = static_cast<InnerIdType>(this->total_count_);
            i2_start = 0;
            i2_end = static_cast<InnerIdType>(self_count);
            i2_entry_point = this->entry_point_id_;
        }

        auto build_data = (has_precise_reorder() and not build_by_base_)
                              ? this->high_precise_codes_
                              : this->basic_flatten_codes_;
        uint64_t max_degree = bottom_graph_->MaximumDegree();
        // lambda = number of cross-index neighbors kept per node. It must be on
        // the order of max_degree for the merged graph to stay well-connected;
        // a small lambda leaves the two sub-indexes weakly linked and recall
        // collapses. merge_lambda_ == 0 means "follow max_degree" (the default).
        uint64_t lambda = (merge_lambda_ == 0) ? max_degree : merge_lambda_;

        // Backward candidate lists: for each node in I2, collect nodes from I1
        // that discovered it during forward search
        std::vector<std::vector<InnerIdType>> backward_candidates(n2);
        std::vector<std::mutex> backward_mutexes(n2);

        // ============ Stage 1: Forward Search ============
        // For each point p in I1, search I2's connected component to find
        // top-lambda nearest neighbors. Since the two subgraphs are disconnected
        // after MergeOther, searching from i2_entry_point naturally stays in I2.
        auto forward_search_task = [&](uint64_t start, uint64_t end) {
            for (uint64_t idx = start; idx < end; ++idx) {
                InnerIdType p_id = static_cast<InnerIdType>(i1_start + idx);

                InnerSearchParam param;
                param.topk = static_cast<int64_t>(lambda);
                // Use the build-time search width (ef_construction) for forward
                // search so cross-index neighbors are found with the same quality
                // as a from-scratch build. lambda only controls how many neighbors
                // are kept, not the search width; ef must stay >= topk.
                param.ef = std::max(this->ef_construct_, lambda);
                param.ep = i2_entry_point;
                param.is_inner_id_allowed = nullptr;

                bool release_p = false;
                const auto* p_data = build_data->GetCodesById(p_id, release_p);

                auto result = this->search_one_graph(p_data,
                                                     this->bottom_graph_,
                                                     build_data,
                                                     param,
                                                     (VisitedListPtr) nullptr,
                                                     nullptr);
                if (release_p) {
                    build_data->Release(p_data);
                }

                // Collect forward search results (cross-index neighbors only)
                Vector<InnerIdType> forward_neighbors(allocator_);
                while (result && !result->Empty()) {
                    InnerIdType neighbor_id = result->Top().second;
                    result->Pop();
                    if (neighbor_id >= i2_start && neighbor_id < i2_end) {
                        forward_neighbors.emplace_back(neighbor_id);
                        // Record backward candidate
                        uint64_t q_local = neighbor_id - i2_start;
                        std::lock_guard<std::mutex> lock(backward_mutexes[q_local]);
                        backward_candidates[q_local].push_back(p_id);
                    }
                }

                // Merge forward neighbors with p's existing neighbors and prune
                Vector<InnerIdType> existing_neighbors(allocator_);
                bottom_graph_->GetNeighbors(p_id, existing_neighbors);

                auto candidates = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                for (auto n_id : forward_neighbors) {
                    float dist = build_data->ComputePairVectors(p_id, n_id);
                    candidates->Push(dist, n_id);
                }
                for (auto n_id : existing_neighbors) {
                    float dist = build_data->ComputePairVectors(p_id, n_id);
                    candidates->Push(dist, n_id);
                }

                select_edges_by_heuristic(candidates, max_degree, build_data, allocator_, alpha_);

                Vector<InnerIdType> new_neighbors(allocator_);
                while (!candidates->Empty()) {
                    new_neighbors.emplace_back(candidates->Top().second);
                    candidates->Pop();
                }
                bottom_graph_->InsertNeighborsById(p_id, new_neighbors);
            }
        };

        // Execute forward search in parallel
        uint64_t thread_count = this->build_thread_count_;
        if (thread_count <= 1 || n1 < thread_count) {
            forward_search_task(0, n1);
        } else {
            uint64_t chunk_size = (n1 + thread_count - 1) / thread_count;
            std::vector<std::future<void>> futures;
            for (uint64_t t = 0; t < thread_count; ++t) {
                uint64_t start = t * chunk_size;
                uint64_t end = std::min(start + chunk_size, n1);
                if (start >= n1) {
                    break;
                }
                futures.push_back(
                    this->thread_pool_->GeneralEnqueue(forward_search_task, start, end));
            }
            for (auto& f : futures) {
                f.get();
            }
        }

        // ============ Stage 2: Backward Direct-Connect ============
        // For each point q in I2, merge backward candidates (nodes from I1 that
        // found q during forward search) with q's existing neighbors, then prune.
        auto backward_connect_task = [&](uint64_t start, uint64_t end) {
            for (uint64_t idx = start; idx < end; ++idx) {
                if (backward_candidates[idx].empty()) {
                    continue;
                }

                InnerIdType q_id = static_cast<InnerIdType>(i2_start + idx);

                Vector<InnerIdType> existing_neighbors(allocator_);
                bottom_graph_->GetNeighbors(q_id, existing_neighbors);

                auto candidates = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                for (auto n_id : backward_candidates[idx]) {
                    float dist = build_data->ComputePairVectors(q_id, n_id);
                    candidates->Push(dist, n_id);
                }
                for (auto n_id : existing_neighbors) {
                    float dist = build_data->ComputePairVectors(q_id, n_id);
                    candidates->Push(dist, n_id);
                }

                select_edges_by_heuristic(candidates, max_degree, build_data, allocator_, alpha_);

                Vector<InnerIdType> new_neighbors(allocator_);
                while (!candidates->Empty()) {
                    new_neighbors.emplace_back(candidates->Top().second);
                    candidates->Pop();
                }
                bottom_graph_->InsertNeighborsById(q_id, new_neighbors);
            }
        };

        if (thread_count <= 1 || n2 < thread_count) {
            backward_connect_task(0, n2);
        } else {
            uint64_t chunk_size = (n2 + thread_count - 1) / thread_count;
            std::vector<std::future<void>> futures;
            for (uint64_t t = 0; t < thread_count; ++t) {
                uint64_t start = t * chunk_size;
                uint64_t end = std::min(start + chunk_size, n2);
                if (start >= n2) {
                    break;
                }
                futures.push_back(
                    this->thread_pool_->GeneralEnqueue(backward_connect_task, start, end));
            }
            for (auto& f : futures) {
                f.get();
            }
        }

        // Update entry point
        if (self_is_smaller) {
            this->entry_point_id_ = i2_entry_point;
        }
    }

    // Rebuild route graphs using ODescent (route graphs are small/sparse)
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }
    auto build_data =
        (has_precise_reorder() and not build_by_base_) ? high_precise_codes_ : basic_flatten_codes_;
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}

void
HGraph::merge_with_odescent(const std::vector<MergeUnit>& merge_units) {
    // ====== Original rebuild-based merge strategy ======
    // Merges sub-index data, truncates every node's neighbor list to half, then
    // rebuilds the bottom graph and route graphs from scratch with ODescent.
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, this->total_count_);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (has_precise_reorder()) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, this->total_count_);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, this->total_count_);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < std::min(other_index->route_graphs_.size(), route_graphs_.size());
             ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], this->total_count_);
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (has_precise_reorder() and not build_by_base_) ? this->high_precise_codes_
                                                                     : this->basic_flatten_codes_;
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        Vector<InnerIdType> neighbors(this->allocator_);
        this->bottom_graph_->GetNeighbors(inner_id, neighbors);
        neighbors.resize(neighbors.size() / 2);
        this->bottom_graph_->InsertNeighborsById(inner_id, neighbors);
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}

}  // namespace vsag
