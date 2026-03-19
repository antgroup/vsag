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

#include "hgraph_shrink_context.h"

#include <iostream>

#include "impl/heap/standard_heap.h"
#include "impl/pruning_strategy.h"
#include "utils/lock_strategy.h"

namespace vsag {

HGraphShrinkContext::HGraphShrinkContext(HGraph* hgraph) : hgraph_(hgraph) {
}

void
HGraphShrinkContext::Run(double timeout_ms) {
    start_time_ = std::chrono::steady_clock::now();
    bool timeout = false;

    if (state_ == HGraphShrinkState::IDLE) {
        timeout = prepare(timeout_ms);
        if (timeout) {
            return;
        }
        if (check_point_.deleted_nodes_.empty()) {
            state_ = HGraphShrinkState::FINISHED;
            return;
        }
        state_ = HGraphShrinkState::COLLECT_FORWARD_EDGES;
    }
    if (state_ != HGraphShrinkState::IDLE) {
        while (this->check_point_.current_deleted_node_index_ <
               this->check_point_.deleted_nodes_.size()) {
            auto node = check_point_.deleted_nodes_[check_point_.current_deleted_node_index_];
            bool is_level0 = hgraph_->CheckIdLevel0(node);
            if (not is_level0) {
                this->check_point_.current_deleted_node_index_++;
                continue;
            }

            if (state_ == HGraphShrinkState::COLLECT_FORWARD_EDGES) {
                timeout = collect_forward_edges(timeout_ms);
                if (timeout) {
                    return;
                }
                state_ = HGraphShrinkState::COLLECT_REVERSE_EDGES;
            }
            if (state_ == HGraphShrinkState::COLLECT_REVERSE_EDGES) {
                timeout = collect_reverse_edges(timeout_ms);
                if (timeout) {
                    return;
                }
                state_ = HGraphShrinkState::PROCESS_REVERSE_EDGES;
            }
            if (state_ == HGraphShrinkState::PROCESS_REVERSE_EDGES) {
                timeout = process_reverse_edges(timeout_ms);
                if (timeout) {
                    return;
                }
                state_ = HGraphShrinkState::PROCESS_FORWARD_EDGES;
            }
            if (state_ == HGraphShrinkState::PROCESS_FORWARD_EDGES) {
                timeout = process_forward_edges(timeout_ms);
                if (timeout) {
                    return;
                }
                state_ = HGraphShrinkState::CLEANUP;
            }
            if (state_ == HGraphShrinkState::CLEANUP) {
                timeout = cleanup(timeout_ms);
                check_point_.current_forward_edges_.edges_.clear();
                check_point_.current_reversion_edges_.edges_.clear();
                check_point_.current_reversion_edge_index_ = 0;
                check_point_.process_forward_edge_index_ = 0;
                check_point_.process_reversion_edge_index_ = 0;
                if (timeout) {
                    return;
                }
            }
            check_point_.current_deleted_node_index_++;
        }
        state_ = HGraphShrinkState::IDLE;
    }
}

bool
HGraphShrinkContext::prepare(double timeout_ms) {
    check_point_.deleted_nodes_ =
        hgraph_->label_table_->GetDeletedIds(HgraphShrinkCheckPoint::BATCH_SIZE);
    return this->check_timeout(timeout_ms);
}

bool
HGraphShrinkContext::collect_forward_edges(double timeout_ms) {
    auto node = check_point_.deleted_nodes_[check_point_.current_deleted_node_index_];

    Vector<InnerIdType> neighbors(hgraph_->allocator_);
    hgraph_->bottom_graph_->GetNeighbors(node, neighbors);
    for (const auto& neighbor : neighbors) {
        check_point_.current_forward_edges_.edges_.push_back(neighbor);
    }

    return this->check_timeout(timeout_ms);
}

bool
HGraphShrinkContext::collect_reverse_edges(double timeout_ms) {
    auto node = check_point_.deleted_nodes_[check_point_.current_deleted_node_index_];
    auto total_count = hgraph_->bottom_graph_->TotalCount();

    for (; check_point_.current_reversion_edge_index_ < total_count;
         ++check_point_.current_reversion_edge_index_) {
        auto candidate = check_point_.current_reversion_edge_index_;

        Vector<InnerIdType> neighbors(hgraph_->allocator_);
        hgraph_->bottom_graph_->GetNeighbors(candidate, neighbors);

        for (const auto& n : neighbors) {
            if (n == node) {
                check_point_.current_reversion_edges_.edges_.push_back(candidate);
                break;
            }
        }

        if (this->check_timeout(timeout_ms)) {
            return true;
        }
    }

    return false;
}

bool
HGraphShrinkContext::process_reverse_edges(double timeout_ms) {
    auto deleted_node = check_point_.deleted_nodes_[check_point_.current_deleted_node_index_];
    auto flatten = hgraph_->basic_flatten_codes_;
    auto max_degree = hgraph_->bottom_graph_->MaximumDegree();

    for (; check_point_.process_reversion_edge_index_ <
           check_point_.current_reversion_edges_.edges_.size();
         ++check_point_.process_reversion_edge_index_) {
        auto node_b = check_point_.current_reversion_edges_
                          .edges_[check_point_.process_reversion_edge_index_];

        LockGuard lock(hgraph_->neighbors_mutex_, node_b);

        Vector<InnerIdType> neighbors_b(hgraph_->allocator_);
        hgraph_->bottom_graph_->GetNeighbors(node_b, neighbors_b);

        std::unordered_set<InnerIdType> candidate_set;
        for (const auto& nb : neighbors_b) {
            if (nb != deleted_node) {
                candidate_set.insert(nb);
            }
        }
        for (const auto& na : check_point_.current_forward_edges_.edges_) {
            if (na != deleted_node && na != node_b) {
                candidate_set.insert(na);
            }
        }

        auto candidates = std::make_shared<StandardHeap<true, false>>(hgraph_->allocator_, -1);
        for (const auto& candidate : candidate_set) {
            float dist = flatten->ComputePairVectors(node_b, candidate);
            candidates->Push(dist, candidate);
        }

        select_edges_by_heuristic(
            candidates, max_degree, flatten, hgraph_->allocator_, hgraph_->alpha_);

        Vector<InnerIdType> new_neighbors(hgraph_->allocator_);
        while (not candidates->Empty()) {
            new_neighbors.push_back(candidates->Top().second);
            candidates->Pop();
        }

        hgraph_->bottom_graph_->InsertNeighborsById(node_b, new_neighbors);

        if (this->check_timeout(timeout_ms)) {
            return true;
        }
    }

    return false;
}

bool
HGraphShrinkContext::process_forward_edges(double timeout_ms) {
    auto flatten = hgraph_->basic_flatten_codes_;
    auto max_degree = hgraph_->bottom_graph_->MaximumDegree();

    for (; check_point_.process_forward_edge_index_ <
           check_point_.current_forward_edges_.edges_.size();
         ++check_point_.process_forward_edge_index_) {
        auto node =
            check_point_.current_forward_edges_.edges_[check_point_.process_forward_edge_index_];

        LockGuard lock(hgraph_->neighbors_mutex_, node);

        Vector<InnerIdType> neighbors(hgraph_->allocator_);
        hgraph_->bottom_graph_->GetNeighbors(node, neighbors);

        auto candidates = std::make_shared<StandardHeap<true, false>>(hgraph_->allocator_, -1);
        for (const auto& neighbor : neighbors) {
            float dist = flatten->ComputePairVectors(node, neighbor);
            candidates->Push(dist, neighbor);
        }

        select_edges_by_heuristic(
            candidates, max_degree, flatten, hgraph_->allocator_, hgraph_->alpha_);

        Vector<InnerIdType> new_neighbors(hgraph_->allocator_);
        while (not candidates->Empty()) {
            new_neighbors.push_back(candidates->Top().second);
            candidates->Pop();
        }

        hgraph_->bottom_graph_->InsertNeighborsById(node, new_neighbors);

        if (this->check_timeout(timeout_ms)) {
            return true;
        }
    }

    return false;
}

bool
HGraphShrinkContext::cleanup(double timeout_ms) {
    auto deleted_node = check_point_.deleted_nodes_[check_point_.current_deleted_node_index_];

    if (deleted_node == hgraph_->entry_point_id_) {
        bool found_new_ep = false;
        for (auto& graph : hgraph_->route_graphs_) {
            Vector<InnerIdType> neighbors(hgraph_->allocator_);
            graph->GetNeighbors(hgraph_->entry_point_id_, neighbors);
            for (const auto& nb_id : neighbors) {
                if (nb_id != deleted_node && !hgraph_->label_table_->IsRemoved(nb_id)) {
                    hgraph_->entry_point_id_ = nb_id;
                    found_new_ep = true;
                    break;
                }
            }
            if (found_new_ep) {
                break;
            }
        }
        if (!found_new_ep && hgraph_->bottom_graph_->TotalCount() > 0) {
            for (InnerIdType i = 0; i < hgraph_->bottom_graph_->TotalCount(); ++i) {
                if (i != deleted_node && !hgraph_->label_table_->IsRemoved(i)) {
                    hgraph_->entry_point_id_ = i;
                    found_new_ep = true;
                    break;
                }
            }
        }
    }

    hgraph_->label_table_->EraseFromDeletedIds(deleted_node);
    hgraph_->label_table_->PushHole(deleted_node);
    hgraph_->delete_count_--;
    return this->check_timeout(timeout_ms);
}

}  // namespace vsag
