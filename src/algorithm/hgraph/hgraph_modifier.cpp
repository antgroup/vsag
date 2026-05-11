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

#include "hgraph_modifier.h"

#include "hgraph.h"
#include "impl/pruning_strategy.h"

namespace vsag {

uint32_t
HGraphModifier::Remove(HGraph& graph, const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(graph.label_lookup_mutex_);
        delete_count = graph.label_table_->MarkRemove(ids);
        graph.delete_count_ += delete_count;
        return delete_count;
    }

    if (mode == RemoveMode::FORCE_REMOVE) {
        std::unique_lock<std::shared_mutex> wlock(graph.force_remove_mutex_);
        for (const auto& id : ids) {
            delete_count += HGraphModifier::ForceRemoveOne(graph, id);
        }
        if (delete_count != 0) {
            HGraphModifier::ShrinkToFit(graph);
        }
        return delete_count;
    }

    throw VsagException(ErrorType::INVALID_ARGUMENT, "RemoveMode not supported");
}

void
HGraphModifier::UpdateAttribute(HGraph& graph, int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = graph.label_table_->GetIdByLabel(id);
    graph.attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraphModifier::UpdateAttribute(HGraph& graph, int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs) {
    auto inner_id = graph.label_table_->GetIdByLabel(id);
    graph.attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

bool
HGraphModifier::UpdateVector(HGraph& graph, int64_t id, const DatasetPtr& new_base, bool force_update) {
    uint32_t inner_id = 0;
    {
        std::shared_lock label_lock(graph.label_lookup_mutex_);
        inner_id = graph.label_table_->GetIdByLabel(id);
    }

    void* new_base_vec = nullptr;
    uint64_t data_size = 0;
    get_vectors(graph.data_type_, graph.dim_, new_base, &new_base_vec, &data_size);

    if (not force_update) {
        std::shared_lock label_lock(graph.label_lookup_mutex_);

        Vector<int8_t> base_data(data_size, graph.allocator_);
        graph.GetVectorByInnerId(inner_id, (float*)base_data.data());
        float old_self_dist = graph.CalcDistanceById((float*)base_data.data(), id);
        float self_dist = graph.CalcDistanceById((float*)new_base_vec, id);
        if (std::abs(old_self_dist - self_dist) < 1e-3) {
            return true;
        }

        Vector<InnerIdType> neighbors(graph.allocator_);
        graph.bottom_graph_->GetNeighbors(inner_id, neighbors);
        for (auto neighbor_inner_id : neighbors) {
            if (neighbor_inner_id == inner_id) {
                continue;
            }

            float neighbor_dist = 0;
            try {
                neighbor_dist = graph.CalcDistanceById(static_cast<float*>(new_base_vec),
                                                       graph.label_table_->GetLabelById(neighbor_inner_id));
            } catch (const std::runtime_error& e) {
                continue;
            }
            if (neighbor_dist < self_dist) {
                return false;
            }
        }
    }

    auto codes = (graph.use_reorder_) ? graph.high_precise_codes_ : graph.basic_flatten_codes_;
    bool update_status = graph.basic_flatten_codes_->UpdateVector(new_base_vec, inner_id);
    if (graph.use_reorder_) {
        update_status = update_status && graph.high_precise_codes_->UpdateVector(new_base_vec, inner_id);
    }
    return update_status;
}

void
HGraphModifier::RecoverRemove(HGraph& graph, int64_t id) {
    std::shared_lock label_lock(graph.label_lookup_mutex_);
    auto inner_id = graph.label_table_->GetIdByLabel(id, true);
    graph.bottom_graph_->RecoverDeleteNeighborsById(inner_id);
    graph.label_table_->RecoverRemove(id);
    graph.delete_count_--;
}

bool
HGraphModifier::TryRecoverTombstone(HGraph& graph, const DatasetPtr& data, std::vector<int64_t>& failed_ids) {
    auto label = data->GetIds()[0];

    bool is_label_valid = false;
    bool is_tombstone = false;
    bool is_recovered = false;
    {
        std::scoped_lock label_lock(graph.label_lookup_mutex_);
        is_label_valid = graph.label_table_->CheckLabel(label);
        if (not is_label_valid) {
            is_tombstone = graph.label_table_->IsTombstoneLabel(label);
        }
    }

    if (is_tombstone) {
        try {
            HGraphModifier::RecoverRemove(graph, label);
            auto update_res = HGraphModifier::UpdateVector(graph, label, data, false);
            if (update_res) {
                is_recovered = true;
                return is_recovered;
            }
            graph.Remove({label});
        } catch (std::runtime_error& e) {
            graph.Remove({label});
        }
    }

    if (is_label_valid) {
        failed_ids.emplace_back(label);
        return true;
    }

    return false;
}

uint32_t
HGraphModifier::ForceRemoveOne(HGraph& graph, int64_t label) {
    InnerIdType inner_id;
    {
        std::shared_lock lock(graph.label_lookup_mutex_);
        inner_id = graph.label_table_->GetIdByLabel(label);
    }
    if (inner_id == graph.entry_point_id_) {
        HGraphModifier::FindNewEntryPoint(graph);
    }

    HGraphModifier::GraphForceRemoveOne(graph, inner_id, graph.basic_flatten_codes_, graph.bottom_graph_);

    for (const auto& route_graph : graph.route_graphs_) {
        HGraphModifier::GraphForceRemoveOne(graph, inner_id, graph.basic_flatten_codes_, route_graph);
    }
    InnerIdType swap_id = graph.total_count_.load() - 1;

    if (swap_id != inner_id) {
        HGraphModifier::MoveId(graph, swap_id, inner_id);
    }
    graph.total_count_--;
    return 1;
}

void
HGraphModifier::FindNewEntryPoint(HGraph& graph) {
    bool find_new_ep = false;
    auto inner_id = graph.entry_point_id_;
    while (not graph.route_graphs_.empty()) {
        auto& upper_graph = graph.route_graphs_.back();
        Vector<InnerIdType> neighbors(graph.allocator_);
        upper_graph->GetNeighbors(graph.entry_point_id_, neighbors);
        for (const auto& nb_id : neighbors) {
            if (inner_id == nb_id) {
                continue;
            }
            graph.entry_point_id_ = nb_id;
            find_new_ep = true;
            break;
        }
        if (find_new_ep) {
            break;
        }
        graph.route_graphs_.pop_back();
    }
}

void
HGraphModifier::GraphForceRemoveOne(HGraph& graph,
                                     const InnerIdType& inner_id,
                                     const FlattenInterfacePtr& flatten,
                                     const GraphInterfacePtr& graph_ptr) {
    Vector<InnerIdType> forward_neighbors(graph.allocator_);
    graph_ptr->GetNeighbors(inner_id, forward_neighbors);
    Vector<InnerIdType> reverse_neighbors(graph.allocator_);
    graph_ptr->GetIncomingNeighbors(inner_id, reverse_neighbors);
    if (forward_neighbors.empty() && reverse_neighbors.empty()) {
        return;
    }

    UnorderedSet<InnerIdType> affected_nodes(graph.allocator_);
    auto current_count = graph.total_count_.load();
    for (const auto& n : forward_neighbors) {
        if (n < current_count) {
            affected_nodes.insert(n);
        }
    }
    for (const auto& n : reverse_neighbors) {
        if (n < current_count) {
            affected_nodes.insert(n);
        }
    }

    auto max_degree = graph_ptr->MaximumDegree();

    for (const auto& neighbor : affected_nodes) {
        LockGuard lock(graph.neighbors_mutex_, neighbor);

        Vector<InnerIdType> neighbors_of_neighbor(graph.allocator_);
        graph_ptr->GetNeighbors(neighbor, neighbors_of_neighbor);

        UnorderedSet<InnerIdType> candidate_set(graph.allocator_);
        for (const auto& nb : neighbors_of_neighbor) {
            if (nb != inner_id) {
                candidate_set.insert(nb);
            }
        }
        for (const auto& nb : forward_neighbors) {
            if (nb != inner_id && nb != neighbor) {
                candidate_set.insert(nb);
            }
        }

        Vector<InnerIdType> candidate_list(graph.allocator_);
        auto current_count = graph.total_count_.load();
        for (const auto& candidate : candidate_set) {
            if (candidate < current_count) {
                candidate_list.emplace_back(candidate);
            }
        }

        select_edges_by_heuristic(candidate_list, neighbor, max_degree, flatten, graph.allocator_, graph.alpha_);

        graph_ptr->InsertNeighborsById(neighbor, candidate_list);
    }

    Vector<InnerIdType> empty_neighbor(graph.allocator_);
    graph_ptr->InsertNeighborsById(inner_id, empty_neighbor);
}

void
HGraphModifier::MoveId(HGraph& graph, InnerIdType from, InnerIdType to) {
    graph.basic_flatten_codes_->Move(from, to);
    if (graph.high_precise_codes_) {
        graph.high_precise_codes_->Move(from, to);
    }

    if (graph.extra_infos_) {
        graph.extra_infos_->Move(from, to);
    }

    graph.bottom_graph_->Move(from, to);
    for (const auto& route_graph : graph.route_graphs_) {
        route_graph->Move(from, to);
    }

    graph.label_table_->Move(from, to);

    if (graph.entry_point_id_ == from) {
        graph.entry_point_id_ = to;
    }
}

void
HGraphModifier::ShrinkToFit(HGraph& graph) {
    auto total_count = graph.total_count_.load();

    graph.basic_flatten_codes_->ShrinkToFit(total_count);
    if (graph.high_precise_codes_) {
        graph.high_precise_codes_->ShrinkToFit(total_count);
    }
    graph.bottom_graph_->ShrinkToFit(total_count);
    for (const auto& route_graph : graph.route_graphs_) {
        route_graph->ShrinkToFit(total_count);
    }
}

}  // namespace vsag
