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

#include "hgraph.h"  // IWYU pragma: keep
#include "impl/pruning_strategy.h"
#include "utils/util_functions.h"

namespace vsag {

uint32_t
HGraph::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
        return delete_count;
    }

    if (mode == RemoveMode::FORCE_REMOVE) {
        std::unique_lock<std::shared_mutex> wlock(this->force_remove_mutex_);
        for (const auto& id : ids) {
            delete_count += this->force_remove_one(id);
        }
        if (delete_count != 0) {
            this->shrink_to_fit();
        }
        return delete_count;
    }

    throw VsagException(ErrorType::INVALID_ARGUMENT, "RemoveMode not supported");
}

void
HGraph::find_new_entry_point() {
    bool find_new_ep = false;
    auto inner_id = this->entry_point_id_;
    while (not route_graphs_.empty()) {
        auto& upper_graph = route_graphs_.back();
        Vector<InnerIdType> neighbors(allocator_);
        upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
        for (const auto& nb_id : neighbors) {
            if (inner_id == nb_id) {
                continue;
            }
            this->entry_point_id_ = nb_id;
            find_new_ep = true;
            break;
        }
        if (find_new_ep) {
            break;
        }
        route_graphs_.pop_back();
    }
}

void
HGraph::graph_force_remove_one(const InnerIdType& inner_id,
                               const FlattenInterfacePtr& flatten,
                               const GraphInterfacePtr& graph) {
    Vector<InnerIdType> forward_neighbors(allocator_);
    graph->GetNeighbors(inner_id, forward_neighbors);
    Vector<InnerIdType> reverse_neighbors(allocator_);
    graph->GetIncomingNeighbors(inner_id, reverse_neighbors);
    if (forward_neighbors.empty() && reverse_neighbors.empty()) {
        return;
    }

    UnorderedSet<InnerIdType> affected_nodes(allocator_);
    auto current_count = this->total_count_.load();
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

    auto max_degree = graph->MaximumDegree();

    for (const auto& neighbor : affected_nodes) {
        LockGuard lock(neighbors_mutex_, neighbor);

        Vector<InnerIdType> neighbors_of_neighbor(allocator_);
        graph->GetNeighbors(neighbor, neighbors_of_neighbor);

        UnorderedSet<InnerIdType> candidate_set(allocator_);
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

        Vector<InnerIdType> candidate_list(allocator_);
        auto current_count = this->total_count_.load();
        for (const auto& candidate : candidate_set) {
            if (candidate < current_count) {
                candidate_list.emplace_back(candidate);
            }
        }

        select_edges_by_heuristic(
            candidate_list, neighbor, max_degree, flatten, allocator_, alpha_);

        graph->InsertNeighborsById(neighbor, candidate_list);
    }

    Vector<InnerIdType> empty_neighbor(allocator_);
    graph->InsertNeighborsById(inner_id, empty_neighbor);
}

void
HGraph::move_id(InnerIdType from, InnerIdType to) {
    basic_flatten_codes_->Move(from, to);
    if (high_precise_codes_) {
        high_precise_codes_->Move(from, to);
    }

    if (extra_infos_) {
        extra_infos_->Move(from, to);
    }

    bottom_graph_->Move(from, to);
    for (const auto& route_graph : route_graphs_) {
        route_graph->Move(from, to);
    }

    label_table_->Move(from, to);

    if (entry_point_id_ == from) {
        entry_point_id_ = to;
    }
}

uint32_t
HGraph::force_remove_one(int64_t label) {
    InnerIdType inner_id;
    {
        std::shared_lock lock(this->label_lookup_mutex_);
        bool found = false;
        std::tie(found, inner_id) = this->label_table_->TryGetIdByLabel(label, true);
        if (not found) {
            return 0;
        }
    }
    if (inner_id == this->entry_point_id_) {
        this->find_new_entry_point();
    }

    graph_force_remove_one(inner_id, basic_flatten_codes_, bottom_graph_);

    for (const auto& route_graph : route_graphs_) {
        graph_force_remove_one(inner_id, basic_flatten_codes_, route_graph);
    }
    InnerIdType swap_id = this->total_count_.load() - 1;

    bool was_mark_removed = false;
    {
        std::unique_lock lock(this->label_lookup_mutex_);
        was_mark_removed = this->label_table_->IsRemoved(inner_id);
        this->label_table_->ForceRemove(label, inner_id);
        if (swap_id != inner_id) {
            this->move_id(swap_id, inner_id);
        }
    }
    if (was_mark_removed) {
        this->delete_count_.fetch_sub(1);
    }
    this->total_count_.fetch_sub(1);
    return 1;
}

void
HGraph::shrink_to_fit() {
    auto total_count = this->total_count_.load();

    basic_flatten_codes_->ShrinkToFit(total_count);
    if (high_precise_codes_) {
        high_precise_codes_->ShrinkToFit(total_count);
    }
    bottom_graph_->ShrinkToFit(total_count);
    for (const auto& route_graph : route_graphs_) {
        route_graph->ShrinkToFit(total_count);
    }
    label_table_->ShrinkToFit(total_count);
}

void
HGraph::recover_remove(int64_t id) {
    // note:
    // 1. this function doesn't recover entry_point and route_graphs caused by Remove()
    // 2. use this function only when is_tombstone is checked

    std::shared_lock label_lock(this->label_lookup_mutex_);
    auto inner_id = this->label_table_->GetIdByLabel(id, true);
    this->bottom_graph_->RecoverDeleteNeighborsById(inner_id);
    this->label_table_->RecoverRemove(id);
    delete_count_--;
}

DatasetPtr
HGraph::get_single_dataset(const DatasetPtr& data, uint32_t j) {
    void* vectors = nullptr;
    uint64_t data_size = 0;
    get_vectors(data_type_, dim_, data, &vectors, &data_size);
    const auto* labels = data->GetIds();
    auto one_data = Dataset::Make();
    one_data->Ids(labels + j)
        ->Float32Vectors((float*)((char*)vectors + data_size * j))
        ->Int8Vectors((int8_t*)((char*)vectors + data_size * j))
        ->NumElements(1)
        ->Owner(false);
    return one_data;
}

bool
HGraph::try_recover_tombstone(const DatasetPtr& data, std::vector<int64_t>& failed_ids) {
    /*
     * return:
     *      True : No processing required — data already exists or was recovered successfully
     *      False: Processing required — data not found or recovery failed
     *
     *
     * [case 1] fail to insert -> continue + record failed id
     * exist + not delete : is_label_valid = true, is_tombstone = false
     *
     * [case 2] fail to recovery -> add process
     * exist + delete + not recovery: is_label_valid = false, is_tombstone = ture, is_recovered = false
     *
     * [case 3] tombstone recovery -> continue
     * exist + delete + recovery: is_label_valid = false, is_tombstone = ture, is_recovered = true
     *
     * [case 4] no old point -> add process
     * not exists + not delete: is_label_valid = false, is_tombstone = false
     *
     * [case 5] error
     * exists + deleted: is_label_valid = true, is_tombstone = true
     */

    auto label = data->GetIds()[0];

    bool is_label_valid = false;
    bool is_tombstone = false;
    bool is_recovered = false;
    {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        is_label_valid = this->label_table_->CheckLabel(label);
        if (not is_label_valid) {
            is_tombstone = this->label_table_->IsTombstoneLabel(label);
        }
    }

    if (is_tombstone) {
        try {
            // try recover and update
            recover_remove(label);
            auto update_res = UpdateVector(label, data, false);
            if (update_res) {
                // [case 3]
                is_recovered = true;
                return is_recovered;
            }
            // recover failed: roll back
            Remove({label});
        } catch (std::runtime_error& e) {
            // recover failed: roll back
            Remove({label});
        }
    }

    // is_recovered = false
    if (is_label_valid) {
        // [case 1]
        failed_ids.emplace_back(label);
        return true;
    }

    // [case 2, 4]
    return false;
}

void
HGraph::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraph::UpdateAttribute(int64_t id,
                        const AttributeSet& new_attrs,
                        const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

}  // namespace vsag
