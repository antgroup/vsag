
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

#include "basic_searcher.h"

#include <limits>

namespace vsag {

BasicSearcher::BasicSearcher(const IndexCommonParam& common_param) {
    this->allocator_ = common_param.allocator_.get();
}

uint32_t
BasicSearcher::visit(const GraphInterfacePtr& graph_data_cell,
                     const std::shared_ptr<VisitedList>& vl,
                     const std::pair<float, uint64_t>& current_node_pair,
                     Vector<InnerIdType>& to_be_visited_rid,
                     Vector<InnerIdType>& to_be_visited_id) const {
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> neighbors(allocator_);

    graph_data_cell->GetNeighbors(current_node_pair.second, neighbors);

    for (uint32_t i = 0; i < prefetch_jump_visit_size_; i++) {
        vl->Prefetch(neighbors[i]);
    }

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_jump_visit_size_ < neighbors.size()) {
            vl->Prefetch(neighbors[i + prefetch_jump_visit_size_]);
        }
        if (not vl->Get(neighbors[i])) {
            to_be_visited_rid[count_no_visited] = i;
            to_be_visited_id[count_no_visited] = neighbors[i];
            count_no_visited++;
            vl->Set(neighbors[i]);
        }
    }
    return count_no_visited;
}

MaxHeap
BasicSearcher::Search(const GraphInterfacePtr& graph_data_cell,
                      const FlattenInterfacePtr& vector_data_cell,
                      const std::shared_ptr<VisitedList>& vl,
                      const float* query,
                      const InnerSearchParam& inner_search_param) const {
    MaxHeap top_candidates(allocator_);
    MaxHeap candidate_set(allocator_);

    if (not graph_data_cell or not vector_data_cell) {
        return top_candidates;
    }

    auto computer = vector_data_cell->FactoryComputer(query);

    float lower_bound = std::numeric_limits<float>::max();
    float dist;
    uint64_t candidate_id;
    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph_data_cell->MaximumDegree(), allocator_);
    Vector<InnerIdType> to_be_visited_id(graph_data_cell->MaximumDegree(), allocator_);
    Vector<float> line_dists(graph_data_cell->MaximumDegree(), allocator_);

    InnerIdType ep_id = inner_search_param.ep_;
    vector_data_cell->Query(&dist, computer, &ep_id, 1);
    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(-dist, ep_id);
    vl->Set(ep_id);

    while (!candidate_set.empty()) {
        hops++;
        std::pair<float, uint64_t> current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lower_bound &&
            (top_candidates.size() == inner_search_param.ef_)) {
            break;
        }
        candidate_set.pop();
        if (not candidate_set.empty()) {
            graph_data_cell->Prefetch(candidate_set.top().second, 0);
        }

        count_no_visited =
            visit(graph_data_cell, vl, current_node_pair, to_be_visited_rid, to_be_visited_id);

        dist_cmp += count_no_visited;

        vector_data_cell->Query(
            line_dists.data(), computer, to_be_visited_id.data(), count_no_visited);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            candidate_id = to_be_visited_id[i];
            if (top_candidates.size() < inner_search_param.ef_ || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                top_candidates.emplace(dist, candidate_id);

                if (top_candidates.size() > inner_search_param.ef_) {
                    top_candidates.pop();
                }

                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    if (inner_search_param.topk_ > 0) {
        while (top_candidates.size() > inner_search_param.topk_) {
            top_candidates.pop();
        }
    } else if (inner_search_param.radius_ > 0) {
        while (top_candidates.top().first > inner_search_param.radius_ + 2e-6) {
            top_candidates.pop();
        }
    }

    return top_candidates;
}

}  // namespace vsag
