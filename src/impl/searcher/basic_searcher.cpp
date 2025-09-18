
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

#include "impl/heap/standard_heap.h"
#include "utils/linear_congruential_generator.h"

namespace vsag {

BasicSearcher::BasicSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)) {
}

uint32_t
BasicSearcher::visit(const GraphInterfacePtr& graph,
                     const VisitedListPtr& vl,
                     const std::pair<float, uint64_t>& current_node_pair,
                     const FilterPtr& filter,
                     float skip_ratio,
                     Vector<InnerIdType>& to_be_visited_rid,
                     Vector<InnerIdType>& to_be_visited_id,
                     Vector<InnerIdType>& neighbors) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        SharedLock lock(this->mutex_array_, current_node_pair.second);
        graph->GetNeighbors(current_node_pair.second, neighbors);
    } else {
        graph->GetNeighbors(current_node_pair.second, neighbors);
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_stride_visit_ < neighbors.size()) {
            vl->Prefetch(neighbors[i + prefetch_stride_visit_]);
        }
        if (not vl->Get(neighbors[i])) {
            if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                filter->CheckValid(neighbors[i])) {
                to_be_visited_rid[count_no_visited] = i;
                to_be_visited_id[count_no_visited] = neighbors[i];
                count_no_visited++;
            }
            vl->Set(neighbors[i]);
        }
    }
    return count_no_visited;
}

DistHeapPtr
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const void* query,
                      const InnerSearchParam& inner_search_param,
                      const LabelTablePtr& label_table) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param, label_table);
    }
    return this->search_impl<RANGE_SEARCH>(
        graph, flatten, vl, query, inner_search_param, label_table);
}

DistHeapPtr
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const void* query,
                      const InnerSearchParam& inner_search_param,
                      IteratorFilterContext* iter_ctx) const {
    return this->search_impl<KNN_SEARCH>(graph, flatten, vl, query, inner_search_param, iter_ctx);
}

template <InnerSearchMode mode>
DistHeapPtr
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const void* query,
                           const InnerSearchParam& inner_search_param,
                           IteratorFilterContext* iter_ctx) const {
    Allocator* alloc =
        inner_search_param.search_alloc == nullptr ? allocator_ : inner_search_param.search_alloc;
    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    float dist = 0.0F;
    uint64_t ids_cnt = 1;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
    Vector<float> line_dists(graph->MaximumDegree(), alloc);

    if (!iter_ctx->IsFirstUsed()) {
        if (iter_ctx->Empty()) {
            return top_candidates;
        }
        while (!iter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_ctx->GetTopID();
            float cur_dist = iter_ctx->GetTopDist();
            if (!vl->Get(cur_inner_id) && iter_ctx->CheckPoint(cur_inner_id)) {
                vl->Set(cur_inner_id);
                lower_bound = std::max(lower_bound, cur_dist);
                flatten->Query(&cur_dist, computer, &cur_inner_id, 1, alloc);
                top_candidates->Push(cur_dist, cur_inner_id);
                candidate_set->Push(cur_dist, cur_inner_id);
                if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                    if (cur_dist > inner_search_param.radius and not top_candidates->Empty()) {
                        top_candidates->Pop();
                    }
                }
            }
            iter_ctx->PopDiscard();
        }
    } else {
        flatten->Query(&dist, computer, &ep, 1, alloc);
        if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
            top_candidates->Push(dist, ep);
            lower_bound = top_candidates->Top().first;
        }
        candidate_set->Push(-dist, ep);
        vl->Set(ep);
    }

    while (not candidate_set->Empty()) {
        hops++;
        auto current_node_pair = candidate_set->Top();

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        if (not candidate_set->Empty()) {
            graph->Prefetch(candidate_set->Top().second, 0);
        }

        count_no_visited = visit(graph,
                                 vl,
                                 current_node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors);

        dist_cmp += count_no_visited;

        flatten->Query(
            line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                if (!iter_ctx->CheckPoint(to_be_visited_id[i])) {
                    continue;
                }
                candidate_set->Push(-dist, to_be_visited_id[i]);
                flatten->Prefetch(candidate_set->Top().second);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        if (iter_ctx->CheckPoint(top_candidates->Top().second)) {
                            auto cur_node_pair = top_candidates->Top();
                            iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
                        }
                        top_candidates->Pop();
                    }
                }

                if (not top_candidates->Empty()) {
                    lower_bound = top_candidates->Top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates->Size() > inner_search_param.topk) {
            auto cur_node_pair = top_candidates->Top();
            if (iter_ctx->CheckPoint(cur_node_pair.second)) {
                iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
            }
            top_candidates->Pop();
        }
    }

    return top_candidates;
}

template <InnerSearchMode mode>
DistHeapPtr
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const void* query,
                           const InnerSearchParam& inner_search_param,
                           const LabelTablePtr& label_table) const {
    Allocator* alloc =
        inner_search_param.search_alloc == nullptr ? allocator_ : inner_search_param.search_alloc;
    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    float dist = 0.0F;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_rid(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
    Vector<float> line_dists(graph->MaximumDegree(), alloc);

    Filter* attr_ft = nullptr;
    if (not inner_search_param.executors.empty() and inner_search_param.executors[0] != nullptr) {
        inner_search_param.executors[0]->Clear();
        attr_ft = inner_search_param.executors[0]->Run();
    }

    auto check_func = [&is_id_allowed, &attr_ft](InnerIdType id) {
        return (is_id_allowed == nullptr or is_id_allowed->CheckValid(id)) and
               (attr_ft == nullptr or attr_ft->CheckValid(id));
    };

    flatten->Query(&dist, computer, &ep, 1, alloc);
    if (check_func(ep)) {
        top_candidates->Push(dist, ep);
        lower_bound = top_candidates->Top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (dist > inner_search_param.radius and not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    }
    if (dist < THRESHOLD_ERROR) {
        inner_search_param.duplicate_id = ep;
    }
    candidate_set->Push(-dist, ep);
    vl->Set(ep);

    while (not candidate_set->Empty()) {
        hops++;
        auto current_node_pair = candidate_set->Top();

        if (inner_search_param.time_cost != nullptr and
            inner_search_param.time_cost->CheckOvertime()) {
            break;
        }

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        if (not candidate_set->Empty()) {
            graph->Prefetch(candidate_set->Top().second, 0);
        }

        count_no_visited = visit(graph,
                                 vl,
                                 current_node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors);

        dist_cmp += count_no_visited;

        flatten->Query(
            line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, to_be_visited_id[i]);
                //                flatten->Prefetch(candidate_set->Top().second);
                if (check_func(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }
                if (inner_search_param.consider_duplicate and label_table != nullptr and
                    label_table->CompressDuplicateData()) {
                    const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                    for (const auto& item : duplicate_ids) {
                        if (check_func(item)) {
                            top_candidates->Push(dist, item);
                        }
                    }
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        top_candidates->Pop();
                    }
                }

                if (not top_candidates->Empty()) {
                    lower_bound = top_candidates->Top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        while (top_candidates->Size() > inner_search_param.topk) {
            top_candidates->Pop();
        }
    } else if constexpr (mode == RANGE_SEARCH) {
        if (inner_search_param.range_search_limit_size > 0) {
            while (top_candidates->Size() > inner_search_param.range_search_limit_size) {
                top_candidates->Pop();
            }
        }
        while (not top_candidates->Empty() &&
               top_candidates->Top().first > inner_search_param.radius + THRESHOLD_ERROR) {
            top_candidates->Pop();
        }
    }

    return top_candidates;
}

bool
BasicSearcher::SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) {
    bool ret = false;
    auto iter = new_params.find(PREFETCH_STRIDE_VISIT);
    if (iter != new_params.end()) {
        prefetch_stride_visit_ = static_cast<uint32_t>(iter->second);
        ret = true;
    }

    ret |= this->mock_flatten_->SetRuntimeParameters(new_params);
    return ret;
}

void
BasicSearcher::SetMockParameters(const GraphInterfacePtr& graph,
                                 const FlattenInterfacePtr& flatten,
                                 const std::shared_ptr<VisitedListPool>& vl_pool,
                                 const InnerSearchParam& inner_search_param,
                                 const uint64_t dim,
                                 const uint32_t n_trials) {
    mock_graph_ = graph;
    mock_flatten_ = flatten;
    mock_vl_pool_ = vl_pool;
    mock_inner_search_param_ = inner_search_param;
    mock_dim_ = dim;
    mock_n_trials_ = n_trials;
}

double
BasicSearcher::MockRun() const {
    uint64_t n_trials = std::min(mock_n_trials_, mock_flatten_->TotalCount());

    double time_cost = 0;
    for (uint32_t i = 0; i < n_trials; ++i) {
        // init param
        Vector<uint8_t> codes(mock_flatten_->code_size_, allocator_);
        mock_flatten_->GetCodesById(i, codes.data());

        Vector<float> raw_data(mock_dim_, allocator_);
        mock_flatten_->Decode(codes.data(), raw_data.data());
        auto vl = mock_vl_pool_->TakeOne();

        // mock run
        auto st = std::chrono::high_resolution_clock::now();
        Search(mock_graph_, mock_flatten_, vl, raw_data.data(), mock_inner_search_param_);
        auto ed = std::chrono::high_resolution_clock::now();
        time_cost += std::chrono::duration<double>(ed - st).count();

        mock_vl_pool_->ReturnOne(vl);
    }
    return time_cost;
}

void
BasicSearcher::SetMutexArray(MutexArrayPtr new_mutex_array) {
    mutex_array_.reset();
    mutex_array_ = std::move(new_mutex_array);
}

}  // namespace vsag
