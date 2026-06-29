
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

#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>

#include "datacell/flatten_interface.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "utils/filter_search_skip_strategy.h"
#include "vsag/allocator.h"

namespace vsag {

BasicSearcher::BasicSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)) {
}

uint32_t
BasicSearcher::visit(const GraphInterfacePtr& graph,
                     const VisitedListPtr& vl,
                     const std::pair<float, uint64_t>& current_node_pair,
                     const FilterPtr& filter,
                     FilterSearchSkipStrategy* skip_strategy,
                     Vector<InnerIdType>& to_be_visited_id,
                     Vector<InnerIdType>& neighbors) const {
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        SharedLock lock(this->mutex_array_, current_node_pair.second);
        graph->GetNeighbors(current_node_pair.second, neighbors);
    } else {
        graph->GetNeighbors(current_node_pair.second, neighbors);
    }

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_stride_visit_ < neighbors.size()) {
            vl->Prefetch(neighbors[i + prefetch_stride_visit_]);
        }
        if (not vl->Get(neighbors[i])) {
            vl->Set(neighbors[i]);
            // Removed filter->CheckValid() to eliminate duplicate filter checking.
            // Filter is applied at result-collection stage.
            // ShouldVisit() probabilistically gates traversal to preserve graph connectivity.
            if (not filter || count_no_visited == 0 || skip_strategy == nullptr ||
                skip_strategy->ShouldVisit()) {
                to_be_visited_id[count_no_visited] = neighbors[i];
                count_no_visited++;
            }
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
                      const LabelTablePtr& label_table,
                      QueryContext* ctx,
                      DistanceRecordVector* rabitq_lower_bound_candidates) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(graph,
                                             flatten,
                                             vl,
                                             query,
                                             inner_search_param,
                                             label_table,
                                             ctx,
                                             rabitq_lower_bound_candidates);
    }
    return this->search_impl<RANGE_SEARCH>(graph,
                                           flatten,
                                           vl,
                                           query,
                                           inner_search_param,
                                           label_table,
                                           ctx,
                                           rabitq_lower_bound_candidates);
}

DistHeapPtr
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const void* query,
                      const InnerSearchParam& inner_search_param,
                      IteratorFilterContext* iter_ctx,
                      QueryContext* ctx,
                      DistanceRecordVector* rabitq_lower_bound_candidates) const {
    return this->search_impl<KNN_SEARCH>(graph,
                                         flatten,
                                         vl,
                                         query,
                                         inner_search_param,
                                         iter_ctx,
                                         ctx,
                                         rabitq_lower_bound_candidates);
}

template <InnerSearchMode mode>
DistHeapPtr
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const void* query,
                           const InnerSearchParam& inner_search_param,
                           IteratorFilterContext* iter_ctx,
                           QueryContext* ctx,
                           DistanceRecordVector* rabitq_lower_bound_candidates) const {
    // set customize query alloctor
    Allocator* alloc = select_query_allocator(ctx, allocator_);

    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    if (not graph or not flatten) {
        return top_candidates;
    }

    auto computer = flatten->FactoryComputer(query);

    auto is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;

    float dist = 0.0F;
    uint64_t ids_cnt = 1;
    auto lower_bound = std::numeric_limits<float>::max();

    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
    Vector<float> line_dists(graph->MaximumDegree(), alloc);
    Vector<float> lower_bound_dists(graph->MaximumDegree(), alloc);
    auto skip_strategy = create_filter_search_skip_strategy(
        inner_search_param.skip_strategy_type,
        inner_search_param.is_inner_id_allowed != nullptr
            ? inner_search_param.is_inner_id_allowed->ValidRatio()
            : 1.0F,
        inner_search_param.skip_ratio);
    if (rabitq_lower_bound_candidates != nullptr) {
        rabitq_lower_bound_candidates->clear();
    }

    if (!iter_ctx->IsFirstUsed()) {
        if (iter_ctx->Empty()) {
            return top_candidates;
        }
        while (!iter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_ctx->GetTopID();
            float cur_dist = iter_ctx->GetTopDist();
            vl->Set(cur_inner_id);
            if (iter_ctx->CheckPoint(cur_inner_id)) {
                flatten->Query(&cur_dist, computer, &cur_inner_id, 1, ctx);
                // Sign convention: top_candidates stores positive distances (nearest = smallest);
                // candidate_set is a max-heap, so distances are negated (nearest = largest, popped first).
                top_candidates->Push(cur_dist, cur_inner_id);
                candidate_set->Push(-cur_dist, cur_inner_id);
                if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
                    if (cur_dist > inner_search_param.radius and not top_candidates->Empty()) {
                        top_candidates->Pop();
                    }
                }
            }
            iter_ctx->PopDiscard();
        }
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            while (top_candidates->Size() > ef) {
                auto cur_node_pair = top_candidates->Top();
                if (iter_ctx->CheckPoint(cur_node_pair.second)) {
                    iter_ctx->AddDiscardNode(cur_node_pair.first, cur_node_pair.second);
                }
                top_candidates->Pop();
            }
        }
        if (not top_candidates->Empty()) {
            lower_bound = top_candidates->Top().first;
        }
    } else {
        if (inner_search_param.enable_rabitq_one_bit_search) {
            flatten->QueryWithDistanceLowerBound(&dist, nullptr, computer, &ep, 1, ctx);
        } else {
            flatten->Query(&dist, computer, &ep, 1, ctx);
        }
        if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
            top_candidates->Push(dist, ep);
            lower_bound = top_candidates->Top().first;
        }
        candidate_set->Push(-dist, ep);
        vl->Set(ep);
    }

    while (not candidate_set->Empty()) {
        hops++;
        if (hops >= inner_search_param.hops_limit) {
            break;
        }
        auto current_node_pair = candidate_set->Top();

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                if (reasoning != nullptr) {
                    reasoning->SetTermination(ReasoningContext::kTerminationLowerBoundReached);
                }
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
                                 skip_strategy.get(),
                                 to_be_visited_id,
                                 neighbors);

        dist_cmp += count_no_visited;

        bool collect_rabitq_lower_bound = false;
        if (inner_search_param.enable_rabitq_one_bit_search and top_candidates->Size() == ef and
            rabitq_lower_bound_candidates != nullptr) {
            collect_rabitq_lower_bound = true;
            flatten->QueryWithDistanceLowerBound(line_dists.data(),
                                                 lower_bound_dists.data(),
                                                 computer,
                                                 to_be_visited_id.data(),
                                                 count_no_visited,
                                                 ctx);
        } else if (inner_search_param.enable_rabitq_one_bit_search) {
            flatten->QueryWithDistanceLowerBound(line_dists.data(),
                                                 nullptr,
                                                 computer,
                                                 to_be_visited_id.data(),
                                                 count_no_visited,
                                                 ctx);
        } else {
            flatten->Query(
                line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, ctx);
        }

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            const auto cur_id = to_be_visited_id[i];
            const bool id_allowed = not is_id_allowed || is_id_allowed->CheckValid(cur_id);
            if constexpr (mode == KNN_SEARCH) {
                if (collect_rabitq_lower_bound and lower_bound_dists[i] < lower_bound and
                    id_allowed and iter_ctx->CheckPoint(cur_id)) {
                    rabitq_lower_bound_candidates->emplace_back(lower_bound_dists[i], cur_id);
                }
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                if (!iter_ctx->CheckPoint(cur_id)) {
                    continue;
                }
                candidate_set->Push(-dist, cur_id);
                flatten->Prefetch(candidate_set->Top().second);
                if (id_allowed) {
                    top_candidates->Push(dist, cur_id);
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
                           const LabelTablePtr& label_table,
                           QueryContext* ctx,
                           DistanceRecordVector* rabitq_lower_bound_candidates) const {
    // set customize query alloctor
    Allocator* alloc = select_query_allocator(ctx, allocator_);

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
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
    Vector<float> line_dists(graph->MaximumDegree(), alloc);
    Vector<float> lower_bound_dists(graph->MaximumDegree(), alloc);
    auto skip_strategy = create_filter_search_skip_strategy(
        inner_search_param.skip_strategy_type,
        inner_search_param.is_inner_id_allowed != nullptr
            ? inner_search_param.is_inner_id_allowed->ValidRatio()
            : 1.0F,
        inner_search_param.skip_ratio);
    if (rabitq_lower_bound_candidates != nullptr) {
        rabitq_lower_bound_candidates->clear();
    }

    Filter* attr_ft = nullptr;
    if (not inner_search_param.executors.empty() and inner_search_param.executors[0] != nullptr) {
        inner_search_param.executors[0]->Clear();
        attr_ft = inner_search_param.executors[0]->Run();
    }

    auto check_func = [&is_id_allowed, &attr_ft](InnerIdType id) {
        return (is_id_allowed == nullptr or is_id_allowed->CheckValid(id)) and
               (attr_ft == nullptr or attr_ft->CheckValid(id));
    };
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;

    if (inner_search_param.enable_rabitq_one_bit_search) {
        flatten->QueryWithDistanceLowerBound(&dist, nullptr, computer, &ep, 1, ctx);
    } else {
        flatten->Query(&dist, computer, &ep, 1, ctx);
    }
    ++dist_cmp;
    if (check_func(ep)) {
        top_candidates->Push(dist, ep);
        lower_bound = top_candidates->Top().first;
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (dist > inner_search_param.radius and not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    }
    candidate_set->Push(-dist, ep);
    vl->Set(ep);

    while (not candidate_set->Empty()) {
        ++hops;
        if (hops >= inner_search_param.hops_limit) {
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationHopsLimitReached);
            }
            break;
        }
        if (reasoning != nullptr) {
            reasoning->AddSearchHop();
        }
        auto current_node_pair = candidate_set->Top();

        if (inner_search_param.time_cost != nullptr and
            inner_search_param.time_cost->CheckOvertime()) {
            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->is_timeout.store(true, std::memory_order_relaxed);
            }
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningContext::kTerminationTimeout);
            }
            break;
        }

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                if (reasoning != nullptr) {
                    reasoning->SetTermination(ReasoningContext::kTerminationLowerBoundReached);
                }
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
                                 skip_strategy.get(),
                                 to_be_visited_id,
                                 neighbors);

        bool collect_rabitq_lower_bound = false;
        if (inner_search_param.enable_rabitq_one_bit_search and top_candidates->Size() == ef and
            rabitq_lower_bound_candidates != nullptr) {
            collect_rabitq_lower_bound = true;
            flatten->QueryWithDistanceLowerBound(line_dists.data(),
                                                 lower_bound_dists.data(),
                                                 computer,
                                                 to_be_visited_id.data(),
                                                 count_no_visited,
                                                 ctx);
        } else if (inner_search_param.enable_rabitq_one_bit_search) {
            flatten->QueryWithDistanceLowerBound(line_dists.data(),
                                                 nullptr,
                                                 computer,
                                                 to_be_visited_id.data(),
                                                 count_no_visited,
                                                 ctx);
        } else {
            flatten->Query(
                line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, ctx);
        }
        dist_cmp += count_no_visited;

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            const auto cur_id = to_be_visited_id[i];
            if (reasoning != nullptr) {
                reasoning->RecordVisit(cur_id, dist, hops);
            }
            if constexpr (mode == KNN_SEARCH) {
                if (collect_rabitq_lower_bound and lower_bound_dists[i] < lower_bound and
                    check_func(cur_id)) {
                    rabitq_lower_bound_candidates->emplace_back(lower_bound_dists[i], cur_id);
                }
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, cur_id);
                //                flatten->Prefetch(candidate_set->Top().second);
                if (check_func(cur_id)) {
                    top_candidates->Push(dist, cur_id);
                } else if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(cur_id);
                }
                if (inner_search_param.consider_duplicate) {
                    const auto duplicate_ids = graph->GetDuplicateIds(cur_id);
                    for (const auto& item : duplicate_ids) {
                        if (check_func(item)) {
                            top_candidates->Push(dist, item);
                        }
                    }
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        if (reasoning != nullptr) {
                            reasoning->RecordEviction(top_candidates->Top().second, hops);
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

    // set duplicate id for query vector
    if (inner_search_param.find_duplicate and not top_candidates->Empty()) {
        const auto* data = top_candidates->GetData();
        auto min_distance = data[0].first;
        auto min_index = data[0].second;
        for (uint32_t i = 1; i < top_candidates->Size(); ++i) {
            if (data[i].first < min_distance) {
                min_distance = data[i].first;
                min_index = data[i].second;
            }
        }
        if (inner_search_param.duplicate_distance_threshold > 0.0F) {
            if (min_distance <= inner_search_param.duplicate_distance_threshold) {
                inner_search_param.duplicate_id = min_index;
            }
        } else if (inner_search_param.duplicate_query_id < flatten->TotalCount() &&
                   flatten->CompareVectors(inner_search_param.duplicate_query_id, min_index)) {
            inner_search_param.duplicate_id = min_index;
        }
    }

    if (ctx != nullptr and ctx->stats != nullptr) {
        auto& stats = *ctx->stats;
        stats.dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        stats.hops.fetch_add(hops, std::memory_order_relaxed);
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
BasicSearcher::MockRun(SearchStatistics& stats) const {
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
        Search(mock_graph_,
               mock_flatten_,
               vl,
               raw_data.data(),
               mock_inner_search_param_,
               (LabelTablePtr) nullptr,
               nullptr);
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
