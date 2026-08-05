
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
#include <cmath>
#include <cstring>
#include <limits>

#include "datacell/flatten_interface.h"
#include "impl/filter/duplicate_group_filter.h"
#include "impl/filter/iterator_filter.h"
#include "impl/heap/standard_heap.h"
#include "impl/reasoning/search_reasoning.h"
#include "utils/filter_search_skip_strategy.h"
#include "vsag/allocator.h"

namespace vsag {

BasicSearcher::BasicSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)) {
}

BasicSearcher::BasicSearcher(Allocator* allocator, MutexArrayPtr mutex_array)
    : allocator_(allocator), mutex_array_(std::move(mutex_array)) {
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
            if (not filter || count_no_visited == 0 || skip_strategy == nullptr ||
                skip_strategy->ShouldVisit() || filter->CheckValid(neighbors[i])) {
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
                      RaBitQCandidateVector* rabitq_lower_bound_candidates) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(graph,
                                             flatten,
                                             vl,
                                             query,
                                             inner_search_param,
                                             label_table,
                                             ctx,
                                             rabitq_lower_bound_candidates,
                                             nullptr);
    }
    return this->search_impl<RANGE_SEARCH>(graph,
                                           flatten,
                                           vl,
                                           query,
                                           inner_search_param,
                                           label_table,
                                           ctx,
                                           rabitq_lower_bound_candidates,
                                           nullptr);
}

DistHeapPtr
BasicSearcher::SearchWithPresetComputer(const GraphInterfacePtr& graph,
                                        const FlattenInterfacePtr& flatten,
                                        const VisitedListPtr& vl,
                                        const void* query,
                                        const InnerSearchParam& inner_search_param,
                                        const LabelTablePtr& label_table,
                                        QueryContext* ctx,
                                        RaBitQCandidateVector* rabitq_lower_bound_candidates,
                                        const ComputerInterfacePtr& preset_computer) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(graph,
                                             flatten,
                                             vl,
                                             query,
                                             inner_search_param,
                                             label_table,
                                             ctx,
                                             rabitq_lower_bound_candidates,
                                             preset_computer);
    }
    return this->search_impl<RANGE_SEARCH>(graph,
                                           flatten,
                                           vl,
                                           query,
                                           inner_search_param,
                                           label_table,
                                           ctx,
                                           rabitq_lower_bound_candidates,
                                           preset_computer);
}

DistHeapPtr
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const void* query,
                      const InnerSearchParam& inner_search_param,
                      IteratorFilterContext* iter_ctx,
                      QueryContext* ctx,
                      RaBitQCandidateVector* rabitq_lower_bound_candidates) const {
    return this->search_impl<KNN_SEARCH>(graph,
                                         flatten,
                                         vl,
                                         query,
                                         inner_search_param,
                                         iter_ctx,
                                         ctx,
                                         rabitq_lower_bound_candidates);
}

DistHeapPtr
BasicSearcher::Search(const GraphInterfacePtr& graph,
                      const DistanceProviderForGraph& distance_provider,
                      const VisitedListPtr& vl,
                      const InnerSearchParam& inner_search_param,
                      Filter* attr_filter,
                      QueryContext* ctx) const {
    if (inner_search_param.search_mode == InnerSearchMode::RANGE_SEARCH) {
        return this->search_impl<InnerSearchMode::RANGE_SEARCH>(
            graph, distance_provider, vl, inner_search_param, attr_filter, ctx);
    }
    return this->search_impl<InnerSearchMode::KNN_SEARCH>(
        graph, distance_provider, vl, inner_search_param, attr_filter, ctx);
}

template <InnerSearchMode mode>
DistHeapPtr
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const DistanceProviderForGraph& distance_provider,
                           const VisitedListPtr& vl,
                           const InnerSearchParam& inner_search_param,
                           Filter* attr_filter,
                           QueryContext* ctx) const {
    Allocator* alloc = select_query_allocator(ctx, allocator_);
    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    if (not graph or not vl) {
        return top_candidates;
    }
    const auto check_func = [&distance_provider, &inner_search_param, attr_filter](InnerIdType id) {
        const auto original_id = distance_provider.OriginalId(id);
        return distance_provider.IsValid(id) &&
               (inner_search_param.is_inner_id_allowed == nullptr ||
                inner_search_param.is_inner_id_allowed->CheckValid(original_id)) &&
               (attr_filter == nullptr || attr_filter->CheckValid(id));
    };
    const auto ep = inner_search_param.ep;
    const auto ef = inner_search_param.ef;
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;
    float dist = distance_provider.QueryDistance(ep, ctx);
    uint32_t hops = 0;
    uint32_t dist_cmp = 1;
    if (reasoning != nullptr) {
        reasoning->RecordVisit(distance_provider.OriginalId(ep), dist, 0);
    }
    if (check_func(ep)) {
        top_candidates->Push(dist, ep);
    } else if (reasoning != nullptr) {
        reasoning->RecordFilterReject(distance_provider.OriginalId(ep));
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        if (dist > inner_search_param.radius + THRESHOLD_ERROR && not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    }
    candidate_set->Push(-dist, ep);
    vl->Set(ep);
    auto lower_bound =
        top_candidates->Empty() ? std::numeric_limits<float>::max() : top_candidates->Top().first;
    Vector<InnerIdType> to_be_visited_id(graph->MaximumDegree(), alloc);
    Vector<InnerIdType> neighbors(graph->MaximumDegree(), alloc);
    Vector<float> line_dists(graph->MaximumDegree(), alloc);
    while (not candidate_set->Empty()) {
        ++hops;
        if (hops >= inner_search_param.hops_limit) {
            break;
        }
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if (-candidate_set->Top().first > lower_bound && top_candidates->Size() >= ef) {
                break;
            }
        }
        const auto current_node_pair = candidate_set->Top();
        candidate_set->Pop();
        if (not candidate_set->Empty()) {
            graph->Prefetch(candidate_set->Top().second, 0);
        }
        const auto count_no_visited =
            visit(graph, vl, current_node_pair, nullptr, nullptr, to_be_visited_id, neighbors);
        distance_provider.BatchQueryDistance(
            line_dists.data(), to_be_visited_id.data(), count_no_visited, ctx);
        dist_cmp += count_no_visited;
        for (uint32_t i = 0; i < count_no_visited; ++i) {
            const auto id = to_be_visited_id[i];
            dist = line_dists[i];
            if (not distance_provider.IsValid(id)) {
                continue;
            }
            if (reasoning != nullptr) {
                reasoning->RecordVisit(distance_provider.OriginalId(id), dist, hops);
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == InnerSearchMode::RANGE_SEARCH &&
                 dist <= inner_search_param.radius + THRESHOLD_ERROR)) {
                candidate_set->Push(-dist, id);
                distance_provider.Prefetch(candidate_set->Top().second);
                if (check_func(id)) {
                    top_candidates->Push(dist, id);
                } else if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(distance_provider.OriginalId(id));
                }
                if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        if (reasoning != nullptr) {
                            reasoning->RecordEviction(
                                distance_provider.OriginalId(top_candidates->Top().second), hops);
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
    if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        while (top_candidates->Size() > inner_search_param.topk) {
            top_candidates->Pop();
        }
    } else {
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
    if (ctx != nullptr && ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
        ctx->stats->hops.fetch_add(hops, std::memory_order_relaxed);
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
                           IteratorFilterContext* iter_ctx,
                           QueryContext* ctx,
                           RaBitQCandidateVector* rabitq_lower_bound_candidates) const {
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
    Vector<float> filter_inner_products(graph->MaximumDegree(), alloc);
    const auto visit_filter = MakeDuplicateGroupFilter(
        inner_search_param.is_inner_id_allowed, graph, inner_search_param.consider_duplicate);
    auto skip_strategy = create_filter_search_skip_strategy(
        inner_search_param.skip_strategy_type,
        visit_filter != nullptr ? visit_filter->ValidRatio() : 1.0F,
        inner_search_param.skip_ratio);
    if (rabitq_lower_bound_candidates != nullptr) {
        rabitq_lower_bound_candidates->clear();
    }

    UnorderedSet<InnerIdType> expanded_duplicate_groups(alloc);
    auto is_result_allowed = [&is_id_allowed](InnerIdType id) {
        return is_id_allowed == nullptr or is_id_allowed->CheckValid(id);
    };
    auto push_result = [&](InnerIdType id, float distance) {
        if (not iter_ctx->CheckPoint(id) or not is_result_allowed(id)) {
            return false;
        }
        top_candidates->Push(distance, id);
        return true;
    };
    auto push_duplicate_group = [&](InnerIdType id, float distance) {
        if (not inner_search_param.consider_duplicate) {
            return push_result(id, distance);
        }

        const auto group_id = graph->GetGroupId(id);
        if (not expanded_duplicate_groups.insert(group_id).second) {
            return false;
        }

        bool pushed = push_result(group_id, distance);
        for (const auto duplicate_id : graph->GetDuplicateIds(group_id)) {
            pushed = push_result(duplicate_id, distance) or pushed;
        }
        return pushed;
    };
    auto append_lower_bound_group = [&](InnerIdType id, float bound, float filter_ip) {
        if (rabitq_lower_bound_candidates == nullptr) {
            return;
        }
        const auto group_id = inner_search_param.consider_duplicate ? graph->GetGroupId(id) : id;
        const auto append = [&](InnerIdType candidate, float candidate_bound, float candidate_ip) {
            if (iter_ctx->CheckPoint(candidate) and is_result_allowed(candidate)) {
                rabitq_lower_bound_candidates->push_back(
                    {candidate_bound, candidate_ip, candidate});
            }
        };
        append(group_id, bound, filter_ip);
        if (inner_search_param.consider_duplicate) {
            for (const auto duplicate_id : graph->GetDuplicateIds(group_id)) {
                if (not iter_ctx->CheckPoint(duplicate_id) or not is_result_allowed(duplicate_id)) {
                    continue;
                }
                float duplicate_distance = 0.0F;
                float duplicate_bound = std::numeric_limits<float>::max();
                float duplicate_filter_ip = std::numeric_limits<float>::quiet_NaN();
                flatten->QueryWithDistanceLowerBoundAndFilterIP(&duplicate_distance,
                                                                &duplicate_bound,
                                                                &duplicate_filter_ip,
                                                                computer,
                                                                &duplicate_id,
                                                                1,
                                                                ctx);
                append(duplicate_id, duplicate_bound, duplicate_filter_ip);
            }
        }
    };
    auto trim_top_candidates = [&](uint64_t limit) {
        while (top_candidates->Size() > limit) {
            const auto candidate = top_candidates->Top();
            if (iter_ctx->CheckPoint(candidate.second)) {
                iter_ctx->AddDiscardNode(candidate.first, candidate.second);
            }
            top_candidates->Pop();
        }
    };

    if (!iter_ctx->IsFirstUsed()) {
        if (iter_ctx->Empty()) {
            return top_candidates;
        }
        while (!iter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_ctx->GetTopID();
            float cur_dist = iter_ctx->GetTopDist();
            if (iter_ctx->CheckPoint(cur_inner_id)) {
                const auto traversal_id = inner_search_param.consider_duplicate
                                              ? graph->GetGroupId(cur_inner_id)
                                              : cur_inner_id;
                const bool group_needs_expansion =
                    not inner_search_param.consider_duplicate or
                    expanded_duplicate_groups.find(traversal_id) == expanded_duplicate_groups.end();
                if (group_needs_expansion) {
                    flatten->Query(&cur_dist, computer, &traversal_id, 1, ctx);
                    push_duplicate_group(traversal_id, cur_dist);
                }
                // Sign convention: top_candidates stores positive distances (nearest = smallest);
                // candidate_set is a max-heap, so distances are negated (nearest = largest,
                // popped first).
                if (not vl->TestAndSet(traversal_id)) {
                    candidate_set->Push(-cur_dist, traversal_id);
                }
            }
            iter_ctx->PopDiscard();
        }
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            trim_top_candidates(ef);
        }
        if (not top_candidates->Empty()) {
            lower_bound = top_candidates->Top().first;
        }
    } else {
        if (inner_search_param.enable_rabitq_one_bit_search) {
            float entry_lower_bound = std::numeric_limits<float>::max();
            float entry_filter_ip = std::numeric_limits<float>::quiet_NaN();
            flatten->QueryWithDistanceLowerBoundAndFilterIP(
                &dist, &entry_lower_bound, &entry_filter_ip, computer, &ep, 1, ctx);
            append_lower_bound_group(ep, entry_lower_bound, entry_filter_ip);
        } else {
            flatten->Query(&dist, computer, &ep, 1, ctx);
        }
        push_duplicate_group(ep, dist);
        trim_top_candidates(ef);
        if (not top_candidates->Empty()) {
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
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() >= ef) {
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
                                 visit_filter,
                                 skip_strategy.get(),
                                 to_be_visited_id,
                                 neighbors);

        dist_cmp += count_no_visited;

        bool collect_rabitq_lower_bound = false;
        if (inner_search_param.enable_rabitq_one_bit_search and
            rabitq_lower_bound_candidates != nullptr) {
            collect_rabitq_lower_bound = true;
            flatten->QueryWithDistanceLowerBoundAndFilterIP(line_dists.data(),
                                                            lower_bound_dists.data(),
                                                            filter_inner_products.data(),
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
            if constexpr (mode == KNN_SEARCH) {
                if (collect_rabitq_lower_bound and
                    (top_candidates->Size() < ef or lower_bound_dists[i] < lower_bound)) {
                    append_lower_bound_group(
                        cur_id, lower_bound_dists[i], filter_inner_products[i]);
                }
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                const bool source_available = iter_ctx->CheckPoint(cur_id);
                const bool pushed_group = push_duplicate_group(cur_id, dist);
                if (not source_available and not pushed_group) {
                    continue;
                }
                candidate_set->Push(-dist, cur_id);
                flatten->Prefetch(candidate_set->Top().second);

                if constexpr (mode == KNN_SEARCH) {
                    trim_top_candidates(ef);
                }

                if (not top_candidates->Empty()) {
                    lower_bound = top_candidates->Top().first;
                }
            }
        }
    }

    if constexpr (mode == KNN_SEARCH) {
        trim_top_candidates(static_cast<uint64_t>(inner_search_param.topk));
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
                           RaBitQCandidateVector* rabitq_lower_bound_candidates,
                           const ComputerInterfacePtr& preset_computer) const {
    // set customize query alloctor
    Allocator* alloc = select_query_allocator(ctx, allocator_);

    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    const bool use_custom_distance = inner_search_param.distance_batch_func != nullptr;
    if (not graph or (not flatten and not use_custom_distance)) {
        return top_candidates;
    }

    ComputerInterfacePtr computer = nullptr;
    if (not use_custom_distance) {
        computer = preset_computer != nullptr ? preset_computer : flatten->FactoryComputer(query);
    }

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
    const uint64_t custom_batch_capacity =
        use_custom_distance
            ? std::max<uint64_t>(1,
                                 std::min<uint64_t>(inner_search_param.distance_batch_size,
                                                    graph->MaximumDegree()))
            : 0;
    Vector<int64_t> custom_labels(custom_batch_capacity, alloc);
    Vector<float> filter_inner_products(graph->MaximumDegree(), alloc);
    const auto visit_filter = MakeDuplicateGroupFilter(
        inner_search_param.is_inner_id_allowed, graph, inner_search_param.consider_duplicate);
    auto skip_strategy = create_filter_search_skip_strategy(
        inner_search_param.skip_strategy_type,
        visit_filter != nullptr ? visit_filter->ValidRatio() : 1.0F,
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
    auto push_duplicate_candidates = [&](InnerIdType id, float distance) {
        if (not inner_search_param.consider_duplicate) {
            return;
        }
        for (const auto duplicate_id : graph->GetDuplicateIds(id)) {
            if (check_func(duplicate_id)) {
                top_candidates->Push(distance, duplicate_id);
            }
        }
    };
    auto append_lower_bound_candidates = [&](InnerIdType id, float bound, float filter_ip) {
        if (rabitq_lower_bound_candidates == nullptr) {
            return;
        }
        const auto group_id = inner_search_param.consider_duplicate ? graph->GetGroupId(id) : id;
        const auto append = [&](InnerIdType candidate, float candidate_bound, float candidate_ip) {
            if (check_func(candidate)) {
                rabitq_lower_bound_candidates->push_back(
                    {candidate_bound, candidate_ip, candidate});
            }
        };
        append(group_id, bound, filter_ip);
        if (inner_search_param.consider_duplicate) {
            for (const auto duplicate_id : graph->GetDuplicateIds(group_id)) {
                if (not check_func(duplicate_id)) {
                    continue;
                }
                float duplicate_distance = 0.0F;
                float duplicate_bound = std::numeric_limits<float>::max();
                float duplicate_filter_ip = std::numeric_limits<float>::quiet_NaN();
                flatten->QueryWithDistanceLowerBoundAndFilterIP(&duplicate_distance,
                                                                &duplicate_bound,
                                                                &duplicate_filter_ip,
                                                                computer,
                                                                &duplicate_id,
                                                                1,
                                                                ctx);
                append(duplicate_id, duplicate_bound, duplicate_filter_ip);
            }
        }
    };
    auto* reasoning = ctx == nullptr ? nullptr : ctx->reasoning_ctx;

    auto score_ids = [&](const InnerIdType* ids, uint64_t count, float* scores) {
        if (not use_custom_distance) {
            flatten->Query(scores, computer, ids, count, ctx);
            return;
        }
        CHECK_ARGUMENT(label_table != nullptr, "custom distance requires a label table");
        CHECK_ARGUMENT(inner_search_param.distance_batch_size > 0,
                       "distance_batch_size must be greater than 0");
        for (uint64_t offset = 0; offset < count;
             offset += inner_search_param.distance_batch_size) {
            const uint64_t batch_count =
                std::min(inner_search_param.distance_batch_size, count - offset);
            for (uint64_t i = 0; i < batch_count; ++i) {
                custom_labels[i] = label_table->GetLabelById(ids[offset + i]);
            }
            inner_search_param.distance_batch_func(
                custom_labels.data(), batch_count, scores + offset);
            for (uint64_t i = 0; i < batch_count; ++i) {
                CHECK_ARGUMENT(std::isfinite(scores[offset + i]),
                               "distance callback must return finite scores");
            }
        }
    };
    auto score_duplicates = [&](const auto& duplicate_ids, uint32_t duplicate_hops) {
        if (not use_custom_distance) {
            return;
        }

        uint64_t duplicate_count = 0;
        auto submit_duplicates = [&]() {
            if (duplicate_count == 0) {
                return;
            }
            score_ids(neighbors.data(), duplicate_count, lower_bound_dists.data());
            dist_cmp += duplicate_count;
            for (uint64_t i = 0; i < duplicate_count; ++i) {
                if (reasoning != nullptr) {
                    reasoning->RecordVisit(neighbors[i], lower_bound_dists[i], duplicate_hops);
                }
                top_candidates->Push(lower_bound_dists[i], neighbors[i]);
            }
            duplicate_count = 0;
        };

        for (const auto& item : duplicate_ids) {
            if (not check_func(item)) {
                continue;
            }
            neighbors[duplicate_count++] = item;
            if (duplicate_count == neighbors.size()) {
                submit_duplicates();
            }
        }
        submit_duplicates();

        if constexpr (mode == KNN_SEARCH) {
            while (top_candidates->Size() > ef) {
                if (reasoning != nullptr) {
                    reasoning->RecordEviction(top_candidates->Top().second, duplicate_hops);
                }
                top_candidates->Pop();
            }
        }
        if (not top_candidates->Empty()) {
            lower_bound = top_candidates->Top().first;
        }
    };

    if (use_custom_distance) {
        score_ids(&ep, 1, &dist);
    } else if (inner_search_param.enable_rabitq_one_bit_search) {
        float entry_lower_bound = std::numeric_limits<float>::max();
        float entry_filter_ip = std::numeric_limits<float>::quiet_NaN();
        flatten->QueryWithDistanceLowerBoundAndFilterIP(
            &dist, &entry_lower_bound, &entry_filter_ip, computer, &ep, 1, ctx);
        append_lower_bound_candidates(ep, entry_lower_bound, entry_filter_ip);
    } else {
        flatten->Query(&dist, computer, &ep, 1, ctx);
    }
    ++dist_cmp;
    if (check_func(ep)) {
        top_candidates->Push(dist, ep);
    }
    if (not use_custom_distance) {
        push_duplicate_candidates(ep, dist);
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        while (dist > inner_search_param.radius and not top_candidates->Empty()) {
            top_candidates->Pop();
        }
    } else if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
        while (top_candidates->Size() > ef) {
            top_candidates->Pop();
        }
    }
    if (not top_candidates->Empty()) {
        lower_bound = top_candidates->Top().first;
    }
    if (use_custom_distance and inner_search_param.consider_duplicate) {
        const auto duplicate_ids = graph->GetDuplicateIds(ep);
        score_duplicates(duplicate_ids, hops);
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
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() >= ef) {
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
                                 visit_filter,
                                 skip_strategy.get(),
                                 to_be_visited_id,
                                 neighbors);

        bool collect_rabitq_lower_bound = false;
        if (use_custom_distance) {
            score_ids(to_be_visited_id.data(), count_no_visited, line_dists.data());
        } else if (inner_search_param.enable_rabitq_one_bit_search and
                   rabitq_lower_bound_candidates != nullptr) {
            collect_rabitq_lower_bound = true;
            flatten->QueryWithDistanceLowerBoundAndFilterIP(line_dists.data(),
                                                            lower_bound_dists.data(),
                                                            filter_inner_products.data(),
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
            if (use_custom_distance and inner_search_param.consider_duplicate) {
                const auto duplicate_ids = graph->GetDuplicateIds(cur_id);
                score_duplicates(duplicate_ids, hops);
            }
            if constexpr (mode == KNN_SEARCH) {
                if (collect_rabitq_lower_bound and
                    (top_candidates->Size() < ef or lower_bound_dists[i] < lower_bound)) {
                    append_lower_bound_candidates(
                        cur_id, lower_bound_dists[i], filter_inner_products[i]);
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
                if (not use_custom_distance) {
                    push_duplicate_candidates(cur_id, dist);
                }

                if constexpr (mode == KNN_SEARCH) {
                    while (top_candidates->Size() > ef) {
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
    if (not use_custom_distance and inner_search_param.find_duplicate and
        not top_candidates->Empty()) {
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
        } else {
            const bool has_stored_query =
                inner_search_param.duplicate_query_id < flatten->TotalCount();
            const bool is_duplicate =
                has_stored_query
                    ? flatten->CompareVectors(inner_search_param.duplicate_query_id, min_index)
                    : flatten->CompareRawVectorWithId(query, min_index);
            if (is_duplicate) {
                inner_search_param.duplicate_id = min_index;
            }
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
