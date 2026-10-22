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

// Definition of the graph traversal BasicSearcher::search_impl. basic_searcher.cpp instantiates the
// traversals without the distance early stop and basic_searcher_early_stop.cpp the one with it, so
// the early-stop instance does not change how the default traversal is compiled.

#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "basic_searcher.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/standard_heap.h"
#include "impl/query_computer_pool.h"
#include "impl/reasoning/reasoning_context.h"
#include "impl/searcher/distance_early_stop.h"
#include "impl/searcher/searcher_utils.h"
#include "utils/filter_search_skip_strategy.h"
#include "vsag/allocator.h"

namespace vsag {

template <InnerSearchMode mode, bool early_stop>
DistHeapPtr
BasicSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const void* query,
                           const InnerSearchParam& inner_search_param,
                           const LabelTablePtr& label_table,
                           QueryContext* ctx,
                           DistanceRecordVector* rabitq_lower_bound_candidates,
                           const ComputerInterfacePtr& preset_computer) const {
    // set customize query alloctor
    Allocator* alloc = select_query_allocator(ctx, allocator_);

    auto top_candidates = std::make_shared<StandardHeap<true, false>>(alloc, -1);
    auto candidate_set = std::make_shared<StandardHeap<true, false>>(alloc, -1);

    DistanceEarlyStop<early_stop> early_stopper(alloc, inner_search_param, ctx);

    const bool use_custom_distance = inner_search_param.distance_batch_func != nullptr;
    if (not graph or (not flatten and not use_custom_distance)) {
        return top_candidates;
    }

    ComputerLease computer_lease;
    ComputerInterfacePtr computer = preset_computer;
    if (not use_custom_distance and computer == nullptr) {
        computer_lease = AcquireQueryComputer(flatten, query, ctx);
        computer = computer_lease.computer;
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
            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->AddDistance(
                    ctx->distance_phase, DistanceEvaluationBackend::UNKNOWN, batch_count);
            }
            for (uint64_t i = 0; i < batch_count; ++i) {
                CHECK_ARGUMENT(is_finite_distance(scores[offset + i]),
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
                    reasoning->RecordVisit(
                        neighbors[i], lower_bound_dists[i], duplicate_hops);  // [reasoning]
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
                    reasoning->RecordEviction(top_candidates->Top().second,
                                              duplicate_hops);  // [reasoning]
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
        flatten->QueryWithDistanceLowerBound(&dist, nullptr, computer, &ep, 1, ctx);
    } else {
        flatten->Query(&dist, computer, &ep, 1, ctx);
    }
    ++dist_cmp;
    if (check_func(ep) and is_result_distance_eligible<mode>(dist, inner_search_param)) {
        top_candidates->Push(dist, ep);
        early_stopper.Record(dist);
    }
    // The entry point is marked visited before its neighbours are expanded, so its own duplicate
    // labels are never reached by the traversal loop. Expand them here whether the seed distance is
    // finite or not.
    if (inner_search_param.consider_duplicate and not use_custom_distance and
        is_result_distance_eligible<mode>(dist, inner_search_param)) {
        for (const auto duplicate_id : graph->GetDuplicateIds(ep)) {
            if (check_func(duplicate_id)) {
                top_candidates->Push(dist, duplicate_id);
                early_stopper.Record(dist);
            }
        }
        if constexpr (mode == KNN_SEARCH) {
            while (top_candidates->Size() > ef) {
                top_candidates->Pop();
            }
        }
    }
    if constexpr (mode == InnerSearchMode::RANGE_SEARCH) {
        while (dist > inner_search_param.radius and not top_candidates->Empty()) {
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
    candidate_set->Push(traversal_priority(dist), ep);
    vl->Set(ep);

    while (not candidate_set->Empty()) {
        ++hops;
        if (hops >= inner_search_param.hops_limit) {
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningTermination::kHopsLimitReached);  // [reasoning]
            }
            break;
        }
        if (reasoning != nullptr) {
            reasoning->AddSearchHop();  // [reasoning]
        }
        auto current_node_pair = candidate_set->Top();

        if (inner_search_param.time_cost != nullptr and
            inner_search_param.time_cost->CheckOvertime()) {
            if (ctx != nullptr and ctx->stats != nullptr) {
                ctx->stats->is_timeout.store(true, std::memory_order_relaxed);
            }
            if (reasoning != nullptr) {
                reasoning->SetTermination(ReasoningTermination::kTimeout);  // [reasoning]
            }
            break;
        }

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && top_candidates->Size() >= ef) {
                if (reasoning != nullptr) {
                    reasoning->SetTermination(
                        ReasoningTermination::kLowerBoundReached);  // [reasoning]
                }
                break;
            }
        }
        if constexpr (early_stop) {
            // Distance termination on the existing fixed-ef traversal. Candidate
            // admission and the normal termination condition remain unchanged.
            if (early_stopper.ShouldStop(-current_node_pair.first)) {
                if (ctx != nullptr and ctx->stats != nullptr) {
                    ctx->stats->early_stop_count.fetch_add(1, std::memory_order_relaxed);
                }
                if (reasoning != nullptr) {
                    reasoning->SetTermination(
                        ReasoningTermination::kDistanceEarlyStop);  // [reasoning]
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
        if (use_custom_distance) {
            score_ids(to_be_visited_id.data(), count_no_visited, line_dists.data());
        } else if (inner_search_param.enable_rabitq_one_bit_search and
                   top_candidates->Size() == ef and rabitq_lower_bound_candidates != nullptr) {
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
                reasoning->RecordVisit(cur_id, dist, hops);  // [reasoning]
            }
            if (use_custom_distance and inner_search_param.consider_duplicate) {
                const auto duplicate_ids = graph->GetDuplicateIds(cur_id);
                score_duplicates(duplicate_ids, hops);
            }
            if (not is_finite_distance(dist)) {
                candidate_set->Push(traversal_priority(dist), cur_id);
                auto push_result = [&](InnerIdType result_id) {
                    if (not is_result_distance_eligible<mode>(dist, inner_search_param) or
                        not check_func(result_id)) {
                        return;
                    }
                    top_candidates->Push(dist, result_id);
                    if constexpr (mode == KNN_SEARCH) {
                        while (top_candidates->Size() > ef) {
                            if (reasoning != nullptr) {
                                reasoning->RecordEviction(top_candidates->Top().second,
                                                          hops);  // [reasoning]
                            }
                            top_candidates->Pop();
                        }
                    } else if (dist > inner_search_param.radius) {
                        top_candidates->Pop();
                    }
                };
                push_result(cur_id);
                if (inner_search_param.consider_duplicate and not use_custom_distance) {
                    for (const auto duplicate_id : graph->GetDuplicateIds(cur_id)) {
                        push_result(duplicate_id);
                    }
                }
                if (not top_candidates->Empty()) {
                    lower_bound = top_candidates->Top().first;
                }
                continue;
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
                    early_stopper.Record(dist);
                } else if (reasoning != nullptr) {
                    reasoning->RecordFilterReject(cur_id);  // [reasoning]
                }
                if (inner_search_param.consider_duplicate and not use_custom_distance) {
                    const auto duplicate_ids = graph->GetDuplicateIds(cur_id);
                    for (const auto& item : duplicate_ids) {
                        if (check_func(item)) {
                            top_candidates->Push(dist, item);
                            early_stopper.Record(dist);
                        }
                    }
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (top_candidates->Size() > ef) {
                        if (reasoning != nullptr) {
                            reasoning->RecordEviction(top_candidates->Top().second,
                                                      hops);  // [reasoning]
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
            auto matches_query = [&](InnerIdType id) {
                return has_stored_query
                           ? flatten->CompareVectors(inner_search_param.duplicate_query_id, id)
                           : flatten->CompareRawVectorWithId(query, id);
            };
            if (matches_query(min_index)) {
                inner_search_param.duplicate_id = min_index;
            } else {
                // Quantization can rank a different code ahead of an identical one.
                for (uint32_t i = 0; i < top_candidates->Size(); ++i) {
                    if (data[i].second != min_index && matches_query(data[i].second)) {
                        inner_search_param.duplicate_id = data[i].second;
                        break;
                    }
                }
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

}  // namespace vsag
