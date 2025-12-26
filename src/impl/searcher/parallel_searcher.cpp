
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

#include "parallel_searcher.h"

#include <limits>
#include <utility>

#include "impl/heap/standard_heap.h"
#include "utils/linear_congruential_generator.h"
#include "utils/spsc_queue.h"

namespace vsag {

ParallelSearcher::ParallelSearcher(const IndexCommonParam& common_param,
                                   std::shared_ptr<SafeThreadPool> search_pool,
                                   MutexArrayPtr mutex_array)
    : allocator_(common_param.allocator_.get()),
      pool(std::move(search_pool)),
      mutex_array_(std::move(mutex_array)) {
}

uint32_t
ParallelSearcher::visit(const GraphInterfacePtr& graph,
                        const VisitedListPtr& vl,
                        const Vector<std::pair<float, uint64_t>>& node_pair,
                        const FilterPtr& filter,
                        float skip_ratio,
                        Vector<InnerIdType>& to_be_visited_rid,
                        Vector<InnerIdType>& to_be_visited_id,
                        std::vector<Vector<InnerIdType>>& neighbors,
                        uint64_t point_visited_num) const {
    LinearCongruentialGenerator generator;
    uint32_t count_no_visited = 0;

    if (this->mutex_array_ != nullptr) {
        for (uint64_t i = 0; i < point_visited_num; i++) {
            SharedLock lock(this->mutex_array_, node_pair[i].second);
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
        }
    } else {
        for (uint64_t i = 0; i < point_visited_num; i++) {
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
        }
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);
    for (uint64_t i = 0; i < point_visited_num; i++) {
        for (uint32_t j = 0; j < neighbors[i].size(); j++) {
            if (j + prefetch_stride_visit_ < neighbors[i].size()) {
                vl->Prefetch(neighbors[i][j + prefetch_stride_visit_]);
            }
            if (not vl->Get(neighbors[i][j])) {
                if (not filter || count_no_visited == 0 || generator.NextFloat() > skip_threshold ||
                    filter->CheckValid(neighbors[i][j])) {
                    to_be_visited_rid[count_no_visited] = j;
                    to_be_visited_id[count_no_visited] = neighbors[i][j];
                    count_no_visited++;
                }
                vl->Set(neighbors[i][j]);
            }
        }
    }
    return count_no_visited;
}

DistHeapPtr
ParallelSearcher::Search(const GraphInterfacePtr& graph,
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

template <InnerSearchMode mode>
DistHeapPtr
ParallelSearcher::search_impl(const GraphInterfacePtr& graph,
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
    uint64_t beam = 1;
    uint32_t vector_size = graph->MaximumDegree() * beam;
    uint32_t current_start = 0;
    Vector<InnerIdType> to_be_visited_rid(vector_size, alloc);
    Vector<InnerIdType> to_be_visited_id(vector_size, alloc);
    std::vector<Vector<InnerIdType>> neighbors(beam,
                                               Vector<InnerIdType>(graph->MaximumDegree(), alloc));
    Vector<float> line_dists(vector_size, alloc);
    Vector<std::pair<float, uint64_t>> node_pair(beam, alloc);

    flatten->Query(&dist, computer, &ep, 1, alloc);
    if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
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

    auto num_threads = inner_search_param.parallel_search_thread_count - 1;

    std::vector<SPSCQueue<std::tuple<float*, InnerIdType*, uint64_t>, 1024>> queues(num_threads);

    std::atomic<uint32_t> num_points{0};

    auto task = [&](uint64_t thread_id) {
        std::tuple<float*, InnerIdType*, uint64_t> item;
        while (true) {
            if (queues[thread_id].Pop(item)) {
                auto [distances, ids, count] = item;
                if (distances == nullptr) {
                    break;
                }
                flatten->Query(distances, computer, ids, count, alloc);
                num_points.fetch_add(count, std::memory_order_release);
            }
        }
    };

    for (uint64_t i = 0; i < num_threads; i++) {
        pool->GeneralEnqueue(task, i);
    }

    while (not candidate_set->Empty()) {
        hops++;
        auto num_explore_nodes = candidate_set->Size() < beam ? candidate_set->Size() : beam;

        auto current_first_node_pair = candidate_set->Top();
        node_pair[0] = current_first_node_pair;

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_first_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        for (uint64_t i = 1; i < num_explore_nodes; i++) {
            node_pair[i] = candidate_set->Top();
            candidate_set->Pop();
        }

        count_no_visited = visit(graph,
                                 vl,
                                 node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors,
                                 num_explore_nodes);

        dist_cmp += count_no_visited;
        num_points.store(0, std::memory_order_relaxed);
        auto point_per_thread = count_no_visited / (num_threads + 1);
        auto hard_task_count = point_per_thread % (num_threads + 1);

        uint64_t offset = 0;
        for (uint64_t i = 0; i < num_threads; ++i) {
            if (i < hard_task_count) {
                queues[i].Push({line_dists.data() + offset,
                                to_be_visited_id.data() + offset,
                                point_per_thread + 1});
                offset += point_per_thread + 1;
            } else {
                queues[i].Push({line_dists.data() + offset,
                                to_be_visited_id.data() + offset,
                                point_per_thread});
                offset += point_per_thread;
            }
        }

        while (num_points.load(std::memory_order_relaxed) < count_no_visited) {
            if (offset >= count_no_visited) {
                continue;
            }
            const auto remaining_work = count_no_visited - offset;
            flatten->Query(line_dists.data() + offset,
                           computer,
                           to_be_visited_id.data() + offset,
                           remaining_work,
                           alloc);
            num_points.fetch_add(remaining_work, std::memory_order_release);
            offset += remaining_work;
        };

        for (uint64_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, to_be_visited_id[i]);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }
                if (inner_search_param.consider_duplicate && label_table &&
                    label_table->CompressDuplicateData()) {
                    const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                    for (const auto& item : duplicate_ids) {
                        top_candidates->Push(dist, item);
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

    for (uint64_t i = 0; i < num_threads; i++) {
        queues[i].Push({nullptr, nullptr, 0});
    }

    return top_candidates;
}

void
ParallelSearcher::SetMutexArray(MutexArrayPtr new_mutex_array) {
    mutex_array_.reset();
    mutex_array_ = std::move(new_mutex_array);
}

}  // namespace vsag
