
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

#include "impl/heap/standard_heap.h"
#include "utils/linear_congruential_generator.h"

namespace vsag {

ParallelSearcher::ParallelSearcher(const IndexCommonParam& common_param, MutexArrayPtr mutex_array, std::shared_ptr<SafeThreadPool> search_pool)
    : allocator_(common_param.allocator_.get()), mutex_array_(std::move(mutex_array)), pool(search_pool) {
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
        for(uint64_t i = 0; i < point_visited_num; i++){
            SharedLock lock(this->mutex_array_, node_pair[i].second);
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
        }
    } else {
        for(uint64_t i = 0; i < point_visited_num; i++)
            graph->GetNeighbors(node_pair[i].second, neighbors[i]);
    }

    float skip_threshold =
        (filter != nullptr
             ? (filter->ValidRatio() == 1.0F ? 0 : (1 - ((1 - filter->ValidRatio()) * skip_ratio)))
             : 0.0F);
    for(uint64_t i = 0; i < point_visited_num; i++){
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
                      const InnerSearchParam& inner_search_param) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param);
    }
    return this->search_impl<RANGE_SEARCH>(
        graph, flatten, vl, query, inner_search_param);
}

DistHeapPtr
ParallelSearcher::Search_plus(const GraphInterfacePtr& graph,
                      const FlattenInterfacePtr& flatten,
                      const VisitedListPtr& vl,
                      const void* query,
                      const InnerSearchParam& inner_search_param) const {
    if (inner_search_param.search_mode == KNN_SEARCH) {
        return this->search_impl_plus<KNN_SEARCH>(
            graph, flatten, vl, query, inner_search_param);
    }
    return this->search_impl_plus<RANGE_SEARCH>(
        graph, flatten, vl, query, inner_search_param);
}

template <InnerSearchMode mode>
DistHeapPtr
ParallelSearcher::search_impl(const GraphInterfacePtr& graph,
                           const FlattenInterfacePtr& flatten,
                           const VisitedListPtr& vl,
                           const void* query,
                           const InnerSearchParam& inner_search_param) const {

    //auto checkpoint1 = std::chrono::high_resolution_clock::now();
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
    uint32_t vector_size = graph->MaximumDegree() * inner_search_param.parallel_search_thread_count_per_query;
    uint32_t current_start = 0;
    Vector<InnerIdType> to_be_visited_rid(vector_size, alloc);
    Vector<InnerIdType> to_be_visited_id(vector_size, alloc);//计算的时候用到
    std::vector<Vector<InnerIdType>> neighbors(inner_search_param.parallel_search_thread_count_per_query, Vector<InnerIdType>(graph->MaximumDegree(), alloc));
    Vector<float> line_dists(vector_size, alloc);
    Vector<std::pair<float, uint64_t>> node_pair(inner_search_param.parallel_search_thread_count_per_query, alloc);
    Vector<uint32_t> tasks_per_thread(inner_search_param.parallel_search_thread_count_per_query, alloc);
    Vector<uint32_t> start_index(inner_search_param.parallel_search_thread_count_per_query, alloc);

    // std::cout << "parallel_search_thread_count_per_query:" << inner_search_param.parallel_search_thread_count_per_query << std::endl;
    // std::cout << "topk:" << inner_search_param.topk << std::endl;

    // std::cout << "数组neighbors.size():" << neighbors[0].size() << std::endl;
    // std::cout << "graph->MaximumDegree():" << graph->MaximumDegree() << std::endl;

    //auto checkpoint2 = std::chrono::high_resolution_clock::now();
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

    //auto checkpoint3 = std::chrono::high_resolution_clock::now();

    // double time1 = 0;
    // double time2 = 0;
    // double time3 = 0;
    // double time4 = 0;
    // double time5 = 0;

    // std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timee(inner_search_param.parallel_search_thread_count_per_query * 2);

    uint64_t min_task = vector_size * 0.2;
    uint64_t min_task_per_thread = static_cast<uint64_t>(std::ceil(graph->MaximumDegree() * 0.2));

    while (not candidate_set->Empty()) {
        //auto checkpoint6 = std::chrono::high_resolution_clock::now();
        hops++;
        auto num_explore_nodes = candidate_set->Size() < inner_search_param.parallel_search_thread_count_per_query ? candidate_set->Size() : inner_search_param.parallel_search_thread_count_per_query;
        //num_threads = 1;
        auto current_first_node_pair = candidate_set->Top();
        node_pair[0] = current_first_node_pair;
 
        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_first_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        for(uint64_t i = 1; i < num_explore_nodes; i++){
            node_pair[i] = candidate_set->Top();
            candidate_set->Pop();
        }

        //auto checkpoint7 = std::chrono::high_resolution_clock::now();

        count_no_visited = visit(graph,
                                 vl,
                                 node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors,
                                 num_explore_nodes);

        //std::cout << "num_threads:" << num_threads << std::endl;
        // std::cout << "count_no_visited:" << count_no_visited << std::endl;
        // if (not candidate_set->Empty()) {
        //     graph->Prefetch(candidate_set->Top().second, 0);
        // }
        //auto checkpoint8 = std::chrono::high_resolution_clock::now();
        //std::sort(to_be_visited_id.begin(), to_be_visited_id.begin() + count_no_visited);
        //auto checkpoint9 = std::chrono::high_resolution_clock::now();
        dist_cmp += count_no_visited;
        uint64_t num_threads = num_explore_nodes;

        if(count_no_visited < min_task){
            //std::cout << "count_no_visited: " << count_no_visited << std::endl;
            num_threads = count_no_visited % min_task_per_thread == 0 ? count_no_visited / min_task_per_thread : count_no_visited / min_task_per_thread + 1;
            //std::cout << "num_threads: " << num_threads << std::endl;
        }
        //num_threads = num_explore_nodes;

        uint32_t base = 0,remainder = 0;

        if(num_threads){
            base = count_no_visited / num_threads;
            remainder = count_no_visited % num_threads;
        }
        
        current_start = 0;
        for (uint64_t i = 0; i < num_threads; ++i) {
            tasks_per_thread[i] = base + (i < remainder ? 1 : 0);
            start_index[i] = current_start;
            current_start += tasks_per_thread[i];
        }

        std::vector<std::future<void>> futures;

        //auto checkpoint1 = std::chrono::high_resolution_clock::now();

        for (uint64_t i = 0; i < num_threads; i++) {
            futures.emplace_back(
                pool->Enqueue([&flatten, &line_dists, &computer, &to_be_visited_id, &tasks_per_thread, &start_index, &alloc, i]() -> void {
                //pool->Enqueue([&flatten, &line_dists, &computer, &to_be_visited_id, &tasks_per_thread, &start_index, &alloc, &timee, i]() -> void {
                    //timee[i * 2] = std::chrono::high_resolution_clock::now();
                    flatten->Query(line_dists.data() + start_index[i], computer, to_be_visited_id.data() + start_index[i], tasks_per_thread[i], alloc);
                    //timee[i * 2 + 1] = std::chrono::high_resolution_clock::now();
                }));
        }
        
        //auto checkpoint2 = std::chrono::high_resolution_clock::now();

        for (auto& f : futures) {
            f.get();
        }

        //auto checkpoint3 = std::chrono::high_resolution_clock::now();

        // time1 = std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint2 - checkpoint1).count();
        // time2 = std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint3 - checkpoint2).count();
        // time3 = std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint3 - checkpoint1).count();


        // std::cout << "任务量:   ";
        // for(uint64_t i = 0; i < num_threads; i++){
        //     std::cout << tasks_per_thread[i] << "    ";
        // }
        // std::cout << std::endl;

        // std::cout << "任务时间: ";
        // for(uint64_t i = 0; i < num_threads; i++){
        //     std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(timee[i * 2 + 1] - timee[i * 2]).count() << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "任务开始时间: ";
        // for(uint64_t i = 0; i < num_threads; i++){
        //     std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(timee[i * 2] - checkpoint1).count() << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "任务结束时间: ";
        // for(uint64_t i = 0; i < num_threads; i++){
        //     std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(timee[i * 2 + 1] - checkpoint1).count() << " ";
        // }
        // std::cout << std::endl;



        // std::cout << "循环: "  << time1 <<"(" << time1/time3 << ")" << std::endl; 
        // std::cout << "get: "  << time2 <<"(" << time2/time3 << ")" << std::endl; 
        // std::cout << "total: "  << time3 <<"(" << time3/time3 << ")" << std::endl; 
        // std::cout << std::endl;


        // time5 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint13 - checkpoint12).count();

        //auto checkpoint10 = std::chrono::high_resolution_clock::now();
        //for (uint64_t i = 0; i < num_threads; i++) {
        //     uint64_t i = 0;
        //     auto function = [&flatten, &line_dists, &computer, &to_be_visited_id, &tasks_per_thread, &start_index, &alloc, &time2, i]() -> void {
        //         auto checkpoint121 = std::chrono::high_resolution_clock::now();
        //         flatten->Query(line_dists.data() + start_index[i], computer, to_be_visited_id.data() + start_index[i], tasks_per_thread[i], alloc);
        //         auto checkpoint122 = std::chrono::high_resolution_clock::now();
        //         time2 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint122 - checkpoint121).count();
                    
        //     };

        //     auto checkpoint11 = std::chrono::high_resolution_clock::now();
        //     //function();
        //     futures.emplace_back(pool->Enqueue(function));
        // //}
        //     auto checkpoint12 = std::chrono::high_resolution_clock::now();
        //     // futures.emplace_back(
        //     //     pool->Enqueue([&flatten, &line_dists, &computer, &to_be_visited_id, &alloc, count_no_visited]() -> void {
        //     //         flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);
        //     //     }));

        // //flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);
        
        // for (auto& f : futures) {
        //     f.get();
        // }
        // auto checkpoint13 = std::chrono::high_resolution_clock::now();
        
        // time1 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint11 - checkpoint10).count();
        // time3 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint12 - checkpoint11).count();
        // time4 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint13 - checkpoint10).count();
        // time5 += std::chrono::duration_cast<std::chrono::nanoseconds>(checkpoint13 - checkpoint12).count();
        
        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, to_be_visited_id[i]);
                //                flatten->Prefetch(candidate_set->Top().second);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }
                // if (inner_search_param.consider_duplicate && label_table &&
                //     label_table->CompressDuplicateData()) {
                //     const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                //     for (const auto& item : duplicate_ids) {
                //         top_candidates->Push(dist, item);
                //     }
                // }

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
        // auto checkpoint13 = std::chrono::high_resolution_clock::now();
        // inner_search_param.latency[4] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint7 - checkpoint6).count();
        // inner_search_param.latency[5] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint8 - checkpoint7).count();
        // inner_search_param.latency[6] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint9 - checkpoint8).count();
        // inner_search_param.latency[7] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint10 - checkpoint9).count();
        // inner_search_param.latency[8] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint11 - checkpoint10).count();
        // inner_search_param.latency[9] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint12 - checkpoint11).count();
        // inner_search_param.latency[10] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint13 - checkpoint12).count();
    }

    //auto checkpoint4 = std::chrono::high_resolution_clock::now();

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
    // std::cout << "lambda define: "  << time1 <<"(" << time1/time4 << ")" << std::endl; 
    // std::cout << "query: "  << time2 <<"(" << time2/time4 << ")" << std::endl; 
    // std::cout << "exec lambda: "  << time3 <<"(" << time3/time4 << ")" << std::endl; 
    // std::cout << "get(): "  << time5 <<"(" << time5/time4 << ")" << std::endl; 
    // std::cout << "total: "  << time4 <<"(" << time4/time4 << ")" << std::endl;
    // std::cout << std::endl;
    // std::cout << hops << std::endl;
    // auto checkpoint5 = std::chrono::high_resolution_clock::now();
    // inner_search_param.latency[0] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint2 - checkpoint1).count();
    // inner_search_param.latency[1] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint3 - checkpoint2).count();
    // inner_search_param.latency[2] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint4 - checkpoint3).count();
    // inner_search_param.latency[3] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint5 - checkpoint4).count();
    return top_candidates;
}

template <InnerSearchMode mode>
DistHeapPtr
ParallelSearcher::search_impl_plus(const GraphInterfacePtr& graph,
                                    const FlattenInterfacePtr& flatten,
                                    const VisitedListPtr& vl,
                                    const void* query,
                                    const InnerSearchParam& inner_search_param) const {

    //auto checkpoint1 = std::chrono::high_resolution_clock::now();
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
    uint32_t vector_size = graph->MaximumDegree() * inner_search_param.parallel_search_thread_count_per_query * 2;
    uint32_t current_start = 0;
    Vector<InnerIdType> to_be_visited_rid(vector_size, alloc);
    Vector<InnerIdType> to_be_visited_id(vector_size, alloc);//计算的时候用到
    std::vector<Vector<InnerIdType>> neighbors(inner_search_param.parallel_search_thread_count_per_query * 2, Vector<InnerIdType>(graph->MaximumDegree(), alloc));
    Vector<float> line_dists(vector_size, alloc);
    Vector<std::pair<float, uint64_t>> node_pair(inner_search_param.parallel_search_thread_count_per_query * 2, alloc);
    Vector<uint32_t> tasks_per_thread(inner_search_param.parallel_search_thread_count_per_query, alloc);
    Vector<uint32_t> start_index(inner_search_param.parallel_search_thread_count_per_query, alloc);

    // std::cout << "parallel_search_thread_count_per_query:" << inner_search_param.parallel_search_thread_count_per_query << std::endl;
    // std::cout << "topk:" << inner_search_param.topk << std::endl;

    // std::cout << "数组neighbors.size():" << neighbors[0].size() << std::endl;
    // std::cout << "graph->MaximumDegree():" << graph->MaximumDegree() << std::endl;

    //auto checkpoint2 = std::chrono::high_resolution_clock::now();
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

    //auto checkpoint3 = std::chrono::high_resolution_clock::now();

    while (not candidate_set->Empty()) {
        //auto checkpoint6 = std::chrono::high_resolution_clock::now();
        hops++;
        uint64_t num_threads = inner_search_param.parallel_search_thread_count_per_query;
        uint64_t will_visit = inner_search_param.parallel_search_thread_count_per_query * 2;
        if(candidate_set->Size() < inner_search_param.parallel_search_thread_count_per_query * 2){
            num_threads = candidate_set->Size() / 2 + 1;
            will_visit = candidate_set->Size();
        }
        
        auto current_first_node_pair = candidate_set->Top();
        node_pair[0] = current_first_node_pair;

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_first_node_pair.first) > lower_bound && top_candidates->Size() == ef) {
                break;
            }
        }
        candidate_set->Pop();

        for(uint64_t i = 1; i < will_visit; i++){
            node_pair[i] = candidate_set->Top();
            candidate_set->Pop();
        }

        //auto checkpoint7 = std::chrono::high_resolution_clock::now();

        count_no_visited = visit(graph,
                                 vl,
                                 node_pair,
                                 inner_search_param.is_inner_id_allowed,
                                 inner_search_param.skip_ratio,
                                 to_be_visited_rid,
                                 to_be_visited_id,
                                 neighbors,
                                 will_visit);

        //std::cout << "num_threads:" << num_threads << std::endl;
        // std::cout << "count_no_visited:" << count_no_visited << std::endl;
        // if (not candidate_set->Empty()) {
        //     graph->Prefetch(candidate_set->Top().second, 0);
        // }
        //auto checkpoint8 = std::chrono::high_resolution_clock::now();
        //std::sort(to_be_visited_id.begin(), to_be_visited_id.begin() + count_no_visited);
        //auto checkpoint9 = std::chrono::high_resolution_clock::now();
        dist_cmp += count_no_visited;

        uint32_t base = count_no_visited / num_threads;
        uint32_t remainder = count_no_visited % num_threads;

        current_start = 0;
        for (uint64_t i = 0; i < num_threads; ++i) {
            tasks_per_thread[i] = base + (i < remainder ? 1 : 0);
            start_index[i] = current_start;
            current_start += tasks_per_thread[i];
        }

        std::vector<std::future<void>> futures;

        //auto checkpoint10 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < num_threads; i++) {
            futures.emplace_back(
                pool->Enqueue([&flatten, &line_dists, &computer, &to_be_visited_id, &tasks_per_thread, &start_index, &alloc, i]() -> void {
                    flatten->Query(line_dists.data() + start_index[i], computer, to_be_visited_id.data() + start_index[i], tasks_per_thread[i], alloc);
                }));
        }

            // futures.emplace_back(
            //     pool->Enqueue([&flatten, &line_dists, &computer, &to_be_visited_id, &alloc, count_no_visited]() -> void {
            //         flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);
            //     }));

        //flatten->Query(line_dists.data(), computer, to_be_visited_id.data(), count_no_visited, alloc);
        //auto checkpoint11 = std::chrono::high_resolution_clock::now();
        for (auto& f : futures) {
            f.get();
        }
        //auto checkpoint12 = std::chrono::high_resolution_clock::now();
        
        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            if (dist < THRESHOLD_ERROR) {
                inner_search_param.duplicate_id = to_be_visited_id[i];
            }
            if (top_candidates->Size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set->Push(-dist, to_be_visited_id[i]);
                //                flatten->Prefetch(candidate_set->Top().second);
                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited_id[i])) {
                    top_candidates->Push(dist, to_be_visited_id[i]);
                }
                // if (inner_search_param.consider_duplicate && label_table &&
                //     label_table->CompressDuplicateData()) {
                //     const auto& duplicate_ids = label_table->GetDuplicateId(to_be_visited_id[i]);
                //     for (const auto& item : duplicate_ids) {
                //         top_candidates->Push(dist, item);
                //     }
                // }

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
        // auto checkpoint13 = std::chrono::high_resolution_clock::now();
        // inner_search_param.latency[4] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint7 - checkpoint6).count();
        // inner_search_param.latency[5] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint8 - checkpoint7).count();
        // inner_search_param.latency[6] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint9 - checkpoint8).count();
        // inner_search_param.latency[7] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint10 - checkpoint9).count();
        // inner_search_param.latency[8] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint11 - checkpoint10).count();
        // inner_search_param.latency[9] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint12 - checkpoint11).count();
        // inner_search_param.latency[10] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint13 - checkpoint12).count();
    }

    //auto checkpoint4 = std::chrono::high_resolution_clock::now();

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
    // auto checkpoint5 = std::chrono::high_resolution_clock::now();
    // inner_search_param.latency[0] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint2 - checkpoint1).count();
    // inner_search_param.latency[1] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint3 - checkpoint2).count();
    // inner_search_param.latency[2] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint4 - checkpoint3).count();
    // inner_search_param.latency[3] += std::chrono::duration_cast<std::chrono::microseconds>(checkpoint5 - checkpoint4).count();
    // inner_search_param.latency[11] += hops;
    return top_candidates;
}

bool
ParallelSearcher::SetRuntimeParameters(const UnorderedMap<std::string, float>& new_params) {
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
ParallelSearcher::SetMockParameters(const GraphInterfacePtr& graph,
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
ParallelSearcher::MockRun() const {
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
ParallelSearcher::SetMutexArray(MutexArrayPtr new_mutex_array) {
    mutex_array_.reset();
    mutex_array_ = std::move(new_mutex_array);
}

}  // namespace vsag
