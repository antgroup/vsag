// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <exception>
#include <future>
#include <memory>
#include <vector>

#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "hgraph.h"
#include "impl/pruning_strategy.h"
#include "impl/thread_pool/safe_thread_pool.h"

namespace vsag {

namespace {

template <typename Function>
void
parallel_for(uint64_t count,
             uint64_t parallelism,
             const SafeThreadPoolPtr& thread_pool,
             const Function& function) {
    const auto worker_count = std::min(count, std::max<uint64_t>(1, parallelism));
    if (worker_count <= 1 || thread_pool == nullptr) {
        for (uint64_t i = 0; i < count; ++i) {
            function(i);
        }
        return;
    }

    const auto batch_size = (count + worker_count - 1) / worker_count;
    std::vector<std::future<void>> futures;
    futures.reserve(worker_count);
    std::exception_ptr first_exception;
    for (uint64_t worker = 0; worker < worker_count; ++worker) {
        const auto begin = worker * batch_size;
        const auto end = std::min(count, begin + batch_size);
        if (begin == end) {
            break;
        }
        try {
            futures.emplace_back(thread_pool->GeneralEnqueue([begin, end, &function]() {
                for (uint64_t i = begin; i < end; ++i) {
                    function(i);
                }
            }));
        } catch (...) {
            first_exception = std::current_exception();
            break;
        }
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (first_exception == nullptr) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

GraphInterfacePtr
materialize_graph(const GraphInterfacePtr& source,
                  const GraphInterfaceParamPtr& target_param,
                  const FlattenInterfacePtr& flatten,
                  Allocator* allocator,
                  bool dense_ids,
                  const SafeThreadPoolPtr& thread_pool,
                  uint64_t parallelism) {
    auto common_param = flatten->ExportCommonParam();
    auto target = GraphInterface::MakeInstance(target_param, common_param);
    CHECK_ARGUMENT(target != nullptr, "failed to create compact graph storage");
    if (dense_ids) {
        target->Resize(source->MaxCapacity());
    }

    const auto copy_neighbors = [&](InnerIdType id) {
        Vector<InnerIdType> neighbors(allocator);
        source->GetNeighbors(id, neighbors);
        const auto target_degree = static_cast<uint64_t>(target_param->max_degree_);
        if (neighbors.size() > target_degree) {
            neighbors.resize(target_degree);
        }
        target->InsertNeighborsById(id, neighbors);
    };

    if (dense_ids) {
        parallel_for(source->TotalCount(), parallelism, thread_pool, [&](uint64_t id) {
            copy_neighbors(static_cast<InnerIdType>(id));
        });
    } else {
        const auto ids = source->GetIds();
        parallel_for(
            ids.size(), parallelism, thread_pool, [&](uint64_t i) { copy_neighbors(ids[i]); });
    }
    target->SetDuplicateTracker(source->GetDuplicateTracker());
    return target;
}

void
rank_graph(const GraphInterfacePtr& graph,
           const FlattenInterfacePtr& flatten,
           Allocator* allocator,
           float alpha,
           bool dense_ids,
           const SafeThreadPoolPtr& thread_pool,
           uint64_t parallelism) {
    const auto rank_neighbors = [&](InnerIdType id) {
        Vector<InnerIdType> neighbors(allocator);
        graph->GetNeighbors(id, neighbors);
        if (!neighbors.empty()) {
            const auto original = neighbors;
            select_edges_by_heuristic(neighbors, id, neighbors.size(), flatten, allocator, alpha);
            std::reverse(neighbors.begin(), neighbors.end());

            std::vector<std::pair<float, InnerIdType>> remaining;
            remaining.reserve(original.size() - neighbors.size());
            for (const auto neighbor : original) {
                if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end()) {
                    remaining.emplace_back(flatten->ComputePairVectors(id, neighbor), neighbor);
                }
            }
            std::sort(remaining.begin(), remaining.end());
            neighbors.reserve(original.size());
            for (const auto& candidate : remaining) {
                neighbors.emplace_back(candidate.second);
            }
            graph->InsertNeighborsById(id, neighbors);
        }
    };

    if (dense_ids) {
        parallel_for(graph->TotalCount(), parallelism, thread_pool, [&](uint64_t id) {
            rank_neighbors(static_cast<InnerIdType>(id));
        });
    } else {
        const auto ids = graph->GetIds();
        parallel_for(
            ids.size(), parallelism, thread_pool, [&](uint64_t i) { rank_neighbors(ids[i]); });
    }
}

}  // namespace

bool
HGraph::can_reduce_max_degree_unlocked() const {
    if (this->immutable_.load(std::memory_order_acquire) ||
        this->graph_type_ != GRAPH_TYPE_VALUE_NSW || this->bottom_graph_ == nullptr ||
        !this->bottom_graph_->InMemory() || this->support_force_remove_) {
        return false;
    }
    const auto hgraph_param = std::dynamic_pointer_cast<HGraphParameter>(this->create_param_ptr_);
    if (hgraph_param == nullptr) {
        return false;
    }
    const auto bottom_param =
        std::dynamic_pointer_cast<GraphDataCellParameter>(hgraph_param->bottom_graph_param);
    if (bottom_param == nullptr || bottom_param->support_remove_ ||
        bottom_param->use_reverse_edges_) {
        return false;
    }
    return this->hierarchical_datacell_param_ != nullptr &&
           !this->hierarchical_datacell_param_->support_delete_ &&
           !this->hierarchical_datacell_param_->use_reverse_edges_ &&
           this->get_degree_reduction_codes() != nullptr;
}

FlattenInterfacePtr
HGraph::get_degree_reduction_codes() const {
    if (this->has_precise_reorder() && !this->build_by_base_) {
        return this->high_precise_codes_;
    }
    if (this->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ) {
        return this->raw_vector_;
    }
    return this->basic_flatten_codes_;
}

bool
HGraph::CanReduceMaxDegree() const {
    std::scoped_lock lock(this->add_mutex_, this->global_mutex_);
    return this->can_reduce_max_degree_unlocked();
}

void
HGraph::PrepareDegreeReduction() {
    std::scoped_lock lock(this->add_mutex_, this->global_mutex_);
    CHECK_ARGUMENT(this->can_reduce_max_degree_unlocked(),
                   "HGraph max-degree reduction does not support this index");
    if (this->degree_reduction_prepared_) {
        return;
    }

    const auto flatten = this->get_degree_reduction_codes();
    rank_graph(this->bottom_graph_,
               flatten,
               this->allocator_,
               this->alpha_,
               true,
               this->thread_pool_,
               this->build_thread_count_);
    for (const auto& route : this->route_graphs_) {
        rank_graph(route,
                   flatten,
                   this->allocator_,
                   this->alpha_,
                   false,
                   this->thread_pool_,
                   this->build_thread_count_);
    }

    this->degree_reduction_prepared_ = true;
}

void
HGraph::ReduceMaxDegree(uint32_t max_degree) {
    std::scoped_lock lock(this->add_mutex_, this->global_mutex_);
    CHECK_ARGUMENT(this->can_reduce_max_degree_unlocked(),
                   "HGraph max-degree reduction does not support this index");
    CHECK_ARGUMENT(this->degree_reduction_prepared_,
                   "PrepareDegreeReduction must be called before ReduceMaxDegree");
    CHECK_ARGUMENT(max_degree >= 4, "target max_degree must be at least 4");
    CHECK_ARGUMENT(max_degree < this->bottom_graph_->MaximumDegree(),
                   "target max_degree must be smaller than the current max_degree");

    const auto hgraph_param = std::dynamic_pointer_cast<HGraphParameter>(this->create_param_ptr_);
    auto bottom_param = std::make_shared<GraphDataCellParameter>(
        *std::dynamic_pointer_cast<GraphDataCellParameter>(hgraph_param->bottom_graph_param));
    bottom_param->max_degree_ = max_degree;
    auto bottom = materialize_graph(this->bottom_graph_,
                                    bottom_param,
                                    this->basic_flatten_codes_,
                                    this->allocator_,
                                    true,
                                    this->thread_pool_,
                                    this->build_thread_count_);

    auto route_param =
        std::make_shared<SparseGraphDatacellParameter>(*this->hierarchical_datacell_param_);
    route_param->max_degree_ = std::max<uint32_t>(1, max_degree / 2);
    Vector<GraphInterfacePtr> routes(this->allocator_);
    routes.reserve(this->route_graphs_.size());
    for (const auto& route : this->route_graphs_) {
        routes.emplace_back(materialize_graph(route,
                                              route_param,
                                              this->basic_flatten_codes_,
                                              this->allocator_,
                                              false,
                                              this->thread_pool_,
                                              this->build_thread_count_));
    }

    this->bottom_graph_ = std::move(bottom);
    this->route_graphs_ = std::move(routes);
    hgraph_param->bottom_graph_param = bottom_param;
    hgraph_param->hierarchical_graph_param = route_param;
    this->hierarchical_datacell_param_ = route_param;
    this->cal_memory_usage();
}

}  // namespace vsag
