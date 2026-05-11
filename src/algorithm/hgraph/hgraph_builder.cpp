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

#include "hgraph_builder.h"

#include "dataset_impl.h"
#include "hgraph.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "impl/searcher/basic_searcher.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/util_functions.h"

namespace vsag {

void
HGraphBuilder::Train(HGraph& graph, const DatasetPtr& base) {
    int64_t total_elements = base->GetNumElements();
    int64_t dim = base->GetDim();
    DatasetPtr train_data = vsag::sample_train_data(
        base, total_elements, dim, graph.train_sample_count_, graph.allocator_);

    const auto* data_ptr = graph.get_data(train_data);
    graph.basic_flatten_codes_->Train(data_ptr, train_data->GetNumElements());
    if (graph.use_reorder_) {
        graph.high_precise_codes_->Train(data_ptr, train_data->GetNumElements());
    }
    if (graph.create_new_raw_vector_) {
        graph.raw_vector_->Train(data_ptr, train_data->GetNumElements());
    }
}

std::vector<int64_t>
HGraphBuilder::BuildByODescent(HGraph& graph, const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* extra_infos = data->GetExtraInfos();
    Vector<int64_t> valid_indices(graph.allocator_);
    UnorderedSet<LabelType> seen_labels(graph.allocator_);
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (graph.label_table_->CheckLabel(label) or seen_labels.find(label) != seen_labels.end()) {
            failed_ids.emplace_back(label);
            continue;
        }
        seen_labels.insert(label);
        valid_indices.emplace_back(i);
    }
    auto inner_ids = graph.get_unique_inner_ids(static_cast<InnerIdType>(valid_indices.size()));
    auto current_count = graph.total_count_.load();
    uint64_t new_ids_count = 0;
    for (auto inner_id : inner_ids) {
        if (inner_id >= current_count) {
            ++new_ids_count;
        }
    }
    HGraphBuilder::Resize(graph, current_count + new_ids_count);
    graph.total_count_ += new_ids_count;
    Vector<Vector<InnerIdType>> route_graph_ids(graph.allocator_);
    for (InnerIdType cur_size = 0; cur_size < valid_indices.size(); ++cur_size) {
        auto i = valid_indices[cur_size];
        auto label = labels[i];
        InnerIdType inner_id = inner_ids.at(cur_size);
        graph.label_table_->Insert(inner_id, label);
        graph.basic_flatten_codes_->InsertVector(vectors + graph.dim_ * i, inner_id);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->InsertVector(vectors + graph.dim_ * i, inner_id);
        }
        if (graph.create_new_raw_vector_) {
            graph.raw_vector_->InsertVector(vectors + graph.dim_ * i, inner_id);
        }
        auto level = graph.get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(graph.allocator_);
                }
                graph.entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    auto build_data = (graph.use_reorder_ and not graph.build_by_base_)
                          ? graph.high_precise_codes_
                          : graph.basic_flatten_codes_;
    {
        graph.odescent_param_->max_degree = graph.bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            graph.odescent_param_, build_data, graph.allocator_, graph.thread_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(graph.bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        graph.odescent_param_->max_degree = graph.bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            graph.odescent_param_, build_data, graph.allocator_, graph.thread_pool_.get());
        auto graph_ptr = graph.generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph_ptr);
        graph.route_graphs_.emplace_back(graph_ptr);
    }
    return failed_ids;
}

void
HGraphBuilder::AddOnePoint(HGraph& graph, const void* data, int level, InnerIdType inner_id) {
    {
        std::shared_lock add_lock(graph.add_mutex_);
        graph.basic_flatten_codes_->InsertVector(data, inner_id);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->InsertVector(data, inner_id);
        }
        if (graph.create_new_raw_vector_) {
            graph.raw_vector_->InsertVector(data, inner_id);
        }
    }
    std::unique_lock add_lock(graph.add_mutex_);
    if (level >= static_cast<int>(graph.route_graphs_.size()) ||
        graph.bottom_graph_->TotalCount() == 0) {
        std::scoped_lock<std::shared_mutex> wlock(graph.global_mutex_);
        for (auto j = static_cast<int>(graph.route_graphs_.size()); j <= level; ++j) {
            graph.route_graphs_.emplace_back(graph.generate_one_route_graph());
        }
        auto insert_success = HGraphBuilder::GraphAddOne(graph, data, level, inner_id);
        if (insert_success) {
            graph.entry_point_id_ = inner_id;
        } else {
            graph.route_graphs_.pop_back();
        }
        add_lock.unlock();
    } else {
        add_lock.unlock();
        std::shared_lock rlock(graph.global_mutex_);
        HGraphBuilder::GraphAddOne(graph, data, level, inner_id);
    }
}

bool
HGraphBuilder::GraphAddOne(HGraph& graph, const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param;
    param.topk = 1;
    param.ep = graph.entry_point_id_;
    param.ef = 1;
    param.is_inner_id_allowed = nullptr;

    auto flatten_codes = graph.basic_flatten_codes_;
    if (graph.use_reorder_ and not graph.build_by_base_) {
        flatten_codes = graph.high_precise_codes_;
    }

    for (auto j = graph.route_graphs_.size() - 1; j > level; --j) {
        result = graph.search_one_graph(
            data, graph.route_graphs_[j], flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
        param.ep = result->Top().second;
    }

    param.ef = graph.ef_construct_;
    param.topk = static_cast<int64_t>(graph.ef_construct_);
    if (graph.support_duplicate_) {
        param.find_duplicate = true;
    }

    if (graph.bottom_graph_->TotalCount() != 0) {
        result = graph.search_one_graph(
            data, graph.bottom_graph_, flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
        if (graph.support_duplicate_ && param.duplicate_id >= 0) {
            std::unique_lock lock(graph.label_lookup_mutex_);
            graph.bottom_graph_->SetDuplicateId(static_cast<InnerIdType>(param.duplicate_id),
                                                inner_id);
            return false;
        }
        auto filtered_result = std::make_shared<StandardHeap<true, false>>(graph.allocator_, -1);
        while (not result->Empty()) {
            auto [dist, id] = result->Top();
            result->Pop();
            if (id != inner_id) {
                filtered_result->Push(dist, id);
            }
        }
        LockGuard cur_lock(graph.neighbors_mutex_, inner_id);
        mutually_connect_new_element(inner_id,
                                     filtered_result,
                                     graph.bottom_graph_,
                                     flatten_codes,
                                     graph.neighbors_mutex_,
                                     graph.allocator_,
                                     graph.alpha_);
    } else {
        LockGuard cur_lock(graph.neighbors_mutex_, inner_id);
        graph.bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(graph.allocator_));
    }

    for (int64_t j = 0; j <= level; ++j) {
        if (graph.route_graphs_[j]->TotalCount() != 0) {
            result = graph.search_one_graph(data,
                                            graph.route_graphs_[j],
                                            flatten_codes,
                                            param,
                                            (VisitedListPtr) nullptr,
                                            nullptr);
            auto filtered_result =
                std::make_shared<StandardHeap<true, false>>(graph.allocator_, -1);
            while (not result->Empty()) {
                auto [dist, id] = result->Top();
                result->Pop();
                if (id != inner_id) {
                    filtered_result->Push(dist, id);
                }
            }
            LockGuard cur_lock(graph.neighbors_mutex_, inner_id);
            mutually_connect_new_element(inner_id,
                                         filtered_result,
                                         graph.route_graphs_[j],
                                         flatten_codes,
                                         graph.neighbors_mutex_,
                                         graph.allocator_,
                                         graph.alpha_);
        } else {
            LockGuard cur_lock(graph.neighbors_mutex_, inner_id);
            graph.route_graphs_[j]->InsertNeighborsById(inner_id,
                                                        Vector<InnerIdType>(graph.allocator_));
        }
    }
    return true;
}

void
HGraphBuilder::Resize(HGraph& graph, uint64_t new_size) {
    auto cur_size = graph.max_capacity_.load();
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, graph.resize_increase_count_bit_);
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::scoped_lock lock(graph.global_mutex_);
    cur_size = graph.max_capacity_.load();
    if (cur_size < new_size_power_2) {
        graph.neighbors_mutex_->Resize(new_size_power_2);
        graph.pool_ = std::make_shared<VisitedListPool>(
            1, graph.allocator_, new_size_power_2, graph.allocator_);
        graph.label_table_->Resize(new_size_power_2);
        graph.bottom_graph_->Resize(new_size_power_2);
        graph.basic_flatten_codes_->Resize(new_size_power_2);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->Resize(new_size_power_2);
        }
        if (graph.create_new_raw_vector_) {
            graph.raw_vector_->Resize(new_size_power_2);
        }
        if (graph.extra_infos_ != nullptr) {
            graph.extra_infos_->Resize(new_size_power_2);
        }
        graph.max_capacity_.store(new_size_power_2);
        graph.cal_memory_usage();
    }
}

void
HGraphBuilder::ELPOptimize(HGraph& graph) {
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 80;
    param.topk = 10;
    param.is_inner_id_allowed = nullptr;
    graph.searcher_->SetMockParameters(
        graph.bottom_graph_, graph.basic_flatten_codes_, graph.pool_, param, graph.dim_);
    graph.optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    graph.optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    graph.optimizer_->Optimize(graph.searcher_);
}

}  // namespace vsag
