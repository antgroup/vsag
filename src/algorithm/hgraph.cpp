
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

#include "hgraph.h"

#include <fmt/format-inl.h>

#include <memory>
#include <stdexcept>

#include "common.h"
#include "data_cell/sparse_graph_datacell.h"
#include "empty_index_binary_set.h"
#include "impl/pruning_strategy.h"
#include "index/hgraph_index_zparameters.h"
#include "logger.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"

namespace vsag {

static uint64_t
next_multiple_of_power_of_two(uint64_t x, uint64_t n) {
    if (n > 63) {
        throw std::runtime_error(fmt::format("n is larger than 63, n is {}", n));
    }
    uint64_t y = 1 << n;
    auto result = (x + y - 1) & ~(y - 1);
    return result;
}

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      dim_(common_param.dim_),
      metric_(common_param.metric_),
      route_graphs_(common_param.allocator_.get()),
      use_reorder_(hgraph_param->use_reorder),
      ef_construct_(hgraph_param->ef_construction),
      build_thread_count_(hgraph_param->build_thread_count) {
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (use_reorder_) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));
    resize(bottom_graph_->max_capacity_);
    if (this->build_thread_count_ > 1) {
        this->build_pool_ = std::make_unique<progschj::ThreadPool>(this->build_thread_count_);
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    this->basic_flatten_codes_->EnableForceInMemory();
    if (use_reorder_) {
        this->high_precise_codes_->EnableForceInMemory();
    }
    auto ret = this->Add(data);
    this->basic_flatten_codes_->DisableForceInMemory();
    if (use_reorder_) {
        this->high_precise_codes_->DisableForceInMemory();
    }
    return ret;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto base_dim = data->GetDim();
    CHECK_ARGUMENT(base_dim == dim_,
                   fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    CHECK_ARGUMENT(data->GetFloat32Vectors() != nullptr, "base.float_vector is nullptr");
    auto split_datasets = this->split_dataset_by_duplicate_label(data, failed_ids);

    for (auto& data_ptr : split_datasets) {
        this->basic_flatten_codes_->Train(data_ptr->GetFloat32Vectors(),
                                          data_ptr->GetNumElements());
        this->basic_flatten_codes_->BatchInsertVector(data_ptr->GetFloat32Vectors(),
                                                      data_ptr->GetNumElements());
        if (use_reorder_) {
            this->high_precise_codes_->Train(data_ptr->GetFloat32Vectors(),
                                             data_ptr->GetNumElements());
            this->high_precise_codes_->BatchInsertVector(data_ptr->GetFloat32Vectors(),
                                                         data_ptr->GetNumElements());
        }
        this->hnsw_add(data_ptr);
    }
    return failed_ids;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    std::shared_ptr<CommonInnerIdFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<CommonInnerIdFilter>(filter, *this->label_table_);
    }
    int64_t query_dim = query->GetDim();
    CHECK_ARGUMENT(query_dim == dim_,
                   fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(query->GetFloat32Vectors(),
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param);
        search_param.ep = result.top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    auto search_result = this->search_one_graph(
        query->GetFloat32Vectors(), this->bottom_graph_, this->basic_flatten_codes_, search_param);

    if (use_reorder_) {
        this->reorder(query->GetFloat32Vectors(), this->high_precise_codes_, search_result, k);
    }

    while (search_result.size() > k) {
        search_result.pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result.empty()) {
        auto result = Dataset::Make();
        result->Dim(0)->NumElements(1);
        return result;
    }
    auto count = static_cast<const int64_t>(search_result.size());
    auto [dataset_results, dists, ids] = CreateFastDataset(count, allocator_);
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result.top().first;
        ids[j] = this->label_table_->GetLabelById(search_result.top().second);
        search_result.pop();
    }
    return std::move(dataset_results);
}

uint64_t
HGraph::EstimateMemory(uint64_t num_elements) const {
    uint64_t estimate_memory = 0;
    auto block_size = Options::Instance().block_size_limit();
    auto element_count =
        next_multiple_of_power_of_two(num_elements, this->resize_increase_count_bit_);

    auto block_memory_ceil = [](uint64_t memory, uint64_t block_size) -> uint64_t {
        return static_cast<uint64_t>(
            std::ceil(static_cast<double>(memory) / static_cast<double>(block_size)) *
            static_cast<double>(block_size));
    };

    if (this->basic_flatten_codes_->InMemory()) {
        auto base_memory = this->basic_flatten_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(base_memory, block_size);
    }

    if (bottom_graph_->InMemory()) {
        auto bottom_graph_memory =
            (this->bottom_graph_->maximum_degree_ + 1) * sizeof(InnerIdType) * element_count;
        estimate_memory += block_memory_ceil(bottom_graph_memory, block_size);
    }

    if (use_reorder_ && this->high_precise_codes_->InMemory()) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    auto label_map_memory =
        element_count * (sizeof(std::pair<LabelType, InnerIdType>) + 2 * sizeof(void*));
    estimate_memory += label_map_memory;

    auto sparse_graph_memory = (this->mult_ * 0.05 * static_cast<double>(element_count)) *
                               sizeof(InnerIdType) *
                               (static_cast<double>(this->bottom_graph_->maximum_degree_) / 2 + 1);
    estimate_memory += static_cast<uint64_t>(sparse_graph_memory);

    auto other_memory = element_count * (sizeof(LabelType) + sizeof(std::shared_mutex) +
                                         sizeof(std::shared_ptr<std::shared_mutex>));
    estimate_memory += other_memory;

    return estimate_memory;
}

void
HGraph::hnsw_add(const DatasetPtr& data) {
    uint64_t total = data->GetNumElements();
    const auto* ids = data->GetIds();
    const auto* datas = data->GetFloat32Vectors();
    auto cur_count = this->bottom_graph_->TotalCount();
    this->resize(total + cur_count);

    std::mutex add_mutex;

    auto build_func = [&](InnerIdType begin, InnerIdType end) -> void {
        for (InnerIdType i = begin; i < end; ++i) {
            int level = this->get_random_level() - 1;
            auto label = ids[i];
            auto inner_id = i + cur_count;
            {
                std::lock_guard<std::shared_mutex> lock(this->label_lookup_mutex_);
                this->label_table_->Insert(inner_id, label);
            }

            std::unique_lock<std::mutex> add_lock(add_mutex);
            if (level >= int64_t(this->max_level_) || bottom_graph_->TotalCount() == 0) {
                std::lock_guard<std::shared_mutex> wlock(this->global_mutex_);
                // level maybe a negative number(-1)
                for (auto j = static_cast<int64_t>(max_level_); j <= level; ++j) {
                    this->route_graphs_.emplace_back(this->generate_one_route_graph());
                }
                max_level_ = level + 1;
                this->add_one_point(datas + i * dim_, level, inner_id);
                entry_point_id_ = inner_id;
                add_lock.unlock();
            } else {
                add_lock.unlock();
                std::shared_lock<std::shared_mutex> rlock(this->global_mutex_);
                this->add_one_point(datas + i * dim_, level, inner_id);
            }
        }
    };

    if (this->build_pool_ != nullptr) {
        auto task_size = (total + this->build_thread_count_ - 1) / this->build_thread_count_;
        for (uint64_t j = 0; j < this->build_thread_count_; ++j) {
            auto end = std::min(j * task_size + task_size, total);
            this->build_pool_->enqueue(build_func, j * task_size, end);
        }
        this->build_pool_->wait_until_nothing_in_flight();
    } else {
        build_func(0, total);
    }
}

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(this->allocator_,
                                                 bottom_graph_->MaximumDegree() / 2);
}

template <InnerSearchMode mode>
MaxHeap
HGraph::search_one_graph(const float* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param) const {
    auto visited_list = this->pool_->getFreeVisitedList();

    auto* visited_array = visited_list->mass;
    auto visited_array_tag = visited_list->curV;
    auto computer = flatten->FactoryComputer(query);
    auto prefetch_neighbor_visit_num = 1;  // TODO(LHT) Optimize the param;

    auto& is_id_allowed = inner_search_param.is_inner_id_allowed;
    auto ep = inner_search_param.ep;
    auto ef = inner_search_param.ef;

    MaxHeap candidate_set(allocator_);
    MaxHeap cur_result(allocator_);
    float dist = 0.0F;
    auto lower_bound = std::numeric_limits<float>::max();
    flatten->Query(&dist, computer, &ep, 1);
    if (not is_id_allowed || is_id_allowed->CheckValid(ep)) {
        cur_result.emplace(dist, ep);
        lower_bound = cur_result.top().first;
    }
    if constexpr (mode == RANGE_SEARCH) {
        if (dist > inner_search_param.radius) {
            cur_result.pop();
        }
    }
    candidate_set.emplace(-dist, ep);
    visited_array[ep] = visited_array_tag;

    Vector<InnerIdType> neighbors(allocator_);
    Vector<InnerIdType> to_be_visited(graph->MaximumDegree(), allocator_);
    Vector<float> tmp_result(graph->MaximumDegree(), allocator_);

    while (not candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();

        if constexpr (mode == InnerSearchMode::KNN_SEARCH) {
            if ((-current_node_pair.first) > lower_bound && cur_result.size() == ef) {
                break;
            }
        }
        candidate_set.pop();

        auto current_node_id = current_node_pair.second;
        {
            SharedLock lock(neighbors_mutex_, current_node_id);
            graph->GetNeighbors(current_node_id, neighbors);
        }
        if (!neighbors.empty()) {
            flatten->Prefetch(neighbors[0]);
#ifdef USE_SSE
            _mm_prefetch((char*)(visited_array + neighbors[0]), _MM_HINT_T0);
            for (uint32_t i = 0; i < prefetch_neighbor_visit_num; i++) {
                _mm_prefetch(visited_list->mass + neighbors[i], _MM_HINT_T0);
            }
#endif
        }
        auto count_no_visited = 0;
        for (uint64_t i = 0; i < neighbors.size(); ++i) {  // NOLINT(modernize-loop-convert)
            const auto& neighbor = neighbors[i];
#if defined(USE_SSE)
            if (i + prefetch_neighbor_visit_num < neighbors.size()) {
                _mm_prefetch(visited_array + neighbors[i + prefetch_neighbor_visit_num],
                             _MM_HINT_T0);
            }
#endif
            if (visited_array[neighbor] != visited_array_tag) {
                to_be_visited[count_no_visited] = neighbor;
                count_no_visited++;
                visited_array[neighbor] = visited_array_tag;
            }
        }

        flatten->Query(tmp_result.data(), computer, to_be_visited.data(), count_no_visited);

        for (auto i = 0; i < count_no_visited; ++i) {
            dist = tmp_result[i];
            if (cur_result.size() < ef || lower_bound > dist ||
                (mode == RANGE_SEARCH && dist <= inner_search_param.radius)) {
                candidate_set.emplace(-dist, to_be_visited[i]);
                flatten->Prefetch(candidate_set.top().second);

                if (not is_id_allowed || is_id_allowed->CheckValid(to_be_visited[i])) {
                    cur_result.emplace(dist, to_be_visited[i]);
                }

                if constexpr (mode == KNN_SEARCH) {
                    if (cur_result.size() > ef) {
                        cur_result.pop();
                    }
                }

                if (not cur_result.empty()) {
                    lower_bound = cur_result.top().first;
                }
            }
        }
    }
    this->pool_->releaseVisitedList(visited_list);
    return cur_result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    std::shared_ptr<CommonInnerIdFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<CommonInnerIdFilter>(filter, *this->label_table_);
    }
    int64_t query_dim = query->GetDim();
    CHECK_ARGUMENT(query_dim == dim_,
                   fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.ef = 1;
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(query->GetFloat32Vectors(),
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             search_param);
        search_param.ep = result.top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    auto search_result = this->search_one_graph(
        query->GetFloat32Vectors(), this->bottom_graph_, this->basic_flatten_codes_, search_param);
    if (use_reorder_) {
        this->reorder(
            query->GetFloat32Vectors(), this->high_precise_codes_, search_result, limited_size);
    }

    if (limited_size > 0) {
        while (search_result.size() > limited_size) {
            search_result.pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result.size());
    auto [dataset_results, dists, ids] = CreateFastDataset(count, allocator_);
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result.top().first;
        ids[j] = this->label_table_->GetLabelById(search_result.top().second);
        search_result.pop();
    }
    return std::move(dataset_results);
}

void
HGraph::serialize_basic_info(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    StreamWriter::WriteObj(writer, this->max_level_);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    StreamWriter::WriteObj(writer, this->max_capacity_);
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);

    uint64_t size = this->label_table_->label_remap_.size();
    StreamWriter::WriteObj(writer, size);
    for (const auto& pair : this->label_table_->label_remap_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraph::Serialize(StreamWriter& writer) const {
    this->serialize_basic_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->use_reorder_) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (auto i = 0; i < this->max_level_; ++i) {
        this->route_graphs_[i]->Serialize(writer);
    }
}

void
HGraph::Deserialize(StreamReader& reader) {
    this->deserialize_basic_info(reader);
    this->basic_flatten_codes_->Deserialize(reader);
    this->bottom_graph_->Deserialize(reader);
    if (this->use_reorder_) {
        this->high_precise_codes_->Deserialize(reader);
    }

    for (uint64_t i = 0; i < this->max_level_; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }

    for (uint64_t i = 0; i < this->max_level_; ++i) {
        this->route_graphs_[i]->Deserialize(reader);
    }
    this->neighbors_mutex_->Resize(max_capacity_);
    pool_ = std::make_shared<hnswlib::VisitedListPool>(max_capacity_, allocator_);
}

void
HGraph::deserialize_basic_info(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    StreamReader::ReadObj(reader, this->max_level_);
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    StreamReader::ReadObj(reader, this->max_capacity_);
    StreamReader::ReadVector(reader, this->label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->label_remap_.emplace(key, value);
    }
}

float
HGraph::CalcDistanceById(const float* query, int64_t id) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_) {
        flat = this->high_precise_codes_;
    }
    float result = 0.0F;
    auto computer = flat->FactoryComputer(query);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        auto iter = this->label_table_->label_remap_.find(id);
        if (iter == this->label_table_->label_remap_.end()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("failed to find id: {}", id));
        }
        auto new_id = iter->second;
        flat->Query(&result, computer, &new_id, 1);
        return result;
    }
}

DatasetPtr
HGraph::CalDistanceById(const float* query, const int64_t* ids, int64_t count) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_) {
        flat = this->high_precise_codes_;
    }
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = (float*)allocator_->Allocate(sizeof(float) * count);
    result->Distances(distances);
    auto computer = flat->FactoryComputer(query);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        for (int64_t i = 0; i < count; ++i) {
            auto iter = this->label_table_->label_remap_.find(ids[i]);
            if (iter == this->label_table_->label_remap_.end()) {
                logger::error(fmt::format("failed to find id: {}", ids[i]));
                distances[i] = -1;
                continue;
            }
            auto new_id = iter->second;
            flat->Query(distances + i, computer, &new_id, 1);
        }
    }
    return result;
}

void
HGraph::add_one_point(const float* data, int level, InnerIdType inner_id) {
    MaxHeap result(allocator_);

    InnerSearchParam param{
        .ep = this->entry_point_id_,
        .ef = 1,
        .is_inner_id_allowed = nullptr,
    };

    LockGuard cur_lock(neighbors_mutex_, inner_id);
    auto flatten_codes = basic_flatten_codes_;
    if (use_reorder_) {
        flatten_codes = high_precise_codes_;
    }
    for (auto j = max_level_ - 1; j > level; --j) {
        result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
        param.ep = result.top().second;
    }

    param.ef = this->ef_construct_;
    for (auto j = level; j >= 0; --j) {
        if (route_graphs_[j]->TotalCount() != 0) {
            result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
            param.ep = mutually_connect_new_element(
                inner_id, result, route_graphs_[j], flatten_codes, neighbors_mutex_, allocator_);
        } else {
            route_graphs_[j]->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        }
    }
    if (bottom_graph_->TotalCount() != 0) {
        result = search_one_graph(data, this->bottom_graph_, flatten_codes, param);
        mutually_connect_new_element(
            inner_id, result, this->bottom_graph_, flatten_codes, neighbors_mutex_, allocator_);
    } else {
        bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    }
}

void
HGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_;
    logger::debug(
        "hgraph resize from ", std::to_string(cur_size), " to " + std::to_string(new_size));
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<hnswlib::VisitedListPool>(new_size_power_2, allocator_);
        this->label_table_->label_table_.resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);
        this->max_capacity_ = new_size_power_2;
    }
}
void
HGraph::init_features() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
    });
    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_RANGE_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
    });
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
    });
    // other
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_ESTIMATE_MEMORY,
        IndexFeature::SUPPORT_CHECK_ID_EXIST,
    });

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    }

    bool have_fp32 = false;
    if (name == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
    }
    if (use_reorder_ and
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
    }
    if (have_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
    }

    // metric
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }
}

Vector<DatasetPtr>
HGraph::split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                         std::vector<LabelType>& failed_ids) const {
    Vector<DatasetPtr> return_datasets(0, this->allocator_);
    auto count = dataset->GetNumElements();
    auto dim = dataset->GetDim();
    const auto* labels = dataset->GetIds();
    const auto* vec = dataset->GetFloat32Vectors();
    UnorderedSet<LabelType> temp_labels(allocator_);

    for (uint64_t i = 0; i < count; ++i) {
        if (this->label_table_->CheckLabel(labels[i]) or
            temp_labels.find(labels[i]) != temp_labels.end()) {
            failed_ids.emplace_back(i);
            continue;
        }
        temp_labels.emplace(labels[i]);
    }
    failed_ids.emplace_back(count);

    if (failed_ids.size() == 1) {
        return_datasets.emplace_back(dataset);
        return return_datasets;
    }
    int64_t start = -1;
    for (auto end : failed_ids) {
        if (end - start == 1) {
            start = end;
            continue;
        }
        auto new_dataset = Dataset::Make();
        new_dataset->NumElements(end - start - 1)
            ->Dim(dim)
            ->Ids(labels + start + 1)
            ->Float32Vectors(vec + dim * (start + 1))
            ->Owner(false);
        return_datasets.emplace_back(new_dataset);
        start = end;
    }
    failed_ids.pop_back();
    for (auto& failed_id : failed_ids) {
        failed_id = labels[failed_id];
    }
    return return_datasets;
}

void
HGraph::reorder(const float* query,
                const FlattenInterfacePtr& flatten_interface,
                MaxHeap& candidate_heap,
                int64_t k) const {
    uint64_t size = candidate_heap.size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    Vector<InnerIdType> ids(size, allocator_);
    Vector<float> dists(size, allocator_);
    uint64_t idx = 0;
    while (not candidate_heap.empty()) {
        ids[idx] = candidate_heap.top().second;
        ++idx;
        candidate_heap.pop();
    }
    auto computer = flatten_interface->FactoryComputer(query);
    flatten_interface->Query(dists.data(), computer, ids.data(), size);
    for (uint64_t i = 0; i < size; ++i) {
        if (candidate_heap.size() < k or dists[i] <= candidate_heap.top().first) {
            candidate_heap.emplace(dists[i], ids[i]);
        }
        if (candidate_heap.size() > k) {
            candidate_heap.pop();
        }
    }
}

static const ConstParamMap EXTERNAL_MAPPING = {
    {
        HGRAPH_USE_REORDER,
        {
            HGRAPH_USE_REORDER_KEY,
        },
    },
    {
        HGRAPH_BASE_QUANTIZATION_TYPE,
        {
            HGRAPH_BASE_CODES_KEY,
            QUANTIZATION_PARAMS_KEY,
            QUANTIZATION_TYPE_KEY,
        },
    },
    {
        HGRAPH_BASE_IO_TYPE,
        {
            HGRAPH_BASE_CODES_KEY,
            IO_PARAMS_KEY,
            IO_TYPE_KEY,
        },
    },
    {
        HGRAPH_PRECISE_IO_TYPE,
        {
            HGRAPH_PRECISE_CODES_KEY,
            IO_PARAMS_KEY,
            IO_TYPE_KEY,
        },
    },
    {
        HGRAPH_BASE_FILE_PATH,
        {
            HGRAPH_BASE_CODES_KEY,
            IO_PARAMS_KEY,
            IO_FILE_PATH,
        },
    },
    {
        HGRAPH_PRECISE_FILE_PATH,
        {
            HGRAPH_PRECISE_CODES_KEY,
            IO_PARAMS_KEY,
            IO_FILE_PATH,
        },
    },
    {
        HGRAPH_PRECISE_QUANTIZATION_TYPE,
        {
            HGRAPH_PRECISE_CODES_KEY,
            QUANTIZATION_PARAMS_KEY,
            QUANTIZATION_TYPE_KEY,
        },
    },
    {
        HGRAPH_GRAPH_MAX_DEGREE,
        {
            HGRAPH_GRAPH_KEY,
            GRAPH_PARAM_MAX_DEGREE,
        },
    },
    {
        HGRAPH_BUILD_EF_CONSTRUCTION,
        {
            BUILD_PARAMS_KEY,
            BUILD_EF_CONSTRUCTION,
        },
    },
    {
        HGRAPH_INIT_CAPACITY,
        {
            HGRAPH_GRAPH_KEY,
            GRAPH_PARAM_INIT_MAX_CAPACITY,
        },
    },
    {
        HGRAPH_BUILD_THREAD_COUNT,
        {
            BUILD_PARAMS_KEY,
            BUILD_THREAD_COUNT,
        },
    },
    {
        SQ4_UNIFORM_TRUNC_RATE,
        {
            HGRAPH_BASE_CODES_KEY,
            QUANTIZATION_PARAMS_KEY,
            SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE,
        },
    },
};

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_HGRAPH}",
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_PARAM_MAX_DEGREE}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_PQ}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "nbits": 8
            }
        },
        "{HGRAPH_PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}"
            }
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_EF_CONSTRUCTION}": 400,
            "{BUILD_THREAD_COUNT}": 100
        }
    })";

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw std::invalid_argument(fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, EXTERNAL_MAPPING, inner_json);

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->FromJson(inner_json);

    return hgraph_parameter;
}
}  // namespace vsag
