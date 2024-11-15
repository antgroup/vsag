
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

#include "hgraph_index.h"

#include <fmt/format-inl.h>

#include "common.h"
#include "data_cell/sparse_graph_datacell.h"
#include "hnsw.h"
#include "hnsw_zparameters.h"

namespace vsag {
BinarySet
empty_binaryset() {
    const std::string empty_str = "EMPTY_INDEX";
    size_t num_bytes = empty_str.length();
    std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
    memcpy(bin.get(), empty_str.c_str(), empty_str.length());
    Binary b{
        .data = bin,
        .size = num_bytes,
    };
    BinarySet bs;
    bs.Set(BLANK_INDEX, b);

    return bs;
}

HGraphIndex::HGraphIndex(const JsonType& index_param,
                         const vsag::IndexCommonParam& common_param) noexcept
    : index_param_(index_param),
      common_param_(common_param),
      label_lookup_(common_param.allocator_),
      label_op_mutex_(MAX_LABEL_OPERATION_LOCKS, common_param_.allocator_),
      neighbors_mutex_(0, common_param_.allocator_),
      route_graphs_(common_param.allocator_) {
    this->dim_ = common_param.dim_;
    this->metric_ = common_param.metric_;
    this->allocator_ = common_param.allocator_;
}

void
HGraphIndex::Init() {
    CHECK_ARGUMENT(this->index_param_.contains(HGRAPH_USE_REORDER_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_USE_REORDER_KEY));
    this->use_reorder_ = this->index_param_[HGRAPH_USE_REORDER_KEY];

    CHECK_ARGUMENT(this->index_param_.contains(HGRAPH_BASE_CODES_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_BASE_CODES_KEY));
    const auto& base_codes_json_obj = this->index_param_[HGRAPH_BASE_CODES_KEY];
    this->basic_flatten_codes_ = FlattenInterface::MakeInstance(base_codes_json_obj, common_param_);

    if (this->use_reorder_) {
        CHECK_ARGUMENT(this->index_param_.contains(HGRAPH_PRECISE_CODES_KEY),
                       fmt::format("hgraph parameters must contains {}", HGRAPH_PRECISE_CODES_KEY));
        const auto& precise_codes_json_obj = this->index_param_[HGRAPH_PRECISE_CODES_KEY];
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(precise_codes_json_obj, common_param_);
    }

    CHECK_ARGUMENT(this->index_param_.contains(HGRAPH_GRAPH_KEY),
                   fmt::format("hgraph parameters must contains {}", HGRAPH_GRAPH_KEY));
    const auto& graph_json_obj = this->index_param_[HGRAPH_GRAPH_KEY];
    this->bottom_graph_ = GraphInterface::MakeInstance(graph_json_obj, common_param_);

    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    this->pool_ = std::make_shared<hnswlib::VisitedListPool>(
        1, this->bottom_graph_->MaxCapacity(), allocator_);

    if (this->build_thread_count_ > 1) {
        this->build_pool_ = std::make_unique<progschj::ThreadPool>(this->build_thread_count_);
    }
}

tl::expected<std::vector<int64_t>, Error>
HGraphIndex::build(const DatasetPtr& data) {
    return this->add(data);
}

tl::expected<std::vector<int64_t>, Error>
HGraphIndex::add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    try {
        this->basic_flatten_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
        this->basic_flatten_codes_->BatchInsertVector(data->GetFloat32Vectors(),
                                                      data->GetNumElements());
        if (use_reorder_) {
            this->high_precise_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
            this->high_precise_codes_->BatchInsertVector(data->GetFloat32Vectors(),
                                                         data->GetNumElements());
        }
        this->hnsw_add(data);
        return failed_ids;  // TODO(LHT): maybe use copy for small batch filtered by label
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to add(invalid argument): ", e.what());
    }
}

tl::expected<DatasetPtr, Error>
HGraphIndex::knn_search(const DatasetPtr& query,
                        int64_t k,
                        const std::string& parameters,
                        const std::function<bool(int64_t)>& filter) const {
    BitsetOrCallbackFilter ft(filter);

    auto ep = this->entry_point_id_;
    for (int64_t i = this->route_graphs_.size() - 1; i >= 0; --i) {
        auto result = this->search_one_graph(query->GetFloat32Vectors(),
                                             this->route_graphs_[i],
                                             this->basic_flatten_codes_,
                                             ep,
                                             1,
                                             nullptr);
        ep = result.top().second;
    }

    auto params = HnswSearchParameters::FromJson(parameters);

    auto ef = params.ef_search;
    auto search_result = this->search_one_graph(
        query->GetFloat32Vectors(), this->bottom_graph_, this->basic_flatten_codes_, ep, ef, &ft);

    while (search_result.size() > k) {
        search_result.pop();
    }

    auto dataset_results = Dataset::Make();
    dataset_results->Dim(search_result.size())->NumElements(1)->Owner(true, allocator_);

    auto* ids = (int64_t*)allocator_->Allocate(sizeof(int64_t) * search_result.size());
    dataset_results->Ids(ids);
    auto* dists = (float*)allocator_->Allocate(sizeof(float) * search_result.size());
    dataset_results->Distances(dists);

    for (int64_t j = search_result.size() - 1; j >= 0; --j) {
        dists[j] = search_result.top().first;
        ids[j] = search_result.top().second;
        search_result.pop();
    }
    return std::move(dataset_results);
}

tl::expected<BinarySet, Error>
HGraphIndex::serialize() const {
    if (GetNumElements() == 0) {
        return empty_binaryset();
    }
    SlowTaskTimer t("hgraph serialize");
    size_t num_bytes = this->cal_serialize_size();
    try {
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        auto buffer = reinterpret_cast<char*>(const_cast<int8_t*>(bin.get()));
        BufferStreamWriter writer(buffer);
        this->serialize(writer);
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(INDEX_HGRAPH, b);

        return bs;
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
HGraphIndex::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hgraph deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    try {
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            reader_set.Get(INDEX_HGRAPH)->Read(offset, len, dest);
        };
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->deserialize(reader);
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
};

void
HGraphIndex::hnsw_add(const DatasetPtr& data) {
    uint64_t total = data->GetNumElements();
    auto* ids = data->GetIds();
    auto* datas = data->GetFloat32Vectors();
    auto cur_count = this->bottom_graph_->TotalCount();
    vsag::Vector<std::shared_mutex>(total + cur_count, allocator_).swap(this->neighbors_mutex_);

    auto build_func = [&](InnerIdType begin, InnerIdType end) -> void {
        for (InnerIdType i = begin; i < end; ++i) {
            int level = this->get_random_level() - 1;
            auto label = ids[i];
            auto inner_id = i + cur_count;
            {
                std::unique_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
                this->label_lookup_[label] = inner_id;
            }

            std::unique_lock<std::mutex> lock(this->global_mutex_);
            bool need_lock = false;
            if (level >= int64_t(this->max_level_) || bottom_graph_->TotalCount() == 0) {
                for (int64_t j = max_level_; j <= level; ++j) {
                    this->route_graphs_.emplace_back(this->generate_one_route_graph());
                }
                max_level_ = level + 1;
                need_lock = true;
            } else {
                lock.unlock();
            }

            {
                auto ep = this->entry_point_id_;
                MaxHeap result(allocator_);
                for (auto j = max_level_ - 1; j > level; --j) {
                    result = search_one_graph(
                        datas + dim_ * i, route_graphs_[j], basic_flatten_codes_, ep, 1, nullptr);
                    ep = result.top().second;
                }

                for (auto j = level; j >= 0; --j) {
                    if (route_graphs_[j]->TotalCount() != 0) {
                        result = search_one_graph(datas + dim_ * i,
                                                  route_graphs_[j],
                                                  basic_flatten_codes_,
                                                  ep,
                                                  this->ef_construct_,
                                                  nullptr);
                        ep = this->mutually_connect_new_element(
                            inner_id, result, route_graphs_[j], basic_flatten_codes_, false);
                    } else {
                        route_graphs_[j]->InsertNeighborsById(inner_id,
                                                              Vector<InnerIdType>(allocator_));
                    }
                    route_graphs_[j]->IncreaseTotalCount(1);
                }
                if (bottom_graph_->TotalCount() != 0) {
                    result = search_one_graph(datas + dim_ * i,
                                              this->bottom_graph_,
                                              basic_flatten_codes_,
                                              ep,
                                              this->ef_construct_,
                                              nullptr);
                    this->mutually_connect_new_element(
                        inner_id, result, this->bottom_graph_, basic_flatten_codes_, false);
                } else {
                    bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
                }
                bottom_graph_->IncreaseTotalCount(1);
            }

            if (need_lock) {
                entry_point_id_ = inner_id;
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
HGraphIndex::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(this->allocator_,
                                                 bottom_graph_->MaximumDegree() / 2);
}

HGraphIndex::MaxHeap
HGraphIndex::search_one_graph(const float* query,
                              const GraphInterfacePtr& graph,
                              const FlattenInterfacePtr& flatten,
                              InnerIdType ep,
                              uint64_t ef,
                              BaseFilterFunctor* is_id_allowed) const {
    auto visited_list = this->pool_->getFreeVisitedList();

    auto* visited_array = visited_list->mass;
    auto visited_array_tag = visited_list->curV;
    auto computer = flatten->FactoryComputer(query);
    auto prefetch_neighbor_visit_num = 1;  // TODO(LHT) Optimize the param;

    MaxHeap candidate_set(allocator_);
    MaxHeap cur_result(allocator_);
    float dist = 0.0f;
    auto lower_bound = std::numeric_limits<float>::max();
    flatten->Query(&dist, computer, &ep, 1);
    if (not is_id_allowed || (*is_id_allowed)(ep)) {
        cur_result.emplace(dist, ep);
        lower_bound = cur_result.top().first;
    }
    candidate_set.emplace(-dist, ep);
    visited_array[ep] = visited_array_tag;

    Vector<InnerIdType> neighbors(allocator_);
    Vector<InnerIdType> to_be_visited(graph->MaximumDegree(), allocator_);
    Vector<float> tmp_result(graph->MaximumDegree(), allocator_);

    while (not candidate_set.empty()) {
        auto current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lower_bound && cur_result.size() == ef) {
            break;
        }
        candidate_set.pop();

        auto current_node_id = current_node_pair.second;
        {
            std::shared_lock<std::shared_mutex> lock(neighbors_mutex_[current_node_id]);
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
        for (uint64_t i = 0; i < neighbors.size(); ++i) {
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
            if (cur_result.size() < ef || lower_bound > dist) {
                candidate_set.emplace(-dist, to_be_visited[i]);
                flatten->Prefetch(candidate_set.top().second);

                if (not is_id_allowed || (*is_id_allowed)(to_be_visited[i])) {
                    cur_result.emplace(dist, to_be_visited[i]);
                }

                if (cur_result.size() > ef)
                    cur_result.pop();

                if (not cur_result.empty())
                    lower_bound = cur_result.top().first;
            }
        }
    }
    this->pool_->releaseVisitedList(visited_list);
    return cur_result;
}

void
HGraphIndex::select_edges_by_heuristic(HGraphIndex::MaxHeap& edges,
                                       uint64_t max_size,
                                       const FlattenInterfacePtr& flatten) {
    if (edges.size() < max_size) {
        return;
    }

    MaxHeap queue_closest(allocator_);
    vsag::Vector<std::pair<float, InnerIdType>> return_list(allocator_);
    while (not edges.empty()) {
        queue_closest.emplace(-edges.top().first, edges.top().second);
        edges.pop();
    }

    while (not queue_closest.empty()) {
        if (return_list.size() >= max_size)
            break;
        std::pair<float, InnerIdType> curent_pair = queue_closest.top();
        float float_query = -curent_pair.first;
        queue_closest.pop();
        bool good = true;

        for (const auto& second_pair : return_list) {
            float curdist = flatten->ComputePairVectors(second_pair.second, curent_pair.second);
            if (curdist < float_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.emplace_back(curent_pair);
        }
    }

    for (const auto& curent_pair : return_list) {
        edges.emplace(-curent_pair.first, curent_pair.second);
    }
}

InnerIdType
HGraphIndex::mutually_connect_new_element(InnerIdType cur_c,
                                          MaxHeap& top_candidates,
                                          GraphInterfacePtr graph,
                                          FlattenInterfacePtr flatten,
                                          bool is_update) {
    const size_t max_size = graph->MaximumDegree();
    this->select_edges_by_heuristic(top_candidates, max_size, flatten);
    if (top_candidates.size() > max_size)
        throw std::runtime_error(
            "Should be not be more than max_size candidates returned by the heuristic");

    Vector<InnerIdType> selected_neighbors(allocator_);
    selected_neighbors.reserve(max_size);
    while (not top_candidates.empty()) {
        selected_neighbors.emplace_back(top_candidates.top().second);
        top_candidates.pop();
    }

    InnerIdType next_closest_entry_point = selected_neighbors.back();

    {
        // because during the addition the lock for cur_c is already acquired
        std::unique_lock<std::shared_mutex> lock(neighbors_mutex_[cur_c], std::defer_lock);
        if (is_update) {
            lock.lock();
        }
        graph->InsertNeighborsById(cur_c, selected_neighbors);
    }

    for (auto selectedNeighbor : selected_neighbors) {
        std::unique_lock<std::shared_mutex> lock(neighbors_mutex_[selectedNeighbor]);

        Vector<InnerIdType> neighbors(allocator_);
        graph->GetNeighbors(selectedNeighbor, neighbors);

        size_t sz_link_list_other = neighbors.size();

        if (sz_link_list_other > max_size)
            throw std::runtime_error("Bad value of sz_link_list_other");
        if (selectedNeighbor == cur_c)
            throw std::runtime_error("Trying to connect an element to itself");

        bool is_cur_c_present = false;
        if (is_update) {
            for (size_t j = 0; j < sz_link_list_other; j++) {
                if (neighbors[j] == cur_c) {
                    is_cur_c_present = true;
                    break;
                }
            }
        }

        // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
        if (!is_cur_c_present) {
            if (sz_link_list_other < max_size) {
                neighbors.emplace_back(cur_c);
                graph->InsertNeighborsById(selectedNeighbor, neighbors);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = flatten->ComputePairVectors(cur_c, selectedNeighbor);

                MaxHeap candidates(allocator_);
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(flatten->ComputePairVectors(neighbors[j], selectedNeighbor),
                                       neighbors[j]);
                }

                this->select_edges_by_heuristic(candidates, max_size, flatten);

                Vector<InnerIdType> cand_neighbors(allocator_);
                while (not candidates.empty()) {
                    cand_neighbors.emplace_back(candidates.top().second);
                    candidates.pop();
                }
                graph->InsertNeighborsById(selectedNeighbor, cand_neighbors);
            }
        }
    }
    return next_closest_entry_point;
}

void
HGraphIndex::serialize_basic_info(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    StreamWriter::WriteObj(writer, this->max_level_);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    uint64_t size = this->label_lookup_.size();
    StreamWriter::WriteObj(writer, size);
    for (auto& pair : this->label_lookup_) {
        auto key = pair.first;
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, pair.second);
    }
}

void
HGraphIndex::serialize(StreamWriter& writer) const {
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
HGraphIndex::deserialize(StreamReader& reader) {
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
    vsag::Vector<std::shared_mutex>(this->GetNumElements(), allocator_)
        .swap(this->neighbors_mutex_);
}

void
HGraphIndex::deserialize_basic_info(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    StreamReader::ReadObj(reader, this->max_level_);
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_lookup_.emplace(key, value);
    }
}

uint64_t
HGraphIndex::cal_serialize_size() const {
    auto calSizeFunc = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(calSizeFunc, 0);
    this->serialize(writer);
    return writer.cursor_;
}
tl::expected<void, Error>
HGraphIndex::serialize(std::ostream& out_stream) const {
    try {
        IOStreamWriter writer(out_stream);
        this->serialize(writer);
        return {};
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
HGraphIndex::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    // check if binary set is an empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        return {};
    }

    Binary b = binary_set.Get(INDEX_HGRAPH);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->deserialize(reader);
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    } catch (const std::out_of_range& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}
tl::expected<void, Error>
HGraphIndex::deserialize(std::istream& in_stream) {
    SlowTaskTimer t("hgraph deserialize");
    if (this->GetNumElements() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }
    try {
        IOStreamReader reader(in_stream);
        this->deserialize(reader);
        return {};
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to deserialize(bad alloc): ", e.what());
    }
}
tl::expected<float, Error>
HGraphIndex::calc_distance_by_id(const float* vector, int64_t id) const {
    auto flat = this->basic_flatten_codes_;
    if (use_reorder_) {
        flat = this->high_precise_codes_;
    }
    float result = 0.0f;
    auto computer = flat->FactoryComputer(vector);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        auto iter = this->label_lookup_.find(id);
        if (iter != this->label_lookup_.end()) {
            auto new_id = iter->second;
            flat->Query(&result, computer, &new_id, 1);
            return result;
        } else {
            LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                  fmt::format("failed to find id: {}", id));
        }
    }
}

}  // namespace vsag
