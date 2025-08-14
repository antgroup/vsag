
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

#include <data_cell/compressed_graph_datacell_parameter.h>
#include <fmt/format.h>

#include <memory>
#include <stdexcept>

#include "attr/argparse.h"
#include "common.h"
#include "data_cell/sparse_graph_datacell.h"
#include "dataset_impl.h"
#include "impl/heap/standard_heap.h"
#include "impl/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "impl/reorder.h"
#include "index/index_impl.h"
#include "index/iterator_filter.h"
#include "io/reader_io_parameter.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "vsag/options.h"

namespace vsag {

HGraph::HGraph(const HGraphParameterPtr& hgraph_param, const vsag::IndexCommonParam& common_param)
    : InnerIndexInterface(hgraph_param, common_param),
      route_graphs_(common_param.allocator_.get()),
      use_reorder_(hgraph_param->use_reorder),
      use_elp_optimizer_(hgraph_param->use_elp_optimizer),
      ignore_reorder_(hgraph_param->ignore_reorder),
      build_by_base_(hgraph_param->build_by_base),
      use_attribute_filter_(hgraph_param->use_attribute_filter),
      ef_construct_(hgraph_param->ef_construction),
      build_thread_count_(hgraph_param->build_thread_count),
      odescent_param_(hgraph_param->odescent_param),
      graph_type_(hgraph_param->graph_type),
      hierarchical_datacell_param_(hgraph_param->hierarchical_graph_param),
      extra_info_size_(common_param.extra_info_size_),
      deleted_ids_(allocator_) {
    this->label_table_->compress_duplicate_data_ = hgraph_param->support_duplicate;
    neighbors_mutex_ = std::make_shared<PointsMutex>(0, common_param.allocator_.get());
    this->basic_flatten_codes_ =
        FlattenInterface::MakeInstance(hgraph_param->base_codes_param, common_param);
    if (use_reorder_) {
        this->high_precise_codes_ =
            FlattenInterface::MakeInstance(hgraph_param->precise_codes_param, common_param);
    }
    this->searcher_ = std::make_shared<BasicSearcher>(common_param, neighbors_mutex_);

    this->bottom_graph_ =
        GraphInterface::MakeInstance(hgraph_param->bottom_graph_param, common_param);
    mult_ = 1 / log(1.0 * static_cast<double>(this->bottom_graph_->MaximumDegree()));

    if (extra_info_size_ > 0) {
        this->extra_infos_ =
            ExtraInfoInterface::MakeInstance(hgraph_param->extra_info_param, common_param);
    }

    auto step_block_size = Options::Instance().block_size_limit();
    auto block_size_per_vector = this->basic_flatten_codes_->code_size_;
    block_size_per_vector =
        std::max(block_size_per_vector,
                 static_cast<uint32_t>(this->bottom_graph_->maximum_degree_ * sizeof(InnerIdType)));
    if (use_reorder_) {
        block_size_per_vector =
            std::max(block_size_per_vector, this->high_precise_codes_->code_size_);
    }
    if (extra_infos_ != nullptr) {
        block_size_per_vector =
            std::max(block_size_per_vector, static_cast<uint32_t>(this->extra_info_size_));
    }
    auto increase_count = step_block_size / block_size_per_vector;
    this->resize_increase_count_bit_ = std::max(
        DEFAULT_RESIZE_BIT, static_cast<uint64_t>(log2(static_cast<double>(increase_count))));

    resize(bottom_graph_->max_capacity_);
    this->build_pool_ = common_param.thread_pool_;
    if (this->build_thread_count_ > 1 && this->build_pool_ == nullptr) {
        this->build_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        this->build_pool_->SetPoolSize(build_thread_count_);
    }

    UnorderedMap<std::string, float> default_param(common_param.allocator_.get());
    default_param.insert(
        {PREFETCH_DEPTH_CODE, (this->basic_flatten_codes_->code_size_ + 63.0) / 64.0});
    this->basic_flatten_codes_->SetRuntimeParameters(default_param);

    if (use_elp_optimizer_) {
        optimizer_ = std::make_shared<Optimizer<BasicSearcher>>(common_param);
    }
    if (use_attribute_filter_) {
        this->attr_filter_index_ =
            AttributeInvertedInterface::MakeInstance(allocator_, false /*have_bucket*/);
    }
}
void
HGraph::Train(const DatasetPtr& base) {
    const auto* base_data = get_data(base);
    this->basic_flatten_codes_->Train(base_data, base->GetNumElements());
    if (use_reorder_) {
        this->high_precise_codes_->Train(base_data, base->GetNumElements());
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    this->Train(data);
    std::vector<int64_t> ret;
    if (graph_type_ == GRAPH_TYPE_NSW) {
        ret = this->Add(data);
    } else {
        ret = this->build_by_odescent(data);
    }
    if (use_elp_optimizer_) {
        elp_optimize();
    }
    return ret;
}

std::vector<int64_t>
HGraph::build_by_odescent(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;

    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* vectors = data->GetFloat32Vectors();
    const auto* extra_infos = data->GetExtraInfos();
    auto inner_ids = this->get_unique_inner_ids(total);
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    InnerIdType cur_size = 0;
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label)) {
            failed_ids.emplace_back(label);
            continue;
        }
        InnerIdType inner_id = inner_ids.at(cur_size);
        cur_size++;
        this->label_table_->Insert(inner_id, label);
        this->basic_flatten_codes_->InsertVector(vectors + dim_ * i, inner_id);
        if (use_reorder_) {
            this->high_precise_codes_->InsertVector(vectors + dim_ * i, inner_id);
        }
        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(Vector<InnerIdType>(allocator_));
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    this->resize(total_count_);
    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(odescent_param_, build_data, allocator_, this->build_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->build_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    return failed_ids;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data) {
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    }
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.float_vector is nullptr");

    {
        std::lock_guard lock(this->add_mutex_);
        if (this->total_count_ == 0) {
            this->Train(data);
        }
    }

    auto add_func = [&](const void* data,
                        int level,
                        InnerIdType inner_id,
                        const char* extra_info,
                        const AttributeSet* attrs) -> void {
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->InsertExtraInfo(extra_info, inner_id);
        }
        if (attrs != nullptr and this->use_attribute_filter_) {
            this->attr_filter_index_->Insert(*attrs, inner_id);
        }
        this->add_one_point(data, level, inner_id);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        auto label = labels[j];
        InnerIdType inner_id;
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            if (this->label_table_->CheckLabel(label)) {
                failed_ids.emplace_back(label);
                continue;
            }
            {
                std::lock_guard lock(this->add_mutex_);
                inner_id = this->get_unique_inner_ids(1).at(0);
                uint64_t new_count = total_count_;
                this->resize(new_count);
            }
            this->label_table_->Insert(inner_id, label);
            inner_ids.emplace_back(inner_id, j);
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        int level;
        {
            std::lock_guard label_lock(this->label_lookup_mutex_);
            level = this->get_random_level() - 1;
        }
        const auto* extra_info = extra_infos + local_idx * extra_info_size_;
        const AttributeSet* cur_attr_set = nullptr;
        if (attr_sets != nullptr) {
            cur_attr_set = attr_sets + local_idx;
        }
        if (this->build_pool_ != nullptr) {
            auto future = this->build_pool_->GeneralEnqueue(
                add_func, get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
            futures.emplace_back(std::move(future));
        } else {
            add_func(get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
        }
    }
    if (this->build_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    return failed_ids;
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter) const {
    return KnnSearch(query, k, parameters, filter, nullptr);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator) const {
    SearchRequest req;
    req.query_ = query;
    req.topk_ = k;
    req.filter_ = filter;
    req.params_str_ = parameters;
    req.search_allocator_ = allocator;
    return this->SearchWithRequest(req);
}

DatasetPtr
HGraph::KnnSearch(const DatasetPtr& query,
                  int64_t k,
                  const std::string& parameters,
                  const FilterPtr& filter,
                  Allocator* allocator,
                  IteratorContext*& iter_ctx,
                  bool is_last_filter) const {
    Allocator* search_allocator = allocator == nullptr ? allocator_ : allocator;
    if (GetNumElements() == 0) {
        return DatasetImpl::MakeEmptyDataset();
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(parameters);
    auto ef_search_threshold = std::max(AMPLIFICATION_FACTOR * k, 1000L);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    FilterPtr ft = nullptr;
    if (filter != nullptr) {
        if (params.use_extra_info_filter) {
            ft = std::make_shared<ExtraInfoWrapperFilter>(filter, this->extra_infos_);
        } else {
            ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
        }
    }

    if (iter_ctx == nullptr) {
        auto cur_count = this->bottom_graph_->TotalCount();
        auto* new_ctx = new IteratorFilterContext();
        if (auto ret = new_ctx->init(cur_count, params.ef_search, search_allocator);
            not ret.has_value()) {
            throw vsag::VsagException(ErrorType::INTERNAL_ERROR,
                                      "failed to init IteratorFilterContext");
        }
        iter_ctx = new_ctx;
    }

    auto* iter_filter_ctx = static_cast<IteratorFilterContext*>(iter_ctx);
    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(search_allocator, k);
    const auto* query_data = get_data(query);
    if (is_last_filter) {
        while (!iter_filter_ctx->Empty()) {
            uint32_t cur_inner_id = iter_filter_ctx->GetTopID();
            float cur_dist = iter_filter_ctx->GetTopDist();
            search_result->Push(cur_dist, cur_inner_id);
            iter_filter_ctx->PopDiscard();
        }
    } else {
        InnerSearchParam search_param;
        search_param.ep = this->entry_point_id_;
        search_param.topk = 1;
        search_param.ef = 1;
        search_param.is_inner_id_allowed = nullptr;
        search_param.search_alloc = search_allocator;
        if (iter_filter_ctx->IsFirstUsed()) {
            for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
                auto result = this->search_one_graph(
                    query_data, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
                search_param.ep = result->Top().second;
            }
        }

        search_param.ef = std::max(params.ef_search, k);
        search_param.is_inner_id_allowed = ft;
        search_param.topk = static_cast<int64_t>(search_param.ef);
        search_result = this->search_one_graph(query_data,
                                               this->bottom_graph_,
                                               this->basic_flatten_codes_,
                                               search_param,
                                               iter_filter_ctx);
    }

    if (use_reorder_) {
        this->reorder(query_data, this->high_precise_codes_, search_result, k);
    }

    while (search_result->Size() > k) {
        auto curr = search_result->Top();
        iter_filter_ctx->AddDiscardNode(curr.first, curr.second);
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, search_allocator);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)search_allocator->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        iter_filter_ctx->SetPoint(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    iter_filter_ctx->SetOFFFirstUsed();
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

    if (use_reorder_ && this->high_precise_codes_->InMemory() && not this->ignore_reorder_) {
        auto precise_memory = this->high_precise_codes_->code_size_ * element_count;
        estimate_memory += block_memory_ceil(precise_memory, block_size);
    }

    if (extra_info_size_ > 0 && this->extra_infos_ != nullptr && this->extra_infos_->InMemory()) {
        auto extra_info_memory = this->extra_infos_->ExtraInfoSize() * element_count;
        estimate_memory += block_memory_ceil(extra_info_memory, block_size);
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

GraphInterfacePtr
HGraph::generate_one_route_graph() {
    return std::make_shared<SparseGraphDataCell>(hierarchical_datacell_param_, this->allocator_);
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param) const {
    auto visited_list = this->pool_->TakeOne();
    auto result = this->searcher_->Search(
        graph, flatten, visited_list, query, inner_search_param, this->label_table_);
    this->pool_->ReturnOne(visited_list);
    return result;
}

template <InnerSearchMode mode>
DistHeapPtr
HGraph::search_one_graph(const void* query,
                         const GraphInterfacePtr& graph,
                         const FlattenInterfacePtr& flatten,
                         InnerSearchParam& inner_search_param,
                         IteratorFilterContext* iter_ctx) const {
    auto visited_list = this->pool_->TakeOne();
    auto result =
        this->searcher_->Search(graph, flatten, visited_list, query, inner_search_param, iter_ctx);
    this->pool_->ReturnOne(visited_list);
    return result;
}

DatasetPtr
HGraph::RangeSearch(const DatasetPtr& query,
                    float radius,
                    const std::string& parameters,
                    const FilterPtr& filter,
                    int64_t limited_size) const {
    std::shared_ptr<InnerIdWrapperFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    int64_t query_dim = query->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }
    // check radius
    CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    // check limited_size
    CHECK_ARGUMENT(limited_size != 0,
                   fmt::format("limited_size({}) must not be equal to 0", limited_size));

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
        search_param.ep = result->Top().second;
    }

    auto params = HGraphSearchParameters::FromJson(parameters);

    CHECK_ARGUMENT((1 <= params.ef_search) and (params.ef_search <= 1000),  // NOLINT
                   fmt::format("ef_search({}) must in range[1, 1000]", params.ef_search));
    search_param.ef = std::max(params.ef_search, limited_size);
    search_param.is_inner_id_allowed = ft;
    search_param.radius = radius;
    search_param.search_mode = RANGE_SEARCH;
    search_param.consider_duplicate = true;
    search_param.range_search_limit_size = static_cast<int>(limited_size);
    auto search_result = this->search_one_graph(
        raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param);
    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, limited_size);
    }

    if (limited_size > 0) {
        while (search_result->Size() > limited_size) {
            search_result->Pop();
        }
    }

    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, allocator_);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)allocator_->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    return std::move(dataset_results);
}

void
HGraph::serialize_basic_info_v0_14(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    uint64_t max_level = this->route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    auto capacity = this->max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
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
HGraph::deserialize_basic_info_v0_14(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    this->max_capacity_.store(capacity);
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

#define TO_JSON(json_obj, var) json_obj[#var] = this->var##_;

#define TO_JSON_BASE64(json_obj, var) json_obj[#var] = base64_encode_obj(this->var##_);

#define TO_JSON_ATOMIC(json_obj, var) json_obj[#var] = this->var##_.load();

JsonType
HGraph::serialize_basic_info() const {
    JsonType jsonify_basic_info;
    TO_JSON(jsonify_basic_info, use_reorder);
    TO_JSON(jsonify_basic_info, dim);
    TO_JSON(jsonify_basic_info, metric);
    TO_JSON(jsonify_basic_info, entry_point_id);
    TO_JSON(jsonify_basic_info, ef_construct);
    TO_JSON(jsonify_basic_info, extra_info_size);
    TO_JSON(jsonify_basic_info, data_type);
    // logger::debug("mult: {}", this->mult_);
    TO_JSON_BASE64(jsonify_basic_info, mult);
    TO_JSON_ATOMIC(jsonify_basic_info, max_capacity);
    jsonify_basic_info["max_level"] = this->route_graphs_.size();
    jsonify_basic_info[INDEX_PARAM] = this->create_param_ptr_->ToString();

    return jsonify_basic_info;
}

#define FROM_JSON(json_obj, var)             \
    do {                                     \
        if ((json_obj).contains(#var)) {     \
            this->var##_ = (json_obj)[#var]; \
        }                                    \
    } while (0)

#define FROM_JSON_BASE64(json_obj, var) base64_decode_obj((json_obj)[#var], this->var##_);

#define FROM_JSON_ATOMIC(json_obj, var) this->var##_.store((json_obj)[#var]);

void
HGraph::deserialize_basic_info(JsonType jsonify_basic_info) {
    logger::debug("jsonify_basic_info: {}", jsonify_basic_info.dump());
    FROM_JSON(jsonify_basic_info, use_reorder);
    FROM_JSON(jsonify_basic_info, dim);
    FROM_JSON(jsonify_basic_info, metric);
    FROM_JSON(jsonify_basic_info, entry_point_id);
    FROM_JSON(jsonify_basic_info, ef_construct);
    FROM_JSON(jsonify_basic_info, extra_info_size);
    FROM_JSON(jsonify_basic_info, data_type);
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    // logger::debug("mult: {}", this->mult_);
    FROM_JSON_ATOMIC(jsonify_basic_info, max_capacity);

    uint64_t max_level = jsonify_basic_info["max_level"];
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    if (jsonify_basic_info.contains(INDEX_PARAM)) {
        std::string index_param_string = jsonify_basic_info[INDEX_PARAM];
        HGraphParameterPtr index_param = std::make_shared<HGraphParameter>();
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("HGraph index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
}

void
HGraph::serialize_label_info(StreamWriter& writer) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Serialize(writer);
        return;
    }
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
HGraph::deserialize_label_info(StreamReader& reader) const {
    if (this->label_table_->CompressDuplicateData()) {
        this->label_table_->Deserialize(reader);
        return;
    }
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

void
HGraph::Serialize(StreamWriter& writer) const {
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }

    // FIXME(wxyu): only for testing, remove before merge into the main branch
    // if (not Options::Instance().new_version()) {
    //     this->serialize_basic_info_v0_14(writer);
    //     this->basic_flatten_codes_->Serialize(writer);
    //     this->bottom_graph_->Serialize(writer);
    //     if (this->use_reorder_) {
    //         this->high_precise_codes_->Serialize(writer);
    //     }
    //     for (const auto& route_graph : this->route_graphs_) {
    //         route_graph->Serialize(writer);
    //     }
    //     if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
    //         this->extra_infos_->Serialize(writer);
    //     }
    //     if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
    //         this->attr_filter_index_->Serialize(writer);
    //     }
    //     return;
    // }

    this->serialize_label_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->use_reorder_) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : this->route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    auto jsonify_basic_info = this->serialize_basic_info();
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    logger::debug(jsonify_basic_info.dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
HGraph::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        this->deserialize_basic_info_v0_14(buffer_reader);

        this->basic_flatten_codes_->Deserialize(buffer_reader);
        this->bottom_graph_->Deserialize(buffer_reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(buffer_reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        // metadata should NOT be nullptr if footer is not nullptr
        this->deserialize_basic_info(metadata->Get(BASIC_INFO));
        this->deserialize_label_info(buffer_reader);

        this->basic_flatten_codes_->Deserialize(buffer_reader);
        this->bottom_graph_->Deserialize(buffer_reader);
        if (this->use_reorder_) {
            this->high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(buffer_reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }
    }

    // post serialize procedure
    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

std::string
HGraph::GetMemoryUsageDetail() const {
    JsonType memory_usage;
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }
    memory_usage["basic_flatten_codes"] = this->basic_flatten_codes_->CalcSerializeSize();
    memory_usage["bottom_graph"] = this->bottom_graph_->CalcSerializeSize();
    if (this->use_reorder_) {
        memory_usage["high_precise_codes"] = this->high_precise_codes_->CalcSerializeSize();
    }
    size_t route_graph_size = 0;
    for (const auto& route_graph : this->route_graphs_) {
        route_graph_size += route_graph->CalcSerializeSize();
    }
    memory_usage["route_graph"] = route_graph_size;
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        memory_usage["extra_infos"] = this->extra_infos_->CalcSerializeSize();
    }
    memory_usage["__total_size__"] = this->CalSerializeSize();
    return memory_usage.dump();
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
        auto new_id = this->label_table_->GetIdByLabel(id);
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
    Vector<InnerIdType> inner_ids(count, 0, allocator_);
    Vector<InnerIdType> invalid_id_loc(allocator_);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        for (int64_t i = 0; i < count; ++i) {
            try {
                inner_ids[i] = this->label_table_->GetIdByLabel(ids[i]);
            } catch (std::runtime_error& e) {
                logger::debug(fmt::format("failed to find id: {}", ids[i]));
                invalid_id_loc.push_back(i);
            }
        }
        flat->Query(distances, computer, inner_ids.data(), count);
        for (unsigned int i : invalid_id_loc) {
            distances[i] = -1;
        }
    }
    return result;
}

std::pair<int64_t, int64_t>
HGraph::GetMinAndMaxId() const {
    int64_t min_id = INT64_MAX;
    int64_t max_id = INT64_MIN;
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    if (this->total_count_ == 0) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Label map size is zero");
    }
    for (int i = 0; i < this->total_count_; ++i) {
        if (not deleted_ids_.empty() && deleted_ids_.count(i) != 0) {
            continue;
        }
        auto label = this->label_table_->label_table_[i];
        max_id = std::max(label, max_id);
        min_id = std::min(label, min_id);
    }
    return {min_id, max_id};
}

void
HGraph::GetExtraInfoByIds(const int64_t* ids, int64_t count, char* extra_infos) const {
    if (this->extra_infos_ == nullptr) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "extra_info is NULL");
    }
    for (int64_t i = 0; i < count; ++i) {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        auto inner_id = this->label_table_->GetIdByLabel(ids[i]);
        this->extra_infos_->GetExtraInfoById(inner_id, extra_infos + i * extra_info_size_);
    }
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id) {
    this->basic_flatten_codes_->InsertVector(data, inner_id);
    if (use_reorder_) {
        this->high_precise_codes_->InsertVector(data, inner_id);
    }
    std::unique_lock add_lock(add_mutex_);
    if (level >= static_cast<int>(this->route_graphs_.size()) || bottom_graph_->TotalCount() == 0) {
        std::lock_guard<std::shared_mutex> wlock(this->global_mutex_);
        // level maybe a negative number(-1)
        for (auto j = static_cast<int>(this->route_graphs_.size()); j <= level; ++j) {
            this->route_graphs_.emplace_back(this->generate_one_route_graph());
        }
        auto insert_success = this->graph_add_one(data, level, inner_id);
        if (insert_success) {
            entry_point_id_ = inner_id;
        } else {
            this->route_graphs_.pop_back();
        }
        add_lock.unlock();
    } else {
        add_lock.unlock();
        std::shared_lock<std::shared_mutex> rlock(this->global_mutex_);
        this->graph_add_one(data, level, inner_id);
    }
}

bool
HGraph::graph_add_one(const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param{
        .topk = 1,
        .ep = this->entry_point_id_,
        .ef = 1,
        .is_inner_id_allowed = nullptr,
    };

    LockGuard cur_lock(neighbors_mutex_, inner_id);
    auto flatten_codes = basic_flatten_codes_;
    if (use_reorder_ and not build_by_base_) {
        flatten_codes = high_precise_codes_;
    }
    for (auto j = this->route_graphs_.size() - 1; j > level; --j) {
        result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
        param.ep = result->Top().second;
    }

    param.ef = this->ef_construct_;
    param.topk = static_cast<int64_t>(ef_construct_);

    if (bottom_graph_->TotalCount() != 0) {
        result = search_one_graph(data, this->bottom_graph_, flatten_codes, param);
        if (this->label_table_->CompressDuplicateData() && param.duplicate_id >= 0) {
            std::unique_lock lock(this->label_lookup_mutex_);
            label_table_->SetDuplicateId(static_cast<InnerIdType>(param.duplicate_id), inner_id);
            return false;
        }
        mutually_connect_new_element(
            inner_id, result, this->bottom_graph_, flatten_codes, neighbors_mutex_, allocator_);
    } else {
        bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    }

    for (int64_t j = 0; j <= level; ++j) {
        if (route_graphs_[j]->TotalCount() != 0) {
            result = search_one_graph(data, route_graphs_[j], flatten_codes, param);
            mutually_connect_new_element(
                inner_id, result, route_graphs_[j], flatten_codes, neighbors_mutex_, allocator_);
        } else {
            route_graphs_[j]->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
        }
    }
    return true;
}

void
HGraph::resize(uint64_t new_size) {
    auto cur_size = this->max_capacity_.load();
    uint64_t new_size_power_2 =
        next_multiple_of_power_of_two(new_size, this->resize_increase_count_bit_);
    if (cur_size >= new_size_power_2) {
        return;
    }
    std::lock_guard lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size_power_2, allocator_);
        this->label_table_->Resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);
        this->max_capacity_.store(new_size_power_2);
        this->basic_flatten_codes_->Resize(new_size_power_2);
        if (use_reorder_) {
            this->high_precise_codes_->Resize(new_size_power_2);
        }
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
    }
}
void
HGraph::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_BUILD_WITH_MULTI_THREAD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_MERGE_INDEX,
    });
    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
        IndexFeature::SUPPORT_KNN_ITERATOR_FILTER_SEARCH,
    });
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_CONCURRENT);
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
        IndexFeature::SUPPORT_CLONE,
        IndexFeature::SUPPORT_EXPORT_MODEL,
    });

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    bool have_fp32 = false;
    bool hold_molds = false;
    if (name == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->basic_flatten_codes_->HoldMolds();
    }
    if (use_reorder_ and not ignore_reorder_ and
        this->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        have_fp32 = true;
        hold_molds |= this->high_precise_codes_->HoldMolds();
    }
    if (have_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
        if (metric_ != MetricType::METRIC_TYPE_COSINE || hold_molds) {
            this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
        }
    }

    // metric
    if (metric_ == MetricType::METRIC_TYPE_IP) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_INNER_PRODUCT);
    } else if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_L2);
    } else if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_METRIC_TYPE_COSINE);
    }

    if (this->extra_infos_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_EXTRA_INFO_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_KNN_SEARCH_WITH_EX_FILTER);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT);
    }
}

void
HGraph::elp_optimize() {
    InnerSearchParam param;
    param.ep = 0;
    param.ef = 80;
    param.topk = 10;
    param.is_inner_id_allowed = nullptr;
    searcher_->SetMockParameters(bottom_graph_, basic_flatten_codes_, pool_, param, dim_);
    // TODO(ZXY): optimize PREFETCH_DEPTH_CODE and add default value for the others
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_CODE, 1, 10, 1));
    optimizer_->RegisterParameter(RuntimeParameter(PREFETCH_STRIDE_VISIT, 1, 10, 1));
    optimizer_->Optimize(searcher_);
}

void
HGraph::reorder(const void* query,
                const FlattenInterfacePtr& flatten,
                DistHeapPtr& candidate_heap,
                int64_t k) const {
    uint64_t size = candidate_heap->Size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    auto reorder_heap = Reorder::ReorderByFlatten(
        candidate_heap, flatten, static_cast<const float*>(query), allocator_, k);
    candidate_heap = reorder_heap;
}

static const std::string HGRAPH_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_HGRAPH}",
        "{HGRAPH_USE_REORDER_KEY}": false,
        "{HGRAPH_USE_ENV_OPTIMIZER}": false,
        "{HGRAPH_IGNORE_REORDER_KEY}": false,
        "{HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY}": false,
        "{HGRAPH_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{HGRAPH_GRAPH_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{GRAPH_TYPE_KEY}": "{GRAPH_TYPE_NSW}",
            "{GRAPH_STORAGE_TYPE_KEY}": "{GRAPH_STORAGE_TYPE_FLAT}",
            "{ODESCENT_PARAMETER_BUILD_BLOCK_SIZE}": 10000,
            "{ODESCENT_PARAMETER_MIN_IN_DEGREE}": 1,
            "{ODESCENT_PARAMETER_ALPHA}": 1.2,
            "{ODESCENT_PARAMETER_GRAPH_ITER_TURN}": 30,
            "{ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE}": 0.2,
            "{GRAPH_PARAM_MAX_DEGREE}": 64,
            "{GRAPH_PARAM_INIT_MAX_CAPACITY}": 100,
            "{GRAPH_SUPPORT_REMOVE}": false,
            "{REMOVE_FLAG_BIT}": 8
        },
        "{HGRAPH_BASE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY}": 32,
                "nbits": 8,
                "{PRODUCT_QUANTIZATION_DIM}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{HGRAPH_PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{PRODUCT_QUANTIZATION_DIM}": 1,
                "{HOLD_MOLDS}": false
            }
        },
        "{BUILD_PARAMS_KEY}": {
            "{BUILD_EF_CONSTRUCTION}": 400,
            "{BUILD_THREAD_COUNT}": 100
        },
        "{HGRAPH_EXTRA_INFO_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            }
        },
        "{HGRAPH_GET_RAW_VECTOR_COSINE}": false,
        "{HGRAPH_SUPPORT_DUPLICATE}": false
    })";

ParamPtr
HGraph::CheckAndMappingExternalParam(const JsonType& external_param,
                                     const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {{
                                                HGRAPH_USE_REORDER,
                                                {
                                                    HGRAPH_USE_REORDER_KEY,
                                                },
                                            },
                                            {
                                                HGRAPH_USE_ELP_OPTIMIZER,
                                                {
                                                    HGRAPH_USE_ELP_OPTIMIZER_KEY,
                                                },
                                            },
                                            {
                                                HGRAPH_IGNORE_REORDER,
                                                {
                                                    HGRAPH_IGNORE_REORDER_KEY,
                                                },
                                            },
                                            {
                                                HGRAPH_BUILD_BY_BASE_QUANTIZATION,
                                                {
                                                    HGRAPH_BUILD_BY_BASE_QUANTIZATION_KEY,
                                                },
                                            },
                                            {
                                                USE_ATTRIBUTE_FILTER,
                                                {
                                                    USE_ATTRIBUTE_FILTER_KEY,
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
                                                HGRAPH_STORE_RAW_VECTOR,
                                                {
                                                    HGRAPH_BASE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    HOLD_MOLDS,
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
                                                HGRAPH_STORE_RAW_VECTOR,
                                                {
                                                    HGRAPH_PRECISE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    HOLD_MOLDS,
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
                                                HGRAPH_GRAPH_TYPE,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    GRAPH_TYPE_KEY,
                                                },
                                            },
                                            {
                                                HGRAPH_GRAPH_STORAGE_TYPE,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    GRAPH_STORAGE_TYPE_KEY,
                                                },
                                            },
                                            {
                                                ODESCENT_PARAMETER_ALPHA,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    ODESCENT_PARAMETER_ALPHA,
                                                },
                                            },
                                            {
                                                ODESCENT_PARAMETER_GRAPH_ITER_TURN,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    ODESCENT_PARAMETER_GRAPH_ITER_TURN,
                                                },
                                            },
                                            {
                                                ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    ODESCENT_PARAMETER_NEIGHBOR_SAMPLE_RATE,
                                                },
                                            },
                                            {
                                                ODESCENT_PARAMETER_MIN_IN_DEGREE,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    ODESCENT_PARAMETER_MIN_IN_DEGREE,
                                                },
                                            },
                                            {
                                                ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
                                                {
                                                    HGRAPH_GRAPH_KEY,
                                                    ODESCENT_PARAMETER_BUILD_BLOCK_SIZE,
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
                                            {
                                                RABITQ_PCA_DIM,
                                                {
                                                    HGRAPH_BASE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    PCA_DIM,
                                                },
                                            },
                                            {
                                                RABITQ_BITS_PER_DIM_QUERY,
                                                {
                                                    HGRAPH_BASE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY,
                                                },
                                            },
                                            {
                                                HGRAPH_BASE_PQ_DIM,
                                                {
                                                    HGRAPH_BASE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    PRODUCT_QUANTIZATION_DIM,
                                                },
                                            },
                                            {
                                                RABITQ_USE_FHT,
                                                {
                                                    HGRAPH_BASE_CODES_KEY,
                                                    QUANTIZATION_PARAMS_KEY,
                                                    USE_FHT,
                                                },
                                            },
                                            {
                                                HGRAPH_SUPPORT_REMOVE,
                                                {HGRAPH_GRAPH_KEY, GRAPH_SUPPORT_REMOVE},
                                            },
                                            {
                                                HGRAPH_REMOVE_FLAG_BIT,
                                                {HGRAPH_GRAPH_KEY, REMOVE_FLAG_BIT},
                                            },
                                            {
                                                HGRAPH_SUPPORT_DUPLICATE,
                                                {
                                                    SUPPORT_DUPLICATE,
                                                },
                                            }};
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("HGraph not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(HGRAPH_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto hgraph_parameter = std::make_shared<HGraphParameter>();
    hgraph_parameter->data_type = common_param.data_type_;
    hgraph_parameter->FromJson(inner_json);
    uint64_t max_degree = hgraph_parameter->bottom_graph_param->max_degree_;

    auto max_degree_threshold = std::max(common_param.dim_, 128L);
    CHECK_ARGUMENT(  // NOLINT
        (4 <= max_degree) and (max_degree <= max_degree_threshold),
        fmt::format("max_degree({}) must in range[4, {}]", max_degree, max_degree_threshold));

    auto construction_threshold = std::max(1000UL, AMPLIFICATION_FACTOR * max_degree);
    CHECK_ARGUMENT((max_degree <= hgraph_parameter->ef_construction) and  // NOLINT
                       (hgraph_parameter->ef_construction <= construction_threshold),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), {}]",
                               hgraph_parameter->ef_construction,
                               max_degree,
                               construction_threshold));
    return hgraph_parameter;
}
InnerIndexPtr
HGraph::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<HGraph>(this->create_param_ptr_, param);
    this->basic_flatten_codes_->ExportModel(index->basic_flatten_codes_);
    if (use_reorder_) {
        this->high_precise_codes_->ExportModel(index->high_precise_codes_);
    }
    return index;
}
void
HGraph::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    if (use_reorder_) {
        high_precise_codes_->GetCodesById(inner_id, data);
    } else {
        basic_flatten_codes_->GetCodesById(inner_id, data);
    }
}

bool
HGraph::Remove(int64_t id) {
    // TODO(inbao): support thread safe remove
    auto inner_id = this->label_table_->GetIdByLabel(id);
    if (inner_id == this->entry_point_id_) {
        bool find_new_ep = false;
        while (not route_graphs_.empty()) {
            auto& upper_graph = route_graphs_.back();
            Vector<InnerIdType> neighbors(allocator_);
            upper_graph->GetNeighbors(this->entry_point_id_, neighbors);
            for (const auto& nb_id : neighbors) {
                if (inner_id == nb_id) {
                    continue;
                }
                this->entry_point_id_ = nb_id;
                find_new_ep = true;
                break;
            }
            if (find_new_ep) {
                break;
            }
            route_graphs_.pop_back();
        }
    }
    for (int level = static_cast<int>(route_graphs_.size()) - 1; level >= 0; --level) {
        this->route_graphs_[level]->DeleteNeighborsById(inner_id);
    }
    this->bottom_graph_->DeleteNeighborsById(inner_id);
    this->label_table_->Remove(id);
    this->deleted_ids_.insert(inner_id);
    delete_count_++;
    return true;
}

void
HGraph::Merge(const std::vector<MergeUnit>& merge_units) {
    int64_t total_count = this->GetNumElements();
    for (const auto& unit : merge_units) {
        total_count += unit.index->GetNumElements();
    }
    if (max_capacity_ < total_count) {
        this->resize(total_count);
    }
    for (const auto& merge_unit : merge_units) {
        const auto other_index = std::dynamic_pointer_cast<HGraph>(
            std::dynamic_pointer_cast<IndexImpl<HGraph>>(merge_unit.index)->GetInnerIndex());
        if (total_count_ == 0) {
            this->entry_point_id_ = other_index->entry_point_id_;
        }
        basic_flatten_codes_->MergeOther(other_index->basic_flatten_codes_, this->total_count_);
        label_table_->MergeOther(other_index->label_table_, merge_unit.id_map_func);
        if (use_reorder_) {
            high_precise_codes_->MergeOther(other_index->high_precise_codes_, this->total_count_);
        }
        bottom_graph_->MergeOther(other_index->bottom_graph_, this->total_count_);
        if (route_graphs_.size() < other_index->route_graphs_.size()) {
            route_graphs_.push_back(this->generate_one_route_graph());
        }
        for (int j = 0; j < other_index->route_graphs_.size(); ++j) {
            route_graphs_[j]->MergeOther(other_index->route_graphs_[j], this->total_count_);
        }
        this->total_count_ += other_index->GetNumElements();
    }
    if (this->odescent_param_ == nullptr) {
        odescent_param_ = std::make_shared<ODescentParameter>();
    }

    auto build_data = (use_reorder_ and not build_by_base_) ? this->high_precise_codes_
                                                            : this->basic_flatten_codes_;
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(odescent_param_, build_data, allocator_, this->build_pool_.get());
        odescent_builder.Build(bottom_graph_);
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& graph : route_graphs_) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->build_pool_.get());
        auto ids = graph->GetIds();
        sparse_odescent_builder.Build(ids, graph);
        sparse_odescent_builder.SaveGraph(graph);
        this->entry_point_id_ = ids.back();
    }
}
bool
HGraph::UpdateExtraInfo(const DatasetPtr& new_base) {
    CHECK_ARGUMENT(new_base != nullptr, "new_base is nullptr");
    CHECK_ARGUMENT(new_base->GetExtraInfos() != nullptr, "extra_infos is nullptr");
    CHECK_ARGUMENT(new_base->GetExtraInfoSize() == extra_info_size_, "extra_infos size mismatch");
    CHECK_ARGUMENT(new_base->GetNumElements() == 1, "new_base size must be one");
    auto label = new_base->GetIds()[0];
    if (this->extra_infos_ != nullptr) {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        if (not this->label_table_->CheckLabel(label)) {
            return false;
        }
        const auto inner_id = this->label_table_->GetIdByLabel(label);
        this->extra_infos_->InsertExtraInfo(new_base->GetExtraInfos(), inner_id);
        return true;
    }
    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION, "extra_infos is not initialized");
}
void
HGraph::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    auto codes = (use_reorder_) ? high_precise_codes_ : basic_flatten_codes_;
    Vector<uint8_t> buffer(codes->code_size_, allocator_);
    codes->GetCodesById(inner_id, buffer.data());
    codes->Decode(buffer.data(), data);
}

void
HGraph::SetImmutable() {
    if (this->immutable_) {
        return;
    }
    std::lock_guard<std::shared_mutex> wlock(this->global_mutex_);
    this->neighbors_mutex_.reset();
    this->neighbors_mutex_ = std::make_shared<EmptyMutex>();
    this->searcher_->SetMutexArray(this->neighbors_mutex_);
    this->immutable_ = true;
}

void
HGraph::SetIO(const std::shared_ptr<Reader> reader) {
    if (use_reorder_) {
        auto reader_param = std::make_shared<ReaderIOParameter>();
        reader_param->reader = reader;
        high_precise_codes_->InitIO(reader_param);
    }
}

[[nodiscard]] DatasetPtr
HGraph::SearchWithRequest(const SearchRequest& request) const {
    const auto& query = request.query_;
    int64_t query_dim = query->GetDim();
    Allocator* search_allocator = this->allocator_;
    if (request.search_allocator_ != nullptr) {
        search_allocator = request.search_allocator_;
    }
    auto k = request.topk_;
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));
    }

    auto params = HGraphSearchParameters::FromJson(request.params_str_);

    auto ef_search_threshold = std::max(AMPLIFICATION_FACTOR * k, 1000L);
    CHECK_ARGUMENT(  // NOLINT
        (1 <= params.ef_search) and (params.ef_search <= ef_search_threshold),
        fmt::format("ef_search({}) must in range[1, {}]", params.ef_search, ef_search_threshold));

    // check k
    CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k));
    k = std::min(k, GetNumElements());

    // check query vector
    CHECK_ARGUMENT(query->GetNumElements() == 1, "query dataset should contain 1 vector only");

    InnerSearchParam search_param;
    search_param.ep = this->entry_point_id_;
    search_param.topk = 1;
    search_param.ef = 1;
    search_param.is_inner_id_allowed = nullptr;
    search_param.search_alloc = search_allocator;
    const auto* raw_query = get_data(query);
    for (auto i = static_cast<int64_t>(this->route_graphs_.size() - 1); i >= 0; --i) {
        auto result = this->search_one_graph(
            raw_query, this->route_graphs_[i], this->basic_flatten_codes_, search_param);
        search_param.ep = result->Top().second;
    }

    FilterPtr ft = nullptr;
    if (request.filter_ != nullptr) {
        if (params.use_extra_info_filter) {
            ft = std::make_shared<ExtraInfoWrapperFilter>(request.filter_, this->extra_infos_);
        } else {
            ft = std::make_shared<InnerIdWrapperFilter>(request.filter_, *this->label_table_);
        }
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        executor->Init();
        search_param.executors.emplace_back(executor);
    }

    search_param.ef = std::max(params.ef_search, k);
    search_param.is_inner_id_allowed = ft;
    search_param.topk = static_cast<int64_t>(search_param.ef);
    search_param.consider_duplicate = true;
    if (params.enable_time_record) {
        search_param.time_cost = std::make_shared<Timer>();
        search_param.time_cost->SetThreshold(params.timeout_ms);
    }
    auto search_result = this->search_one_graph(
        raw_query, this->bottom_graph_, this->basic_flatten_codes_, search_param);

    if (use_reorder_) {
        this->reorder(raw_query, this->high_precise_codes_, search_result, k);
    }

    while (search_result->Size() > k) {
        search_result->Pop();
    }

    // return an empty dataset directly if searcher returns nothing
    if (search_result->Empty()) {
        return DatasetImpl::MakeEmptyDataset();
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, ids] = create_fast_dataset(count, search_allocator);
    char* extra_infos = nullptr;
    if (extra_info_size_ > 0) {
        extra_infos = (char*)search_allocator->Allocate(extra_info_size_ * search_result->Size());
        dataset_results->ExtraInfos(extra_infos);
    }
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        ids[j] = this->label_table_->GetLabelById(search_result->Top().second);
        if (extra_infos != nullptr) {
            this->extra_infos_->GetExtraInfoById(search_result->Top().second,
                                                 extra_infos + extra_info_size_ * j);
        }
        search_result->Pop();
    }
    return std::move(dataset_results);
}

void
HGraph::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0);
}

void
HGraph::UpdateAttribute(int64_t id,
                        const AttributeSet& new_attrs,
                        const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, inner_id, 0, origin_attrs);
}

const static uint64_t QUERY_SAMPLE_SIZE = 100;
const static int64_t DEFAULT_TOPK = 100;

std::string
HGraph::GetStats() const {
    JsonType stats;
    int64_t topk = DEFAULT_TOPK;
    uint64_t sample_size = std::min(QUERY_SAMPLE_SIZE, this->total_count_);
    Vector<float> sample_base_datas(dim_ * sample_size, 0.0F, allocator_);
    if (this->total_count_ == 0) {
        stats["total_count"] = 0;
        return stats.dump();
    }
    // TODO(inabao): configure in next pr
    std::string search_params = R"({
        "hgraph": {
            "ef_search": 200
        }
    })";
    stats["total_count"] = this->total_count_;
    // duplicate rate
    size_t duplicate_num = 0;
    if (this->label_table_->CompressDuplicateData()) {
        for (int i = 0; i < this->total_count_; ++i) {
            if (this->label_table_->duplicate_records_[i] != nullptr) {
                duplicate_num += this->label_table_->duplicate_records_[i]->duplicate_ids.size();
            }
        }
    }
    stats["duplicate_rate"] =
        static_cast<float>(duplicate_num) / static_cast<float>(this->total_count_);
    stats["deleted_count"] = delete_count_.load();
    this->analyze_graph_connection(stats);
    this->analyze_graph_recall(stats, sample_base_datas, sample_size, topk, search_params);
    this->analyze_quantizer(stats, sample_base_datas, sample_size, topk, search_params);
    return stats.dump(4);
}

void
HGraph::analyze_quantizer(JsonType& stats,
                          const Vector<float>& data,
                          uint64_t sample_data_size,
                          int64_t topk,
                          const std::string& search_param) const {
    // record quantized information
    if (this->use_reorder_) {
        logger::info("analyze_quantizer: sample_data_size = {}, topk = {}", sample_data_size, topk);
        float bias_ratio = 0.0F;
        float inversion_count_rate = 0.0F;
        for (uint64_t i = 0; i < sample_data_size; ++i) {
            float tmp_bias_ratio = 0.0F;
            float tmp_inversion_count_rate = 0.0F;
            this->use_reorder_ = false;
            const auto* query_data = data.data() + i * dim_;
            auto query = Dataset::Make();
            query->Owner(false)->NumElements(1)->Float32Vectors(query_data)->Dim(dim_);
            auto search_result = this->KnnSearch(query, topk, search_param, nullptr);
            this->use_reorder_ = true;
            auto distance_result =
                this->CalDistanceById(query_data, search_result->GetIds(), search_result->GetDim());
            const auto* ground_distances = distance_result->GetDistances();
            const auto* approximate_distances = search_result->GetDistances();
            for (int64_t j = 0; j < topk; ++j) {
                if (ground_distances[j] > 0) {
                    tmp_bias_ratio += std::abs(approximate_distances[j] - ground_distances[j]) /
                                      ground_distances[j];
                }
            }
            tmp_bias_ratio /= static_cast<float>(topk);
            bias_ratio += tmp_bias_ratio;
            // calculate inversion count rate
            int64_t inversion_count = 0;
            for (int64_t j = 0; j < search_result->GetDim() - 1; ++j) {
                for (int64_t k = j + 1; k < search_result->GetDim(); ++k) {
                    if (ground_distances[j] > ground_distances[k]) {
                        inversion_count++;
                    }
                }
            }
            int64_t search_count = search_result->GetDim();
            tmp_inversion_count_rate =
                static_cast<float>(inversion_count) /
                (static_cast<float>(search_count * (search_count - 1)) / 2.0F);
            inversion_count_rate += tmp_inversion_count_rate;
        }
        stats["quantization_bias_ratio"] = bias_ratio / static_cast<float>(sample_data_size);
        stats["quantization_inversion_count_rate"] =
            inversion_count_rate / static_cast<float>(sample_data_size);
    }
}

void
HGraph::analyze_graph_recall(JsonType& stats,
                             Vector<float>& data,
                             uint64_t sample_data_size,
                             int64_t topk,
                             const std::string& search_param) const {
    // recall of "base" when searching for "base"
    logger::info("analyze_graph_recall: sample_data_size = {}, topk = {}", sample_data_size, topk);
    auto codes = this->use_reorder_ ? this->high_precise_codes_ : this->basic_flatten_codes_;
    int64_t hit_count = 0;
    size_t all_neighbor_count = 0;
    int64_t hit_neighbor_count = 0;
    float avg_distance_base = 0.0F;
    for (uint64_t i = 0; i < sample_data_size; ++i) {
        InnerIdType sample_id = rand() % this->total_count_;
        GetVectorByInnerId(sample_id, data.data() + i * dim_);
        // generate groundtruth
        DistHeapPtr groundtruth = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        if (i % 10 == 0) {
            logger::info("calculate groundtruth for sample {} of {}", i, i + 10);
        }
        for (uint64_t j = 0; j < this->total_count_; ++j) {
            float dist = codes->ComputePairVectors(sample_id, j);
            if (groundtruth->Size() < topk) {
                groundtruth->Push({dist, j});
            } else if (dist < groundtruth->Top().first) {
                groundtruth->Pop();
                groundtruth->Push({dist, j});
            }
        }
        // neighbors of a point and the proximity relationship of a point
        Vector<InnerIdType> neighbors(allocator_);
        this->bottom_graph_->GetNeighbors(sample_id, neighbors);
        size_t neighbor_size = neighbors.size();
        UnorderedSet<LabelType> groundtruth_ids(allocator_);
        UnorderedSet<LabelType> neighbor_groundtruth_ids(allocator_);
        while (not groundtruth->Empty()) {
            auto id = groundtruth->Top().second;
            groundtruth_ids.insert(this->label_table_->GetLabelById(id));
            if (groundtruth->Size() <= neighbor_size) {
                neighbor_groundtruth_ids.insert(this->label_table_->GetLabelById(id));
            }
            avg_distance_base += groundtruth->Top().first;
            groundtruth->Pop();
        }
        all_neighbor_count += neighbor_size;
        for (const auto& id : neighbors) {
            if (neighbor_groundtruth_ids.count(this->label_table_->GetLabelById(id)) > 0) {
                hit_neighbor_count++;
            }
        }

        // search
        auto query = Dataset::Make();
        query->Owner(false)->NumElements(1)->Float32Vectors(data.data() + i * dim_)->Dim(dim_);
        auto result = this->KnnSearch(query, topk, search_param, nullptr);
        // calculate recall
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            auto id = result->GetIds()[j];
            if (groundtruth_ids.count(id) > 0) {
                hit_count++;
            }
        }
    }
    stats["recall_base"] =
        static_cast<float>(hit_count) / static_cast<float>(sample_data_size * topk);
    stats["proximity_recall_neighbor"] =
        static_cast<float>(hit_neighbor_count) / static_cast<float>(all_neighbor_count);
    stats["avg_distance_base"] =
        avg_distance_base / static_cast<float>(sample_data_size * (topk - 1));
}

void
HGraph::analyze_graph_connection(JsonType& stats) const {
    // graph connection
    Vector<bool> visited(total_count_, false, allocator_);
    int64_t connect_components = 0;
    if (this->label_table_->CompressDuplicateData()) {
        for (int i = 0; i < this->total_count_; ++i) {
            if (this->label_table_->duplicate_records_[i] != nullptr) {
                for (const auto& dup_id :
                     this->label_table_->duplicate_records_[i]->duplicate_ids) {
                    visited[dup_id] = true;
                }
            }
        }
    }
    for (int64_t i = 0; i < total_count_; ++i) {
        if (not visited[i] and (deleted_ids_.count(i) == 0)) {
            connect_components++;
            int64_t component_size = 0;
            std::queue<int64_t> q;
            q.push(i);
            visited[i] = true;
            while (not q.empty()) {
                auto node = q.front();
                q.pop();
                component_size++;
                Vector<InnerIdType> neighbors(allocator_);
                this->bottom_graph_->GetNeighbors(node, neighbors);
                for (const auto& nb : neighbors) {
                    if (not visited[nb] and (deleted_ids_.count(nb) == 0)) {
                        visited[nb] = true;
                        q.push(nb);
                    }
                }
            }
        }
    }
    stats["connect_components"] = connect_components;
}

}  // namespace vsag
