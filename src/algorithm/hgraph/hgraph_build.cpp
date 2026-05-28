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

#include <fmt/format.h>

#include "datacell/flatten_datacell_parameter.h"
#include "dataset_impl.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "impl/searcher/basic_searcher.h"
#include "io/memory_io_parameter.h"
#include "quantization/scalar_quantization/scalar_quantizer_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/util_functions.h"

namespace vsag {

static FlattenInterfacePtr
make_temporary_sq8_flatten(MetricType metric,
                           DataTypes data_type,
                           int64_t dim,
                           int64_t extra_info_size,
                           const std::shared_ptr<SafeThreadPool>& thread_pool,
                           Allocator* allocator) {
    auto sq8_param = std::make_shared<FlattenDataCellParameter>();
    sq8_param->quantizer_parameter = std::make_shared<ScalarQuantizerParameter<8>>();
    sq8_param->io_parameter = std::make_shared<MemoryIOParameter>();

    IndexCommonParam common_param;
    common_param.metric_ = metric;
    common_param.data_type_ = data_type;
    common_param.dim_ = dim;
    common_param.extra_info_size_ = extra_info_size;
    common_param.thread_pool_ = thread_pool;
    common_param.allocator_ = std::shared_ptr<Allocator>(allocator, [](Allocator*) {});
    return FlattenInterface::MakeInstance(sq8_param, common_param);
}

static bool
need_temporary_sq8_build_data(const FlattenInterfacePtr& basic_flatten_codes,
                              bool has_precise_reorder) {
    return not has_precise_reorder and
           basic_flatten_codes->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_RABITQ;
}

void
HGraph::Train(const DatasetPtr& base) {
    int64_t total_elements = base->GetNumElements();
    int64_t dim = base->GetDim();
    DatasetPtr train_data =
        vsag::sample_train_data(base, total_elements, dim, train_sample_count_, allocator_);

    const auto* data_ptr = get_data(train_data);
    this->basic_flatten_codes_->Train(data_ptr, train_data->GetNumElements());
    if (has_precise_reorder()) {
        this->high_precise_codes_->Train(data_ptr, train_data->GetNumElements());
    }
    if (create_new_raw_vector_) {
        // nothing to do since raw_vector_ is fp32
        this->raw_vector_->Train(data_ptr, train_data->GetNumElements());
    }
}

std::vector<int64_t>
HGraph::Build(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    this->Train(data);
    std::vector<int64_t> ret;
    if (graph_type_ == GRAPH_TYPE_VALUE_NSW) {
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
    Vector<int64_t> valid_indices(allocator_);
    UnorderedSet<LabelType> seen_labels(allocator_);
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label) or seen_labels.find(label) != seen_labels.end()) {
            failed_ids.emplace_back(label);
            continue;
        }
        seen_labels.insert(label);
        valid_indices.emplace_back(i);
    }
    auto inner_ids = this->get_unique_inner_ids(static_cast<InnerIdType>(valid_indices.size()));
    auto current_count = total_count_.load();
    uint64_t new_ids_count = 0;
    for (auto inner_id : inner_ids) {
        if (inner_id >= current_count) {
            ++new_ids_count;
        }
    }
    this->resize(current_count + new_ids_count);
    this->total_count_ += new_ids_count;
    Vector<Vector<InnerIdType>> route_graph_ids(allocator_);
    auto need_sq8_build_data =
        need_temporary_sq8_build_data(this->basic_flatten_codes_, this->has_precise_reorder());
    FlattenInterfacePtr temporary_sq8_build_data = nullptr;
    if (need_sq8_build_data and raw_vector_ == nullptr) {
        temporary_sq8_build_data =
            make_temporary_sq8_flatten(this->metric_,
                                       this->data_type_,
                                       this->dim_,
                                       static_cast<int64_t>(this->extra_info_size_),
                                       this->thread_pool_,
                                       this->allocator_);
        temporary_sq8_build_data->Train(vectors, total);
    }
    bool defer_persistent_codes = temporary_sq8_build_data != nullptr;
    if (not defer_persistent_codes) {
        this->Train(data);
    }
    Vector<std::pair<InnerIdType, int64_t>> deferred_code_ids(allocator_);
    for (InnerIdType cur_size = 0; cur_size < valid_indices.size(); ++cur_size) {
        auto i = valid_indices[cur_size];
        auto label = labels[i];
        InnerIdType inner_id = inner_ids.at(cur_size);
        this->label_table_->Insert(inner_id, label);
        if (not defer_persistent_codes) {
            this->insert_persistent_codes(vectors + dim_ * i, inner_id);
        } else {
            deferred_code_ids.emplace_back(inner_id, i);
        }
        if (temporary_sq8_build_data != nullptr) {
            temporary_sq8_build_data->InsertVector(vectors + dim_ * i, inner_id);
        }
        auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(route_graph_ids.size()) || route_graph_ids.empty()) {
                for (auto k = static_cast<int>(route_graph_ids.size()); k <= level; ++k) {
                    route_graph_ids.emplace_back(allocator_);
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    auto build_data = (has_precise_reorder() and not build_by_base_) ? this->high_precise_codes_
                                                                     : this->basic_flatten_codes_;
    if (need_sq8_build_data) {
        build_data = raw_vector_ != nullptr ? raw_vector_ : temporary_sq8_build_data;
    }
    {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree();
        ODescent odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        odescent_builder.Build();
        odescent_builder.SaveGraph(bottom_graph_);
    }
    for (auto& route_graph_id : route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    if (defer_persistent_codes) {
        build_data.reset();
        temporary_sq8_build_data.reset();
        this->Train(data);
        for (const auto& [inner_id, local_idx] : deferred_code_ids) {
            this->insert_persistent_codes(vectors + dim_ * local_idx, inner_id);
        }
    }
    return failed_ids;
}

std::vector<int64_t>
HGraph::Add(const DatasetPtr& data, AddMode mode) {
    std::shared_lock force_remove_rlock(this->force_remove_mutex_);
    std::vector<int64_t> failed_ids;
    auto base_dim = data->GetDim();
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));
    }
    CHECK_ARGUMENT(get_data(data) != nullptr, "base.float_vector is nullptr");

    auto need_sq8_build_data =
        need_temporary_sq8_build_data(this->basic_flatten_codes_, this->has_precise_reorder());
    CHECK_ARGUMENT(not(need_sq8_build_data and this->total_count_ != 0 and
                       raw_vector_ == nullptr and temporary_build_flatten_codes_ == nullptr),
                   "adding to a non-empty HGraph that needs temporary SQ8 build data requires "
                   "raw vectors");
    bool created_temporary_build_data = false;
    if (need_sq8_build_data and this->total_count_ == 0 and raw_vector_ == nullptr and
        temporary_build_flatten_codes_ == nullptr) {
        temporary_build_flatten_codes_ =
            make_temporary_sq8_flatten(this->metric_,
                                       this->data_type_,
                                       this->dim_,
                                       static_cast<int64_t>(this->extra_info_size_),
                                       this->thread_pool_,
                                       this->allocator_);
        temporary_build_flatten_codes_->Train(get_data(data), data->GetNumElements());
        created_temporary_build_data = true;
    }
    struct temporary_build_flatten_guard {
        HGraph* hgraph;
        bool enabled;
        ~temporary_build_flatten_guard() {
            if (enabled) {
                hgraph->temporary_build_flatten_codes_.reset();
            }
        }
    } temporary_build_flatten_guard_instance{this, created_temporary_build_data};
    bool defer_persistent_codes = created_temporary_build_data;

    {
        std::scoped_lock lock(this->add_mutex_);
        if (this->total_count_ == 0 and not defer_persistent_codes) {
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
        this->add_one_point(data, level, inner_id, not defer_persistent_codes);
    };

    std::vector<std::future<void>> futures;
    auto total = data->GetNumElements();
    const auto* labels = data->GetIds();
    const auto* source_id = data->GetSourceID();
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    bool use_parallel_add = this->thread_pool_ != nullptr;
    Vector<std::pair<InnerIdType, LabelType>> inner_ids(allocator_);
    for (int64_t j = 0; j < total; ++j) {
        InnerIdType inner_id;

        // try recover tombstone
        if (this->data_type_ != DataTypes::DATA_TYPE_SPARSE) {
            auto one_base = get_single_dataset(data, j);
            bool is_process_finished = try_recover_tombstone(one_base, failed_ids);
            if (is_process_finished) {
                continue;
            }
        }

        {
            std::scoped_lock lock(this->add_mutex_);
            inner_id = this->get_unique_inner_ids(1).at(0);
            if (inner_id >= total_count_) {
                this->resize(total_count_.load() + 1);
                ++total_count_;
            }
        }

        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            this->label_table_->Insert(inner_id, labels[j]);
            if (source_id != nullptr) {
                this->label_table_->InsertSourceId(inner_id, *source_id);
            }
            inner_ids.emplace_back(inner_id, j);
        }
    }
    if (temporary_build_flatten_codes_ != nullptr) {
        for (const auto& [inner_id, local_idx] : inner_ids) {
            temporary_build_flatten_codes_->InsertVector(get_data(data, local_idx), inner_id);
        }
    }
    for (auto& [inner_id, local_idx] : inner_ids) {
        int level;
        {
            std::scoped_lock label_lock(this->label_lookup_mutex_);
            level = this->get_random_level() - 1;
        }
        const auto* extra_info = extra_infos + local_idx * extra_info_size_;
        const AttributeSet* cur_attr_set = nullptr;
        if (attr_sets != nullptr) {
            cur_attr_set = attr_sets + local_idx;
        }
        if (use_parallel_add) {
            auto future = this->thread_pool_->GeneralEnqueue(
                add_func, get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
            futures.emplace_back(std::move(future));
        } else {
            add_func(get_data(data, local_idx), level, inner_id, extra_info, cur_attr_set);
        }
    }
    if (use_parallel_add) {
        for (auto& future : futures) {
            future.get();
        }
    }
    if (defer_persistent_codes) {
        temporary_build_flatten_codes_.reset();
        {
            std::scoped_lock lock(this->add_mutex_);
            this->Train(data);
        }
        futures.clear();
        for (const auto& id_pair : inner_ids) {
            auto inner_id = id_pair.first;
            auto local_idx = id_pair.second;
            if (use_parallel_add) {
                auto future =
                    this->thread_pool_->GeneralEnqueue([this, data, inner_id, local_idx]() {
                        this->insert_persistent_codes(get_data(data, local_idx), inner_id);
                    });
                futures.emplace_back(std::move(future));
            } else {
                this->insert_persistent_codes(get_data(data, local_idx), inner_id);
            }
        }
        if (use_parallel_add) {
            for (auto& future : futures) {
                future.get();
            }
        }
    }
    return failed_ids;
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id) {
    this->add_one_point(data, level, inner_id, true);
}

void
HGraph::insert_persistent_codes(const void* data, InnerIdType inner_id) {
    std::shared_lock add_lock(add_mutex_);
    this->basic_flatten_codes_->InsertVector(data, inner_id);
    if (has_precise_reorder()) {
        this->high_precise_codes_->InsertVector(data, inner_id);
    }
    if (create_new_raw_vector_) {
        raw_vector_->InsertVector(data, inner_id);
    }
}

void
HGraph::add_one_point(const void* data, int level, InnerIdType inner_id, bool insert_codes) {
    if (insert_codes) {
        this->insert_persistent_codes(data, inner_id);
    }
    std::unique_lock add_lock(add_mutex_);
    if (level >= static_cast<int>(this->route_graphs_.size()) || bottom_graph_->TotalCount() == 0) {
        std::scoped_lock<std::shared_mutex> wlock(this->global_mutex_);
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
        std::shared_lock rlock(this->global_mutex_);
        this->graph_add_one(data, level, inner_id);
    }
}

bool
HGraph::graph_add_one(const void* data, int level, InnerIdType inner_id) {
    DistHeapPtr result = nullptr;
    InnerSearchParam param;
    param.topk = 1;
    param.ep = this->entry_point_id_;
    param.ef = 1;
    param.is_inner_id_allowed = nullptr;

    auto flatten_codes = basic_flatten_codes_;
    if (temporary_build_flatten_codes_ != nullptr) {
        flatten_codes = temporary_build_flatten_codes_;
    } else if (need_temporary_sq8_build_data(this->basic_flatten_codes_,
                                             this->has_precise_reorder()) and
               raw_vector_ != nullptr) {
        flatten_codes = raw_vector_;
    } else if (has_precise_reorder() and not build_by_base_) {
        flatten_codes = high_precise_codes_;
    }

    for (auto j = this->route_graphs_.size() - 1; j > level; --j) {
        result = search_one_graph(
            data, route_graphs_[j], flatten_codes, param, (VisitedListPtr) nullptr, nullptr);
        param.ep = result->Top().second;
    }

    param.ef = this->ef_construct_;
    param.topk = static_cast<int64_t>(ef_construct_);
    if (this->support_duplicate_) {
        param.find_duplicate = true;
        param.duplicate_query_id = inner_id;
        param.duplicate_distance_threshold = this->duplicate_distance_threshold_;
    }

    if (bottom_graph_->TotalCount() != 0) {
        result = search_one_graph(data,
                                  this->bottom_graph_,
                                  flatten_codes,
                                  param,
                                  // to specify which overloaded function to call
                                  (VisitedListPtr) nullptr,
                                  nullptr);
        if (this->support_duplicate_ && param.duplicate_id >= 0) {
            std::unique_lock lock(this->label_lookup_mutex_);
            bottom_graph_->SetDuplicateId(static_cast<InnerIdType>(param.duplicate_id), inner_id);
            return false;
        }
        auto filtered_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
        while (not result->Empty()) {
            auto [dist, id] = result->Top();
            result->Pop();
            if (id != inner_id) {
                filtered_result->Push(dist, id);
            }
        }
        LockGuard cur_lock(neighbors_mutex_, inner_id);
        mutually_connect_new_element(inner_id,
                                     filtered_result,
                                     this->bottom_graph_,
                                     flatten_codes,
                                     neighbors_mutex_,
                                     allocator_,
                                     alpha_);
    } else {
        LockGuard cur_lock(neighbors_mutex_, inner_id);
        bottom_graph_->InsertNeighborsById(inner_id, Vector<InnerIdType>(allocator_));
    }

    for (int64_t j = 0; j <= level; ++j) {
        if (route_graphs_[j]->TotalCount() != 0) {
            result = search_one_graph(data,
                                      route_graphs_[j],
                                      flatten_codes,
                                      param,
                                      // to specify which overloaded function to call
                                      (VisitedListPtr) nullptr,
                                      nullptr);
            auto filtered_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
            while (not result->Empty()) {
                auto [dist, id] = result->Top();
                result->Pop();
                if (id != inner_id) {
                    filtered_result->Push(dist, id);
                }
            }
            LockGuard cur_lock(neighbors_mutex_, inner_id);
            mutually_connect_new_element(inner_id,
                                         filtered_result,
                                         route_graphs_[j],
                                         flatten_codes,
                                         neighbors_mutex_,
                                         allocator_,
                                         alpha_);
        } else {
            LockGuard cur_lock(neighbors_mutex_, inner_id);
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
    std::scoped_lock lock(this->global_mutex_);
    cur_size = this->max_capacity_.load();
    if (cur_size < new_size_power_2) {
        this->neighbors_mutex_->Resize(new_size_power_2);
        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size_power_2, allocator_);
        this->label_table_->Resize(new_size_power_2);
        bottom_graph_->Resize(new_size_power_2);
        this->basic_flatten_codes_->Resize(new_size_power_2);
        if (has_precise_reorder()) {
            this->high_precise_codes_->Resize(new_size_power_2);
        }
        if (create_new_raw_vector_) {
            this->raw_vector_->Resize(new_size_power_2);
        }
        if (this->extra_infos_ != nullptr) {
            this->extra_infos_->Resize(new_size_power_2);
        }
        this->max_capacity_.store(new_size_power_2);
        this->cal_memory_usage();
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
    // update
    if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
        this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_VECTOR_CONCURRENT});
    }
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_UPDATE_ID_CONCURRENT});
    // concurrency
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_CONCURRENT);
    this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_SEARCH_DELETE_CONCURRENT);
    // serialize
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_DESERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_DESERIALIZE_FILE,
        IndexFeature::SUPPORT_DESERIALIZE_READER_SET,
        IndexFeature::SUPPORT_SERIALIZE_BINARY_SET,
        IndexFeature::SUPPORT_SERIALIZE_FILE,
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });
    // other
    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_ESTIMATE_MEMORY,
                                            IndexFeature::SUPPORT_GET_MEMORY_USAGE,
                                            IndexFeature::SUPPORT_CHECK_ID_EXIST,
                                            IndexFeature::SUPPORT_CLONE,
                                            IndexFeature::SUPPORT_EXPORT_MODEL,
                                            IndexFeature::SUPPORT_TUNE});

    // About Train
    auto name = this->basic_flatten_codes_->GetQuantizerName();

    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16 and
        name != QUANTIZATION_TYPE_VALUE_FP16) {
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
    if (has_precise_reorder() and not ignore_reorder_ and
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

    if (raw_vector_ != nullptr) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_RAW_VECTOR_BY_IDS);
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
                int64_t k,
                IteratorFilterContext* iter_ctx,
                QueryContext& ctx,
                const DistanceRecordVector* rabitq_lower_bound_candidates) const {
    uint64_t size = candidate_heap->Size();
    if (k <= 0) {
        k = static_cast<int64_t>(size);
    }
    auto reorder_impl = reorder_;
    if (reorder_impl == nullptr) {
        reorder_impl = std::make_shared<FlattenReorder>(flatten, allocator_);
    }
    auto reorder_heap = reorder_impl->Reorder(candidate_heap,
                                              static_cast<const float*>(query),
                                              k,
                                              ctx,
                                              iter_ctx,
                                              rabitq_lower_bound_candidates);
    candidate_heap = reorder_heap;
}

void
HGraph::ExportCache(std::ostream& out_stream) {
    IOStreamWriter writer(out_stream);
    this->fullfill_cache();
    this->cache_->Serialize(writer);
}

void
HGraph::ImportCache(std::istream& in_stream) {
    IOStreamReader reader(in_stream);
    this->cache_->Deserialize(reader);
}

void
HGraph::fullfill_cache() {
    auto& source_ids = this->cache_->source_ids_;
    auto& source_cache_map = this->cache_->neighbors_;
    source_ids.clear();
    source_cache_map.clear();
    source_ids.reserve(this->total_count_);
    Vector<InnerIdType> inner_id_list(allocator_);
    for (InnerIdType inner_id = 0; inner_id < this->total_count_; ++inner_id) {
        auto source_id = this->label_table_->GetSourceId(inner_id);
        source_ids.push_back(source_id);
        if (source_id.empty()) {
            continue;
        }
        auto [it, _] = source_cache_map.try_emplace(source_id, Vector<InnerIdType>(allocator_));
        auto& cached = it.value();
        cached.push_back(inner_id);
        inner_id_list.clear();
        this->bottom_graph_->GetNeighbors(inner_id, inner_id_list);
        cached.insert(cached.end(), inner_id_list.begin(), inner_id_list.end());
    }
}

}  // namespace vsag
