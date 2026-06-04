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

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "datacell/flatten_datacell_parameter.h"
#include "dataset_impl.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/heap/standard_heap.h"
#include "impl/logger/logger.h"
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
    if (this->has_loaded_cache()) {
        // A previously exported cache has been imported via ImportCache().
        // Take the accelerated build path that warm-starts neighbours from
        // the cache and refines them, instead of building from scratch.
        auto ret = this->build_with_cache(data);
        if (use_elp_optimizer_) {
            elp_optimize();
        }
        return ret;
    }
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
    const auto* source_id = data->GetSourceID();
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
        // Persist source_id alongside label so day2 ExportCache produces a
        // non-empty source_id_table_. Same array-indexing semantics as Add().
        if (source_id != nullptr && not source_id[i].empty()) {
            this->label_table_->InsertSourceId(inner_id, source_id[i]);
        }
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
    std::shared_lock<std::shared_mutex> force_remove_rlock;
    if (this->support_force_remove()) {
        force_remove_rlock = std::shared_lock<std::shared_mutex>(this->force_remove_mutex_);
    }
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
        // Check if label already exists (skip removed IDs)
        {
            std::shared_lock label_lock(this->label_lookup_mutex_);
            auto [found, _] = this->label_table_->TryGetIdByLabel(labels[j]);
            if (found) {
                failed_ids.emplace_back(labels[j]);
                continue;
            }
        }

        InnerIdType inner_id;

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
            // NOTE: Dataset::GetSourceID() returns a pointer to an array of N
            // source_id strings (one per row), matching the semantics already
            // used by build_with_cache() at line ~1234. Use array indexing
            // instead of dereferencing the head pointer.
            if (source_id != nullptr && not source_id[j].empty()) {
                this->label_table_->InsertSourceId(inner_id, source_id[j]);
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
    std::shared_lock<std::shared_mutex> add_lock;
    if (not this->support_force_remove()) {
        add_lock = std::shared_lock<std::shared_mutex>(this->add_mutex_);
    }
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
    std::unique_lock<std::shared_mutex> add_lock(this->add_mutex_, std::defer_lock);
    if (this->support_force_remove()) {
        add_lock.lock();
    }
    if (insert_codes) {
        this->insert_persistent_codes(data, inner_id);
    }
    if (not this->support_force_remove()) {
        add_lock.lock();
    }
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
HGraph::ExportCache(std::ostream& out_stream) const {
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
HGraph::fullfill_cache() const {
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

namespace {

uint64_t
build_cache_now_us() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

constexpr uint32_t HIT_REFINE_ROUNDS = 1;
// MISSED side requires >=3 rounds: cold-start nodes (no warm seed) need
// multiple sweeps to (a) discover neighbors via search, (b) install reverse
// edges into hit-node adjacency lists, and (c) re-search with the newly
// populated frontier. Bisection on the miss-only smoke test
// (ef_construction=50, 200 vectors) shows ROUNDS<=2 fails REQUIRE(found_self);
// ROUNDS=3 passes consistently. Upstream uses 4; we keep 3 to bound cost.
constexpr uint32_t MISSED_REFINE_ROUNDS = 3;
// refine ef tuned down from ef_construct_ (=300/400) to 64 to avoid the
// pathological hit_refine bottleneck where each node spends 308us re-exploring
// the bottom graph with ef=300 (10092s total for 32.96M nodes, 1.73x slower
// than full rebuild). With ef=64, expected per-node ~60us (5x faster).
// Reference gby BuildCacheOptions default for hit_refine_ef is also 64.
constexpr uint32_t HIT_REFINE_EF = 64;
// MISSED_REFINE_EF kept higher than HIT because missed nodes have NO seed
// neighbors and need wider exploration. User-directed asymmetric config.
constexpr uint32_t MISSED_REFINE_EF = 200;

}  // namespace

DistHeapPtr
HGraph::collect_refine_candidates(const DatasetPtr& data,
                                  InnerIdType inner_id,
                                  uint32_t input_idx,
                                  const FlattenInterfacePtr& flatten_codes,
                                  uint32_t refine_ef,
                                  bool use_self_as_entry) const {
    const uint32_t effective_refine_ef = refine_ef == 0 ? this->ef_construct_ : refine_ef;
    CHECK_ARGUMENT(effective_refine_ef > 0, "refine ef must be greater than 0");

    auto candidates = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
    std::unordered_set<InnerIdType> seen;

    Vector<InnerIdType> current_neighbors(allocator_);
    this->bottom_graph_->GetNeighbors(inner_id, current_neighbors);
    seen.reserve(current_neighbors.size() + effective_refine_ef);

    // Optimisation #2: replace the per-neighbour ComputePairVectors loop with
    // a single batched flatten->Query call. The original code looped 64 times
    // (max_degree per node) and each call paid for a separate query-vector
    // load + distance kernel invocation; with N nodes refined we performed
    // N*64 scalar distance computations. The batched path uses
    // FactoryComputer(query) once and then asks the flatten cell to fill an
    // array of N distances in one shot, which lets the underlying SIMD /
    // prefetching code amortise the query load across all neighbours. Bit
    // equivalence with the scalar path is preserved because Query() is the
    // canonical primitive that ComputePairVectors itself delegates to.
    if (not current_neighbors.empty()) {
        // De-duplicate against `seen` and skip self-loops before issuing the
        // batched Query so we never spend SIMD cycles on rows we will discard.
        Vector<InnerIdType> filtered_ids(allocator_);
        filtered_ids.reserve(current_neighbors.size());
        for (const auto neighbor : current_neighbors) {
            if (neighbor == inner_id || not seen.emplace(neighbor).second) {
                continue;
            }
            filtered_ids.push_back(neighbor);
        }
        if (not filtered_ids.empty()) {
            auto computer = flatten_codes->FactoryComputer(get_data(data, input_idx));
            Vector<float> filtered_dists(filtered_ids.size(), allocator_);
            flatten_codes->Query(filtered_dists.data(),
                                 computer,
                                 filtered_ids.data(),
                                 static_cast<InnerIdType>(filtered_ids.size()),
                                 nullptr);
            for (uint64_t k = 0; k < filtered_ids.size(); ++k) {
                candidates->Push(filtered_dists[k], filtered_ids[k]);
            }
        }
    }

    if (this->entry_point_id_ != INVALID_ENTRY_POINT && this->bottom_graph_->TotalCount() > 0) {
        // For cache-hit nodes, start the search from inner_id itself so the
        // warm-started stale neighbours act as the initial frontier and refine
        // exploits the local neighbourhood efficiently. For cache-missed /
        // cold-start nodes, use the global entry point for broad exploration.
        // When the node has no cached neighbours, self-entry is meaningless,
        // so fall back to the global entry point.
        const auto search_entry_point =
            (use_self_as_entry && not current_neighbors.empty()) ? inner_id : this->entry_point_id_;
        InnerSearchParam param;
        param.topk = static_cast<int64_t>(effective_refine_ef);
        param.ef = effective_refine_ef;
        param.ep = search_entry_point;
        param.is_inner_id_allowed = nullptr;
        auto result = this->search_one_graph(get_data(data, input_idx),
                                             this->bottom_graph_,
                                             flatten_codes,
                                             param,
                                             (VisitedListPtr) nullptr,
                                             nullptr);
        // Optimisation #1: reuse the (dist, inner_id) records produced by
        // search_one_graph instead of recomputing the same distance via
        // ComputePairVectors. The searcher already computed dist(query, id)
        // for every node it visited; the heap returned here stores those
        // exact (dist, id) pairs. Re-running ComputePairVectors would do the
        // same fp32 kernel call a second time for every candidate (~64 per
        // node for hit_refine, ~400 for missed_refine), which dominates the
        // candidate-collection cost. Dropping the recompute is bit-equivalent:
        // both paths go through flatten_codes->ComputePairVectors with the
        // same query and same inner_ids.
        while (not result->Empty()) {
            auto candidate = result->Top();
            const InnerIdType candidate_id = candidate.second;
            if (candidate_id == inner_id || not seen.emplace(candidate_id).second) {
                result->Pop();
                continue;
            }
            candidates->Push(candidate.first, candidate.second);
            result->Pop();
        }
    }

    return candidates;
}

void
HGraph::select_refine_neighbors_with_distances(const DatasetPtr& data,
                                               InnerIdType inner_id,
                                               uint32_t input_idx,
                                               const FlattenInterfacePtr& flatten_codes,
                                               uint32_t refine_ef,
                                               bool use_self_as_entry,
                                               Vector<InnerIdType>& out_neighbors,
                                               Vector<float>& out_distances) const {
    out_neighbors.clear();
    out_distances.clear();

    auto candidates = this->collect_refine_candidates(
        data, inner_id, input_idx, flatten_codes, refine_ef, use_self_as_entry);

    if (candidates->Empty()) {
        return;
    }

    const uint64_t max_size = this->bottom_graph_->MaximumDegree();
    select_edges_by_heuristic(candidates, max_size, flatten_codes, allocator_, this->alpha_);

    out_neighbors.reserve(candidates->Size());
    out_distances.reserve(candidates->Size());
    while (not candidates->Empty()) {
        out_neighbors.emplace_back(candidates->Top().second);
        out_distances.emplace_back(candidates->Top().first);
        candidates->Pop();
    }
}

void
HGraph::refine_nodes_two_phase(
    const DatasetPtr& data,
    const std::vector<InnerIdType>& ids_to_refine,
    std::string_view phase_name,
    uint32_t rounds,
    uint32_t refine_ef,
    bool use_self_as_entry,
    const FlattenInterfacePtr& flatten_codes,
    const std::unordered_map<InnerIdType, uint32_t>& inner_id_to_input_idx) {
    if (ids_to_refine.empty() || rounds == 0) {
        return;
    }

    uint32_t parallelism = 1;
    if (this->thread_pool_ != nullptr && this->build_thread_count_ > 1 &&
        ids_to_refine.size() > 1) {
        parallelism = std::min<uint32_t>(static_cast<uint32_t>(this->build_thread_count_),
                                         static_cast<uint32_t>(ids_to_refine.size()));
    }

    const uint32_t effective_refine_ef = refine_ef == 0 ? this->ef_construct_ : refine_ef;
    CHECK_ARGUMENT(effective_refine_ef > 0, "refine ef must be greater than 0");

    logger::info("[hgraph_build_cache] starting {} nodes={} rounds={} parallelism={}",
                 phase_name,
                 ids_to_refine.size(),
                 rounds,
                 parallelism);

    constexpr int64_t block_size = 128;
    const auto begin = build_cache_now_us();

    for (uint32_t round = 0; round < rounds; ++round) {
        const auto round_begin = build_cache_now_us();

        // ===== Phase 1: parallel search & local select (also keep distances) =====
        Vector<Vector<InnerIdType>> selected_neighbors(
            ids_to_refine.size(), Vector<InnerIdType>(allocator_), allocator_);
        Vector<Vector<float>> selected_distances(
            ids_to_refine.size(), Vector<float>(allocator_), allocator_);

        if (parallelism <= 1) {
            for (uint64_t i = 0; i < ids_to_refine.size(); ++i) {
                const auto inner_id = ids_to_refine[i];
                auto data_iter = inner_id_to_input_idx.find(inner_id);
                CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                               fmt::format("missing input row for inner_id {}", inner_id));
                this->select_refine_neighbors_with_distances(data,
                                                             inner_id,
                                                             data_iter->second,
                                                             flatten_codes,
                                                             effective_refine_ef,
                                                             use_self_as_entry,
                                                             selected_neighbors[i],
                                                             selected_distances[i]);
            }
        } else {
            std::vector<std::future<void>> futures;
            futures.reserve((ids_to_refine.size() + block_size - 1) / block_size);
            for (int64_t i = 0; i < static_cast<int64_t>(ids_to_refine.size()); i += block_size) {
                const auto end =
                    std::min(i + block_size, static_cast<int64_t>(ids_to_refine.size()));
                futures.emplace_back(this->thread_pool_->GeneralEnqueue([this,
                                                                         &data,
                                                                         &ids_to_refine,
                                                                         &inner_id_to_input_idx,
                                                                         &flatten_codes,
                                                                         effective_refine_ef,
                                                                         use_self_as_entry,
                                                                         &selected_neighbors,
                                                                         &selected_distances,
                                                                         i,
                                                                         end]() {
                    for (int64_t idx = i; idx < end; ++idx) {
                        const auto inner_id = ids_to_refine[idx];
                        auto data_iter = inner_id_to_input_idx.find(inner_id);
                        CHECK_ARGUMENT(data_iter != inner_id_to_input_idx.end(),
                                       fmt::format("missing input row for inner_id {}", inner_id));
                        this->select_refine_neighbors_with_distances(data,
                                                                     inner_id,
                                                                     data_iter->second,
                                                                     flatten_codes,
                                                                     effective_refine_ef,
                                                                     use_self_as_entry,
                                                                     selected_neighbors[idx],
                                                                     selected_distances[idx]);
                    }
                }));
            }
            std::exception_ptr search_ex = nullptr;
            for (auto& future : futures) {
                try {
                    future.get();
                } catch (...) {
                    if (not search_ex) {
                        search_ex = std::current_exception();
                    }
                }
            }
            if (search_ex) {
                std::rethrow_exception(search_ex);
            }
        }
        const auto search_elapsed = build_cache_now_us() - round_begin;

        // ===== Phase 2: serial writeback of selected neighbours =====
        const auto writeback_begin = build_cache_now_us();
        for (uint64_t i = 0; i < ids_to_refine.size(); ++i) {
            LockGuard lock(neighbors_mutex_, ids_to_refine[i]);
            this->bottom_graph_->InsertNeighborsById(ids_to_refine[i], selected_neighbors[i]);
        }
        const auto writeback_elapsed = build_cache_now_us() - writeback_begin;

        // ===== Phase 3: reverse-edge install (scatter -> materialise -> shard prune) =====
        const auto reverse_begin = build_cache_now_us();
        const uint32_t reverse_shard_count =
            (parallelism > 1)
                ? std::max<uint32_t>(parallelism, static_cast<uint32_t>(this->build_thread_count_))
                : 1U;

        struct reverse_edge_entry {
            InnerIdType src_id;
            float dist;
        };
        struct scatter_record {
            InnerIdType target_id;
            InnerIdType src_id;
            float dist;
        };
        struct reverse_shard {
            std::unordered_map<InnerIdType, std::vector<reverse_edge_entry>> pending;
        };
        std::vector<reverse_shard> shards(reverse_shard_count);

        // ----- (3a) scatter (target,src,dist) into per-(worker,shard) buffers -----
        const uint32_t scatter_worker_count =
            (parallelism > 1)
                ? std::min<uint32_t>(parallelism, static_cast<uint32_t>(ids_to_refine.size()))
                : 1U;

        std::vector<std::vector<std::vector<scatter_record>>> worker_buckets(
            scatter_worker_count, std::vector<std::vector<scatter_record>>(reverse_shard_count));

        const uint64_t max_degree = this->bottom_graph_->MaximumDegree();
        const uint64_t avg_per_bucket =
            (ids_to_refine.size() * max_degree +
             static_cast<uint64_t>(scatter_worker_count) * reverse_shard_count - 1) /
            (static_cast<uint64_t>(scatter_worker_count) * reverse_shard_count);
        for (auto& wb : worker_buckets) {
            for (auto& shard_buf : wb) {
                shard_buf.reserve(avg_per_bucket + 16);
            }
        }

        auto scatter_slice = [&](uint32_t worker_idx, uint64_t lo, uint64_t hi) {
            auto& my_buckets = worker_buckets[worker_idx];
            for (uint64_t i = lo; i < hi; ++i) {
                const auto src_id = ids_to_refine[i];
                const auto& neighbors = selected_neighbors[i];
                const auto& dists = selected_distances[i];
                const uint64_t k = neighbors.size();
                CHECK_ARGUMENT(dists.size() == k,
                               "selected_neighbors and selected_distances size mismatch");
                for (uint64_t j = 0; j < k; ++j) {
                    const auto neighbor_id = neighbors[j];
                    const uint32_t shard_idx =
                        static_cast<uint32_t>(neighbor_id) % reverse_shard_count;
                    my_buckets[shard_idx].push_back({neighbor_id, src_id, dists[j]});
                }
            }
        };

        if (scatter_worker_count <= 1) {
            scatter_slice(0, 0, ids_to_refine.size());
        } else {
            const uint64_t per_worker =
                (ids_to_refine.size() + scatter_worker_count - 1) / scatter_worker_count;
            std::vector<std::future<void>> scatter_futures;
            scatter_futures.reserve(scatter_worker_count);
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                const uint64_t lo = static_cast<uint64_t>(w) * per_worker;
                if (lo >= ids_to_refine.size()) {
                    break;
                }
                const uint64_t hi =
                    std::min(lo + per_worker, static_cast<uint64_t>(ids_to_refine.size()));
                scatter_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&scatter_slice, w, lo, hi]() { scatter_slice(w, lo, hi); }));
            }
            std::exception_ptr scatter_ex = nullptr;
            for (auto& f : scatter_futures) {
                try {
                    f.get();
                } catch (...) {
                    if (not scatter_ex) {
                        scatter_ex = std::current_exception();
                    }
                }
            }
            if (scatter_ex) {
                std::rethrow_exception(scatter_ex);
            }
        }

        // ----- (3b) materialise per-shard (target -> [(src,dist), ...]) maps -----
        auto materialise_shard = [&](uint32_t shard_idx) {
            uint64_t total = 0;
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                total += worker_buckets[w][shard_idx].size();
            }
            auto& pending = shards[shard_idx].pending;
            const uint64_t hint = total / std::max<uint64_t>(max_degree / 2, 1) + 16;
            pending.reserve(hint);
            for (uint32_t w = 0; w < scatter_worker_count; ++w) {
                auto& buf = worker_buckets[w][shard_idx];
                for (const auto& rec : buf) {
                    pending[rec.target_id].push_back({rec.src_id, rec.dist});
                }
                std::vector<scatter_record>().swap(buf);
            }
        };

        if (parallelism <= 1 || reverse_shard_count <= 1) {
            for (uint32_t s = 0; s < reverse_shard_count; ++s) {
                materialise_shard(s);
            }
        } else {
            std::vector<std::future<void>> mat_futures;
            mat_futures.reserve(reverse_shard_count);
            for (uint32_t s = 0; s < reverse_shard_count; ++s) {
                mat_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&materialise_shard, s]() { materialise_shard(s); }));
            }
            std::exception_ptr mat_ex = nullptr;
            for (auto& f : mat_futures) {
                try {
                    f.get();
                } catch (...) {
                    if (not mat_ex) {
                        mat_ex = std::current_exception();
                    }
                }
            }
            if (mat_ex) {
                std::rethrow_exception(mat_ex);
            }
        }
        std::vector<std::vector<std::vector<scatter_record>>>().swap(worker_buckets);

        // ----- (3c) per-shard merge + heuristic prune (parallel across shards) -----
        auto process_shard = [this, &flatten_codes, max_degree](reverse_shard& shard) {
            Vector<InnerIdType> current_neighbors(allocator_);
            current_neighbors.reserve(max_degree + 16);
            std::unordered_set<InnerIdType> existing_set;
            existing_set.reserve(max_degree * 2 + 16);
            std::unordered_map<InnerIdType, float> reuse_dist;

            for (auto& entry : shard.pending) {
                const auto target_id = entry.first;
                auto& reverse_adds = entry.second;

                LockGuard lock(neighbors_mutex_, target_id);

                current_neighbors.clear();
                this->bottom_graph_->GetNeighbors(target_id, current_neighbors);

                existing_set.clear();
                for (auto nid : current_neighbors) {
                    existing_set.insert(nid);
                }

                bool changed = false;
                for (const auto& add : reverse_adds) {
                    if (add.src_id == target_id) {
                        continue;
                    }
                    if (existing_set.insert(add.src_id).second) {
                        current_neighbors.push_back(add.src_id);
                        changed = true;
                    }
                }
                if (not changed) {
                    continue;
                }

                if (current_neighbors.size() > max_degree) {
                    // Reuse src->target distances captured during the select phase.
                    reuse_dist.clear();
                    reuse_dist.reserve(reverse_adds.size());
                    for (const auto& add : reverse_adds) {
                        auto [it, inserted] = reuse_dist.emplace(add.src_id, add.dist);
                        if (not inserted && add.dist < it->second) {
                            it->second = add.dist;
                        }
                    }

                    auto edges = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
                    for (auto neighbor_id : current_neighbors) {
                        auto rit = reuse_dist.find(neighbor_id);
                        const float d =
                            (rit != reuse_dist.end())
                                ? rit->second
                                : flatten_codes->ComputePairVectors(neighbor_id, target_id);
                        edges->Push(d, neighbor_id);
                    }
                    select_edges_by_heuristic(
                        edges, max_degree, flatten_codes, allocator_, this->alpha_);
                    current_neighbors.clear();
                    while (not edges->Empty()) {
                        current_neighbors.emplace_back(edges->Top().second);
                        edges->Pop();
                    }
                }
                this->bottom_graph_->InsertNeighborsById(target_id, current_neighbors);
            }
        };

        if (parallelism <= 1 || reverse_shard_count <= 1) {
            for (auto& shard : shards) {
                process_shard(shard);
            }
        } else {
            std::vector<std::future<void>> shard_futures;
            shard_futures.reserve(shards.size());
            for (auto& shard : shards) {
                shard_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    [&process_shard, &shard]() { process_shard(shard); }));
            }
            std::exception_ptr shard_ex = nullptr;
            for (auto& f : shard_futures) {
                try {
                    f.get();
                } catch (...) {
                    if (not shard_ex) {
                        shard_ex = std::current_exception();
                    }
                }
            }
            if (shard_ex) {
                std::rethrow_exception(shard_ex);
            }
        }
        const auto reverse_elapsed = build_cache_now_us() - reverse_begin;
        const auto round_elapsed = build_cache_now_us() - round_begin;
        logger::info(
            "[hgraph_build_cache] {} round {}/{} finished in {:.3f}s "
            "(search={:.3f}s writeback={:.3f}s reverse={:.3f}s) processed_nodes={}",
            phase_name,
            round + 1,
            rounds,
            static_cast<double>(round_elapsed) / 1000000.0,
            static_cast<double>(search_elapsed) / 1000000.0,
            static_cast<double>(writeback_elapsed) / 1000000.0,
            static_cast<double>(reverse_elapsed) / 1000000.0,
            ids_to_refine.size());
    }
    const auto total_elapsed = build_cache_now_us() - begin;
    logger::info("[hgraph_build_cache] {} finished in {:.3f}s",
                 phase_name,
                 static_cast<double>(total_elapsed) / 1000000.0);
}

std::vector<int64_t>
HGraph::build_with_cache(const DatasetPtr& data) {
    CHECK_ARGUMENT(GetNumElements() == 0, "index is not empty");
    CHECK_ARGUMENT(this->cache_ != nullptr, "build_with_cache requires an imported cache");
    CHECK_ARGUMENT(data->GetSourceID() != nullptr,
                   "build_with_cache requires Dataset::SourceID to be set");

    const auto build_begin = build_cache_now_us();
    this->Train(data);

    BuildCachePlan plan(allocator_);
    this->cache_collect_valid_indices(data, plan);
    this->cache_setup_metadata_serial(data, plan);
    this->cache_encode_codes_parallel(data, plan);
    this->cache_warm_start_and_classify(plan);
    this->cache_run_refine_two_phase(data, plan);
    this->cache_rebuild_route_graphs(plan);

    const auto build_total_elapsed = build_cache_now_us() - build_begin;
    logger::info("[hgraph_build_cache] build_with_cache total elapsed {:.3f}s",
                 static_cast<double>(build_total_elapsed) / 1000000.0);
    return std::move(plan.failed_ids);
}

// Step 1: scan dataset, dedup labels against existing index + intra-batch
// duplicates, allocate inner_ids for the surviving rows, grow capacity.
// Outputs (in plan): valid_indices, inner_ids, failed_ids.
void
HGraph::cache_collect_valid_indices(const DatasetPtr& data, BuildCachePlan& plan) {
    const auto* labels = data->GetIds();
    const int64_t total = data->GetNumElements();
    UnorderedSet<LabelType> seen_labels(allocator_);
    for (int64_t i = 0; i < total; ++i) {
        auto label = labels[i];
        if (this->label_table_->CheckLabel(label) || seen_labels.find(label) != seen_labels.end()) {
            plan.failed_ids.emplace_back(label);
            continue;
        }
        seen_labels.insert(label);
        plan.valid_indices.emplace_back(i);
    }
    plan.inner_ids =
        this->get_unique_inner_ids(static_cast<InnerIdType>(plan.valid_indices.size()));
    auto current_count = total_count_.load();
    uint64_t new_ids_count = 0;
    for (auto inner_id : plan.inner_ids) {
        if (inner_id >= current_count) {
            ++new_ids_count;
        }
    }
    this->resize(current_count + new_ids_count);
    this->total_count_ += new_ids_count;
    plan.inserted_inner_ids.reserve(static_cast<uint64_t>(plan.valid_indices.size()));
    plan.inner_id_to_input_idx.reserve(static_cast<uint64_t>(plan.valid_indices.size()));
    plan.source_id_to_new_inner.reserve(static_cast<uint64_t>(plan.valid_indices.size()));
}

// Step 1a: serial setup of label_table / source_id_table / route_graph_ids
// and bookkeeping containers. LabelTable::Insert and the route graph id
// buckets are NOT thread-safe (vector resize race), so this short prelude
// stays serial.
void
HGraph::cache_setup_metadata_serial(const DatasetPtr& data, BuildCachePlan& plan) {
    const auto* labels = data->GetIds();
    const auto* source_ids = data->GetSourceID();
    const auto prepare_meta_begin = build_cache_now_us();
    for (uint64_t cur = 0; cur < plan.valid_indices.size(); ++cur) {
        const auto i = plan.valid_indices[cur];
        const auto inner_id = plan.inner_ids.at(cur);
        const auto label = labels[i];
        this->label_table_->Insert(inner_id, label);
        if (source_ids != nullptr && not source_ids[i].empty()) {
            this->label_table_->InsertSourceId(inner_id, source_ids[i]);
            plan.source_id_to_new_inner.emplace(source_ids[i], inner_id);
        }
        plan.inserted_inner_ids.push_back(inner_id);
        plan.inner_id_to_input_idx.emplace(inner_id, static_cast<uint32_t>(i));

        const auto level = this->get_random_level() - 1;
        if (level >= 0) {
            if (level >= static_cast<int>(plan.route_graph_ids.size()) ||
                plan.route_graph_ids.empty()) {
                for (auto k = static_cast<int>(plan.route_graph_ids.size()); k <= level; ++k) {
                    plan.route_graph_ids.emplace_back(allocator_);
                }
                entry_point_id_ = inner_id;
            }
            for (int j = 0; j <= level; ++j) {
                plan.route_graph_ids[j].emplace_back(inner_id);
            }
        }
    }
    const auto prepare_meta_elapsed = build_cache_now_us() - prepare_meta_begin;
    logger::info("[hgraph_build_cache] prepare_meta finished in {:.3f}s nodes={}",
                 static_cast<double>(prepare_meta_elapsed) / 1000000.0,
                 static_cast<uint64_t>(plan.valid_indices.size()));
}

// Step 1b: parallel encode (RaBitQ + SQ8) + extra_info/attr_filter writes.
// Each task touches a unique inner_id, FlattenDataCell::InsertVector is
// per-cell-mutex-protected, ExtraInfo and AttrFilter writes are also
// per-inner_id and independent.
void
HGraph::cache_encode_codes_parallel(const DatasetPtr& data, BuildCachePlan& plan) {
    const auto* extra_infos = data->GetExtraInfos();
    const auto* attr_sets = data->GetAttributeSets();
    const auto prepare_encode_begin = build_cache_now_us();
    const bool use_parallel_prepare =
        this->thread_pool_ != nullptr && this->build_thread_count_ > 1;
    auto& valid_indices = plan.valid_indices;
    auto& inner_ids = plan.inner_ids;
    if (use_parallel_prepare) {
        const uint64_t num_threads = this->build_thread_count_;
        const uint64_t total_jobs = valid_indices.size();
        const uint64_t block_size =
            std::max<uint64_t>(1, (total_jobs + num_threads - 1) / num_threads);
        std::vector<std::future<void>> futures;
        futures.reserve(num_threads);
        for (uint64_t lo = 0; lo < total_jobs; lo += block_size) {
            const uint64_t hi = std::min<uint64_t>(lo + block_size, total_jobs);
            futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                [this, lo, hi, &valid_indices, &inner_ids, data, extra_infos, attr_sets]() {
                    for (uint64_t cur = lo; cur < hi; ++cur) {
                        const auto i = valid_indices[cur];
                        const auto inner_id = inner_ids.at(cur);
                        this->insert_persistent_codes(get_data(data, static_cast<uint32_t>(i)),
                                                      inner_id);
                        if (this->extra_infos_ != nullptr && extra_infos != nullptr) {
                            this->extra_infos_->InsertExtraInfo(extra_infos + i * extra_info_size_,
                                                                inner_id);
                        }
                        if (attr_sets != nullptr && this->use_attribute_filter_) {
                            this->attr_filter_index_->Insert(attr_sets[i], inner_id);
                        }
                    }
                }));
        }
        // CRITICAL: drain ALL futures even if one throws. Background tasks
        // capture valid_indices/inner_ids by reference; early return on first
        // exception would leave tasks running against soon-destroyed stack
        // variables, causing use-after-free / UB. Mirror the warm_start
        // join-then-rethrow pattern below.
        std::exception_ptr prepare_ex = nullptr;
        for (auto& f : futures) {
            try {
                f.get();
            } catch (...) {
                if (not prepare_ex) {
                    prepare_ex = std::current_exception();
                }
            }
        }
        if (prepare_ex) {
            std::rethrow_exception(prepare_ex);
        }
    } else {
        for (uint64_t cur = 0; cur < valid_indices.size(); ++cur) {
            const auto i = valid_indices[cur];
            const auto inner_id = inner_ids.at(cur);
            this->insert_persistent_codes(get_data(data, static_cast<uint32_t>(i)), inner_id);
            if (this->extra_infos_ != nullptr && extra_infos != nullptr) {
                this->extra_infos_->InsertExtraInfo(extra_infos + i * extra_info_size_, inner_id);
            }
            if (attr_sets != nullptr && this->use_attribute_filter_) {
                this->attr_filter_index_->Insert(attr_sets[i], inner_id);
            }
        }
    }
    const auto prepare_encode_elapsed = build_cache_now_us() - prepare_encode_begin;
    logger::info("[hgraph_build_cache] prepare_encode finished in {:.3f}s nodes={} parallelism={}",
                 static_cast<double>(prepare_encode_elapsed) / 1000000.0,
                 static_cast<uint64_t>(valid_indices.size()),
                 use_parallel_prepare ? this->build_thread_count_ : 1);
    if (entry_point_id_ == INVALID_ENTRY_POINT && not plan.inserted_inner_ids.empty()) {
        entry_point_id_ = plan.inserted_inner_ids.front();
    }
}

// Step 2: warm_start - seed neighbours using the cache, classify nodes into
// hit_ids (have warm seed) / missed_ids (no warm seed, need full search).
//
// The cache encodes neighbours by source_id:
//   cache_->source_ids_[old_inner_id] -> source_id (string)
//   cache_->neighbors_[source_id]     -> [old_inner_id, neighbor_old_inner_id...]
// We translate the old neighbour list into the new inner_id space via
// (old neighbor inner_id) -> (old source_id) -> (new inner_id).
//
// Optimisations vs. the original O(N) serial implementation:
//   A. Replace the per-node std::unordered_set<InnerIdType> dedup
//      structure (which forced 30M+ malloc/free pairs) with a thread-local
//      Vector<InnerIdType> reused across nodes, deduped via std::sort +
//      std::unique on a tiny (<= max_degree+1) array.
//   B. Run the outer loop in parallel via thread_pool_->GeneralEnqueue,
//      using std::atomic<uint64_t> slots to claim positions in
//      hit_ids/missed_ids without locking.
//   C. Pre-build a flat Vector<InnerIdType> old_to_new_map indexed by
//      old_inner_id, eliminating the per-edge string hash lookup
//      (cache_source_ids[idx] -> source_id_to_new_inner.find(source_id))
//      on the hot path. This drops the inner-loop translation cost from
//      two string-keyed unordered_map probes to a single integer index.
void
HGraph::cache_warm_start_and_classify(BuildCachePlan& plan) {
    const auto warm_start_begin = build_cache_now_us();
    const auto& cache_source_ids = this->cache_->source_ids_;
    const auto& cache_map = this->cache_->neighbors_;
    const uint64_t max_degree = this->bottom_graph_->MaximumDegree();

    // Step 2.0 (optimisation C): build old_inner_id -> new_inner_id map.
    constexpr InnerIdType invalid_id = std::numeric_limits<InnerIdType>::max();
    Vector<InnerIdType> old_to_new_map(cache_source_ids.size(), invalid_id, allocator_);
    for (uint64_t old_inner = 0; old_inner < cache_source_ids.size(); ++old_inner) {
        const auto& sid = cache_source_ids[old_inner];
        if (sid.empty()) {
            continue;
        }
        auto fit = plan.source_id_to_new_inner.find(sid);
        if (fit != plan.source_id_to_new_inner.end()) {
            old_to_new_map[old_inner] = fit->second;
        }
    }

    const uint64_t total_nodes = plan.inserted_inner_ids.size();
    auto& hit_ids = plan.hit_ids;
    auto& missed_ids = plan.missed_ids;
    hit_ids.assign(total_nodes, invalid_id);
    missed_ids.assign(total_nodes, invalid_id);
    std::atomic<uint64_t> hit_idx{0};
    std::atomic<uint64_t> missed_idx{0};
    std::atomic<uint64_t> hit_empty_seed_atomic{0};
    std::atomic<uint64_t> hit_seed_neighbor_atomic{0};

    // Block size sized so that each task processes ~16 K nodes; with 32-thread
    // pool and 30 M nodes this yields ~1900 tasks, plenty to amortise enqueue
    // overhead while keeping per-task data small.
    constexpr uint64_t warm_start_block = 16384;
    const uint64_t num_blocks = (total_nodes + warm_start_block - 1) / warm_start_block;

    auto warm_start_worker = [&, this](uint64_t lo, uint64_t hi) {
        Vector<InnerIdType> mapped(allocator_);
        Vector<InnerIdType> empty_neighbours(allocator_);
        mapped.reserve(max_degree + 8);
        for (uint64_t k = lo; k < hi; ++k) {
            const auto inner_id = plan.inserted_inner_ids[k];
            const auto& source_id = this->label_table_->GetSourceId(inner_id);
            if (source_id.empty()) {
                this->bottom_graph_->InsertNeighborsById(inner_id, empty_neighbours);
                missed_ids[missed_idx.fetch_add(1, std::memory_order_relaxed)] = inner_id;
                continue;
            }
            auto it = cache_map.find(source_id);
            if (it == cache_map.end()) {
                this->bottom_graph_->InsertNeighborsById(inner_id, empty_neighbours);
                missed_ids[missed_idx.fetch_add(1, std::memory_order_relaxed)] = inner_id;
                continue;
            }
            const auto& cached_list = it->second;
            mapped.clear();
            for (uint64_t kk = 1; kk < cached_list.size(); ++kk) {
                const auto neighbor_old_inner = cached_list[kk];
                if (static_cast<uint64_t>(neighbor_old_inner) >= old_to_new_map.size()) {
                    continue;
                }
                const auto new_neighbor = old_to_new_map[neighbor_old_inner];
                if (new_neighbor == invalid_id || new_neighbor == inner_id) {
                    continue;
                }
                mapped.push_back(new_neighbor);
            }
            // Tiny array (<= ~64 elements typically): sort + unique is faster
            // than building+tearing down an unordered_set per node.
            std::sort(mapped.begin(), mapped.end());
            mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
            if (mapped.size() > max_degree) {
                mapped.resize(max_degree);
            }
            this->bottom_graph_->InsertNeighborsById(inner_id, mapped);
            if (mapped.empty()) {
                // Nodes whose cached neighbours could not be translated into
                // the current index have no warm-start signal. Treat them as
                // cold-start nodes so they receive the full missed-refine
                // budget (global entry-point search + more rounds) instead of
                // the cheap hit-refine path.
                hit_empty_seed_atomic.fetch_add(1, std::memory_order_relaxed);
                missed_ids[missed_idx.fetch_add(1, std::memory_order_relaxed)] = inner_id;
            } else {
                hit_seed_neighbor_atomic.fetch_add(mapped.size(), std::memory_order_relaxed);
                hit_ids[hit_idx.fetch_add(1, std::memory_order_relaxed)] = inner_id;
            }
        }
    };

    if (this->thread_pool_ == nullptr || num_blocks <= 1) {
        warm_start_worker(0, total_nodes);
    } else {
        std::vector<std::future<void>> warm_futures;
        warm_futures.reserve(num_blocks);
        for (uint64_t b = 0; b < num_blocks; ++b) {
            const uint64_t lo = b * warm_start_block;
            const uint64_t hi = std::min(lo + warm_start_block, total_nodes);
            warm_futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                [&warm_start_worker, lo, hi]() { warm_start_worker(lo, hi); }));
        }
        std::exception_ptr warm_ex = nullptr;
        for (auto& fut : warm_futures) {
            try {
                fut.get();
            } catch (...) {
                if (not warm_ex) {
                    warm_ex = std::current_exception();
                }
            }
        }
        if (warm_ex) {
            std::rethrow_exception(warm_ex);
        }
    }
    hit_ids.resize(hit_idx.load(std::memory_order_relaxed));
    missed_ids.resize(missed_idx.load(std::memory_order_relaxed));
    const uint64_t hit_empty_seed_nodes = hit_empty_seed_atomic.load(std::memory_order_relaxed);
    const uint64_t hit_seed_neighbor_total =
        hit_seed_neighbor_atomic.load(std::memory_order_relaxed);
    const auto warm_start_elapsed = build_cache_now_us() - warm_start_begin;
    const uint64_t total_classified = hit_ids.size() + missed_ids.size();
    const float hit_rate = total_classified > 0 ? static_cast<float>(hit_ids.size()) /
                                                      static_cast<float>(total_classified)
                                                : 0.0F;
    logger::info(
        "[hgraph_build_cache] warm_start finished in {:.3f}s hit_nodes={} missed_nodes={} "
        "hit_empty_seed_nodes={} hit_seed_neighbor_total={} hit_rate={:.4f}",
        static_cast<double>(warm_start_elapsed) / 1000000.0,
        hit_ids.size(),
        missed_ids.size(),
        hit_empty_seed_nodes,
        hit_seed_neighbor_total,
        hit_rate);
}

// Step 3: refine missed nodes (global entry) first, then hit nodes.
// Rationale: running missed_refine first lets cold-start nodes install
// both their own forward neighbours and their reverse-edge contributions
// back into hit-node adjacency lists before hit_refine kicks in. The
// subsequent hit_refine therefore sees a richer local frontier (warm
// seed merged with reverse edges produced by missed_refine) and can
// still exploit the cheap self-entry path on nodes whose neighbour list
// remains non-empty.
//
// Hit nodes whose neighbour list is empty when hit_refine starts
// (either because warm_start could not map any cached neighbour, or
// because every warm seed was evicted by the missed_refine reverse
// pruning) will automatically fall back to the global entry point via
// the existing guard inside collect_refine_candidates().
//
// Parallelism note: refine_nodes_two_phase fully drains its internal
// futures before returning, so swapping the two invocations does not
// introduce any cross-phase data races.
void
HGraph::cache_run_refine_two_phase(const DatasetPtr& data, BuildCachePlan& plan) {
    auto flatten_codes = this->basic_flatten_codes_;
    if (this->has_precise_reorder()) {
        flatten_codes = this->high_precise_codes_;
    }
    logger::debug(
        "[DIAG] refine flatten_codes quantizer = {} (basic={}, high_precise={})",
        flatten_codes->GetQuantizerName(),
        this->basic_flatten_codes_ ? this->basic_flatten_codes_->GetQuantizerName() : "null",
        this->high_precise_codes_ ? this->high_precise_codes_->GetQuantizerName() : "null");
    this->refine_nodes_two_phase(data,
                                 plan.missed_ids,
                                 "missed_refine",
                                 MISSED_REFINE_ROUNDS,
                                 MISSED_REFINE_EF,
                                 /*use_self_as_entry=*/false,
                                 flatten_codes,
                                 plan.inner_id_to_input_idx);
    this->refine_nodes_two_phase(data,
                                 plan.hit_ids,
                                 "hit_refine",
                                 HIT_REFINE_ROUNDS,
                                 HIT_REFINE_EF,
                                 /*use_self_as_entry=*/true,
                                 flatten_codes,
                                 plan.inner_id_to_input_idx);
}

// Step 4: rebuild route graphs via ODescent.
void
HGraph::cache_rebuild_route_graphs(BuildCachePlan& plan) {
    this->route_graphs_.clear();
    if (plan.route_graph_ids.empty()) {
        return;
    }
    const auto route_graph_begin = build_cache_now_us();
    if (this->odescent_param_ == nullptr) {
        this->odescent_param_ = std::make_shared<ODescentParameter>();
    }
    auto build_data =
        this->has_precise_reorder() ? this->high_precise_codes_ : this->basic_flatten_codes_;
    for (auto& route_graph_id : plan.route_graph_ids) {
        odescent_param_->max_degree = bottom_graph_->MaximumDegree() / 2;
        ODescent sparse_odescent_builder(
            odescent_param_, build_data, allocator_, this->thread_pool_.get());
        auto graph = this->generate_one_route_graph();
        sparse_odescent_builder.Build(route_graph_id);
        sparse_odescent_builder.SaveGraph(graph);
        this->route_graphs_.emplace_back(graph);
    }
    const auto route_graph_elapsed = build_cache_now_us() - route_graph_begin;
    logger::info("[hgraph_build_cache] route_graph_build finished in {:.3f}s levels={}",
                 static_cast<double>(route_graph_elapsed) / 1000000.0,
                 this->route_graphs_.size());
}

}  // namespace vsag
