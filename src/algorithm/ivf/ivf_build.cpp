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

#include <algorithm>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "datacell/graph_datacell_parameter.h"
#include "impl/heap/standard_heap.h"
#include "impl/inner_search_param.h"
#include "impl/pruning_strategy.h"
#include "impl/searcher/basic_searcher.h"
#include "index_feature_list.h"
#include "ivf.h"  // IWYU pragma: keep
#include "query_context.h"
#include "utils/util_functions.h"
#include "utils/visited_list.h"
#include "vsag_exception.h"

namespace vsag {

void
IVF::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
        IndexFeature::SUPPORT_ADD_CONCURRENT,
    });

    // search
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_KNN_SEARCH,
        IndexFeature::SUPPORT_KNN_SEARCH_WITH_ID_FILTER,
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
        IndexFeature::SUPPORT_SERIALIZE_WRITE_FUNC,
    });

    auto name = this->bucket_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16 and
        name != QUANTIZATION_TYPE_VALUE_FP16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }

    bool has_fp32 = false;
    if (use_reorder_) {
        const auto precise_quantizer_name = precise_bucket_ != nullptr
                                                ? precise_bucket_->GetQuantizerName()
                                                : reorder_codes_->GetQuantizerName();
        has_fp32 = precise_quantizer_name == QUANTIZATION_TYPE_VALUE_FP32;
    }
    if (name == QUANTIZATION_TYPE_VALUE_FP32 or has_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_BATCH_CALC_DISTANCE_BY_ID);
    }

    if (name == QUANTIZATION_TYPE_VALUE_FP32 and
        this->bucket_->GetMetricType() != MetricType::METRIC_TYPE_COSINE and
        not bucket_->UseResidual()) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_GET_DATA_BY_IDS);
    }

    this->index_feature_list_->SetFeatures({IndexFeature::SUPPORT_CLONE,
                                            IndexFeature::SUPPORT_EXPORT_MODEL,
                                            IndexFeature::SUPPORT_GET_MEMORY_USAGE,
                                            IndexFeature::SUPPORT_MERGE_INDEX});
    if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_ADD_AFTER_BUILD, false);
    }
}

std::vector<int64_t>
IVF::Build(const DatasetPtr& base) {
    if (graph_build_threshold_ > 0) {
        CHECK_ARGUMENT(this->total_elements_ == 0,
                       "graph bucket searcher requires a fresh Build with no prior data");
    }
    this->Train(base);
    // TODO(LHT): duplicate
    auto result = this->Add(base);
    if (graph_build_threshold_ > 0) {
        this->build_bucket_graphs();
    }
    this->cal_memory_usage();
    return result;
}

void
IVF::Train(const DatasetPtr& data) {
    if (this->is_trained_) {
        return;
    }

    int64_t total_elements = data->GetNumElements();
    int64_t dim = data->GetDim();
    DatasetPtr train_data =
        vsag::sample_train_data(data, total_elements, dim, train_sample_count_, allocator_);
    int64_t sample_count = train_data->GetNumElements();

    partition_strategy_->Train(train_data);

    const auto* data_ptr = train_data->GetFloat32Vectors();
    this->bucket_->Train(data_ptr, sample_count);
    if (use_reorder_) {
        if (precise_bucket_ != nullptr) {
            this->precise_bucket_->Train(data->GetFloat32Vectors(), data->GetNumElements());
        } else {
            this->reorder_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
        }
    }
    this->is_trained_ = true;
}

std::vector<int64_t>
IVF::Add(const DatasetPtr& base) {
    // TODO(LHT): duplicate
    if (not partition_strategy_->is_trained_) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "ivf index add without train error");
    }
    this->bucket_->Unpack();
    if (precise_bucket_ != nullptr) {
        this->precise_bucket_->Unpack();
    }
    auto num_element = base->GetNumElements();
    const auto* ids = base->GetIds();
    const auto* vectors = base->GetFloat32Vectors();
    const auto* attr_sets = base->GetAttributeSets();
    const auto* extra_info = base->GetExtraInfos();
    const auto extra_info_size = base->GetExtraInfoSize();
    auto buckets =
        partition_strategy_->ClassifyDatas(vectors, num_element, buckets_per_data_, nullptr);

    int64_t current_num;
    bool need_cal_memory_usage = false;
    {
        std::lock_guard lock(label_lookup_mutex_);
        current_num = this->total_elements_;
        if (precise_bucket_ != nullptr) {
            if (num_element < 0 or current_num < 0) {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    "invalid IVF precise bucket element count");
            }
            const auto posting_count = static_cast<uint64_t>(num_element);
            const auto current_count = static_cast<uint64_t>(current_num);
            const auto max_inner_id =
                static_cast<uint64_t>(std::numeric_limits<InnerIdType>::max());
            if (posting_count > max_inner_id or current_count > max_inner_id - posting_count) {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    "IVF precise bucket batch exceeds inner id capacity");
            }
        }
        if (use_reorder_ and precise_bucket_ == nullptr) {
            this->reorder_codes_->BatchInsertVector(base->GetFloat32Vectors(),
                                                    base->GetNumElements());
        }
        for (int64_t i = 0; i < num_element; ++i) {
            this->label_table_->Insert(i + total_elements_, ids[i]);
        }
        this->total_elements_ += num_element;
        if (this->total_elements_ - last_cal_memory_element_ >= cal_memory_element_interval_) {
            need_cal_memory_usage = true;
            last_cal_memory_element_ = this->total_elements_;
        }
        location_map_.resize(this->total_elements_);
    }

    Vector<InnerIdType> precise_offsets(allocator_);
    if (precise_bucket_ != nullptr) {
        const auto posting_count = static_cast<uint64_t>(num_element);
        Vector<InnerIdType> posting_ids(posting_count, allocator_);
        precise_offsets.resize(posting_count);
        for (uint64_t i = 0; i < posting_count; ++i) {
            posting_ids[i] = static_cast<InnerIdType>(i + static_cast<uint64_t>(current_num));
        }
        precise_bucket_->BatchInsertVector(vectors,
                                           buckets.data(),
                                           posting_ids.data(),
                                           static_cast<InnerIdType>(posting_count),
                                           precise_offsets.data());
    }

    auto add_func = [&](int64_t i) -> void {
        for (int64_t j = 0; j < buckets_per_data_; ++j) {
            const auto* data_ptr = vectors + i * dim_;
            auto idx = i * buckets_per_data_ + j;
            auto posting_id = static_cast<InnerIdType>(idx + current_num * buckets_per_data_);
            InnerIdType offset_id;
            if (precise_bucket_ != nullptr) {
                // Publish the basic posting only after its precise mirror is ready.
                offset_id = precise_offsets[idx];
                bucket_->InsertVectorWithOffset(data_ptr, buckets[idx], posting_id, offset_id);
            } else {
                offset_id = bucket_->InsertVector(data_ptr, buckets[idx], posting_id);
            }
            if (j == 0) {
                std::lock_guard lock(label_lookup_mutex_);
                location_map_[i + current_num] =
                    (static_cast<uint64_t>(buckets[idx]) << LOCATION_SPLIT_BIT) |
                    static_cast<uint64_t>(offset_id);
            }
            if (use_attribute_filter_ and this->attr_filter_index_ != nullptr and
                attr_sets != nullptr) {
                const auto& attr_set = attr_sets[i];
                this->attr_filter_index_->Insert(attr_set, offset_id, buckets[idx]);
            }
            if (extra_info_size > 0) {
                this->extra_infos_->InsertExtraInfo(extra_info + i * extra_info_size,
                                                    i + current_num);
            }
        }
    };
    std::vector<std::future<void>> futures;
    std::exception_ptr first_exception = nullptr;
    try {
        for (int64_t i = 0; i < num_element; ++i) {
            if (this->thread_pool_ != nullptr) {
                futures.emplace_back(thread_pool_->GeneralEnqueue(add_func, i));
            } else {
                add_func(i);
            }
        }
    } catch (...) {
        first_exception = std::current_exception();
    }

    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (first_exception == nullptr) {
                    first_exception = std::current_exception();
                }
            }
        }
    }
    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
    this->bucket_->Package();
    if (precise_bucket_ != nullptr) {
        this->precise_bucket_->Package();
    }
    if (need_cal_memory_usage) {
        this->cal_memory_usage();
    }
    return {};
}

class PairwiseBucketDistanceProvider final : public DistanceProviderForGraph {
public:
    PairwiseBucketDistanceProvider(std::shared_ptr<BucketInterface> bucket,
                                   BucketIdType bucket_id,
                                   InnerIdType query_id)
        : bucket_(std::move(bucket)), bucket_id_(bucket_id), query_id_(query_id) {
    }

    [[nodiscard]] float
    QueryDistance(InnerIdType id, QueryContext* ctx = nullptr) const override {
        return bucket_->ComputePairVectors(bucket_id_, query_id_, id);
    }

    void
    BatchQueryDistance(float* distances,
                       const InnerIdType* ids,
                       InnerIdType count,
                       QueryContext* ctx = nullptr) const override {
        for (InnerIdType i = 0; i < count; ++i) {
            distances[i] = bucket_->ComputePairVectors(bucket_id_, query_id_, ids[i]);
        }
    }

    [[nodiscard]] float
    PairwiseDistance(InnerIdType id1,
                     InnerIdType id2,
                     const ComputerInterfacePtr& computer = nullptr) const override {
        return bucket_->ComputePairVectors(bucket_id_, id1, id2);
    }

    [[nodiscard]] ComputerInterfacePtr
    FactoryComputerById(InnerIdType id) const override {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "PairwiseBucketDistanceProvider does not support FactoryComputerById");
    }

private:
    std::shared_ptr<BucketInterface> bucket_;
    InnerIdType query_id_;
    BucketIdType bucket_id_;
};

void
IVF::build_bucket_graphs() {
    if (graph_build_threshold_ <= 0) {
        return;
    }
    if (graph_param_ == nullptr) {
        graph_param_ = std::make_shared<GraphDataCellParameter>();
    }

    constexpr int64_t max_degree = 64;
    constexpr uint64_t ef_construction = 300;
    const auto bucket_count = bucket_->bucket_count_;
    bucket_graphs_.resize(bucket_count);

    auto build_one_bucket = [&](BucketIdType b) {
        const auto bucket_size = bucket_->GetBucketSize(b);
        if (bucket_size < graph_build_threshold_) {
            return;
        }

        const auto* inner_ids = bucket_->GetInnerIds(b);
        Vector<InnerIdType> valid_ids(allocator_);
        valid_ids.reserve(bucket_size);
        for (InnerIdType i = 0; i < static_cast<InnerIdType>(bucket_size); ++i) {
            if (inner_ids[i] != std::numeric_limits<InnerIdType>::max()) {
                valid_ids.push_back(i);
            }
        }
        if (valid_ids.size() < static_cast<uint64_t>(graph_build_threshold_)) {
            return;
        }

        const auto effective_degree =
            std::min(max_degree, static_cast<int64_t>(valid_ids.size()) - 1);
        if (effective_degree <= 0) {
            return;
        }

        auto graph = GraphInterface::MakeInstance(graph_param_, this->common_param_);
        graph->Resize(bucket_size);
        graph->SetTotalCount(bucket_size);
        graph->SetMaximumDegree(static_cast<uint32_t>(effective_degree));

        auto mutexes = std::make_shared<EmptyMutex>();
        BasicSearcher searcher(common_param_);

        const auto entry = valid_ids.front();
        graph->InsertNeighborsById(entry, Vector<InnerIdType>(allocator_));
        auto visited = std::make_shared<VisitedList>(bucket_size, allocator_);
        for (uint64_t node_pos = 1; node_pos < valid_ids.size(); ++node_pos) {
            const auto node = valid_ids[node_pos];
            InnerSearchParam search_param;
            search_param.ep = entry;
            search_param.ef = std::min(ef_construction, node_pos);
            search_param.topk = static_cast<int64_t>(search_param.ef);
            PairwiseBucketDistanceProvider distance_provider(bucket_, b, node);
            visited->Reset();
            auto candidates =
                searcher.Search(graph, distance_provider, visited, search_param, nullptr, nullptr);
            mutually_connect_new_element(
                node, candidates, graph, distance_provider, mutexes, allocator_);
        }

        bucket_graphs_[b] = std::move(graph);
    };

    if (this->thread_pool_ != nullptr) {
        std::vector<std::future<void>> futures;
        futures.reserve(bucket_count);
        for (BucketIdType b = 0; b < bucket_count; ++b) {
            futures.emplace_back(this->thread_pool_->GeneralEnqueue(build_one_bucket, b));
        }
        for (auto& future : futures) {
            future.get();
        }
    } else {
        for (BucketIdType b = 0; b < bucket_count; ++b) {
            build_one_bucket(b);
        }
    }
}
void
IVF::fill_location_map() {
    this->location_map_.resize(this->total_elements_ * buckets_per_data_);
    auto bucket_count = this->bucket_->bucket_count_;
    if (precise_bucket_ != nullptr and precise_bucket_->GetBucketCount() != bucket_count) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "basic and precise bucket counts do not match");
    }
    for (BucketIdType i = 0; i < bucket_count; ++i) {
        auto* ids = this->bucket_->GetInnerIds(i);
        auto bucket_size = this->bucket_->GetBucketSize(i);
        InnerIdType* precise_ids = nullptr;
        if (precise_bucket_ != nullptr) {
            auto precise_bucket_size = precise_bucket_->GetBucketSize(i);
            if (precise_bucket_size != bucket_size) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "basic and precise bucket sizes do not match");
            }
            precise_ids = precise_bucket_->GetInnerIds(i);
        }
        for (uint64_t j = 0; j < bucket_size; ++j) {
            if (precise_ids != nullptr and precise_ids[j] != ids[j]) {
                throw VsagException(ErrorType::INTERNAL_ERROR,
                                    "basic and precise bucket inner ids do not match");
            }
            if (ids[j] == std::numeric_limits<InnerIdType>::max()) {
                continue;
            }
            if (ids[j] >= this->total_elements_ * buckets_per_data_) {
                throw VsagException(ErrorType::INTERNAL_ERROR, "invalid inner_id");
            }
            this->location_map_[ids[j] / buckets_per_data_] =
                (static_cast<uint64_t>(i) << LOCATION_SPLIT_BIT) | static_cast<uint64_t>(j);
        }
    }
}

void
IVF::RebuildBucketGraphs() {
    if (graph_build_threshold_ <= 0) {
        throw VsagException(
            ErrorType::UNSUPPORTED_INDEX_OPERATION,
            "RebuildIVFBucketGraphs: index was not configured with graph_build_threshold > 0");
    }
    if (common_param_.data_type_ != DataTypes::DATA_TYPE_FLOAT) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "RebuildIVFBucketGraphs: only supports float32 data type");
    }

    // Build new graphs in temp storage first (exception safety)
    auto old_graphs = std::move(bucket_graphs_);
    bucket_graphs_.resize(old_graphs.size(), nullptr);

    try {
        this->build_bucket_graphs();
        this->cal_memory_usage();
    } catch (...) {
        // Restore old graphs on failure
        bucket_graphs_ = std::move(old_graphs);
        throw;
    }
}
}  // namespace vsag
