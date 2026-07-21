
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

#include "ivf.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <random>
#include <set>

#include "algorithm/inner_index_interface.h"
#include "attr/argparse.h"
#include "attr/executor/executor.h"
#include "datacell/flatten_interface.h"
#include "flat_bucket_searcher.h"
#include "gno_imi_partition.h"
#include "impl/heap/standard_heap.h"
#include "impl/inner_search_param.h"
#include "impl/reasoning/search_reasoning.h"
#include "impl/reorder/flatten_reorder.h"
#include "impl/searcher/basic_searcher.h"
#include "index/index_impl.h"
#include "index_feature_list.h"
#include "inner_string_params.h"
#include "ivf_nearest_partition.h"
#include "query_context.h"
#include "simd/normalize.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "storage/tlv_section.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

static constexpr BucketIdType INVALID_BUCKET_ID = static_cast<BucketIdType>(-1);

static constexpr const char* IVF_PARAMS_TEMPLATE =
    R"(
    {
        "{TYPE_KEY}": "{INDEX_TYPE_IVF}",
        "{IVF_TRAIN_TYPE_KEY}": "{IVF_TRAIN_TYPE_KMEANS}",
        "{USE_ATTRIBUTE_FILTER_KEY}": false,
        "{USE_REORDER_KEY}": false,
        "{BUILD_THREAD_COUNT_KEY}": 1,
        "{BUCKET_PARAMS_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE_KEY}": 0.05,
                "{PCA_DIM_KEY}": 0,
                "{RABITQ_QUANTIZATION_VERSION_KEY}": "standard",
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY}": 32,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY}": 1,
                "{RABITQ_QUANTIZATION_ERROR_RATE_KEY}": 1.9,
                "{USE_FHT_KEY}": false,
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 1
            },
            "{BUCKETS_COUNT_KEY}": 10,
            "{BUCKET_USE_RESIDUAL_KEY}": false
        },
        "{IVF_PARTITION_STRATEGY_PARAMS_KEY}": {
            "{IVF_PARTITION_STRATEGY_TYPE_KEY}": "{IVF_PARTITION_STRATEGY_TYPE_NEAREST}",
            "{IVF_TRAIN_TYPE_KEY}": "{IVF_TRAIN_TYPE_KMEANS}",
            "{IVF_PARTITION_STRATEGY_TYPE_GNO_IMI}": {
                "{GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY}": 10,
                "{GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY}": 10
            }
        },
        "{BUCKET_PER_DATA_KEY}": 1,
        "{USE_REORDER_KEY}": false,
        "{PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH_KEY}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{FAST_ENCODE_RABITQ_KEY}": true,
                "{FAST_ENCODE_RABITQ_ROUNDS_KEY}": 6,
                "{PRODUCT_QUANTIZATION_DIM_KEY}": 0
            }
        },
        "{ATTR_PARAMS_KEY}": {
            "{ATTR_HAS_BUCKETS_KEY}": true
        }
    })";

ParamPtr
IVF::CheckAndMappingExternalParam(const JsonType& external_param,
                                  const IndexCommonParam& common_param) {
    const ConstParamMap external_mapping = {
        {
            IVF_BASE_QUANTIZATION_TYPE,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            IVF_BASE_IO_TYPE,
            {
                BUCKET_PARAMS_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            IVF_BASE_FILE_PATH,
            {
                BUCKET_PARAMS_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            IVF_PRECISE_QUANTIZATION_TYPE,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            IVF_PRECISE_IO_TYPE,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                TYPE_KEY,
            },
        },
        {
            IVF_PRECISE_FILE_PATH,
            {
                PRECISE_CODES_KEY,
                IO_PARAMS_KEY,
                IO_FILE_PATH_KEY,
            },
        },
        {
            IVF_BUCKETS_COUNT,
            {
                BUCKET_PARAMS_KEY,
                BUCKETS_COUNT_KEY,
            },
        },
        {
            IVF_TRAIN_TYPE,
            {
                IVF_PARTITION_STRATEGY_PARAMS_KEY,
                IVF_TRAIN_TYPE_KEY,
            },
        },
        {
            IVF_PARTITION_STRATEGY_TYPE_KEY,
            {
                IVF_PARTITION_STRATEGY_PARAMS_KEY,
                IVF_PARTITION_STRATEGY_TYPE_KEY,
            },
        },
        {
            GNO_IMI_FIRST_ORDER_BUCKETS_COUNT,
            {
                IVF_PARTITION_STRATEGY_PARAMS_KEY,
                IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
                GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY,
            },
        },
        {
            GNO_IMI_SECOND_ORDER_BUCKETS_COUNT,
            {
                IVF_PARTITION_STRATEGY_PARAMS_KEY,
                IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
                GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY,
            },
        },
        {
            BUCKET_PER_DATA_KEY,
            {
                BUCKET_PER_DATA_KEY,
            },
        },
        {
            IVF_USE_REORDER,
            {
                USE_REORDER_KEY,
            },
        },
        {
            IVF_USE_RESIDUAL,
            {
                BUCKET_PARAMS_KEY,
                BUCKET_USE_RESIDUAL_KEY,
            },
        },
        {
            USE_ATTRIBUTE_FILTER,
            {
                USE_ATTRIBUTE_FILTER_KEY,
            },
        },
        {
            IVF_BASE_PQ_DIM,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM_KEY,
            },
        },
        {
            RABITQ_PCA_DIM,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                PCA_DIM_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_QUERY,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY,
            },
        },
        {
            RABITQ_BITS_PER_DIM_BASE,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY,
            },
        },
        {
            RABITQ_VERSION,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_VERSION_KEY,
            },
        },
        {
            RABITQ_ERROR_RATE,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                RABITQ_QUANTIZATION_ERROR_RATE_KEY,
            },
        },
        {
            RABITQ_USE_FHT,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                USE_FHT_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ_ROUNDS,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_ROUNDS_KEY,
            },
        },
        {
            FAST_ENCODE_RABITQ_ROUNDS,
            {
                PRECISE_CODES_KEY,
                QUANTIZATION_PARAMS_KEY,
                FAST_ENCODE_RABITQ_ROUNDS_KEY,
            },
        },
        {
            IVF_THREAD_COUNT,
            {
                BUILD_THREAD_COUNT_KEY,
            },
        },
        {
            TRAIN_SAMPLE_COUNT_KEY,
            {
                TRAIN_SAMPLE_COUNT_KEY,
            },
        },
    };

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("IVF not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(IVF_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::Parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto ivf_parameter = std::make_shared<IVFParameter>();
    ivf_parameter->FromJson(inner_json);

    return ivf_parameter;
}

IVF::IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      buckets_per_data_(param->buckets_per_data),
      location_map_(common_param.allocator_.get()),
      bucket_searcher_(std::make_shared<FlatBucketSearcher>()) {
    this->bucket_ = BucketInterface::MakeInstance(param->bucket_param, common_param);
    if (this->bucket_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bucket init error");
    }
    if (param->ivf_partition_strategy_parameter->partition_strategy_type ==
        IVFPartitionStrategyType::IVF) {
        this->partition_strategy_ = std::make_shared<IVFNearestPartition>(
            bucket_->bucket_count_, common_param, param->ivf_partition_strategy_parameter);
    } else if (param->ivf_partition_strategy_parameter->partition_strategy_type ==
               IVFPartitionStrategyType::GNO_IMI) {
        this->partition_strategy_ = std::make_shared<GNOIMIPartition>(
            common_param, param->ivf_partition_strategy_parameter);
    }
    if (this->use_reorder_) {
        this->reorder_codes_ =
            FlattenInterface::MakeInstance(param->precise_codes_param, common_param);
        reorder_ = std::make_shared<FlattenReorder>(this->reorder_codes_, allocator_);
    }
    if (param->bucket_param->use_residual_) {
        this->bucket_->SetStrategy(partition_strategy_);
    }
    this->thread_pool_ = common_param.thread_pool_;
    if (param->build_thread_count > 1 and this->thread_pool_ == nullptr) {
        this->thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        this->thread_pool_->SetPoolSize(param->build_thread_count);
    }

    if (bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        this->has_raw_vector_ = true;
    }
}

void
IVF::GetCodeByInnerId(InnerIdType inner_id, uint8_t* data) const {
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    this->bucket_->GetCodesById(bucket_id, offset_id, data);
}

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
    if (use_reorder_ && reorder_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        has_fp32 = true;
    }
    if (name == QUANTIZATION_TYPE_VALUE_FP32 or has_fp32) {
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_CAL_DISTANCE_BY_ID);
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
    this->Train(base);
    // TODO(LHT): duplicate
    auto result = this->Add(base);
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
        this->reorder_codes_->Train(data->GetFloat32Vectors(), data->GetNumElements());
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
        if (use_reorder_) {
            this->reorder_codes_->BatchInsertVector(base->GetFloat32Vectors(),
                                                    base->GetNumElements());
        }
        for (int64_t i = 0; i < num_element; ++i) {
            this->label_table_->Insert(i + total_elements_, ids[i]);
        }
        current_num = this->total_elements_;
        this->total_elements_ += num_element;
        if (this->total_elements_ - last_cal_memory_element_ >= cal_memory_element_interval_) {
            need_cal_memory_usage = true;
            last_cal_memory_element_ = this->total_elements_;
        }
        location_map_.resize(this->total_elements_);
    }

    auto add_func = [&](int64_t i) -> void {
        for (int64_t j = 0; j < buckets_per_data_; ++j) {
            const auto* data_ptr = vectors + i * dim_;
            auto idx = i * buckets_per_data_ + j;
            InnerIdType offset_id = bucket_->InsertVector(
                data_ptr, buckets[idx], idx + current_num * buckets_per_data_);
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
    for (int64_t i = 0; i < num_element; ++i) {
        if (this->thread_pool_ != nullptr) {
            auto future = thread_pool_->GeneralEnqueue(add_func, i);
            futures.emplace_back(std::move(future));
        } else {
            add_func(i);
        }
    }

    if (this->thread_pool_ != nullptr) {
        for (auto& future : futures) {
            future.get();
        }
    }
    this->bucket_->Package();
    if (need_cal_memory_usage) {
        this->cal_memory_usage();
    }
    return {};
}

DatasetPtr
IVF::KnnSearch(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter) const {
    SearchRequest req;
    req.mode_ = SearchMode::KNN_SEARCH;
    req.query_ = query;
    req.topk_ = k;
    req.params_str_ = parameters;
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

DatasetPtr
IVF::RangeSearch(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const FilterPtr& filter,
                 int64_t limited_size) const {
    SearchRequest req;
    req.mode_ = SearchMode::RANGE_SEARCH;
    req.query_ = query;
    req.radius_ = radius;
    req.limited_size_ = limited_size;
    req.params_str_ = parameters;
    if (filter != nullptr) {
        req.filter_ = filter;
    }
    return this->SearchWithRequest(req);
}

int64_t
IVF::GetNumElements() const {
    return this->total_elements_ - this->delete_count_;
}

void
IVF::Merge(const std::vector<MergeUnit>& merge_units) {
    this->bucket_->Unpack();
    for (const auto& unit : merge_units) {
        this->merge_one_unit(unit);
    }
    this->fill_location_map();
    this->bucket_->Package();
}

std::pair<BucketIdType, InnerIdType>
IVF::get_location(InnerIdType inner_id) const {
    auto loc = this->location_map_[inner_id];
    constexpr uint64_t mask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;
    auto bucket_id = static_cast<BucketIdType>(loc >> LOCATION_SPLIT_BIT);
    auto offset_id = static_cast<InnerIdType>(loc & mask);
    return {bucket_id, offset_id};
}

uint32_t
IVF::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
    }
    return delete_count;
}

void
IVF::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, offset_id, bucket_id);
}

void
IVF::UpdateAttribute(int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, offset_id, bucket_id, origin_attrs);
}

#define WRITE_DATACELL_WITH_NAME(writer, name, datacell)            \
    datacell_offsets[(name)].SetInt(offset);                        \
    auto datacell##_start = (writer).GetCursor();                   \
    (datacell)->Serialize(writer);                                  \
    auto datacell##_size = (writer).GetCursor() - datacell##_start; \
    datacell_sizes[(name)].SetInt(datacell##_size);                 \
    offset += datacell##_size;

void
IVF::Serialize(StreamWriter& writer) const {
    JsonType datacell_offsets;
    JsonType datacell_sizes;
    uint64_t offset = 0;

    WRITE_DATACELL_WITH_NAME(writer, "bucket", bucket_);
    WRITE_DATACELL_WITH_NAME(writer, "partition_strategy", partition_strategy_);
    WRITE_DATACELL_WITH_NAME(writer, "label_table", label_table_);

    if (use_reorder_) {
        WRITE_DATACELL_WITH_NAME(writer, "reorder_codes", reorder_codes_);
    }

    if (use_attribute_filter_) {
        WRITE_DATACELL_WITH_NAME(writer, "attr_filter_index", attr_filter_index_);
    }

    // serialize footer (introduced since v0.15)
    JsonType basic_info;
    basic_info["total_elements"].SetInt(this->total_elements_);
    basic_info["use_reorder"].SetBool(this->use_reorder_);
    basic_info["is_trained"].SetBool(this->is_trained_);
    basic_info[DIM].SetInt(this->dim_);
    basic_info[EXTRA_INFO_SIZE].SetInt(0);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));

    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, basic_info);
    metadata->Set("datacell_offsets", datacell_offsets);
    metadata->Set("datacell_sizes", datacell_sizes);

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

MetadataPtr
IVF::collect_streaming_header() const {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("format", "vsag_stream_v1");
    metadata->Set("index_name", this->GetName());

    JsonType basic_info;
    basic_info["total_elements"].SetInt(this->total_elements_);
    basic_info["use_reorder"].SetBool(this->use_reorder_);
    basic_info["is_trained"].SetBool(this->is_trained_);
    basic_info[DIM].SetInt(this->dim_);
    basic_info[EXTRA_INFO_SIZE].SetInt(0);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));
    metadata->Set(BASIC_INFO, basic_info);

    JsonType manifest;
    auto bucket_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET);
    auto partition_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_PARTITION_STRATEGY);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    AppendStreamingManifestBlock(manifest,
                                 bucket_tag,
                                 StreamSerializationBlockCurrentVersion(bucket_tag),
                                 StreamSerializationTagCritical(bucket_tag));
    AppendStreamingManifestBlock(manifest,
                                 partition_tag,
                                 StreamSerializationBlockCurrentVersion(partition_tag),
                                 StreamSerializationTagCritical(partition_tag));
    AppendStreamingManifestBlock(manifest,
                                 label_tag,
                                 StreamSerializationBlockCurrentVersion(label_tag),
                                 StreamSerializationTagCritical(label_tag));
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    if (this->use_attribute_filter_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    metadata->Set("block_manifest", manifest);
    metadata->SetEmptyIndex(this->GetNumElements() == 0);
    return metadata;
}

void
IVF::serialize_streaming_body(StreamWriter& writer) const {
    auto bucket_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET);
    auto partition_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_PARTITION_STRATEGY);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);

    WriteStreamingBlock(
        writer, bucket_tag, StreamSerializationTagCritical(bucket_tag), [this](StreamWriter& w) {
            this->bucket_->Serialize(w);
        });
    WriteStreamingBlock(writer,
                        partition_tag,
                        StreamSerializationTagCritical(partition_tag),
                        [this](StreamWriter& w) { this->partition_strategy_->Serialize(w); });
    WriteStreamingBlock(
        writer, label_tag, StreamSerializationTagCritical(label_tag), [this](StreamWriter& w) {
            this->label_table_->Serialize(w);
        });
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->reorder_codes_->Serialize(w);
            });
    }
    if (this->use_attribute_filter_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->attr_filter_index_->Serialize(w);
            });
    }
}

void
IVF::deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    this->read_streaming_body(reader, metadata);
}

void
IVF::load_streaming_body(StreamReader& reader,
                         const MetadataPtr& metadata,
                         const LoadParameters& parameters) {
    (void)parameters;
    this->read_streaming_body(reader, metadata);
}

void
IVF::read_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    auto basic_info = metadata->Get(BASIC_INFO);
    this->total_elements_ = basic_info["total_elements"].GetInt();
    this->use_reorder_ = basic_info["use_reorder"].GetBool();
    this->is_trained_ = basic_info["is_trained"].GetBool();
    if (basic_info.Contains(INDEX_PARAM)) {
        auto index_param = std::make_shared<IVFParameter>();
        index_param->FromString(basic_info[INDEX_PARAM].GetString());
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("IVF index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }

    bool loaded_bucket = false;
    bool loaded_partition = false;
    bool loaded_label_table = false;
    bool loaded_reorder_codes = false;
    bool loaded_attribute_filter = false;

    while (true) {
        auto block_header = StreamBlockHeader::Read(reader);
        if (block_header.IsSectionEnd()) {
            break;
        }
        BoundedForwardReader block_reader(&reader, block_header.value_len);
        if (!StreamSerializationBlockVersionSupported(block_header.tag,
                                                      block_header.block_version)) {
            if (block_header.IsCritical()) {
                throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                    fmt::format("unsupported IVF streaming block version: tag={}, "
                                                "name={}, version={}, flags={}, value_len={}",
                                                block_header.tag,
                                                StreamSerializationTagName(block_header.tag),
                                                block_header.block_version,
                                                block_header.flags,
                                                block_header.value_len));
            }
            block_reader.SkipRemaining();
            continue;
        }

        switch (static_cast<StreamSerializationTag>(block_header.tag)) {
            case StreamSerializationTag::IVF_BUCKET:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->bucket_->Deserialize(block);
                });
                loaded_bucket = true;
                break;
            case StreamSerializationTag::IVF_PARTITION_STRATEGY:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->partition_strategy_->Deserialize(block);
                });
                loaded_partition = true;
                break;
            case StreamSerializationTag::LABEL_TABLE:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->label_table_->Deserialize(block);
                });
                loaded_label_table = true;
                break;
            case StreamSerializationTag::HIGH_PRECISION_CODES:
                if (this->use_reorder_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->reorder_codes_->Deserialize(block);
                        });
                    loaded_reorder_codes = true;
                }
                break;
            case StreamSerializationTag::ATTRIBUTE_FILTER:
                if (this->use_attribute_filter_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->attr_filter_index_->Deserialize(block);
                        });
                    loaded_attribute_filter = true;
                    this->has_attribute_ = true;
                }
                break;
            default:
                if (block_header.IsCritical()) {
                    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        fmt::format("unknown IVF streaming serialization block: "
                                                    "tag={}, name={}, version={}, flags={}, "
                                                    "value_len={}",
                                                    block_header.tag,
                                                    StreamSerializationTagName(block_header.tag),
                                                    block_header.block_version,
                                                    block_header.flags,
                                                    block_header.value_len));
                }
                break;
        }
        block_reader.SkipRemaining();
    }

    if (!loaded_bucket || !loaded_partition || !loaded_label_table) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization required block is missing");
    }
    if (this->use_reorder_ && !loaded_reorder_codes) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization reorder block is missing");
    }
    if (this->use_attribute_filter_ && !loaded_attribute_filter) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization attribute filter block is missing");
    }
    if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        this->has_raw_vector_ = true;
    }
    this->fill_location_map();
    this->cal_memory_usage();
}

#define READ_DATACELL_WITH_NAME(reader, name, datacell)                       \
    reader.PushSeek(datacell_offsets[(name)].GetInt());                       \
    (datacell)->Deserialize((reader).Slice(datacell_sizes[(name)].GetInt())); \
    (reader).PopSeek();

void
IVF::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        StreamReader::ReadObj(buffer_reader, this->total_elements_);
        StreamReader::ReadObj(buffer_reader, this->use_reorder_);
        StreamReader::ReadObj(buffer_reader, this->is_trained_);

        this->bucket_->Deserialize(buffer_reader);
        this->partition_strategy_->Deserialize(buffer_reader);
        this->label_table_->Deserialize(buffer_reader);
        if (use_reorder_) {
            this->reorder_codes_->Deserialize(buffer_reader);
        }

        if (use_attribute_filter_) {
            this->attr_filter_index_->Deserialize(buffer_reader);
            this->has_attribute_ = true;
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        if (metadata->EmptyIndex()) {
            return;
        }

        auto basic_info = metadata->Get(BASIC_INFO);
        this->total_elements_ = basic_info["total_elements"].GetInt();
        this->use_reorder_ = basic_info["use_reorder"].GetBool();
        this->is_trained_ = basic_info["is_trained"].GetBool();
        if (basic_info.Contains(INDEX_PARAM)) {
            auto param_str = basic_info[INDEX_PARAM].GetString();
            auto index_param = std::make_shared<IVFParameter>();
            index_param->FromString(param_str);
            if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                auto message = fmt::format("IVF index parameter not match, current: {}, new: {}",
                                           this->create_param_ptr_->ToString(),
                                           index_param->ToString());
                logger::error(message);
                throw VsagException(ErrorType::INVALID_ARGUMENT, message);
            }
        }

        JsonType datacell_offsets = metadata->Get(DATACELL_OFFSETS);
        logger::debug("datacell_offsets: {}", datacell_offsets.Dump());
        JsonType datacell_sizes = metadata->Get(DATACELL_SIZES);
        logger::debug("datacell_sizes: {}", datacell_sizes.Dump());

        READ_DATACELL_WITH_NAME(buffer_reader, "bucket", this->bucket_);
        READ_DATACELL_WITH_NAME(buffer_reader, "partition_strategy", this->partition_strategy_);
        READ_DATACELL_WITH_NAME(buffer_reader, "label_table", this->label_table_);
        if (use_reorder_) {
            READ_DATACELL_WITH_NAME(buffer_reader, "reorder_codes", this->reorder_codes_);
        }
        if (use_attribute_filter_) {
            READ_DATACELL_WITH_NAME(buffer_reader, "attr_filter_index", this->attr_filter_index_);
            this->has_attribute_ = true;
        }
        if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
            this->has_raw_vector_ = true;
        }
    }
    this->fill_location_map();
    this->cal_memory_usage();
}

InnerSearchParam
IVF::create_search_param(const std::string& parameters, const FilterPtr& filter) const {
    InnerSearchParam param;
    param.is_inner_id_allowed = this->create_search_filter(filter);
    auto search_param = IVFSearchParameters::FromJson(parameters);
    if (search_param.disable_bucket_scan) {
        param.scan_bucket_size = static_cast<BucketIdType>(search_param.scan_buckets_count);
    } else {
        param.scan_bucket_size = std::min(
            static_cast<BucketIdType>(search_param.scan_buckets_count), bucket_->bucket_count_);
    }
    param.disable_bucket_scan = search_param.disable_bucket_scan;
    param.factor = search_param.topk_factor;
    param.enable_reorder = search_param.enable_reorder;
    param.first_order_scan_ratio = search_param.first_order_scan_ratio;
    param.parallel_search_thread_count = search_param.parallel_search_thread_count;
    if (search_param.enable_time_record) {
        param.time_cost = std::make_shared<Timer>();
        param.time_cost->SetThreshold(search_param.timeout_ms);
    }
    return param;
}

DatasetPtr
IVF::route_buckets_only(const DatasetPtr& query,
                        const InnerSearchParam& param,
                        QueryContext& ctx) const {
    const auto num_queries = query->GetNumElements();
    const auto* query_data = query->GetFloat32Vectors();
    const auto buckets_per_query = param.scan_bucket_size;
    const auto candidate_buckets =
        partition_strategy_->ClassifyDatasForSearch(query_data, num_queries, param, &ctx);

    auto result = Dataset::Make();
    if (num_queries == 0 || buckets_per_query == 0) {
        return result->NumElements(0)->Dim(0);
    }

    auto* alloc = (ctx.alloc != nullptr) ? ctx.alloc : allocator_;
    const auto total_slots = num_queries * buckets_per_query;
    auto* ids = static_cast<int64_t*>(alloc->Allocate(sizeof(int64_t) * total_slots));
    auto* distances = static_cast<float*>(alloc->Allocate(sizeof(float) * total_slots));
    const auto dim = partition_strategy_->dim_;
    const auto metric = partition_strategy_->metric_type_;

    Vector<float> centroid(dim, allocator_);
    Vector<float> norm_query(dim, allocator_);
    Vector<float> norm_centroid(dim, allocator_);
    for (int64_t q = 0; q < num_queries; ++q) {
        const auto* query_vec = query_data + q * dim;
        if (metric == MetricType::METRIC_TYPE_COSINE) {
            Normalize(query_vec, norm_query.data(), dim);
        }
        for (int64_t b = 0; b < buckets_per_query; ++b) {
            const auto idx = q * buckets_per_query + b;
            const auto bucket_id = candidate_buckets[idx];
            if (bucket_id == INVALID_BUCKET_ID) {
                ids[idx] = -1;
                distances[idx] = std::numeric_limits<float>::infinity();
                continue;
            }
            partition_strategy_->GetCentroid(bucket_id, centroid);
            float dist = 0.0F;
            if (metric == MetricType::METRIC_TYPE_L2SQR) {
                for (int64_t d = 0; d < dim; ++d) {
                    auto diff = query_vec[d] - centroid[d];
                    dist += diff * diff;
                }
            } else if (metric == MetricType::METRIC_TYPE_COSINE) {
                Normalize(centroid.data(), norm_centroid.data(), dim);
                for (int64_t d = 0; d < dim; ++d) {
                    dist += norm_query[d] * norm_centroid[d];
                }
                dist = 1.0F - dist;
            } else {
                for (int64_t d = 0; d < dim; ++d) {
                    dist += query_vec[d] * centroid[d];
                }
                dist = 1.0F - dist;
            }
            ids[idx] = static_cast<int64_t>(bucket_id);
            distances[idx] = dist;
        }
    }

    return result->NumElements(num_queries)
        ->Dim(buckets_per_query)
        ->Ids(ids)
        ->Distances(distances)
        ->Owner(true, alloc);
}

DatasetPtr
IVF::reorder(int64_t topk,
             DistHeapPtr& input,
             const float* query,
             const InnerSearchParam& param,
             QueryContext& ctx,
             ReasoningContext* reasoning_ctx) const {
    auto reorder_heap = reorder_->Reorder(input, query, topk, ctx);
    auto dataset_results = this->pack_knn_result(reorder_heap, ctx.alloc);

    this->AttachReasoningReport(dataset_results, reasoning_ctx);

    return dataset_results;
}

InnerIndexPtr
IVF::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<IVF>(this->create_param_ptr_, param);
    IVFPartitionStrategy::Clone(this->partition_strategy_, index->partition_strategy_);
    this->bucket_->ExportModel(index->bucket_);
    if (use_reorder_) {
        this->reorder_codes_->ExportModel(index->reorder_codes_);
    }
    index->is_trained_ = this->is_trained_;
    return index;
}

template <InnerSearchMode mode>
DistHeapPtr
IVF::search(const DatasetPtr& query,
            const InnerSearchParam& param,
            QueryContext& ctx,
            ReasoningContext* reasoning_ctx) const {
    const auto* query_data = query->GetFloat32Vectors();
    Vector<float> normalize_data(dim_, allocator_);
    auto candidate_buckets =
        partition_strategy_->ClassifyDatasForSearch(query_data, 1, param, &ctx);
    if (reasoning_ctx != nullptr) {
        reasoning_ctx->RecordBucketSelection(candidate_buckets);
    }
    auto computer = bucket_->FactoryComputer(query_data);

    int64_t topk = param.topk;
    if constexpr (mode == RANGE_SEARCH) {
        topk = param.range_search_limit_size;
        if (topk < 0) {
            topk = this->GetNumElements();
        }
    }
    // Scale topk to ensure sufficient candidates after deduplication when buckets_per_data_ > 1
    int64_t origin_topk = topk;
    if (buckets_per_data_ > 1) {
        if (topk <= std::numeric_limits<int64_t>::max() / buckets_per_data_) {
            topk *= buckets_per_data_;
        } else {
            topk = std::numeric_limits<int64_t>::max();
        }
    }

    DistHeapPtr search_result = nullptr;

    auto bucket_count = candidate_buckets.size();
    auto search_thread_count = param.parallel_search_thread_count;
    if (this->thread_pool_ == nullptr) {
        search_thread_count = 1;
    }
    std::vector<DistHeapPtr> heaps(search_thread_count);
    std::atomic<uint64_t> cur_bucket_num(0);
    auto search_func = [&](int64_t thread_id) -> void {
        heaps[thread_id] = DistanceHeap::MakeInstanceBySize<true, false>(this->allocator_, topk);
        auto& heap = heaps[thread_id];
        Vector<float> dist(allocator_);
        uint64_t i = cur_bucket_num.fetch_add(1);
        for (; i < bucket_count; i = cur_bucket_num.fetch_add(1)) {
            if (param.time_cost != nullptr and param.time_cost->CheckOvertime() and
                ctx.stats != nullptr) {
                ctx.stats->is_timeout.store(true, std::memory_order_relaxed);
                break;
            }
            auto bucket_id = candidate_buckets[i];
            if (bucket_id == INVALID_BUCKET_ID) {
                break;
            }
            bucket_searcher_->Search(bucket_id,
                                     bucket_,
                                     computer,
                                     param,
                                     thread_id,
                                     topk,
                                     buckets_per_data_,
                                     heap,
                                     dist,
                                     reasoning_ctx);
        }
    };
    std::vector<std::future<void>> futures;
    if (this->thread_pool_ != nullptr and search_thread_count > 1) {
        for (int64_t thread_id = 0; thread_id < search_thread_count; ++thread_id) {
            auto future = this->thread_pool_->GeneralEnqueue(search_func, thread_id);
            futures.emplace_back(std::move(future));
        }
    } else {
        search_func(0);
        search_result = heaps[0];
    }

    if (this->thread_pool_ != nullptr and search_thread_count > 1) {
        for (auto& future : futures) {
            future.get();
        }
        search_result = DistanceHeap::MakeInstanceBySize<true, true>(this->allocator_, topk);
        for (auto& heap : heaps) {
            auto size = heap->Size();
            const auto* data = heap->GetData();
            for (int i = 0; i < size; ++i) {
                search_result->Push(data[i]);
            }
        }
    }

    // Deduplicate ids when buckets_per_data_ > 1
    if (buckets_per_data_ > 1) {
        std::unordered_map<InnerIdType, float> id_to_min_dist;
        while (!search_result->Empty()) {
            const auto& [dist_val, id] = search_result->Top();
            auto origin_id = id / buckets_per_data_;
            // Keep the smallest distance for each id
            if (id_to_min_dist.find(origin_id) == id_to_min_dist.end() ||
                dist_val < id_to_min_dist[origin_id]) {
                id_to_min_dist[origin_id] = dist_val;
            }
            search_result->Pop();
        }

        auto cur_heap_top2 = std::numeric_limits<float>::max();
        for (const auto& [origin_id, dist_val] : id_to_min_dist) {
            if (dist_val < cur_heap_top2) {
                search_result->Push(dist_val, origin_id);
            }
            if (search_result->Size() > origin_topk) {
                search_result->Pop();
            }
            if (not search_result->Empty() and search_result->Size() == origin_topk) {
                cur_heap_top2 = search_result->Top().first;
            }
        }
    }

    return search_result;
}

void
IVF::merge_one_unit(const MergeUnit& unit) {
    check_merge_illegal(unit);
    const auto other_index = std::dynamic_pointer_cast<IVF>(
        std::dynamic_pointer_cast<IndexImpl<IVF>>(unit.index)->GetInnerIndex());
    auto bucket_bias = static_cast<InnerIdType>(this->total_elements_ * this->buckets_per_data_);
    this->label_table_->MergeOther(other_index->label_table_, unit.id_map_func);
    other_index->bucket_->Unpack();
    this->bucket_->MergeOther(other_index->bucket_, bucket_bias);
    other_index->bucket_->Package();

    if (this->use_reorder_) {
        this->reorder_codes_->MergeOther(other_index->reorder_codes_, this->total_elements_);
    }
    this->total_elements_ += other_index->total_elements_;
}

void
IVF::check_merge_illegal(const vsag::MergeUnit& unit) const {
    auto index = std::dynamic_pointer_cast<IndexImpl<IVF>>(unit.index);
    if (index == nullptr) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "Merge Failed: index type not match, try to merge a non-ivf index to an IVF index");
    }
    auto other_ivf_index = std::dynamic_pointer_cast<IVF>(
        std::dynamic_pointer_cast<IndexImpl<IVF>>(unit.index)->GetInnerIndex());
    if (other_ivf_index->use_reorder_ != this->use_reorder_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format(
                "Merge Failed: ivf use_reorder not match, current index is {}, other index is {}",
                this->use_reorder_,
                other_ivf_index->use_reorder_));
    }
    auto cur_model = this->ExportModel(index->GetCommonParam());
    std::stringstream ss1;
    std::stringstream ss2;
    IOStreamWriter writer1(ss1);
    cur_model->Serialize(writer1);

    cur_model.reset();
    auto other_model = other_ivf_index->ExportModel(index->GetCommonParam());
    IOStreamWriter writer2(ss2);
    other_model->Serialize(writer2);

    other_model.reset();

    if (not check_equal_on_string_stream(ss1, ss2)) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "Merge Failed: IVF model not match, try to merge a different model ivf index");
    }
}

DatasetPtr
IVF::SearchWithRequest(const SearchRequest& request) const {
    SearchStatistics stats;
    QueryContext ctx{.alloc = request.search_allocator_, .stats = &stats};

    bool is_range = (request.mode_ == SearchMode::RANGE_SEARCH);

    auto param = this->create_search_param(request.params_str_, request.filter_);

    auto query = request.query_;
    if (param.disable_bucket_scan) {
        CHECK_ARGUMENT(query != nullptr, "query dataset cannot be null");
        CHECK_ARGUMENT(query->GetNumElements() >= 1,
                       "disable bucket scan requires at least one query");
        CHECK_ARGUMENT(query->GetFloat32Vectors() != nullptr,
                       "query float32 vectors cannot be null");
        CHECK_ARGUMENT(query->GetDim() == this->dim_, "query dimension must match index dimension");
        auto result = this->route_buckets_only(query, param, ctx);
        result->Statistics(stats.Dump());
        return result;
    }

    if (request.enable_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        auto& schema = this->attr_filter_index_->field_type_map_;
        auto expr = AstParse(request.attribute_filter_str_, &schema);
        for (int64_t i = 0; i < param.parallel_search_thread_count; ++i) {
            auto executor =
                Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
            executor->Init();
            param.executors.emplace_back(executor);
        }
    }
    std::shared_ptr<ReasoningContext> reasoning_ctx;
    if (not request.expected_labels_.empty()) {
        reasoning_ctx = std::make_shared<ReasoningContext>(this->allocator_);
        reasoning_ctx->SetSearchParams(
            request.topk_, "IVF", use_reorder_, request.filter_ != nullptr);

        UnorderedMap<int64_t, InnerIdType> label_to_inner_id(this->allocator_);
        std::vector<std::tuple<InnerIdType, BucketIdType, InnerIdType>> locations;
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            locations.reserve(request.expected_labels_.size());
            for (const auto& label : request.expected_labels_) {
                auto [success, inner_id] = this->label_table_->TryGetIdByLabel(label, true);
                if (success) {
                    label_to_inner_id[label] = inner_id;
                    auto [bucket_id, offset_id] = this->get_location(inner_id);
                    locations.emplace_back(inner_id, bucket_id, offset_id);
                }
            }
        }

        Vector<int64_t> expected_labels_vec(this->allocator_);
        expected_labels_vec.reserve(request.expected_labels_.size());
        for (const auto& label : request.expected_labels_) {
            expected_labels_vec.push_back(label);
        }
        reasoning_ctx->InitializeExpectedTargets(expected_labels_vec, label_to_inner_id);

        const auto* query_data = query->GetFloat32Vectors();
        auto computer = this->bucket_->FactoryComputer(query_data);
        for (const auto& [inner_id, bucket_id, offset_id] : locations) {
            float dist = this->bucket_->QueryOneById(computer, bucket_id, offset_id);
            reasoning_ctx->SetTrueDistance(inner_id, dist);
        }
        ctx.reasoning_ctx = reasoning_ctx.get();
    }

    if (is_range) {
        param.search_mode = RANGE_SEARCH;
        param.radius = request.radius_;
        param.range_search_limit_size = static_cast<int>(request.limited_size_);
        if (use_reorder_ and param.enable_reorder and request.limited_size_ > 0) {
            CHECK_ARGUMENT(param.factor > 0.0F,
                           fmt::format("factor must be positive when use_reorder is true, got {}",
                                       param.factor));
            param.range_search_limit_size =
                static_cast<int>(param.factor * static_cast<float>(request.limited_size_));
        }
        auto search_result = this->search<RANGE_SEARCH>(query, param, ctx, reasoning_ctx.get());
        if (use_reorder_ and param.enable_reorder) {
            int64_t k = (request.limited_size_ > 0) ? request.limited_size_
                                                    : static_cast<int64_t>(search_result->Size());
            auto result = reorder(
                k, search_result, query->GetFloat32Vectors(), param, ctx, reasoning_ctx.get());
            result->Statistics(stats.Dump());
            return result;
        }
        auto dataset_results = this->pack_knn_result(search_result, ctx.alloc);
        dataset_results->Statistics(stats.Dump());
        this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
        return dataset_results;
    }

    // KNN mode
    param.search_mode = KNN_SEARCH;
    param.topk = request.topk_;
    if (use_reorder_ and param.enable_reorder) {
        CHECK_ARGUMENT(
            param.factor > 0.0F,
            fmt::format("factor must be positive when use_reorder is true, got {}", param.factor));
        param.topk = static_cast<int64_t>(param.factor * static_cast<float>(request.topk_));
    }
    auto search_result = this->search<KNN_SEARCH>(query, param, ctx, reasoning_ctx.get());
    if (use_reorder_ and param.enable_reorder) {
        auto result = reorder(request.topk_,
                              search_result,
                              query->GetFloat32Vectors(),
                              param,
                              ctx,
                              reasoning_ctx.get());
        result->Statistics(stats.Dump());
        return result;
    }
    if (search_result == nullptr || search_result->Empty()) {
        auto dataset_results = DatasetImpl::MakeEmptyDataset();
        this->AttachReasoningReport(dataset_results, reasoning_ctx.get());
        dataset_results->Statistics(stats.Dump());
        return dataset_results;
    }

    auto dataset_results = this->pack_knn_result(search_result, ctx.alloc);
    dataset_results->Statistics(stats.Dump());

    this->AttachReasoningReport(dataset_results, reasoning_ctx.get());

    return dataset_results;
}

void
IVF::AttachReasoningReport(const DatasetPtr& dataset_results,
                           ReasoningContext* reasoning_ctx) const {
    if (reasoning_ctx == nullptr) {
        return;
    }
    auto count = dataset_results->GetNumElements();
    if (count > 0 and dataset_results->GetIds() != nullptr) {
        Vector<InnerIdType> result_inner_ids(static_cast<uint64_t>(count), this->allocator_);
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            for (int64_t i = 0; i < count; ++i) {
                result_inner_ids[i] =
                    this->label_table_->GetIdByLabel(dataset_results->GetIds()[i]);
            }
        }
        reasoning_ctx->MarkResult(result_inner_ids);
    }
    reasoning_ctx->DiagnoseExpectedTargets();
    dataset_results->Reasoning(reasoning_ctx->GenerateReport());
}

void
IVF::fill_location_map() {
    this->location_map_.resize(this->total_elements_ * buckets_per_data_);
    auto bucket_count = this->bucket_->bucket_count_;
    for (BucketIdType i = 0; i < bucket_count; ++i) {
        auto* ids = this->bucket_->GetInnerIds(i);
        auto bucket_size = this->bucket_->GetBucketSize(i);
        for (uint64_t j = 0; j < bucket_size; ++j) {
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
IVF::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    auto [bucket_id, bucket_offset] = this->get_location(inner_id);
    this->attr_filter_index_->GetAttribute(bucket_id, bucket_offset, attr);
}

DatasetPtr
IVF::CalDistanceById(const float* query,
                     const int64_t* ids,
                     int64_t count,
                     bool calculate_precise_distance) const {
    if (this->use_reorder_ && calculate_precise_distance) {
        return this->cal_distance_by_id(query, ids, count, this->reorder_codes_);
    }
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
    result->Distances(distances);
    auto computer = this->bucket_->FactoryComputer(query);
    for (int64_t i = 0; i < count; ++i) {
        bool success = false;
        InnerIdType inner_id = 0;
        {
            std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
            std::tie(success, inner_id) = this->label_table_->TryGetIdByLabel(ids[i]);
        }
        if (not success) {
            distances[i] = -1;
            continue;
        }
        auto [bucket_id, offset_id] = this->get_location(inner_id);
        distances[i] = this->bucket_->QueryOneById(computer, bucket_id, offset_id);
    }
    return result;
}

float
IVF::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    auto [success, inner_id] = this->label_table_->TryGetIdByLabel(id);
    if (not success) {
        return -1.0F;
    }
    if (this->use_reorder_ && calculate_precise_distance) {
        float dist = 0.0F;
        auto computer = this->reorder_codes_->FactoryComputer(query);
        this->reorder_codes_->Query(&dist, computer, &inner_id, 1);
        return dist;
    }
    auto computer = this->bucket_->FactoryComputer(query);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    return this->bucket_->QueryOneById(computer, bucket_id, offset_id);
}

void
IVF::GetVectorByInnerId(InnerIdType inner_id, float* data) const {
    auto [bucket_id, bucket_offset] = this->get_location(inner_id);
    this->bucket_->GetCodesById(bucket_id, bucket_offset, reinterpret_cast<uint8_t*>(data));
}

float
calculate_percentile(const std::vector<float>& sorted_data, float percentile) {
    uint64_t n = sorted_data.size();
    float index = percentile * static_cast<float>(n - 1);
    auto floor_index = static_cast<uint64_t>(std::floor(index));
    uint64_t ceil_index = floor_index + 1;

    if (ceil_index >= n) {
        return sorted_data[floor_index];
    }

    float fractional = index - static_cast<float>(floor_index);
    return sorted_data[floor_index] * (1.0F - fractional) + sorted_data[ceil_index] * fractional;
}

JsonType
get_data_stats(const Vector<float>& data) {
    JsonType json;
    if (data.empty()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Vector cannot be empty.");
    }

    float sum = 0.0;
    for (float val : data) {
        sum += val;
    }
    float mean = sum / static_cast<float>(data.size());
    json["mean"].SetFloat(mean);

    float sq_diff_sum = 0.0;
    for (float val : data) {
        sq_diff_sum += (val - mean) * (val - mean);
    }
    float variance = sq_diff_sum / static_cast<float>(data.size());
    json["std"].SetFloat(std::sqrt(variance));

    float min_val = *std::min_element(data.begin(), data.end());
    json["min"].SetFloat(min_val);
    float max_val = *std::max_element(data.begin(), data.end());
    json["max"].SetFloat(max_val);

    std::vector<float> sorted_data(data.begin(), data.end());
    std::sort(sorted_data.begin(), sorted_data.end());

    float q25 = calculate_percentile(sorted_data, 0.25);
    float q50 = calculate_percentile(sorted_data, 0.5);
    float q75 = calculate_percentile(sorted_data, 0.75);
    json["q25"].SetFloat(q25);
    json["q50"].SetFloat(q50);
    json["q75"].SetFloat(q75);

    return json;
}

std::string
IVF::GetStats() const {
    JsonType stats;
    // bucket_radius
    stats["bucket_count"].SetInt(this->bucket_->bucket_count_);
    Vector<float> centroids(this->dim_, allocator_);
    Vector<float> bucket_counts(allocator_);
    Vector<float> bucket_radius(allocator_);
    for (int i = 0; i < this->bucket_->bucket_count_; ++i) {
        auto size = bucket_->GetBucketSize(i);
        if (size == 0) {
            bucket_counts.push_back(0);
            continue;
        }
        bucket_counts.push_back(static_cast<float>(size));
        Vector<float> dists(size, allocator_);
        partition_strategy_->GetCentroid(i, centroids);
        auto computer = bucket_->FactoryComputer(centroids.data());
        bucket_->ScanBucketById(dists.data(), computer, i);
        float max_distance = *std::max_element(dists.begin(), dists.end());
        bucket_radius.push_back(max_distance);
    }
    // bucket_count_std
    stats["bucket_num"].SetJson(get_data_stats(bucket_counts));
    // bucket_radius
    stats["bucket_radius"].SetJson(get_data_stats(bucket_radius));
    return stats.Dump(4);
}

std::string
IVF::AnalyzeIndexBySearch(const SearchRequest& request) {
    JsonType stats;
    auto querys = request.query_;
    auto topk = std::min(request.topk_, GetNumElements());
    auto num_elements = querys->GetNumElements();
    auto param_str = request.params_str_;
    // quantization error
    this->analyze_quantizer(stats, querys->GetFloat32Vectors(), num_elements, topk, param_str);
    return stats.Dump(4);
}

void
IVF::cal_memory_usage() {
    auto memory = sizeof(IVF);
    memory += this->bucket_->GetMemoryUsage();
    if (use_reorder_) {
        memory += this->reorder_codes_->GetMemoryUsage();
    }
    if (this->extra_info_size_ > 0 and this->extra_infos_ != nullptr) {
        memory += this->extra_infos_->GetMemoryUsage();
    }
    memory += this->label_table_->GetMemoryUsage();
    memory += location_map_.size() * sizeof(uint64_t);
    memory += partition_strategy_->GetMemoryUsage();
    std::unique_lock lock(this->memory_usage_mutex_);
    this->current_memory_usage_.store(memory);
}

uint64_t
IVF::GetMemoryUsage() const {
    uint64_t memory = 0;
    {
        std::shared_lock lock(this->memory_usage_mutex_);
        memory = this->current_memory_usage_.load();
    }
    if (this->attr_filter_index_ != nullptr) {
        memory += this->attr_filter_index_->GetMemoryUsage();
    }
    return memory;
}

}  // namespace vsag
