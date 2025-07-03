
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

#include <fstream>
#include <set>

#include "attr/executor/executor.h"
#include "attr/expression_visitor.h"
#include "impl/basic_searcher.h"
#include "impl/heap/standard_heap.h"
#include "impl/reorder.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "ivf_partition/gno_imi_partition.h"
#include "ivf_partition/ivf_nearest_partition.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/util_functions.h"

namespace vsag {
static constexpr const int64_t MAX_TRAIN_SIZE = 65536L;
static constexpr const char* IVF_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_IVF}",
        "{IVF_TRAIN_TYPE_KEY}": "{IVF_TRAIN_TYPE_KMEANS}",
        "{IVF_USE_ATTRIBUTE_FILTER_KEY}": false,
        "{IVF_USE_REORDER_KEY}": false,
        "{BUCKET_PARAMS_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_MEMORY_IO}"
            },
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY}": 32,
                "{PRODUCT_QUANTIZATION_DIM}": 0
            },
            "{BUCKETS_COUNT_KEY}": 10,
            "{BUCKET_USE_RESIDUAL}": false
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
        "{IVF_USE_REORDER_KEY}": false,
        "{IVF_PRECISE_CODES_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}",
                "{IO_FILE_PATH}": "{DEFAULT_FILE_PATH_VALUE}"
            },
            "codes_type": "flatten_codes",
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{PRODUCT_QUANTIZATION_DIM}": 0
            }
        }
    })";

ParamPtr
IVF::CheckAndMappingExternalParam(const JsonType& external_param,
                                  const IndexCommonParam& common_param) {
    const std::unordered_map<std::string, std::vector<std::string>> external_mapping = {
        {
            IVF_BASE_QUANTIZATION_TYPE,
            {BUCKET_PARAMS_KEY, QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY},
        },
        {
            IVF_BASE_IO_TYPE,
            {BUCKET_PARAMS_KEY, IO_PARAMS_KEY, IO_TYPE_KEY},
        },
        {
            IVF_PRECISE_QUANTIZATION_TYPE,
            {IVF_PRECISE_CODES_KEY, QUANTIZATION_PARAMS_KEY, QUANTIZATION_TYPE_KEY},
        },
        {
            IVF_PRECISE_IO_TYPE,
            {IVF_PRECISE_CODES_KEY, IO_PARAMS_KEY, IO_TYPE_KEY},
        },
        {
            IVF_BUCKETS_COUNT,
            {BUCKET_PARAMS_KEY, BUCKETS_COUNT_KEY},
        },
        {
            IVF_TRAIN_TYPE,
            {IVF_PARTITION_STRATEGY_PARAMS_KEY, IVF_TRAIN_TYPE_KEY},
        },
        {
            IVF_PARTITION_STRATEGY_TYPE_KEY,
            {IVF_PARTITION_STRATEGY_PARAMS_KEY, IVF_PARTITION_STRATEGY_TYPE_KEY},
        },
        {
            GNO_IMI_FIRST_ORDER_BUCKETS_COUNT,
            {IVF_PARTITION_STRATEGY_PARAMS_KEY,
             IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
             GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY},
        },
        {
            GNO_IMI_SECOND_ORDER_BUCKETS_COUNT,
            {IVF_PARTITION_STRATEGY_PARAMS_KEY,
             IVF_PARTITION_STRATEGY_TYPE_GNO_IMI,
             GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY},
        },
        {
            BUCKET_PER_DATA_KEY,
            {BUCKET_PER_DATA_KEY},
        },
        {
            IVF_USE_REORDER,
            {IVF_USE_REORDER_KEY},
        },
        {IVF_USE_RESIDUAL, {BUCKET_PARAMS_KEY, BUCKET_USE_RESIDUAL}},
        {
            IVF_USE_ATTRIBUTE_FILTER,
            {IVF_USE_ATTRIBUTE_FILTER_KEY},
        },
        {
            IVF_BASE_PQ_DIM,
            {
                BUCKET_PARAMS_KEY,
                QUANTIZATION_PARAMS_KEY,
                PRODUCT_QUANTIZATION_DIM,
            },
        },
    };

    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("IVF not support {} datatype", DATATYPE_INT8));
    }

    std::string str = format_map(IVF_PARAMS_TEMPLATE, DEFAULT_MAP);
    auto inner_json = JsonType::parse(str);
    mapping_external_param_to_inner(external_param, external_mapping, inner_json);

    auto ivf_parameter = std::make_shared<IVFParameter>();
    ivf_parameter->FromJson(inner_json);

    return ivf_parameter;
}

IVF::IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param), buckets_per_data_(param->buckets_per_data) {
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
    this->use_reorder_ = param->use_reorder;
    if (this->use_reorder_) {
        this->reorder_codes_ = FlattenInterface::MakeInstance(param->flatten_param, common_param);
    }
    this->use_residual_ = param->bucket_param->use_residual_;
    this->use_attribute_filter_ = param->use_attribute_filter;
    if (this->use_attribute_filter_) {
        this->attr_filter_index_ =
            AttributeInvertedInterface::MakeInstance(allocator_, true /*have_bucket*/);
    }
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
    });

    auto name = this->bucket_->GetQuantizerName();
    if (name != QUANTIZATION_TYPE_VALUE_FP32 and name != QUANTIZATION_TYPE_VALUE_BF16) {
        this->index_feature_list_->SetFeature(IndexFeature::NEED_TRAIN);
    } else {
        this->index_feature_list_->SetFeatures({
            IndexFeature::SUPPORT_RANGE_SEARCH,
            IndexFeature::SUPPORT_RANGE_SEARCH_WITH_ID_FILTER,
        });
    }
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_CLONE,
        IndexFeature::SUPPORT_EXPORT_MODEL,
        IndexFeature::SUPPORT_MERGE_INDEX,
    });

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
    partition_strategy_->Train(data);
    const auto* data_ptr = data->GetFloat32Vectors();
    Vector<float> train_data_buffer(allocator_);
    auto num_element = std::min(data->GetNumElements(), MAX_TRAIN_SIZE);
    if (use_residual_) {
        train_data_buffer.resize(num_element * dim_);
        if (metric_ == MetricType::METRIC_TYPE_COSINE) {
            for (int i = 0; i < num_element; ++i) {
                Normalize(data_ptr + i * dim_, train_data_buffer.data() + i * dim_, dim_);
            }
            data_ptr = train_data_buffer.data();
        }
        Vector<float> centroid(dim_, allocator_);
        auto buckets = partition_strategy_->ClassifyDatas(data_ptr, num_element, 1);
        for (int i = 0; i < num_element; ++i) {
            partition_strategy_->GetCentroid(buckets[i], centroid);
            for (int j = 0; j < dim_; ++j) {
                train_data_buffer[i * dim_ + j] = data_ptr[i * dim_ + j] - centroid[j];
            }
        }
        data_ptr = train_data_buffer.data();
    }
    this->bucket_->Train(data_ptr, num_element);
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
    auto buckets = partition_strategy_->ClassifyDatas(vectors, num_element, buckets_per_data_);
    Vector<float> normalize_data(dim_, allocator_);
    Vector<float> residual_data(dim_, allocator_);
    Vector<float> centroid(dim_, allocator_);
    int64_t current_num;
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
    }
    for (int64_t i = 0; i < num_element; ++i) {
        const auto* data_ptr = vectors + i * dim_;
        for (int64_t j = 0; j < buckets_per_data_; ++j) {
            auto idx = i * buckets_per_data_ + j;

            if (use_residual_) {
                partition_strategy_->GetCentroid(buckets[idx], centroid);
                if (metric_ == MetricType::METRIC_TYPE_COSINE) {
                    Normalize(data_ptr, normalize_data.data(), dim_);
                    data_ptr = normalize_data.data();
                }
                FP32Sub(data_ptr, centroid.data(), residual_data.data(), dim_);
                bucket_->InsertVector(residual_data.data(),
                                      buckets[idx],
                                      idx + current_num * buckets_per_data_,
                                      centroid.data());
            } else {
                bucket_->InsertVector(
                    data_ptr, buckets[idx], idx + current_num * buckets_per_data_);
            }
        }
    }

    this->bucket_->Package();
    if (use_attribute_filter_ and this->attr_filter_index_ != nullptr and attr_sets != nullptr) {
        for (uint64_t i = 0; i < this->bucket_->bucket_count_; ++i) {
            auto bucket_id = static_cast<BucketIdType>(i);
            auto bucket_size = this->bucket_->GetBucketSize(bucket_id);
            if (bucket_size == 0) {
                continue;
            }
            auto* inner_ids = this->bucket_->GetInnerIds(bucket_id);
            for (InnerIdType j = 0; j < bucket_size; ++j) {
                auto inner_id = inner_ids[j];
                const auto& attr_set = attr_sets[inner_id - current_num];
                this->attr_filter_index_->InsertWithBucket(attr_set, j, bucket_id);
            }
        }
    }
    return {};
}

DatasetPtr
IVF::KnnSearch(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const FilterPtr& filter) const {
    auto param = this->create_search_param(parameters, filter);
    param.search_mode = KNN_SEARCH;
    param.topk = k;
    if (use_reorder_) {
        param.topk = static_cast<int64_t>(param.factor * static_cast<float>(k));
    }
    auto search_result = this->search<KNN_SEARCH>(query, param);
    if (use_reorder_) {
        return reorder(k, search_result, query->GetFloat32Vectors());
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, labels] = create_fast_dataset(count, allocator_);
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        labels[j] = label_table_->GetLabelById(search_result->Top().second);
        search_result->Pop();
    }
    return std::move(dataset_results);
}

DatasetPtr
IVF::RangeSearch(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 const FilterPtr& filter,
                 int64_t limited_size) const {
    auto param = this->create_search_param(parameters, filter);
    param.search_mode = RANGE_SEARCH;
    param.radius = radius;
    param.range_search_limit_size = static_cast<int>(limited_size);
    if (use_reorder_ and limited_size > 0) {
        param.range_search_limit_size =
            static_cast<int>(param.factor * static_cast<float>(limited_size));
    }
    auto search_result = this->search<RANGE_SEARCH>(query, param);
    if (use_reorder_) {
        int64_t k = (limited_size > 0) ? limited_size : static_cast<int64_t>(search_result->Size());
        return reorder(k, search_result, query->GetFloat32Vectors());
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, labels] = create_fast_dataset(count, allocator_);
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        labels[j] = label_table_->GetLabelById(search_result->Top().second);
        search_result->Pop();
    }
    return std::move(dataset_results);
}

int64_t
IVF::GetNumElements() const {
    return this->total_elements_;
}

void
IVF::Merge(const std::vector<MergeUnit>& merge_units) {
    this->bucket_->Unpack();
    for (const auto& unit : merge_units) {
        this->merge_one_unit(unit);
    }
    this->bucket_->Package();
}

#define WRITE_DATACELL_WITH_NAME(writer, name, datacell)            \
    datacell_offsets[(name)] = offset;                              \
    auto datacell##_start = (writer).GetCursor();                   \
    (datacell)->Serialize(writer);                                  \
    auto datacell##_size = (writer).GetCursor() - datacell##_start; \
    datacell_sizes[(name)] = datacell##_size;                       \
    offset += datacell##_size;

void
IVF::Serialize(StreamWriter& writer) const {
    // FIXME(wxyu): only for testing, remove before merge into the main branch
    // if (not Options::Instance().new_version()) {
    //     StreamWriter::WriteObj(writer, this->total_elements_);
    //     StreamWriter::WriteObj(writer, this->use_reorder_);
    //     StreamWriter::WriteObj(writer, this->is_trained_);

    //     this->bucket_->Serialize(writer);
    //     this->partition_strategy_->Serialize(writer);
    //     this->label_table_->Serialize(writer);
    //     if (use_reorder_) {
    //         this->reorder_codes_->Serialize(writer);
    //     }
    //     if (use_attribute_filter_) {
    //         this->attr_filter_index_->Serialize(writer);
    //     }
    //     return;
    // }

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
    basic_info["total_elements"] = this->total_elements_;
    basic_info["use_reorder"] = this->use_reorder_;
    basic_info["is_trained"] = this->is_trained_;

    auto metadata = std::make_shared<Metadata>();
    metadata->Set("basic_info", basic_info);
    metadata->Set("datacell_offsets", datacell_offsets);
    metadata->Set("datacell_sizes", datacell_sizes);

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

#define READ_DATACELL_WITH_NAME(reader, name, datacell)              \
    reader.PushSeek(datacell_offsets[(name)].get<uint64_t>());       \
    (datacell)->Deserialize((reader).Slice(datacell_sizes[(name)])); \
    (reader).PopSeek();

void
IVF::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        StreamReader::ReadObj(reader, this->total_elements_);
        StreamReader::ReadObj(reader, this->use_reorder_);
        StreamReader::ReadObj(reader, this->is_trained_);

        this->bucket_->Deserialize(reader);
        this->partition_strategy_->Deserialize(reader);
        this->label_table_->Deserialize(reader);
        if (use_reorder_) {
            this->reorder_codes_->Deserialize(reader);
        }

        if (use_attribute_filter_) {
            this->attr_filter_index_->Deserialize(reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        if (metadata->EmptyIndex()) {
            return;
        }

        auto basic_info = metadata->Get("basic_info");
        this->total_elements_ = basic_info["total_elements"];
        this->use_reorder_ = basic_info["use_reorder"];
        this->is_trained_ = basic_info["is_trained"];

        JsonType datacell_offsets = metadata->Get("datacell_offsets");
        logger::debug("datacell_offsets: {}", datacell_offsets.dump());
        JsonType datacell_sizes = metadata->Get("datacell_sizes");
        logger::debug("datacell_sizes: {}", datacell_sizes.dump());

        READ_DATACELL_WITH_NAME(reader, "bucket", this->bucket_);
        READ_DATACELL_WITH_NAME(reader, "partition_strategy", this->partition_strategy_);
        READ_DATACELL_WITH_NAME(reader, "label_table", this->label_table_);
        if (use_reorder_) {
            READ_DATACELL_WITH_NAME(reader, "reorder_codes", this->reorder_codes_);
        }
        if (use_attribute_filter_) {
            READ_DATACELL_WITH_NAME(reader, "attr_filter_index", this->attr_filter_index_);
        }
    }

    // post serialize procedure
}

InnerSearchParam
IVF::create_search_param(const std::string& parameters, const FilterPtr& filter) const {
    InnerSearchParam param;
    std::shared_ptr<InnerIdWrapperFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<InnerIdWrapperFilter>(filter, *this->label_table_);
    }
    param.is_inner_id_allowed = ft;
    auto search_param = IVFSearchParameters::FromJson(parameters);
    param.scan_bucket_size = std::min(static_cast<BucketIdType>(search_param.scan_buckets_count),
                                      bucket_->bucket_count_);
    param.factor = search_param.topk_factor;
    param.first_order_scan_ratio = search_param.first_order_scan_ratio;
    return param;
}

DatasetPtr
IVF::reorder(int64_t topk, DistHeapPtr& input, const float* query) const {
    auto [dataset_results, dists, labels] = create_fast_dataset(topk, allocator_);
    auto reorder_heap = Reorder::ReorderByFlatten(input, reorder_codes_, query, allocator_, topk);
    for (int64_t j = topk - 1; j >= 0; --j) {
        dists[j] = reorder_heap->Top().first;
        labels[j] = label_table_->GetLabelById(reorder_heap->Top().second);
        reorder_heap->Pop();
    }
    return std::move(dataset_results);
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
IVF::search(const DatasetPtr& query, const InnerSearchParam& param) const {
    const auto* query_data = query->GetFloat32Vectors();
    Vector<float> normalize_data(dim_, allocator_);
    if (use_residual_ && metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query_data, normalize_data.data(), dim_);
        query_data = normalize_data.data();
    }
    auto candidate_buckets = partition_strategy_->ClassifyDatasForSearch(query_data, 1, param);
    auto computer = bucket_->FactoryComputer(query_data);

    Vector<float> dist(allocator_);
    auto cur_heap_top = std::numeric_limits<float>::max();
    int64_t topk = param.topk;
    if constexpr (mode == RANGE_SEARCH) {
        topk = param.range_search_limit_size;
        if (topk < 0) {
            topk = std::numeric_limits<int64_t>::max();
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

    auto search_result = DistanceHeap::MakeInstanceBySize<true, false>(this->allocator_, topk);
    const auto& ft = param.is_inner_id_allowed;
    Vector<float> centroid(dim_, allocator_);

    for (auto& bucket_id : candidate_buckets) {
        if (bucket_id == -1) {
            break;
        }
        auto bucket_size = bucket_->GetBucketSize(bucket_id);
        const auto* ids = bucket_->GetInnerIds(bucket_id);
        if (bucket_size > dist.size()) {
            dist.resize(bucket_size);
        }
        auto ip_distance = 0.0F;
        if (use_residual_) {
            partition_strategy_->GetCentroid(bucket_id, centroid);
            ip_distance = FP32ComputeIP(query_data, centroid.data(), dim_);
            if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
                ip_distance *= 2;
            }
        }

        bucket_->ScanBucketById(dist.data(), computer, bucket_id);
        FilterPtr attr_ft = nullptr;
        if (param.executor != nullptr) {
            param.executor->Clear();
            attr_ft = param.executor->RunWithBucket(bucket_id);
        }
        for (int j = 0; j < bucket_size; ++j) {
            auto origin_id = ids[j] / buckets_per_data_;
            if ((ft == nullptr or ft->CheckValid(origin_id)) and
                (attr_ft == nullptr or attr_ft->CheckValid(j))) {
                dist[j] -= ip_distance;
                if constexpr (mode == KNN_SEARCH) {
                    if (search_result->Size() < topk or dist[j] < cur_heap_top) {
                        search_result->Push(dist[j], ids[j]);
                    }
                } else if constexpr (mode == RANGE_SEARCH) {
                    if (dist[j] <= param.radius + THRESHOLD_ERROR and dist[j] < cur_heap_top) {
                        search_result->Push(dist[j], ids[j]);
                    }
                }
                if (search_result->Size() > topk) {
                    search_result->Pop();
                }
                if (not search_result->Empty() and search_result->Size() == topk) {
                    cur_heap_top = search_result->Top().first;
                }
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
    auto bias = this->total_elements_;
    this->label_table_->MergeOther(other_index->label_table_, unit.id_map_func);
    other_index->bucket_->Unpack();
    this->bucket_->MergeOther(other_index->bucket_, bias);
    other_index->bucket_->Package();

    if (this->use_reorder_) {
        this->reorder_codes_->MergeOther(other_index->reorder_codes_, bias);
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

    // std::ofstream of1("/tmp/vsag-f1.index", std::ios::binary | std::ios::out);
    // IOStreamWriter os1(of1);
    // cur_model->Serialize(os1);
    // of1.close();

    cur_model.reset();
    auto other_model = other_ivf_index->ExportModel(index->GetCommonParam());
    IOStreamWriter writer2(ss2);
    other_model->Serialize(writer2);

    // std::ofstream of2("/tmp/vsag-f2.index", std::ios::binary | std::ios::out);
    // IOStreamWriter os2(of2);
    // other_model->Serialize(os2);
    // of2.close();

    other_model.reset();

    if (not check_equal_on_string_stream(ss1, ss2)) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            "Merge Failed: IVF model not match, try to merge a different model ivf index");
    }
}

DatasetPtr
IVF::SearchWithRequest(const SearchRequest& request) const {
    auto param = this->create_search_param(request.params_str_, request.filter_);
    param.search_mode = KNN_SEARCH;
    param.topk = request.topk_;
    if (use_reorder_) {
        param.topk = static_cast<int64_t>(param.factor * static_cast<float>(request.topk_));
    }
    auto query = request.query_;
    if (request.enable_attribute_filter_) {
        auto expr = AstParse(request.attribute_filter_str_);
        auto executor = Executor::MakeInstance(this->allocator_, expr, this->attr_filter_index_);
        param.executor = executor;
    }
    auto search_result = this->search<KNN_SEARCH>(query, param);
    if (use_reorder_) {
        return reorder(request.topk_, search_result, query->GetFloat32Vectors());
    }
    auto count = static_cast<const int64_t>(search_result->Size());
    auto [dataset_results, dists, labels] = create_fast_dataset(count, allocator_);
    for (int64_t j = count - 1; j >= 0; --j) {
        dists[j] = search_result->Top().first;
        labels[j] = label_table_->GetLabelById(search_result->Top().second);
        search_result->Pop();
    }
    return std::move(dataset_results);
}

}  // namespace vsag
