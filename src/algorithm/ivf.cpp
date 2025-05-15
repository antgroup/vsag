
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

#include "impl/basic_searcher.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "ivf_partition/ivf_nearest_partition.h"
#include "utils/standard_heap.h"
#include "utils/util_functions.h"

namespace vsag {

static constexpr const char* IVF_PARAMS_TEMPLATE =
    R"(
    {
        "type": "{INDEX_TYPE_IVF}",
        "{IVF_TRAIN_TYPE_KEY}": "{IVF_TRAIN_TYPE_KMEANS}",
        "{BUCKET_PARAMS_KEY}": {
            "{IO_PARAMS_KEY}": {
                "{IO_TYPE_KEY}": "{IO_TYPE_VALUE_BLOCK_MEMORY_IO}"
            },
            "{QUANTIZATION_PARAMS_KEY}": {
                "{QUANTIZATION_TYPE_KEY}": "{QUANTIZATION_TYPE_VALUE_FP32}",
                "{SQ4_UNIFORM_QUANTIZATION_TRUNC_RATE}": 0.05,
                "{PCA_DIM}": 0,
                "{RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY}": 32,
                "{PRODUCT_QUANTIZATION_DIM}": 0
            },
            "{BUCKETS_COUNT_KEY}": 10
        },
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
            {IVF_TRAIN_TYPE_KEY},
        },
        {
            IVF_USE_REORDER,
            {IVF_USE_REORDER_KEY},
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
    : InnerIndexInterface(param, common_param) {
    this->bucket_ = BucketInterface::MakeInstance(param->bucket_param, common_param);
    if (this->bucket_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bucket init error");
    }
    this->partition_strategy_ = std::make_shared<IVFNearestPartition>(
        bucket_->bucket_count_, common_param, param->partition_train_type);
    this->use_reorder_ = param->use_reorder;
    if (this->use_reorder_) {
        this->reorder_codes_ = FlattenInterface::MakeInstance(param->flatten_param, common_param);
    }
}

void
IVF::InitFeatures() {
    // Common Init
    // Build & Add
    this->index_feature_list_->SetFeatures({
        IndexFeature::SUPPORT_BUILD,
        IndexFeature::SUPPORT_ADD_AFTER_BUILD,
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
        // TODO(LHT): merge on ivfpqfs
        this->index_feature_list_->SetFeature(IndexFeature::SUPPORT_MERGE_INDEX, false);
    }
}

std::vector<int64_t>
IVF::Build(const DatasetPtr& base) {
    this->Train(base);
    // TODO(LHT): duplicate
    auto result = this->Add(base);
    if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_PQFS) {
        this->bucket_->Package();
    }
    return result;
}

void
IVF::Train(const DatasetPtr& data) {
    if (this->is_trained_) {
        return;
    }
    partition_strategy_->Train(data);
    this->bucket_->Train(data->GetFloat32Vectors(), data->GetNumElements());
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
    auto num_element = base->GetNumElements();
    const auto* ids = base->GetIds();
    const auto* vectors = base->GetFloat32Vectors();
    auto buckets = partition_strategy_->ClassifyDatas(vectors, num_element, 1);
    for (int64_t i = 0; i < num_element; ++i) {
        bucket_->InsertVector(vectors + i * dim_, buckets[i], i + total_elements_);
        this->label_table_->Insert(i + total_elements_, ids[i]);
    }
    if (use_reorder_) {
        this->reorder_codes_->BatchInsertVector(base->GetFloat32Vectors(), base->GetNumElements());
    }
    this->total_elements_ += num_element;
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
    auto [dataset_results, dists, labels] = CreateFastDataset(count, allocator_);
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
    auto [dataset_results, dists, labels] = CreateFastDataset(count, allocator_);
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
    for (const auto& unit : merge_units) {
        this->merge_one_unit(unit);
    }
}

void
IVF::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->total_elements_);
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->is_trained_);

    this->bucket_->Serialize(writer);
    this->partition_strategy_->Serialize(writer);
    this->label_table_->Serialize(writer);
    if (use_reorder_) {
        this->reorder_codes_->Serialize(writer);
    }
}

void
IVF::Deserialize(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->total_elements_);
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->is_trained_);

    this->bucket_->Deserialize(reader);
    this->partition_strategy_->Deserialize(reader);
    this->label_table_->Deserialize(reader);
    if (use_reorder_) {
        this->reorder_codes_->Deserialize(reader);
    }
}
InnerSearchParam
IVF::create_search_param(const std::string& parameters, const FilterPtr& filter) const {
    InnerSearchParam param;
    std::shared_ptr<CommonInnerIdFilter> ft = nullptr;
    if (filter != nullptr) {
        ft = std::make_shared<CommonInnerIdFilter>(filter, *this->label_table_);
    }
    param.is_inner_id_allowed = ft;
    auto search_param = IVFSearchParameters::FromJson(parameters);
    param.scan_bucket_size = std::min(static_cast<BucketIdType>(search_param.scan_buckets_count),
                                      bucket_->bucket_count_);
    param.factor = search_param.topk_factor;
    return param;
}

DatasetPtr
IVF::reorder(int64_t topk, DistHeapPtr& input, const float* query) const {
    auto [dataset_results, dists, labels] = CreateFastDataset(topk, allocator_);
    StandardHeap<true, true> reorder_heap(allocator_, topk);
    auto computer = this->reorder_codes_->FactoryComputer(query);
    while (not input->Empty()) {
        auto [dist, id] = input->Top();
        this->reorder_codes_->Query(&dist, computer, &id, 1);
        reorder_heap.Push(dist, id);
        input->Pop();
    }
    for (int64_t j = topk - 1; j >= 0; --j) {
        dists[j] = reorder_heap.Top().first;
        labels[j] = label_table_->GetLabelById(reorder_heap.Top().second);
        reorder_heap.Pop();
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
    auto search_result = std::make_shared<StandardHeap<true, false>>(allocator_, -1);
    auto candidate_buckets =
        partition_strategy_->ClassifyDatas(query->GetFloat32Vectors(), 1, param.scan_bucket_size);
    auto computer = bucket_->FactoryComputer(query->GetFloat32Vectors());
    Vector<float> dist(allocator_);
    auto cur_heap_top = std::numeric_limits<float>::max();
    int64_t topk = param.topk;
    if constexpr (mode == RANGE_SEARCH) {
        topk = param.range_search_limit_size;
        if (topk < 0) {
            topk = std::numeric_limits<int64_t>::max();
        }
    }
    const auto& ft = param.is_inner_id_allowed;
    for (auto& bucket_id : candidate_buckets) {
        auto bucket_size = bucket_->GetBucketSize(bucket_id);
        const auto* ids = bucket_->GetInnerIds(bucket_id);
        if (bucket_size > dist.size()) {
            dist.resize(bucket_size);
        }
        bucket_->ScanBucketById(dist.data(), computer, bucket_id);
        for (int j = 0; j < bucket_size; ++j) {
            if (ft == nullptr or ft->CheckValid(ids[j])) {
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
    return search_result;
}

void
IVF::merge_one_unit(const MergeUnit& unit) {
    check_merge_illegal(unit);
    const auto other_index = std::dynamic_pointer_cast<IVF>(
        std::dynamic_pointer_cast<IndexImpl<IVF>>(unit.index)->GetInnerIndex());
    auto bias = this->total_elements_;
    this->label_table_->MergeOther(other_index->label_table_, bias);
    this->bucket_->MergeOther(other_index->bucket_, bias);

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

}  // namespace vsag
