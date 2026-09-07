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

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <set>
#include <unordered_map>

#include "algorithm/inner_index_interface.h"
#include "datacell/bucket_datacell_parameter.h"
#include "datacell/flatten_datacell_parameter.h"
#include "datacell/flatten_interface.h"
#include "flat_bucket_searcher.h"
#include "gno_imi_partition.h"
#include "graph_bucket_searcher.h"
#include "impl/reorder/bucket_reorder.h"
#include "impl/reorder/flatten_reorder.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "ivf_nearest_partition.h"
#include "storage/stream_writer.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

BucketDataCellParamPtr
make_precise_bucket_param(const IVFParameterPtr& param) {
    auto precise_bucket_param = std::make_shared<BucketDataCellParameter>();
    precise_bucket_param->io_parameter = param->precise_codes_param->io_parameter;
    precise_bucket_param->quantizer_parameter = param->precise_codes_param->quantizer_parameter;
    precise_bucket_param->buckets_count = param->bucket_param->buckets_count;
    precise_bucket_param->use_residual_ = false;
    return precise_bucket_param;
}

}  // namespace

JsonType
build_default_ivf_param(const JsonType& external_param) {
    const auto base_quantization_type = external_param.Contains(IVF_BASE_QUANTIZATION_TYPE)
                                            ? external_param[IVF_BASE_QUANTIZATION_TYPE].GetString()
                                            : QUANTIZATION_TYPE_VALUE_FP32;
    const auto base_io_type = external_param.Contains(IVF_BASE_IO_TYPE)
                                  ? external_param[IVF_BASE_IO_TYPE].GetString()
                                  : IO_TYPE_VALUE_MEMORY_IO;
    const auto precise_quantization_type =
        external_param.Contains(IVF_PRECISE_QUANTIZATION_TYPE)
            ? external_param[IVF_PRECISE_QUANTIZATION_TYPE].GetString()
            : QUANTIZATION_TYPE_VALUE_FP32;
    const auto precise_io_type = external_param.Contains(IVF_PRECISE_IO_TYPE)
                                     ? external_param[IVF_PRECISE_IO_TYPE].GetString()
                                     : IO_TYPE_VALUE_BLOCK_MEMORY_IO;
    JsonType json;
    json[TYPE_KEY].SetString(INDEX_TYPE_IVF);
    json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_KMEANS);
    json[USE_ATTRIBUTE_FILTER_KEY].SetBool(false);
    json[USE_REORDER_KEY].SetBool(false);
    json[BUILD_THREAD_COUNT_KEY].SetInt(1);

    auto bucket =
        BucketDataCellParameter::CreateDefault(base_quantization_type, base_io_type)->ToJson();
    bucket[BUCKETS_COUNT_KEY].SetInt(10);
    bucket[BUCKET_USE_RESIDUAL_KEY].SetBool(false);
    json[BUCKET_PARAMS_KEY].SetJson(bucket);

    JsonType partition;
    partition[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_NEAREST);
    partition[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_KMEANS);
    partition[IVF_USE_ROUTE_GRAPH_KEY].SetBool(true);
    partition[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI][GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY].SetInt(
        10);
    partition[IVF_PARTITION_STRATEGY_TYPE_GNO_IMI][GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY].SetInt(
        10);
    json[IVF_PARTITION_STRATEGY_PARAMS_KEY].SetJson(partition);
    json[BUCKET_PER_DATA_KEY].SetInt(1);
    json[PRECISE_CODES_LAYOUT_KEY].SetString(PRECISE_CODES_LAYOUT_VALUE_FLAT);

    auto precise =
        FlattenDataCellParameter::CreateDefault(precise_quantization_type, precise_io_type)
            ->ToJson();
    json[PRECISE_CODES_KEY].SetJson(precise);
    json[ATTR_PARAMS_KEY][ATTR_HAS_BUCKETS_KEY].SetBool(true);
    json[GRAPH_BUILD_THRESHOLD_KEY].SetInt(0);
    return json;
}

ParamPtr
IVF::CheckAndMappingExternalParam(const JsonType& external_param,
                                  const IndexCommonParam& common_param) {
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("IVF not support {} datatype", DATATYPE_INT8));
    }

    auto inner_json = build_default_ivf_param(external_param);
    for (const auto& [key, ignored] : external_param.GetInnerJson()->items()) {
        (void)ignored;
        auto value = external_param[key];
        if (key == IVF_BASE_QUANTIZATION_TYPE) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == IVF_BASE_IO_TYPE) {
            inner_json[BUCKET_PARAMS_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == IVF_BASE_FILE_PATH) {
            inner_json[BUCKET_PARAMS_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == IVF_BASE_CACHE_TOTAL_SIZE) {
            inner_json[BUCKET_PARAMS_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(
                value);
        } else if (key == IVF_PRECISE_QUANTIZATION_TYPE) {
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == IVF_PRECISE_IO_TYPE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][TYPE_KEY].SetJson(value);
        } else if (key == IVF_PRECISE_FILE_PATH) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetJson(value);
        } else if (key == IVF_PRECISE_CACHE_TOTAL_SIZE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetJson(
                value);
        } else if (key == IVF_BUCKETS_COUNT) {
            inner_json[BUCKET_PARAMS_KEY][BUCKETS_COUNT_KEY].SetJson(value);
        } else if (key == IVF_TRAIN_TYPE) {
            inner_json[IVF_PARTITION_STRATEGY_PARAMS_KEY][IVF_TRAIN_TYPE_KEY].SetJson(value);
        } else if (key == IVF_PARTITION_STRATEGY_TYPE_KEY) {
            inner_json[IVF_PARTITION_STRATEGY_PARAMS_KEY][IVF_PARTITION_STRATEGY_TYPE_KEY].SetJson(
                value);
        } else if (key == IVF_USE_ROUTE_GRAPH) {
            inner_json[IVF_PARTITION_STRATEGY_PARAMS_KEY][IVF_USE_ROUTE_GRAPH_KEY].SetJson(value);
        } else if (key == GNO_IMI_FIRST_ORDER_BUCKETS_COUNT) {
            inner_json[IVF_PARTITION_STRATEGY_PARAMS_KEY][IVF_PARTITION_STRATEGY_TYPE_GNO_IMI]
                      [GNO_IMI_FIRST_ORDER_BUCKETS_COUNT_KEY]
                          .SetJson(value);
        } else if (key == GNO_IMI_SECOND_ORDER_BUCKETS_COUNT) {
            inner_json[IVF_PARTITION_STRATEGY_PARAMS_KEY][IVF_PARTITION_STRATEGY_TYPE_GNO_IMI]
                      [GNO_IMI_SECOND_ORDER_BUCKETS_COUNT_KEY]
                          .SetJson(value);
        } else if (key == BUCKET_PER_DATA_KEY) {
            inner_json[BUCKET_PER_DATA_KEY].SetJson(value);
        } else if (key == IVF_USE_REORDER) {
            inner_json[USE_REORDER_KEY].SetJson(value);
        } else if (key == IVF_PRECISE_CODES_LAYOUT) {
            inner_json[PRECISE_CODES_LAYOUT_KEY].SetJson(value);
        } else if (key == IVF_USE_RESIDUAL) {
            inner_json[BUCKET_PARAMS_KEY][BUCKET_USE_RESIDUAL_KEY].SetJson(value);
        } else if (key == USE_ATTRIBUTE_FILTER) {
            inner_json[USE_ATTRIBUTE_FILTER_KEY].SetJson(value);
        } else if (key == IVF_BASE_PQ_DIM) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][PRODUCT_QUANTIZATION_DIM_KEY]
                .SetJson(value);
        } else if (key == RABITQ_PCA_DIM) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][PCA_DIM_KEY].SetJson(value);
        } else if (key == RABITQ_BITS_PER_DIM_QUERY) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_BITS_PER_DIM_QUERY_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_BITS_PER_DIM_BASE) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_BITS_PER_DIM_BASE_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_VERSION) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][RABITQ_QUANTIZATION_VERSION_KEY]
                .SetJson(value);
        } else if (key == RABITQ_ERROR_RATE) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY]
                      [RABITQ_QUANTIZATION_ERROR_RATE_KEY]
                          .SetJson(value);
        } else if (key == RABITQ_USE_FHT) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][USE_FHT_KEY].SetJson(value);
        } else if (key == FAST_ENCODE_RABITQ) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_KEY].SetJson(
                value);
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_KEY].SetJson(
                value);
        } else if (key == FAST_ENCODE_RABITQ_ROUNDS) {
            inner_json[BUCKET_PARAMS_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_ROUNDS_KEY]
                .SetJson(value);
            inner_json[PRECISE_CODES_KEY][QUANTIZATION_PARAMS_KEY][FAST_ENCODE_RABITQ_ROUNDS_KEY]
                .SetJson(value);
        } else if (key == IVF_THREAD_COUNT) {
            inner_json[BUILD_THREAD_COUNT_KEY].SetJson(value);
        } else if (key == TRAIN_SAMPLE_COUNT_KEY) {
            inner_json[TRAIN_SAMPLE_COUNT_KEY].SetJson(value);
        } else if (key == GRAPH_BUILD_THRESHOLD_KEY) {
            inner_json[GRAPH_BUILD_THRESHOLD_KEY].SetJson(value);
        } else if (key == IVF_BASE_ENABLE_READ_CACHE) {
            inner_json[BUCKET_PARAMS_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else if (key == IVF_PRECISE_ENABLE_READ_CACHE) {
            inner_json[PRECISE_CODES_KEY][IO_PARAMS_KEY][READ_CACHE_ENABLED_KEY].SetJson(value);
        } else {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("invalid config param: {}", key));
        }
    }

    auto ivf_parameter = std::make_shared<IVFParameter>();
    ivf_parameter->FromJson(inner_json);

    return ivf_parameter;
}

IVF::IVF(const IVFParameterPtr& param, const IndexCommonParam& common_param)
    : InnerIndexInterface(param, common_param),
      buckets_per_data_(param->buckets_per_data),
      location_map_(common_param.allocator_.get()),
      bucket_graphs_(common_param.allocator_.get()),
      common_param_(common_param),
      bucket_searcher_(std::make_shared<FlatBucketSearcher>()) {
    this->bucket_ = BucketInterface::MakeInstance(param->bucket_param, common_param);
    if (this->bucket_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "bucket init error");
    }

    // Initialize thread pool before partition strategy construction
    this->thread_pool_ = common_param.thread_pool_;
    if (param->build_thread_count > 1 and this->thread_pool_ == nullptr) {
        this->thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
        this->thread_pool_->SetPoolSize(param->build_thread_count);
    }

    // Create modified common_param with the initialized thread_pool_
    IndexCommonParam modified_common_param = common_param;
    modified_common_param.thread_pool_ = this->thread_pool_;

    if (param->ivf_partition_strategy_parameter->partition_strategy_type ==
        IVFPartitionStrategyType::IVF) {
        this->partition_strategy_ = std::make_shared<IVFNearestPartition>(
            bucket_->bucket_count_, modified_common_param, param->ivf_partition_strategy_parameter);
    } else if (param->ivf_partition_strategy_parameter->partition_strategy_type ==
               IVFPartitionStrategyType::GNO_IMI) {
        this->partition_strategy_ = std::make_shared<GNOIMIPartition>(
            modified_common_param, param->ivf_partition_strategy_parameter);
    }
    if (this->use_reorder_) {
        if (param->precise_codes_layout == PRECISE_CODES_LAYOUT_VALUE_BUCKET) {
            this->precise_bucket_ = BucketInterface::MakeInstance(make_precise_bucket_param(param),
                                                                  modified_common_param);
            CHECK_ARGUMENT(this->precise_bucket_ != nullptr,
                           "unsupported IO or quantizer for IVF precise bucket");
            this->reorder_ = std::make_shared<BucketReorder>(
                this->precise_bucket_,
                [this](InnerIdType inner_id) { return this->get_location(inner_id); },
                allocator_);
        } else {
            this->reorder_codes_ =
                FlattenInterface::MakeInstance(param->precise_codes_param, modified_common_param);
            reorder_ = std::make_shared<FlattenReorder>(this->reorder_codes_, allocator_);
        }
    }
    if (param->bucket_param->use_residual_) {
        this->bucket_->SetStrategy(partition_strategy_);
    }

    this->graph_param_ = param->graph_param;
    this->graph_build_threshold_ = param->graph_build_threshold;
    if (this->graph_build_threshold_ > 0) {
        this->bucket_searcher_ = std::make_shared<GraphBucketSearcher>(
            this->graph_build_threshold_, this->bucket_graphs_, this->allocator_);
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

int64_t
IVF::GetNumElements() const {
    return this->total_elements_ - this->delete_count_;
}

void
IVF::Merge(const std::vector<MergeUnit>& merge_units) {
    this->bucket_->Unpack();
    if (precise_bucket_ != nullptr) {
        this->precise_bucket_->Unpack();
    }
    for (const auto& unit : merge_units) {
        this->merge_one_unit(unit);
    }
    this->fill_location_map();
    this->bucket_->Package();
    if (precise_bucket_ != nullptr) {
        this->precise_bucket_->Package();
    }
}

std::pair<BucketIdType, InnerIdType>
IVF::get_location(InnerIdType inner_id) const {
    auto loc = this->location_map_[inner_id];
    constexpr uint64_t mask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;
    auto bucket_id = static_cast<BucketIdType>(loc >> LOCATION_SPLIT_BIT);
    auto offset_id = static_cast<InnerIdType>(loc & mask);
    return {bucket_id, offset_id};
}

InnerIndexPtr
IVF::ExportModel(const IndexCommonParam& param) const {
    auto index = std::make_shared<IVF>(this->create_param_ptr_, param);
    IVFPartitionStrategy::Clone(this->partition_strategy_, index->partition_strategy_);
    this->bucket_->ExportModel(index->bucket_);
    if (use_reorder_) {
        if (precise_bucket_ != nullptr) {
            this->precise_bucket_->ExportModel(index->precise_bucket_);
        } else {
            this->reorder_codes_->ExportModel(index->reorder_codes_);
        }
    }
    index->is_trained_ = this->is_trained_;
    return index;
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
        if (precise_bucket_ != nullptr) {
            other_index->precise_bucket_->Unpack();
            this->precise_bucket_->MergeOther(other_index->precise_bucket_, bucket_bias);
            other_index->precise_bucket_->Package();
        } else {
            this->reorder_codes_->MergeOther(other_index->reorder_codes_, this->total_elements_);
        }
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
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("Merge Failed: ivf use_reorder not match, current "
                                        "index is {}, other index is {}",
                                        this->use_reorder_,
                                        other_ivf_index->use_reorder_));
    }
    if ((other_ivf_index->precise_bucket_ == nullptr) != (this->precise_bucket_ == nullptr)) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Merge Failed: IVF precise codes layout does not match");
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

void
IVF::GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const {
    auto [bucket_id, bucket_offset] = this->get_location(inner_id);
    this->attr_filter_index_->GetAttribute(bucket_id, bucket_offset, attr);
}

DatasetPtr
IVF::CalcDistancesById(const float* query,
                       const int64_t* ids,
                       int64_t count,
                       bool calculate_precise_distance) const {
    return this->CalDistanceById(query, ids, count, calculate_precise_distance);
}

DatasetPtr
IVF::CalDistanceById(const float* query,
                     const int64_t* ids,
                     int64_t count,
                     bool calculate_precise_distance,
                     int64_t topk) const {
    CHECK_ARGUMENT(count >= 0, "CalDistanceById count must be non-negative");
    const bool invalid_topk = topk != -1 && topk <= 0;
    CHECK_ARGUMENT(not invalid_topk, "CalDistanceById topk must be -1 or positive");
    if (count > 0) {
        CHECK_ARGUMENT(query != nullptr, "CalDistanceById query must not be null");
        CHECK_ARGUMENT(ids != nullptr, "CalDistanceById ids must not be null");
    }
    const int64_t result_count = (topk == -1) ? count : std::min(topk, count);
    auto result = Dataset::Make();
    result->NumElements(1)->Dim(result_count)->Owner(true, allocator_);
    if (count == 0) {
        return result;
    }
    auto* distances = static_cast<float*>(allocator_->Allocate(sizeof(float) * count));
    result->Distances(distances);
    Vector<InnerIdType> inner_ids(count, 0, allocator_);
    std::vector<bool> validity(count, false);
    {
        std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
        for (int64_t i = 0; i < count; ++i) {
            auto [success, inner_id] = this->label_table_->TryGetIdByLabel(ids[i]);
            if (success) {
                inner_ids[i] = inner_id;
                validity[i] = true;
            }
        }
    }
    if (this->use_reorder_ && calculate_precise_distance && reorder_codes_ != nullptr) {
        auto computer = this->reorder_codes_->FactoryComputer(query);
        this->reorder_codes_->Query(distances, computer, inner_ids.data(), count);
    } else if (this->use_reorder_ && calculate_precise_distance && precise_bucket_ != nullptr) {
        auto computer = this->precise_bucket_->FactoryComputer(query);
        Vector<BucketIdType> bucket_ids(allocator_);
        Vector<InnerIdType> offset_ids(allocator_);
        Vector<int64_t> result_indices(allocator_);
        bucket_ids.reserve(count);
        offset_ids.reserve(count);
        result_indices.reserve(count);
        for (int64_t i = 0; i < count; ++i) {
            if (validity[i]) {
                auto [bucket_id, offset_id] = this->get_location(inner_ids[i]);
                bucket_ids.emplace_back(bucket_id);
                offset_ids.emplace_back(offset_id);
                result_indices.emplace_back(i);
            }
        }
        Vector<float> valid_distances(result_indices.size(), allocator_);
        this->precise_bucket_->Query(valid_distances.data(),
                                     computer,
                                     bucket_ids.data(),
                                     offset_ids.data(),
                                     static_cast<InnerIdType>(result_indices.size()));
        for (uint64_t i = 0; i < result_indices.size(); ++i) {
            distances[result_indices[i]] = valid_distances[i];
        }
    } else {
        auto computer = this->bucket_->FactoryComputer(query);
        for (int64_t i = 0; i < count; ++i) {
            if (validity[i]) {
                auto [bucket_id, offset_id] = this->get_location(inner_ids[i]);
                distances[i] = this->bucket_->QueryOneById(computer, bucket_id, offset_id);
            }
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        if (not validity[i]) {
            distances[i] = -1.0F;
        }
    }
    if (topk == -1) {
        return result;
    }
    return ApplyTopkWithValidity(distances, ids, count, 1, topk, validity, allocator_);
}

float
IVF::CalcDistanceById(const float* query, int64_t id, bool calculate_precise_distance) const {
    std::shared_lock<std::shared_mutex> lock(this->label_lookup_mutex_);
    auto [success, inner_id] = this->label_table_->TryGetIdByLabel(id);
    if (not success) {
        return -1.0F;
    }
    if (this->use_reorder_ && calculate_precise_distance && reorder_codes_ != nullptr) {
        float dist = 0.0F;
        auto computer = this->reorder_codes_->FactoryComputer(query);
        this->reorder_codes_->Query(&dist, computer, &inner_id, 1);
        return dist;
    }
    auto codes = this->use_reorder_ && calculate_precise_distance ? precise_bucket_ : bucket_;
    auto computer = codes->FactoryComputer(query);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    return codes->QueryOneById(computer, bucket_id, offset_id);
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
        memory += precise_bucket_ != nullptr ? precise_bucket_->GetMemoryUsage()
                                             : reorder_codes_->GetMemoryUsage();
    }
    if (this->extra_info_size_ > 0 and this->extra_infos_ != nullptr) {
        memory += this->extra_infos_->GetMemoryUsage();
    }
    memory += this->label_table_->GetMemoryUsage();
    memory += location_map_.size() * sizeof(uint64_t);
    memory += partition_strategy_->GetMemoryUsage();
    for (auto& g : bucket_graphs_) {
        if (g != nullptr) {
            memory += g->GetMemoryUsage();
        }
    }
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
