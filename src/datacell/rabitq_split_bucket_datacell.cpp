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

#include "rabitq_split_bucket_datacell.h"

#include <algorithm>
#include <cstring>

#include "inner_string_params.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"

namespace vsag {

RaBitQSplitBucketDataCell::RaBitQSplitBucketDataCell(const BucketDataCellParamPtr& param,
                                                     const IndexCommonParam& common_param)
    : common_param_(common_param),
      allocator_(common_param.allocator_.get()),
      dim_(common_param.dim_),
      metric_(common_param.metric_),
      inner_ids_(param->buckets_count,
                 Vector<InnerIdType>(common_param.allocator_.get()),
                 common_param.allocator_.get()),
      residual_bias_(param->buckets_count,
                     Vector<float>(common_param.allocator_.get()),
                     common_param.allocator_.get()),
      fastscan_blocks_(param->buckets_count,
                       Vector<uint8_t>(common_param.allocator_.get()),
                       common_param.allocator_.get()),
      bucket_mutexes_(param->buckets_count, common_param.allocator_.get()),
      locations_(common_param.allocator_.get()) {
    auto flatten_param = std::make_shared<FlattenDataCellParameter>();
    flatten_param->name = RABITQ_SPLIT_DATA_CELL;
    flatten_param->quantizer_parameter = param->quantizer_parameter;
    flatten_param->io_parameter = param->io_parameter;
    flatten_param->supplement_io_parameter = param->supplement_io_parameter;
    this->codes_ = FlattenInterface::MakeInstance(flatten_param, common_param);
    CHECK_ARGUMENT(this->codes_ != nullptr and this->codes_->SupportSplitCodeStorage(),
                   "failed to create RaBitQ split bucket codes");

    this->bucket_count_ = static_cast<BucketIdType>(param->buckets_count);
    this->code_size_ = this->codes_->code_size_;
    this->use_residual_ = param->use_residual_;
    this->backend_ = DistanceEvaluationBackend::RABITQ;
    this->fastscan_block_size_ = this->codes_->GetFastScan32BlockSize();
}

void
RaBitQSplitBucketDataCell::check_valid_bucket_id(BucketIdType bucket_id) const {
    if (bucket_id < 0 or bucket_id >= this->bucket_count_) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "visited invalid bucket id");
    }
}

void
RaBitQSplitBucketDataCell::check_valid_offset(BucketIdType bucket_id, InnerIdType offset_id) const {
    this->check_valid_bucket_id(bucket_id);
    if (offset_id >= this->inner_ids_[bucket_id].size() or
        this->inner_ids_[bucket_id][offset_id] == EMPTY_INNER_ID) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid offset id for bucket");
    }
}

const ComputerInterfacePtr&
RaBitQSplitBucketDataCell::get_inner_computer(const ComputerInterfacePtr& computer) {
    return RaBitQSplitBucketDataCell::get_bucket_computer(computer).inner_;
}

const RaBitQSplitBucketDataCell::SplitBucketComputer&
RaBitQSplitBucketDataCell::get_bucket_computer(const ComputerInterfacePtr& computer) {
    auto* bucket_computer = dynamic_cast<SplitBucketComputer*>(computer.get());
    CHECK_ARGUMENT(bucket_computer != nullptr, "invalid RaBitQ split bucket computer");
    return *bucket_computer;
}

const float*
RaBitQSplitBucketDataCell::prepare_vector(const void* vector,
                                          BucketIdType bucket_id,
                                          Vector<float>& prepared,
                                          float& residual_bias) const {
    const auto* input = static_cast<const float*>(vector);
    residual_bias = 0.0F;
    if (not this->use_residual_) {
        return input;
    }

    Vector<float> centroid(this->dim_, this->allocator_);
    this->strategy_->GetCentroid(bucket_id, centroid);
    prepared.resize(this->dim_);
    if (this->metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(input, prepared.data(), this->dim_);
        for (uint64_t i = 0; i < this->dim_; ++i) {
            prepared[i] -= centroid[i];
        }
    } else {
        FP32Sub(input, centroid.data(), prepared.data(), this->dim_);
    }

    if (this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
        ByteBuffer full_code(this->code_size_, this->allocator_);
        Vector<float> decoded(this->dim_, this->allocator_);
        this->codes_->Encode(prepared.data(), full_code.data);
        this->codes_->Decode(full_code.data, decoded.data());
        residual_bias = -2.0F * FP32ComputeIP(centroid.data(), decoded.data(), this->dim_) -
                        FP32ComputeIP(centroid.data(), centroid.data(), this->dim_);
    }
    return prepared.data();
}

float
RaBitQSplitBucketDataCell::residual_adjustment(const SplitBucketComputer& computer,
                                               BucketIdType bucket_id,
                                               InnerIdType offset_id) const {
    if (not this->use_residual_) {
        return 0.0F;
    }
    Vector<float> centroid(this->dim_, this->allocator_);
    this->strategy_->GetCentroid(bucket_id, centroid);
    float adjustment = FP32ComputeIP(computer.raw_query_.data(), centroid.data(), this->dim_);
    if (this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
        adjustment *= 2.0F;
        adjustment += this->residual_bias_[bucket_id][offset_id];
    }
    return adjustment;
}

void
RaBitQSplitBucketDataCell::ScanBucketById(float* result_dists,
                                          const ComputerInterfacePtr& computer,
                                          const BucketIdType& bucket_id,
                                          QueryContext* ctx) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    const auto bucket_size = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
    const auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    const uint64_t block_count =
        (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    if (bucket_computer.fastscan_ != nullptr and this->fastscan_block_size_ > 0 and
        this->fastscan_blocks_[bucket_id].size() == block_count * this->fastscan_block_size_) {
        Vector<InnerIdType> fallback_ids(this->allocator_);
        Vector<InnerIdType> fallback_positions(this->allocator_);
        fallback_ids.reserve(bucket_size);
        fallback_positions.reserve(bucket_size);
        bool computed[FASTSCAN_BATCH_SIZE] = {};
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
            const InnerIdType valid_size =
                std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
            this->codes_->QueryFastScan32(
                result_dists + begin,
                computed,
                bucket_computer.inner_,
                bucket_computer.fastscan_,
                this->fastscan_blocks_[bucket_id].data() + block_index * this->fastscan_block_size_,
                valid_size,
                ctx);
            for (InnerIdType i = 0; i < valid_size; ++i) {
                const InnerIdType offset = begin + i;
                const auto inner_id = this->inner_ids_[bucket_id][offset];
                if (inner_id == EMPTY_INNER_ID) {
                    result_dists[offset] = std::numeric_limits<float>::max();
                } else if (computed[i]) {
                    result_dists[offset] -=
                        this->residual_adjustment(bucket_computer, bucket_id, offset);
                } else {
                    fallback_ids.push_back(inner_id);
                    fallback_positions.push_back(offset);
                }
            }
        }

        if (not fallback_ids.empty()) {
            Vector<float> fallback_dists(fallback_ids.size(), this->allocator_);
            this->codes_->QueryWithDistanceLowerBound(fallback_dists.data(),
                                                      nullptr,
                                                      bucket_computer.inner_,
                                                      fallback_ids.data(),
                                                      static_cast<InnerIdType>(fallback_ids.size()),
                                                      ctx);
            for (uint64_t i = 0; i < fallback_ids.size(); ++i) {
                const auto offset = fallback_positions[i];
                result_dists[offset] = fallback_dists[i] - this->residual_adjustment(
                                                               bucket_computer, bucket_id, offset);
            }
        }
        return;
    }

    Vector<InnerIdType> ids(this->allocator_);
    Vector<InnerIdType> positions(this->allocator_);
    ids.reserve(bucket_size);
    positions.reserve(bucket_size);
    for (InnerIdType offset = 0; offset < bucket_size; ++offset) {
        const auto inner_id = this->inner_ids_[bucket_id][offset];
        if (inner_id == EMPTY_INNER_ID) {
            result_dists[offset] = std::numeric_limits<float>::max();
            continue;
        }
        ids.push_back(inner_id);
        positions.push_back(offset);
    }
    if (ids.empty()) {
        return;
    }

    Vector<float> compact_dists(ids.size(), this->allocator_);
    this->codes_->QueryWithDistanceLowerBound(
        compact_dists.data(),
        nullptr,
        RaBitQSplitBucketDataCell::get_inner_computer(computer),
        ids.data(),
        static_cast<InnerIdType>(ids.size()),
        ctx);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        const auto offset = positions[i];
        result_dists[offset] =
            compact_dists[i] - this->residual_adjustment(bucket_computer, bucket_id, offset);
    }
}

void
RaBitQSplitBucketDataCell::ScanBucketWithDistanceLowerBound(float* result_dists,
                                                            float* lower_bounds,
                                                            float* filter_inner_products,
                                                            const ComputerInterfacePtr& computer,
                                                            const BucketIdType& bucket_id,
                                                            QueryContext* ctx) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    const auto bucket_size = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
    const auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    const uint64_t block_count =
        (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    if (bucket_computer.fastscan_ != nullptr and this->fastscan_block_size_ > 0 and
        this->fastscan_blocks_[bucket_id].size() == block_count * this->fastscan_block_size_) {
        Vector<InnerIdType> fallback_ids(this->allocator_);
        Vector<InnerIdType> fallback_positions(this->allocator_);
        fallback_ids.reserve(bucket_size);
        fallback_positions.reserve(bucket_size);
        bool computed[FASTSCAN_BATCH_SIZE] = {};
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
            const InnerIdType valid_size =
                std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
            this->codes_->QueryFastScan32WithDistanceLowerBoundAndFilterInnerProduct(
                result_dists + begin,
                lower_bounds + begin,
                filter_inner_products + begin,
                computed,
                bucket_computer.inner_,
                bucket_computer.fastscan_,
                this->fastscan_blocks_[bucket_id].data() + block_index * this->fastscan_block_size_,
                valid_size,
                ctx);
            for (InnerIdType i = 0; i < valid_size; ++i) {
                const InnerIdType offset = begin + i;
                const auto inner_id = this->inner_ids_[bucket_id][offset];
                if (inner_id == EMPTY_INNER_ID) {
                    result_dists[offset] = std::numeric_limits<float>::max();
                    lower_bounds[offset] = std::numeric_limits<float>::max();
                    filter_inner_products[offset] = std::numeric_limits<float>::quiet_NaN();
                } else if (computed[i]) {
                    const float adjustment =
                        this->residual_adjustment(bucket_computer, bucket_id, offset);
                    result_dists[offset] -= adjustment;
                    lower_bounds[offset] -= adjustment;
                } else {
                    fallback_ids.push_back(inner_id);
                    fallback_positions.push_back(offset);
                }
            }
        }

        if (not fallback_ids.empty()) {
            Vector<float> fallback_dists(fallback_ids.size(), this->allocator_);
            Vector<float> fallback_lower_bounds(fallback_ids.size(), this->allocator_);
            Vector<float> fallback_filter_inner_products(fallback_ids.size(), this->allocator_);
            this->codes_->QueryWithDistanceLowerBoundAndFilterInnerProduct(
                fallback_dists.data(),
                fallback_lower_bounds.data(),
                fallback_filter_inner_products.data(),
                bucket_computer.inner_,
                fallback_ids.data(),
                static_cast<InnerIdType>(fallback_ids.size()),
                ctx);
            for (uint64_t i = 0; i < fallback_ids.size(); ++i) {
                const auto offset = fallback_positions[i];
                const float adjustment =
                    this->residual_adjustment(bucket_computer, bucket_id, offset);
                result_dists[offset] = fallback_dists[i] - adjustment;
                lower_bounds[offset] = fallback_lower_bounds[i] - adjustment;
                filter_inner_products[offset] = fallback_filter_inner_products[i];
            }
        }
        return;
    }

    Vector<InnerIdType> ids(this->allocator_);
    Vector<InnerIdType> positions(this->allocator_);
    ids.reserve(bucket_size);
    positions.reserve(bucket_size);
    for (InnerIdType offset = 0; offset < bucket_size; ++offset) {
        const auto inner_id = this->inner_ids_[bucket_id][offset];
        if (inner_id == EMPTY_INNER_ID) {
            result_dists[offset] = std::numeric_limits<float>::max();
            lower_bounds[offset] = std::numeric_limits<float>::max();
            filter_inner_products[offset] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        ids.push_back(inner_id);
        positions.push_back(offset);
    }
    if (ids.empty()) {
        return;
    }

    Vector<float> compact_dists(ids.size(), this->allocator_);
    Vector<float> compact_lower_bounds(ids.size(), this->allocator_);
    Vector<float> compact_filter_inner_products(ids.size(), this->allocator_);
    this->codes_->QueryWithDistanceLowerBoundAndFilterInnerProduct(
        compact_dists.data(),
        compact_lower_bounds.data(),
        compact_filter_inner_products.data(),
        bucket_computer.inner_,
        ids.data(),
        static_cast<InnerIdType>(ids.size()),
        ctx);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        const auto offset = positions[i];
        const float adjustment = this->residual_adjustment(bucket_computer, bucket_id, offset);
        result_dists[offset] = compact_dists[i] - adjustment;
        lower_bounds[offset] = compact_lower_bounds[i] - adjustment;
        filter_inner_products[offset] = compact_filter_inner_products[i];
    }
}

float
RaBitQSplitBucketDataCell::QueryOneById(const ComputerInterfacePtr& computer,
                                        const BucketIdType& bucket_id,
                                        const InnerIdType& offset_id) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_offset(bucket_id, offset_id);
    const auto inner_id = this->inner_ids_[bucket_id][offset_id];
    float distance = 0.0F;
    this->codes_->QueryWithDistanceLowerBound(
        &distance,
        nullptr,
        RaBitQSplitBucketDataCell::get_inner_computer(computer),
        &inner_id,
        1,
        nullptr);
    return distance -
           this->residual_adjustment(
               RaBitQSplitBucketDataCell::get_bucket_computer(computer), bucket_id, offset_id);
}

void
RaBitQSplitBucketDataCell::QueryWithFilterInnerProductByInnerId(
    float* result_dists,
    const float* filter_inner_products,
    const ComputerInterfacePtr& computer,
    const InnerIdType* inner_ids,
    InnerIdType id_count,
    QueryContext* ctx) {
    Vector<float> adjustments(id_count, 0.0F, this->allocator_);
    const auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    for (InnerIdType i = 0; i < id_count; ++i) {
        const auto [bucket_id, offset_id] = this->get_location(inner_ids[i]);
        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        adjustments[i] = this->residual_adjustment(bucket_computer, bucket_id, offset_id);
    }
    this->codes_->QueryWithFilterInnerProduct(
        result_dists, filter_inner_products, bucket_computer.inner_, inner_ids, id_count, ctx);
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[i] -= adjustments[i];
    }
}

void
RaBitQSplitBucketDataCell::QueryWithDistanceHintByInnerId(float* result_dists,
                                                          const float* hint_dists,
                                                          const ComputerInterfacePtr& computer,
                                                          const InnerIdType* inner_ids,
                                                          InnerIdType id_count,
                                                          QueryContext* ctx) {
    Vector<float> raw_hints(id_count, this->allocator_);
    Vector<float> adjustments(id_count, 0.0F, this->allocator_);
    const auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    for (InnerIdType i = 0; i < id_count; ++i) {
        const auto [bucket_id, offset_id] = this->get_location(inner_ids[i]);
        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        adjustments[i] = this->residual_adjustment(bucket_computer, bucket_id, offset_id);
        raw_hints[i] = hint_dists == nullptr ? std::numeric_limits<float>::max()
                                             : hint_dists[i] + adjustments[i];
    }
    this->codes_->QueryWithDistanceHint(result_dists,
                                        raw_hints.data(),
                                        RaBitQSplitBucketDataCell::get_inner_computer(computer),
                                        inner_ids,
                                        id_count,
                                        ctx);
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[i] -= adjustments[i];
    }
}

float
RaBitQSplitBucketDataCell::ComputePairVectors(BucketIdType bucket_id,
                                              InnerIdType id1,
                                              InnerIdType id2) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_offset(bucket_id, id1);
    this->check_valid_offset(bucket_id, id2);
    return this->codes_->ComputePairVectors(this->inner_ids_[bucket_id][id1],
                                            this->inner_ids_[bucket_id][id2]);
}

ComputerInterfacePtr
RaBitQSplitBucketDataCell::FactoryComputer(const void* query) {
    const auto* input = static_cast<const float*>(query);
    auto inner = this->codes_->FactoryComputer(input);
    auto fastscan = this->codes_->FactoryFastScan32Computer(inner);
    auto computer = std::make_shared<SplitBucketComputer>(
        std::move(inner), std::move(fastscan), this->dim_, this->allocator_);
    if (this->metric_ == MetricType::METRIC_TYPE_COSINE and this->use_residual_) {
        Normalize(input, computer->raw_query_.data(), this->dim_);
    } else {
        std::memcpy(computer->raw_query_.data(), input, sizeof(float) * this->dim_);
    }
    return computer;
}

void
RaBitQSplitBucketDataCell::Train(const void* data, uint64_t count) {
    if (not this->use_residual_) {
        this->codes_->Train(data, count);
        return;
    }

    const auto* input = static_cast<const float*>(data);
    Vector<float> train_data(count * this->dim_, this->allocator_);
    Vector<float> normalized(this->dim_, this->allocator_);
    Vector<float> centroid(this->dim_, this->allocator_);
    const float* classify_data = input;
    Vector<float> normalized_data(this->allocator_);
    if (this->metric_ == MetricType::METRIC_TYPE_COSINE) {
        normalized_data.resize(count * this->dim_);
        for (uint64_t i = 0; i < count; ++i) {
            Normalize(input + i * this->dim_, normalized_data.data() + i * this->dim_, this->dim_);
        }
        classify_data = normalized_data.data();
    }
    CHECK_ARGUMENT(count <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                   "RaBitQ split training count is too large");
    auto buckets =
        this->strategy_->ClassifyDatas(classify_data, static_cast<int64_t>(count), 1, nullptr);
    for (uint64_t i = 0; i < count; ++i) {
        this->strategy_->GetCentroid(buckets[i], centroid);
        FP32Sub(classify_data + i * this->dim_,
                centroid.data(),
                train_data.data() + i * this->dim_,
                this->dim_);
    }
    this->codes_->Train(train_data.data(), count);
}

InnerIdType
RaBitQSplitBucketDataCell::InsertVector(const void* vector,
                                        BucketIdType bucket_id,
                                        InnerIdType inner_id) {
    this->check_valid_bucket_id(bucket_id);
    CHECK_ARGUMENT(inner_id != EMPTY_INNER_ID, "invalid inner id for bucket");
    Vector<float> prepared(this->allocator_);
    float bias = 0.0F;
    const auto* input = this->prepare_vector(vector, bucket_id, prepared, bias);
    std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
    const auto offset_id = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
    {
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        this->codes_->InsertVector(input, inner_id);
    }
    this->fastscan_blocks_[bucket_id].clear();
    this->inner_ids_[bucket_id].push_back(inner_id);
    if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->residual_bias_[bucket_id].push_back(bias);
    }
    {
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id) {
            this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
        }
        this->locations_[inner_id] = RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
    }
    return offset_id;
}

void
RaBitQSplitBucketDataCell::BatchInsertVector(const void* vectors,
                                             const BucketIdType* bucket_ids,
                                             const InnerIdType* inner_ids,
                                             InnerIdType count,
                                             InnerIdType* out_offsets) {
    if (count == 0) {
        return;
    }
    if (vectors == nullptr or bucket_ids == nullptr or inner_ids == nullptr or
        out_offsets == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "batch bucket insert requires non-null inputs and output");
    }
    for (InnerIdType i = 0; i < count; ++i) {
        this->check_valid_bucket_id(bucket_ids[i]);
        CHECK_ARGUMENT(inner_ids[i] != EMPTY_INNER_ID, "invalid inner id for bucket");
    }

    const auto* float_vectors = static_cast<const float*>(vectors);
    for (InnerIdType i = 0; i < count; ++i) {
        out_offsets[i] = this->InsertVector(
            float_vectors + static_cast<uint64_t>(i) * this->dim_, bucket_ids[i], inner_ids[i]);
    }
}

void
RaBitQSplitBucketDataCell::InsertVectorWithOffset(const void* vector,
                                                  BucketIdType bucket_id,
                                                  InnerIdType inner_id,
                                                  InnerIdType offset_id) {
    this->check_valid_bucket_id(bucket_id);
    CHECK_ARGUMENT(inner_id != EMPTY_INNER_ID, "invalid inner id for bucket");
    CHECK_ARGUMENT(offset_id != EMPTY_INNER_ID, "invalid offset id for bucket");
    Vector<float> prepared(this->allocator_);
    float bias = 0.0F;
    const auto* input = this->prepare_vector(vector, bucket_id, prepared, bias);
    std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
    {
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        this->codes_->InsertVector(input, inner_id);
    }
    this->fastscan_blocks_[bucket_id].clear();
    if (this->inner_ids_[bucket_id].size() <= offset_id) {
        this->inner_ids_[bucket_id].resize(static_cast<uint64_t>(offset_id) + 1, EMPTY_INNER_ID);
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id].resize(static_cast<uint64_t>(offset_id) + 1, 0.0F);
        }
    }
    this->inner_ids_[bucket_id][offset_id] = inner_id;
    if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
        this->residual_bias_[bucket_id][offset_id] = bias;
    }
    {
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id) {
            this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
        }
        this->locations_[inner_id] = RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
    }
}

InnerIdType*
RaBitQSplitBucketDataCell::GetInnerIds(BucketIdType bucket_id) {
    this->check_valid_bucket_id(bucket_id);
    return this->inner_ids_[bucket_id].data();
}

void
RaBitQSplitBucketDataCell::Prefetch(BucketIdType bucket_id, InnerIdType offset_id) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_offset(bucket_id, offset_id);
    this->codes_->Prefetch(this->inner_ids_[bucket_id][offset_id]);
}

void
RaBitQSplitBucketDataCell::GetCodesById(BucketIdType bucket_id,
                                        InnerIdType offset_id,
                                        uint8_t* data) const {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_offset(bucket_id, offset_id);
    CHECK_ARGUMENT(this->codes_->GetCodesById(this->inner_ids_[bucket_id][offset_id], data),
                   "failed to read RaBitQ split bucket code");
}

std::string
RaBitQSplitBucketDataCell::GetQuantizerName() {
    return this->codes_->GetQuantizerName();
}

MetricType
RaBitQSplitBucketDataCell::GetMetricType() {
    return this->metric_;
}

InnerIdType
RaBitQSplitBucketDataCell::GetBucketSize(BucketIdType bucket_id) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    return static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
}

void
RaBitQSplitBucketDataCell::ExportModel(const BucketInterfacePtr& other) const {
    auto target = std::dynamic_pointer_cast<RaBitQSplitBucketDataCell>(other);
    CHECK_ARGUMENT(target != nullptr, "export model's RaBitQ split bucket datacell failed");
    this->codes_->ExportModel(target->codes_);
}

void
RaBitQSplitBucketDataCell::MergeOther(const BucketInterfacePtr& other, InnerIdType bias) {
    auto source = std::dynamic_pointer_cast<RaBitQSplitBucketDataCell>(other);
    CHECK_ARGUMENT(source != nullptr, "merge RaBitQ split bucket datacell failed");
    this->codes_->MergeOther(source->codes_, bias);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        std::scoped_lock lock(this->bucket_mutexes_[bucket_id], source->bucket_mutexes_[bucket_id]);
        const auto old_size = this->inner_ids_[bucket_id].size();
        this->inner_ids_[bucket_id].resize(old_size + source->inner_ids_[bucket_id].size(),
                                           EMPTY_INNER_ID);
        for (uint64_t i = 0; i < source->inner_ids_[bucket_id].size(); ++i) {
            const auto source_id = source->inner_ids_[bucket_id][i];
            if (source_id != EMPTY_INNER_ID) {
                this->inner_ids_[bucket_id][old_size + i] = source_id + bias;
            }
        }
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id].insert(this->residual_bias_[bucket_id].end(),
                                                   source->residual_bias_[bucket_id].begin(),
                                                   source->residual_bias_[bucket_id].end());
        }
        this->fastscan_blocks_[bucket_id].clear();
    }
    this->rebuild_locations();
}

void
RaBitQSplitBucketDataCell::Package() {
    this->package_fastscan();
}

void
RaBitQSplitBucketDataCell::Unpack() {
    for (auto& blocks : this->fastscan_blocks_) {
        blocks.clear();
    }
}

void
RaBitQSplitBucketDataCell::package_fastscan() {
    if (not this->codes_->SupportFastScan32()) {
        this->fastscan_block_size_ = 0;
        return;
    }

    this->fastscan_block_size_ = this->codes_->GetFastScan32BlockSize();
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        const auto bucket_size = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
        const uint64_t block_count =
            (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
        auto& blocks = this->fastscan_blocks_[bucket_id];
        blocks.resize(block_count * this->fastscan_block_size_);
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
            const InnerIdType valid_size =
                std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
            this->codes_->PackageFastScan32(
                this->inner_ids_[bucket_id].data() + begin,
                valid_size,
                blocks.data() + block_index * this->fastscan_block_size_);
        }
    }
}

void
RaBitQSplitBucketDataCell::Serialize(StreamWriter& writer) {
    BucketInterface::Serialize(writer);
    this->codes_->Serialize(writer);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        StreamWriter::WriteVector(writer, this->inner_ids_[bucket_id]);
        if (this->use_residual_) {
            StreamWriter::WriteVector(writer, this->residual_bias_[bucket_id]);
        }
    }
}

void
RaBitQSplitBucketDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    const auto expected_bucket_count = this->bucket_count_;
    BucketInterface::Deserialize(reader);
    CHECK_ARGUMENT(this->bucket_count_ == expected_bucket_count,
                   "RaBitQ split bucket count does not match serialized data");
    this->codes_->Deserialize(reader);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        StreamReader::ReadVector(reader, this->inner_ids_[bucket_id]);
        if (this->use_residual_) {
            StreamReader::ReadVector(reader, this->residual_bias_[bucket_id]);
        }
    }
    this->backend_ = DistanceEvaluationBackend::RABITQ;
    this->rebuild_locations();
    this->package_fastscan();
}

uint64_t
RaBitQSplitBucketDataCell::GetMemoryUsage() const {
    uint64_t memory = sizeof(RaBitQSplitBucketDataCell) + this->codes_->GetMemoryUsage();
    memory += this->locations_.capacity() * sizeof(uint64_t);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        memory += this->inner_ids_[bucket_id].capacity() * sizeof(InnerIdType);
        memory += this->residual_bias_[bucket_id].capacity() * sizeof(float);
        memory += this->fastscan_blocks_[bucket_id].capacity() * sizeof(uint8_t);
        memory += sizeof(std::shared_mutex);
    }
    return memory;
}

uint64_t
RaBitQSplitBucketDataCell::pack_location(BucketIdType bucket_id, InnerIdType offset_id) {
    return (static_cast<uint64_t>(bucket_id) << LOCATION_SPLIT_BIT) |
           static_cast<uint64_t>(offset_id);
}

std::pair<BucketIdType, InnerIdType>
RaBitQSplitBucketDataCell::get_location(InnerIdType inner_id) const {
    std::lock_guard lock(this->locations_mutex_);
    CHECK_ARGUMENT(inner_id < this->locations_.size(), "invalid inner id for RaBitQ split bucket");
    CHECK_ARGUMENT(this->locations_[inner_id] != INVALID_LOCATION,
                   "invalid inner id for RaBitQ split bucket");
    const auto location = this->locations_[inner_id];
    constexpr uint64_t mask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;
    return {static_cast<BucketIdType>(location >> LOCATION_SPLIT_BIT),
            static_cast<InnerIdType>(location & mask)};
}

void
RaBitQSplitBucketDataCell::rebuild_locations() {
    std::lock_guard lock(this->locations_mutex_);
    this->locations_.clear();
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        for (InnerIdType offset = 0; offset < this->inner_ids_[bucket_id].size(); ++offset) {
            const auto inner_id = this->inner_ids_[bucket_id][offset];
            if (inner_id == EMPTY_INNER_ID) {
                continue;
            }
            if (this->locations_.size() <= inner_id) {
                this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
            }
            this->locations_[inner_id] = this->pack_location(bucket_id, offset);
        }
    }
}

}  // namespace vsag
