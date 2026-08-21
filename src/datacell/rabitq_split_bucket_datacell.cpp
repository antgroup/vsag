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
#include <future>
#include <memory>

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
      residual_centroid_transforms_(common_param.allocator_.get()),
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
    this->codes_->EnableExternalFilterCodeStorage();

    this->bucket_count_ = static_cast<BucketIdType>(param->buckets_count);
    this->code_size_ = this->codes_->code_size_;
    this->use_residual_ = param->use_residual_;
    this->backend_ = DistanceEvaluationBackend::RABITQ;
    this->fastscan_block_size_ = this->codes_->GetFastScan32BlockSize();
    if (this->use_l2_residual_query()) {
        this->residual_transform_size_ = this->codes_->GetResidualQueryTransformSize();
    }
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

RaBitQSplitBucketDataCell::SplitBucketComputer&
RaBitQSplitBucketDataCell::get_bucket_computer(const ComputerInterfacePtr& computer) {
    auto* bucket_computer = dynamic_cast<SplitBucketComputer*>(computer.get());
    CHECK_ARGUMENT(bucket_computer != nullptr, "invalid RaBitQ split bucket computer");
    return *bucket_computer;
}

bool
RaBitQSplitBucketDataCell::use_l2_residual_query() const {
    return this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR and
           this->codes_->SupportResidualQueryTransform();
}

std::pair<ComputerInterfacePtr, ComputerInterfacePtr>
RaBitQSplitBucketDataCell::get_scan_computers(SplitBucketComputer& computer,
                                              BucketIdType bucket_id) {
    if (not this->use_l2_residual_query()) {
        return {computer.inner_, computer.fastscan_};
    }

    if (computer.routed_buckets_prepared_) {
        for (uint64_t i = 0; i < computer.bucket_ids_.size(); ++i) {
            if (computer.bucket_ids_[i] == bucket_id) {
                return {computer.bucket_inner_computers_[i],
                        computer.bucket_fastscan_computers_[i]};
            }
        }
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "residual query computer was not prepared for the requested bucket");
    }

    std::lock_guard lock(computer.bucket_computers_mutex_);
    for (uint64_t i = 0; i < computer.bucket_ids_.size(); ++i) {
        if (computer.bucket_ids_[i] == bucket_id) {
            return {computer.bucket_inner_computers_[i], computer.bucket_fastscan_computers_[i]};
        }
    }
    this->append_scan_computer(computer, bucket_id);
    return {computer.bucket_inner_computers_.back(), computer.bucket_fastscan_computers_.back()};
}

void
RaBitQSplitBucketDataCell::prepare_scan_computers(SplitBucketComputer& computer,
                                                  const BucketIdType* bucket_ids,
                                                  uint64_t bucket_count) {
    if (not this->use_l2_residual_query()) {
        return;
    }
    if (bucket_count != 0 and bucket_ids == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "bucket ids are required to prepare residual query computers");
    }

    std::lock_guard lock(computer.bucket_computers_mutex_);
    CHECK_ARGUMENT(not computer.routed_buckets_prepared_,
                   "residual query computers were already prepared");
    computer.bucket_ids_.reserve(bucket_count);
    computer.bucket_inner_computers_.reserve(bucket_count);
    computer.bucket_fastscan_computers_.reserve(bucket_count);
    CHECK_ARGUMENT(this->residual_centroid_transforms_.size() ==
                       static_cast<uint64_t>(this->bucket_count_) * this->residual_transform_size_,
                   "missing transformed residual centroids");
    for (uint64_t i = 0; i < bucket_count; ++i) {
        const auto bucket_id = bucket_ids[i];
        if (bucket_id < 0) {
            continue;
        }
        this->check_valid_bucket_id(bucket_id);
        if (std::find(computer.bucket_ids_.begin(), computer.bucket_ids_.end(), bucket_id) !=
            computer.bucket_ids_.end()) {
            continue;
        }
        computer.bucket_ids_.push_back(bucket_id);
    }

    const uint64_t prepared_count = computer.bucket_ids_.size();
    Vector<float> residual_queries(prepared_count * this->residual_transform_size_,
                                   this->allocator_);
    for (uint64_t i = 0; i < prepared_count; ++i) {
        const auto bucket_id = computer.bucket_ids_[i];
        FP32Sub(computer.transformed_query_.data(),
                this->residual_centroid_transforms_.data() +
                    static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                residual_queries.data() + i * this->residual_transform_size_,
                this->residual_transform_size_);
    }
    computer.bucket_inner_computers_.resize(prepared_count);
    computer.bucket_fastscan_computers_.resize(prepared_count);
    if (prepared_count != 0) {
        this->codes_->FactoryFastScan32ComputersFromResidualQueries(
            residual_queries.data(),
            prepared_count,
            computer.bucket_inner_computers_.data(),
            computer.bucket_fastscan_computers_.data());
    }
    computer.routed_buckets_prepared_ = true;
}

void
RaBitQSplitBucketDataCell::append_scan_computer(SplitBucketComputer& computer,
                                                BucketIdType bucket_id) {
    CHECK_ARGUMENT(this->residual_centroid_transforms_.size() ==
                       static_cast<uint64_t>(this->bucket_count_) * this->residual_transform_size_,
                   "missing transformed residual centroids");
    const uint64_t required_size = computer.bucket_ids_.size() + 1;
    computer.bucket_ids_.reserve(required_size);
    computer.bucket_inner_computers_.reserve(required_size);
    computer.bucket_fastscan_computers_.reserve(required_size);
    FP32Sub(computer.transformed_query_.data(),
            this->residual_centroid_transforms_.data() +
                static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
            computer.residual_query_scratch_.data(),
            this->residual_transform_size_);
    auto inner =
        this->codes_->FactoryComputerFromResidualQuery(computer.residual_query_scratch_.data());
    auto fastscan = this->codes_->FactoryFastScan32Computer(inner);
    computer.bucket_ids_.push_back(bucket_id);
    computer.bucket_inner_computers_.push_back(std::move(inner));
    computer.bucket_fastscan_computers_.push_back(std::move(fastscan));
}

void
RaBitQSplitBucketDataCell::rebuild_residual_centroid_transforms() {
    this->residual_centroid_transforms_ready_.store(false, std::memory_order_relaxed);
    if (not this->use_l2_residual_query()) {
        this->residual_centroid_transforms_.clear();
        this->residual_centroid_transforms_ready_.store(true, std::memory_order_release);
        return;
    }

    Vector<float> zero(this->dim_, 0.0F, this->allocator_);
    Vector<float> zero_transform(this->residual_transform_size_, this->allocator_);
    this->codes_->TransformResidualQuery(zero.data(), zero_transform.data());
    this->residual_centroid_transforms_.resize(static_cast<uint64_t>(this->bucket_count_) *
                                               this->residual_transform_size_);
    auto transform_centroid = [this, &zero_transform](BucketIdType bucket_id) {
        Vector<float> centroid(this->dim_, this->allocator_);
        this->strategy_->GetCentroid(bucket_id, centroid);
        float* transformed_centroid =
            this->residual_centroid_transforms_.data() +
            static_cast<uint64_t>(bucket_id) * this->residual_transform_size_;
        this->codes_->TransformResidualQuery(centroid.data(), transformed_centroid);
        for (uint64_t i = 0; i < this->residual_transform_size_; ++i) {
            transformed_centroid[i] -= zero_transform[i];
        }
    };
    if (this->common_param_.thread_pool_ == nullptr) {
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            transform_centroid(bucket_id);
        }
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(this->bucket_count_);
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            futures.emplace_back(
                this->common_param_.thread_pool_->GeneralEnqueue(transform_centroid, bucket_id));
        }
        for (auto& future : futures) {
            future.get();
        }
    }
    this->residual_centroid_transforms_ready_.store(true, std::memory_order_release);
}

void
RaBitQSplitBucketDataCell::ensure_residual_centroid_transforms() {
    if (not this->use_l2_residual_query()) {
        return;
    }
    if (this->residual_centroid_transforms_ready_.load(std::memory_order_acquire)) {
        return;
    }
    const uint64_t expected_size =
        static_cast<uint64_t>(this->bucket_count_) * this->residual_transform_size_;
    std::lock_guard lock(this->residual_centroid_transforms_mutex_);
    if (not this->residual_centroid_transforms_ready_.load(std::memory_order_relaxed) or
        this->residual_centroid_transforms_.size() != expected_size) {
        this->rebuild_residual_centroid_transforms();
    }
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
        // The residual code estimates ||q - (x - c)||^2.  Store
        // ||x - c||^2 - ||x||^2 so search can recover the original-vector distance without
        // decoding the multi-bit RaBitQ code.
        residual_bias = FP32ComputeIP(prepared.data(), prepared.data(), this->dim_) -
                        FP32ComputeIP(input, input, this->dim_);
    }
    return prepared.data();
}

float
RaBitQSplitBucketDataCell::query_centroid_adjustment(const SplitBucketComputer& computer,
                                                     BucketIdType bucket_id) const {
    if (not this->use_residual_) {
        return 0.0F;
    }
    return computer.query_centroid_adjustments_[bucket_id];
}

float
RaBitQSplitBucketDataCell::residual_adjustment(float query_centroid_adjustment,
                                               BucketIdType bucket_id,
                                               InnerIdType offset_id) const {
    if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
        query_centroid_adjustment += this->residual_bias_[bucket_id][offset_id];
    }
    return query_centroid_adjustment;
}

void
RaBitQSplitBucketDataCell::ScanBucketById(float* result_dists,
                                          const ComputerInterfacePtr& computer,
                                          const BucketIdType& bucket_id,
                                          QueryContext* ctx) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    const auto bucket_size = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    const auto [scan_inner, scan_fastscan] = this->get_scan_computers(bucket_computer, bucket_id);
    const float query_centroid_adjustment =
        this->use_l2_residual_query() ? 0.0F
                                      : this->query_centroid_adjustment(bucket_computer, bucket_id);
    const uint64_t block_count =
        (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    if (scan_fastscan != nullptr and this->fastscan_block_size_ > 0 and
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
                scan_inner,
                scan_fastscan,
                this->fastscan_blocks_[bucket_id].data() + block_index * this->fastscan_block_size_,
                valid_size,
                ctx);
            for (InnerIdType i = 0; i < valid_size; ++i) {
                const InnerIdType offset = begin + i;
                const auto inner_id = this->inner_ids_[bucket_id][offset];
                if (inner_id == EMPTY_INNER_ID) {
                    result_dists[offset] = std::numeric_limits<float>::max();
                } else if (computed[i]) {
                    if (not this->use_l2_residual_query()) {
                        result_dists[offset] -=
                            this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
                    }
                } else {
                    fallback_ids.push_back(inner_id);
                    fallback_positions.push_back(offset);
                }
            }
        }

        if (not fallback_ids.empty()) {
            Vector<float> fallback_dists(fallback_ids.size(), this->allocator_);
            Vector<uint8_t> fallback_filter_codes(
                fallback_ids.size() * this->codes_->GetFilterCodeSize(), this->allocator_);
            for (uint64_t i = 0; i < fallback_positions.size(); ++i) {
                this->get_filter_code(
                    bucket_id,
                    fallback_positions[i],
                    fallback_filter_codes.data() + i * this->codes_->GetFilterCodeSize());
            }
            this->codes_->QueryWithFilterCodes(fallback_dists.data(),
                                               nullptr,
                                               nullptr,
                                               scan_inner,
                                               fallback_ids.data(),
                                               fallback_filter_codes.data(),
                                               static_cast<InnerIdType>(fallback_ids.size()),
                                               ctx);
            for (uint64_t i = 0; i < fallback_ids.size(); ++i) {
                const auto offset = fallback_positions[i];
                result_dists[offset] = fallback_dists[i];
                if (not this->use_l2_residual_query()) {
                    result_dists[offset] -=
                        this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
                }
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
    Vector<uint8_t> filter_codes(ids.size() * this->codes_->GetFilterCodeSize(), this->allocator_);
    for (uint64_t i = 0; i < positions.size(); ++i) {
        this->get_filter_code(
            bucket_id, positions[i], filter_codes.data() + i * this->codes_->GetFilterCodeSize());
    }
    this->codes_->QueryWithFilterCodes(compact_dists.data(),
                                       nullptr,
                                       nullptr,
                                       scan_inner,
                                       ids.data(),
                                       filter_codes.data(),
                                       static_cast<InnerIdType>(ids.size()),
                                       ctx);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        const auto offset = positions[i];
        result_dists[offset] = compact_dists[i];
        if (not this->use_l2_residual_query()) {
            result_dists[offset] -=
                this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
        }
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
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    const auto [scan_inner, scan_fastscan] = this->get_scan_computers(bucket_computer, bucket_id);
    const float query_centroid_adjustment =
        this->use_l2_residual_query() ? 0.0F
                                      : this->query_centroid_adjustment(bucket_computer, bucket_id);
    const uint64_t block_count =
        (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    if (scan_fastscan != nullptr and this->fastscan_block_size_ > 0 and
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
                scan_inner,
                scan_fastscan,
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
                    if (not this->use_l2_residual_query()) {
                        const float adjustment =
                            this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
                        result_dists[offset] -= adjustment;
                        lower_bounds[offset] -= adjustment;
                    }
                } else {
                    fallback_ids.push_back(inner_id);
                    fallback_positions.push_back(offset);
                }
            }
        }

        if (not fallback_ids.empty()) {
            Vector<float> fallback_dists(fallback_ids.size(), this->allocator_);
            Vector<uint8_t> fallback_filter_codes(
                fallback_ids.size() * this->codes_->GetFilterCodeSize(), this->allocator_);
            for (uint64_t i = 0; i < fallback_positions.size(); ++i) {
                this->get_filter_code(
                    bucket_id,
                    fallback_positions[i],
                    fallback_filter_codes.data() + i * this->codes_->GetFilterCodeSize());
            }
            this->codes_->QueryWithFilterCodes(fallback_dists.data(),
                                               nullptr,
                                               nullptr,
                                               scan_inner,
                                               fallback_ids.data(),
                                               fallback_filter_codes.data(),
                                               static_cast<InnerIdType>(fallback_ids.size()),
                                               ctx);
            for (uint64_t i = 0; i < fallback_ids.size(); ++i) {
                const auto offset = fallback_positions[i];
                result_dists[offset] = fallback_dists[i];
                if (not this->use_l2_residual_query()) {
                    result_dists[offset] -=
                        this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
                }
                lower_bounds[offset] = -std::numeric_limits<float>::infinity();
                filter_inner_products[offset] = std::numeric_limits<float>::quiet_NaN();
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
    Vector<uint8_t> filter_codes(ids.size() * this->codes_->GetFilterCodeSize(), this->allocator_);
    for (uint64_t i = 0; i < positions.size(); ++i) {
        this->get_filter_code(
            bucket_id, positions[i], filter_codes.data() + i * this->codes_->GetFilterCodeSize());
    }
    this->codes_->QueryWithFilterCodes(compact_dists.data(),
                                       nullptr,
                                       nullptr,
                                       scan_inner,
                                       ids.data(),
                                       filter_codes.data(),
                                       static_cast<InnerIdType>(ids.size()),
                                       ctx);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        const auto offset = positions[i];
        result_dists[offset] = compact_dists[i];
        if (not this->use_l2_residual_query()) {
            result_dists[offset] -=
                this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
        }
        lower_bounds[offset] = -std::numeric_limits<float>::infinity();
        filter_inner_products[offset] = std::numeric_limits<float>::quiet_NaN();
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
    ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
    this->get_filter_code(bucket_id, offset_id, filter_code.data);
    float distance = 0.0F;
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    if (this->use_l2_residual_query()) {
        const auto scan_inner = this->get_scan_computers(bucket_computer, bucket_id).first;
        this->codes_->QueryWithFilterCodes(
            &distance, nullptr, nullptr, scan_inner, &inner_id, filter_code.data, 1, nullptr);
        return distance;
    }
    this->codes_->QueryWithFilterCodes(&distance,
                                       nullptr,
                                       nullptr,
                                       bucket_computer.inner_,
                                       &inner_id,
                                       filter_code.data,
                                       1,
                                       nullptr);
    const float query_centroid_adjustment =
        this->query_centroid_adjustment(bucket_computer, bucket_id);
    return distance - this->residual_adjustment(query_centroid_adjustment, bucket_id, offset_id);
}

void
RaBitQSplitBucketDataCell::QueryWithFilterInnerProductByInnerId(
    float* result_dists,
    const float* filter_inner_products,
    const ComputerInterfacePtr& computer,
    const InnerIdType* inner_ids,
    InnerIdType id_count,
    QueryContext* ctx) {
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    if (this->use_l2_residual_query()) {
        this->query_residual_by_inner_ids(result_dists,
                                          nullptr,
                                          filter_inner_products,
                                          bucket_computer,
                                          inner_ids,
                                          id_count,
                                          ctx);
        return;
    }

    Vector<float> adjustments(id_count, 0.0F, this->allocator_);
    Vector<uint8_t> filter_codes(this->allocator_);
    this->collect_filter_codes(inner_ids, id_count, filter_codes, &adjustments, &bucket_computer);
    this->codes_->QueryWithFilterCodes(result_dists,
                                       nullptr,
                                       filter_inner_products,
                                       bucket_computer.inner_,
                                       inner_ids,
                                       filter_codes.data(),
                                       id_count,
                                       ctx);
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
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    if (this->use_l2_residual_query()) {
        this->query_residual_by_inner_ids(
            result_dists, hint_dists, nullptr, bucket_computer, inner_ids, id_count, ctx);
        return;
    }

    Vector<float> raw_hints(id_count, this->allocator_);
    Vector<float> adjustments(id_count, 0.0F, this->allocator_);
    Vector<uint8_t> filter_codes(this->allocator_);
    this->collect_filter_codes(inner_ids, id_count, filter_codes, &adjustments, &bucket_computer);
    for (InnerIdType i = 0; i < id_count; ++i) {
        raw_hints[i] = hint_dists == nullptr ? std::numeric_limits<float>::max()
                                             : hint_dists[i] + adjustments[i];
    }
    this->codes_->QueryWithFilterCodes(result_dists,
                                       raw_hints.data(),
                                       nullptr,
                                       RaBitQSplitBucketDataCell::get_inner_computer(computer),
                                       inner_ids,
                                       filter_codes.data(),
                                       id_count,
                                       ctx);
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[i] -= adjustments[i];
    }
}

void
RaBitQSplitBucketDataCell::query_residual_by_inner_ids(float* result_dists,
                                                       const float* hint_dists,
                                                       const float* filter_inner_products,
                                                       SplitBucketComputer& computer,
                                                       const InnerIdType* inner_ids,
                                                       InnerIdType id_count,
                                                       QueryContext* ctx) {
    struct reorder_entry {
        BucketIdType bucket_id;
        InnerIdType offset_id;
        InnerIdType inner_id;
        InnerIdType result_offset;
    };

    Vector<reorder_entry> entries(id_count, this->allocator_);
    {
        std::lock_guard lock(this->locations_mutex_);
        for (InnerIdType i = 0; i < id_count; ++i) {
            CHECK_ARGUMENT(inner_ids[i] < this->locations_.size(),
                           "invalid inner id for RaBitQ split bucket");
            const uint64_t location = this->locations_[inner_ids[i]];
            CHECK_ARGUMENT(location != INVALID_LOCATION,
                           "invalid inner id for RaBitQ split bucket");
            entries[i] = {static_cast<BucketIdType>(location >> LOCATION_SPLIT_BIT),
                          static_cast<InnerIdType>(location & 0xFFFFFFFFULL),
                          inner_ids[i],
                          i};
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bucket_id < rhs.bucket_id;
    });

    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    Vector<InnerIdType> sorted_ids(id_count, this->allocator_);
    Vector<float> sorted_dists(id_count, this->allocator_);
    Vector<float> sorted_hints(hint_dists == nullptr ? 0 : id_count, this->allocator_);
    Vector<float> sorted_filter_inner_products(filter_inner_products == nullptr ? 0 : id_count,
                                               this->allocator_);
    Vector<uint8_t> filter_codes(static_cast<uint64_t>(id_count) * filter_code_size,
                                 this->allocator_);
    for (InnerIdType i = 0; i < id_count; ++i) {
        sorted_ids[i] = entries[i].inner_id;
        if (hint_dists != nullptr) {
            sorted_hints[i] = hint_dists[entries[i].result_offset];
        }
        if (filter_inner_products != nullptr) {
            sorted_filter_inner_products[i] = filter_inner_products[entries[i].result_offset];
        }
    }

    InnerIdType group_begin = 0;
    while (group_begin < id_count) {
        const BucketIdType bucket_id = entries[group_begin].bucket_id;
        InnerIdType group_end = group_begin + 1;
        while (group_end < id_count and entries[group_end].bucket_id == bucket_id) {
            ++group_end;
        }
        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        for (InnerIdType i = group_begin; i < group_end; ++i) {
            this->check_valid_offset(bucket_id, entries[i].offset_id);
            this->get_filter_code(
                bucket_id,
                entries[i].offset_id,
                filter_codes.data() + static_cast<uint64_t>(i) * filter_code_size);
        }
        const auto scan_inner = this->get_scan_computers(computer, bucket_id).first;
        this->codes_->QueryWithFilterCodes(
            sorted_dists.data() + group_begin,
            hint_dists == nullptr ? nullptr : sorted_hints.data() + group_begin,
            filter_inner_products == nullptr ? nullptr
                                             : sorted_filter_inner_products.data() + group_begin,
            scan_inner,
            sorted_ids.data() + group_begin,
            filter_codes.data() + static_cast<uint64_t>(group_begin) * filter_code_size,
            group_end - group_begin,
            ctx);
        group_begin = group_end;
    }
    for (InnerIdType i = 0; i < id_count; ++i) {
        result_dists[entries[i].result_offset] = sorted_dists[i];
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
    ByteBuffer filter_code1(this->codes_->GetFilterCodeSize(), this->allocator_);
    ByteBuffer filter_code2(this->codes_->GetFilterCodeSize(), this->allocator_);
    this->get_filter_code(bucket_id, id1, filter_code1.data);
    this->get_filter_code(bucket_id, id2, filter_code2.data);
    return this->codes_->ComputePairVectorsWithFilterCodes(this->inner_ids_[bucket_id][id1],
                                                           filter_code1.data,
                                                           this->inner_ids_[bucket_id][id2],
                                                           filter_code2.data);
}

ComputerInterfacePtr
RaBitQSplitBucketDataCell::FactoryComputer(const void* query) {
    return this->factory_computer(query, nullptr, 0);
}

ComputerInterfacePtr
RaBitQSplitBucketDataCell::FactoryComputerForBuckets(const void* query,
                                                     const BucketIdType* bucket_ids,
                                                     uint64_t bucket_count) {
    return this->factory_computer(query, bucket_ids, bucket_count);
}

ComputerInterfacePtr
RaBitQSplitBucketDataCell::factory_computer(const void* query,
                                            const BucketIdType* bucket_ids,
                                            uint64_t bucket_count) {
    const auto* input = static_cast<const float*>(query);
    this->ensure_residual_centroid_transforms();
    ComputerInterfacePtr inner = nullptr;
    ComputerInterfacePtr fastscan = nullptr;
    if (not this->use_l2_residual_query()) {
        inner = this->codes_->FactoryComputer(input);
        fastscan = this->codes_->FactoryFastScan32Computer(inner);
    }
    const bool use_l2_residual_query = this->use_l2_residual_query();
    const uint64_t adjustment_count = this->use_residual_ and not use_l2_residual_query
                                          ? static_cast<uint64_t>(this->bucket_count_)
                                          : 0;
    const uint64_t bucket_computer_capacity = use_l2_residual_query ? bucket_count : 0;
    auto computer = std::make_shared<SplitBucketComputer>(
        std::move(inner),
        std::move(fastscan),
        use_l2_residual_query ? 0 : this->dim_,
        adjustment_count,
        bucket_computer_capacity,
        use_l2_residual_query ? this->residual_transform_size_ : 0,
        this->allocator_);
    if (not use_l2_residual_query) {
        if (this->metric_ == MetricType::METRIC_TYPE_COSINE and this->use_residual_) {
            Normalize(input, computer->raw_query_.data(), this->dim_);
        } else {
            std::memcpy(computer->raw_query_.data(), input, sizeof(float) * this->dim_);
        }
    }
    if (use_l2_residual_query) {
        this->codes_->TransformResidualQuery(input, computer->transformed_query_.data());
        if (bucket_ids != nullptr or bucket_count != 0) {
            this->prepare_scan_computers(*computer, bucket_ids, bucket_count);
        }
    }
    if (this->use_residual_ and not use_l2_residual_query) {
        // A computer is shared by all scanned buckets for one query.  Precompute the query-only
        // centroid term once so neither FastScan nor exact-distance heap pays a dot product per
        // candidate.
        Vector<float> centroid(this->dim_, this->allocator_);
        for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
            this->strategy_->GetCentroid(bucket_id, centroid);
            float adjustment =
                FP32ComputeIP(computer->raw_query_.data(), centroid.data(), this->dim_);
            if (this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
                adjustment *= 2.0F;
            }
            computer->query_centroid_adjustments_[bucket_id] = adjustment;
        }
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
    this->rebuild_residual_centroid_transforms();
}

bool
RaBitQSplitBucketDataCell::BeginOptimizedBuild(const FlattenOptimizedBuildContext& context,
                                               InnerIdType capacity) {
    if (this->optimized_build_codes_ != nullptr) {
        return false;
    }
    auto optimized_build = std::dynamic_pointer_cast<FlattenOptimizedBuildInterface>(this->codes_);
    if (optimized_build == nullptr or not optimized_build->BeginOptimizedBuild(context)) {
        return false;
    }
    try {
        this->codes_->Resize(capacity);
    } catch (...) {
        optimized_build->AbortOptimizedBuild();
        throw;
    }
    this->optimized_build_codes_ = std::move(optimized_build);
    return true;
}

void
RaBitQSplitBucketDataCell::FinalizeOptimizedBuild() {
    if (this->optimized_build_codes_ == nullptr) {
        return;
    }
    try {
        // Package while the temporary scalar build codes are still available. The lower data cell
        // then materializes only the supplement bits and releases the temporary scalar codes.
        this->package_fastscan(true);
        this->optimized_build_codes_->FinalizeOptimizedBuild();
        {
            std::lock_guard codes_lock(this->codes_insert_mutex_);
            this->codes_->DiscardFilterCodes();
        }
    } catch (...) {
        this->optimized_build_codes_->AbortOptimizedBuild();
        this->optimized_build_codes_.reset();
        throw;
    }
    this->optimized_build_codes_.reset();
}

void
RaBitQSplitBucketDataCell::AbortOptimizedBuild() noexcept {
    if (this->optimized_build_codes_ == nullptr) {
        return;
    }
    this->optimized_build_codes_->AbortOptimizedBuild();
    this->optimized_build_codes_.reset();
}

bool
RaBitQSplitBucketDataCell::IsOptimizedBuildActive() const {
    return this->optimized_build_codes_ != nullptr and
           this->optimized_build_codes_->IsOptimizedBuildActive();
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
    if (this->optimized_build_codes_ != nullptr) {
        // BeginOptimizedBuild pre-sizes scalar storage, and IVF assigns disjoint inner IDs.
        this->codes_->InsertVector(input, inner_id);
    } else {
        ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        this->codes_->InsertVectorWithFilterCode(input, inner_id, filter_code.data);
        this->set_filter_code(this->fastscan_blocks_[bucket_id], offset_id, filter_code.data);
    }
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
    if (this->optimized_build_codes_ != nullptr) {
        // BeginOptimizedBuild pre-sizes scalar storage, and IVF assigns disjoint inner IDs.
        this->codes_->InsertVector(input, inner_id);
    } else {
        ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        this->codes_->InsertVectorWithFilterCode(input, inner_id, filter_code.data);
        this->set_filter_code(this->fastscan_blocks_[bucket_id], offset_id, filter_code.data);
    }
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
    this->codes_->PrefetchSupplement(this->inner_ids_[bucket_id][offset_id]);
}

void
RaBitQSplitBucketDataCell::GetCodesById(BucketIdType bucket_id,
                                        InnerIdType offset_id,
                                        uint8_t* data) const {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    this->check_valid_offset(bucket_id, offset_id);
    ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
    this->get_filter_code(bucket_id, offset_id, filter_code.data);
    CHECK_ARGUMENT(this->codes_->GetCodesByIdWithFilterCode(
                       this->inner_ids_[bucket_id][offset_id], filter_code.data, data),
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
    CHECK_ARGUMENT(source->bucket_count_ == this->bucket_count_,
                   "merge RaBitQ split bucket count mismatch");
    CHECK_ARGUMENT(source->fastscan_block_size_ == this->fastscan_block_size_,
                   "merge RaBitQ split bucket FastScan layout mismatch");
    this->codes_->MergeSupplementCodes(source->codes_, bias);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        std::scoped_lock lock(this->bucket_mutexes_[bucket_id], source->bucket_mutexes_[bucket_id]);
        const auto old_size = this->inner_ids_[bucket_id].size();
        const auto source_size = source->inner_ids_[bucket_id].size();
        Vector<uint8_t> merged_blocks(this->allocator_);
        ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
        for (InnerIdType offset = 0; offset < old_size; ++offset) {
            if (this->inner_ids_[bucket_id][offset] == EMPTY_INNER_ID) {
                continue;
            }
            this->get_filter_code(this->fastscan_blocks_[bucket_id], offset, filter_code.data);
            this->set_filter_code(merged_blocks, offset, filter_code.data);
        }
        for (InnerIdType offset = 0; offset < source_size; ++offset) {
            if (source->inner_ids_[bucket_id][offset] == EMPTY_INNER_ID) {
                continue;
            }
            source->get_filter_code(source->fastscan_blocks_[bucket_id], offset, filter_code.data);
            this->set_filter_code(
                merged_blocks, static_cast<InnerIdType>(old_size + offset), filter_code.data);
        }
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
        this->fastscan_blocks_[bucket_id] = std::move(merged_blocks);
    }
    this->rebuild_locations();
}

void
RaBitQSplitBucketDataCell::Package() {
    this->package_fastscan();
    std::lock_guard codes_lock(this->codes_insert_mutex_);
    this->codes_->DiscardFilterCodes();
}

void
RaBitQSplitBucketDataCell::Unpack() {
    // Packed 32-vector blocks are the canonical IVF filter-code storage. Keeping them packed also
    // allows Add and search after Package without recreating a scalar x-bit copy.
}

void
RaBitQSplitBucketDataCell::package_fastscan(bool force) {
    if (not this->codes_->SupportFastScan32()) {
        return;
    }

    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        if (not force and this->packed_filter_codes_complete(bucket_id)) {
            continue;
        }
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
RaBitQSplitBucketDataCell::set_filter_code(Vector<uint8_t>& blocks,
                                           InnerIdType offset_id,
                                           const uint8_t* filter_code) const {
    CHECK_ARGUMENT(this->fastscan_block_size_ > 0, "invalid FastScan block size");
    const uint64_t block_index = static_cast<uint64_t>(offset_id) / FASTSCAN_BATCH_SIZE;
    const auto index_in_block =
        static_cast<InnerIdType>(static_cast<uint64_t>(offset_id) % FASTSCAN_BATCH_SIZE);
    const uint64_t required_size = (block_index + 1) * this->fastscan_block_size_;
    if (blocks.size() < required_size) {
        blocks.resize(required_size, 0);
    }
    this->codes_->SetFastScan32Code(
        filter_code, index_in_block, blocks.data() + block_index * this->fastscan_block_size_);
}

void
RaBitQSplitBucketDataCell::get_filter_code(BucketIdType bucket_id,
                                           InnerIdType offset_id,
                                           uint8_t* filter_code) const {
    this->get_filter_code(this->fastscan_blocks_[bucket_id], offset_id, filter_code);
}

void
RaBitQSplitBucketDataCell::get_filter_code(const Vector<uint8_t>& blocks,
                                           InnerIdType offset_id,
                                           uint8_t* filter_code) const {
    CHECK_ARGUMENT(this->fastscan_block_size_ > 0, "invalid FastScan block size");
    const uint64_t block_index = static_cast<uint64_t>(offset_id) / FASTSCAN_BATCH_SIZE;
    const auto index_in_block =
        static_cast<InnerIdType>(static_cast<uint64_t>(offset_id) % FASTSCAN_BATCH_SIZE);
    CHECK_ARGUMENT((block_index + 1) * this->fastscan_block_size_ <= blocks.size(),
                   "missing packed RaBitQ filter code");
    this->codes_->UnpackFastScan32Code(
        blocks.data() + block_index * this->fastscan_block_size_, index_in_block, filter_code);
}

void
RaBitQSplitBucketDataCell::collect_filter_codes(const InnerIdType* inner_ids,
                                                InnerIdType id_count,
                                                Vector<uint8_t>& filter_codes,
                                                Vector<float>* adjustments,
                                                const SplitBucketComputer* computer) const {
    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    filter_codes.resize(static_cast<uint64_t>(id_count) * filter_code_size);
    if (adjustments != nullptr) {
        CHECK_ARGUMENT(computer != nullptr, "residual adjustments require a bucket computer");
        adjustments->resize(id_count, 0.0F);
    }
    for (InnerIdType i = 0; i < id_count; ++i) {
        const auto [bucket_id, offset_id] = this->get_location(inner_ids[i]);
        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        this->check_valid_offset(bucket_id, offset_id);
        this->get_filter_code(bucket_id,
                              offset_id,
                              filter_codes.data() + static_cast<uint64_t>(i) * filter_code_size);
        if (adjustments != nullptr) {
            const float query_centroid_adjustment =
                this->query_centroid_adjustment(*computer, bucket_id);
            (*adjustments)[i] =
                this->residual_adjustment(query_centroid_adjustment, bucket_id, offset_id);
        }
    }
}

bool
RaBitQSplitBucketDataCell::packed_filter_codes_complete(BucketIdType bucket_id) const {
    const uint64_t bucket_size = this->inner_ids_[bucket_id].size();
    const uint64_t block_count = (bucket_size + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    return this->fastscan_blocks_[bucket_id].size() == block_count * this->fastscan_block_size_;
}

void
RaBitQSplitBucketDataCell::serialize_packed_filter_codes(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, PACKED_FILTER_STORAGE_MAGIC);
    StreamWriter::WriteObj(writer, PACKED_FILTER_STORAGE_VERSION);
    StreamWriter::WriteObj(writer, this->fastscan_block_size_);
    const auto bucket_count = static_cast<uint64_t>(this->bucket_count_);
    StreamWriter::WriteObj(writer, bucket_count);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        CHECK_ARGUMENT(this->packed_filter_codes_complete(bucket_id),
                       "cannot serialize incomplete packed RaBitQ filter codes");
        StreamWriter::WriteVector(writer, this->fastscan_blocks_[bucket_id]);
    }
}

void
RaBitQSplitBucketDataCell::deserialize_packed_filter_codes(StreamReader& reader) {
    uint64_t magic = 0;
    uint32_t version = 0;
    uint64_t block_size = 0;
    uint64_t bucket_count = 0;
    StreamReader::ReadObj(reader, magic);
    StreamReader::ReadObj(reader, version);
    StreamReader::ReadObj(reader, block_size);
    StreamReader::ReadObj(reader, bucket_count);
    CHECK_ARGUMENT(magic == PACKED_FILTER_STORAGE_MAGIC,
                   "invalid packed RaBitQ filter-code storage marker");
    CHECK_ARGUMENT(version == PACKED_FILTER_STORAGE_VERSION,
                   "unsupported packed RaBitQ filter-code storage version");
    CHECK_ARGUMENT(block_size == this->fastscan_block_size_,
                   "packed RaBitQ FastScan block size mismatch");
    CHECK_ARGUMENT(bucket_count == static_cast<uint64_t>(this->bucket_count_),
                   "packed RaBitQ filter-code bucket count mismatch");
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        StreamReader::ReadVector(reader, this->fastscan_blocks_[bucket_id]);
        CHECK_ARGUMENT(this->packed_filter_codes_complete(bucket_id),
                       "invalid packed RaBitQ filter-code block count");
    }
}

void
RaBitQSplitBucketDataCell::Serialize(StreamWriter& writer) {
    this->Package();
    BucketInterface::Serialize(writer);
    this->codes_->Serialize(writer);
    for (BucketIdType bucket_id = 0; bucket_id < this->bucket_count_; ++bucket_id) {
        StreamWriter::WriteVector(writer, this->inner_ids_[bucket_id]);
        if (this->use_residual_) {
            StreamWriter::WriteVector(writer, this->residual_bias_[bucket_id]);
        }
    }
    this->serialize_packed_filter_codes(writer);
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
    bool has_packed_filter_codes = false;
    if (reader->GetCursor() + sizeof(PACKED_FILTER_STORAGE_MAGIC) <= reader->Length()) {
        const uint64_t cursor = reader->GetCursor();
        uint64_t magic = 0;
        StreamReader::ReadObj(*reader, magic);
        reader->Seek(cursor);
        has_packed_filter_codes = magic == PACKED_FILTER_STORAGE_MAGIC;
    }
    if (has_packed_filter_codes) {
        this->deserialize_packed_filter_codes(*reader);
    } else {
        // Legacy indexes stored scalar xbits in the lower datacell. Convert them once at load.
        if (not this->codes_->HasFilterCodes() and this->codes_->TotalCount() != 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "serialized RaBitQ split bucket has no filter codes");
        }
        this->package_fastscan(true);
    }
    this->codes_->DiscardFilterCodes();
    this->backend_ = DistanceEvaluationBackend::RABITQ;
    this->rebuild_locations();
    this->residual_centroid_transforms_.clear();
    this->residual_centroid_transforms_ready_.store(false, std::memory_order_release);
}

void
RaBitQSplitBucketDataCell::FinalizeLoad() {
    this->rebuild_residual_centroid_transforms();
}

uint64_t
RaBitQSplitBucketDataCell::GetMemoryUsage() const {
    uint64_t memory = sizeof(RaBitQSplitBucketDataCell) + this->codes_->GetMemoryUsage();
    memory += this->locations_.capacity() * sizeof(uint64_t);
    memory += this->residual_centroid_transforms_.capacity() * sizeof(float);
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
