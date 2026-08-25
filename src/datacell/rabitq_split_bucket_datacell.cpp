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
#include <array>
#include <cstring>
#include <future>
#include <memory>

#include "inner_string_params.h"
#include "rabitq_split_datacell.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"

namespace vsag {

namespace {

bool
IsFiniteFloat(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
}

RaBitQSplitResidualOriginalQueryInterface&
GetResidualOriginalQueryCodes(const FlattenInterfacePtr& codes) {
    auto* residual_codes = dynamic_cast<RaBitQSplitResidualOriginalQueryInterface*>(codes.get());
    CHECK_ARGUMENT(residual_codes != nullptr,
                   "RaBitQ split codes do not support original-query residual factors");
    return *residual_codes;
}

}  // namespace

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
      bucket_versions_(param->buckets_count, 0, common_param.allocator_.get()),
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
    this->residual_original_query_codes_ = &GetResidualOriginalQueryCodes(this->codes_);

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
                                              BucketIdType bucket_id,
                                              bool require_fastscan) {
    if (not this->use_l2_residual_query()) {
        return {computer.inner_, computer.fastscan_};
    }

    this->check_routed_bucket(computer, bucket_id);

    std::lock_guard lock(computer.bucket_computers_mutex_);
    for (uint64_t i = 0; i < computer.bucket_ids_.size(); ++i) {
        if (computer.bucket_ids_[i] == bucket_id) {
            if (require_fastscan and computer.bucket_fastscan_computers_[i] == nullptr) {
                computer.bucket_fastscan_computers_[i] =
                    this->codes_->FactoryFastScan32Computer(computer.bucket_inner_computers_[i]);
            }
            return {computer.bucket_inner_computers_[i], computer.bucket_fastscan_computers_[i]};
        }
    }
    this->append_scan_computer(computer, bucket_id, require_fastscan);
    return {computer.bucket_inner_computers_.back(), computer.bucket_fastscan_computers_.back()};
}

void
RaBitQSplitBucketDataCell::check_routed_bucket(const SplitBucketComputer& computer,
                                               BucketIdType bucket_id) const {
    if (computer.routed_buckets_prepared_ and
        this->get_routed_bucket_index(computer, bucket_id) == INVALID_ROUTED_BUCKET_INDEX) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "residual query computer was not prepared for the requested bucket");
    }
}

uint64_t
RaBitQSplitBucketDataCell::get_routed_bucket_index(const SplitBucketComputer& computer,
                                                   BucketIdType bucket_id) const {
    const auto it = std::find(
        computer.routed_bucket_ids_.begin(), computer.routed_bucket_ids_.end(), bucket_id);
    if (it == computer.routed_bucket_ids_.end()) {
        return INVALID_ROUTED_BUCKET_INDEX;
    }
    return static_cast<uint64_t>(std::distance(computer.routed_bucket_ids_.begin(), it));
}

void
RaBitQSplitBucketDataCell::prepare_scan_computers(SplitBucketComputer& computer,
                                                  const BucketIdType* bucket_ids,
                                                  uint64_t bucket_count) {
    if (bucket_count != 0 and bucket_ids == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "bucket ids are required to prepare query caches");
    }

    std::lock_guard lock(computer.bucket_computers_mutex_);
    CHECK_ARGUMENT(not computer.routed_buckets_prepared_, "query caches were already prepared");
    Vector<float> residual_scratch(
        this->use_l2_residual_query() ? this->residual_transform_size_ : 0, this->allocator_);
    for (uint64_t i = 0; i < bucket_count; ++i) {
        const auto bucket_id = bucket_ids[i];
        if (bucket_id < 0) {
            continue;
        }
        this->check_valid_bucket_id(bucket_id);
        if (std::find(computer.routed_bucket_ids_.begin(),
                      computer.routed_bucket_ids_.end(),
                      bucket_id) != computer.routed_bucket_ids_.end()) {
            continue;
        }

        computer.routed_bucket_ids_.push_back(bucket_id);
        float query_bucket_norm_sqr = 0.0F;
        if (this->use_l2_residual_query()) {
            FP32Sub(computer.transformed_query_.data(),
                    this->residual_centroid_transforms_.data() +
                        static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                    residual_scratch.data(),
                    this->residual_transform_size_);
            query_bucket_norm_sqr =
                this->codes_->ComputeTransformedResidualQueryNormSqr(residual_scratch.data());
        }
        computer.routed_bucket_norm_sqrs_.push_back(query_bucket_norm_sqr);
    }
    computer.routed_buckets_prepared_ = true;
}

void
RaBitQSplitBucketDataCell::append_scan_computer(SplitBucketComputer& computer,
                                                BucketIdType bucket_id,
                                                bool require_fastscan) {
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
    auto fastscan = require_fastscan ? this->codes_->FactoryFastScan32Computer(inner) : nullptr;
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
                                          QueryContext* ctx,
                                          InnerIdType* scanned_inner_ids,
                                          InnerIdType max_scan_size,
                                          InnerIdType* scanned_size) {
    this->scan_bucket_by_id(result_dists,
                            nullptr,
                            computer,
                            bucket_id,
                            ctx,
                            scanned_inner_ids,
                            max_scan_size,
                            scanned_size);
}

uint64_t
RaBitQSplitBucketDataCell::ScanBucketWithFilterInnerProduct(float* result_dists,
                                                            float* filter_inner_products,
                                                            const ComputerInterfacePtr& computer,
                                                            const BucketIdType& bucket_id,
                                                            QueryContext* ctx,
                                                            InnerIdType* scanned_inner_ids,
                                                            InnerIdType max_scan_size,
                                                            InnerIdType* scanned_size) {
    CHECK_ARGUMENT(filter_inner_products != nullptr,
                   "filter inner-product output is required for candidate scan");
    return this->scan_bucket_by_id(result_dists,
                                   filter_inner_products,
                                   computer,
                                   bucket_id,
                                   ctx,
                                   scanned_inner_ids,
                                   max_scan_size,
                                   scanned_size);
}

uint64_t
RaBitQSplitBucketDataCell::scan_bucket_by_id(float* result_dists,
                                             float* direct_filter_inner_products,
                                             const ComputerInterfacePtr& computer,
                                             const BucketIdType& bucket_id,
                                             QueryContext* ctx,
                                             InnerIdType* scanned_inner_ids,
                                             InnerIdType max_scan_size,
                                             InnerIdType* scanned_size) {
    this->check_valid_bucket_id(bucket_id);
    std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
    const uint64_t source_version = this->bucket_versions_[bucket_id];
    const auto stored_bucket_size = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
    const auto bucket_size = std::min(stored_bucket_size, max_scan_size);
    if (scanned_size != nullptr) {
        *scanned_size = bucket_size;
    }
    if (scanned_inner_ids != nullptr) {
        std::copy_n(this->inner_ids_[bucket_id].begin(), bucket_size, scanned_inner_ids);
    }
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    ComputerInterfacePtr scan_inner = bucket_computer.inner_;
    ComputerInterfacePtr scan_fastscan = bucket_computer.fastscan_;
    float query_bucket_norm_sqr = 0.0F;
    if (this->use_l2_residual_query()) {
        this->check_routed_bucket(bucket_computer, bucket_id);
        const uint64_t routed_bucket_index =
            this->get_routed_bucket_index(bucket_computer, bucket_id);
        if (routed_bucket_index != INVALID_ROUTED_BUCKET_INDEX) {
            query_bucket_norm_sqr = bucket_computer.routed_bucket_norm_sqrs_[routed_bucket_index];
        } else {
            Vector<float> residual_scratch(this->residual_transform_size_, this->allocator_);
            FP32Sub(bucket_computer.transformed_query_.data(),
                    this->residual_centroid_transforms_.data() +
                        static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                    residual_scratch.data(),
                    this->residual_transform_size_);
            query_bucket_norm_sqr =
                this->codes_->ComputeTransformedResidualQueryNormSqr(residual_scratch.data());
        }
    } else {
        std::tie(scan_inner, scan_fastscan) = this->get_scan_computers(bucket_computer, bucket_id);
    }
    const float query_centroid_adjustment =
        this->use_l2_residual_query() ? 0.0F
                                      : this->query_centroid_adjustment(bucket_computer, bucket_id);
    const uint64_t block_count =
        (static_cast<uint64_t>(bucket_size) + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    if (scan_fastscan != nullptr and this->fastscan_block_size_ > 0 and
        this->fastscan_blocks_[bucket_id].size() >= block_count * this->fastscan_block_size_) {
        float* scan_filter_inner_products = direct_filter_inner_products;
        constexpr uint64_t kStackComputedMaskCount = 256;
        std::array<uint32_t, kStackComputedMaskCount> stack_computed_masks{};
        Vector<uint32_t> dynamic_computed_masks(this->allocator_);
        uint32_t* computed_masks = stack_computed_masks.data();
        if (block_count > kStackComputedMaskCount) {
            dynamic_computed_masks.resize(block_count, 0);
            computed_masks = dynamic_computed_masks.data();
        }
        if (this->use_l2_residual_query()) {
            this->codes_->QueryFastScan32BatchSharedResidual(
                result_dists,
                computed_masks,
                scan_inner,
                scan_fastscan,
                this->fastscan_blocks_[bucket_id].data(),
                bucket_size,
                query_bucket_norm_sqr,
                scan_filter_inner_products,
                ctx);
        } else {
            this->codes_->QueryFastScan32Batch(result_dists,
                                               computed_masks,
                                               scan_inner,
                                               scan_fastscan,
                                               this->fastscan_blocks_[bucket_id].data(),
                                               bucket_size,
                                               scan_filter_inner_products,
                                               ctx);
        }
        if (this->use_l2_residual_query()) {
            bool all_computed = true;
            for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
                const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
                const auto valid_size =
                    std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
                const uint32_t expected_mask = valid_size == FASTSCAN_BATCH_SIZE
                                                   ? std::numeric_limits<uint32_t>::max()
                                                   : (1U << valid_size) - 1U;
                all_computed &= computed_masks[block_index] == expected_mask;
            }
            if (all_computed) {
                return source_version;
            }
        }
        Vector<InnerIdType> fallback_ids(this->allocator_);
        Vector<InnerIdType> fallback_positions(this->allocator_);
        fallback_ids.reserve(bucket_size);
        fallback_positions.reserve(bucket_size);
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
            const InnerIdType valid_size =
                std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
            for (InnerIdType i = 0; i < valid_size; ++i) {
                const InnerIdType offset = begin + i;
                const auto inner_id = this->inner_ids_[bucket_id][offset];
                if (inner_id == EMPTY_INNER_ID) {
                    result_dists[offset] = std::numeric_limits<float>::max();
                    if (scan_filter_inner_products != nullptr) {
                        scan_filter_inner_products[offset] =
                            std::numeric_limits<float>::quiet_NaN();
                    }
                } else if ((computed_masks[block_index] & (1U << i)) != 0U) {
                    if (not this->use_l2_residual_query()) {
                        result_dists[offset] -=
                            this->residual_adjustment(query_centroid_adjustment, bucket_id, offset);
                    }
                } else {
                    if (scan_filter_inner_products != nullptr) {
                        scan_filter_inner_products[offset] =
                            std::numeric_limits<float>::quiet_NaN();
                    }
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
            if (this->use_l2_residual_query()) {
                scan_inner = this->get_scan_computers(bucket_computer, bucket_id).first;
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
        return source_version;
    }

    if (direct_filter_inner_products != nullptr) {
        std::fill_n(
            direct_filter_inner_products, bucket_size, std::numeric_limits<float>::quiet_NaN());
    }
    Vector<InnerIdType> ids(this->allocator_);
    Vector<InnerIdType> positions(this->allocator_);
    ids.reserve(bucket_size);
    if (this->use_l2_residual_query()) {
        scan_inner = this->get_scan_computers(bucket_computer, bucket_id).first;
    }

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
        return source_version;
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
    return source_version;
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
RaBitQSplitBucketDataCell::QueryWithCandidateFilterInnerProductBySource(
    float* result_dists,
    const float* hint_dists,
    const float* filter_inner_products,
    const BucketIdType* source_bucket_ids,
    const InnerIdType* source_offset_ids,
    const uint64_t* source_versions,
    const ComputerInterfacePtr& computer,
    const InnerIdType* inner_ids,
    InnerIdType id_count,
    QueryContext* ctx) {
    (void)hint_dists;
    CHECK_ARGUMENT(
        id_count == 0 or (filter_inner_products != nullptr and source_bucket_ids != nullptr and
                          source_offset_ids != nullptr and source_versions != nullptr),
        "candidate provenance inputs are required");
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    if (this->use_l2_residual_query()) {
        this->query_residual_by_inner_ids(result_dists,
                                          filter_inner_products,
                                          FilterInnerProductMode::CANDIDATE_SCAN,
                                          bucket_computer,
                                          inner_ids,
                                          id_count,
                                          ctx,
                                          source_bucket_ids,
                                          source_offset_ids,
                                          source_versions);
        return;
    }

    this->query_non_residual_by_inner_ids(result_dists,
                                          filter_inner_products,
                                          FilterInnerProductMode::CANDIDATE_SCAN,
                                          bucket_computer,
                                          inner_ids,
                                          id_count,
                                          ctx,
                                          source_bucket_ids,
                                          source_offset_ids,
                                          source_versions);
}

void
RaBitQSplitBucketDataCell::QueryWithDistanceHintByInnerId(float* result_dists,
                                                          const float* hint_dists,
                                                          const ComputerInterfacePtr& computer,
                                                          const InnerIdType* inner_ids,
                                                          InnerIdType id_count,
                                                          QueryContext* ctx) {
    (void)hint_dists;
    auto& bucket_computer = RaBitQSplitBucketDataCell::get_bucket_computer(computer);
    if (this->use_l2_residual_query()) {
        this->query_residual_by_inner_ids(result_dists,
                                          nullptr,
                                          FilterInnerProductMode::CACHED,
                                          bucket_computer,
                                          inner_ids,
                                          id_count,
                                          ctx);
        return;
    }

    this->query_non_residual_by_inner_ids(result_dists,
                                          nullptr,
                                          FilterInnerProductMode::CACHED,
                                          bucket_computer,
                                          inner_ids,
                                          id_count,
                                          ctx);
}

bool
RaBitQSplitBucketDataCell::query_current_by_inner_id(float& result_dist,
                                                     SplitBucketComputer& computer,
                                                     InnerIdType inner_id,
                                                     QueryContext* ctx) {
    result_dist = std::numeric_limits<float>::max();
    constexpr uint32_t kMaxLocationRetries = 3;
    constexpr uint64_t kLocationMask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;
    for (uint32_t retry = 0; retry < kMaxLocationRetries; ++retry) {
        uint64_t location = INVALID_LOCATION;
        {
            std::lock_guard location_lock(this->locations_mutex_);
            if (inner_id >= this->locations_.size() or
                this->locations_[inner_id] == INVALID_LOCATION) {
                return false;
            }
            location = this->locations_[inner_id];
        }

        const auto bucket_id = static_cast<BucketIdType>(location >> LOCATION_SPLIT_BIT);
        const auto offset_id = static_cast<InnerIdType>(location & kLocationMask);
        if (bucket_id < 0 or bucket_id >= this->bucket_count_) {
            return false;
        }

        std::shared_lock bucket_lock(this->bucket_mutexes_[bucket_id]);
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        {
            // Writers publish locations while holding the corresponding bucket lock and follow the
            // bucket -> codes -> locations order. Taking the same locks before rechecking makes the
            // lane and supplement record a consistent snapshot without inverting locks.
            std::lock_guard location_lock(this->locations_mutex_);
            if (inner_id >= this->locations_.size() or this->locations_[inner_id] != location) {
                continue;
            }
        }
        if (offset_id >= this->inner_ids_[bucket_id].size() or
            this->inner_ids_[bucket_id][offset_id] != inner_id) {
            continue;
        }

        ByteBuffer filter_code(this->codes_->GetFilterCodeSize(), this->allocator_);
        this->get_filter_code(bucket_id, offset_id, filter_code.data);
        if (this->use_l2_residual_query()) {
            Vector<float> residual_query(this->residual_transform_size_, this->allocator_);
            FP32Sub(computer.transformed_query_.data(),
                    this->residual_centroid_transforms_.data() +
                        static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                    residual_query.data(),
                    this->residual_transform_size_);
            ComputerInterfacePtr residual_computer = nullptr;
            this->codes_->ResetComputerFromResidualQuery(residual_query.data(), residual_computer);
            this->codes_->QueryWithFilterCodes(&result_dist,
                                               nullptr,
                                               nullptr,
                                               residual_computer,
                                               &inner_id,
                                               filter_code.data,
                                               1,
                                               ctx);
        } else {
            this->codes_->QueryWithFilterCodes(&result_dist,
                                               nullptr,
                                               nullptr,
                                               computer.inner_,
                                               &inner_id,
                                               filter_code.data,
                                               1,
                                               ctx);
            const float query_centroid_adjustment =
                this->query_centroid_adjustment(computer, bucket_id);
            result_dist -=
                this->residual_adjustment(query_centroid_adjustment, bucket_id, offset_id);
        }
        return true;
    }
    return false;
}

void
RaBitQSplitBucketDataCell::query_non_residual_by_inner_ids(
    float* result_dists,
    const float* filter_inner_products,
    FilterInnerProductMode filter_inner_product_mode,
    SplitBucketComputer& computer,
    const InnerIdType* inner_ids,
    InnerIdType id_count,
    QueryContext* ctx,
    const BucketIdType* source_bucket_ids,
    const InnerIdType* source_offset_ids,
    const uint64_t* source_versions) {
    (void)filter_inner_product_mode;
    struct ReorderEntry {
        BucketIdType bucket_id;
        InnerIdType offset_id;
        InnerIdType inner_id;
        InnerIdType result_offset;
        uint64_t source_version;
    };

    const bool has_source_provenance = source_bucket_ids != nullptr;
    Vector<ReorderEntry> entries(this->allocator_);
    entries.reserve(id_count);
    Vector<InnerIdType> stale_ids(this->allocator_);
    Vector<InnerIdType> stale_result_offsets(this->allocator_);
    stale_ids.reserve(id_count);
    stale_result_offsets.reserve(id_count);
    if (has_source_provenance) {
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (source_bucket_ids[i] < 0 or source_bucket_ids[i] >= this->bucket_count_) {
                stale_ids.push_back(inner_ids[i]);
                stale_result_offsets.push_back(i);
                continue;
            }
            entries.push_back(
                {source_bucket_ids[i], source_offset_ids[i], inner_ids[i], i, source_versions[i]});
        }
    } else {
        std::lock_guard lock(this->locations_mutex_);
        for (InnerIdType i = 0; i < id_count; ++i) {
            CHECK_ARGUMENT(inner_ids[i] < this->locations_.size(),
                           "invalid inner id for RaBitQ split bucket");
            const uint64_t location = this->locations_[inner_ids[i]];
            CHECK_ARGUMENT(location != INVALID_LOCATION,
                           "invalid inner id for RaBitQ split bucket");
            entries.push_back({static_cast<BucketIdType>(location >> LOCATION_SPLIT_BIT),
                               static_cast<InnerIdType>(location & 0xFFFFFFFFULL),
                               inner_ids[i],
                               i,
                               INVALID_CACHE_VERSION});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bucket_id < rhs.bucket_id;
    });

    const auto entry_count = static_cast<InnerIdType>(entries.size());
    Vector<InnerIdType> sorted_ids(entry_count, this->allocator_);
    Vector<float> sorted_dists(entry_count, this->allocator_);
    Vector<float> sorted_filter_inner_products(
        entry_count, std::numeric_limits<float>::quiet_NaN(), this->allocator_);
    Vector<float> adjustments(entry_count, 0.0F, this->allocator_);
    Vector<uint8_t> source_stale(entry_count, 0, this->allocator_);
    for (InnerIdType i = 0; i < entry_count; ++i) {
        sorted_ids[i] = entries[i].inner_id;
        if (filter_inner_products != nullptr) {
            sorted_filter_inner_products[i] = filter_inner_products[entries[i].result_offset];
        }
    }

    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    Vector<InnerIdType> reusable_ids(entry_count, this->allocator_);
    Vector<InnerIdType> reusable_positions(entry_count, this->allocator_);
    Vector<float> reusable_inner_products(entry_count, this->allocator_);
    Vector<float> reusable_dists(entry_count, this->allocator_);
    Vector<uint8_t> reusable_computed(entry_count, 0, this->allocator_);
    Vector<InnerIdType> fallback_ids(entry_count, this->allocator_);
    Vector<InnerIdType> fallback_positions(entry_count, this->allocator_);
    Vector<float> fallback_dists(entry_count, this->allocator_);
    Vector<uint8_t> fallback_filter_codes(static_cast<uint64_t>(entry_count) * filter_code_size,
                                          this->allocator_);

    InnerIdType group_begin = 0;
    while (group_begin < entry_count) {
        const BucketIdType bucket_id = entries[group_begin].bucket_id;
        InnerIdType group_end = group_begin + 1;
        while (group_end < entry_count and entries[group_end].bucket_id == bucket_id) {
            ++group_end;
        }

        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        const bool direct_filter_inner_products_valid = has_source_provenance;
        const float query_centroid_adjustment =
            this->query_centroid_adjustment(computer, bucket_id);
        InnerIdType reusable_count = 0;
        InnerIdType fallback_count = 0;
        for (InnerIdType i = group_begin; i < group_end; ++i) {
            bool entry_is_current =
                entries[i].offset_id < this->inner_ids_[bucket_id].size() and
                this->inner_ids_[bucket_id][entries[i].offset_id] == entries[i].inner_id;
            if (entry_is_current and has_source_provenance) {
                entry_is_current = entries[i].source_version == this->bucket_versions_[bucket_id];
            } else if (entry_is_current) {
                const uint64_t expected_location =
                    RaBitQSplitBucketDataCell::pack_location(bucket_id, entries[i].offset_id);
                std::lock_guard location_lock(this->locations_mutex_);
                entry_is_current = entries[i].inner_id < this->locations_.size() and
                                   this->locations_[entries[i].inner_id] == expected_location;
            }
            if (not entry_is_current) {
                source_stale[i] = 1U;
                stale_ids.push_back(entries[i].inner_id);
                stale_result_offsets.push_back(entries[i].result_offset);
                continue;
            }
            adjustments[i] = this->residual_adjustment(
                query_centroid_adjustment, bucket_id, entries[i].offset_id);

            float reusable_inner_product = 0.0F;
            bool reusable_inner_product_valid = false;
            if (filter_inner_products != nullptr) {
                if (direct_filter_inner_products_valid) {
                    reusable_inner_product = sorted_filter_inner_products[i];
                    reusable_inner_product_valid = IsFiniteFloat(reusable_inner_product);
                }
            }
            if (reusable_inner_product_valid) {
                reusable_ids[reusable_count] = sorted_ids[i];
                reusable_positions[reusable_count] = i;
                reusable_inner_products[reusable_count] = reusable_inner_product;
                ++reusable_count;
            } else {
                fallback_ids[fallback_count] = sorted_ids[i];
                fallback_positions[fallback_count] = i;
                ++fallback_count;
            }
        }

        if (reusable_count != 0) {
            this->codes_->QueryWithFilterInnerProducts(reusable_dists.data(),
                                                       reusable_computed.data(),
                                                       reusable_inner_products.data(),
                                                       computer.inner_,
                                                       reusable_ids.data(),
                                                       reusable_count,
                                                       ctx);
            for (InnerIdType i = 0; i < reusable_count; ++i) {
                const InnerIdType position = reusable_positions[i];
                if (reusable_computed[i] != 0U) {
                    sorted_dists[position] = reusable_dists[i];
                    continue;
                }
                fallback_ids[fallback_count] = reusable_ids[i];
                fallback_positions[fallback_count] = position;
                ++fallback_count;
            }
        }

        for (InnerIdType i = 0; i < fallback_count; ++i) {
            const InnerIdType position = fallback_positions[i];
            this->get_filter_code(
                bucket_id,
                entries[position].offset_id,
                fallback_filter_codes.data() + static_cast<uint64_t>(i) * filter_code_size);
        }
        if (fallback_count != 0) {
            this->codes_->QueryWithFilterCodes(fallback_dists.data(),
                                               nullptr,
                                               nullptr,
                                               computer.inner_,
                                               fallback_ids.data(),
                                               fallback_filter_codes.data(),
                                               fallback_count,
                                               ctx);
            for (InnerIdType i = 0; i < fallback_count; ++i) {
                sorted_dists[fallback_positions[i]] = fallback_dists[i];
            }
        }
        group_begin = group_end;
    }

    for (InnerIdType i = 0; i < entry_count; ++i) {
        if (source_stale[i] == 0U) {
            result_dists[entries[i].result_offset] = sorted_dists[i] - adjustments[i];
        }
    }
    for (uint64_t i = 0; i < stale_ids.size(); ++i) {
        float stale_dist = std::numeric_limits<float>::max();
        this->query_current_by_inner_id(stale_dist, computer, stale_ids[i], ctx);
        result_dists[stale_result_offsets[i]] = stale_dist;
    }
}

void
RaBitQSplitBucketDataCell::query_residual_by_inner_ids(
    float* result_dists,
    const float* filter_inner_products,
    FilterInnerProductMode filter_inner_product_mode,
    SplitBucketComputer& computer,
    const InnerIdType* inner_ids,
    InnerIdType id_count,
    QueryContext* ctx,
    const BucketIdType* source_bucket_ids,
    const InnerIdType* source_offset_ids,
    const uint64_t* source_versions) {
    struct ReorderEntry {
        BucketIdType bucket_id;
        InnerIdType offset_id;
        InnerIdType inner_id;
        InnerIdType result_offset;
        uint64_t source_version;
    };

    const bool has_source_provenance = source_bucket_ids != nullptr;
    Vector<ReorderEntry> entries(this->allocator_);
    entries.reserve(id_count);
    Vector<InnerIdType> stale_ids(this->allocator_);
    Vector<InnerIdType> stale_result_offsets(this->allocator_);
    if (has_source_provenance) {
        for (InnerIdType i = 0; i < id_count; ++i) {
            if (source_bucket_ids[i] < 0 or source_bucket_ids[i] >= this->bucket_count_) {
                stale_ids.push_back(inner_ids[i]);
                stale_result_offsets.push_back(i);
                continue;
            }
            entries.push_back(
                {source_bucket_ids[i], source_offset_ids[i], inner_ids[i], i, source_versions[i]});
        }
    } else {
        std::lock_guard lock(this->locations_mutex_);
        for (InnerIdType i = 0; i < id_count; ++i) {
            CHECK_ARGUMENT(inner_ids[i] < this->locations_.size(),
                           "invalid inner id for RaBitQ split bucket");
            const uint64_t location = this->locations_[inner_ids[i]];
            CHECK_ARGUMENT(location != INVALID_LOCATION,
                           "invalid inner id for RaBitQ split bucket");
            entries.push_back({static_cast<BucketIdType>(location >> LOCATION_SPLIT_BIT),
                               static_cast<InnerIdType>(location & 0xFFFFFFFFULL),
                               inner_ids[i],
                               i,
                               INVALID_CACHE_VERSION});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.bucket_id < rhs.bucket_id;
    });

    const auto entry_count = static_cast<InnerIdType>(entries.size());
    Vector<InnerIdType> sorted_ids(entry_count, this->allocator_);
    Vector<float> sorted_dists(entry_count, this->allocator_);
    Vector<float> sorted_filter_inner_products(
        entry_count, std::numeric_limits<float>::quiet_NaN(), this->allocator_);
    Vector<uint8_t> source_stale(entry_count, 0, this->allocator_);
    for (InnerIdType i = 0; i < entry_count; ++i) {
        sorted_ids[i] = entries[i].inner_id;
        if (filter_inner_products != nullptr) {
            sorted_filter_inner_products[i] = filter_inner_products[entries[i].result_offset];
        }
    }

    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    RaBitQSplitResidualOriginalQueryInterface* residual_codes =
        filter_inner_product_mode == FilterInnerProductMode::CANDIDATE_SCAN
            ? this->residual_original_query_codes_
            : nullptr;
    Vector<InnerIdType> reusable_ids(entry_count, this->allocator_);
    Vector<InnerIdType> reusable_positions(entry_count, this->allocator_);
    Vector<float> reusable_inner_products(entry_count, this->allocator_);
    Vector<float> reusable_full_adds(entry_count, this->allocator_);
    Vector<float> reusable_dists(entry_count, this->allocator_);
    Vector<uint8_t> reusable_computed(entry_count, 0, this->allocator_);
    Vector<InnerIdType> fallback_ids(this->allocator_);
    Vector<InnerIdType> fallback_positions(this->allocator_);
    Vector<float> fallback_dists(this->allocator_);
    Vector<uint8_t> fallback_filter_codes(this->allocator_);

    Vector<float> candidate_residual_query(this->allocator_);
    ComputerInterfacePtr candidate_reorder_computer = nullptr;
    InnerIdType group_begin = 0;
    while (group_begin < entry_count) {
        const BucketIdType bucket_id = entries[group_begin].bucket_id;
        InnerIdType group_end = group_begin + 1;
        while (group_end < entry_count and entries[group_end].bucket_id == bucket_id) {
            ++group_end;
        }

        std::shared_lock lock(this->bucket_mutexes_[bucket_id]);
        const bool use_original_query = residual_codes != nullptr;
        ComputerInterfacePtr scan_inner = use_original_query
                                              ? computer.inner_
                                              : this->get_scan_computers(computer, bucket_id).first;
        bool candidate_residual_query_ready = false;
        auto prepare_candidate_residual_query = [&]() {
            if (candidate_residual_query_ready) {
                return;
            }
            candidate_residual_query.resize(this->residual_transform_size_);
            FP32Sub(computer.transformed_query_.data(),
                    this->residual_centroid_transforms_.data() +
                        static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                    candidate_residual_query.data(),
                    this->residual_transform_size_);
            candidate_residual_query_ready = true;
        };
        float query_bucket_norm_sqr = 0.0F;
        if (use_original_query) {
            const uint64_t routed_bucket_index = this->get_routed_bucket_index(computer, bucket_id);
            if (routed_bucket_index != INVALID_ROUTED_BUCKET_INDEX) {
                query_bucket_norm_sqr = computer.routed_bucket_norm_sqrs_[routed_bucket_index];
            } else {
                prepare_candidate_residual_query();
                query_bucket_norm_sqr = this->codes_->ComputeTransformedResidualQueryNormSqr(
                    candidate_residual_query.data());
            }
        }
        const bool direct_filter_inner_products_valid = has_source_provenance;
        InnerIdType reusable_count = 0;
        fallback_ids.clear();
        fallback_positions.clear();
        auto convert_shared_filter_inner_product = [&](InnerIdType position,
                                                       float shared_filter_inner_product) {
            if (not IsFiniteFloat(shared_filter_inner_product)) {
                return false;
            }
            const uint64_t block_index =
                static_cast<uint64_t>(entries[position].offset_id) / FASTSCAN_BATCH_SIZE;
            const auto index_in_block = static_cast<InnerIdType>(
                static_cast<uint64_t>(entries[position].offset_id) % FASTSCAN_BATCH_SIZE);
            const uint8_t* block =
                this->fastscan_blocks_[bucket_id].data() + block_index * this->fastscan_block_size_;
            return residual_codes->RecoverFastScan32OriginalQueryFilterInnerProduct(
                       computer.inner_,
                       block,
                       index_in_block,
                       shared_filter_inner_product,
                       sorted_filter_inner_products.data() + position) and
                   IsFiniteFloat(sorted_filter_inner_products[position]);
        };
        for (InnerIdType i = group_begin; i < group_end; ++i) {
            bool entry_is_current =
                entries[i].offset_id < this->inner_ids_[bucket_id].size() and
                this->inner_ids_[bucket_id][entries[i].offset_id] == entries[i].inner_id;
            if (entry_is_current and has_source_provenance) {
                entry_is_current = entries[i].source_version == this->bucket_versions_[bucket_id];
            } else if (entry_is_current) {
                const uint64_t expected_location =
                    RaBitQSplitBucketDataCell::pack_location(bucket_id, entries[i].offset_id);
                std::lock_guard location_lock(this->locations_mutex_);
                entry_is_current = entries[i].inner_id < this->locations_.size() and
                                   this->locations_[entries[i].inner_id] == expected_location;
            }
            if (not entry_is_current) {
                source_stale[i] = 1U;
                stale_ids.push_back(entries[i].inner_id);
                stale_result_offsets.push_back(entries[i].result_offset);
                continue;
            }
            bool reusable_inner_product_valid = false;
            if (filter_inner_products != nullptr) {
                if (direct_filter_inner_products_valid and
                    filter_inner_product_mode == FilterInnerProductMode::CANDIDATE_SCAN) {
                    const float shared_filter_inner_product = sorted_filter_inner_products[i];
                    reusable_inner_product_valid =
                        convert_shared_filter_inner_product(i, shared_filter_inner_product);
                }
            }

            float full_add = std::numeric_limits<float>::quiet_NaN();
            if (reusable_inner_product_valid and use_original_query) {
                if (entries[i].offset_id < this->residual_bias_[bucket_id].size()) {
                    full_add = this->residual_bias_[bucket_id][entries[i].offset_id];
                }
                reusable_inner_product_valid = IsFiniteFloat(full_add);
            }
            if (reusable_inner_product_valid) {
                reusable_ids[reusable_count] = sorted_ids[i];
                reusable_positions[reusable_count] = i;
                reusable_inner_products[reusable_count] = sorted_filter_inner_products[i];
                if (use_original_query) {
                    reusable_full_adds[reusable_count] = full_add;
                }
                ++reusable_count;
            } else {
                fallback_ids.push_back(sorted_ids[i]);
                fallback_positions.push_back(i);
            }
        }

        if (reusable_count != 0) {
            if (use_original_query) {
                residual_codes->QueryWithOriginalQueryFilterInnerProducts(
                    reusable_dists.data(),
                    reusable_computed.data(),
                    reusable_inner_products.data(),
                    reusable_full_adds.data(),
                    query_bucket_norm_sqr,
                    computer.inner_,
                    reusable_ids.data(),
                    reusable_count,
                    ctx);
            } else {
                this->codes_->QueryWithFilterInnerProducts(reusable_dists.data(),
                                                           reusable_computed.data(),
                                                           reusable_inner_products.data(),
                                                           scan_inner,
                                                           reusable_ids.data(),
                                                           reusable_count,
                                                           ctx);
            }
            for (InnerIdType i = 0; i < reusable_count; ++i) {
                const InnerIdType position = reusable_positions[i];
                if (reusable_computed[i] != 0U) {
                    sorted_dists[position] = reusable_dists[i];
                    continue;
                }
                fallback_ids.push_back(reusable_ids[i]);
                fallback_positions.push_back(position);
            }
        }

        const auto fallback_count = static_cast<InnerIdType>(fallback_ids.size());
        if (fallback_count != 0) {
            fallback_dists.resize(fallback_count);
            fallback_filter_codes.resize(static_cast<uint64_t>(fallback_count) * filter_code_size);
        }
        for (InnerIdType i = 0; i < fallback_count; ++i) {
            const InnerIdType position = fallback_positions[i];
            this->get_filter_code(
                bucket_id,
                entries[position].offset_id,
                fallback_filter_codes.data() + static_cast<uint64_t>(i) * filter_code_size);
        }
        if (fallback_count != 0) {
            if (use_original_query) {
                prepare_candidate_residual_query();
                this->codes_->ResetComputerFromResidualQuery(candidate_residual_query.data(),
                                                             candidate_reorder_computer);
                scan_inner = candidate_reorder_computer;
            }
            this->codes_->QueryWithFilterCodes(fallback_dists.data(),
                                               nullptr,
                                               nullptr,
                                               scan_inner,
                                               fallback_ids.data(),
                                               fallback_filter_codes.data(),
                                               fallback_count,
                                               ctx);
            for (InnerIdType i = 0; i < fallback_count; ++i) {
                sorted_dists[fallback_positions[i]] = fallback_dists[i];
            }
        }
        group_begin = group_end;
    }

    for (InnerIdType i = 0; i < entry_count; ++i) {
        if (source_stale[i] == 0U) {
            result_dists[entries[i].result_offset] = sorted_dists[i];
        }
    }
    for (uint64_t i = 0; i < stale_ids.size(); ++i) {
        float stale_dist = std::numeric_limits<float>::max();
        this->query_current_by_inner_id(stale_dist, computer, stale_ids[i], ctx);
        result_dists[stale_result_offsets[i]] = stale_dist;
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
    const bool use_l2_residual_query = this->use_l2_residual_query();
    Vector<float> transformed_query(use_l2_residual_query ? this->residual_transform_size_ : 0,
                                    this->allocator_);
    ComputerInterfacePtr inner = nullptr;
    ComputerInterfacePtr fastscan = nullptr;
    if (use_l2_residual_query) {
        this->codes_->TransformResidualQuery(input, transformed_query.data());
        inner = this->codes_->FactoryComputerFromResidualQuery(transformed_query.data());
        fastscan = this->codes_->FactoryFastScan32Computer(inner);
    } else {
        inner = this->codes_->FactoryComputer(input);
        fastscan = this->codes_->FactoryFastScan32Computer(inner);
    }
    const uint64_t adjustment_count = this->use_residual_ and not use_l2_residual_query
                                          ? static_cast<uint64_t>(this->bucket_count_)
                                          : 0;
    const uint64_t bucket_computer_capacity = bucket_ids == nullptr ? 0 : bucket_count;
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
        std::copy(transformed_query.begin(),
                  transformed_query.end(),
                  computer->transformed_query_.begin());
    }
    if (bucket_ids != nullptr or bucket_count != 0) {
        this->prepare_scan_computers(*computer, bucket_ids, bucket_count);
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
    if (this->optimized_build_codes_ != nullptr) {
        // BeginOptimizedBuild pre-sizes scalar storage, and IVF assigns disjoint inner IDs.
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        const auto offset_id = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
        this->codes_->InsertVector(input, inner_id);
        this->inner_ids_[bucket_id].push_back(inner_id);
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id].push_back(bias);
        }
        {
            std::lock_guard location_lock(this->locations_mutex_);
            if (this->locations_.size() <= inner_id) {
                this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
            }
            this->locations_[inner_id] =
                RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
        }
        ++this->bucket_versions_[bucket_id];
        return offset_id;
    }

    RaBitQSplitResidualOriginalQueryInterface* residual_codes = nullptr;
    const float* transformed_centroid = nullptr;
    if (this->use_l2_residual_query()) {
        this->ensure_residual_centroid_transforms();
        residual_codes = this->residual_original_query_codes_;
        transformed_centroid = this->residual_centroid_transforms_.data() +
                               static_cast<uint64_t>(bucket_id) * this->residual_transform_size_;
    }
    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    ByteBuffer filter_code(filter_code_size, this->allocator_);
    constexpr uint64_t location_mask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;

    auto write_target = [&](InnerIdType offset_id) {
        if (residual_codes != nullptr) {
            residual_codes->InsertVectorWithFilterCodeAndResidualFullFactor(
                input, inner_id, transformed_centroid, filter_code.data, &bias);
        } else {
            this->codes_->InsertVectorWithFilterCode(input, inner_id, filter_code.data);
        }
        this->set_filter_code(this->fastscan_blocks_[bucket_id], offset_id, filter_code.data);
        this->set_residual_filter_code(bucket_id, offset_id, filter_code.data);
        this->inner_ids_[bucket_id].push_back(inner_id);
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id].push_back(bias);
        }
    };
    auto invalidate_old_lane = [&](BucketIdType old_bucket_id, InnerIdType old_offset_id) {
        std::memset(filter_code.data, 0, filter_code_size);
        this->set_filter_code(
            this->fastscan_blocks_[old_bucket_id], old_offset_id, filter_code.data);
        this->set_residual_filter_code(old_bucket_id, old_offset_id, filter_code.data);
        this->inner_ids_[old_bucket_id][old_offset_id] = EMPTY_INNER_ID;
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[old_bucket_id][old_offset_id] = 0.0F;
        }
    };

    // The common append path retains one bucket/codes/locations critical section. Existing IDs are
    // detected before their global supplement code is overwritten.
    {
        std::unique_lock bucket_lock(this->bucket_mutexes_[bucket_id]);
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id) {
            this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
        }
        const uint64_t old_location = this->locations_[inner_id];
        const auto old_bucket_id = static_cast<BucketIdType>(old_location >> LOCATION_SPLIT_BIT);
        if (old_location == INVALID_LOCATION or old_bucket_id == bucket_id) {
            const auto offset_id = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
            const uint64_t target_location =
                RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
            write_target(offset_id);
            if (old_location != INVALID_LOCATION) {
                const auto old_offset_id = static_cast<InnerIdType>(old_location & location_mask);
                CHECK_ARGUMENT(old_offset_id < offset_id and
                                   this->inner_ids_[bucket_id][old_offset_id] == inner_id,
                               "stale RaBitQ split bucket location during relocation");
                invalidate_old_lane(bucket_id, old_offset_id);
            }
            ++this->bucket_versions_[bucket_id];
            this->locations_[inner_id] = target_location;
            return offset_id;
        }
    }

    while (true) {
        uint64_t old_location = INVALID_LOCATION;
        {
            std::lock_guard location_lock(this->locations_mutex_);
            if (this->locations_.size() > inner_id) {
                old_location = this->locations_[inner_id];
            }
        }
        if (old_location == INVALID_LOCATION) {
            // A concurrent fixed-offset overwrite displaced this ID. Restart as a regular append.
            return this->InsertVector(vector, bucket_id, inner_id);
        }

        const auto old_bucket_id = static_cast<BucketIdType>(old_location >> LOCATION_SPLIT_BIT);
        const auto old_offset_id = static_cast<InnerIdType>(old_location & location_mask);
        this->check_valid_bucket_id(old_bucket_id);

        const auto first_bucket_id = std::min(old_bucket_id, bucket_id);
        const auto second_bucket_id = std::max(old_bucket_id, bucket_id);
        std::unique_lock first_bucket_lock(this->bucket_mutexes_[first_bucket_id]);
        std::unique_lock<std::shared_mutex> second_bucket_lock;
        if (first_bucket_id != second_bucket_id) {
            second_bucket_lock = std::unique_lock(this->bucket_mutexes_[second_bucket_id]);
        }
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id or this->locations_[inner_id] != old_location) {
            continue;
        }

        CHECK_ARGUMENT(old_offset_id < this->inner_ids_[old_bucket_id].size() and
                           this->inner_ids_[old_bucket_id][old_offset_id] == inner_id,
                       "stale RaBitQ split bucket location during relocation");
        const auto offset_id = static_cast<InnerIdType>(this->inner_ids_[bucket_id].size());
        const uint64_t target_location =
            RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
        write_target(offset_id);
        invalidate_old_lane(old_bucket_id, old_offset_id);
        ++this->bucket_versions_[bucket_id];
        if (old_bucket_id != bucket_id) {
            ++this->bucket_versions_[old_bucket_id];
        }
        this->locations_[inner_id] = target_location;
        return offset_id;
    }
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
    if (this->optimized_build_codes_ != nullptr) {
        // BeginOptimizedBuild pre-sizes scalar storage, and IVF assigns disjoint inner IDs.
        std::unique_lock lock(this->bucket_mutexes_[bucket_id]);
        this->codes_->InsertVector(input, inner_id);
        if (this->inner_ids_[bucket_id].size() <= offset_id) {
            this->inner_ids_[bucket_id].resize(static_cast<uint64_t>(offset_id) + 1,
                                               EMPTY_INNER_ID);
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
            this->locations_[inner_id] =
                RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
        }
        ++this->bucket_versions_[bucket_id];
        return;
    }

    RaBitQSplitResidualOriginalQueryInterface* residual_codes = nullptr;
    const float* transformed_centroid = nullptr;
    if (this->use_l2_residual_query()) {
        this->ensure_residual_centroid_transforms();
        residual_codes = this->residual_original_query_codes_;
        transformed_centroid = this->residual_centroid_transforms_.data() +
                               static_cast<uint64_t>(bucket_id) * this->residual_transform_size_;
    }
    const uint64_t target_location = RaBitQSplitBucketDataCell::pack_location(bucket_id, offset_id);
    const uint64_t filter_code_size = this->codes_->GetFilterCodeSize();
    ByteBuffer filter_code(filter_code_size, this->allocator_);
    constexpr uint64_t location_mask = (1ULL << LOCATION_SPLIT_BIT) - 1ULL;

    auto resize_target = [&]() {
        if (this->inner_ids_[bucket_id].size() <= offset_id) {
            this->inner_ids_[bucket_id].resize(static_cast<uint64_t>(offset_id) + 1,
                                               EMPTY_INNER_ID);
            if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
                this->residual_bias_[bucket_id].resize(static_cast<uint64_t>(offset_id) + 1, 0.0F);
            }
        }
    };
    auto invalidate_target_occupant = [&]() {
        const auto displaced_id = this->inner_ids_[bucket_id][offset_id];
        if (displaced_id != EMPTY_INNER_ID and displaced_id != inner_id and
            displaced_id < this->locations_.size() and
            this->locations_[displaced_id] == target_location) {
            this->locations_[displaced_id] = INVALID_LOCATION;
        }
    };
    auto write_target = [&]() {
        if (residual_codes != nullptr) {
            residual_codes->InsertVectorWithFilterCodeAndResidualFullFactor(
                input, inner_id, transformed_centroid, filter_code.data, &bias);
        } else {
            this->codes_->InsertVectorWithFilterCode(input, inner_id, filter_code.data);
        }
        this->set_filter_code(this->fastscan_blocks_[bucket_id], offset_id, filter_code.data);
        this->set_residual_filter_code(bucket_id, offset_id, filter_code.data);
        this->inner_ids_[bucket_id][offset_id] = inner_id;
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[bucket_id][offset_id] = bias;
        }
    };
    auto invalidate_old_lane = [&](BucketIdType old_bucket_id, InnerIdType old_offset_id) {
        std::memset(filter_code.data, 0, filter_code_size);
        this->set_filter_code(
            this->fastscan_blocks_[old_bucket_id], old_offset_id, filter_code.data);
        this->set_residual_filter_code(old_bucket_id, old_offset_id, filter_code.data);
        this->inner_ids_[old_bucket_id][old_offset_id] = EMPTY_INNER_ID;
        if (this->use_residual_ and this->metric_ == MetricType::METRIC_TYPE_L2SQR) {
            this->residual_bias_[old_bucket_id][old_offset_id] = 0.0F;
        }
    };

    // Keep the unique-ID path on the existing bucket -> codes -> locations lock order. Cross-bucket
    // relocation is detected before mutating the global supplement code and retried below with both
    // buckets exclusively locked.
    {
        std::unique_lock bucket_lock(this->bucket_mutexes_[bucket_id]);
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id) {
            this->locations_.resize(static_cast<uint64_t>(inner_id) + 1, INVALID_LOCATION);
        }
        const uint64_t old_location = this->locations_[inner_id];
        const auto old_bucket_id = static_cast<BucketIdType>(old_location >> LOCATION_SPLIT_BIT);
        if (old_location == INVALID_LOCATION or old_bucket_id == bucket_id) {
            resize_target();
            invalidate_target_occupant();
            write_target();
            if (old_location != INVALID_LOCATION and old_location != target_location) {
                const auto old_offset_id = static_cast<InnerIdType>(old_location & location_mask);
                CHECK_ARGUMENT(old_offset_id < this->inner_ids_[bucket_id].size() and
                                   this->inner_ids_[bucket_id][old_offset_id] == inner_id,
                               "stale RaBitQ split bucket location during relocation");
                invalidate_old_lane(bucket_id, old_offset_id);
            }
            ++this->bucket_versions_[bucket_id];
            this->locations_[inner_id] = target_location;
            return;
        }
    }

    while (true) {
        uint64_t old_location = INVALID_LOCATION;
        {
            std::lock_guard location_lock(this->locations_mutex_);
            if (this->locations_.size() > inner_id) {
                old_location = this->locations_[inner_id];
            }
        }
        if (old_location == INVALID_LOCATION) {
            // A concurrent relocation displaced this ID. Restart through the regular path, which
            // publishes a new lane without trying to invalidate a location that no longer exists.
            this->InsertVectorWithOffset(vector, bucket_id, inner_id, offset_id);
            return;
        }

        const auto old_bucket_id = static_cast<BucketIdType>(old_location >> LOCATION_SPLIT_BIT);
        const auto old_offset_id = static_cast<InnerIdType>(old_location & location_mask);
        this->check_valid_bucket_id(old_bucket_id);

        const auto first_bucket_id = std::min(old_bucket_id, bucket_id);
        const auto second_bucket_id = std::max(old_bucket_id, bucket_id);
        std::unique_lock first_bucket_lock(this->bucket_mutexes_[first_bucket_id]);
        std::unique_lock<std::shared_mutex> second_bucket_lock;
        if (first_bucket_id != second_bucket_id) {
            second_bucket_lock = std::unique_lock(this->bucket_mutexes_[second_bucket_id]);
        }
        std::lock_guard codes_lock(this->codes_insert_mutex_);
        std::lock_guard location_lock(this->locations_mutex_);
        if (this->locations_.size() <= inner_id or this->locations_[inner_id] != old_location) {
            continue;
        }

        CHECK_ARGUMENT(old_offset_id < this->inner_ids_[old_bucket_id].size() and
                           this->inner_ids_[old_bucket_id][old_offset_id] == inner_id,
                       "stale RaBitQ split bucket location during relocation");
        resize_target();
        invalidate_target_occupant();
        write_target();
        if (old_location != target_location) {
            invalidate_old_lane(old_bucket_id, old_offset_id);
        }
        ++this->bucket_versions_[bucket_id];
        if (old_bucket_id != bucket_id) {
            ++this->bucket_versions_[old_bucket_id];
        }
        // The new location is the publication point: both packed lanes and both source versions
        // are already consistent while all affected bucket locks are still held.
        this->locations_[inner_id] = target_location;
        return;
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
    if (this->use_l2_residual_query()) {
        this->ensure_residual_centroid_transforms();
    }
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
            if (this->use_l2_residual_query()) {
                this->residual_bias_[bucket_id].resize(this->inner_ids_[bucket_id].size(),
                                                       std::numeric_limits<float>::quiet_NaN());
            } else {
                this->residual_bias_[bucket_id].insert(this->residual_bias_[bucket_id].end(),
                                                       source->residual_bias_[bucket_id].begin(),
                                                       source->residual_bias_[bucket_id].end());
            }
        }
        this->fastscan_blocks_[bucket_id] = std::move(merged_blocks);
        if (this->use_l2_residual_query()) {
            auto& residual_codes = *this->residual_original_query_codes_;
            const float* transformed_centroid =
                this->residual_centroid_transforms_.data() +
                static_cast<uint64_t>(bucket_id) * this->residual_transform_size_;
            for (InnerIdType offset = 0; offset < this->inner_ids_[bucket_id].size(); ++offset) {
                if (this->inner_ids_[bucket_id][offset] == EMPTY_INNER_ID) {
                    continue;
                }
                this->get_filter_code(bucket_id, offset, filter_code.data);
                this->set_residual_filter_code(bucket_id, offset, filter_code.data);
                float full_add = std::numeric_limits<float>::quiet_NaN();
                residual_codes.ComputeResidualFullFactorForId(filter_code.data,
                                                              this->inner_ids_[bucket_id][offset],
                                                              transformed_centroid,
                                                              &full_add);
                this->residual_bias_[bucket_id][offset] = full_add;
            }
        }
        ++this->bucket_versions_[bucket_id];
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

    RaBitQSplitResidualOriginalQueryInterface* residual_codes = nullptr;
    if (this->use_l2_residual_query()) {
        this->ensure_residual_centroid_transforms();
        residual_codes = this->residual_original_query_codes_;
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
        if (residual_codes != nullptr) {
            this->residual_bias_[bucket_id].resize(bucket_size,
                                                   std::numeric_limits<float>::quiet_NaN());
        }
        for (uint64_t block_index = 0; block_index < block_count; ++block_index) {
            const auto begin = static_cast<InnerIdType>(block_index * FASTSCAN_BATCH_SIZE);
            const InnerIdType valid_size =
                std::min<InnerIdType>(FASTSCAN_BATCH_SIZE, bucket_size - begin);
            auto* block = blocks.data() + block_index * this->fastscan_block_size_;
            if (this->use_l2_residual_query()) {
                residual_codes->PackageFastScan32ResidualWithFullFactors(
                    this->inner_ids_[bucket_id].data() + begin,
                    valid_size,
                    this->residual_centroid_transforms_.data() +
                        static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
                    block,
                    this->residual_bias_[bucket_id].data() + begin);
            } else {
                this->codes_->PackageFastScan32(
                    this->inner_ids_[bucket_id].data() + begin, valid_size, block);
            }
        }
        ++this->bucket_versions_[bucket_id];
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
RaBitQSplitBucketDataCell::set_residual_filter_code(BucketIdType bucket_id,
                                                    InnerIdType offset_id,
                                                    const uint8_t* filter_code) {
    if (not this->use_l2_residual_query()) {
        return;
    }
    this->ensure_residual_centroid_transforms();
    const uint64_t block_index = static_cast<uint64_t>(offset_id) / FASTSCAN_BATCH_SIZE;
    const auto index_in_block =
        static_cast<InnerIdType>(static_cast<uint64_t>(offset_id) % FASTSCAN_BATCH_SIZE);
    auto& blocks = this->fastscan_blocks_[bucket_id];
    CHECK_ARGUMENT((block_index + 1) * this->fastscan_block_size_ <= blocks.size(),
                   "missing packed RaBitQ filter block");
    this->codes_->SetFastScan32ResidualCode(
        filter_code,
        this->residual_centroid_transforms_.data() +
            static_cast<uint64_t>(bucket_id) * this->residual_transform_size_,
        index_in_block,
        blocks.data() + block_index * this->fastscan_block_size_);
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

bool
RaBitQSplitBucketDataCell::packed_filter_codes_complete(BucketIdType bucket_id) const {
    const uint64_t bucket_size = this->inner_ids_[bucket_id].size();
    const uint64_t block_count = (bucket_size + FASTSCAN_BATCH_SIZE - 1) / FASTSCAN_BATCH_SIZE;
    return this->fastscan_blocks_[bucket_id].size() == block_count * this->fastscan_block_size_;
}

void
RaBitQSplitBucketDataCell::serialize_packed_filter_codes(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, PACKED_FILTER_STORAGE_MAGIC);
    const uint32_t version = this->use_l2_residual_query()
                                 ? PACKED_RESIDUAL_FULL_FACTOR_STORAGE_VERSION
                                 : PACKED_FILTER_STORAGE_VERSION;
    StreamWriter::WriteObj(writer, version);
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
    const uint32_t expected_version = this->use_l2_residual_query()
                                          ? PACKED_RESIDUAL_FULL_FACTOR_STORAGE_VERSION
                                          : PACKED_FILTER_STORAGE_VERSION;
    CHECK_ARGUMENT(version == expected_version,
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
