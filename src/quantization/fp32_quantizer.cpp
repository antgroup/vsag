
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

#include "fp32_quantizer.h"

#include <cstring>

#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/simd.h"
#include "typing.h"

namespace vsag {

namespace {
bool
cosine_exceptional(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // IEEE classification remains valid under -ffast-math.
    return (bits & 0x7fffffffU) == 0U or (bits & 0x7f800000U) == 0x7f800000U;
}

bool
cosine_non_finite(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) == 0x7f800000U;
}

bool
cosine_zero(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7fffffffU) == 0U;
}

float
normalized_code_similarity(const float* codes,
                           const float* normalized_query,
                           uint64_t dim,
                           Allocator* allocator) {
    Vector<float> normalized(dim, allocator);
    Normalize(codes, normalized.data(), dim);
    return FP32ComputeIP(normalized.data(), normalized_query, dim);
}
}  // namespace

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(int dim, Allocator* allocator)
    : Quantizer<FP32Quantizer<metric>>(dim, allocator) {
    this->code_size_ = dim * sizeof(float);
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const FP32QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP32Quantizer<metric>(common_param.dim_, common_param.allocator_.get()) {
    this->hold_molds_ = param->hold_molds;
    if (metric == MetricType::METRIC_TYPE_COSINE && this->hold_molds_) {
        this->code_size_ += sizeof(float);
    }
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP32Quantizer<metric>(std::dynamic_pointer_cast<FP32QuantizerParameter>(param),
                            common_param) {
}

template <MetricType metric>
bool
FP32Quantizer<metric>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) {
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        if (this->hold_molds_) {
            // Store the mold for cosine similarity
            const auto* data_float = reinterpret_cast<const float*>(data);
            float mold = std::sqrt(FP32ComputeIP(data_float, data_float, this->dim_));
            // Match Normalize's zero-vector divisor without changing nonfinite
            // propagation. Existing serialized zero molds are not rewritten.
            if (mold == 0.0F) {
                mold = 1.0F;
            }
            std::memcpy(codes, data, this->dim_ * sizeof(float));
            std::memcpy(codes + this->dim_ * sizeof(float), &mold, sizeof(float));
        } else {
            Normalize(data, reinterpret_cast<float*>(codes), this->dim_);
        }
    } else {
        std::memcpy(codes, data, this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    std::memcpy(data, codes, this->dim_ * sizeof(float));
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
    if (this->code_size_ == this->dim_ * sizeof(float)) {
        std::memcpy(data, codes, count * this->code_size_);
    } else {
        for (uint64_t i = 0; i < count; ++i) {
            std::memcpy(
                data + i * this->dim_, codes + i * this->code_size_, this->dim_ * sizeof(float));
        }
    }
    return true;
}

template <MetricType metric>
float
FP32Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if (metric == MetricType::METRIC_TYPE_IP) {
        return 1.0F - FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                    reinterpret_cast<const float*>(codes2),
                                    this->dim_);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        auto similarity = FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                        reinterpret_cast<const float*>(codes2),
                                        this->dim_);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            const float dot = similarity;
            const float product = mold1[0] * mold2[0];
            similarity /= product;
            if (cosine_exceptional(mold1[0]) or cosine_exceptional(mold2[0]) or
                cosine_exceptional(product) or cosine_non_finite(dot) or
                cosine_non_finite(similarity)) {
                // Match the existing normalized-storage path on exceptional
                // finite arithmetic, without changing stored raw rows.
                Vector<float> row1(this->dim_, this->allocator_);
                Vector<float> row2(this->dim_, this->allocator_);
                Normalize(reinterpret_cast<const float*>(codes1), row1.data(), this->dim_);
                Normalize(reinterpret_cast<const float*>(codes2), row2.data(), this->dim_);
                similarity = FP32ComputeIP(row1.data(), row2.data(), this->dim_);
            }
        }
        return 1.0F - similarity;
    }
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes1),
                                reinterpret_cast<const float*>(codes2),
                                this->dim_);
    }
    return 0.0F;
}

template <MetricType metric>
void
FP32Quantizer<metric>::ScanBatchDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                         uint64_t count,
                                         const uint8_t* codes,
                                         float* dists) const {
    // TODO(LHT): Optimize batch for simd
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ProcessQueryImpl(const float* query,
                                        Computer<FP32Quantizer<metric>>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        std::memcpy(computer.buf_, query, this->code_size_);
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    if (metric == MetricType::METRIC_TYPE_IP) {
        *dists = 1.0F - FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                      reinterpret_cast<const float*>(computer.buf_),
                                      this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        auto similarity = FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                        reinterpret_cast<const float*>(computer.buf_),
                                        this->dim_);
        if (this->hold_molds_) {
            const auto* mold = reinterpret_cast<const float*>(codes + this->dim_ * sizeof(float));
            const float dot = similarity;
            similarity /= mold[0];
            if (cosine_exceptional(mold[0]) or cosine_non_finite(dot) or
                cosine_non_finite(similarity)) {
                // The computer query was normalized by ProcessQueryImpl already.
                similarity =
                    normalized_code_similarity(reinterpret_cast<const float*>(codes),
                                               reinterpret_cast<const float*>(computer.buf_),
                                               this->dim_,
                                               this->allocator_);
            }
        }
        *dists = 1.0F - similarity;
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes),
                                  reinterpret_cast<const float*>(computer.buf_),
                                  this->dim_);
    } else {
        *dists = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistsBatch4Impl(Computer<FP32Quantizer<metric>>& computer,
                                              const uint8_t* codes1,
                                              const uint8_t* codes2,
                                              const uint8_t* codes3,
                                              const uint8_t* codes4,
                                              float& dists1,
                                              float& dists2,
                                              float& dists3,
                                              float& dists4) const {
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        FP32ComputeIPBatch4(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            dists1,
                            dists2,
                            dists3,
                            dists4);
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        const float initial[4] = {dists1, dists2, dists3, dists4};
        FP32ComputeIPBatch4(reinterpret_cast<const float*>(computer.buf_),
                            this->dim_,
                            reinterpret_cast<const float*>(codes1),
                            reinterpret_cast<const float*>(codes2),
                            reinterpret_cast<const float*>(codes3),
                            reinterpret_cast<const float*>(codes4),
                            dists1,
                            dists2,
                            dists3,
                            dists4);
        if (this->hold_molds_) {
            const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(float));
            const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(float));
            const auto* mold3 = reinterpret_cast<const float*>(codes3 + this->dim_ * sizeof(float));
            const auto* mold4 = reinterpret_cast<const float*>(codes4 + this->dim_ * sizeof(float));
            dists1 /= mold1[0];
            dists2 /= mold2[0];
            dists3 /= mold3[0];
            dists4 /= mold4[0];
            const uint8_t* rows[4] = {codes1, codes2, codes3, codes4};
            const float norms[4] = {mold1[0], mold2[0], mold3[0], mold4[0]};
            float* outputs[4] = {&dists1, &dists2, &dists3, &dists4};
            for (uint64_t i = 0; i < 4; ++i) {
                // FlattenDataCell zeroes its outputs. Low-level callers may
                // accumulate: retain their original hold-molds result exactly.
                if (cosine_zero(initial[i]) and
                    (cosine_exceptional(norms[i]) or cosine_non_finite(*outputs[i]))) {
                    *outputs[i] =
                        normalized_code_similarity(reinterpret_cast<const float*>(rows[i]),
                                                   reinterpret_cast<const float*>(computer.buf_),
                                                   this->dim_,
                                                   this->allocator_);
                }
            }
        }
        dists1 = 1.0F - dists1;
        dists2 = 1.0F - dists2;
        dists3 = 1.0F - dists3;
        dists4 = 1.0F - dists4;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        FP32ComputeL2SqrBatch4(reinterpret_cast<const float*>(computer.buf_),
                               this->dim_,
                               reinterpret_cast<const float*>(codes1),
                               reinterpret_cast<const float*>(codes2),
                               reinterpret_cast<const float*>(codes3),
                               reinterpret_cast<const float*>(codes4),
                               dists1,
                               dists2,
                               dists3,
                               dists4);
    } else {
        dists1 = 0.0F;
        dists2 = 0.0F;
        dists3 = 0.0F;
        dists4 = 0.0F;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

TEMPLATE_QUANTIZER(FP32Quantizer);
}  // namespace vsag
