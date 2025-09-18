
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

#include "quantization/int8_quantizer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>

#include "metric_type.h"
#include "quantization/computer.h"
#include "quantization/int8_quantizer_parameter.h"
#include "quantization/quantizer.h"
#include "simd/int8_simd.h"
#include "vsag_exception.h"

namespace vsag {

template <MetricType metric>
INT8Quantizer<metric>::INT8Quantizer(int dim, Allocator* allocator)
    : Quantizer<INT8Quantizer<metric>>(dim, allocator) {
    this->code_size_ = dim * sizeof(int8_t);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        this->code_size_ += sizeof(float);
    }
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <MetricType metric>
INT8Quantizer<metric>::INT8Quantizer(const INT8QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : INT8Quantizer<metric>(common_param.dim_, common_param.allocator_.get()) {
    this->hold_molds_ = param->hold_molds;
}

template <MetricType metric>
INT8Quantizer<metric>::INT8Quantizer(const QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : INT8Quantizer<metric>(std::dynamic_pointer_cast<INT8QuantizerParameter>(param),
                            common_param) {
}

template <MetricType metric>
bool
INT8Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
INT8Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    memcpy(codes, data, this->dim_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        // Store the mold for cosine similarity
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        float mold = std::sqrt(INT8ComputeIP(data_int8, data_int8, this->dim_));
        memcpy(codes + this->dim_ * sizeof(int8_t), &mold, sizeof(float));
    }
    return true;
}

template <MetricType metric>
bool
INT8Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    const auto* data_ptr = reinterpret_cast<const int8_t*>(data);
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(reinterpret_cast<const float*>(data_ptr + i * this->dim_),
                      codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
INT8Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    memcpy(data, codes, this->dim_);
    return true;
}

template <MetricType metric>
bool
INT8Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    auto* data_ptr = reinterpret_cast<int8_t*>(data);
    for (uint64_t i = 0; i < count; ++i) {
        memcpy(data_ptr + i * this->dim_, codes + i * this->code_size_, this->dim_);
    }
    return true;
}

template <MetricType metric>
float
INT8Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        return 1.0F - INT8ComputeIP(reinterpret_cast<const int8_t*>(codes1),
                                    reinterpret_cast<const int8_t*>(codes2),
                                    this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        const auto* mold1 = reinterpret_cast<const float*>(codes1 + this->dim_ * sizeof(uint8_t));
        const auto* mold2 = reinterpret_cast<const float*>(codes2 + this->dim_ * sizeof(uint8_t));

        if (*mold1 == 0 or *mold2 == 0) {
            return 1.0F;
        }
        auto similarity = INT8ComputeIP(reinterpret_cast<const int8_t*>(codes1),
                                        reinterpret_cast<const int8_t*>(codes2),
                                        this->dim_);
        similarity /= mold1[0] * mold2[0];
        return 1.0F - std::max(-1.0F, std::min(1.0F, similarity));
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return INT8ComputeL2Sqr(reinterpret_cast<const int8_t*>(codes1),
                                reinterpret_cast<const int8_t*>(codes2),
                                this->dim_);
    }
}

template <MetricType metric>
void
INT8Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<INT8Quantizer<metric>>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    memcpy(computer.buf_, query, this->dim_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        // Store the mold for cosine similarity
        const auto* data_int8 = reinterpret_cast<const int8_t*>(query);
        float mold = std::sqrt(INT8ComputeIP(data_int8, data_int8, this->dim_));
        memcpy(computer.buf_ + this->dim_ * sizeof(int8_t), &mold, sizeof(float));
    }
}

template <MetricType metric>
void
INT8Quantizer<metric>::ComputeDistImpl(Computer<INT8Quantizer<metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    static_assert(metric == MetricType::METRIC_TYPE_IP ||
                      metric == MetricType::METRIC_TYPE_COSINE ||
                      metric == MetricType::METRIC_TYPE_L2SQR,
                  "Unsupported metric type for INT8Quantizer");
    if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        *dists = 1.0F - INT8ComputeIP(reinterpret_cast<const int8_t*>(codes),
                                      reinterpret_cast<const int8_t*>(computer.buf_),
                                      this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        const auto* mold = reinterpret_cast<const float*>(codes + this->dim_ * sizeof(uint8_t));
        const auto* query_mold =
            reinterpret_cast<const float*>(computer.buf_ + this->dim_ * sizeof(uint8_t));
        if (*mold == 0 or *query_mold == 0) {
            *dists = 1.0F;
            return;
        }
        const auto similarity = INT8ComputeIP(reinterpret_cast<const int8_t*>(codes),
                                              reinterpret_cast<const int8_t*>(computer.buf_),
                                              this->dim_);
        *dists = 1.0F - std::max(-1.0F, std::min(1.0F, similarity / (mold[0] * query_mold[0])));
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = INT8ComputeL2Sqr(reinterpret_cast<const int8_t*>(codes),
                                  reinterpret_cast<const int8_t*>(computer.buf_),
                                  this->dim_);
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "invalid metric type");
    }
}

template <MetricType metric>
void
INT8Quantizer<metric>::ScanBatchDistImpl(Computer<INT8Quantizer<metric>>& computer,
                                         uint64_t count,
                                         const uint8_t* codes,
                                         float* dists) const {
    for (uint64_t i = 0; i < count; i++) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
INT8Quantizer<metric>::ReleaseComputerImpl(Computer<INT8Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

TEMPLATE_QUANTIZER(INT8Quantizer)
}  // namespace vsag
