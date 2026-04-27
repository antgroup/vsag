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

#include "half_precision_quantizer.h"

#include "simd/normalize.h"
#include "typing.h"
#include "vsag_exception.h"

namespace vsag {

template <typename Format, MetricType metric>
HalfPrecisionQuantizer<Format, metric>::HalfPrecisionQuantizer(int dim, Allocator* allocator)
    : Quantizer<HalfPrecisionQuantizer<Format, metric>>(dim, allocator) {
    this->code_size_ = dim * 2;
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <typename Format, MetricType metric>
HalfPrecisionQuantizer<Format, metric>::HalfPrecisionQuantizer(
    const HalfPrecisionQuantizerParameter<Format>& param, const IndexCommonParam& common_param)
    : HalfPrecisionQuantizer<Format, metric>(common_param.dim_, common_param.allocator_.get()) {
}

template <typename Format, MetricType metric>
HalfPrecisionQuantizer<Format, metric>::HalfPrecisionQuantizer(const QuantizerParamPtr& param,
                                                               const IndexCommonParam& common_param)
    : HalfPrecisionQuantizer<Format, metric>(common_param.dim_, common_param.allocator_.get()) {
    if (param && param->GetTypeName() != Format::TYPE_NAME) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Parameter type mismatch: expected " + std::string(Format::TYPE_NAME) +
                                " but got " + param->GetTypeName());
    }
}

template <typename Format, MetricType metric>
bool
HalfPrecisionQuantizer<Format, metric>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return data != nullptr;
}

template <typename Format, MetricType metric>
bool
HalfPrecisionQuantizer<Format, metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    auto* codes_half = reinterpret_cast<uint16_t*>(codes);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Vector<float> tmp(this->dim_, this->allocator_);
        Normalize(data, tmp.data(), this->dim_);
        for (uint64_t i = 0; i < this->dim_; ++i) {
            codes_half[i] = Format::FloatToHalf(tmp[i]);
        }
    } else {
        for (uint64_t i = 0; i < this->dim_; ++i) {
            codes_half[i] = Format::FloatToHalf(data[i]);
        }
    }
    return true;
}

template <typename Format, MetricType metric>
bool
HalfPrecisionQuantizer<Format, metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    const auto* codes_half = reinterpret_cast<const uint16_t*>(codes);
    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = Format::HalfToFloat(codes_half[d]);
    }
    return true;
}

template <typename Format, MetricType metric>
float
HalfPrecisionQuantizer<Format, metric>::ComputeImpl(const uint8_t* codes1,
                                                    const uint8_t* codes2) const {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return Format::ComputeL2Sqr(codes1, codes2, this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - Format::ComputeIP(codes1, codes2, this->dim_);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "unsupported metric type");
    }
}

template <typename Format, MetricType metric>
void
HalfPrecisionQuantizer<Format, metric>::ProcessQueryImpl(
    const float* query, Computer<HalfPrecisionQuantizer<Format, metric>>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        this->EncodeOneImpl(query, computer.buf_);
    } catch (const std::bad_alloc& e) {
        throw VsagException(
            ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf", e.what());
    }
}

template <typename Format, MetricType metric>
void
HalfPrecisionQuantizer<Format, metric>::ComputeDistImpl(
    Computer<HalfPrecisionQuantizer<Format, metric>>& computer,
    const uint8_t* codes,
    float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

template class HalfPrecisionQuantizer<FP16Format, MetricType::METRIC_TYPE_L2SQR>;
template class HalfPrecisionQuantizer<FP16Format, MetricType::METRIC_TYPE_IP>;
template class HalfPrecisionQuantizer<FP16Format, MetricType::METRIC_TYPE_COSINE>;
template class HalfPrecisionQuantizer<BF16Format, MetricType::METRIC_TYPE_L2SQR>;
template class HalfPrecisionQuantizer<BF16Format, MetricType::METRIC_TYPE_IP>;
template class HalfPrecisionQuantizer<BF16Format, MetricType::METRIC_TYPE_COSINE>;

}  // namespace vsag
