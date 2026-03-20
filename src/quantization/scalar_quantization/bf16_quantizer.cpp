
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

#include "bf16_quantizer.h"

#include <cstring>

#include "simd/bf16_simd.h"
#include "simd/normalize.h"
#include "typing.h"
#include "utils/byte_buffer.h"

namespace vsag {

template <MetricType metric>
BF16Quantizer<metric>::BF16Quantizer(int dim, Allocator* allocator)
    : Quantizer<BF16Quantizer<metric>>(dim, allocator) {
    this->code_size_ = dim * 2;
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <MetricType metric>
BF16Quantizer<metric>::BF16Quantizer(const BF16QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : BF16Quantizer<metric>(common_param.dim_, common_param.allocator_.get()) {
    this->input_data_type_ = common_param.data_type_;
};

template <MetricType metric>
BF16Quantizer<metric>::BF16Quantizer(const QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : BF16Quantizer<metric>(std::dynamic_pointer_cast<BF16QuantizerParameter>(param),
                            common_param){};

template <MetricType metric>
bool
BF16Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    this->is_trained_ = true;
    return data != nullptr;
}

template <MetricType metric>
bool
BF16Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    auto* codes_bf16 = reinterpret_cast<uint16_t*>(codes);

    if (input_data_type_ == DataTypes::DATA_TYPE_BF16) {
        const auto* input_bf16 = reinterpret_cast<const uint16_t*>(data);
        std::memcpy(codes_bf16, input_bf16, this->dim_ * sizeof(uint16_t));
        return true;
    }

    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    for (int i = 0; i < this->dim_; ++i) {
        codes_bf16[i] = generic::FloatToBF16(cur[i]);
    }

    return true;
}

template <MetricType metric>
bool
BF16Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    const auto* codes_bf16 = reinterpret_cast<const uint16_t*>(codes);

    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = generic::BF16ToFloat(codes_bf16[d]);
    }
    return true;
}

template <MetricType metric>
inline float
BF16Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return BF16ComputeL2Sqr(codes1, codes2, this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - BF16ComputeIP(codes1, codes2, this->dim_);
    } else {
        return 0;
    }
}

template <MetricType metric>
void
BF16Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<BF16Quantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        auto* codes_bf16 = reinterpret_cast<uint16_t*>(computer.buf_);
        Vector<float> tmp(this->allocator_);
        const DataType* cur = query;
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            tmp.resize(this->dim_);
            Normalize(query, tmp.data(), this->dim_);
            cur = tmp.data();
        }
        for (int i = 0; i < this->dim_; ++i) {
            codes_bf16[i] = generic::FloatToBF16(cur[i]);
        }
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

template <MetricType metric>
void
BF16Quantizer<metric>::ComputeDistImpl(Computer<BF16Quantizer>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    auto* buf = computer.buf_;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = BF16ComputeL2Sqr(buf, codes, this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        dists[0] = 1 - BF16ComputeIP(buf, codes, this->dim_);
    } else {
        logger::error("unsupported metric type");
        dists[0] = 0;
    }
}

TEMPLATE_QUANTIZER(BF16Quantizer)
}  // namespace vsag
