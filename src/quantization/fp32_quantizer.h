
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

#pragma once

#include <cstdint>
#include <cstring>

#include "index/index_common_param.h"
#include "nlohmann/json.hpp"
#include "quantizer.h"
#include "simd/simd.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP32Quantizer : public Quantizer<FP32Quantizer<metric>> {
public:
    explicit FP32Quantizer(int dim);

    FP32Quantizer(const nlohmann::json& quantization_obj, const IndexCommonParam& common_param);

    ~FP32Quantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    SerializeImpl(StreamWriter& writer){};

    void
    DeserializeImpl(StreamReader& reader){};

    inline void
    ProcessQueryImpl(const DataType* query, Computer<FP32Quantizer<metric>>& computer) const;

    inline void
    ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;
};

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const nlohmann::json& quantization_obj,
                                     const IndexCommonParam& common_param)
    : Quantizer<FP32Quantizer<metric>>(common_param.dim_) {
    this->code_size_ = common_param.dim_ * sizeof(float);
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(int dim) : Quantizer<FP32Quantizer<metric>>(dim) {
    this->code_size_ = dim * sizeof(float);
}

template <MetricType metric>
bool
FP32Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    memcpy(codes, data, this->code_size_);
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        memcpy(codes + i * this->code_size_, data + i * this->dim_, this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    memcpy(data, codes, this->code_size_);
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        memcpy(data + i * this->dim_, codes + i * this->code_size_, this->code_size_);
    }
    return true;
}

template <MetricType metric>
float
FP32Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if (metric == MetricType::METRIC_TYPE_IP) {
        return InnerProduct(codes1, codes2, &this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return L2Sqr(codes1, codes2, &this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        return InnerProduct(codes1, codes2, &this->dim_);  // TODO
    } else {
        return 0.0f;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<FP32Quantizer<metric>>& computer) const {
    computer.buf_ = new uint8_t[this->code_size_];
    memcpy(computer.buf_, query, this->code_size_);
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    if (metric == MetricType::METRIC_TYPE_IP) {
        *dists = InnerProduct(codes, computer.buf_, &this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = L2Sqr(codes, computer.buf_, &this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        *dists = InnerProduct(codes, computer.buf_, &this->dim_);  // TODO
    } else {
        *dists = 0.0f;
    }
}

}  // namespace vsag
