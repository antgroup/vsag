
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

#include "quantizer.h"
#include "simd/simd.h"

namespace vsag {

template <MetricType Metric = MetricType::METRIC_TYPE_L2SQR>
class FP32Quantizer : public Quantizer<FP32Quantizer<Metric>> {
public:
    explicit FP32Quantizer(int dim);

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

    inline void
    ProcessQueryImpl(const DataType* query, Computer<FP32Quantizer<Metric>>& computer) const;

    inline void
    ComputeDistImpl(Computer<FP32Quantizer<Metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;
};

template <MetricType Metric>
FP32Quantizer<Metric>::FP32Quantizer(int dim) : Quantizer<FP32Quantizer<Metric>>(dim) {
    this->code_size_ = dim * sizeof(float);
}

template <MetricType Metric>
bool
FP32Quantizer<Metric>::TrainImpl(const DataType* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType Metric>
bool
FP32Quantizer<Metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    memcpy(codes, data, this->code_size_);
    return true;
}

template <MetricType Metric>
bool
FP32Quantizer<Metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        memcpy(codes + i * this->code_size_, data + i * this->dim_, this->code_size_);
    }
    return true;
}

template <MetricType Metric>
bool
FP32Quantizer<Metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    memcpy(data, codes, this->code_size_);
    return true;
}

template <MetricType Metric>
bool
FP32Quantizer<Metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        memcpy(data + i * this->dim_, codes + i * this->code_size_, this->code_size_);
    }
    return true;
}

template <MetricType Metric>
float
FP32Quantizer<Metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if (Metric == MetricType::METRIC_TYPE_IP) {
        return InnerProduct(codes1, codes2, &this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_L2SQR) {
        return L2Sqr(codes1, codes2, &this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_COSINE) {
        return InnerProduct(codes1, codes2, &this->dim_);  // TODO
    } else {
        return 0.0f;
    }
}

template <MetricType Metric>
void
FP32Quantizer<Metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<FP32Quantizer<Metric>>& computer) const {
    computer.buf_ = new uint8_t[this->code_size_];
    memcpy(computer.buf_, query, this->code_size_);
}

template <MetricType Metric>
void
FP32Quantizer<Metric>::ComputeDistImpl(Computer<FP32Quantizer<Metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    if (Metric == MetricType::METRIC_TYPE_IP) {
        *dists = InnerProduct(codes, computer.buf_, &this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = L2Sqr(codes, computer.buf_, &this->dim_);
    } else if (Metric == MetricType::METRIC_TYPE_COSINE) {
        *dists = InnerProduct(codes, computer.buf_, &this->dim_);  // TODO
    } else {
        *dists = 0.0f;
    }
}
}  // namespace vsag
