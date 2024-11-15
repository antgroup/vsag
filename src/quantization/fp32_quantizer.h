
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
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/simd.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class FP32Quantizer : public Quantizer<FP32Quantizer<metric>> {
public:
    explicit FP32Quantizer(int dim, Allocator* allocator);

    FP32Quantizer(const JsonType& quantization_param, const IndexCommonParam& common_param);

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

    inline void
    ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const;
};

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(const JsonType& quantization_param,
                                     const IndexCommonParam& common_param)
    : Quantizer<FP32Quantizer<metric>>(common_param.dim_, common_param.allocator_) {
    this->code_size_ = common_param.dim_ * sizeof(float);
}

template <MetricType metric>
FP32Quantizer<metric>::FP32Quantizer(int dim, Allocator* allocator)
    : Quantizer<FP32Quantizer<metric>>(dim, allocator) {
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
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(data, reinterpret_cast<float*>(codes), this->dim_);
    } else {
        memcpy(codes, data, this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP32Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
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
        return 1 - FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                 reinterpret_cast<const float*>(codes2),
                                 this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes1),
                                reinterpret_cast<const float*>(codes2),
                                this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - FP32ComputeIP(reinterpret_cast<const float*>(codes1),
                                 reinterpret_cast<const float*>(codes2),
                                 this->dim_);  // TODO
    } else {
        return 0.0f;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<FP32Quantizer<metric>>& computer) const {
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        memcpy(computer.buf_, query, this->code_size_);
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ComputeDistImpl(Computer<FP32Quantizer<metric>>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    if (metric == MetricType::METRIC_TYPE_IP) {
        *dists = 1 - FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                   reinterpret_cast<const float*>(computer.buf_),
                                   this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = FP32ComputeL2Sqr(reinterpret_cast<const float*>(codes),
                                  reinterpret_cast<const float*>(computer.buf_),
                                  this->dim_);
    } else if (metric == MetricType::METRIC_TYPE_COSINE) {
        *dists = 1 - FP32ComputeIP(reinterpret_cast<const float*>(codes),
                                   reinterpret_cast<const float*>(computer.buf_),
                                   this->dim_);  // TODO
    } else {
        *dists = 0.0f;
    }
}

template <MetricType metric>
void
FP32Quantizer<metric>::ReleaseComputerImpl(Computer<FP32Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag
