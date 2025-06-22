
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

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq8_uniform_simd.h"
#include "sq8_row_quantizer_parameter.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8RowQuantizer : public Quantizer<SQ8RowQuantizer<metric>> {
public:
    explicit SQ8RowQuantizer(int dim, Allocator* allocator);

    SQ8RowQuantizer(const SQ8RowQuantizerParamPtr& param, const IndexCommonParam& common_param);

    SQ8RowQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~SQ8RowQuantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ8RowQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ8RowQuantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ScanBatchDistImpl(Computer<SQ8RowQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    inline void
    ReleaseComputerImpl(Computer<SQ8RowQuantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ8;
    }
};

template <MetricType metric>
SQ8RowQuantizer<metric>::SQ8RowQuantizer(int dim, Allocator* allocator)
    : Quantizer<SQ8RowQuantizer<metric>>(dim, allocator) {
    // align 64 bytes (512 bits) to avoid illegal memory access in SIMD
    this->code_size_ = this->dim_ + 4 * sizeof(float);
    this->metric_ = metric;
}

template <MetricType metric>
SQ8RowQuantizer<metric>::SQ8RowQuantizer(const SQ8RowQuantizerParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SQ8RowQuantizer<metric>(common_param.dim_, common_param.allocator_.get()) {
}

template <MetricType metric>
SQ8RowQuantizer<metric>::SQ8RowQuantizer(const QuantizerParamPtr& param,
                                         const IndexCommonParam& common_param)
    : SQ8RowQuantizer<metric>(std::dynamic_pointer_cast<SQ8RowQuantizerParameter>(param),
                              common_param) {
}

template <MetricType metric>
bool
SQ8RowQuantizer<metric>::TrainImpl(const vsag::DataType* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ8RowQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    float min_value = cur[0];
    float max_value = cur[0];
    for (int i = 1; i < this->dim_; ++i) {
        min_value = std::min(min_value, cur[i]);
        max_value = std::max(max_value, cur[i]);
    }
    float diff = max_value - min_value;
    float sum = 0.0f;
    int32_t scalar = 0;
    for (int i = 0; i < this->dim_; ++i) {
        float xi = 0;
        if (diff != 0) {
            xi = (cur[i] - min_value) / diff;
            if (xi < 0.0) {
                xi = 0;
            }
            if (xi > 0.999) {
                xi = 1.0;
            }
        }
        codes[i] = int(xi * 255);
        scalar += codes[i] * codes[i];
        sum += cur[i];
    }
    // Store lower bound and diff for decoding
    auto base = reinterpret_cast<float*>(codes + this->dim_);
    base[0] = min_value;
    base[1] = diff;
    base[2] = static_cast<float>(scalar) / (255.0f * 255.0f) * diff * diff;
    base[3] = static_cast<float>(sum);
    return true;
}

template <MetricType metric>
bool
SQ8RowQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ8RowQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
bool
SQ8RowQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    auto base = reinterpret_cast<const float*>(codes + this->dim_);
    for (uint64_t i = 0; i < this->dim_; ++i) {
        data[i] = static_cast<DataType>(static_cast<float>(codes[i]) / 255.0 * base[1] + base[0]);
    }
    return true;
}

template <MetricType metric>
inline float
SQ8RowQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        auto min_value1 = reinterpret_cast<const float*>(codes1 + this->dim_)[0];
        auto min_value2 = reinterpret_cast<const float*>(codes2 + this->dim_)[0];
        auto diff1 = reinterpret_cast<const float*>(codes1 + this->dim_)[1];
        auto diff2 = reinterpret_cast<const float*>(codes2 + this->dim_)[1];
        auto scalar1 = reinterpret_cast<const float*>(codes1 + this->dim_)[2];
        auto scalar2 = reinterpret_cast<const float*>(codes2 + this->dim_)[2];
        auto sum1 = reinterpret_cast<const float*>(codes1 + this->dim_)[3];
        auto sum2 = reinterpret_cast<const float*>(codes2 + this->dim_)[3];
        auto result = SQ8UniformComputeCodesIP(codes1, codes2, this->dim_) / (255.0F * 255.0F);
        auto part1 = -2 * result * (diff1 * diff2);
        auto part2 = scalar1 + scalar2;
        auto part3 = this->dim_ * (min_value1 - min_value2) * (min_value1 - min_value2);
        auto part4 = 2 * (min_value1 - min_value2) * (sum1 - sum2);
        return part1 + part2 + part3 + part4;
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        return 0.0f;
    } else {
        return 0.0f;
    }
}

template <MetricType metric>
void
SQ8RowQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                          Computer<SQ8RowQuantizer>& computer) const {
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
        this->EncodeOneImpl(query, computer.buf_);
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

template <MetricType metric>
void
SQ8RowQuantizer<metric>::ComputeDistImpl(Computer<SQ8RowQuantizer>& computer,
                                         const uint8_t* codes,
                                         float* dists) const {
    *dists = this->ComputeImpl(codes, computer.buf_);
}

template <MetricType metric>
void
SQ8RowQuantizer<metric>::ScanBatchDistImpl(Computer<SQ8RowQuantizer<metric>>& computer,
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
SQ8RowQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
}

template <MetricType metric>
void
SQ8RowQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
}

template <MetricType metric>
void
SQ8RowQuantizer<metric>::ReleaseComputerImpl(Computer<SQ8RowQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag
