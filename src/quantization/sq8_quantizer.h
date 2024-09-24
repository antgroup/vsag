
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
#include <vector>

#include "quantizer.h"
#include "simd/sq8_simd.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8Quantizer : public Quantizer<SQ8Quantizer<metric>> {
public:
    explicit SQ8Quantizer(int dim);

    ~SQ8Quantizer() = default;

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

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ8Quantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ8Quantizer>& computer, const uint8_t* codes, float* dists) const;

public:
    std::vector<DataType> diff_{};
    std::vector<DataType> lower_bound_{};
};

template <MetricType Metric>
SQ8Quantizer<Metric>::SQ8Quantizer(int dim) : Quantizer<SQ8Quantizer<Metric>>(dim) {
    this->code_size_ = this->dim_;
    this->diff_.resize(dim, 0);
    this->lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
}

template <MetricType Metric>
bool
SQ8Quantizer<Metric>::TrainImpl(const vsag::DataType* data, uint64_t count) {
    if (this->is_trained_) {
        return true;
    }
    std::vector<DataType> upperBound(this->dim_, std::numeric_limits<DataType>::lowest());
    for (uint64_t i = 0; i < this->dim_; ++i) {
        for (uint64_t j = 0; j < count; ++j) {
            upperBound[i] = std::max(upperBound[i], data[j * this->dim_ + i]);
            lower_bound_[i] = std::min(lower_bound_[i], data[j * this->dim_ + i]);
        }
    }
    for (uint64_t i = 0; i < this->dim_; ++i) {
        this->diff_[i] = upperBound[i] - this->lower_bound_[i];
    }
    this->is_trained_ = true;
    return true;
}

template <MetricType Metric>
bool
SQ8Quantizer<Metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    for (int i = 0; i < this->dim_; ++i) {
        float xi = 0;
        if (diff_[i] != 0) {
            xi = (data[i] - lower_bound_[i]) / diff_[i];
            if (xi < 0.0) {
                xi = 0;
            }
            if (xi > 0.999) {
                xi = 1.0;
            }
        }
        codes[i] = int(xi * 255);
    }
    return true;
}

template <MetricType Metric>
bool
SQ8Quantizer<Metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType Metric>
bool
SQ8Quantizer<Metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType Metric>
bool
SQ8Quantizer<Metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t i = 0; i < this->dim_; ++i) {
        data[i] = static_cast<DataType>(static_cast<float>(codes[i]) / 255.0 * diff_[i] +
                                        lower_bound_[i]);
    }
    return true;
}

template <MetricType Metric>
inline float
SQ8Quantizer<Metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (Metric == MetricType::METRIC_TYPE_L2SQR) {
        return SQ8ComputeCodesL2Sqr(
            codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (Metric == MetricType::METRIC_TYPE_IP) {
        return SQ8ComputeCodesIP(
            codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (Metric == MetricType::METRIC_TYPE_COSINE) {
        return SQ8ComputeCodesIP(
            codes1, codes2, this->lower_bound_.data(), this->diff_.data(), this->dim_);  // TODO
    } else {
        return 0.;
    }
}

template <MetricType Metric>
void
SQ8Quantizer<Metric>::ProcessQueryImpl(const DataType* query,
                                       Computer<SQ8Quantizer>& computer) const {
    computer.buf_ = new uint8_t[this->dim_ * sizeof(float)];
    std::memcpy(computer.buf_, query, this->dim_ * sizeof(float));
}

template <MetricType Metric>
void
SQ8Quantizer<Metric>::ComputeDistImpl(Computer<SQ8Quantizer>& computer,
                                      const uint8_t* codes,
                                      float* dists) const {
    auto* query = reinterpret_cast<float*>(computer.buf_);

    if constexpr (Metric == MetricType::METRIC_TYPE_L2SQR) {
        *dists = SQ8ComputeL2Sqr(
            query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (Metric == MetricType::METRIC_TYPE_IP) {
        *dists =
            SQ8ComputeIP(query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);
    } else if constexpr (Metric == MetricType::METRIC_TYPE_COSINE) {
        *dists = SQ8ComputeIP(
            query, codes, this->lower_bound_.data(), this->diff_.data(), this->dim_);  // TODO
    } else {
        *dists = 0.;
    }
}

}  // namespace vsag
