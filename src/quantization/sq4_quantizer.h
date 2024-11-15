
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

#include <cstring>
#include <limits>
#include <vector>

#include "index/index_common_param.h"
#include "quantizer.h"
#include "simd/normalize.h"
#include "simd/sq4_simd.h"
#include "typing.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ4Quantizer : public Quantizer<SQ4Quantizer<metric>> {
public:
    explicit SQ4Quantizer(int dim, Allocator* allocator);

    explicit SQ4Quantizer(const JsonType& quantization_param, const IndexCommonParam& common_param);

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
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ4Quantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ4Quantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ReleaseComputerImpl(Computer<SQ4Quantizer<metric>>& computer) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

private:
    std::vector<DataType> lower_bound_{};
    std::vector<DataType> diff_{};
};

template <MetricType metric>
SQ4Quantizer<metric>::SQ4Quantizer(int dim, Allocator* allocator)
    : Quantizer<SQ4Quantizer<metric>>(dim, allocator) {
    this->code_size_ = (dim + (1 << 6) - 1) >> 6 << 6;
    lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
    diff_.resize(dim, std::numeric_limits<DataType>::lowest());
}

template <MetricType metric>
SQ4Quantizer<metric>::SQ4Quantizer(const JsonType& quantization_param,
                                   const IndexCommonParam& common_param)
    : SQ4Quantizer<metric>(common_param.dim_, common_param.allocator_){};

template <MetricType metric>
bool
SQ4Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr) {
        return false;
    }

    std::fill(lower_bound_.begin(), lower_bound_.end(), std::numeric_limits<DataType>::max());
    std::fill(diff_.begin(), diff_.end(), std::numeric_limits<DataType>::lowest());

    Vector<float> norms(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norms.resize(count);
        Vector<float> tmp(this->dim_, this->allocator_);
        for (uint64_t i = 0; i < count; ++i) {
            norms[i] = Normalize(data + i * this->dim_, tmp.data(), this->dim_);
        }
    }

    for (uint64_t d = 0; d < this->dim_; d++) {
        for (uint64_t i = 0; i < count; i++) {
            auto val = data[i * this->dim_ + d];
            if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
                if (norms[i] != 0) {
                    val /= norms[i];
                }
            }
            if (val > diff_[d]) {
                diff_[d] = val;
            }
            if (val < lower_bound_[d]) {
                lower_bound_[d] = val;
            }
        }
    }

    for (uint64_t d = 0; d < this->dim_; d++) {
        diff_[d] -= lower_bound_[d];
        if (diff_[d] < 1e-4) {
            diff_[d] = 1;
        }
    }

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ4Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = 1.0f * (cur[d] - lower_bound_[d]) / diff_[d];
        if (delta < 0.0) {
            delta = 0;
        } else if (delta > 0.999) {
            delta = 1;
        }
        scaled = 15 * delta;

        if (d & 1) {
            codes[d >> 1] |= scaled << 4;
        } else {
            codes[d >> 1] = 0;
            codes[d >> 1] |= scaled;
        }
    }

    return true;
}

template <MetricType metric>
bool
SQ4Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ4Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        if (d & 1) {
            data[d] = ((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff_[d] + lower_bound_[d];
        } else {
            data[d] = (codes[d >> 1] & 0x0f) / 15.0 * diff_[d] + lower_bound_[d];
        }
    }

    return true;
}

template <MetricType metric>
bool
SQ4Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
SQ4Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        return 1 - SQ4ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - SQ4ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
    } else {
        return 0;
    }
}

template <MetricType metric>
void
SQ4Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                       Computer<SQ4Quantizer>& computer) const {
    try {
        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->dim_ * sizeof(float)));

    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        memcpy(computer.buf_, query, this->dim_ * sizeof(float));
    }
}

template <MetricType metric>
void
SQ4Quantizer<metric>::ComputeDistImpl(Computer<SQ4Quantizer>& computer,
                                      const uint8_t* codes,
                                      float* dists) const {
    auto* buf = reinterpret_cast<float*>(computer.buf_);
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = SQ4ComputeL2Sqr(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1 - SQ4ComputeIP(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        dists[0] = 1 - SQ4ComputeIP(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else {
        logger::error("unsupported metric type");
        dists[0] = 0;
    }
}

template <MetricType metric>
void
SQ4Quantizer<metric>::ReleaseComputerImpl(Computer<SQ4Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
SQ4Quantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->diff_);
    StreamWriter::WriteVector(writer, this->lower_bound_);
}

template <MetricType metric>
void
SQ4Quantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->diff_);
    StreamReader::ReadVector(reader, this->lower_bound_);
}

}  // namespace vsag
