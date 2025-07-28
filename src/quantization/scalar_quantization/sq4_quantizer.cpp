
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

#include "sq4_quantizer.h"

#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq4_simd.h"
#include "typing.h"

namespace vsag {

template <MetricType metric>
SQ4Quantizer<metric>::SQ4Quantizer(int dim, Allocator* allocator)
    : Quantizer<SQ4Quantizer<metric>>(dim, allocator) {
    this->code_size_ = (dim + (1 << 6) - 1) >> 6 << 6;
    this->query_code_size_ = this->dim_ * sizeof(float);
    this->metric_ = metric;
    lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
    diff_.resize(dim, std::numeric_limits<DataType>::lowest());
}

template <MetricType metric>
SQ4Quantizer<metric>::SQ4Quantizer(const SQ4QuantizerParamPtr& param,
                                   const IndexCommonParam& common_param)
    : SQ4Quantizer<metric>(common_param.dim_, common_param.allocator_.get()){};

template <MetricType metric>
SQ4Quantizer<metric>::SQ4Quantizer(const QuantizerParamPtr& param,
                                   const IndexCommonParam& common_param)
    : SQ4Quantizer<metric>(std::dynamic_pointer_cast<SQ4QuantizerParameter>(param), common_param){};

template <MetricType metric>
bool
SQ4Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr) {
        return false;
    }

    if (this->is_trained_) {
        return true;
    }
    bool need_normalize = false;
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        need_normalize = true;
    }

    ScalarQuantizationTrainer trainer(this->dim_, 4);
    trainer.Train(data, count, this->diff_.data(), this->lower_bound_.data(), need_normalize);

    for (uint64_t i = 0; i < this->dim_; ++i) {
        this->diff_[i] -= this->lower_bound_[i];
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
        delta = 1.0F * (cur[d] - lower_bound_[d]) / diff_[d];
        if (delta < 0.0F) {
            delta = 0;
        } else if (delta > 0.999F) {
            delta = 1;
        }
        scaled = static_cast<uint8_t>(15.0F * delta);

        if ((d & 1) != 0U) {
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
        if ((d & 1) != 0U) {
            data[d] = static_cast<float>((codes[d >> 1] & 0xF0) >> 4) / 15.0F * diff_[d] +
                      lower_bound_[d];
        } else {
            data[d] = static_cast<float>(codes[d >> 1] & 0x0F) / 15.0F * diff_[d] + lower_bound_[d];
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
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
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
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        dists[0] = 1 - SQ4ComputeIP(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
    } else {
        logger::error("unsupported metric type");
        dists[0] = 0;
    }
}

template <MetricType metric>
void
SQ4Quantizer<metric>::ScanBatchDistImpl(Computer<SQ4Quantizer<metric>>& computer,
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

TEMPLATE_QUANTIZER(SQ4Quantizer);
}  // namespace vsag
