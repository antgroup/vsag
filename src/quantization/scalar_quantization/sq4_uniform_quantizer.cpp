
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

#include "sq4_uniform_quantizer.h"

#include <cstddef>

#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq4_uniform_simd.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

using norm_type = uint64_t;

template <MetricType metric>
SQ4UniformQuantizer<metric>::SQ4UniformQuantizer(int dim, Allocator* allocator, float trunc_rate)
    : Quantizer<SQ4UniformQuantizer<metric>>(dim, allocator), trunc_rate_(trunc_rate) {
    lower_bound_ = std::numeric_limits<DataType>::max();
    diff_ = std::numeric_limits<DataType>::lowest();

    size_t align_size = 1;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        align_size = std::max(align_size, sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        align_size = std::max(align_size, sizeof(sum_type));
    }
    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    dim = (dim + 1) / 2;
    this->code_size_ += ceil_int(dim, static_cast<int64_t>(align_size));

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_norm_ = this->code_size_;
        this->code_size_ += ceil_int(sizeof(norm_type), static_cast<int64_t>(align_size));
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_sum_ = this->code_size_;
        this->code_size_ += ceil_int(sizeof(sum_type), static_cast<int64_t>(align_size));

        offset_codes_sum_ = this->code_size_;
        this->code_size_ += ceil_int(sizeof(sum_type), static_cast<int64_t>(align_size));
    }

    // align 64 bytes (512 bits) to avoid illegal memory access in SIMD
    this->code_size_ = ceil_int(this->code_size_, 64);

    this->query_code_size_ = this->code_size_;
}

template <MetricType metric>
SQ4UniformQuantizer<metric>::SQ4UniformQuantizer(const SQ4UniformQuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : SQ4UniformQuantizer<metric>(
          common_param.dim_, common_param.allocator_.get(), param->trunc_rate_){};

template <MetricType metric>
SQ4UniformQuantizer<metric>::SQ4UniformQuantizer(const QuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : SQ4UniformQuantizer<metric>(std::dynamic_pointer_cast<SQ4UniformQuantizerParameter>(param),
                                  common_param){};

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
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
    trainer.SetSQ4UniformTruncRate(this->trunc_rate_);
    trainer.TrainUniform(data, count, this->diff_, this->lower_bound_, need_normalize);

    this->diff_ -= this->lower_bound_;
    this->scalar_rate_ = (diff_ / 15.0) * (diff_ / 15.0);

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    norm_type norm = 0;
    sum_type sum = 0;
    sum_type codes_sum = 0;

    Vector<DataType> norm_data(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_data.resize(this->dim_);
        Normalize(data, norm_data.data(), this->dim_);
        data = norm_data.data();
    }

    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = 1.0F * (data[d] - lower_bound_) / diff_;
        if (delta < 0.0F) {
            delta = 0.0F;
        } else if (delta > 0.999F) {
            delta = 1.0F;
        }
        scaled = static_cast<uint8_t>(15.0F * delta);

        if ((d & 1) != 0U) {
            codes[offset_code_ + (d >> 1)] |= scaled << 4;
        } else {
            codes[offset_code_ + (d >> 1)] = 0;
            codes[offset_code_ + (d >> 1)] |= scaled;
        }
        norm += static_cast<uint64_t>(scaled) * static_cast<uint64_t>(scaled);
        sum += data[d];
        codes_sum += static_cast<float>(scaled);
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        *(norm_type*)(codes + offset_norm_) = norm;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(sum_type*)(codes + offset_sum_) = sum;
        *(sum_type*)(codes + offset_codes_sum_) = codes_sum;
    }

    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        if ((d & 1) != 0U) {
            data[d] = ((codes[d >> 1] & 0xf0) >> 4) / 15.0 * diff_ + lower_bound_;
        } else {
            data[d] = (codes[d >> 1] & 0x0f) / 15.0 * diff_ + lower_bound_;
        }
    }

    return true;
}

template <MetricType metric>
bool
SQ4UniformQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
SQ4UniformQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    float result = 0.0F;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        norm_type norm1 = *((norm_type*)(codes1 + offset_norm_));
        norm_type norm2 = *((norm_type*)(codes2 + offset_norm_));

        result =
            (static_cast<float>(norm1) + static_cast<float>(norm2) - 2 * result) * scalar_rate_;
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        result = SQ4UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offset_sum_));
        sum_type sum2 = *((sum_type*)(codes2 + offset_sum_));

        result = lower_bound_ * (sum1 + sum2) + scalar_rate_ * result + lower_bound_ * lower_bound_;

        result = 1 - result;

    } else {
        logger::error("unsupported metric type");
        result = 0;
    }
    return result;
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<SQ4UniformQuantizer>& computer) const {
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
        this->EncodeOneImpl(query, computer.buf_);
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::ComputeDistImpl(Computer<SQ4UniformQuantizer>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::ScanBatchDistImpl(Computer<SQ4UniformQuantizer<metric>>& computer,
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
SQ4UniformQuantizer<metric>::ReleaseComputerImpl(
    Computer<SQ4UniformQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->diff_);
    StreamWriter::WriteObj(writer, this->lower_bound_);
    StreamWriter::WriteObj(writer, this->offset_code_);
    StreamWriter::WriteObj(writer, this->offset_norm_);
    StreamWriter::WriteObj(writer, this->offset_sum_);
    StreamWriter::WriteObj(writer, this->offset_codes_sum_);
}

template <MetricType metric>
void
SQ4UniformQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->diff_);
    StreamReader::ReadObj(reader, this->lower_bound_);
    StreamReader::ReadObj(reader, this->offset_code_);
    StreamReader::ReadObj(reader, this->offset_norm_);
    StreamReader::ReadObj(reader, this->offset_sum_);
    StreamReader::ReadObj(reader, this->offset_codes_sum_);
    this->scalar_rate_ = (diff_ / 15.0) * (diff_ / 15.0);
}

TEMPLATE_QUANTIZER(SQ4UniformQuantizer)

}  // namespace vsag
