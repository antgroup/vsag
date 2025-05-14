
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

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq8_uniform_simd.h"
#include "sq8_uniform_quantizer_parameter.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQ8UniformQuantizer : public Quantizer<SQ8UniformQuantizer<metric>> {
public:
    using norm_type = uint64_t;
    using sum_type = float;

    explicit SQ8UniformQuantizer(int dim, Allocator* allocator);

    SQ8UniformQuantizer(const SQ8UniformQuantizerParamPtr& param,
                        const IndexCommonParam& common_param);

    SQ8UniformQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~SQ8UniformQuantizer() = default;

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

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    inline void
    ProcessQueryImpl(const DataType* query, Computer<SQ8UniformQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<SQ8UniformQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    inline void
    ScanBatchDistImpl(Computer<SQ8UniformQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    inline void
    ReleaseComputerImpl(Computer<SQ8UniformQuantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM;
    }

private:
    DataType lower_bound_{0};
    DataType diff_{0};

    /***
     * code layout: sq-code(fixed) + norm(opt) + sum(opt)
     * for L2 and COSINE, norm is needed for fast computation
     * for IP and COSINE, sum is needed for restoring original distance
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_sum_{0};
};

template <MetricType metric>
SQ8UniformQuantizer<metric>::SQ8UniformQuantizer(int dim, Allocator* allocator)
    : Quantizer<SQ8UniformQuantizer<metric>>(dim, allocator) {
    lower_bound_ = std::numeric_limits<DataType>::max();
    diff_ = std::numeric_limits<DataType>::lowest();

    size_t align_size = 1;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        align_size = std::max(align_size, sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        align_size = std::max(align_size, sizeof(sum_type));
    }
    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += ceil_int(dim, align_size);

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        offset_norm_ = this->code_size_;
        this->code_size_ += ceil_int(sizeof(norm_type), align_size);
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_sum_ = this->code_size_;
        this->code_size_ += ceil_int(sizeof(sum_type), align_size);
    }
}

template <MetricType metric>
SQ8UniformQuantizer<metric>::SQ8UniformQuantizer(const SQ8UniformQuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : SQ8UniformQuantizer<metric>(common_param.dim_, common_param.allocator_.get()){};

template <MetricType metric>
SQ8UniformQuantizer<metric>::SQ8UniformQuantizer(const QuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : SQ8UniformQuantizer<metric>(std::dynamic_pointer_cast<SQ8UniformQuantizerParameter>(param),
                                  common_param){};

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
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

    ScalarQuantizationTrainer trainer(this->dim_, 8);
    trainer.TrainUniform(data, count, this->diff_, this->lower_bound_, need_normalize);

    this->diff_ -= this->lower_bound_;

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    norm_type norm = 0;
    sum_type sum = 0;

    Vector<DataType> norm_data(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_data.resize(this->dim_);
        Normalize(data, norm_data.data(), this->dim_);
        data = norm_data.data();
    }

    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = 1.0f * (data[d] - lower_bound_) / diff_;
        if (delta < 0.0) {
            delta = 0;
        } else if (delta > 0.999) {
            delta = 1;
        }
        scaled = 255 * delta;
        codes[offset_code_ + d] = scaled;

        norm += scaled * scaled;
        sum += data[d];
    }

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        *(norm_type*)(codes + offset_norm_) = norm;
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        *(sum_type*)(codes + offset_sum_) = sum;
    }

    return true;
}

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = codes[d] / 255.0 * diff_ + lower_bound_;
    }
    return true;
}

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
inline float
SQ8UniformQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    float result;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        result = SQ8UniformComputeCodesIP(codes1, codes2, this->dim_);

        norm_type norm1 = *((norm_type*)(codes1 + offset_norm_));
        norm_type norm2 = *((norm_type*)(codes2 + offset_norm_));

        result = norm1 + norm2 - 2 * result;
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        result = SQ8UniformComputeCodesIP(codes1, codes2, this->dim_);

        sum_type sum1 = *((sum_type*)(codes1 + offset_sum_));
        sum_type sum2 = *((sum_type*)(codes2 + offset_sum_));

        result = lower_bound_ * (sum1 + sum2) + (diff_ / 255.0) * (diff_ / 255.0) * result -
                 lower_bound_ * lower_bound_;

        result = 1 - result;

    } else {
        logger::error("unsupported metric type");
        result = 0;
    }
    return result;
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<SQ8UniformQuantizer>& computer) const {
    try {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->code_size_));
        this->EncodeOneImpl(query, computer.buf_);
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::ComputeDistImpl(Computer<SQ8UniformQuantizer>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    dists[0] = this->ComputeImpl(computer.buf_, codes);
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::ScanBatchDistImpl(Computer<SQ8UniformQuantizer<metric>>& computer,
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
SQ8UniformQuantizer<metric>::ReleaseComputerImpl(
    Computer<SQ8UniformQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->diff_);
    StreamWriter::WriteObj(writer, this->lower_bound_);
    StreamWriter::WriteObj(writer, this->offset_code_);
    StreamWriter::WriteObj(writer, this->offset_norm_);
    StreamWriter::WriteObj(writer, this->offset_sum_);
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->diff_);
    StreamReader::ReadObj(reader, this->lower_bound_);
    StreamReader::ReadObj(reader, this->offset_code_);
    StreamReader::ReadObj(reader, this->offset_norm_);
    StreamReader::ReadObj(reader, this->offset_sum_);
}

}  // namespace vsag
