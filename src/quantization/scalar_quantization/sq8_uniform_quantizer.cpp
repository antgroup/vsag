
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

#include "sq8_uniform_quantizer.h"

#include <cstddef>
#include <cstring>

#include "scalar_quantization_trainer.h"
#include "scalar_quantization_utils.h"
#include "simd/normalize.h"
#include "simd/sq8_uniform_simd.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

template <MetricType metric>
SQ8UniformQuantizer<metric>::SQ8UniformQuantizer(int dim, Allocator* allocator)
    : Quantizer<SQ8UniformQuantizer<metric>>(dim, allocator) {
    lower_bound_ = std::numeric_limits<float>::max();
    diff_ = std::numeric_limits<float>::lowest();

    uint64_t align_size = 1;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        align_size = std::max<uint64_t>(align_size, sizeof(norm_type));
    }
    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        align_size = std::max<uint64_t>(align_size, sizeof(sum_type));
    }
    this->code_size_ = 0;

    offset_code_ = this->code_size_;
    this->code_size_ += align_up(dim, static_cast<int64_t>(align_size));

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        offset_norm_ = this->code_size_;
        this->code_size_ += align_up(sizeof(norm_type), static_cast<int64_t>(align_size));
    }

    if constexpr (metric == MetricType::METRIC_TYPE_IP or
                  metric == MetricType::METRIC_TYPE_COSINE) {
        offset_sum_ = this->code_size_;
        this->code_size_ += align_up(sizeof(sum_type), static_cast<int64_t>(align_size));
    }

    this->query_code_size_ = this->code_size_;
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
SQ8UniformQuantizer<metric>::TrainImpl(const float* data, uint64_t count) {
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
    trainer.TrainUniform(
        data, count, this->diff_, this->lower_bound_, need_normalize, SQTrainMode::CLASSIC);

    this->diff_ -= this->lower_bound_;
    this->scalar_rate_ = (diff_ / 255.0) * (diff_ / 255.0);

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
SQ8UniformQuantizer<metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    norm_type norm = 0;
    sum_type sum = 0;

    Vector<float> norm_data(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_data.resize(this->dim_);
        Normalize(data, norm_data.data(), this->dim_);
        data = norm_data.data();
    }

    float inv_diff = diff_ == 0.0F ? 0.0F : 1.0F / diff_;
    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = (data[d] - lower_bound_) * inv_diff;
        delta = ClampScalarQuantizationDelta(delta);
        scaled = static_cast<uint8_t>(255.0F * delta);
        codes[offset_code_ + d] = scaled;

        norm += static_cast<uint64_t>(scaled) * static_cast<uint64_t>(scaled);
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
SQ8UniformQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, float* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = codes[d] / 255.0 * diff_ + lower_bound_;
    }
    return true;
}

template <MetricType metric>
float
SQ8UniformQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    float result;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        result = SQ8UniformComputeCodesIP(codes1, codes2, this->dim_);

        norm_type norm1 = *((norm_type*)(codes1 + offset_norm_));
        norm_type norm2 = *((norm_type*)(codes2 + offset_norm_));

        result =
            (static_cast<float>(norm1) + static_cast<float>(norm2) - 2 * result) * scalar_rate_;
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        result = SQ8UniformComputeCodesIP(codes1, codes2, this->dim_);

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
SQ8UniformQuantizer<metric>::ProcessQueryImpl(const float* query,
                                              Computer<SQ8UniformQuantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
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

// Batch-scan one query against `count` codes laid out with row stride =
// code_size_ (fixed by the quantizer; includes optional trailing norm/sum
// metadata). Computes raw IP via SQ8UniformComputeCodesIPBatch -- a single
// AMX/AVX-512 call -- then folds in the per-row metric correction.
template <MetricType metric>
void
SQ8UniformQuantizer<metric>::ScanBatchDistImpl(Computer<SQ8UniformQuantizer>& computer,
                                               uint64_t count,
                                               const uint8_t* codes,
                                               float* dists) const {
    if (count == 0) {
        return;
    }
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        // Stash raw IPs into the output buffer (sized for `count` floats).
        SQ8UniformComputeCodesIPBatch(
            computer.buf_, codes, this->dim_, count, this->code_size_, dists);

        norm_type query_norm = 0;
        std::memcpy(&query_norm, computer.buf_ + offset_norm_, sizeof(norm_type));
        const auto query_norm_f = static_cast<float>(query_norm);
        for (uint64_t i = 0; i < count; ++i) {
            norm_type code_norm = 0;
            std::memcpy(&code_norm, codes + i * this->code_size_ + offset_norm_, sizeof(norm_type));
            dists[i] =
                (query_norm_f + static_cast<float>(code_norm) - 2.F * dists[i]) * scalar_rate_;
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        SQ8UniformComputeCodesIPBatch(
            computer.buf_, codes, this->dim_, count, this->code_size_, dists);

        sum_type query_sum = 0;
        std::memcpy(&query_sum, computer.buf_ + offset_sum_, sizeof(sum_type));
        const float lb = lower_bound_;
        const float lb2 = lb * lb;
        for (uint64_t i = 0; i < count; ++i) {
            sum_type code_sum = 0;
            std::memcpy(&code_sum, codes + i * this->code_size_ + offset_sum_, sizeof(sum_type));
            const float ip = lb * (query_sum + code_sum) + scalar_rate_ * dists[i] + lb2;
            dists[i] = 1.F - ip;
        }
    } else {
        // Fallback for unsupported metric: per-pair compute. Skip the batch
        // IP call so we don't compute results that would be immediately
        // overwritten.
        for (uint64_t i = 0; i < count; ++i) {
            dists[i] = this->ComputeImpl(computer.buf_, codes + i * this->code_size_);
        }
    }
}

template <MetricType metric>
void
SQ8UniformQuantizer<metric>::ComputeDistsBatch4Impl(Computer<SQ8UniformQuantizer>& computer,
                                                    const uint8_t* codes1,
                                                    const uint8_t* codes2,
                                                    const uint8_t* codes3,
                                                    const uint8_t* codes4,
                                                    float& dists1,
                                                    float& dists2,
                                                    float& dists3,
                                                    float& dists4) const {
    // The 4 codes are not contiguous in memory (HGraph passes pointers
    // from a neighbor list). Calling SQ8UniformComputeCodesIPBatch four
    // times with n_codes=1 throws away the batching benefit, so for
    // batch4 we just delegate to the per-pair ComputeImpl.
    dists1 = this->ComputeImpl(computer.buf_, codes1);
    dists2 = this->ComputeImpl(computer.buf_, codes2);
    dists3 = this->ComputeImpl(computer.buf_, codes3);
    dists4 = this->ComputeImpl(computer.buf_, codes4);
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
    this->scalar_rate_ = (diff_ / 255.0) * (diff_ / 255.0);
}

TEMPLATE_QUANTIZER(SQ8UniformQuantizer)

}  // namespace vsag
