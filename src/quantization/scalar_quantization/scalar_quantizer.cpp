
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

#include "scalar_quantizer.h"

#include "scalar_quantization_trainer.h"
#include "simd/normalize.h"
#include "simd/sq4_simd.h"
#include "simd/sq8_simd.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

template <MetricType metric, int bit>
ScalarQuantizer<metric, bit>::ScalarQuantizer(int dim, Allocator* allocator)
    : Quantizer<ScalarQuantizer<metric, bit>>(dim, allocator) {
    auto bit_count = static_cast<int64_t>(dim) * static_cast<int64_t>(BIT_PER_DIM);
    this->code_size_ = ceil_int(bit_count, 8);
    this->query_code_size_ = this->dim_ * sizeof(float);
    this->metric_ = metric;
    lower_bound_.resize(dim, std::numeric_limits<DataType>::max());
    diff_.resize(dim, std::numeric_limits<DataType>::lowest());
}

template <MetricType metric, int bit>
ScalarQuantizer<metric, bit>::ScalarQuantizer(
    const std::shared_ptr<ScalarQuantizerParameter<bit>>& param,
    const IndexCommonParam& common_param)
    : ScalarQuantizer<metric, bit>(common_param.dim_, common_param.allocator_.get()){};

template <MetricType metric, int bit>
ScalarQuantizer<metric, bit>::ScalarQuantizer(const QuantizerParamPtr& param,
                                              const IndexCommonParam& common_param)
    : ScalarQuantizer<metric, bit>(std::dynamic_pointer_cast<ScalarQuantizerParameter<bit>>(param),
                                   common_param){};

template <MetricType metric, int bit>
bool
ScalarQuantizer<metric, bit>::TrainImpl(const DataType* data, uint64_t count) {
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

    ScalarQuantizationTrainer trainer(this->dim_, BIT_PER_DIM);
    trainer.Train(data, count, this->diff_.data(), this->lower_bound_.data(), need_normalize);

    for (uint64_t i = 0; i < this->dim_; ++i) {
        this->diff_[i] -= this->lower_bound_[i];
    }
    this->is_trained_ = true;
    return true;
}

static inline void
fill_codes(uint8_t* codes, uint8_t value, uint8_t value_bit_size, uint64_t index) {
    // fill one value to codes, in index * value_bit_size, value is in [0, 2^value_bit_size)
    // value_bit_size must be in {1, 2, 4, 8}
    auto idx = (index * value_bit_size) / 8;
    auto offset = (index * value_bit_size) % 8;
    codes[idx] |= value << offset;
}

template <MetricType metric, int bit>
bool
ScalarQuantizer<metric, bit>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    float delta = 0;
    uint8_t scaled = 0;
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    memset(codes, 0, this->code_size_);
    for (uint64_t d = 0; d < this->dim_; d++) {
        delta = 1.0F * (cur[d] - lower_bound_[d]) / diff_[d];
        if (delta < 0.0F) {
            delta = 0;
        } else if (delta > 0.999F) {
            delta = 1;
        }
        scaled = static_cast<uint8_t>(static_cast<float>(MAX_CODE_PER_DIM - 1) * delta);
        fill_codes(codes, scaled, BIT_PER_DIM, d);
    }

    return true;
}

template <MetricType metric, int bit>
bool
ScalarQuantizer<metric, bit>::EncodeBatchImpl(const DataType* data,
                                              uint8_t* codes,
                                              uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric, int bit>
bool
ScalarQuantizer<metric, bit>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (uint64_t d = 0; d < this->dim_; d++) {
        auto idx = (d * BIT_PER_DIM) / 8;
        auto offset = (d * BIT_PER_DIM) % 8;
        data[d] = static_cast<float>((codes[idx] >> offset) & ((1 << BIT_PER_DIM) - 1)) /
                      (static_cast<float>(MAX_CODE_PER_DIM) - 1.0F) * diff_[d] +
                  lower_bound_[d];
    }

    return true;
}

template <MetricType metric, int bit>
bool
ScalarQuantizer<metric, bit>::DecodeBatchImpl(const uint8_t* codes,
                                              DataType* data,
                                              uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric, int bit>
inline float
ScalarQuantizer<metric, bit>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    static_assert(bit == 4 || bit == 8, "bit must be 4 or 8");
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        if constexpr (bit == 8) {
            return SQ8ComputeCodesL2Sqr(
                codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
        } else if constexpr (bit == 4) {
            return SQ4ComputeCodesL2Sqr(
                codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        if constexpr (bit == 8) {
            return 1 -
                   SQ8ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
        } else if constexpr (bit == 4) {
            return 1 -
                   SQ4ComputeCodesIP(codes1, codes2, lower_bound_.data(), diff_.data(), this->dim_);
        }
    } else {
        return 0;
    }
}

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::ProcessQueryImpl(const DataType* query,
                                               Computer<ScalarQuantizer>& computer) const {
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

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::ComputeDistImpl(Computer<ScalarQuantizer>& computer,
                                              const uint8_t* codes,
                                              float* dists) const {
    static_assert(bit == 4 || bit == 8, "bit must be 4 or 8");
    auto* buf = reinterpret_cast<float*>(computer.buf_);
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        if constexpr (bit == 8) {
            dists[0] = SQ8ComputeL2Sqr(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
        } else if constexpr (bit == 4) {
            dists[0] = SQ4ComputeL2Sqr(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
        }
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        if constexpr (bit == 8) {
            dists[0] = 1 - SQ8ComputeIP(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
        } else if constexpr (bit == 4) {
            dists[0] = 1 - SQ4ComputeIP(buf, codes, lower_bound_.data(), diff_.data(), this->dim_);
        }
    } else {
        logger::error("unsupported metric type");
        dists[0] = 0;
    }
}

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::ScanBatchDistImpl(Computer<ScalarQuantizer<metric, bit>>& computer,
                                                uint64_t count,
                                                const uint8_t* codes,
                                                float* dists) const {
    // TODO(LHT): Optimize batch for simd
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::ReleaseComputerImpl(
    Computer<ScalarQuantizer<metric, bit>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->diff_);
    StreamWriter::WriteVector(writer, this->lower_bound_);
}

template <MetricType metric, int bit>
void
ScalarQuantizer<metric, bit>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->diff_);
    StreamReader::ReadVector(reader, this->lower_bound_);
}

template class ScalarQuantizer<MetricType::METRIC_TYPE_L2SQR, 8>;
template class ScalarQuantizer<MetricType::METRIC_TYPE_IP, 8>;
template class ScalarQuantizer<MetricType::METRIC_TYPE_COSINE, 8>;
template class ScalarQuantizer<MetricType::METRIC_TYPE_L2SQR, 4>;
template class ScalarQuantizer<MetricType::METRIC_TYPE_IP, 4>;
template class ScalarQuantizer<MetricType::METRIC_TYPE_COSINE, 4>;
}  // namespace vsag
