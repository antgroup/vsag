
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

#include "fp16_quantizer.h"

#include "byte_buffer.h"
#include "simd/fp16_simd.h"
#include "simd/normalize.h"
#include "typing.h"

namespace vsag {

template <MetricType metric>
FP16Quantizer<metric>::FP16Quantizer(int dim, Allocator* allocator)
    : Quantizer<FP16Quantizer<metric>>(dim, allocator) {
    this->code_size_ = dim * 2;
    this->query_code_size_ = this->code_size_;
    this->metric_ = metric;
}

template <MetricType metric>
FP16Quantizer<metric>::FP16Quantizer(const FP16QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP16Quantizer<metric>(common_param.dim_, common_param.allocator_.get()){};

template <MetricType metric>
FP16Quantizer<metric>::FP16Quantizer(const QuantizerParamPtr& param,
                                     const IndexCommonParam& common_param)
    : FP16Quantizer<metric>(std::dynamic_pointer_cast<FP16QuantizerParameter>(param),
                            common_param){};

template <MetricType metric>
bool
FP16Quantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    return data != nullptr;
}

template <MetricType metric>
bool
FP16Quantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    auto* codes_fp16 = reinterpret_cast<uint16_t*>(codes);
    for (int i = 0; i < this->dim_; ++i) {
        codes_fp16[i] = generic::FloatToFP16(cur[i]);
    }

    return true;
}

template <MetricType metric>
bool
FP16Quantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
FP16Quantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    const auto* codes_fp16 = reinterpret_cast<const uint16_t*>(codes);

    for (uint64_t d = 0; d < this->dim_; d++) {
        data[d] = generic::FP16ToFloat(codes_fp16[d]);
    }
    return true;
}

template <MetricType metric>
bool
FP16Quantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
float
FP16Quantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP16ComputeL2Sqr(codes1, codes2, this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        return 1 - FP16ComputeIP(codes1, codes2, this->dim_);
    } else {
        return 0;
    }
}

template <MetricType metric>
void
FP16Quantizer<metric>::ProcessQueryImpl(const DataType* query,
                                        Computer<FP16Quantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        this->EncodeOneImpl(query, computer.buf_);
    } catch (std::bad_alloc& e) {
        throw VsagException(
            ErrorType::INTERNAL_ERROR, "bad alloc when init computer buf", e.what());
    }
}

template <MetricType metric>
void
FP16Quantizer<metric>::ComputeDistImpl(Computer<FP16Quantizer>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    auto* buf = computer.buf_;
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = FP16ComputeL2Sqr(buf, codes, this->dim_);
    } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                         metric == MetricType::METRIC_TYPE_COSINE) {
        dists[0] = 1 - FP16ComputeIP(buf, codes, this->dim_);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "unsupported metric type");
    }
}

template <MetricType metric>
void
FP16Quantizer<metric>::ScanBatchDistImpl(Computer<FP16Quantizer<metric>>& computer,
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
FP16Quantizer<metric>::ReleaseComputerImpl(Computer<FP16Quantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

TEMPLATE_QUANTIZER(FP16Quantizer)

}  // namespace vsag
