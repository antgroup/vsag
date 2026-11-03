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

#include "binary_quantizer.h"

#include <cstring>

#include "simd/bit_simd.h"

namespace vsag {

BinaryQuantizer::BinaryQuantizer(int dim, Allocator* allocator)
    : Quantizer<BinaryQuantizer>(dim, allocator) {
    this->code_size_ = (static_cast<uint64_t>(dim) + 7U) / 8U;
    this->query_code_size_ = this->code_size_;
    this->metric_ = MetricType::METRIC_TYPE_HAMMING;
}

BinaryQuantizer::BinaryQuantizer(const BinaryQuantizerParamPtr& param,
                                 const IndexCommonParam& common_param)
    : BinaryQuantizer(common_param.dim_, common_param.allocator_.get()) {
}

BinaryQuantizer::BinaryQuantizer(const QuantizerParamPtr& param,
                                 const IndexCommonParam& common_param)
    : BinaryQuantizer(common_param.dim_, common_param.allocator_.get()) {
}

void
BinaryQuantizer::MaskUnusedBits(uint8_t* codes) const {
    if (codes == nullptr or this->dim_ == 0) {
        return;
    }
    const uint64_t used_bits = this->dim_ % 8U;
    if (used_bits == 0) {
        return;
    }
    codes[this->code_size_ - 1U] &= static_cast<uint8_t>((1U << used_bits) - 1U);
}

bool
BinaryQuantizer::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

bool
BinaryQuantizer::EncodeOneImpl(const float* data, uint8_t* codes) {
    std::memcpy(codes, static_cast<const void*>(data), this->code_size_);
    this->MaskUnusedBits(codes);
    return true;
}

bool
BinaryQuantizer::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    const auto* source = reinterpret_cast<const uint8_t*>(data);
    for (uint64_t i = 0; i < count; ++i) {
        std::memcpy(codes + i * this->code_size_, source + i * this->code_size_, this->code_size_);
        this->MaskUnusedBits(codes + i * this->code_size_);
    }
    return true;
}

bool
BinaryQuantizer::DecodeOneImpl(const uint8_t* codes, float* data) {
    for (uint64_t d = 0; d < this->dim_; ++d) {
        data[d] = static_cast<float>((codes[d / 8U] >> (d % 8U)) & 1U);
    }
    return true;
}

bool
BinaryQuantizer::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

float
BinaryQuantizer::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    return static_cast<float>(HammingDistance(codes1, codes2, this->code_size_));
}

void
BinaryQuantizer::ProcessQueryImpl(const float* query, Computer<BinaryQuantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr and this->query_code_size_ > 0) {
            computer.buf_ =
                static_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    if (this->query_code_size_ == 0) {
        return;
    }
    std::memcpy(computer.buf_, static_cast<const void*>(query), this->query_code_size_);
    this->MaskUnusedBits(computer.buf_);
}

void
BinaryQuantizer::ComputeDistImpl(Computer<BinaryQuantizer>& computer,
                                 const uint8_t* codes,
                                 float* dists) const {
    *dists = static_cast<float>(HammingDistance(codes, computer.buf_, this->code_size_));
}

void
BinaryQuantizer::ScanBatchDistImpl(Computer<BinaryQuantizer>& computer,
                                   uint64_t count,
                                   const uint8_t* codes,
                                   float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

void
BinaryQuantizer::ComputeDistsBatch4Impl(Computer<BinaryQuantizer>& computer,
                                        const uint8_t* codes1,
                                        const uint8_t* codes2,
                                        const uint8_t* codes3,
                                        const uint8_t* codes4,
                                        float& dists1,
                                        float& dists2,
                                        float& dists3,
                                        float& dists4) const {
    dists1 = static_cast<float>(HammingDistance(codes1, computer.buf_, this->code_size_));
    dists2 = static_cast<float>(HammingDistance(codes2, computer.buf_, this->code_size_));
    dists3 = static_cast<float>(HammingDistance(codes3, computer.buf_, this->code_size_));
    dists4 = static_cast<float>(HammingDistance(codes4, computer.buf_, this->code_size_));
}

void
BinaryQuantizer::ReleaseComputerImpl(Computer<BinaryQuantizer>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag