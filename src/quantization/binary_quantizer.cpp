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

#include "quantization/binary_quantizer.h"

#include <cstring>
#include <new>

#include "data_type.h"
#include "simd/hamming_simd.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

// Distances never exceed dim < 2^63: converting through int64_t gives the same float as the
// unsigned conversion and compiles to one instruction on x86-64.
inline float
to_float(uint64_t distance) {
    return static_cast<float>(static_cast<int64_t>(distance));
}

}  // namespace

BinaryQuantizer::BinaryQuantizer(int dim, Allocator* allocator)
    : Quantizer<BinaryQuantizer>(dim, allocator) {
    this->code_size_ = BinaryVectorBytes(static_cast<uint64_t>(dim));
    this->query_code_size_ = this->code_size_;
    this->metric_ = MetricType::METRIC_TYPE_HAMMING;
}

BinaryQuantizer::BinaryQuantizer(const BinaryQuantizerParamPtr& param,
                                 const IndexCommonParam& common_param)
    : BinaryQuantizer(static_cast<int>(common_param.dim_), common_param.allocator_.get()) {
    (void)param;
}

BinaryQuantizer::BinaryQuantizer(const QuantizerParamPtr& param,
                                 const IndexCommonParam& common_param)
    : BinaryQuantizer(std::dynamic_pointer_cast<BinaryQuantizerParameter>(param), common_param) {
}

void
BinaryQuantizer::copy_canonical(const uint8_t* src, uint8_t* dst) const {
    std::memcpy(dst, src, this->code_size_);
    const auto tail_bits = this->dim_ % 8;
    if (tail_bits != 0) {
        // MSB-first packing: the valid bits of the last byte are its high `tail_bits` bits.
        dst[this->code_size_ - 1] &= static_cast<uint8_t>(0xFF00U >> tail_bits);
    }
}

bool
BinaryQuantizer::TrainImpl(const float* data, uint64_t count) {
    (void)data;
    (void)count;
    this->is_trained_ = true;
    return true;
}

bool
BinaryQuantizer::EncodeOneImpl(const float* data, uint8_t* codes) {
    copy_canonical(reinterpret_cast<const uint8_t*>(data), codes);
    return true;
}

bool
BinaryQuantizer::EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) {
    // Source rows are packed back to back, so the input stride equals the code size.
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    for (uint64_t i = 0; i < count; ++i) {
        copy_canonical(bytes + i * this->code_size_, codes + i * this->code_size_);
    }
    return true;
}

bool
BinaryQuantizer::DecodeOneImpl(const uint8_t* codes, float* data) {
    std::memcpy(data, codes, this->code_size_);
    return true;
}

bool
BinaryQuantizer::DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
    std::memcpy(data, codes, count * this->code_size_);
    return true;
}

float
BinaryQuantizer::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    return to_float(HammingDistance(codes1, codes2, this->dim_));
}

void
BinaryQuantizer::ProcessQueryImpl(const float* query, Computer<BinaryQuantizer>& computer) const {
    try {
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
    copy_canonical(reinterpret_cast<const uint8_t*>(query), computer.buf_);
}

void
BinaryQuantizer::ComputeDistImpl(Computer<BinaryQuantizer>& computer,
                                 const uint8_t* codes,
                                 float* dists) const {
    *dists = to_float(HammingDistance(computer.buf_, codes, this->dim_));
}

void
BinaryQuantizer::ScanBatchDistImpl(Computer<BinaryQuantizer>& computer,
                                   uint64_t count,
                                   const uint8_t* codes,
                                   float* dists) const {
    // Codes are stored back to back with code_size_ == ceil(dim_ / 8), the layout of the scan.
    HammingDistanceScan(computer.buf_, codes, this->dim_, count, dists);
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
    uint64_t results[4];
    HammingDistanceBatch4(computer.buf_, codes1, codes2, codes3, codes4, this->dim_, results);
    dists1 = to_float(results[0]);
    dists2 = to_float(results[1]);
    dists3 = to_float(results[2]);
    dists4 = to_float(results[3]);
}

void
BinaryQuantizer::ReleaseComputerImpl(Computer<BinaryQuantizer>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag
