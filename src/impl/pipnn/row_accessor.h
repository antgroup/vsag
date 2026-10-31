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

#include <cstdint>
#include <cstring>

#include "../../data_type.h"
#include "../../simd/bf16_simd.h"
#include "../../simd/fp16_simd.h"

namespace vsag {

/**
 * Thin abstraction that decouples PiPNN row access from the concrete storage type.
 *
 * When the input is FP32 the caller passes RawFloat() pointers and PiPNN pays zero conversion
 * overhead for use sites that opt into the raw pointer.  On non-FP32 types CopyRow() converts
 * a single row into the caller-supplied fp32 buffer using an IEEE-754–compliant decoder.
 *
 * One RowAccessor per PiPNN build replaces a whole-dataset fp32 conversion buffer, keeping
 * the temporary memory at O(1) instead of O(N*dim).
 */
struct RowAccessor {
    const void* data{nullptr};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    uint64_t count_{0};
    uint64_t dim_{0};
    const uint64_t* remap{nullptr};  // input_indices: local_id → storage offset

    [[nodiscard]] uint64_t
    Count() const {
        return count_;
    }

    /// @return unchanged float* for FP32; nullptr for fp16 / bf16.
    [[nodiscard]] const float*
    RawFloat(uint32_t local_id) const {
        if (data_type_ != DataTypes::DATA_TYPE_FLOAT) {
            return nullptr;
        }
        return static_cast<const float*>(data) + static_cast<uint64_t>(local_id) * dim_;
    }

    /// Copy a single vector to a pre-allocated float32 buffer.
    inline void
    CopyRow(uint32_t local_id, float* dst) const {
        const uint64_t storage_idx =
            remap != nullptr ? remap[local_id] : static_cast<uint64_t>(local_id);
        const uint64_t offset = storage_idx * dim_;
        if (data_type_ == DataTypes::DATA_TYPE_FP16) {
            const auto* src = static_cast<const uint16_t*>(data) + offset;
            for (uint64_t d = 0; d < dim_; ++d) {
                uint16_t h = src[d];
                uint32_t sign = (h & 0x8000U) << 16;
                uint32_t exp = (h >> 10) & 0x1FU;
                uint32_t mant = h & 0x3FFU;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) {
                        bits = sign;
                    } else {
                        int e = 1;
                        while ((mant & 0x400U) == 0) {
                            mant <<= 1;
                            --e;
                        }
                        mant &= 0x3FFU;
                        bits = sign | ((static_cast<uint32_t>(e + 127 - 15)) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    bits = sign | 0x7F800000U | (mant << 13);
                } else {
                    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                std::memcpy(&dst[d], &bits, sizeof(bits));
            }
        } else if (data_type_ == DataTypes::DATA_TYPE_BF16) {
            const auto* src = static_cast<const uint16_t*>(data) + offset;
            for (uint64_t d = 0; d < dim_; ++d) {
                dst[d] = generic::BF16ToFloat(src[d]);
            }
        } else {
            const auto* src = static_cast<const float*>(data) + offset;
            std::memcpy(dst, src, dim_ * sizeof(float));
        }
    }

    /// Copy a batch of `count` consecutive vectors.
    inline void
    CopyRows(uint32_t first_local_id, uint32_t count, float* dst) const {
        // NOTE: remap is not used for batch copies — callers using remap should use
        // per-row CopyRow instead.  If remap is set, fall back to per-row.
        if (remap != nullptr) {
            for (uint32_t i = 0; i < count; ++i) {
                CopyRow(first_local_id + i, dst + static_cast<uint64_t>(i) * dim_);
            }
            return;
        }
        const uint64_t offset = static_cast<uint64_t>(first_local_id) * dim_;
        const uint64_t total = static_cast<uint64_t>(count) * dim_;
        if (data_type_ == DataTypes::DATA_TYPE_FP16) {
            const auto* src = static_cast<const uint16_t*>(data) + offset;
            for (uint64_t i = 0; i < total; ++i) {
                uint16_t h = src[i];
                uint32_t sign = (h & 0x8000U) << 16;
                uint32_t exp = (h >> 10) & 0x1FU;
                uint32_t mant = h & 0x3FFU;
                uint32_t bits;
                if (exp == 0) {
                    if (mant == 0) {
                        bits = sign;
                    } else {
                        int e = 1;
                        while ((mant & 0x400U) == 0) {
                            mant <<= 1;
                            --e;
                        }
                        mant &= 0x3FFU;
                        bits = sign | ((static_cast<uint32_t>(e + 127 - 15)) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    bits = sign | 0x7F800000U | (mant << 13);
                } else {
                    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                std::memcpy(&dst[i], &bits, sizeof(bits));
            }
        } else if (data_type_ == DataTypes::DATA_TYPE_BF16) {
            const auto* src = static_cast<const uint16_t*>(data) + offset;
            for (uint64_t i = 0; i < total; ++i) {
                dst[i] = generic::BF16ToFloat(src[i]);
            }
        } else {
            const auto* src = static_cast<const float*>(data) + offset;
            std::memcpy(dst, src, total * sizeof(float));
        }
    }
};

}  // namespace vsag
