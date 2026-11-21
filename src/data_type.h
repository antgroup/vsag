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

#include "vsag/constants.h"

namespace vsag {
enum class DataTypes {
    DATA_TYPE_FLOAT = 0,
    DATA_TYPE_INT8 = 1,
    DATA_TYPE_FP16 = 2,
    DATA_TYPE_BF16 = 3,
    DATA_TYPE_SPARSE = 4,
    // Bit-packed binary vectors: dim counts bits, each row holds ceil(dim / 8) bytes.
    DATA_TYPE_BINARY = 5,
};

/**
 * @brief Number of bytes that hold one bit-packed binary vector with `dim` bits.
 *
 * Bits are packed MSB-first: logical bit i lives in byte i / 8 under mask 0x80 >> (i % 8), which
 * is the layout produced by numpy.packbits with its default bitorder="big". When dim is not a
 * multiple of eight, the low (8 - dim % 8) bits of the last byte are padding.
 */
constexpr uint64_t
BinaryVectorBytes(uint64_t dim) noexcept {
    return (dim + 7) / 8;
}

/**
 * @brief Largest accepted binary dimension.
 *
 * Distances are returned as float, which represents every integer up to 2^24 exactly, so every
 * Hamming distance of a binary index with at most this many bits is an exact integer.
 */
constexpr int64_t MAX_BINARY_DIM = int64_t{1} << 24;

constexpr const char*
ToString(DataTypes t) noexcept {
    switch (t) {
        case DataTypes::DATA_TYPE_FLOAT:
            return DATATYPE_FLOAT32;
        case DataTypes::DATA_TYPE_INT8:
            return DATATYPE_INT8;
        case DataTypes::DATA_TYPE_FP16:
            return DATATYPE_FLOAT16;
        case DataTypes::DATA_TYPE_SPARSE:
            return DATATYPE_SPARSE;
        case DataTypes::DATA_TYPE_BF16:
            return DATATYPE_BFLOAT16;
        case DataTypes::DATA_TYPE_BINARY:
            return DATATYPE_BINARY;
    }
    return "unknown";
}

enum class RecordRepr {
    DENSE = 0,
    SPARSE = 1,
    MULTI_VECTOR = 2,
};

constexpr const char*
ToString(RecordRepr r) noexcept {
    switch (r) {
        case RecordRepr::DENSE:
            return "dense";
        case RecordRepr::SPARSE:
            return "sparse";
        case RecordRepr::MULTI_VECTOR:
            return "multi_vector";
    }
    return "unknown";
}

}  // namespace vsag
