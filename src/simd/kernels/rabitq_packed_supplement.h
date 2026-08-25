// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace vsag::simd {

struct RaBitQPackedSupplementLayout {
    static constexpr uint64_t CHUNK_DIM = 64;

    uint64_t dim{0};
    uint32_t bits{0};

    [[nodiscard]] bool
    UsesCompactChunks() const {
        return bits >= 5U and bits <= 7U;
    }

    [[nodiscard]] uint64_t
    PlaneBytes() const {
        return (dim + 7U) / 8U;
    }

    [[nodiscard]] uint64_t
    FullChunkCount() const {
        return UsesCompactChunks() ? dim / CHUNK_DIM : 0U;
    }

    [[nodiscard]] uint64_t
    FullDimension() const {
        return FullChunkCount() * CHUNK_DIM;
    }

    [[nodiscard]] uint64_t
    ChunkBytes() const {
        return static_cast<uint64_t>(bits) * CHUNK_DIM / 8U;
    }

    [[nodiscard]] uint64_t
    FullChunksSize() const {
        return FullChunkCount() * ChunkBytes();
    }

    [[nodiscard]] uint64_t
    TailDimension() const {
        return dim - FullDimension();
    }

    [[nodiscard]] uint64_t
    TailPlaneBytes() const {
        return (TailDimension() + 7U) / 8U;
    }

    [[nodiscard]] uint64_t
    PackedSize() const {
        return PlaneBytes() * static_cast<uint64_t>(bits);
    }
};

inline uint8_t
RaBitQSupplementMask(uint32_t bits) {
    return bits >= 8U ? 0xFFU : static_cast<uint8_t>((1U << bits) - 1U);
}

inline void
RaBitQPackCompactSupplementChunk(const uint8_t* values, uint8_t* packed, uint32_t bits) {
    if (bits == 5U) {
        for (uint64_t lane = 0; lane < 16U; ++lane) {
            packed[lane] =
                static_cast<uint8_t>((values[lane] & 0x0FU) | ((values[16U + lane] & 0x0FU) << 4U));
            packed[16U + lane] = static_cast<uint8_t>((values[32U + lane] & 0x0FU) |
                                                      ((values[48U + lane] & 0x0FU) << 4U));
        }
        std::memset(packed + 32U, 0, 8U);
        for (uint64_t lane = 0; lane < RaBitQPackedSupplementLayout::CHUNK_DIM; ++lane) {
            packed[32U + lane / 8U] |=
                static_cast<uint8_t>(((values[lane] >> 4U) & 1U) << (lane % 8U));
        }
        return;
    }

    for (uint64_t lane = 0; lane < 16U; ++lane) {
        const uint8_t last = values[48U + lane];
        packed[lane] = static_cast<uint8_t>((values[lane] & 0x3FU) | ((last & 0x03U) << 6U));
        packed[16U + lane] =
            static_cast<uint8_t>((values[16U + lane] & 0x3FU) | (((last >> 2U) & 0x03U) << 6U));
        packed[32U + lane] =
            static_cast<uint8_t>((values[32U + lane] & 0x3FU) | (((last >> 4U) & 0x03U) << 6U));
    }
    if (bits == 7U) {
        std::memset(packed + 48U, 0, 8U);
        for (uint64_t lane = 0; lane < RaBitQPackedSupplementLayout::CHUNK_DIM; ++lane) {
            packed[48U + lane / 8U] |=
                static_cast<uint8_t>(((values[lane] >> 6U) & 1U) << (lane % 8U));
        }
    }
}

inline void
RaBitQUnpackCompactSupplementChunk(const uint8_t* packed, uint8_t* values, uint32_t bits) {
    if (bits == 5U) {
        for (uint64_t lane = 0; lane < 16U; ++lane) {
            values[lane] = packed[lane] & 0x0FU;
            values[16U + lane] = packed[lane] >> 4U;
            values[32U + lane] = packed[16U + lane] & 0x0FU;
            values[48U + lane] = packed[16U + lane] >> 4U;
        }
        for (uint64_t lane = 0; lane < RaBitQPackedSupplementLayout::CHUNK_DIM; ++lane) {
            values[lane] |=
                static_cast<uint8_t>(((packed[32U + lane / 8U] >> (lane % 8U)) & 1U) << 4U);
        }
        return;
    }

    for (uint64_t lane = 0; lane < 16U; ++lane) {
        values[lane] = packed[lane] & 0x3FU;
        values[16U + lane] = packed[16U + lane] & 0x3FU;
        values[32U + lane] = packed[32U + lane] & 0x3FU;
        values[48U + lane] = static_cast<uint8_t>(((packed[lane] >> 6U) & 0x03U) |
                                                  ((packed[16U + lane] >> 4U) & 0x0CU) |
                                                  ((packed[32U + lane] >> 2U) & 0x30U));
    }
    if (bits == 7U) {
        for (uint64_t lane = 0; lane < RaBitQPackedSupplementLayout::CHUNK_DIM; ++lane) {
            values[lane] |=
                static_cast<uint8_t>(((packed[48U + lane / 8U] >> (lane % 8U)) & 1U) << 6U);
        }
    }
}

inline void
RaBitQPackScalarSupplementTail(const uint8_t* scalar_codes,
                               uint8_t* packed_supplement,
                               const RaBitQPackedSupplementLayout& layout) {
    const uint8_t mask = RaBitQSupplementMask(layout.bits);
    const uint64_t tail_begin = layout.FullDimension();
    auto* tail = packed_supplement + layout.FullChunksSize();
    for (uint64_t lane = 0; lane < layout.TailDimension(); ++lane) {
        const uint8_t value = scalar_codes[tail_begin + lane] & mask;
        for (uint32_t bit = 0; bit < layout.bits; ++bit) {
            tail[static_cast<uint64_t>(bit) * layout.TailPlaneBytes() + lane / 8U] |=
                static_cast<uint8_t>(((value >> bit) & 1U) << (lane % 8U));
        }
    }
}

inline void
RaBitQPackScalarSupplementCode(const uint8_t* scalar_codes,
                               uint8_t* packed_supplement,
                               uint64_t dim,
                               uint32_t supplement_bits) {
    const RaBitQPackedSupplementLayout layout{dim, supplement_bits};
    if (layout.PackedSize() == 0U) {
        return;
    }
    std::memset(packed_supplement, 0, layout.PackedSize());

    const uint8_t mask = RaBitQSupplementMask(supplement_bits);
    for (uint64_t chunk = 0; chunk < layout.FullChunkCount(); ++chunk) {
        uint8_t values[RaBitQPackedSupplementLayout::CHUNK_DIM];
        const uint64_t begin = chunk * RaBitQPackedSupplementLayout::CHUNK_DIM;
        for (uint64_t lane = 0; lane < RaBitQPackedSupplementLayout::CHUNK_DIM; ++lane) {
            values[lane] = scalar_codes[begin + lane] & mask;
        }
        RaBitQPackCompactSupplementChunk(
            values, packed_supplement + chunk * layout.ChunkBytes(), supplement_bits);
    }
    RaBitQPackScalarSupplementTail(scalar_codes, packed_supplement, layout);
}

inline void
RaBitQPackScalarFilterPlanesTail(const uint8_t* scalar_codes,
                                 uint8_t* filter_planes,
                                 uint64_t dim,
                                 uint32_t total_bits,
                                 uint32_t filter_bits,
                                 uint64_t begin) {
    const uint64_t plane_bytes = (dim + 7U) / 8U;
    for (uint64_t d = begin; d < dim; ++d) {
        for (uint32_t plane = 0; plane < filter_bits; ++plane) {
            const uint32_t logical_bit = total_bits - plane - 1U;
            filter_planes[static_cast<uint64_t>(plane) * plane_bytes + d / 8U] |=
                static_cast<uint8_t>(((scalar_codes[d] >> logical_bit) & 1U) << (d % 8U));
        }
    }
}

}  // namespace vsag::simd
