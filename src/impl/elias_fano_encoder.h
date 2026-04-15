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

/// @file elias_fano_encoder.h
/// @brief Elias-Fano encoder for compressing ordered adjacency lists.

#pragma once

#include "typing.h"

namespace vsag {

/// @brief Elias-Fano encoder for compressing ordered adjacency lists.
///
/// Elias-Fano encoding is a quasi-succinct representation of monotonically increasing
/// integer sequences. It provides near-optimal space usage while allowing efficient
/// random access and iteration over the encoded values.
///
/// @note Our adjacency list size is mostly no more than 255, so we use uint8_t
///       to store the number of elements.
class EliasFanoEncoder {
public:
    EliasFanoEncoder() = default;

    /// @brief Encodes an ordered sequence of values.
    /// @param values Ordered vector of values to encode.
    /// @param max_value Maximum possible value in the sequence.
    /// @param allocator Allocator for memory management.
    void
    Encode(const Vector<InnerIdType>& values, InnerIdType max_value, Allocator* allocator);

    /// @brief Decompresses all encoded values.
    /// @param neighbors Output vector to store decompressed neighbor IDs.
    void
    DecompressAll(Vector<InnerIdType>& neighbors) const;

    /// @brief Clears the encoder and releases memory.
    /// @param allocator Allocator used for memory deallocation.
    void
    Clear(Allocator* allocator) {
        if (bits != nullptr) {
            allocator->Deallocate(bits);
            bits = nullptr;
        }
        num_elements = 0;
        low_bits_width = 0;
        low_bits_size = 0;
        high_bits_size = 0;
    }

    /// @brief Gets the total size of the encoded data in bytes.
    /// @return Size in bytes including header.
    [[nodiscard]] uint64_t
    SizeInBytes() const {
        return sizeof(EliasFanoEncoder) + (low_bits_size + high_bits_size) * sizeof(uint64_t);
    }

    /// @brief Gets the number of encoded elements.
    /// @return Number of elements (max 255).
    [[nodiscard]] uint8_t
    Size() const {
        return num_elements;
    }

    /// Combined storage for low bits and high bits.
    uint64_t* bits{nullptr};
    /// Number of elements, max 255.
    uint8_t num_elements{0};
    /// Width of low bits.
    uint8_t low_bits_width{0};
    /// Size of low_bits_ array in uint64_t units.
    uint8_t low_bits_size{0};
    /// Size of high_bits_ array in uint64_t units.
    uint8_t high_bits_size{0};

private:
    /// @brief Sets low bits for a given index.
    /// @param index Index to set.
    /// @param value Value to encode in low bits.
    /// @note Requires "const" to pass lint check; modifies the values pointed by bits.
    void
    set_low_bits(uint64_t index, InnerIdType value) const;

    /// @brief Gets low bits for a given index.
    /// @param index Index to query.
    /// @return Low bits value at the index.
    [[nodiscard]] InnerIdType
    get_low_bits(uint64_t index) const;
};

}  // namespace vsag