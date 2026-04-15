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

/**
 * @file fast_bitset.h
 * @brief Fast bitset implementation using contiguous memory array.
 */

#pragma once

#include <shared_mutex>

#include "computable_bitset.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class Allocator;
DEFINE_POINTER(FastBitset);

/**
 * @brief Fast bitset implementation using a contiguous memory array.
 *
 * This class provides a bitset implementation optimized for dense bit
 * patterns, using a raw array of 64-bit integers for efficient storage
 * and bitwise operations.
 */
class FastBitset : public ComputableBitset {
public:
    /**
     * @brief Constructs a FastBitset with an allocator.
     *
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit FastBitset(Allocator* allocator)
        : ComputableBitset(), data_(nullptr), size_(0), capacity_(0){};

    /**
     * @brief Destructor that frees the allocated memory.
     */
    ~FastBitset() override {
        delete[] data_;
        data_ = nullptr;
    }

    /**
     * @brief Sets a bit at the specified position.
     *
     * @param pos The position of the bit to set.
     * @param value The value to set (true for 1, false for 0).
     */
    void
    Set(int64_t pos, bool value) override;

    /**
     * @brief Tests the bit at the specified position.
     *
     * @param pos The position of the bit to test.
     * @return true if the bit is set, false otherwise.
     */
    bool
    Test(int64_t pos) const override;

    /**
     * @brief Counts the number of set bits.
     *
     * @return The number of bits set to 1.
     */
    uint64_t
    Count() override;

    /**
     * @brief Performs bitwise OR with another bitset.
     *
     * @param another The bitset to OR with.
     */
    void
    Or(const ComputableBitset& another) override;

    /**
     * @brief Performs bitwise AND with another bitset.
     *
     * @param another The bitset to AND with.
     */
    void
    And(const ComputableBitset& another) override;

    /**
     * @brief Performs bitwise OR with another bitset pointer.
     *
     * @param another Pointer to the bitset to OR with.
     */
    void
    Or(const ComputableBitset* another) override;

    /**
     * @brief Performs bitwise AND with another bitset pointer.
     *
     * @param another Pointer to the bitset to AND with.
     */
    void
    And(const ComputableBitset* another) override;

    /**
     * @brief Performs bitwise AND with multiple bitsets.
     *
     * @param other_bitsets Vector of bitsets to AND with.
     */
    void
    And(const std::vector<const ComputableBitset*>& other_bitsets) override;

    /**
     * @brief Performs bitwise OR with multiple bitsets.
     *
     * @param other_bitsets Vector of bitsets to OR with.
     */
    void
    Or(const std::vector<const ComputableBitset*>& other_bitsets) override;

    /**
     * @brief Performs bitwise NOT (inverts all bits).
     */
    void
    Not() override;

    /**
     * @brief Serializes the bitset to a stream.
     *
     * @param writer The stream writer to write to.
     */
    void
    Serialize(StreamWriter& writer) const override;

    /**
     * @brief Deserializes the bitset from a stream.
     *
     * @param reader The stream reader to read from.
     */
    void
    Deserialize(StreamReader& reader) override;

    /**
     * @brief Clears all bits in the bitset.
     */
    void
    Clear() override;

    /**
     * @brief Dumps the bitset to a string representation.
     *
     * @return String representation of the bitset.
     */
    std::string
    Dump() override;

    /**
     * @brief Gets the memory usage of the bitset.
     *
     * @return Memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override;

private:
    /**
     * @brief Resizes the bitset to a new size.
     *
     * @param new_size The new size in number of 64-bit words.
     * @param fill The fill value for new words (default 0).
     */
    void
    resize(uint32_t new_size, uint64_t fill = 0);

    /**
     * @brief Gets the fill bit value.
     *
     * @return The fill bit value stored in capacity_.
     */
    constexpr bool
    get_fill_bit() const {
        return (capacity_ >> 31) & 1;
    }

    /**
     * @brief Sets the fill bit value.
     *
     * @param value The fill bit value to set.
     */
    constexpr void
    set_fill_bit(bool value) {
        if (value) {
            capacity_ |= (1UL << 31);
        } else {
            capacity_ &= ~(1UL << 31);
        }
    }

    /**
     * @brief Gets the actual capacity (without fill bit).
     *
     * @return The capacity in number of 64-bit words.
     */
    constexpr uint32_t
    get_capacity() const {
        return capacity_ & 0x7FFFFFFF;
    }

    /**
     * @brief Sets the capacity (preserving fill bit).
     *
     * @param cap The capacity to set in number of 64-bit words.
     */
    constexpr void
    set_capacity(uint32_t cap) {
        capacity_ = (capacity_ & (1UL << 31)) | (cap & 0x7FFFFFFF);
    }

private:
    /// Pointer to the data array (64-bit words).
    uint64_t* data_{nullptr};

    /// Current size in number of 64-bit words.
    uint32_t size_{0};

    /// Capacity with fill bit stored in bit 31.
    uint32_t capacity_{0};
};

}  // namespace vsag