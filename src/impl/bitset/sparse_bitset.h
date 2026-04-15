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
 * @file sparse_bitset.h
 * @brief Sparse bitset implementation using Roaring bitmap.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <roaring.hh>
#include <vector>

#include "computable_bitset.h"

namespace vsag {

/**
 * @brief Sparse bitset implementation optimized for sparse bit patterns.
 *
 * This class uses Roaring bitmap internally, which provides efficient
 * storage and operations for sparse bitsets where only a small fraction
 * of bits are set.
 */
class SparseBitset : public ComputableBitset {
public:
    /**
     * @brief Default constructor.
     */
    explicit SparseBitset() : ComputableBitset() {
    }

    /**
     * @brief Default destructor.
     */
    ~SparseBitset() override = default;

    /**
     * @brief Constructs a SparseBitset with an allocator.
     *
     * @param allocator Pointer to the allocator (unused in this implementation).
     */
    explicit SparseBitset(Allocator* allocator) : SparseBitset(){};

    SparseBitset(const SparseBitset&) = delete;
    SparseBitset&
    operator=(const SparseBitset&) = delete;
    SparseBitset(SparseBitset&&) = delete;

public:
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
     * @brief Dumps the bitset to a string representation.
     *
     * @return String representation of the bitset.
     */
    std::string
    Dump() override;

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
     * @brief Gets the memory usage of the bitset.
     *
     * @return Memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override;

private:
    /// Mutex for thread-safe access.
    mutable std::mutex mutex_;
    /// Roaring bitmap storing the bitset data.
    roaring::Roaring r_;
};

}  //namespace vsag