
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

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/bitset.h"

namespace vsag {
class ComputableBitset;
using ComputableBitsetPtr = std::shared_ptr<ComputableBitset>;

enum class ComputableBitsetType { SparseBitset, FastBitset };

/**
 * @brief ComputableBitset is a base class for bitsets that can be computed.
 *
 * @note ComputableBitset is a base class for bitsets that can be computed.
 *       It provides a set of methods that can be used to perform bitwise operations on the bitset.
 */
class ComputableBitset : public Bitset {
public:
    static ComputableBitsetPtr
    MakeInstance(ComputableBitsetType type, Allocator* allocator = nullptr);

    static ComputableBitset*
    MakeRawInstance(ComputableBitsetType type, Allocator* allocator = nullptr);

public:
    ComputableBitset() = default;

    ~ComputableBitset() override = default;

    /**
     * @brief Performs a bitwise OR operation on the current bitset with another bitset.
     *
     * @param another The bitset to perform the OR operation with.
     * @return void
     */
    virtual void
    Or(const ComputableBitset& another) = 0;

    /**
     * @brief Performs a bitwise AND operation on the current bitset with another bitset.
     *
     * @param another The bitset to perform the AND operation with.
     * @return void
     */
    virtual void
    And(const ComputableBitset& another) = 0;

    /**
     * @brief Performs a bitwise NOT operation on the current bitset.
     *
     * @return void
     */
    virtual void
    Not() = 0;

    /**
     * @brief Performs a bitwise OR operation on the current computable bitset with another.
     *
     * @param another The computable pointer to perform the OR operation with.
     * @return void
     */
    virtual void
    Or(const ComputableBitset* another) = 0;

    /**
     * @brief Performs a bitwise AND operation on the current computable bitset with another.
     *
     * @param another The computable pointer to perform the AND operation with.
     * @return void
     */
    virtual void
    And(const ComputableBitset* another) = 0;

    /**
     * @brief Performs a bitwise And operation on the current computable bitset with a vector of other computable bitsets.
     *
     * @param other_bitsets The vector of computable bitsets to perform the And operation with.
     * @return void
     */
    virtual void
    And(const std::vector<const ComputableBitset*>& other_bitsets);

    /**
     * @brief Performs a bitwise OR operation on the current computable bitset with a vector of other computable bitsets.
     *
     * @param other_bitsets The vector of computable bitsets to perform the OR operation with.
     * @return void
     */
    virtual void
    Or(const std::vector<const ComputableBitset*>& other_bitsets);

    /**
     * @brief Serializes the bitset to a stream.
     *
     * @param writer The stream writer to write the serialized bitset to.
     * @return void
     * @note The serialized bitset is written to the stream in a format that can be deserialized by the Deserialize method.
     */
    virtual void
    Serialize(StreamWriter& writer) const = 0;

    /**
     * @brief Deserializes the bitset from a stream.
     *
     * @param reader The stream reader to read the serialized bitset from.
     * @return void
     * @note The serialized bitset is read from the stream and deserialized into the current bitset.
     */
    virtual void
    Deserialize(StreamReader& reader) = 0;

    /**
     * @brief Clear the bitset.
     *
     * @return void
     * @note The bitset is cleared, all bits are set to 0.
     */
    virtual void
    Clear() = 0;

public:
    ComputableBitsetType type_{ComputableBitsetType::SparseBitset};
};

}  // namespace vsag
