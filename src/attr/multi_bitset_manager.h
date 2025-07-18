
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

#include "impl/bitset/computable_bitset.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"

namespace vsag {
/**
 * @class MultiBitsetManager
 * @brief Manages multiple ComputableBitset instances.
 * 
 * This class provides an interface to manage a collection of ComputableBitset objects.
 * It allows for creation, retrieval, and count modification of the bitsets.
 */
class MultiBitsetManager {
public:
    /**
     * @brief Constructs a MultiBitsetManager with a specified allocator, count, and bitset type.
     * 
     * @param allocator A pointer to the allocator used for memory management.
     * @param count The initial number of ComputableBitset instances to create.
     * @param bitset_type The type of ComputableBitset instances to create. Defaults to ComputableBitsetType::FastBitset.
     */
    explicit MultiBitsetManager(Allocator* allocator,
                                uint64_t count,
                                ComputableBitsetType bitset_type);

    /**
     * @brief Constructs a MultiBitsetManager with a specified allocator and count.
     * 
     * @param allocator A pointer to the allocator used for memory management.
     * @param count The initial number of ComputableBitset instances to create.
     */
    explicit MultiBitsetManager(Allocator* allocator, uint64_t count);

    /**
         * @brief Constructs a MultiBitsetManager with a specified allocator and default count.
         * 
         * @param allocator A pointer to the allocator used for memory management.
         */
    explicit MultiBitsetManager(Allocator* allocator);

    /**
     * @brief Virtual destructor for the MultiBitsetManager.
     * 
     * Ensures proper cleanup of resources when the object is destroyed.
     */
    virtual ~MultiBitsetManager();

    /**
     * @brief Sets a new count for the number of ComputableBitset instances. If new count is less
     *        than origin count, this function will do nothing.
     * 
     * @param new_count The new number of ComputableBitset instances to manage.
     */
    void
    SetNewCount(uint64_t new_count);

    /**
     * @brief Retrieves a pointer to a ComputableBitset instance by its ID.
     * 
     * @param id The ID of the ComputableBitset instance to retrieve.
     * @return A pointer to the ComputableBitset instance, or nullptr if the ID is invalid.
     */
    ComputableBitset*
    GetOneBitset(uint64_t id) const;

    /**
     * @brief Inserts a value into a ComputableBitset instance at the specified offset.
     * 
     * @param id The ID of the ComputableBitset instance to insert the value into.
     * @param offset The offset within the ComputableBitset instance where the value should be inserted.
     * @param value The value to insert into the ComputableBitset instance. Defaults to true.
     */
    void
    InsertValue(uint64_t id, uint64_t offset, bool value = true);

    /**
     * @brief Serializes the MultiBitsetManager to a StreamWriter.
     * 
     * @param writer The StreamWriter to serialize the MultiBitsetManager to.
     */
    void
    Serialize(StreamWriter& writer);

    /**
     * @brief Deserializes the MultiBitsetManager from a StreamReader.
     * 
     * @param reader The StreamReader to deserialize the MultiBitsetManager from.
     */
    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader);

    ComputableBitsetType
    GetBitsetType() const {
        return this->bitset_type_;
    }

private:
    /// A vector containing pointers to ComputableBitset instances.
    Vector<ComputableBitset*> bitsets_;

    /// The current number of ComputableBitset instances managed by this object. Defaults to 1.
    uint64_t count_{1};

    /// A constant pointer to the allocator used for memory management. Initialized to nullptr.
    Allocator* const allocator_{nullptr};

    /// The type of ComputableBitset instances managed by this object. Defaults to ComputableBitsetType::FastBitset.
    const ComputableBitsetType bitset_type_{ComputableBitsetType::FastBitset};
};

}  // namespace vsag
