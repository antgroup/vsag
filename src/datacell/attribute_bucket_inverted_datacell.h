
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
 * @file attribute_bucket_inverted_datacell.h
 * @brief Attribute bucket inverted data cell implementation for attribute filtering.
 *
 * This file provides the AttributeBucketInvertedDataCell class which implements
 * the AttributeInvertedInterface for managing inverted index structures
 * organized by buckets, supporting efficient attribute-based filtering.
 */

#pragma once

#include <memory>
#include <shared_mutex>

#include "attr/attr_value_map.h"
#include "attribute_inverted_interface.h"
#include "vsag_exception.h"

namespace vsag {

/**
 * @brief Attribute bucket inverted data cell for attribute-based filtering.
 *
 * This class implements AttributeInvertedInterface and provides functionality for:
 * - Managing inverted index structures organized by buckets
 * - Efficient attribute-based filtering using bitsets
 * - Supporting insert, update, and query operations for attributes
 */
class AttributeBucketInvertedDataCell : public AttributeInvertedInterface {
public:
    /**
     * @brief Constructs an AttributeBucketInvertedDataCell.
     * @param allocator The allocator for memory management.
     * @param bitset_type The type of computable bitset to use.
     */
    AttributeBucketInvertedDataCell(
        Allocator* allocator, ComputableBitsetType bitset_type = ComputableBitsetType::FastBitset)
        : AttributeInvertedInterface(allocator, bitset_type), field_2_value_map_(allocator){};

    ~AttributeBucketInvertedDataCell() override = default;

    /**
     * @brief Inserts attribute set for a vector.
     * @param attr_set The attribute set to insert.
     * @param inner_id The internal ID of the vector.
     * @param bucket_id The bucket ID for the vector.
     */
    void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id, BucketIdType bucket_id) override;

    /**
     * @brief Gets bitsets for a given attribute.
     * @param attr The attribute to query.
     * @return Vector of pointers to multi-bitset managers.
     */
    std::vector<const MultiBitsetManager*>
    GetBitsetsByAttr(const Attribute& attr) override;

    /**
     * @brief Updates bitsets for attributes.
     * @param attributes The new attribute set.
     * @param offset_id The offset ID within the bucket.
     * @param bucket_id The bucket ID.
     */
    void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id) override;

    /**
     * @brief Updates bitsets for attributes with original attributes.
     * @param attributes The new attribute set.
     * @param offset_id The offset ID within the bucket.
     * @param bucket_id The bucket ID.
     * @param origin_attributes The original attribute set being replaced.
     */
    void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id,
                        const AttributeSet& origin_attributes) override;

    /**
     * @brief Serializes the data cell to a stream.
     * @param writer The stream writer for output.
     */
    void
    Serialize(StreamWriter& writer) override;

    /**
     * @brief Deserializes the data cell from a stream.
     * @param reader The stream reader for input.
     */
    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

    /**
     * @brief Gets the attribute set for a vector.
     * @param bucket_id The bucket ID.
     * @param inner_id The internal ID within the bucket.
     * @param attr Output pointer to store the attribute set.
     */
    void
    GetAttribute(BucketIdType bucket_id, InnerIdType inner_id, AttributeSet* attr) override;

    /**
     * @brief Gets the memory usage of this data cell.
     * @return The memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const override;

private:
    /// Mapping from field name to value map
    UnorderedMap<std::string, ValueMapPtr> field_2_value_map_;

    /// Global mutex for thread-safe access
    std::shared_mutex global_mutex_{};
};

}  // namespace vsag
