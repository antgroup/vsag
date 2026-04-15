/**
 * @file attribute_inverted_interface.h
 * @brief Attribute inverted interface for attribute-based filtering.
 *
 * This file defines the abstract interface for managing attribute-based
 * inverted indexes that enable efficient filtering of vectors by attributes.
 */

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

#include <memory>

#include "attr/attr_type_schema.h"
#include "attr/multi_bitset_manager.h"
#include "attribute_inverted_interface_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/attribute.h"
#include "vsag_exception.h"

namespace vsag {
DEFINE_POINTER2(AttrInvertedInterface, AttributeInvertedInterface);

/**
 * @brief Abstract interface for attribute-based inverted indexing.
 *
 * AttributeInvertedInterface provides operations for managing attribute
 * sets associated with vectors, enabling efficient filtering during search
 * operations using bitset-based inverted indexes.
 */
class AttributeInvertedInterface {
public:
    /**
     * @brief Creates an AttributeInvertedInterface instance.
     *
     * @param allocator Memory allocator for the instance.
     * @param have_bucket Whether the index uses buckets.
     * @return Shared pointer to the created instance.
     */
    static AttrInvertedInterfacePtr
    MakeInstance(Allocator* allocator, bool have_bucket = false);

    /**
     * @brief Creates an AttributeInvertedInterface instance with parameters.
     *
     * @param allocator Memory allocator for the instance.
     * @param param Configuration parameters.
     * @return Shared pointer to the created instance.
     */
    static AttrInvertedInterfacePtr
    MakeInstance(Allocator* allocator, const AttributeInvertedInterfaceParamPtr& param);

public:
    /**
     * @brief Constructs the interface with allocator and bitset type.
     *
     * @param allocator Memory allocator for the instance.
     * @param bitset_type Type of computable bitset to use.
     */
    AttributeInvertedInterface(Allocator* allocator, ComputableBitsetType bitset_type)
        : allocator_(allocator), field_type_map_(allocator), bitset_type_(bitset_type){};

    virtual ~AttributeInvertedInterface() = default;

    /**
     * @brief Inserts attributes for a vector (without bucket).
     *
     * @param attr_set Set of attributes to insert.
     * @param inner_id Internal ID of the vector.
     */
    virtual void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id) {
        this->Insert(attr_set, inner_id, 0);
    }

    /**
     * @brief Inserts attributes for a vector with bucket ID.
     *
     * @param attr_set Set of attributes to insert.
     * @param inner_id Internal ID of the vector.
     * @param bucket_id Bucket ID for the vector.
     */
    virtual void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id, BucketIdType bucket_id) = 0;

    /**
     * @brief Gets bitsets for filtering by a specific attribute.
     *
     * @param attr The attribute to filter by.
     * @return Vector of multi-bitset managers for the attribute.
     */
    virtual std::vector<const MultiBitsetManager*>
    GetBitsetsByAttr(const Attribute& attr) = 0;

    /**
     * @brief Updates bitsets for attributes of a vector.
     *
     * @param attributes New attribute set for the vector.
     * @param offset_id Internal ID of the vector.
     * @param bucket_id Bucket ID for the vector.
     */
    virtual void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id) = 0;

    /**
     * @brief Updates bitsets for attributes with origin attributes.
     *
     * @param attributes New attribute set for the vector.
     * @param offset_id Internal ID of the vector.
     * @param bucket_id Bucket ID for the vector.
     * @param origin_attributes Original attribute set being replaced.
     */
    virtual void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id,
                        const AttributeSet& origin_attributes) = 0;

    /**
     * @brief Gets the attribute set for a vector.
     *
     * @param bucket_id Bucket ID of the vector.
     * @param inner_id Internal ID of the vector.
     * @param attr Output attribute set.
     */
    virtual void
    GetAttribute(BucketIdType bucket_id, InnerIdType inner_id, AttributeSet* attr) = 0;

    /**
     * @brief Serializes the inverted index to a stream writer.
     *
     * @param writer The stream writer for output.
     */
    virtual void
    Serialize(StreamWriter& writer) {
        this->field_type_map_.Serialize(writer);
    }

    /**
     * @brief Deserializes the inverted index from a stream reader.
     *
     * @param reader The stream reader for input.
     */
    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        this->field_type_map_.Deserialize(reader);
    }

    /**
     * @brief Gets the value type of a field.
     *
     * @param field_name Name of the field to query.
     * @return The attribute value type.
     */
    AttrValueType
    GetTypeOfField(const std::string& field_name) {
        return this->field_type_map_.GetTypeOfField(field_name);
    }

    /**
     * @brief Gets the type of computable bitset being used.
     *
     * @return The bitset type.
     */
    ComputableBitsetType
    GetBitsetType() {
        return this->bitset_type_;
    }

    /**
     * @brief Gets the memory usage of the inverted index.
     *
     * @return Memory usage in bytes.
     */
    virtual int64_t
    GetMemoryUsage() const = 0;

public:
    /// Memory allocator for the instance
    Allocator* const allocator_{nullptr};

    /// Schema mapping field names to their types
    AttrTypeSchema field_type_map_;

    /// Type of computable bitset being used
    ComputableBitsetType bitset_type_{ComputableBitsetType::FastBitset};
};
}  // namespace vsag