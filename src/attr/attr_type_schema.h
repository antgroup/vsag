
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
 * @file attr_type_schema.h
 * @brief Attribute type schema definition for managing field type information.
 *
 * This file provides the AttrTypeSchema class which stores and manages
 * the type information for attribute fields used in filter conditions.
 */

#pragma once

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/attribute.h"

namespace vsag {

/**
 * @class AttrTypeSchema
 * @brief Schema for managing attribute field type information.
 *
 * This class provides a mapping from field names to their corresponding
 * attribute value types, supporting serialization and deserialization.
 */
class AttrTypeSchema {
public:
    /**
     * @brief Constructs an AttrTypeSchema with the specified allocator.
     * @param allocator Pointer to the allocator for memory management.
     */
    explicit AttrTypeSchema(Allocator* allocator);

    virtual ~AttrTypeSchema() = default;

    /**
     * @brief Gets the type of a field by name.
     * @param field_name The name of the field to query.
     * @return The AttrValueType of the field.
     */
    AttrValueType
    GetTypeOfField(const std::string& field_name);

    /**
     * @brief Sets the type of a field.
     * @param field_name The name of the field.
     * @param type The AttrValueType to set for the field.
     */
    void
    SetTypeOfField(const std::string& field_name, AttrValueType type);

    /**
     * @brief Serializes the schema to a stream writer.
     * @param writer The stream writer to write to.
     */
    void
    Serialize(StreamWriter& writer);

    /**
     * @brief Deserializes the schema from a stream reader.
     * @param reader The stream reader to read from.
     */
    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader);

private:
    UnorderedMap<std::string, AttrValueType> schema_;  ///< Mapping from field name to type

    Allocator* const allocator_{nullptr};  ///< Allocator for memory management
};

}  // namespace vsag
