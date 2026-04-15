
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
 * @file attr_value_map.h
 * @brief Attribute value to bitset mapping for efficient attribute filtering.
 *
 * This file provides the AttrValueMap class which maintains mappings from
 * attribute values to bitsets, enabling fast filtering operations based on
 * attribute values.
 */

#pragma once
#include <memory>

#include "impl/allocator/safe_allocator.h"
#include "multi_bitset_manager.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/attribute.h"
#include "vsag_exception.h"

namespace vsag {

DEFINE_POINTER2(ValueMap, AttrValueMap);

/**
 * @class AttrValueMap
 * @brief Maps attribute values to bitsets for efficient filtering.
 *
 * This class maintains multiple maps from different attribute value types
 * to MultiBitsetManager instances, supporting type-safe attribute storage
 * and retrieval.
 */
class AttrValueMap {
public:
    /**
     * @brief Constructs an AttrValueMap with the specified allocator.
     * @param allocator Pointer to the allocator for memory management.
     * @param bitset_type The type of computable bitset to use.
     */
    explicit AttrValueMap(Allocator* allocator,
                          ComputableBitsetType bitset_type = ComputableBitsetType::FastBitset);

    virtual ~AttrValueMap();

    /**
     * @brief Inserts a value with its associated inner ID into the map.
     * @tparam T The type of the value (int64_t, int32_t, int16_t, int8_t,
     *           uint64_t, uint32_t, uint16_t, uint8_t, or std::string).
     * @param value The attribute value to insert.
     * @param inner_id The inner ID associated with this value.
     * @param bucket_id The bucket ID for multi-bitset scenarios. Defaults to 0.
     */
    template <class T>
    void
    Insert(T value, InnerIdType inner_id, BucketIdType bucket_id = 0) {
        auto& map = this->get_map_by_type<T>();
        if (map.find(value) == map.end()) {
            map[value] = new MultiBitsetManager(allocator_, 1, this->bitset_type_);
        }
        map[value]->InsertValue(bucket_id, inner_id, true);
    }

    /**
     * @brief Retrieves the bitset manager for a specific value.
     * @tparam T The type of the value.
     * @param value The attribute value to look up.
     * @return Pointer to the MultiBitsetManager, or nullptr if not found.
     */
    template <class T>
    MultiBitsetManager*
    GetBitsetByValue(T value) {
        auto& map = this->get_map_by_type<T>();
        auto iter = map.find(value);
        if (iter == map.end()) {
            return nullptr;
        }
        return iter->second;
    }

    /**
     * @brief Erases all values associated with an inner ID from the map.
     * @tparam T The type of values in the map.
     * @param inner_id The inner ID to erase.
     * @param bucket_id The bucket ID. Defaults to 0.
     */
    template <class T>
    void
    Erase(InnerIdType inner_id, BucketIdType bucket_id = 0) {
        auto& map = this->get_map_by_type<T>();
        for (auto& [key, manager] : map) {
            if (manager != nullptr) {
                auto* bitset = manager->GetOneBitset(bucket_id);
                if (bitset != nullptr) {
                    bitset->Set(inner_id, false);
                }
            }
        }
    }

    /**
     * @brief Erases specific attribute values associated with an inner ID.
     * @tparam T The type of values in the map.
     * @param inner_id The inner ID to erase.
     * @param attr Pointer to the attribute containing values to erase.
     * @param bucket_id The bucket ID. Defaults to 0.
     */
    template <class T>
    void
    Erase(InnerIdType inner_id, const Attribute* attr, BucketIdType bucket_id = 0) {
        auto& map = this->get_map_by_type<T>();
        const auto* attr_values = dynamic_cast<const AttributeValue<T>*>(attr);
        if (attr_values == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Attribute type not match");
        }
        const auto& values = attr_values->GetValue();
        for (const auto& value : values) {
            auto iter = map.find(value);
            if (iter != map.end() and iter->second != nullptr) {
                auto* bitset = iter->second->GetOneBitset(bucket_id);
                if (bitset != nullptr) {
                    bitset->Set(inner_id, false);
                }
            }
        }
    }

    /**
     * @brief Retrieves an attribute containing all values for an inner ID.
     * @tparam T The type of values to retrieve.
     * @param inner_id The inner ID to look up.
     * @param bucket_id The bucket ID. Defaults to 0.
     * @return Pointer to a new AttributeValue<T>, or nullptr if no values found.
     */
    template <class T>
    Attribute*
    GetAttr(InnerIdType inner_id, BucketIdType bucket_id = 0) {
        auto& map = this->get_map_by_type<T>();
        AttributeValue<T>* result = nullptr;
        bool is_new = true;
        for (auto& [key, manager] : map) {
            if (manager != nullptr) {
                auto* bitset = manager->GetOneBitset(bucket_id);
                if (bitset != nullptr and bitset->Test(inner_id)) {
                    if (is_new) {
                        result = new AttributeValue<T>();
                        is_new = false;
                    }
                    result->GetValue().emplace_back(key);
                }
            }
        }
        return result;
    }

    /**
     * @brief Serializes the map to a stream writer.
     * @param writer The stream writer to write to.
     */
    void
    Serialize(StreamWriter& writer);

    /**
     * @brief Deserializes the map from a stream reader.
     * @param reader The stream reader to read from.
     */
    void
    Deserialize(StreamReader& reader);

    /**
     * @brief Gets the total memory usage of this map.
     * @return Memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() const;

private:
    /**
     * @brief Gets the appropriate map reference for the given type.
     * @tparam T The type of values in the map.
     * @return Reference to the appropriate map.
     */
    template <class T>
    UnorderedMap<T, MultiBitsetManager*>&
    get_map_by_type() {
        if constexpr (std::is_same_v<T, int64_t>) {
            return this->int64_to_bitset_;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return this->int32_to_bitset_;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return this->int16_to_bitset_;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return this->int8_to_bitset_;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return this->uint64_to_bitset_;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return this->uint32_to_bitset_;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return this->uint16_to_bitset_;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return this->uint8_to_bitset_;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return this->string_to_bitset_;
        }
    }

private:
    UnorderedMap<int64_t, MultiBitsetManager*> int64_to_bitset_;       ///< Map for int64_t values
    UnorderedMap<int32_t, MultiBitsetManager*> int32_to_bitset_;       ///< Map for int32_t values
    UnorderedMap<int16_t, MultiBitsetManager*> int16_to_bitset_;       ///< Map for int16_t values
    UnorderedMap<int8_t, MultiBitsetManager*> int8_to_bitset_;         ///< Map for int8_t values
    UnorderedMap<uint64_t, MultiBitsetManager*> uint64_to_bitset_;     ///< Map for uint64_t values
    UnorderedMap<uint32_t, MultiBitsetManager*> uint32_to_bitset_;     ///< Map for uint32_t values
    UnorderedMap<uint16_t, MultiBitsetManager*> uint16_to_bitset_;     ///< Map for uint16_t values
    UnorderedMap<uint8_t, MultiBitsetManager*> uint8_to_bitset_;       ///< Map for uint8_t values
    UnorderedMap<std::string, MultiBitsetManager*> string_to_bitset_;  ///< Map for string values

    Allocator* const allocator_{nullptr};  ///< Allocator for memory management

    const ComputableBitsetType bitset_type_{ComputableBitsetType::SparseBitset};  ///< Bitset type
};
}  // namespace vsag
