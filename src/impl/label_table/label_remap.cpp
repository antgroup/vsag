
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

#include "label_remap.h"

namespace vsag {

/**
 * @brief LabelRemap constructor implementation
 *
 * Initializes the appropriate hash table instance based on
 * the specified type and sets a reasonable load factor
 * for performance optimization.
 */
LabelRemap::LabelRemap(Allocator* allocator, LabelRemapType remap_type)
    : allocator_(allocator), remap_type_(remap_type) {
    if (remap_type_ == LabelRemapType::ROBIN) {
        robin_map_ = std::make_unique<UnorderedMap<LabelType, InnerIdType>>(0, allocator);
        robin_map_->max_load_factor(0.75F);
    } else {
        pg_map_ = std::make_unique<PGUnorderedMap<LabelType, InnerIdType>>(0, allocator);
        pg_map_->max_load_factor(0.75F);
    }
}

/**
 * @brief Reset implementation
 *
 * Creates a new empty hash table instance, releasing the memory
 * of the previous hash table. This is more thorough than Clear(),
 * which only removes elements without releasing memory.
 */
void
LabelRemap::Reset() {
    if (remap_type_ == LabelRemapType::ROBIN) {
        robin_map_ = std::make_unique<UnorderedMap<LabelType, InnerIdType>>(0, allocator_);
        robin_map_->max_load_factor(0.75F);
        return;
    }

    pg_map_ = std::make_unique<PGUnorderedMap<LabelType, InnerIdType>>(0, allocator_);
    pg_map_->max_load_factor(0.75F);
}

/**
 * @brief Clear implementation
 *
 * Removes all elements from the hash table without releasing
 * the underlying memory. Useful when the table will be reused
 * with a similar number of elements.
 */
void
LabelRemap::Clear() {
    if (pg_map_ != nullptr) {
        pg_map_->clear();
        return;
    }
    robin_map_->clear();
}

/**
 * @brief Reserve implementation
 *
 * Pre-allocates memory for the specified number of elements,
 * improving performance when the expected size is known.
 */
void
LabelRemap::Reserve(uint64_t size) {
    if (pg_map_ != nullptr) {
        pg_map_->reserve(size);
        return;
    }
    robin_map_->reserve(size);
}

/**
 * @brief Size implementation
 *
 * Returns the current number of label-ID pairs stored in the
 * hash table.
 */
uint64_t
LabelRemap::Size() const {
    if (pg_map_ != nullptr) {
        return pg_map_->size();
    }
    return robin_map_->size();
}

/**
 * @brief InsertOrAssign implementation
 *
 * Inserts a new label-ID pair or updates the existing ID if
 * the label already exists. Uses direct indexing operator.
 */
void
LabelRemap::InsertOrAssign(LabelType label, InnerIdType inner_id) {
    if (pg_map_ != nullptr) {
        (*pg_map_)[label] = inner_id;
        return;
    }
    (*robin_map_)[label] = inner_id;
}

/**
 * @brief Emplace implementation
 *
 * Inserts a new label-ID pair only if the label does not exist.
 * Does not update existing mappings.
 */
void
LabelRemap::Emplace(LabelType label, InnerIdType inner_id) {
    if (pg_map_ != nullptr) {
        pg_map_->emplace(label, inner_id);
        return;
    }
    robin_map_->emplace(label, inner_id);
}

/**
 * @brief Erase implementation
 *
 * Removes the mapping for the specified label. Returns true
 * if the label was found and removed, false otherwise.
 */
bool
LabelRemap::Erase(LabelType label) {
    if (pg_map_ != nullptr) {
        return pg_map_->erase(label) > 0;
    }
    return robin_map_->erase(label) > 0;
}

/**
 * @brief Find implementation
 *
 * Searches for the specified label and returns the associated
 * internal ID if found. Returns false if the label is not found.
 */
bool
LabelRemap::Find(LabelType label, InnerIdType& inner_id) const {
    if (pg_map_ != nullptr) {
        const auto iter = pg_map_->find(label);
        if (iter == pg_map_->end()) {
            return false;
        }
        inner_id = iter->second;
        return true;
    }

    const auto iter = robin_map_->find(label);
    if (iter == robin_map_->end()) {
        return false;
    }
    inner_id = iter->second;
    return true;
}

}  // namespace vsag
