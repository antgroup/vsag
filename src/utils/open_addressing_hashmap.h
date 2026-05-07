// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace vsag {

/**
 * @brief Open-addressing HashMap optimized for Build Cache's FeatureID → inner_id mapping.
 *
 * Design choices:
 * - Uses FNV-1a hash of std::string keys for fast lookup.
 * - Supports variable-length FeatureID strings.
 * - Slot layout: stores std::string key + ValueType value per slot.
 * - Supports "write-once, read-many" access pattern typical of Build Cache.
 * - Load factor of 0.5 by default to keep probe chains short.
 */
template <typename ValueType = uint32_t>
class OpenAddressingHashMap {
public:
    /**
     * @brief Construct a new Open Addressing HashMap.
     *
     * @param expected_count Expected number of entries (capacity will be ~2x this).
     * @param feature_id_len Hint for average key length (unused in variable-length mode).
     */
    explicit OpenAddressingHashMap(uint64_t expected_count, uint32_t feature_id_len = 32) {
        (void)feature_id_len;  // unused, kept for API compatibility
        // Use load factor 0.5: capacity is 2x expected_count
        uint64_t min_slots = expected_count * 2;
        // Round up to next power of 2 for efficient modular arithmetic
        slot_count_ = 1;
        while (slot_count_ < min_slots) {
            slot_count_ <<= 1;
        }
        slot_mask_ = slot_count_ - 1;
        keys_.resize(slot_count_);
        values_.resize(slot_count_);
        occupied_.assign(slot_count_, false);
        size_ = 0;
    }

    /**
     * @brief Insert a key-value pair. Key must not already exist.
     *
     * @param feature_id Pointer to the FeatureID string data (must be null-terminated or
     *                   have at least feature_id_len bytes).
     * @param feature_id_len Length of the FeatureID string in bytes.
     * @param value The value to associate with the key.
     * @return true if inserted, false if the key already exists or table is full.
     */
    bool
    Insert(const char* feature_id, uint32_t feature_id_len, ValueType value) {
        std::string key(feature_id, feature_id_len);
        return Insert(std::move(key), value);
    }

    /**
     * @brief Insert a key-value pair using a std::string key.
     */
    bool
    Insert(std::string key, ValueType value) {
        uint64_t hash_val = HashString(key);
        uint64_t slot_idx = hash_val & slot_mask_;

        for (uint64_t probe = 0; probe < slot_count_; ++probe) {
            uint64_t idx = (slot_idx + probe) & slot_mask_;
            if (!occupied_[idx]) {
                keys_[idx] = std::move(key);
                values_[idx] = value;
                occupied_[idx] = true;
                ++size_;
                return true;
            }
            if (keys_[idx] == key) {
                // Key already exists
                return false;
            }
        }
        // Table is full
        return false;
    }

    /**
     * @brief Look up a key and return its associated value.
     *
     * @param feature_id Pointer to the FeatureID string data.
     * @param feature_id_len Length of the FeatureID string in bytes.
     * @param[out] value The found value is written here on success.
     * @return true if key was found, false otherwise.
     */
    bool
    Lookup(const char* feature_id, uint32_t feature_id_len, ValueType& value) const {
        std::string key(feature_id, feature_id_len);
        return Lookup(key, value);
    }

    /**
     * @brief Look up a key using a std::string.
     */
    bool
    Lookup(const std::string& key, ValueType& value) const {
        uint64_t hash_val = HashString(key);
        uint64_t slot_idx = hash_val & slot_mask_;

        for (uint64_t probe = 0; probe < slot_count_; ++probe) {
            uint64_t idx = (slot_idx + probe) & slot_mask_;
            if (!occupied_[idx]) {
                return false;
            }
            if (keys_[idx] == key) {
                value = values_[idx];
                return true;
            }
        }
        return false;
    }

    /** @brief Current number of entries in the map. */
    uint64_t
    Size() const {
        return size_;
    }

    /** @brief Total number of slots. */
    uint64_t
    Capacity() const {
        return slot_count_;
    }

    /** @brief Memory usage in bytes. */
    uint64_t
    MemoryUsage() const {
        uint64_t base = slot_count_ * (sizeof(ValueType) + sizeof(bool) + 32 /* avg string */);
        uint64_t key_mem = 0;
        for (uint64_t i = 0; i < slot_count_; ++i) {
            if (occupied_[i]) {
                key_mem += keys_[i].capacity();
            }
        }
        return base + key_mem;
    }

private:
    static uint64_t
    HashString(const std::string& key) {
        // FNV-1a hash for fast, good distribution
        uint64_t hash = 14695981039346656037ULL;
        for (char c : key) {
            hash ^= static_cast<uint64_t>(static_cast<uint8_t>(c));
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    uint64_t slot_count_{0};
    uint64_t slot_mask_{0};
    uint64_t size_{0};
    std::vector<std::string> keys_;
    std::vector<ValueType> values_;
    std::vector<bool> occupied_;
};

}  // namespace vsag
