
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

#include "impl/allocator/safe_allocator.h"
#include "multi_bitset_manager.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag_exception.h"

namespace vsag {

class AttrValueMap;
using ValueMapPtr = std::shared_ptr<AttrValueMap>;

class AttrValueMap {
public:
    explicit AttrValueMap(Allocator* allocator,
                          ComputableBitsetType bitset_type = ComputableBitsetType::FastBitset);

    virtual ~AttrValueMap();

    template <class T>
    void
    Insert(T value, InnerIdType inner_id, BucketIdType bucket_id = 0) {
        auto& map = this->get_map_by_type<T>();
        if (map.find(value) == map.end()) {
            map[value] = new MultiBitsetManager(allocator_, 1, this->bitset_type_);
        }
        map[value]->InsertValue(bucket_id, inner_id, true);
    }

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

    void
    Serialize(StreamWriter& writer);

    void
    Deserialize(StreamReader& reader);

private:
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
    UnorderedMap<int64_t, MultiBitsetManager*> int64_to_bitset_;
    UnorderedMap<int32_t, MultiBitsetManager*> int32_to_bitset_;
    UnorderedMap<int16_t, MultiBitsetManager*> int16_to_bitset_;
    UnorderedMap<int8_t, MultiBitsetManager*> int8_to_bitset_;
    UnorderedMap<uint64_t, MultiBitsetManager*> uint64_to_bitset_;
    UnorderedMap<uint32_t, MultiBitsetManager*> uint32_to_bitset_;
    UnorderedMap<uint16_t, MultiBitsetManager*> uint16_to_bitset_;
    UnorderedMap<uint8_t, MultiBitsetManager*> uint8_to_bitset_;
    UnorderedMap<std::string, MultiBitsetManager*> string_to_bitset_;

    Allocator* const allocator_{nullptr};

    const ComputableBitsetType bitset_type_{ComputableBitsetType::SparseBitset};
};
}  // namespace vsag
