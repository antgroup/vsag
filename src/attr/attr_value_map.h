
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
#include "impl/bitset/computable_bitset.h"
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
                          ComputableBitsetType bitset_type = ComputableBitsetType::SparseBitset)
        : allocator_(allocator),
          int64_to_bitset_(allocator),
          int32_to_bitset_(allocator),
          int16_to_bitset_(allocator),
          int8_to_bitset_(allocator),
          uint64_to_bitset_(allocator),
          uint32_to_bitset_(allocator),
          uint16_to_bitset_(allocator),
          uint8_to_bitset_(allocator),
          string_to_bitset_(allocator),
          bitset_type_(bitset_type){};

    virtual ~AttrValueMap() = default;

    template <class T>
    void
    Insert(T value, InnerIdType inner_id) {
        auto& map = GetMapByType<T>();
        if (map.find(value) == map.end()) {
            map[value] = ComputableBitset::MakeInstance(this->bitset_type_, allocator_);
        }
        map[value]->Set(inner_id);
    }

    template <class T>
    ComputableBitsetPtr
    GetBitsetByValue(T value) {
        auto& map = this->GetMapByType<T>();
        auto iter = map.find(value);
        if (iter == map.end()) {
            return nullptr;
        }
        return iter->second;
    }

    template <class T>
    UnorderedMap<T, ComputableBitsetPtr>&
    GetMapByType() {
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

    void
    Serialize(StreamWriter& writer) {
        serialize_map(writer, int64_to_bitset_);
        serialize_map(writer, int32_to_bitset_);
        serialize_map(writer, int16_to_bitset_);
        serialize_map(writer, int8_to_bitset_);
        serialize_map(writer, uint64_to_bitset_);
        serialize_map(writer, uint32_to_bitset_);
        serialize_map(writer, uint16_to_bitset_);
        serialize_map(writer, uint8_to_bitset_);
        StreamWriter::WriteObj(writer, string_to_bitset_.size());
        for (auto& [key, value] : string_to_bitset_) {
            StreamWriter::WriteString(writer, key);
            value->Serialize(writer);
        }
    }

    void
    Deserialize(StreamReader& reader) {
        deserialize_map(reader, int64_to_bitset_);
        deserialize_map(reader, int32_to_bitset_);
        deserialize_map(reader, int16_to_bitset_);
        deserialize_map(reader, int8_to_bitset_);
        deserialize_map(reader, uint64_to_bitset_);
        deserialize_map(reader, uint32_to_bitset_);
        deserialize_map(reader, uint16_to_bitset_);
        deserialize_map(reader, uint8_to_bitset_);
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        for (uint64_t i = 0; i < size; ++i) {
            std::string key;
            key = StreamReader::ReadString(reader);
            auto bitset = ComputableBitset::MakeInstance(this->bitset_type_, allocator_);
            bitset->Deserialize(reader);
            string_to_bitset_[key] = bitset;
        }
    }

private:
    template <class T>
    void
    serialize_map(StreamWriter& writer, const UnorderedMap<T, ComputableBitsetPtr>& map) {
        StreamWriter::WriteObj(writer, map.size());
        for (auto& [key, value] : map) {
            StreamWriter::WriteObj(writer, key);
            value->Serialize(writer);
        }
    }

    template <class T>
    void
    deserialize_map(StreamReader& reader, UnorderedMap<T, ComputableBitsetPtr>& map) {
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        for (uint64_t i = 0; i < size; ++i) {
            T key;
            StreamReader::ReadObj(reader, key);
            auto bitset = ComputableBitset::MakeInstance(this->bitset_type_, allocator_);
            bitset->Deserialize(reader);
            map[key] = bitset;
        }
    }

private:
    UnorderedMap<int64_t, ComputableBitsetPtr> int64_to_bitset_;
    UnorderedMap<int32_t, ComputableBitsetPtr> int32_to_bitset_;
    UnorderedMap<int16_t, ComputableBitsetPtr> int16_to_bitset_;
    UnorderedMap<int8_t, ComputableBitsetPtr> int8_to_bitset_;
    UnorderedMap<uint64_t, ComputableBitsetPtr> uint64_to_bitset_;
    UnorderedMap<uint32_t, ComputableBitsetPtr> uint32_to_bitset_;
    UnorderedMap<uint16_t, ComputableBitsetPtr> uint16_to_bitset_;
    UnorderedMap<uint8_t, ComputableBitsetPtr> uint8_to_bitset_;
    UnorderedMap<std::string, ComputableBitsetPtr> string_to_bitset_;

    Allocator* const allocator_{nullptr};

    const ComputableBitsetType bitset_type_{ComputableBitsetType::SparseBitset};
};
}  // namespace vsag
