
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

#include "attr_value_map.h"

namespace vsag {

template <class T>
void
clear_map(const UnorderedMap<T, MultiBitsetManager*>& map) {
    for (auto& iter : map) {
        delete iter.second;
    }
}

AttrValueMap::AttrValueMap(Allocator* allocator, ComputableBitsetType bitset_type)
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
      bitset_type_(bitset_type) {
}

AttrValueMap::~AttrValueMap() {
    clear_map(this->int16_to_bitset_);
    clear_map(this->int8_to_bitset_);
    clear_map(this->uint16_to_bitset_);
    clear_map(this->uint8_to_bitset_);
    clear_map(this->string_to_bitset_);
    clear_map(this->int64_to_bitset_);
    clear_map(this->int32_to_bitset_);
    clear_map(this->uint64_to_bitset_);
    clear_map(this->uint32_to_bitset_);
}

template <class T>
void
serialize_map(StreamWriter& writer, const UnorderedMap<T, MultiBitsetManager*>& map) {
    StreamWriter::WriteObj(writer, map.size());
    for (auto& [key, value] : map) {
        StreamWriter::WriteObj(writer, key);
        value->Serialize(writer);
    }
}

void
AttrValueMap::Serialize(StreamWriter& writer) {
    serialize_map(writer, int64_to_bitset_);
    serialize_map(writer, int32_to_bitset_);
    serialize_map(writer, int16_to_bitset_);
    serialize_map(writer, int8_to_bitset_);
    serialize_map(writer, uint64_to_bitset_);
    serialize_map(writer, uint32_to_bitset_);
    serialize_map(writer, uint16_to_bitset_);
    serialize_map(writer, uint8_to_bitset_);
    StreamWriter::WriteObj(writer, string_to_bitset_.size());
    for (const auto& [key, value] : string_to_bitset_) {
        StreamWriter::WriteString(writer, key);
        value->Serialize(writer);
    }
}

template <class T>
void
deserialize_map(StreamReader& reader,
                UnorderedMap<T, MultiBitsetManager*>& map,
                Allocator* allocator,
                ComputableBitsetType type) {
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        T key;
        StreamReader::ReadObj(reader, key);
        auto* manager = new MultiBitsetManager(allocator, 1, type);
        manager->Deserialize(reader);
        map[key] = manager;
    }
}

void
AttrValueMap::Deserialize(StreamReader& reader) {
    deserialize_map(reader, int64_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, int32_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, int16_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, int8_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, uint64_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, uint32_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, uint16_to_bitset_, allocator_, bitset_type_);
    deserialize_map(reader, uint8_to_bitset_, allocator_, bitset_type_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    for (uint64_t i = 0; i < size; ++i) {
        std::string key;
        key = StreamReader::ReadString(reader);
        auto* manager = new MultiBitsetManager(allocator_, 1, bitset_type_);
        manager->Deserialize(reader);
        string_to_bitset_[key] = manager;
    }
}
}  // namespace vsag
