
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

#include "attribute_bucket_inverted_datacell.h"
namespace vsag {

void
AttributeBucketInvertedDataCell::Insert(const AttributeSet& attr_set, InnerIdType inner_id) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "Insert Not implemented");
}

void
AttributeBucketInvertedDataCell::InsertWithBucket(const AttributeSet& attr_set,
                                                  InnerIdType inner_id,
                                                  BucketIdType bucket_id) {
    {
        std::lock_guard lock(this->multi_term_2_value_map_mutex_);
        auto start = this->multi_term_2_value_map_.size();
        while (start < bucket_id + 1) {
            this->multi_term_2_value_map_.emplace_back(
                std::make_unique<UnorderedMap<std::string, ValueMapPtr>>(allocator_));
            ++start;
            this->bucket_mutexes_.emplace_back(std::make_shared<std::shared_mutex>());
        }
    }
    std::shared_lock lock(this->multi_term_2_value_map_mutex_);
    auto& cur_bucket = this->multi_term_2_value_map_[bucket_id];
    std::lock_guard bucket_lock(*this->bucket_mutexes_[bucket_id]);
    for (auto* attr : attr_set.attrs_) {
        if (cur_bucket->find(attr->name_) == cur_bucket->end()) {
            (*cur_bucket)[attr->name_] =
                std::make_shared<AttrValueMap>(allocator_, ComputableBitsetType::FastBitset);
        }
        auto& value_map = (*cur_bucket)[attr->name_];
        auto value_type = attr->GetValueType();
        if (value_type == AttrValueType::INT32) {
            this->insert_by_type<int32_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::INT64) {
            this->insert_by_type<int64_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::INT16) {
            this->insert_by_type<int16_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::INT8) {
            this->insert_by_type<int8_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::UINT32) {
            this->insert_by_type<uint32_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::UINT64) {
            this->insert_by_type<uint64_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::UINT16) {
            this->insert_by_type<uint16_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::UINT8) {
            this->insert_by_type<uint8_t>(value_map, attr, inner_id);
        } else if (value_type == AttrValueType::STRING) {
            this->insert_by_type<std::string>(value_map, attr, inner_id);
        } else {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
        }
    }
}

std::vector<ComputableBitsetPtr>
AttributeBucketInvertedDataCell::GetBitsetsByAttr(const Attribute& attr) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "GetBitsetsByAttr Not implemented");
}

std::vector<ComputableBitsetPtr>
AttributeBucketInvertedDataCell::GetBitsetsByAttrAndBucketId(const Attribute& attr,
                                                             BucketIdType bucket_id) {
    std::shared_lock lock(this->multi_term_2_value_map_mutex_);
    if (bucket_id >= this->bucket_mutexes_.size()) {
        return {attr.GetValueCount(), nullptr};
    }
    auto& value_maps = multi_term_2_value_map_[bucket_id];

    std::shared_lock bucket_lock(*this->bucket_mutexes_[bucket_id]);

    if (value_maps == nullptr or value_maps->find(attr.name_) == value_maps->end()) {
        return {attr.GetValueCount(), nullptr};
    }
    auto& value_map = (*value_maps)[attr.name_];
    auto value_type = attr.GetValueType();
    std::vector<ComputableBitsetPtr> bitsets;
    if (value_type == AttrValueType::INT32) {
        this->get_bitsets_by_type<int32_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT64) {
        this->get_bitsets_by_type<int64_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT16) {
        this->get_bitsets_by_type<int16_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT8) {
        this->get_bitsets_by_type<int8_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT32) {
        this->get_bitsets_by_type<uint32_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT64) {
        this->get_bitsets_by_type<uint64_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT16) {
        this->get_bitsets_by_type<uint16_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT8) {
        this->get_bitsets_by_type<uint8_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::STRING) {
        this->get_bitsets_by_type<std::string>(value_map, &attr, bitsets);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
    return bitsets;
}

void
AttributeBucketInvertedDataCell::Serialize(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, multi_term_2_value_map_.size());
    for (auto& term_2_bucket_value_map : multi_term_2_value_map_) {
        StreamWriter::WriteObj(writer, term_2_bucket_value_map->size());
        for (auto& [term, value_map] : *term_2_bucket_value_map) {
            StreamWriter::WriteString(writer, term);
            value_map->Serialize(writer);
        }
    }
}

void
AttributeBucketInvertedDataCell::Deserialize(StreamReader& reader) {
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    multi_term_2_value_map_.reserve(size);
    bucket_mutexes_.resize(size);
    for (uint64_t i = 0; i < size; i++) {
        bucket_mutexes_[i] = std::make_shared<std::shared_mutex>();
        uint64_t map_size;
        StreamReader::ReadObj(reader, map_size);
        Term2ValueMap map = std::make_unique<UnorderedMap<std::string, ValueMapPtr>>(allocator_);
        map->reserve(map_size);
        for (uint64_t j = 0; j < map_size; ++j) {
            auto term = StreamReader::ReadString(reader);
            auto value_map =
                std::make_shared<AttrValueMap>(this->allocator_, ComputableBitsetType::FastBitset);
            value_map->Deserialize(reader);
            (*map)[term] = value_map;
        }
        multi_term_2_value_map_.emplace_back(std::move(map));
    }
}

}  // namespace vsag
