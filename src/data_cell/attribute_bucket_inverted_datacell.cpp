
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

template <class T>
static void
insert_by_type(ValueMapPtr& value_map, const Attribute* attr, InnerIdType inner_id) {
    auto* attr_value = dynamic_cast<const AttributeValue<T>*>(attr);
    if (attr_value == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Invalid attribute type");
    }
    for (auto& value : attr_value->GetValue()) {
        value_map->Insert(value, inner_id);
    }
}

static void
insert_by_type(ValueMapPtr& value_map, const Attribute* attr, InnerIdType inner_id) {
    auto value_type = attr->GetValueType();
    if (value_type == AttrValueType::INT32) {
        insert_by_type<int32_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::INT64) {
        insert_by_type<int64_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::INT16) {
        insert_by_type<int16_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::INT8) {
        insert_by_type<int8_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::UINT32) {
        insert_by_type<uint32_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::UINT64) {
        insert_by_type<uint64_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::UINT16) {
        insert_by_type<uint16_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::UINT8) {
        insert_by_type<uint8_t>(value_map, attr, inner_id);
    } else if (value_type == AttrValueType::STRING) {
        insert_by_type<std::string>(value_map, attr, inner_id);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
}

static void
erase_by_type(ValueMapPtr& value_map, const AttrValueType value_type, InnerIdType inner_id) {
    if (value_type == AttrValueType::INT32) {
        value_map->Erase<int32_t>(inner_id);
    } else if (value_type == AttrValueType::INT64) {
        value_map->Erase<int64_t>(inner_id);
    } else if (value_type == AttrValueType::INT16) {
        value_map->Erase<int16_t>(inner_id);
    } else if (value_type == AttrValueType::INT8) {
        value_map->Erase<int8_t>(inner_id);
    } else if (value_type == AttrValueType::UINT32) {
        value_map->Erase<uint32_t>(inner_id);
    } else if (value_type == AttrValueType::UINT64) {
        value_map->Erase<uint64_t>(inner_id);
    } else if (value_type == AttrValueType::UINT16) {
        value_map->Erase<uint16_t>(inner_id);
    } else if (value_type == AttrValueType::UINT8) {
        value_map->Erase<uint8_t>(inner_id);
    } else if (value_type == AttrValueType::STRING) {
        value_map->Erase<std::string>(inner_id);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
}

template <class T>
static void
get_bitsets_by_type(const ValueMapPtr& value_map,
                    const Attribute* attr,
                    std::vector<const ComputableBitset*>& bitsets) {
    auto* attr_value = dynamic_cast<const AttributeValue<T>*>(attr);
    if (attr_value == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Invalid attribute type");
    }
    auto values = attr_value->GetValue();
    auto count = values.size();
    for (int i = 0; i < count; ++i) {
        bitsets[i] = value_map->GetBitsetByValue(values[i]);
    }
}

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
        this->field_type_map_.SetTypeOfField(attr->name_, value_type);
        insert_by_type(value_map, attr, inner_id);
    }
}

std::vector<const ComputableBitset*>
AttributeBucketInvertedDataCell::GetBitsetsByAttr(const Attribute& attr) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "GetBitsetsByAttr Not implemented");
}

std::vector<const ComputableBitset*>
AttributeBucketInvertedDataCell::GetBitsetsByAttrAndBucketId(const Attribute& attr,
                                                             BucketIdType bucket_id) {
    std::shared_lock lock(this->multi_term_2_value_map_mutex_);
    if (bucket_id >= this->bucket_mutexes_.size()) {
        return {attr.GetValueCount(), nullptr};
    }
    auto& value_maps = multi_term_2_value_map_[bucket_id];

    std::shared_lock bucket_lock(*this->bucket_mutexes_[bucket_id]);

    if (value_maps == nullptr) {
        return {attr.GetValueCount(), nullptr};
    }
    auto iter = value_maps->find(attr.name_);
    if (iter == value_maps->end()) {
        return {attr.GetValueCount(), nullptr};
    }
    const auto& value_map = iter->second;
    auto value_type = attr.GetValueType();
    std::vector<const ComputableBitset*> bitsets(attr.GetValueCount());
    if (value_type == AttrValueType::INT32) {
        get_bitsets_by_type<int32_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT64) {
        get_bitsets_by_type<int64_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT16) {
        get_bitsets_by_type<int16_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::INT8) {
        get_bitsets_by_type<int8_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT32) {
        get_bitsets_by_type<uint32_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT64) {
        get_bitsets_by_type<uint64_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT16) {
        get_bitsets_by_type<uint16_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::UINT8) {
        get_bitsets_by_type<uint8_t>(value_map, &attr, bitsets);
    } else if (value_type == AttrValueType::STRING) {
        get_bitsets_by_type<std::string>(value_map, &attr, bitsets);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
    return bitsets;
}

void
AttributeBucketInvertedDataCell::Serialize(StreamWriter& writer) {
    AttributeInvertedInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, multi_term_2_value_map_.size());
    for (auto& term_2_bucket_value_map : multi_term_2_value_map_) {
        StreamWriter::WriteObj(writer, term_2_bucket_value_map->size());
        for (const auto& [term, value_map] : *term_2_bucket_value_map) {
            StreamWriter::WriteString(writer, term);
            value_map->Serialize(writer);
        }
    }
}

void
AttributeBucketInvertedDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    AttributeInvertedInterface::Deserialize(reader);
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
void
AttributeBucketInvertedDataCell::UpdateBitsetsByAttrAndBucketId(const AttributeSet& attributes,
                                                                const BucketIdType bucket_id,
                                                                const InnerIdType offset_id) {
    auto& value_maps = this->multi_term_2_value_map_[bucket_id];
    for (const auto* attr : attributes.attrs_) {
        const auto& name = attr->name_;
        auto& value_map = (*value_maps)[name];
        auto type = attr->GetValueType();
        erase_by_type(value_map, type, offset_id);
        insert_by_type(value_map, attr, offset_id);
    }
}

}  // namespace vsag
