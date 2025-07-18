
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
insert_by_type(ValueMapPtr& value_map,
               const Attribute* attr,
               InnerIdType inner_id,
               BucketIdType bucket_id) {
    auto* attr_value = dynamic_cast<const AttributeValue<T>*>(attr);
    if (attr_value == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Invalid attribute type");
    }
    for (auto& value : attr_value->GetValue()) {
        value_map->Insert(value, inner_id, bucket_id);
    }
}

static void
insert_by_type(ValueMapPtr& value_map,
               const Attribute* attr,
               InnerIdType inner_id,
               BucketIdType bucket_id) {
    auto value_type = attr->GetValueType();
    if (value_type == AttrValueType::INT32) {
        insert_by_type<int32_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT64) {
        insert_by_type<int64_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT16) {
        insert_by_type<int16_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT8) {
        insert_by_type<int8_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT32) {
        insert_by_type<uint32_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT64) {
        insert_by_type<uint64_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT16) {
        insert_by_type<uint16_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT8) {
        insert_by_type<uint8_t>(value_map, attr, inner_id, bucket_id);
    } else if (value_type == AttrValueType::STRING) {
        insert_by_type<std::string>(value_map, attr, inner_id, bucket_id);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
}

static void
erase_by_type(ValueMapPtr& value_map,
              const AttrValueType value_type,
              InnerIdType inner_id,
              BucketIdType bucket_id) {
    if (value_type == AttrValueType::INT32) {
        value_map->Erase<int32_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT64) {
        value_map->Erase<int64_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT16) {
        value_map->Erase<int16_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::INT8) {
        value_map->Erase<int8_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT32) {
        value_map->Erase<uint32_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT64) {
        value_map->Erase<uint64_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT16) {
        value_map->Erase<uint16_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::UINT8) {
        value_map->Erase<uint8_t>(inner_id, bucket_id);
    } else if (value_type == AttrValueType::STRING) {
        value_map->Erase<std::string>(inner_id, bucket_id);
    } else {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
    }
}

template <class T>
static void
get_bitsets_by_type(const ValueMapPtr& value_map,
                    const Attribute* attr,
                    std::vector<const MultiBitsetManager*>& managers) {
    auto* attr_value = dynamic_cast<const AttributeValue<T>*>(attr);
    if (attr_value == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "Invalid attribute type");
    }
    auto values = attr_value->GetValue();
    auto count = values.size();
    for (int i = 0; i < count; ++i) {
        managers[i] = value_map->GetBitsetByValue(values[i]);
    }
}

void
AttributeBucketInvertedDataCell::Insert(const AttributeSet& attr_set,
                                        InnerIdType inner_id,
                                        BucketIdType bucket_id) {
    std::lock_guard lock(this->global_mutex_);

    for (auto* attr : attr_set.attrs_) {
        auto iter = field_2_value_map_.find(attr->name_);
        if (iter == field_2_value_map_.end()) {
            field_2_value_map_[attr->name_] =
                std::make_shared<AttrValueMap>(allocator_, this->bitset_type_);
        }
        auto& value_map = field_2_value_map_[attr->name_];
        auto value_type = attr->GetValueType();
        this->field_type_map_.SetTypeOfField(attr->name_, value_type);

        insert_by_type(value_map, attr, inner_id, bucket_id);
    }
}

std::vector<const MultiBitsetManager*>
AttributeBucketInvertedDataCell::GetBitsetsByAttr(const Attribute& attr) {
    std::shared_lock lock(this->global_mutex_);
    std::vector<const MultiBitsetManager*> bitsets(attr.GetValueCount(), nullptr);
    auto iter = field_2_value_map_.find(attr.name_);
    if (iter == field_2_value_map_.end()) {
        return std::move(bitsets);
    }
    const auto& value_map = iter->second;
    auto value_type = attr.GetValueType();
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
    return std::move(bitsets);
}

void
AttributeBucketInvertedDataCell::Serialize(StreamWriter& writer) {
    AttributeInvertedInterface::Serialize(writer);
    auto size = field_2_value_map_.size();
    StreamWriter::WriteObj(writer, size);

    for (const auto& [term, value_map] : field_2_value_map_) {
        StreamWriter::WriteString(writer, term);
        value_map->Serialize(writer);
    }
}

void
AttributeBucketInvertedDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    AttributeInvertedInterface::Deserialize(reader);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    this->field_2_value_map_.reserve(size);
    for (uint64_t i = 0; i < size; i++) {
        auto term = StreamReader::ReadString(reader);
        auto value_map = std::make_shared<AttrValueMap>(this->allocator_, this->bitset_type_);
        value_map->Deserialize(reader);
        field_2_value_map_[term] = value_map;
    }
}
void
AttributeBucketInvertedDataCell::UpdateBitsetsByAttr(const AttributeSet& attributes,
                                                     const InnerIdType offset_id,
                                                     const BucketIdType bucket_id) {
    for (const auto* attr : attributes.attrs_) {
        const auto& name = attr->name_;
        auto& value_map = this->field_2_value_map_[name];
        auto type = attr->GetValueType();
        erase_by_type(value_map, type, offset_id, bucket_id);
        insert_by_type(value_map, attr, offset_id, bucket_id);
    }
}

}  // namespace vsag
