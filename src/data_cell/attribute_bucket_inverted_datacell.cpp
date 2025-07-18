
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
        if (value_type == AttrValueType::INT32) {
            this->insert_by_type<int32_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::INT64) {
            this->insert_by_type<int64_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::INT16) {
            this->insert_by_type<int16_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::INT8) {
            this->insert_by_type<int8_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::UINT32) {
            this->insert_by_type<uint32_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::UINT64) {
            this->insert_by_type<uint64_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::UINT16) {
            this->insert_by_type<uint16_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::UINT8) {
            this->insert_by_type<uint8_t>(value_map, attr, inner_id, bucket_id);
        } else if (value_type == AttrValueType::STRING) {
            this->insert_by_type<std::string>(value_map, attr, inner_id, bucket_id);
        } else {
            throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported value type");
        }
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

}  // namespace vsag
