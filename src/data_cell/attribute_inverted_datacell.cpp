
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

#include "attribute_inverted_datacell.h"
namespace vsag {

void
AttributeInvertedDataCell::Insert(const AttributeSet& attr_set, InnerIdType inner_id) {
    std::lock_guard lock(this->term_2_value_map_mutex_);
    for (auto* attr : attr_set.attrs_) {
        if (term_2_value_map_.find(attr->name_) == term_2_value_map_.end()) {
            term_2_value_map_[attr->name_] = std::make_shared<AttrValueMap>(this->allocator_);
        }
        auto& value_map = term_2_value_map_[attr->name_];
        auto value_type = attr->GetValueType();
        this->field_type_map_[attr->name_] = value_type;
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

void
AttributeInvertedDataCell::InsertWithBucket(const AttributeSet& attr_set,
                                            InnerIdType inner_id,
                                            BucketIdType bucket_id) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "InsertWithBucket Not implemented");
}

std::vector<ComputableBitsetPtr>
AttributeInvertedDataCell::GetBitsetsByAttr(const Attribute& attr) {
    std::shared_lock lock(this->term_2_value_map_mutex_);
    if (term_2_value_map_.find(attr.name_) == term_2_value_map_.end()) {
        return {};
    }
    auto& value_map = term_2_value_map_[attr.name_];
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

std::vector<ComputableBitsetPtr>
AttributeInvertedDataCell::GetBitsetsByAttrAndBucketId(const Attribute& attr_name,
                                                       BucketIdType bucket_id) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "GetBitsetsByAttrAndBucketId Not implemented");
}

void
AttributeInvertedDataCell::Serialize(StreamWriter& writer) {
    AttributeInvertedInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, term_2_value_map_.size());
    for (const auto& [term, value_map] : term_2_value_map_) {
        StreamWriter::WriteString(writer, term);
        value_map->Serialize(writer);
    }
}

void
AttributeInvertedDataCell::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    AttributeInvertedInterface::Deserialize(reader);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    term_2_value_map_.reserve(size);
    for (uint64_t i = 0; i < size; i++) {
        auto term = StreamReader::ReadString(reader);
        auto value_map = std::make_shared<AttrValueMap>(this->allocator_);
        value_map->Deserialize(reader);
        term_2_value_map_[term] = value_map;
    }
}

}  // namespace vsag
