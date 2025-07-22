
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

#include "attr/attr_type_schema.h"
#include "attr/multi_bitset_manager.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/attribute.h"
#include "vsag_exception.h"

namespace vsag {
class AttributeInvertedInterface;
using AttrInvertedInterfacePtr = std::shared_ptr<AttributeInvertedInterface>;

class AttributeInvertedInterface {
public:
    static AttrInvertedInterfacePtr
    MakeInstance(Allocator* allocator, bool have_bucket = false);

public:
    AttributeInvertedInterface(Allocator* allocator, ComputableBitsetType bitset_type)
        : allocator_(allocator), field_type_map_(allocator), bitset_type_(bitset_type){};

    virtual ~AttributeInvertedInterface() = default;

    virtual void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id) {
        this->Insert(attr_set, inner_id, 0);
    }

    virtual void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id, BucketIdType bucket_id) = 0;

    virtual std::vector<const MultiBitsetManager*>
    GetBitsetsByAttr(const Attribute& attr) = 0;

    virtual void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id) = 0;

    virtual void
    Serialize(StreamWriter& writer) {
        this->field_type_map_.Serialize(writer);
    }

    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        this->field_type_map_.Deserialize(reader);
    }

    AttrValueType
    GetTypeOfField(const std::string& field_name) {
        return this->field_type_map_.GetTypeOfField(field_name);
    }

    ComputableBitsetType
    GetBitsetType() {
        return this->bitset_type_;
    }

public:
    Allocator* const allocator_{nullptr};

    AttrTypeSchema field_type_map_;

    ComputableBitsetType bitset_type_{ComputableBitsetType::FastBitset};
};
}  // namespace vsag
