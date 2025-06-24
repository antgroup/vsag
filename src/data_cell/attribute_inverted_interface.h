
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

#include "impl/bitset/computable_bitset.h"
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
    AttributeInvertedInterface(Allocator* allocator)
        : allocator_(allocator), field_type_map_(allocator){};
    virtual ~AttributeInvertedInterface() = default;

    virtual void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id) = 0;

    virtual void
    InsertWithBucket(const AttributeSet& attr_set,
                     InnerIdType inner_id,
                     BucketIdType bucket_id) = 0;

    virtual std::vector<ComputableBitsetPtr>
    GetBitsetsByAttr(const Attribute& attr) = 0;

    virtual std::vector<ComputableBitsetPtr>
    GetBitsetsByAttrAndBucketId(const Attribute& attr_name, BucketIdType bucket_id) = 0;

    virtual void
    Serialize(StreamWriter& writer) {
        auto size = this->field_type_map_.size();
        StreamWriter::WriteObj(writer, size);
        for (auto& [k, v] : this->field_type_map_) {
            StreamWriter::WriteString(writer, k);
            StreamWriter::WriteObj(writer, static_cast<int64_t>(v));
        }
    }

    virtual void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        this->field_type_map_.reserve(size);
        std::string key;
        int64_t value;
        for (int64_t i = 0; i < size; ++i) {
            key = StreamReader::ReadString(reader);
            StreamReader::ReadObj(reader, value);
            this->field_type_map_[key] = static_cast<AttrValueType>(value);
        }
    }

    AttrValueType
    GetTypeOfField(const std::string& field_name) {
        auto iter = this->field_type_map_.find(field_name);
        if (iter == this->field_type_map_.end()) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "field not found");
        }
        return iter->second;
    }

public:
    Allocator* const allocator_{nullptr};

    UnorderedMap<std::string, AttrValueType> field_type_map_;
};
}  // namespace vsag
