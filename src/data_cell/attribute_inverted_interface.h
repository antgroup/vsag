
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
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"
#include "vsag/attribute.h"

namespace vsag {
class AttributeInvertedInterface;
using AttrInvertedInterfacePtr = std::shared_ptr<AttributeInvertedInterface>;

class AttributeInvertedInterface {
public:
    static AttrInvertedInterfacePtr
    MakeInstance(Allocator* allocator, bool have_bucket = false);

public:
    AttributeInvertedInterface(Allocator* allocator) : allocator_(allocator){};
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
    }

    virtual void
    Deserialize(StreamReader& reader) {
    }

public:
    Allocator* const allocator_{nullptr};
};
}  // namespace vsag
