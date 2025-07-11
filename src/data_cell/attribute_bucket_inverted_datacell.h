
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
#include <shared_mutex>

#include "attr/attr_value_map.h"
#include "attribute_inverted_interface.h"
#include "vsag_exception.h"

namespace vsag {

class AttributeBucketInvertedDataCell : public AttributeInvertedInterface {
public:
    using Term2ValueMap = std::unique_ptr<UnorderedMap<std::string, ValueMapPtr>>;

    AttributeBucketInvertedDataCell(Allocator* allocator)
        : AttributeInvertedInterface(allocator), multi_term_2_value_map_(allocator){};
    ~AttributeBucketInvertedDataCell() override = default;

    void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id) override;

    void
    InsertWithBucket(const AttributeSet& attr_set,
                     InnerIdType inner_id,
                     BucketIdType bucket_id) override;

    std::vector<const ComputableBitset*>
    GetBitsetsByAttr(const Attribute& attr) override;

    std::vector<const ComputableBitset*>
    GetBitsetsByAttrAndBucketId(const Attribute& attr_name, BucketIdType bucket_id) override;

    void
    UpdateBitsetsByAttrAndBucketId(const AttributeSet& attributes,
                                   const BucketIdType bucket_id,
                                   const InnerIdType offset_id) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

private:
    Vector<Term2ValueMap> multi_term_2_value_map_;

    std::vector<std::shared_ptr<std::shared_mutex>> bucket_mutexes_{};

    std::shared_mutex multi_term_2_value_map_mutex_{};
};
}  // namespace vsag
