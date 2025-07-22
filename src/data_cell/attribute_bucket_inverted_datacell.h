
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
    AttributeBucketInvertedDataCell(
        Allocator* allocator, ComputableBitsetType bitset_type = ComputableBitsetType::FastBitset)
        : AttributeInvertedInterface(allocator, bitset_type), field_2_value_map_(allocator){};

    ~AttributeBucketInvertedDataCell() override = default;

    void
    Insert(const AttributeSet& attr_set, InnerIdType inner_id, BucketIdType bucket_id) override;

    std::vector<const MultiBitsetManager*>
    GetBitsetsByAttr(const Attribute& attr) override;

    void
    UpdateBitsetsByAttr(const AttributeSet& attributes,
                        const InnerIdType offset_id,
                        const BucketIdType bucket_id) override;

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

private:
    UnorderedMap<std::string, ValueMapPtr> field_2_value_map_;

    std::shared_mutex global_mutex_{};
};

}  // namespace vsag
