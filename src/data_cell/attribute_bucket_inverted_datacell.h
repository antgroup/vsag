
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
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) override;

private:
    template <class T>
    void
    insert_by_type(ValueMapPtr& value_map,
                   const Attribute* attr,
                   InnerIdType inner_id,
                   BucketIdType bucket_id);

    template <class T>
    void
    get_bitsets_by_type(const ValueMapPtr& value_map,
                        const Attribute* attr,
                        std::vector<const MultiBitsetManager*>& bitsets);

private:
    UnorderedMap<std::string, ValueMapPtr> field_2_value_map_;

    std::shared_mutex global_mutex_{};
};

template <class T>
void
AttributeBucketInvertedDataCell::insert_by_type(ValueMapPtr& value_map,
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

template <class T>
void
AttributeBucketInvertedDataCell::get_bitsets_by_type(
    const ValueMapPtr& value_map,
    const Attribute* attr,
    std::vector<const MultiBitsetManager*>& bitsets) {
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

}  // namespace vsag
