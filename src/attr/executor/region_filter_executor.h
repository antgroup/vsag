
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

#include "attr/attr_value_map.h"
#include "attr/multi_bitset_manager.h"
#include "executor.h"
#include "vsag/attribute.h"

namespace vsag {
class RegionFilterExecutor : public Executor {
public:
    explicit RegionFilterExecutor(Allocator* allocator,
                                  const ExprPtr& expr,
                                  const AttrInvertedInterfacePtr& attr_index);

    void
    Clear() override;

    void
    Init() override;

    Filter*
    Run(BucketIdType bucket_id) override;

private:
    static const AttributeValue<int16_t>
    init_type_query();

private:
    static const std::string region_type_field_name;
    static const std::string region_list_field_name;
    static const std::string residence_list_field_name;

    static const AttributeValue<int16_t> type_query_attr;

    ComputableBitset* region_bitset_{nullptr};
    ComputableBitset* residence_bitset_{nullptr};
    ComputableBitset* trigger_bitset_{nullptr};

    std::vector<ComputableBitset*> type_bitsets_{6, nullptr};

    std::vector<const MultiBitsetManager*> region_type_managers_;

    std::vector<const MultiBitsetManager*> region_list_managers_;
    std::vector<const MultiBitsetManager*> residence_list_managers_;
    std::vector<const MultiBitsetManager*> trigger_list_managers_;
};

}  // namespace vsag
