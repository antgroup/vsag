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

#include <array>
#include <memory>

#include "attr/multi_bitset_manager.h"
#include "executor.h"

namespace vsag {

class RegionFilterExecutor : public Executor {
public:
    RegionFilterExecutor(Allocator* allocator,
                         const ExprPtr& expression,
                         const AttrInvertedInterfacePtr& attr_index);

    ~RegionFilterExecutor() override = default;

    void
    Clear() override;

    void
    Init() override;

    Filter*
    Run(BucketIdType bucket_id) override;

private:
    enum ManagerIndex : uint64_t {
        TRIGGER_OR_INTERSECTION = 0,
        RESIDENCE = 1,
        REGION = 2,
        REGION_OR_RESIDENCE = 3,
        REGION_AND_RESIDENCE = 4,
        REGION_WITHOUT_RESIDENCE = 5,
        MATCH_ALL = 6,           // Manager slot for stored type -1, after types 0 through 5.
        TYPE_COUNT = MATCH_ALL,  // Only types before MATCH_ALL need scratch bitmaps.
    };

    static std::vector<const MultiBitsetManager*>
    GetIntegerManagers(const AttrInvertedInterfacePtr& attr_index,
                       const std::string& field_name,
                       const std::vector<int64_t>& values);

    ComputableBitset*
    NewBitset();

    static void
    Accumulate(ComputableBitset* target,
               const std::vector<const MultiBitsetManager*>& managers,
               BucketIdType bucket_id);

    std::vector<const MultiBitsetManager*> region_type_managers_;
    std::vector<const MultiBitsetManager*> region_managers_;
    std::vector<const MultiBitsetManager*> residence_managers_;
    std::vector<const MultiBitsetManager*> trigger_managers_;

    std::unique_ptr<ComputableBitset> region_bitset_;
    std::unique_ptr<ComputableBitset> residence_bitset_;
    std::unique_ptr<ComputableBitset> trigger_bitset_;
    std::unique_ptr<ComputableBitset> intersection_bitset_;
    std::array<std::unique_ptr<ComputableBitset>, TYPE_COUNT> type_bitsets_;
};

}  // namespace vsag
