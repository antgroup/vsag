
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

#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/reorder.h"
#include "utils/pointer_define.h"

namespace vsag {
class FlattenReorder : public ReorderInterface {
public:
    FlattenReorder(const FlattenInterfacePtr& flatten, Allocator* allocator)
        : flatten_(flatten), allocator_(allocator) {
    }

    DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const float* query,
            int64_t topk,
            Allocator* allocator = nullptr) override;

private:
    const FlattenInterfacePtr flatten_;
    Allocator* allocator_{nullptr};
};
}  // namespace vsag