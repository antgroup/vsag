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

#include "algorithm/inner_index_interface.h"
#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/reorder.h"
#include "utils/pointer_define.h"

namespace vsag {

/**
 * @file flatten_reorder.h
 * @brief Flatten-based implementation of result reordering.
 */

/**
 * @brief Reorders search results using flatten data cell for distance computation.
 *
 * FlattenReorder refines search results by recomputing distances using
 * the original or higher-precision vectors stored in a flatten data cell.
 * This provides more accurate distances than quantized representations.
 */
class FlattenReorder : public ReorderInterface {
public:
    /**
     * @brief Constructs a FlattenReorder with the given flatten interface.
     *
     * @param flatten Flatten interface for vector data access.
     * @param allocator Allocator for memory management.
     */
    FlattenReorder(const FlattenInterfacePtr& flatten, Allocator* allocator)
        : flatten_(flatten), allocator_(allocator) {
    }

    /**
     * @brief Reorders search results using flatten data for distance computation.
     *
     * @param input Input heap containing candidate results.
     * @param query Query vector pointer.
     * @param topk Number of top results to return.
     * @param ctx Query context for execution.
     * @param iter_ctx Optional iterator filter context for progressive filtering.
     * @return DistHeapPtr Reordered heap containing refined results.
     */
    DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const float* query,
            int64_t topk,
            QueryContext& ctx,
            IteratorFilterContext* iter_ctx = nullptr) override;

private:
    /// Flatten interface for accessing original vector data.
    const FlattenInterfacePtr flatten_;

    /// Allocator for memory management.
    Allocator* allocator_{nullptr};
};
}  // namespace vsag