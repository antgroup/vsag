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

#include "algorithm/inner_index_interface.h"
#include "impl/heap/distance_heap.h"
#include "index/iterator_filter.h"
#include "utils/pointer_define.h"

namespace vsag {

/**
 * @file reorder.h
 * @brief Interface for result reordering in search pipelines.
 */

DEFINE_POINTER(ReorderInterface)

/**
 * @brief Abstract interface for reordering search results.
 *
 * ReorderInterface defines the contract for reordering components that
 * refine search results by recomputing distances or applying additional
 * criteria. Implementations can use different data representations
 * for distance computation.
 */
class ReorderInterface {
public:
    /**
     * @brief Reorders search results using more accurate distance computation.
     *
     * Takes a heap of candidate results and reorders them by computing
     * distances using a potentially more accurate representation.
     *
     * @param input Input heap containing candidate results.
     * @param query Query vector pointer.
     * @param topk Number of top results to return.
     * @param ctx Query context for execution.
     * @param iter_ctx Optional iterator filter context for progressive filtering.
     * @return DistHeapPtr Reordered heap containing refined results.
     */
    virtual DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const float* query,
            int64_t topk,
            QueryContext& ctx,
            IteratorFilterContext* iter_ctx = nullptr) = 0;
};

}  // namespace vsag