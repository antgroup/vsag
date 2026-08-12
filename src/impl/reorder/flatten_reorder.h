
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
#include "datacell/hgraph_rabitq_fused_datacell.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/reorder.h"
#include "utils/pointer_define.h"

namespace vsag {
class FlattenReorder : public ReorderInterface {
public:
    FlattenReorder(const FlattenInterfacePtr& flatten,
                   Allocator* allocator,
                   HGraphRaBitQFusedDataCellPtr fused_graph = nullptr)
        : flatten_(flatten), allocator_(allocator), fused_graph_(std::move(fused_graph)) {
    }

    DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const void* query,
            int64_t topk,
            QueryContext& ctx,
            IteratorFilterContext* iter_ctx = nullptr,
            const DistanceRecordVector* rabitq_lower_bound_candidates = nullptr,
            const std::optional<float>& distance_threshold = std::nullopt) override;

    DistHeapPtr
    ReorderFused(const DistHeapPtr& input,
                 const void* query,
                 int64_t topk,
                 QueryContext& ctx,
                 IteratorFilterContext* iter_ctx = nullptr,
                 const RaBitQCandidateVector* rabitq_lower_bound_candidates = nullptr,
                 const std::optional<float>& distance_threshold = std::nullopt);

private:
    void
    QueryLowerBound(float* distances,
                    float* lower_bounds,
                    float* filter_inner_products,
                    const ComputerInterfacePtr& computer,
                    const InnerIdType* ids,
                    uint64_t count,
                    QueryContext* ctx) const;

    void
    QueryFullWithHint(float* distances,
                      const float* filter_inner_products,
                      const ComputerInterfacePtr& computer,
                      const InnerIdType* ids,
                      uint64_t count,
                      QueryContext* ctx) const;

    const FlattenInterfacePtr flatten_;
    Allocator* allocator_{nullptr};
    HGraphRaBitQFusedDataCellPtr fused_graph_{nullptr};
};
}  // namespace vsag
