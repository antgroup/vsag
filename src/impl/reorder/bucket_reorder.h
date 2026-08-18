// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <functional>
#include <utility>

#include "datacell/bucket_interface.h"
#include "impl/reorder/reorder.h"

namespace vsag {

class BucketReorder final : public ReorderInterface {
public:
    using LocationResolver =
        std::function<std::pair<BucketIdType, InnerIdType>(InnerIdType inner_id)>;

    BucketReorder(BucketInterfacePtr bucket,
                  LocationResolver location_resolver,
                  Allocator* allocator)
        : bucket_(std::move(bucket)),
          location_resolver_(std::move(location_resolver)),
          allocator_(allocator) {
    }

    DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const void* query,
            int64_t topk,
            QueryContext& ctx,
            IteratorFilterContext* iter_ctx = nullptr,
            const DistanceRecordVector* rabitq_lower_bound_candidates = nullptr,
            const std::optional<float>& distance_threshold = std::nullopt) override;

private:
    const BucketInterfacePtr bucket_;
    const LocationResolver location_resolver_;
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
