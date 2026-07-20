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

#include <cstdint>
#include <memory>

#include "container_types.h"
#include "datacell/bucket_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/inner_search_param.h"
#include "typing.h"

namespace vsag {

class ReasoningContext;
class ComputerInterface;
using ComputerInterfacePtr = std::shared_ptr<ComputerInterface>;

class IVFBucketSearcher {
public:
    virtual ~IVFBucketSearcher() = default;

    /// Search a single bucket for candidate vectors.
    /// Implementations must be thread-safe and must not mutate shared state
    /// unless synchronized. The heap and dist parameters are per-thread scratch.
    virtual void
    Search(BucketIdType bucket_id,
           const BucketInterfacePtr& bucket,
           const ComputerInterfacePtr& computer,
           const InnerSearchParam& param,
           int64_t thread_id,
           int64_t topk,
           BucketIdType buckets_per_data,
           DistHeapPtr& heap,
           Vector<float>& dist,
           ReasoningContext* reasoning_ctx) const = 0;
};

using IVFBucketSearcherPtr = std::shared_ptr<IVFBucketSearcher>;

}  // namespace vsag
