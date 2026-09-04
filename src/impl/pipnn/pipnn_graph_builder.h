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

#include <cstdint>
#include <vector>

#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

class Allocator;
class SafeThreadPool;
DEFINE_POINTER(GraphInterface);

/** Internal PiPNN policy shared by indexes that use the batch graph builder. */
struct PiPNNGraphBuilderParameter {
    uint64_t max_leaf_size{512};
    uint64_t min_leaf_size{64};
    float leader_sample_rate{0.005F};
    std::vector<uint64_t> fanout{10, 2};
    uint64_t leaf_neighbor_count{4};
    uint64_t hash_plane_count{12};
    uint64_t reservoir_size{64};
    float alpha{1.0F};

    void
    Validate(uint64_t max_degree) const;
};

/**
 * One-shot PiPNN graph builder.
 *
 * Implements the construction approach described in "PiPNN: Ultra-Scalable Graph-Based Nearest
 * Neighbor Indexing" (arXiv:2602.21247).
 *
 * The builder consumes normalized internal IDs and dense, in-memory FP32 codes. It owns only
 * graph construction; labels, source IDs, metadata, route layers, and serialization stay in the
 * index layer.
 */
class PiPNNGraphBuilder {
public:
    PiPNNGraphBuilder(PiPNNGraphBuilderParameter parameter,
                      uint64_t dimensions,
                      Allocator* allocator,
                      SafeThreadPool* thread_pool = nullptr,
                      uint64_t thread_count = 1);

    void
    Build(const GraphInterfacePtr& graph,
          const Vector<InnerIdType>& ids_sequence,
          const Vector<const float*>& rows) const;

private:
    PiPNNGraphBuilderParameter parameter_;
    uint64_t dimensions_;
    Allocator* allocator_;
    SafeThreadPool* thread_pool_;
    uint64_t thread_count_;
};

}  // namespace vsag
