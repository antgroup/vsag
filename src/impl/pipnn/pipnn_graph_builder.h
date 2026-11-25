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
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "data_type.h"
#include "json_wrapper.h"
#include "metric_type.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

struct PiPNNIncrementalState;
struct PiPNNProjectionState;
class FlattenInterface;
class Allocator;
class SafeThreadPool;
struct PiPNNBuildCache;
DEFINE_POINTER(GraphInterface);

/** Internal PiPNN policy shared by indexes that use the batch graph builder. */
struct PiPNNGraphBuilderParameter {
    uint64_t max_leaf_size{1024};
    uint64_t min_leaf_size{64};
    float leader_sample_rate{0.005F};
    std::vector<uint64_t> fanout{10, 2};
    uint64_t leaf_neighbor_count{5};
    uint64_t hash_plane_count{12};
    uint64_t reservoir_size{64};
    float alpha{1.0F};
    bool incremental_add{false};
    // Use symmetric distances between stored codes during Add. Build borrows its FP32 input.
    bool add_use_codes{false};
    bool adaptive_pruning{false};
    float adaptive_pruning_adjust_step{0.06F};

    void
    FromJson(const JsonType& json);

    JsonType
    ToJson() const;

    void
    Validate(uint64_t max_degree) const;
    void
    ValidateAddDistanceCodes(FlattenInterface& codes) const;
};

/**
 * One-shot PiPNN graph builder.
 *
 * Implements the construction approach described in "PiPNN: Ultra-Scalable Graph-Based Nearest
 * Neighbor Indexing" (arXiv:2602.21247).
 *
 * The builder consumes normalized internal IDs and dense, in-memory FP32, FP16, or BF16 codes. It owns only
 * graph construction; labels, source IDs, metadata, route layers, and serialization stay in the
 * index layer.
 */
class PiPNNGraphBuilder {
public:
    PiPNNGraphBuilder(PiPNNGraphBuilderParameter parameter,
                      uint64_t dimensions,
                      MetricType metric,
                      Allocator* allocator,
                      SafeThreadPool* thread_pool = nullptr,
                      uint64_t thread_count = 1);

    void
    Build(const GraphInterfacePtr& graph,
          const Vector<InnerIdType>& ids_sequence,
          const Vector<const float*>& rows,
          const std::vector<std::string>* source_ids = nullptr,
          const PiPNNBuildCache* imported_cache = nullptr,
          std::shared_ptr<PiPNNBuildCache>* exported_cache = nullptr) const;

    /** Native half rows are borrowed, subset-aligned with ids_sequence, and never indexed by ID. */
    void
    Build(const GraphInterfacePtr& graph,
          const Vector<InnerIdType>& ids_sequence,
          const Vector<const uint16_t*>& rows,
          DataTypes data_type) const;

    void
    Update(const GraphInterfacePtr&,
           const Vector<InnerIdType>& batch_ids,
           const Vector<const float*>& batch_vectors,
           PiPNNIncrementalState& state) const;

    void
    UpdateFromSharedRows(const GraphInterfacePtr& graph,
                         const Vector<InnerIdType>& ids,
                         std::span<const float* const> rows_by_id,
                         PiPNNIncrementalState& state) const;

    void
    AppendProjections(PiPNNProjectionState& sketches, std::span<const float* const> rows) const;

    void
    UpdateWithCodes(const GraphInterfacePtr& graph,
                    const Vector<InnerIdType>& ids,
                    const Vector<const float*>& input,
                    PiPNNIncrementalState& state,
                    FlattenInterface& codes,
                    const PiPNNProjectionState* shared = nullptr) const;

private:
    void
    update(const GraphInterfacePtr& graph,
           const Vector<InnerIdType>& ids,
           const Vector<const float*>& input,
           PiPNNIncrementalState& state,
           std::span<const float* const> rows_by_id,
           FlattenInterface* codes = nullptr,
           const PiPNNProjectionState* shared = nullptr) const;

    template <typename ElementType>
    void
    build_typed(const GraphInterfacePtr& graph,
                const Vector<InnerIdType>& ids_sequence,
                const Vector<const ElementType*>& rows,
                DataTypes data_type,
                const std::vector<std::string>* source_ids,
                const PiPNNBuildCache* imported_cache,
                std::shared_ptr<PiPNNBuildCache>* exported_cache) const;

    PiPNNGraphBuilderParameter parameter_;
    uint64_t dimensions_;
    MetricType metric_;
    Allocator* allocator_;
    SafeThreadPool* thread_pool_;
    uint64_t thread_count_;
};

}  // namespace vsag
