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
#include <vector>

#include "basic_types.h"
#include "container_types.h"

namespace vsag {

class HGraph;
class SearchStatistics;

struct FGIMNeighbor {
    // CrossQuery returns source-local IDs; FGIMKnnGraph stores merged internal IDs.
    InnerIdType id;
    float distance;
};

// Own the result independently of the source indexes and their allocators.
using FGIMNeighborList = std::vector<FGIMNeighbor>;
using FGIMKnnGraph = std::vector<FGIMNeighborList>;

struct FGIMRefinementStatistics {
    uint64_t iterations{0};
    uint64_t distance_evaluations{0};
    uint64_t new_new_pairs{0};
    uint64_t new_old_pairs{0};
    uint64_t same_source_pairs_skipped{0};
    uint64_t neighbor_updates{0};
    uint64_t indegree_repairs{0};
};

class HGraphFGIM {
    friend class HGraphFGIMTestAccessor;

public:
    // Algorithm 5. Input rows are not modified; output rows remain distance-sorted.
    static FGIMKnnGraph
    KnnGraphToPG(const Vector<const HGraph*>& source_graphs,
                 const FGIMKnnGraph& graph,
                 uint64_t k,
                 uint64_t* distance_evaluations = nullptr);

    // Destination vectors and labels must already occupy the merged internal ID space.
    static void
    WriteBottomGraph(HGraph& destination, const FGIMKnnGraph& graph, uint64_t k);
    // Paper Algorithm 3: bounded persistent lists, unbounded temporary reverse candidates.
    // A zero iteration budget validates the graph but leaves it unchanged.
    static FGIMRefinementStatistics
    RefineKnnGraph(const Vector<const HGraph*>& source_graphs,
                   FGIMKnnGraph& graph,
                   uint64_t k,
                   uint64_t max_iterations);

    // Sources are non-owning, non-empty, disjoint HGraphs with successful Builds.
    // No concurrent mutation is allowed. Only float32/L2/fp32 without deletion
    // or duplicate grouping/deduplicate storage is supported. Result rows use merged internal IDs.
    static FGIMKnnGraph
    BuildInitialKnnGraph(const Vector<const HGraph*>& source_graphs, uint64_t k);

private:
    static void
    ValidateKnnGraph(const FGIMKnnGraph& graph, uint64_t k, uint64_t expected_count);

    // Internal-only search, not a public index API. The caller must validate the target
    // as FGIM-compatible and provide at least target.dim_ floats in query.
    // CrossQuery does not repeat full source validation.
    // Optional statistics are for diagnostics; normal FGIM runs do not collect them.
    static FGIMNeighborList
    CrossQuery(const HGraph& target,
               const float* query,
               int64_t l,
               SearchStatistics* stats = nullptr);

    static std::vector<InnerIdType>
    ValidateSourcesAndGetOffsets(const Vector<const HGraph*>& source_graphs, uint64_t k);

    static void
    AppendOriginalCandidates(const HGraph& source,
                             InnerIdType local_u,
                             InnerIdType source_offset,
                             FGIMNeighborList& candidates);
};

}  // namespace vsag
