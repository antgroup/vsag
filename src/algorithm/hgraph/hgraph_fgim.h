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

#include <cstddef>
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

class HGraphFGIM {
public:
    // Sources are non-owning, non-empty, disjoint HGraphs with successful Builds.
    // No concurrent mutation is allowed. Only float32/L2/fp32 without deletion
    // or deduplicate storage is supported. Result rows use merged internal IDs.
    static FGIMKnnGraph
    BuildInitialKnnGraph(const Vector<const HGraph*>& source_graphs, std::size_t k);

    // Internal-only search, not a public index API. The caller must validate the target
    // as FGIM-compatible and provide at least target.dim_ floats in query.
    // CrossQuery does not repeat full source validation.
    // Optional statistics are for diagnostics; normal FGIM runs do not collect them.
    static FGIMNeighborList
    CrossQuery(const HGraph& target,
               const float* query,
               int64_t l,
               SearchStatistics* stats = nullptr);

private:
    static std::vector<InnerIdType>
    ValidateSourcesAndGetOffsets(const Vector<const HGraph*>& source_graphs, std::size_t k);

    static void
    AppendOriginalCandidates(const HGraph& source,
                             InnerIdType local_u,
                             InnerIdType source_offset,
                             FGIMNeighborList& candidates);
};

}  // namespace vsag
