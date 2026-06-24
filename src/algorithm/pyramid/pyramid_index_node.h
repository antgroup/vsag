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

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "datacell/graph_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/distance_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "utils/lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {

class IndexNode;
class ReasoningContext;
using SearchFunc = std::function<DistHeapPtr(const IndexNode* node, const VisitedListPtr& vl)>;

/**
 * @brief IndexNode: a tree node in the Pyramid hierarchy.
 *
 * Each IndexNode optionally holds a small graph (when the number of ids
 * exceeds index_min_size_) and a map of child nodes keyed by path segment.
 * The tree structure mirrors the hierarchical path labels (e.g. "a/b/c")
 * assigned to vectors at insertion time.
 */
class IndexNode {
public:
    enum class Status { NO_INDEX = 0, GRAPH = 1, FLAT = 2 };

public:
    IndexNode(Allocator* allocator_, GraphInterfaceParamPtr graph_param, uint32_t index_min_size);

    /// Build the internal graph using ODescent over the stored ids.
    void
    Build(ODescent& odescent);

    /// Allocate the graph storage if not yet done.
    void
    Init();

    /**
     * @brief Recursively search this node and its matching children.
     *
     * @param search_func  functor that searches a single node's graph;
     *                     typically bound to the caller's query and ef.
     * @param vl           visited-list for dedup across the recursion.
     * @param search_result  output heap accumulating candidates.
     * @param ef_search    expansion factor passed to the graph search.
     */
    void
    Search(const SearchFunc& search_func,
           const VisitedListPtr& vl,
           const DistHeapPtr& search_result,
           uint64_t ef_search,
           ReasoningContext* reasoning_ctx = nullptr) const;

    void
    AddChild(const std::string& key);

    IndexNode*
    GetChild(const std::string& key, bool need_init = false);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    friend class Pyramid;
    friend class PyramidAnalyzer;

public:
    GraphInterfacePtr graph_{nullptr};  // graph over the ids in this node
    InnerIdType entry_point_{0};        // entry point for graph search
    uint32_t level_{0};                 // depth in the tree (root = 0)
    mutable std::shared_mutex mutex_;   // per-node lock for concurrent add/search

    Vector<InnerIdType> ids_;          // internal ids stored at this node
    uint32_t index_min_size_{0};       // threshold to trigger graph build
    Status status_{Status::NO_INDEX};  // current build state

private:
    UnorderedMap<std::string, std::unique_ptr<IndexNode>> children_;  // keyed by path segment
    Allocator* allocator_{nullptr};
    GraphInterfaceParamPtr graph_param_{nullptr};
};

}  // namespace vsag
