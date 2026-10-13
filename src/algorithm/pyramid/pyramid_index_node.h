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

#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "datacell/flatten_interface.h"
#include "datacell/graph_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/heap/distance_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "index_common_param.h"
#include "utils/lock_strategy.h"
#include "utils/visited_list.h"

namespace vsag {

class IndexNode;
class ReasoningContext;
using SearchFunc = std::function<DistHeapPtr(const IndexNode* node, const VisitedListPtr& vl)>;
using GraphBuildFunc =
    std::function<void(GraphInterfacePtr& graph, const Vector<InnerIdType>& ids, uint32_t level)>;

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
    IndexNode(Allocator* allocator,
              GraphInterfaceParamPtr graph_param,
              uint32_t index_min_size,
              const IndexCommonParam& common_param,
              GraphInterfaceParamPtr child_graph_param);

    /// Build the internal graph using ODescent over the stored ids.
    void
    Build(ODescent& odescent);

    /// Build the internal graph using a batch builder selected by the caller.
    void
    Build(const GraphBuildFunc& build_graph);

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
    friend class PyramidDuplicateTestPeer;

public:
    GraphInterfacePtr graph_{nullptr};  // graph over the ids in this node
    // Bottom entry when no route exists; otherwise the highest route entry. A routed entry is also
    // a physical member of the complete bottom graph.
    InnerIdType entry_point_{0};
    uint32_t level_{0};  // depth in the tree (root = 0)
    // Guards node topology metadata: status, graph ownership, routing graph vector, and entry.
    // Graph adjacency rows remain protected by Pyramid::points_mutex_.
    mutable std::shared_mutex mutex_;

    Vector<InnerIdType> ids_;          // internal ids stored at this node
    uint32_t index_min_size_{0};       // threshold to trigger graph build
    Status status_{Status::NO_INDEX};  // current build state

private:
    // Exact code admission uses immutable construction codes, not copied vectors.
    struct DuplicateAdmission {
        using Key = std::pair<uint64_t, InnerIdType>;
        struct CodeLess {
            FlattenInterfacePtr codes;
            bool
            operator()(const Key& lhs, const Key& rhs) const {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                auto left = codes->AcquireCodesById(lhs.second);
                auto right = codes->AcquireCodesById(rhs.second);
                CHECK_ARGUMENT(left && right, "Pyramid duplicate admission requires codes");
                return std::memcmp(left.Data(), right.Data(), codes->code_size_) < 0;
            }
        };
        struct Stripe {
            static constexpr uint64_t kStripeCount = 64;
            std::mutex mutex;
            using Entry = std::pair<const Key, InnerIdType>;
            std::map<Key, InnerIdType, CodeLess, AllocatorWrapper<Entry>> representatives;
            Stripe(const FlattenInterfacePtr& codes, Allocator* allocator)
                : representatives(CodeLess{codes}, AllocatorWrapper<Entry>(allocator)) {
            }
        };
        // Existing encoded IDs are immutable: Add trains only an empty index,
        // and Pyramid removal marks labels without moving or rewriting codes.
        DuplicateAdmission(const FlattenInterfacePtr& codes, Allocator* allocator) {
            for (auto& stripe : stripes) {
                stripe = std::make_unique<Stripe>(codes, allocator);
            }
        }
        std::array<std::unique_ptr<Stripe>, Stripe::kStripeCount> stripes;
        std::atomic<uint64_t> representative_count{0};
    };
    std::unique_ptr<DuplicateAdmission> duplicate_admission_;

    class RoutingOverlay {
    public:
        RoutingOverlay(Allocator* allocator, GraphInterfaceParamPtr param)
            : graphs(allocator), graph_param(std::move(param)) {
        }

        Vector<GraphInterfacePtr> graphs;
        GraphInterfaceParamPtr graph_param{nullptr};
    };

    void
    enable_routing(GraphInterfaceParamPtr graph_param);

    [[nodiscard]] bool
    has_routing() const {
        return routing_ != nullptr;
    }

    GraphInterfacePtr
    make_route_graph() const;

    void
    serialize_routing_unlocked(StreamWriter& writer) const;

    void
    deserialize_routing_unlocked(StreamReader& reader);

    std::pair<uint64_t, uint64_t>
    get_memory_usage_detail() const;

    JsonType
    get_graph_stats() const;

    Vector<InnerIdType>
    get_ids_unlocked() const;

    void
    resize_graph(InnerIdType new_capacity);

    UnorderedMap<std::string, std::unique_ptr<IndexNode>> children_;  // keyed by path segment
    Allocator* allocator_{nullptr};
    const IndexCommonParam& common_param_;
    GraphInterfaceParamPtr graph_param_{nullptr};
    GraphInterfaceParamPtr child_graph_param_{nullptr};
    InnerIdType graph_capacity_{0};
    std::unique_ptr<RoutingOverlay> routing_{nullptr};
};

}  // namespace vsag
