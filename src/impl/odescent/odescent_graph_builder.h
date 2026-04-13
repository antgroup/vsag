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

#include <iostream>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "datacell/graph_datacell.h"
#include "datacell/sparse_graph_datacell.h"
#include "diskann_logger.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/logger/logger.h"
#include "odescent_graph_parameter.h"
#include "simd/simd.h"
#include "utils.h"
#include "vsag/dataset.h"

namespace vsag {

/**
 * @file odescent_graph_builder.h
 * @brief ODescent algorithm for graph-based approximate nearest neighbor search.
 */

/**
 * @brief Node representation for priority queue in graph traversal.
 *
 * Node stores neighbor information including ID, distance, and a flag
 * indicating whether the edge is from an older iteration.
 */
struct Node {
    /// Flag indicating if the edge is from a previous iteration.
    bool old = false;

    /// Node identifier.
    uint32_t id;

    /// Distance to the query or reference point.
    float distance;

    /**
     * @brief Constructs a Node with id and distance.
     *
     * @param id Node identifier.
     * @param distance Distance value.
     */
    Node(uint32_t id, float distance) {
        this->id = id;
        this->distance = distance;
    }

    /**
     * @brief Constructs a Node with id, distance, and old flag.
     *
     * @param id Node identifier.
     * @param distance Distance value.
     * @param old Whether the edge is from a previous iteration.
     */
    Node(uint32_t id, float distance, bool old) {
        this->id = id;
        this->distance = distance;
        this->old = old;
    }

    /**
     * @brief Default constructor.
     */
    Node() {
    }

    /**
     * @brief Comparison operator for priority queue ordering.
     *
     * @param other Another node to compare with.
     * @return true if this node should come before other.
     */
    bool
    operator<(const Node& other) const {
        if (distance != other.distance) {
            return distance < other.distance;
        }
        if (id != other.id) {
            return id < other.id;
        }
        return old && not other.old;
    }

    /**
     * @brief Equality comparison operator.
     *
     * @param other Another node to compare with.
     * @return true if nodes have the same id.
     */
    bool
    operator==(const Node& other) const {
        return id == other.id;
    }
};

/**
 * @brief Neighbor list for a node in the proximity graph.
 *
 * Linklist maintains the list of neighbors for a node along with
 * the greatest neighbor distance for pruning decisions.
 */
struct Linklist {
    /// Vector of neighbor nodes.
    Vector<Node> neighbors;

    /// Greatest distance among neighbors (used for pruning).
    float greast_neighbor_distance;

    /**
     * @brief Constructs a Linklist with the given allocator.
     *
     * @param allocator Allocator for neighbor vector.
     */
    Linklist(Allocator* allocator)
        : neighbors(allocator), greast_neighbor_distance(std::numeric_limits<float>::max()) {
    }
};

/**
 * @brief ODescent algorithm implementation for proximity graph construction.
 *
 * ODescent builds a proximity graph by iteratively sampling candidate
 * neighbors and optimizing the graph structure. It supports parallel
 * processing and pruning of edges to control graph degree.
 */
class ODescent {
public:
    /**
     * @brief Constructs an ODescent builder with the given parameters.
     *
     * @param odescent_parameter Algorithm configuration parameters.
     * @param flatten_interface Interface for vector data access.
     * @param allocator Allocator for memory management.
     * @param thread_pool Thread pool for parallel processing.
     * @param pruning Whether to enable edge pruning.
     */
    ODescent(ODescentParameterPtr odescent_parameter,
             const FlattenInterfacePtr& flatten_interface,
             Allocator* allocator,
             SafeThreadPool* thread_pool,
             bool pruning = true)
        : odescent_param_(std::move(odescent_parameter)),
          flatten_interface_(flatten_interface),
          pruning_(pruning),
          allocator_(allocator),
          graph_(allocator),
          points_lock_(allocator),
          thread_pool_(thread_pool) {
    }

    /**
     * @brief Builds the proximity graph for all vectors.
     *
     * @param graph_storage Optional storage to save the built graph.
     * @return true on success, false on failure.
     */
    bool
    Build(const GraphInterfacePtr& graph_storage = nullptr) {
        return Build(Vector<InnerIdType>(allocator_), graph_storage);
    }

    /**
     * @brief Builds the proximity graph for specified vectors.
     *
     * @param ids_sequence Vector of IDs to include in the graph.
     * @param graph_storage Optional storage to save the built graph.
     * @return true on success, false on failure.
     */
    bool
    Build(const Vector<InnerIdType>& ids_sequence,
          const GraphInterfacePtr& graph_storage = nullptr);

    /**
     * @brief Saves the graph to a stream.
     *
     * @param out Output stream for graph data.
     */
    void
    SaveGraph(std::stringstream& out);

    /**
     * @brief Saves the graph to a storage interface.
     *
     * @param graph_storage Graph storage interface to save to.
     */
    void
    SaveGraph(GraphInterfacePtr& graph_storage);

    /**
     * @brief Sets the maximum degree for graph nodes.
     *
     * @param max_degree Maximum degree value.
     */
    void
    SetMaxDegree(int32_t max_degree) {
        odescent_param_->max_degree = max_degree;
    }

private:
    /**
     * @brief Computes distance between two vectors.
     *
     * @param loc1 Index of first vector.
     * @param loc2 Index of second vector.
     * @return Distance between the vectors.
     */
    inline float
    get_distance(uint32_t loc1, uint32_t loc2) {
        if (valid_ids_ != nullptr) {
            return flatten_interface_->ComputePairVectors(valid_ids_[loc1], valid_ids_[loc2]);
        }
        return flatten_interface_->ComputePairVectors(loc1, loc2);
    }

    /**
     * @brief Initializes a single edge for a node.
     *
     * @param i Node index.
     * @param graph_storage Graph storage for existing edges.
     * @param id_map_func Function to map indices.
     * @param k_generate Random distribution for neighbor selection.
     * @param rng Random number generator.
     */
    void
    init_one_edge(int64_t i,
                  const GraphInterfacePtr& graph_storage,
                  const std::function<uint32_t(uint32_t)>& id_map_func,
                  std::uniform_int_distribution<int64_t>& k_generate,
                  std::mt19937& rng);

    /**
     * @brief Initializes the graph structure.
     *
     * @param graph_storage Graph storage for existing edges.
     */
    void
    init_graph(const GraphInterfacePtr& graph_storage);

    /**
     * @brief Updates neighbor lists with new candidates.
     *
     * @param old_neighbors Old neighbor sets to update.
     * @param new_neighbors New neighbor sets to merge.
     */
    void
    update_neighbors(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                     Vector<UnorderedSet<uint32_t>>& new_neighbors);

    /**
     * @brief Adds reverse edges to ensure connectivity.
     */
    void
    add_reverse_edges();

    /**
     * @brief Samples candidate neighbors from current graph.
     *
     * @param old_neighbors Old neighbor sets to sample from.
     * @param new_neighbors New neighbor sets to populate.
     * @param sample_rate Rate for sampling neighbors.
     */
    void
    sample_candidates(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                      Vector<UnorderedSet<uint32_t>>& new_neighbors,
                      float sample_rate);

    /**
     * @brief Repairs nodes with no incoming edges.
     */
    void
    repair_no_in_edge();

    /**
     * @brief Prunes edges to control graph degree.
     */
    void
    prune_graph();

private:
    /**
     * @brief Parallelizes a task across multiple threads.
     *
     * @param task Function to execute for each range of indices.
     */
    void
    parallelize_task(const std::function<void(int64_t i, int64_t end)>& task);

    /// Vector dimension.
    uint64_t dim_;

    /// Number of data points.
    int64_t data_num_;

    /// Graph adjacency lists.
    Vector<Linklist> graph_;

    /// Mutex for each point during parallel updates.
    Vector<std::mutex> points_lock_;

    /// Thread pool for parallel execution.
    SafeThreadPool* thread_pool_{nullptr};

    /// Pointer to valid ID array (for partial builds).
    const InnerIdType* valid_ids_{nullptr};

    /// Whether to enable edge pruning.
    bool pruning_{true};

    /// Allocator for memory management.
    Allocator* const allocator_;

    /// Algorithm parameters.
    const ODescentParameterPtr odescent_param_;

    /// Interface for vector data access.
    const FlattenInterfacePtr& flatten_interface_;
};

}  // namespace vsag