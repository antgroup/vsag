
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

#include "data_cell/flatten_datacell.h"
#include "data_cell/graph_datacell.h"
#include "data_cell/sparse_graph_datacell.h"
#include "diskann_logger.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/odescent_graph_parameter.h"
#include "logger.h"
#include "simd/simd.h"
#include "utils.h"
#include "vsag/dataset.h"

namespace vsag {

struct Node {
    bool old = false;
    uint32_t id;
    float distance;

    Node(uint32_t id, float distance) {
        this->id = id;
        this->distance = distance;
    }

    Node(uint32_t id, float distance, bool old) {
        this->id = id;
        this->distance = distance;
        this->old = old;
    }
    Node() {
    }

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

    bool
    operator==(const Node& other) const {
        return id == other.id;
    }
};

struct Linklist {
    Vector<Node> neighbors;
    float greast_neighbor_distance;
    Linklist(Allocator* allocator)
        : neighbors(allocator), greast_neighbor_distance(std::numeric_limits<float>::max()) {
    }
};

class ODescent {
public:
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

    bool
    Build(const GraphInterfacePtr& graph_storage = nullptr) {
        return Build(Vector<InnerIdType>(allocator_), graph_storage);
    }

    bool
    Build(const Vector<InnerIdType>& ids_sequence,
          const GraphInterfacePtr& graph_storage = nullptr);

    void
    SaveGraph(std::stringstream& out);

    void
    SaveGraph(GraphInterfacePtr& graph_storage);

private:
    inline float
    get_distance(uint32_t loc1, uint32_t loc2) {
        if (valid_ids_ != nullptr) {
            return flatten_interface_->ComputePairVectors(valid_ids_[loc1], valid_ids_[loc2]);
        }
        return flatten_interface_->ComputePairVectors(loc1, loc2);
    }

    void
    init_one_edge(int64_t i,
                  const GraphInterfacePtr& graph_storage,
                  const std::function<uint32_t(uint32_t)>& id_map_func,
                  std::uniform_int_distribution<int64_t>& k_generate,
                  std::mt19937& rng);

    void
    init_graph(const GraphInterfacePtr& graph_storage);

    void
    update_neighbors(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                     Vector<UnorderedSet<uint32_t>>& new_neighbors);

    void
    add_reverse_edges();

    void
    sample_candidates(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                      Vector<UnorderedSet<uint32_t>>& new_neighbors,
                      float sample_rate);

    void
    repair_no_in_edge();

    void
    prune_graph();

private:
    void
    parallelize_task(const std::function<void(int64_t i, int64_t end)>& task);

    size_t dim_;
    int64_t data_num_;
    Vector<Linklist> graph_;
    Vector<std::mutex> points_lock_;
    SafeThreadPool* thread_pool_{nullptr};

    const InnerIdType* valid_ids_{nullptr};

    bool pruning_{true};
    Allocator* const allocator_;

    const ODescentParameterPtr odescent_param_;

    const FlattenInterfacePtr& flatten_interface_;
};

}  // namespace vsag
