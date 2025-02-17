
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
#include "logger.h"
#include "safe_allocator.h"
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
    Linklist(Allocator*& allocator)
        : neighbors(allocator), greast_neighbor_distance(std::numeric_limits<float>::max()) {
    }
};
class ODescent {
public:
    ODescent(uint64_t max_degree,
             float alpha,
             int64_t turn,
             float sample_rate,
             const FlattenInterfacePtr& flatten_interface,
             Allocator* allocator,
             SafeThreadPool* thread_pool,
             bool pruning = true)
        : max_degree_(max_degree),
          alpha_(alpha),
          turn_(turn),
          sample_rate_(sample_rate),
          flatten_interface_(flatten_interface),
          pruning_(pruning),
          allocator_(allocator),
          graph(allocator),
          points_lock_(allocator),
          thread_pool_(thread_pool) {
    }

    bool
    Build(const uint32_t* valid_ids = nullptr, uint64_t data_num = 0);

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
    init_graph();

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
    parallelize_task(std::function<void(int64_t i, int64_t end)> task);

    size_t dim_;
    uint64_t data_num_;
    bool is_build_ = false;

    uint64_t max_degree_;
    float alpha_;
    int64_t turn_;
    Vector<Linklist> graph;
    uint64_t min_in_degree_ = 1;
    uint64_t block_size_{10000};
    Vector<std::mutex> points_lock_;
    SafeThreadPool* thread_pool_;

    const uint32_t* valid_ids_{nullptr};

    bool pruning_{true};
    float sample_rate_{0.3};
    Allocator* allocator_;

    const FlattenInterfacePtr& flatten_interface_;
};

}  // namespace vsag
