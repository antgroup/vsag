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
#include <thread>
#include <unordered_set>
#include <vector>

#include "../logger.h"
#include "../simd/simd.h"
#include "../utils.h"
#include "vsag/dataset.h"
#include <ThreadPool.h>

namespace vsag {


class LinearCongruentialGenerator {
public:
    LinearCongruentialGenerator() {
        auto now = std::chrono::steady_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current_ = static_cast<unsigned int>(timestamp);
    }

    float
    NextFloat() {
        current_ = (A * current_ + C) % M;
        return static_cast<float>(current_) / static_cast<float>(M);
    }

private:
    unsigned int current_;
    static const uint32_t A = 1664525;
    static const uint32_t C = 1013904223;
    static const uint32_t M = 4294967295;  // 2^32 - 1
};



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
    std::vector<Node> neighbors;
    float greast_neighbor_distance = std::numeric_limits<float>::max();
};

class Graph {
public:
    virtual bool
    Build(const DatasetPtr dataset) = 0;

    virtual std::vector<std::vector<uint32_t>>
    GetGraph() = 0;

    virtual std::vector<std::vector<uint32_t>>
    GetHGraph(int level) {
        return {};
    }

    virtual int
    GetLevel() {
        return 0;
    }
};

class NNdescent : public Graph {
public:
    NNdescent(int64_t max_degree, int64_t turn, DistanceFunc distance)
        : max_degree_(max_degree), turn_(turn), distance_(distance) {
    }

    bool
    Build(const DatasetPtr dataset) override {
        if (is_build_) {
            return false;
        }
        is_build_ = true;
        this->thread_pool_ = std::make_shared<progschj::ThreadPool>(16);
        dim_ = dataset->GetDim();
        data_num_ = dataset->GetNumElements();
        data_ = dataset->GetFloat32Vectors();
        min_in_degree_ = std::min(min_in_degree_, data_num_ - 1);
        std::vector<std::mutex>(data_num_).swap(points_lock_);
        std::vector<std::unordered_set<uint32_t>> old_neighbors;
        std::vector<std::unordered_set<uint32_t>> new_neighbors;
        new_candidates_.resize(data_num_);
        old_neighbors.resize(data_num_);
        new_neighbors.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            new_candidates_[i].reserve(max_degree_);
            old_neighbors[i].reserve(max_degree_);
            new_neighbors[i].reserve(max_degree_);
        }
        init_graph();
        check_turn();
        {
            for (int i = 0; i < turn_; ++i) {
                sample_candidates(old_neighbors, new_neighbors, 0.3);
                update_neighbors(old_neighbors, new_neighbors);
                repair_no_in_edge();
                check_turn();
            }
            prune_graph();
            add_reverse_edges();
            repair_no_in_edge();
            check_turn();
        }
        return true;
    }

    std::vector<std::vector<uint32_t>>
    GetGraph() override {
        std::vector<std::vector<uint32_t>> extract_graph;
        extract_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            extract_graph[i].resize(graph[i].neighbors.size());
            for (int j = 0; j < graph[i].neighbors.size(); ++j) {
                extract_graph[i][j] = graph[i].neighbors[j].id;
            }
        }

        return extract_graph;
    }

private:
    inline float
    get_distance(uint32_t loc1, uint32_t loc2) {
        return distance_(get_data_by_loc(loc1), get_data_by_loc(loc2), &dim_);
    }

    inline const float*
    get_data_by_loc(uint32_t loc) {
        return data_ + loc * dim_;
    }

    void
    init_graph() {
        graph.resize(data_num_);
        visited_.resize(data_num_);
        std::random_device rd;
        std::uniform_int_distribution<int> k_generate(0, data_num_ - 1);
        for (int i = 0; i < data_num_; ++i) {
            std::mt19937 rng(rd());
            std::unordered_set<uint32_t> ids_set;
            graph[i].neighbors.reserve(max_degree_);
            for (int j = 0; j < std::min(data_num_ - 1, max_degree_); ++j) {
                auto id = i;
                if (data_num_ - 1 < max_degree_) {
                    id = (i + j + 1) % data_num_;
                } else {
                    while (id == i || ids_set.find(id) != ids_set.end()) {
                        id = k_generate(rng);
                    }
                }
                ids_set.insert(id);
                auto dist = get_distance(i, id);
                graph[i].neighbors.emplace_back(id, dist);
                graph[i].greast_neighbor_distance =
                    std::min(graph[i].greast_neighbor_distance, dist);
                visited_[i] = false;
            }
        }
    }

    void
    update_neighbors(std::vector<std::unordered_set<uint32_t>>& old_neighbors,
                     std::vector<std::unordered_set<uint32_t>>& new_neighbors) {
        std::vector<std::future<void>> futures;
        auto task = [&, this](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                std::vector<uint32_t> new_neighbors_candidates;
                for (uint32_t node_id : new_neighbors[i]) {
                    for (unsigned int neighbor_id : new_neighbors_candidates) {
                        float dist = get_distance(node_id, neighbor_id);
                        if (dist < graph[node_id].greast_neighbor_distance) {
                            std::lock_guard<std::mutex> lock(points_lock_[node_id]);
                            graph[node_id].neighbors.emplace_back(neighbor_id, dist);
                        }
                        if (dist < graph[neighbor_id].greast_neighbor_distance) {
                            std::lock_guard<std::mutex> lock(points_lock_[neighbor_id]);
                            graph[neighbor_id].neighbors.emplace_back(node_id, dist);
                        }
                    }
                    new_neighbors_candidates.push_back(node_id);

                    for (uint32_t neighbor_id : old_neighbors[i]) {
                        if (node_id == neighbor_id) {
                            continue;
                        }
                        float dist = get_distance(neighbor_id, node_id);
                        if (dist < graph[node_id].greast_neighbor_distance) {
                            std::lock_guard<std::mutex> lock(points_lock_[node_id]);
                            graph[node_id].neighbors.emplace_back(neighbor_id, dist);
                        }
                        if (dist < graph[neighbor_id].greast_neighbor_distance) {
                            std::lock_guard<std::mutex> lock(points_lock_[neighbor_id]);
                            graph[neighbor_id].neighbors.emplace_back(node_id, dist);
                        }
                    }
                }
                old_neighbors[i].clear();
                new_neighbors[i].clear();
            }
        };
        parallelize_task(task);

        auto resize_task = [&, this](int64_t start, int64_t end) {
            for (uint32_t i = start; i < end; ++i) {
                auto& neighbors = graph[i].neighbors;
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                if (neighbors.size() > max_degree_) {
                    neighbors.resize(max_degree_);
                }
                graph[i].greast_neighbor_distance = neighbors.back().distance;
            }
        };
        parallelize_task(resize_task);
    }

    void
    add_reverse_edges() {
        std::vector<Linklist> reverse_graph;
        reverse_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            reverse_graph[i].neighbors.reserve(max_degree_);
        }
        for (int i = 0; i < data_num_; ++i) {
            for (const auto& node : graph[i].neighbors) {
                reverse_graph[node.id].neighbors.emplace_back(i, node.distance);
            }
        }

        auto task = [&, this](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                auto& neighbors = graph[i].neighbors;
                neighbors.insert(neighbors.end(),
                                 reverse_graph[i].neighbors.begin(),
                                 reverse_graph[i].neighbors.end());
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                if (neighbors.size() > max_degree_) {
                    neighbors.resize(max_degree_);
                }
            }
        };
        parallelize_task(task);
    }

    void
    sample_candidates(std::vector<std::unordered_set<uint32_t>>& old_neighbors,
                      std::vector<std::unordered_set<uint32_t>>& new_neighbors,
                      float sample_rate) {
        auto task = [&, this](int64_t start, int64_t end) {
            LinearCongruentialGenerator r;
            for (int64_t i = start; i < end; ++i) {
                auto& neighbors = graph[i].neighbors;
                for (auto& neighbor : neighbors) {
                    float current_state = r.NextFloat();
                    if (current_state < sample_rate) {
                        if (neighbor.old) {
                            {
                                std::lock_guard<std::mutex> lock(points_lock_[i]);
                                old_neighbors[i].insert(neighbor.id);
                            }
                            {
                                std::lock_guard<std::mutex> inner_lock(points_lock_[neighbor.id]);
                                old_neighbors[neighbor.id].insert(i);
                            }
                        } else {
                            {
                                std::lock_guard<std::mutex> lock(points_lock_[i]);
                                new_neighbors[i].insert(neighbor.id);
                            }
                            {
                                std::lock_guard<std::mutex> inner_lock(points_lock_[neighbor.id]);
                                new_neighbors[neighbor.id].insert(i);
                            }
                            neighbor.old = true;
                        }
                    }
                }
            }
        };
        parallelize_task(task);
    }


    void
    repair_no_in_edge() {
        std::vector<int> in_edges_count(data_num_, 0);
        for (int i = 0; i < data_num_; ++i) {
            for (auto& neighbor : graph[i].neighbors) {
                in_edges_count[neighbor.id]++;
            }
        }

        std::vector<int> replace_pos(
            data_num_,
            static_cast<int32_t>(std::min(data_num_ - 1, max_degree_) - 1));
        auto min_in_degree = std::min(min_in_degree_, data_num_ - 1);
        for (int i = 0; i < data_num_; ++i) {
            auto& link = graph[i].neighbors;
            int need_replace_loc = 0;
            while (in_edges_count[i] < min_in_degree &&
                   need_replace_loc < max_degree_) {
                uint32_t need_replace_id = link[need_replace_loc].id;
                bool has_connect = false;
                for (auto& neighbor : graph[need_replace_id].neighbors) {
                    if (neighbor.id == i) {
                        has_connect = true;
                        break;
                    }
                }
                if (replace_pos[need_replace_id] > 0 && not has_connect) {
                    auto& replace_node =
                        graph[need_replace_id].neighbors[replace_pos[need_replace_id]];
                    auto replace_id = replace_node.id;
                    if (in_edges_count[replace_id] > min_in_degree) {
                        in_edges_count[replace_id]--;
                        replace_node.id = i;
                        replace_node.distance = link[need_replace_loc].distance;
                        in_edges_count[i]++;
                    }
                    replace_pos[need_replace_id]--;
                }
                need_replace_loc++;
            }
        }

    }

    void
    prune_graph() {
        std::vector<int> in_edges_count(data_num_, 0);
        for (int i = 0; i < data_num_; ++i) {
            for (auto& neighbor : graph[i].neighbors) {
                in_edges_count[neighbor.id]++;
            }
        }

        auto min_in_degree = std::min(min_in_degree_, data_num_ - 1);
        auto task = [&, this](int64_t start, int64_t end) {
            for (int64_t loc = start; loc < end; ++loc) {
                auto& neighbors = graph[loc].neighbors;
                std::sort(neighbors.begin(), neighbors.end());
                neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
                std::vector<Node> candidates;
                candidates.reserve(max_degree_);
                for (auto& neighbor : neighbors) {
                    bool flag = true;
                    int cur_in_edge = 0;
                    {
                        std::lock_guard<std::mutex> lock(points_lock_[neighbor.id]);
                        cur_in_edge = in_edges_count[neighbor.id];
                    }
                    if (cur_in_edge > min_in_degree) {
                        for (auto& candidate : candidates) {
                            if (get_distance(neighbor.id, candidate.id) * alpha_ <
                                neighbor.distance) {
                                flag = false;
                                {
                                    std::lock_guard<std::mutex> lock(points_lock_[neighbor.id]);
                                    in_edges_count[neighbor.id]--;
                                }
                                break;
                            }
                        }
                    }
                    if (flag) {
                        candidates.push_back(neighbor);
                    }
                }
                neighbors.swap(candidates);
                if (neighbors.size() > max_degree_) {
                    neighbors.resize(max_degree_);
                }
            }
        };
        parallelize_task(task);
    }

    void
    check_turn() {
        int edge_count = 0;
        float loss = 0;
        int no_in_edge_count = 0;

        std::vector<int> in_edges_count(data_num_, 0);
        for (int i = 0; i < data_num_; ++i) {
            //            std::cout <<"check: ";
            for (int j = 0; j < graph[i].neighbors.size(); ++j) {
                loss += graph[i].neighbors[j].distance;
                in_edges_count[graph[i].neighbors[j].id]++;
                //                                std::cout << graph[i].neighbors[j].distance << " ";
            }
            //                        std::cout << std::endl;
            edge_count += graph[i].neighbors.size();
        }
        for (int i = 0; i < data_num_; ++i) {
            if (in_edges_count[i] == 0) {
                no_in_edge_count++;
            }
        }

        std::vector<bool> visits(data_num_, false);
        int connect_count = 0;
        for (int i = 0; i < data_num_; ++i) {
            if (visits[i]) {
                continue;
            }
            std::vector<uint32_t> candidates;
            candidates.push_back(i);
            while (not candidates.empty()) {
                std::vector<uint32_t> new_candidates;
                for (unsigned int cur_node : candidates) {
                    for (auto& neigbor : graph[cur_node].neighbors) {
                        if (not visits[neigbor.id]) {
                            new_candidates.push_back(neigbor.id);
                            visits[neigbor.id] = true;
                        }
                    }
                }
                candidates.swap(new_candidates);
            }
            connect_count++;
        }
        loss /= edge_count;
        logger::info(
            fmt::format("loss:{} edge_count:{} no_in_edge_count:{}  connections:{} ",
                        loss,
                        edge_count,
                        no_in_edge_count,
                        connect_count));
    }

private:
    void
    parallelize_task(const std::function<void(int64_t i, int64_t end)>& task) {
        if (this->thread_pool_ != nullptr) {
            std::vector<std::future<void>> futures;
            for (int64_t i = 0; i < data_num_; i += block_size_) {
                int64_t end = std::min(i + block_size_, data_num_);
                futures.push_back(thread_pool_->enqueue(task, i, end));
            }
            for (auto& future : futures) {
                future.get();
            }
        } else {
            for (int64_t i = 0; i < data_num_; i += block_size_) {
                int64_t end = std::min(i + block_size_, data_num_);
                task(i, end);
            }
        }
    }

    size_t dim_;
    int64_t data_num_;
    int64_t is_build_ = false;
    const float* data_;

    int64_t max_degree_;
    int64_t turn_;
    std::vector<Linklist> graph;
    std::vector<bool> visited_;
    int64_t min_in_degree_ = 3;
    float alpha_ = 1.3;

    float all_calculate_ = 0;
    float valid_calculate_ = 0;
    float duplicate_rate = 0;
    int64_t block_size_ = 10000;
    std::vector<std::mutex> points_lock_;
    std::vector<std::vector<Node>> new_candidates_;
    std::vector<std::mutex> mutexs_;
    std::shared_ptr<progschj::ThreadPool> thread_pool_{nullptr};

    DistanceFunc distance_;
};

class HierarchicalGraph : public Graph {
public:
    HierarchicalGraph(int64_t max_degree, int64_t turn, DistanceFunc distance)
        : max_degree_(max_degree), turn_(turn), distance_(distance) {
    }

    bool
    Build(const DatasetPtr dataset) override {
        auto sub_dataset = Dataset::Make();
        sub_dataset->Owner(false)
            ->NumElements(dataset->GetNumElements())
            ->Float32Vectors(dataset->GetFloat32Vectors())
            ->Dim(dataset->GetDim());

        int current_size = dataset->GetNumElements();
        while (current_size) {
            std::cout << "build level:" << level_ << " has nodes:" << current_size << std::endl;
            h_graph_[level_] =
                std::make_shared<NNdescent>(max_degree_ * (level_ == 0 ? 2 : 1), turn_, distance_);
            h_graph_[level_]->Build(sub_dataset);
            current_size /= max_degree_;
            sub_dataset->NumElements(current_size);
            level_++;
        }
        return true;
    }

    std::vector<std::vector<uint32_t>>
    GetGraph() override {
        auto level_graph = GetHGraph(0);
        std::cout << "GetGraph in HNNDecent:" << level_graph.size() << std::endl;
        return level_graph;
    }

    std::vector<std::vector<uint32_t>>
    GetHGraph(int level) override {
        auto level_graph = h_graph_[level]->GetGraph();
        std::cout << "GetHGraph in HNNDecent:" << level_graph.size() << std::endl;
        return level_graph;
    }

    int
    GetLevel() override {
        return level_;
    }

private:
    size_t dim_;
    int64_t data_num_;
    const float* data_;

    int64_t max_degree_;
    int64_t turn_;
    std::vector<Linklist> graph;
    std::vector<bool> visited_;

    DistanceFunc distance_;

    std::unordered_map<int, std::shared_ptr<NNdescent>> h_graph_;
    int level_ = 0;
};

}  // namespace vsag