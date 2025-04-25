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
    std::vector<Node> neigbors;
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

class RNNdescent : public Graph {
public:
    RNNdescent(int64_t max_degree, int64_t turn, DistanceFunc distance)
        : max_degree_(max_degree), turn_(turn), distance_(distance) {
    }

    bool
    Build(const DatasetPtr dataset) override {
        if (is_build_) {
            return false;
        }
        is_build_ = true;
        dim_ = dataset->GetDim();
        data_num_ = dataset->GetNumElements();
        data_ = dataset->GetFloat32Vectors();
        init_graph();
        check_turn();
        {
            SlowTaskTimer t("hnsw graph");
            for (int i = 0; i < 10; ++i) {
                update_neighbors();
                check_turn();
                if (i != 9) {
                    add_reverse_edges();
                }
            }
            for (int i = 0; i < data_num_; ++i) {
                reduce_graph(i);
            }
            check_turn();
        }
        return true;
    }

    std::vector<std::vector<uint32_t>>
    GetGraph() override {
        std::vector<std::vector<uint32_t>> extract_graph;
        extract_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            extract_graph[i].resize(graph[i].neigbors.size());
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                extract_graph[i][j] = graph[i].neigbors[j].id;
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
        std::random_device rd;
        std::uniform_int_distribution<int> k_generate(0, data_num_ - 1);
#pragma omp for
        for (int i = 0; i < data_num_; ++i) {
            std::mt19937 rng(rd());
            for (int j = 0; j < max_degree_; ++j) {
                auto id = k_generate(rng);
                graph[i].neigbors.emplace_back(id, get_distance(i, id));
            }
        }
    }

    void
    update_neighbors() {
        for (int i = 0; i < data_num_; ++i) {
            std::vector<Node> old_neighbors;
            { graph[i].neigbors.swap(old_neighbors); }
            std::sort(old_neighbors.begin(), old_neighbors.end());
            std::vector<Node> new_neighbors;
            uint32_t last_id = -1;
            for (int j = 0; j < old_neighbors.size(); ++j) {
                bool flag = true;
                if (j > 0 && last_id == old_neighbors[j].id) {
                    continue;
                }
                last_id = old_neighbors[j].id;
                for (int k = 0; k < new_neighbors.size(); ++k) {
                    if ((old_neighbors[j].old && new_neighbors[k].old)) {
                        continue;
                    }
                    if (old_neighbors[j].id == new_neighbors[k].id) {
                        break;
                    }
                    float d = get_distance(old_neighbors[j].id, new_neighbors[k].id);
                    if (d < old_neighbors[j].distance) {
                        flag = false;
                        {
                            graph[new_neighbors[k].id].neigbors.emplace_back(old_neighbors[j].id,
                                                                             d);
                        }
                        break;
                    }
                }
                if (flag) {
                    new_neighbors.push_back(old_neighbors[j]);
                }
            }
            for (int j = 0; j < new_neighbors.size(); ++j) {
                new_neighbors[j].old = true;
            }
            {
                graph[i].neigbors.insert(
                    graph[i].neigbors.end(), new_neighbors.begin(), new_neighbors.end());
                reduce_graph(i);
            }
        }
    }

    void
    add_reverse_edges() {
        std::vector<Linklist> reverse_graph;
        reverse_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                auto& node = graph[i].neigbors[j];
                reverse_graph[node.id].neigbors.emplace_back(i, node.distance);
            }
        }
        for (int i = 0; i < data_num_; ++i) {
            graph[i].neigbors.insert(graph[i].neigbors.end(),
                                     reverse_graph[i].neigbors.begin(),
                                     reverse_graph[i].neigbors.end());
            reduce_graph(i);
        }
    }

    void
    reduce_graph(uint32_t loc) {
        std::sort(graph[loc].neigbors.begin(), graph[loc].neigbors.end());
        graph[loc].neigbors.erase(
            std::unique(graph[loc].neigbors.begin(), graph[loc].neigbors.end()),
            graph[loc].neigbors.end());
        if (graph[loc].neigbors.size() > max_degree_) {
            graph[loc].neigbors.resize(max_degree_);
        }
    }

    void
    check_turn() {
        int edge_count = 0;
        float loss = 0;
        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                loss += graph[i].neigbors[j].distance;
                //                std::cout << graph[i].neigbors[j].distance << " ";
            }
            //            std::cout << std::endl;
            edge_count += graph[i].neigbors.size();
        }
        loss /= edge_count;
        std::cout << "loss:" << loss << "  edge_count:" << edge_count << std::endl;
    }

private:
    size_t dim_;
    int64_t data_num_;
    int64_t is_build_ = false;
    const float* data_;

    int64_t max_degree_;
    int64_t turn_;
    std::vector<Linklist> graph;

    DistanceFunc distance_;
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
        dim_ = dataset->GetDim();
        data_num_ = dataset->GetNumElements();
        data_ = dataset->GetFloat32Vectors();
        min_in_degree_ = std::min(min_in_degree_, data_num_ - 1);
        std::vector<std::vector<uint32_t>> old_neigbors;
        std::vector<std::vector<uint32_t>> new_neigbors;
        new_candidates_.resize(data_num_);
        old_neigbors.resize(data_num_);
        new_neigbors.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            new_candidates_[i].reserve(max_degree_);
            old_neigbors[i].reserve(max_degree_);
            new_neigbors[i].reserve(max_degree_);
        }
        init_graph();
        check_turn();
        {
            for (int i = 0; i < turn_; ++i) {
                sample_candidates(old_neigbors, new_neigbors, 0.3);
                update_neighbors(old_neigbors, new_neigbors);
                repair_no_in_edge();
                check_turn();
            }
            prune_graph();
            add_reverse_edges();
            check_turn();
        }
        return true;
    }

    std::vector<std::vector<uint32_t>>
    GetGraph() override {
        std::vector<std::vector<uint32_t>> extract_graph;
        extract_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            extract_graph[i].resize(graph[i].neigbors.size());
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                extract_graph[i][j] = graph[i].neigbors[j].id;
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
            graph[i].neigbors.reserve(max_degree_);
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
                graph[i].neigbors.emplace_back(id, dist);
                graph[i].greast_neighbor_distance =
                    std::min(graph[i].greast_neighbor_distance, dist);
                visited_[i] = false;
            }
        }
    }

    void
    update_neighbors(std::vector<std::vector<uint32_t>>& old_neigbors,
                     std::vector<std::vector<uint32_t>>& new_neigbors) {
        all_calculate_ = 0;
        valid_calculate_ = 0;

        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < new_neigbors[i].size(); ++j) {
                for (int k = j + 1; k < new_neigbors[i].size(); ++k) {
                    if (new_neigbors[i][j] == new_neigbors[i][k]) {
                        continue;
                    }
                    float dist = get_distance(new_neigbors[i][j], new_neigbors[i][k]);
                    all_calculate_ += 1.0;
                    float update = false;
                    if (dist < graph[new_neigbors[i][j]].greast_neighbor_distance) {
                        new_candidates_[new_neigbors[i][j]].emplace_back(new_neigbors[i][k], dist);
                        update = true;
                    }
                    if (dist < graph[new_neigbors[i][k]].greast_neighbor_distance) {
                        new_candidates_[new_neigbors[i][k]].emplace_back(new_neigbors[i][j], dist);
                        update = true;
                    }
                    if (update) {
                        valid_calculate_ += 1.0;
                    }
                }

                for (int k = 0; k < old_neigbors[i].size(); ++k) {
                    if (new_neigbors[i][j] == old_neigbors[i][k]) {
                        continue;
                    }
                    float dist = get_distance(new_neigbors[i][j], old_neigbors[i][k]);
                    all_calculate_ += 1.0;
                    float update = false;
                    if (dist < graph[new_neigbors[i][j]].greast_neighbor_distance) {
                        new_candidates_[new_neigbors[i][j]].emplace_back(old_neigbors[i][k], dist);
                        update = true;
                    }
                    if (dist < graph[old_neigbors[i][k]].greast_neighbor_distance) {
                        new_candidates_[old_neigbors[i][k]].emplace_back(new_neigbors[i][j], dist);
                        update = true;
                    }
                    if (update) {
                        valid_calculate_ += 1.0;
                    }
                }
            }
            old_neigbors[i].clear();
            new_neigbors[i].clear();
        }
        float unique_count = 0;
        float all_count = 0;

        for (uint32_t i = 0; i < data_num_; ++i) {
            graph[i].neigbors.insert(
                graph[i].neigbors.end(), new_candidates_[i].begin(), new_candidates_[i].end());
            all_count += graph[i].neigbors.size();
            std::sort(graph[i].neigbors.begin(), graph[i].neigbors.end());
            graph[i].neigbors.erase(std::unique(graph[i].neigbors.begin(), graph[i].neigbors.end()),
                                    graph[i].neigbors.end());
            unique_count += graph[i].neigbors.size();
            if (graph[i].neigbors.size() > max_degree_) {
                graph[i].neigbors.resize(max_degree_);
            }
            graph[i].greast_neighbor_distance = graph[i].neigbors.back().distance;
            new_candidates_[i].clear();
        }
        duplicate_rate = (all_count - unique_count) / all_count;
    }

    void
    add_reverse_edges() {
        std::vector<Linklist> reverse_graph;
        reverse_graph.resize(data_num_);
        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                auto& node = graph[i].neigbors[j];
                reverse_graph[node.id].neigbors.emplace_back(i, node.distance);
            }
        }

        for (int i = 0; i < data_num_; ++i) {
            graph[i].neigbors.insert(graph[i].neigbors.end(),
                                     reverse_graph[i].neigbors.begin(),
                                     reverse_graph[i].neigbors.end());
            std::sort(graph[i].neigbors.begin(), graph[i].neigbors.end());
            graph[i].neigbors.erase(std::unique(graph[i].neigbors.begin(), graph[i].neigbors.end()),
                                    graph[i].neigbors.end());
            if (graph[i].neigbors.size() > max_degree_) {
                graph[i].neigbors.resize(max_degree_);
            }
        }
    }

    void
    search_and_connect() {
        std::vector<bool> visits(data_num_, false);
        std::vector<uint32_t> entry_point;
        for (int i = 0; i < data_num_; ++i) {
            if (visits[i]) {
                continue;
            }
            std::vector<uint32_t> candidates;
            candidates.push_back(i);
            while (not candidates.empty()) {
                std::vector<uint32_t> new_candidates;
                for (unsigned int cur_node : candidates) {
                    for (auto& neigbor : graph[cur_node].neigbors) {
                        if (not visits[neigbor.id]) {
                            new_candidates.push_back(neigbor.id);
                            visits[neigbor.id] = true;
                        }
                    }
                }
                candidates.swap(new_candidates);
            }
            entry_point.push_back(i);
        }
        std::vector<std::vector<Node>> new_candidates;
        new_candidates.resize(data_num_);
        for (int i = 1; i < data_num_; ++i) {
            auto nearest_nerigbors = find_close_neighbor(i, entry_point, max_degree_ * 2);
            for (int j = 0; j < max_degree_; ++j) {
                auto& node = nearest_nerigbors.top();
                if (node.id != i) {
                    new_candidates[node.id].emplace_back(i, node.distance);
                }
                nearest_nerigbors.pop();
            }
        }
        for (int i = 1; i < data_num_; ++i) {
            graph[i].neigbors.insert(
                graph[i].neigbors.end(), new_candidates[i].begin(), new_candidates[i].end());
            std::sort(graph[i].neigbors.begin(), graph[i].neigbors.end());
            graph[i].neigbors.erase(std::unique(graph[i].neigbors.begin(), graph[i].neigbors.end()),
                                    graph[i].neigbors.end());
            graph[i].greast_neighbor_distance = graph[i].neigbors.back().distance;
        }
        for (int i = 1; i < entry_point.size(); ++i) {
            std::cout << entry_point[i] << " ";
            graph[0].neigbors.emplace_back(entry_point[i], 0.0);
        }
        std::cout << std::endl;
    }

    void
    sample_candidates(std::vector<std::vector<uint32_t>>& old_neigbors,
                      std::vector<std::vector<uint32_t>>& new_neigbors,
                      float sample_rate) {
        std::random_device rd;
        std::uniform_real_distribution<float> p;
        std::mt19937 rng(rd());
        old_neigbors.resize(data_num_);
        new_neigbors.resize(data_num_);

        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                float current_state = p(rng);
                if (current_state < sample_rate) {
                    if (graph[i].neigbors[j].old) {
                        old_neigbors[i].push_back(graph[i].neigbors[j].id);
                        old_neigbors[graph[i].neigbors[j].id].push_back(i);
                    } else {
                        new_neigbors[i].push_back(graph[i].neigbors[j].id);
                        new_neigbors[graph[i].neigbors[j].id].push_back(i);
                        graph[i].neigbors[j].old = true;
                    }
                }
            }
        }
    }

    std::priority_queue<Node>
    find_close_neighbor(uint32_t id, const std::vector<uint32_t>& start_points, int64_t topk) {
        std::priority_queue<Node> candidates;
        std::priority_queue<Node> nearest_nerigbors;
        std::unordered_set<uint32_t> visited_set;
        for (int j = 0; j < start_points.size(); ++j) {
            candidates.emplace(start_points[j], -get_distance(id, start_points[j]));
            visited_set.insert(start_points[j]);
        }
        for (int j = 0; j < topk * 2; ++j) {
            auto cur_node = candidates.top();
            candidates.pop();
            for (int k = 0; k < graph[cur_node.id].neigbors.size(); ++k) {
                if (visited_set.find(graph[cur_node.id].neigbors[k].id) != visited_set.end() ||
                    graph[cur_node.id].neigbors[k].id == id) {
                    continue;
                }
                visited_set.insert(graph[cur_node.id].neigbors[k].id);
                auto dist = get_distance(id, graph[cur_node.id].neigbors[k].id);
                candidates.emplace(graph[cur_node.id].neigbors[k].id, -dist);
            }
            nearest_nerigbors.emplace(cur_node.id, cur_node.distance);
        }
        return nearest_nerigbors;
    }

    void
    search_neigbors() {
        std::vector<std::vector<Node>> new_candidates;
        new_candidates.resize(data_num_);
        std::random_device rd;
        std::uniform_int_distribution<int> k_generate(0, data_num_ - 1);
        std::mt19937 rng(rd());
        for (int i = 0; i < data_num_ / 100; ++i) {
            int id = k_generate(rng);
            while (visited_[id]) {
                id = k_generate(rng);
            }
            float max_distance = graph[id].neigbors.back().distance;
            std::vector<uint32_t> start_points;
            for (int j = 0; j < graph[id].neigbors.size(); ++j) {
                start_points.push_back(graph[id].neigbors[j].id);
            }
            auto nearest_nerigbors = find_close_neighbor(id, start_points, max_degree_);
            for (int j = 0; j < graph[id].neigbors.size(); ++j) {
                graph[id].neigbors[j] = nearest_nerigbors.top();
                graph[id].neigbors[j].distance = -graph[id].neigbors[j].distance;
                nearest_nerigbors.pop();
            }
            if (max_distance == graph[id].neigbors.back().distance) {
                visited_[id] = true;
            }
        }
    }

    void
    repair_no_in_edge() {
        std::vector<int> in_edges_count(data_num_, 0);
        for (int i = 0; i < data_num_; ++i) {
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                in_edges_count[graph[i].neigbors[j].id]++;
            }
        }

        std::vector<int> replace_pos(data_num_, std::min(data_num_ - 1, max_degree_) - 1);
        for (int i = 0; i < data_num_; ++i) {
            auto& link = graph[i].neigbors;
            int need_replace_loc = 0;
            while (in_edges_count[i] < min_in_degree_ && need_replace_loc < max_degree_) {
                uint32_t need_replace_id = link[need_replace_loc].id;
                bool has_connect = false;
                for (int j = 0; j < graph[need_replace_id].neigbors.size(); ++j) {
                    if (graph[need_replace_id].neigbors[j].id == i) {
                        has_connect = true;
                        break;
                    }
                }
                if (replace_pos[need_replace_id] > 0 && not has_connect) {
                    auto& replace_node =
                        graph[need_replace_id].neigbors[replace_pos[need_replace_id]];
                    auto replace_id = replace_node.id;
                    if (in_edges_count[replace_id] > min_in_degree_) {
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
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                in_edges_count[graph[i].neigbors[j].id]++;
            }
        }

        for (int loc = 1; loc < data_num_; ++loc) {
            std::sort(graph[loc].neigbors.begin(), graph[loc].neigbors.end());
            graph[loc].neigbors.erase(
                std::unique(graph[loc].neigbors.begin(), graph[loc].neigbors.end()),
                graph[loc].neigbors.end());
            std::vector<Node> candidates;
            candidates.reserve(max_degree_);
            for (int i = 0; i < graph[loc].neigbors.size(); ++i) {
                bool flag = true;
                if (in_edges_count[graph[loc].neigbors[i].id] > min_in_degree_) {
                    for (int j = 0; j < candidates.size(); ++j) {
                        if (get_distance(graph[loc].neigbors[i].id, candidates[j].id) <
                            graph[loc].neigbors[i].distance) {
                            flag = false;
                            in_edges_count[graph[loc].neigbors[i].id]--;
                            break;
                        }
                    }
                }
                if (flag) {
                    candidates.push_back(graph[loc].neigbors[i]);
                }
            }
            graph[loc].neigbors.swap(candidates);
            if (graph[loc].neigbors.size() > max_degree_) {
                graph[loc].neigbors.resize(max_degree_);
            }
        }
    }

    void
    check_turn() {
        int edge_count = 0;
        float loss = 0;
        int no_in_edge_count = 0;

        std::vector<int> in_edges_count(data_num_, 0);
        for (int i = 0; i < data_num_; ++i) {
            //            std::cout <<"check: ";
            for (int j = 0; j < graph[i].neigbors.size(); ++j) {
                loss += graph[i].neigbors[j].distance;
                in_edges_count[graph[i].neigbors[j].id]++;
                //                                std::cout << graph[i].neigbors[j].distance << " ";
            }
            //                        std::cout << std::endl;
            edge_count += graph[i].neigbors.size();
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
                    for (auto& neigbor : graph[cur_node].neigbors) {
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
        auto rate = (all_calculate_ == 0 ? 0 : valid_calculate_ / all_calculate_);
        loss /= edge_count;
        logger::info(
            fmt::format("loss:{} edge_count:{} no_in_edge_count:{}  connections:{} "
                        "valid_cal_rate:{} duplicate_rate:{}",
                        loss,
                        edge_count,
                        no_in_edge_count,
                        connect_count,
                        rate,
                        duplicate_rate));
    }

private:
    size_t dim_;
    int64_t data_num_;
    int64_t is_build_ = false;
    const float* data_;

    int64_t max_degree_;
    int64_t turn_;
    std::vector<Linklist> graph;
    std::vector<bool> visited_;
    int64_t min_in_degree_ = 5;

    float all_calculate_ = 0;
    float valid_calculate_ = 0;
    float duplicate_rate = 0;

    std::vector<std::vector<Node>> new_candidates_;
    std::vector<std::mutex> mutexs_;

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