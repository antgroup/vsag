
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

#include "odescent_graph_builder.h"

#include <chrono>
#include <ios>

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

bool
ODescent::Build(const uint32_t* valid_ids, int64_t data_num) {
    if (is_build_) {
        return false;
    }
    is_build_ = true;
    valid_ids_ = valid_ids;
    if (valid_ids_ != nullptr) {
        data_num_ = data_num;
    } else {
        data_num_ = flatten_interface_->TotalCount();
    }
    if (data_num_ <= 1) {
        throw std::runtime_error("ODescent cannot build a graph with data_num less than 1");
    }
    min_in_degree_ = std::min(min_in_degree_, data_num_ - 1);
    Vector<std::mutex>(data_num_, allocator_).swap(points_lock_);
    Vector<UnorderedSet<uint32_t>> old_neighbors(allocator_);
    Vector<UnorderedSet<uint32_t>> new_neighbors(allocator_);
    old_neighbors.resize(data_num_, UnorderedSet<uint32_t>(allocator_));
    new_neighbors.resize(data_num_, UnorderedSet<uint32_t>(allocator_));
    for (int i = 0; i < data_num_; ++i) {
        old_neighbors[i].reserve(max_degree_);
        new_neighbors[i].reserve(max_degree_);
    }
    init_graph();
    {
        for (int i = 0; i < turn_; ++i) {
            sample_candidates(old_neighbors, new_neighbors, sample_rate_);
            update_neighbors(old_neighbors, new_neighbors);
            repair_no_in_edge();
        }
        if (pruning_) {
            prune_graph();
            add_reverse_edges();
        }
    }
    return true;
}

void
ODescent::SaveGraph(std::stringstream& out) {
    std::streamoff file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&max_degree, sizeof(uint32_t));
    uint32_t ep_u32 = 0;
    size_t num_frozen = 0;
    out.write((char*)&ep_u32, sizeof(uint32_t));
    out.write((char*)&num_frozen, sizeof(size_t));
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
    // location limit.
    for (uint32_t i = 0; i < static_cast<uint32_t>(data_num_); i++) {
        Vector<uint32_t> edges(allocator_);
        edges.resize(graph[i].neighbors.size());
        for (int j = 0; j < graph[i].neighbors.size(); ++j) {
            edges[j] = graph[i].neighbors[j].id;
        }
        auto gk = (uint32_t)edges.size();
        out.write((char*)&gk, sizeof(uint32_t));
        out.write((char*)edges.data(), static_cast<std::streamsize>(gk * sizeof(uint32_t)));
        max_degree = edges.size() > max_degree ? (uint32_t)edges.size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (gk + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&max_degree, sizeof(uint32_t));
}

void
ODescent::init_graph() {
    graph.resize(data_num_, Linklist(allocator_));

    auto task = [&, this](int64_t start, int64_t end) {
        std::random_device rd;
        std::uniform_int_distribution<int64_t> k_generate(0, data_num_ - 1);
        std::mt19937 rng(rd());
        for (int64_t i = start; i < end; ++i) {
            UnorderedSet<uint32_t> ids_set(allocator_);
            ids_set.insert(i);
            graph[i].neighbors.reserve(max_degree_);
            int64_t max_neighbors = std::min(data_num_ - 1, max_degree_);
            for (int j = 0; j < max_neighbors; ++j) {
                uint32_t id = i;
                if (data_num_ - 1 < max_degree_) {
                    id = (i + j + 1) % data_num_;
                } else {
                    while (ids_set.find(id) != ids_set.end()) {
                        id = k_generate(rng);
                    }
                }
                ids_set.insert(id);
                auto dist = get_distance(i, id);
                graph[i].neighbors.emplace_back(id, dist);
                graph[i].greast_neighbor_distance =
                    std::max(graph[i].greast_neighbor_distance, dist);
            }
        }
    };
    parallelize_task(task);
}

void
ODescent::update_neighbors(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                           Vector<UnorderedSet<uint32_t>>& new_neighbors) {
    Vector<std::future<void>> futures(allocator_);
    auto task = [&, this](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            Vector<uint32_t> new_neighbors_candidates(allocator_);
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
ODescent::add_reverse_edges() {
    Vector<Linklist> reverse_graph(allocator_);
    reverse_graph.resize(data_num_, Linklist(allocator_));
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
ODescent::sample_candidates(Vector<UnorderedSet<uint32_t>>& old_neighbors,
                            Vector<UnorderedSet<uint32_t>>& new_neighbors,
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
ODescent::repair_no_in_edge() {
    Vector<int> in_edges_count(data_num_, 0, allocator_);
    for (int i = 0; i < data_num_; ++i) {
        for (auto& neigbor : graph[i].neighbors) {
            in_edges_count[neigbor.id]++;
        }
    }

    Vector<int> replace_pos(
        data_num_, static_cast<int32_t>(std::min(data_num_ - 1, max_degree_) - 1), allocator_);
    for (int i = 0; i < data_num_; ++i) {
        auto& link = graph[i].neighbors;
        int need_replace_loc = 0;
        while (in_edges_count[i] < min_in_degree_ && need_replace_loc < max_degree_) {
            uint32_t need_replace_id = link[need_replace_loc].id;
            bool has_connect = false;
            for (auto& neigbor : graph[need_replace_id].neighbors) {
                if (neigbor.id == i) {
                    has_connect = true;
                    break;
                }
            }
            if (replace_pos[need_replace_id] > 0 && not has_connect) {
                auto& replace_node = graph[need_replace_id].neighbors[replace_pos[need_replace_id]];
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
ODescent::prune_graph() {
    Vector<int> in_edges_count(data_num_, 0, allocator_);
    for (int i = 0; i < data_num_; ++i) {
        for (auto& neighbor : graph[i].neighbors) {
            in_edges_count[neighbor.id]++;
        }
    }

    auto task = [&, this](int64_t start, int64_t end) {
        for (int64_t loc = start; loc < end; ++loc) {
            auto& neighbors = graph[loc].neighbors;
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
            Vector<Node> candidates(allocator_);
            candidates.reserve(max_degree_);
            for (auto& neighbor : neighbors) {
                bool flag = true;
                int cur_in_edge = 0;
                {
                    std::lock_guard<std::mutex> lock(points_lock_[neighbor.id]);
                    cur_in_edge = in_edges_count[neighbor.id];
                }
                if (cur_in_edge > min_in_degree_) {
                    for (auto& candidate : candidates) {
                        if (get_distance(neighbor.id, candidate.id) * alpha_ < neighbor.distance) {
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
ODescent::parallelize_task(std::function<void(int64_t, int64_t)> task) {
    Vector<std::future<void>> futures(allocator_);
    for (int64_t i = 0; i < data_num_; i += block_size_) {
        int64_t end = std::min(i + block_size_, data_num_);
        futures.push_back(thread_pool_->GeneralEnqueue(task, i, end));
    }
    for (auto& future : futures) {
        future.get();
    }
}

void
ODescent::SaveGraph(GraphInterfacePtr& graph_storage) {
    for (int i = 0; i < data_num_; ++i) {
        uint32_t id = i;
        if (valid_ids_) {
            id = valid_ids_[i];
        }
        Vector<uint32_t> edges(allocator_);
        edges.resize(graph[i].neighbors.size());
        for (int j = 0; j < graph[i].neighbors.size(); ++j) {
            edges[j] = graph[i].neighbors[j].id;
            if (valid_ids_) {
                edges[j] = valid_ids_[graph[i].neighbors[j].id];
            }
        }
        graph_storage->InsertNeighborsById(id, edges);
    }
}

}  // namespace vsag
