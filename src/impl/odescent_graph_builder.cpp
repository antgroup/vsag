
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

namespace vsag {

class LCG {
public:
    LCG() {
        auto now = std::chrono::steady_clock::now();
        auto timestamp =
            std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        current = static_cast<unsigned int>(timestamp);
    }

    float
    nextFloat() {
        current = (a * current + c) % m;
        return static_cast<float>(current) / static_cast<float>(m);
    }

private:
    unsigned int current;
    static const uint32_t a = 1664525;
    static const uint32_t c = 1013904223;
    static const uint32_t m = 4294967295;  // 2^32 - 1
};

bool
Odescent::Build(const DatasetPtr& dataset) {
    if (is_build_) {
        return false;
    }
    is_build_ = true;
    dim_ = dataset->GetDim();
    data_num_ = dataset->GetNumElements();
    data_ = dataset->GetFloat32Vectors();
    min_in_degree_ = std::min(min_in_degree_, data_num_ - 1);
    Vector<Vector<uint32_t>> old_neigbors(allocator_);
    Vector<Vector<uint32_t>> new_neigbors(allocator_);
    new_candidates_.resize(data_num_, Vector<Node>(allocator_));
    old_neigbors.resize(data_num_, Vector<uint32_t>(allocator_));
    new_neigbors.resize(data_num_, Vector<uint32_t>(allocator_));
    for (int i = 0; i < data_num_; ++i) {
        new_candidates_[i].reserve(max_degree_);
        old_neigbors[i].reserve(max_degree_);
        new_neigbors[i].reserve(max_degree_);
    }
    init_graph();
    {
        for (int i = 0; i < turn_; ++i) {
            sample_candidates(old_neigbors, new_neigbors, sample_rate_);
            update_neighbors(old_neigbors, new_neigbors);
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
Odescent::SaveGraph(std::stringstream& out) {
    size_t file_offset = 0;  // we will use this if we want
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&max_degree, sizeof(uint32_t));
    uint32_t ep_u32 = 0;
    out.write((char*)&ep_u32, sizeof(uint32_t));
    out.write((char*)&ep_u32, sizeof(size_t));
    // Note: at this point, either _nd == _max_points or any frozen points have
    // been temporarily moved to _nd, so _nd + _num_frozen_points is the valid
    // location limit.
    auto _final_graph = GetGraph();
    for (uint32_t i = 0; i < data_num_; i++) {
        uint32_t GK = (uint32_t)_final_graph[i].size();
        out.write((char*)&GK, sizeof(uint32_t));
        out.write((char*)_final_graph[i].data(), GK * sizeof(uint32_t));
        max_degree =
            _final_graph[i].size() > max_degree ? (uint32_t)_final_graph[i].size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char*)&index_size, sizeof(uint64_t));
    out.write((char*)&max_degree, sizeof(uint32_t));
}

Vector<Vector<uint32_t>>
Odescent::GetGraph() {
    Vector<Vector<uint32_t>> extract_graph(allocator_);
    extract_graph.resize(data_num_, Vector<uint32_t>(allocator_));
    for (int i = 0; i < data_num_; ++i) {
        extract_graph[i].resize(graph[i].neigbors.size());
        for (int j = 0; j < graph[i].neigbors.size(); ++j) {
            extract_graph[i][j] = graph[i].neigbors[j].id;
        }
    }

    return extract_graph;
}

void
Odescent::init_graph() {
    graph.resize(data_num_, Linklist(allocator_));
    std::random_device rd;
    std::uniform_int_distribution<int> k_generate(0, data_num_ - 1);
    std::mt19937 rng(rd());

    for (int i = 0; i < data_num_; ++i) {
        UnorderedSet<uint32_t> ids_set(allocator_);
        graph[i].neigbors.reserve(max_degree_);
        int max_neighbors = std::min(data_num_ - 1, max_degree_);

        for (int j = 0; j < max_neighbors; ++j) {
            uint32_t id = i;
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
            graph[i].greast_neighbor_distance = std::max(graph[i].greast_neighbor_distance, dist);
        }
    }
}

void
Odescent::update_neighbors(Vector<Vector<uint32_t>>& old_neigbors,
                           Vector<Vector<uint32_t>>& new_neigbors) {
    for (int i = 0; i < data_num_; ++i) {
        for (int j = 0; j < new_neigbors[i].size(); ++j) {
            for (int k = j + 1; k < new_neigbors[i].size(); ++k) {
                if (new_neigbors[i][j] == new_neigbors[i][k]) {
                    continue;
                }
                float dist = get_distance(new_neigbors[i][j], new_neigbors[i][k]);
                if (dist < graph[new_neigbors[i][j]].greast_neighbor_distance) {
                    new_candidates_[new_neigbors[i][j]].emplace_back(new_neigbors[i][k], dist);
                }
                if (dist < graph[new_neigbors[i][k]].greast_neighbor_distance) {
                    new_candidates_[new_neigbors[i][k]].emplace_back(new_neigbors[i][j], dist);
                }
            }

            for (int k = 0; k < old_neigbors[i].size(); ++k) {
                if (new_neigbors[i][j] == old_neigbors[i][k]) {
                    continue;
                }
                float dist = get_distance(new_neigbors[i][j], old_neigbors[i][k]);
                if (dist < graph[new_neigbors[i][j]].greast_neighbor_distance) {
                    new_candidates_[new_neigbors[i][j]].emplace_back(old_neigbors[i][k], dist);
                }
                if (dist < graph[old_neigbors[i][k]].greast_neighbor_distance) {
                    new_candidates_[old_neigbors[i][k]].emplace_back(new_neigbors[i][j], dist);
                }
            }
        }
        old_neigbors[i].clear();
        new_neigbors[i].clear();
    }

    for (uint32_t i = 0; i < data_num_; ++i) {
        graph[i].neigbors.insert(
            graph[i].neigbors.end(), new_candidates_[i].begin(), new_candidates_[i].end());
        std::sort(graph[i].neigbors.begin(), graph[i].neigbors.end());
        graph[i].neigbors.erase(std::unique(graph[i].neigbors.begin(), graph[i].neigbors.end()),
                                graph[i].neigbors.end());
        if (graph[i].neigbors.size() > max_degree_) {
            graph[i].neigbors.resize(max_degree_);
        }
        graph[i].greast_neighbor_distance = graph[i].neigbors.back().distance;
        new_candidates_[i].clear();
    }
}

void
Odescent::add_reverse_edges() {
    Vector<Linklist> reverse_graph(allocator_);
    reverse_graph.resize(data_num_, Linklist(allocator_));
    for (int i = 0; i < data_num_; ++i) {
        reverse_graph[i].neigbors.reserve(max_degree_);
    }
    for (int i = 0; i < data_num_; ++i) {
        for (const auto& node : graph[i].neigbors) {
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
Odescent::sample_candidates(Vector<Vector<uint32_t>>& old_neigbors,
                            Vector<Vector<uint32_t>>& new_neigbors,
                            float sample_rate) {
    LCG r;
    for (int i = 0; i < data_num_; ++i) {
        for (int j = 0; j < graph[i].neigbors.size(); ++j) {
            float current_state = r.nextFloat();
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

void
Odescent::repair_no_in_edge() {
    std::vector<int> in_edges_count(data_num_, 0);
    for (int i = 0; i < data_num_; ++i) {
        for (auto& neigbor : graph[i].neigbors) {
            in_edges_count[neigbor.id]++;
        }
    }

    std::vector<int> replace_pos(data_num_, std::min(data_num_ - 1, max_degree_) - 1);
    for (int i = 0; i < data_num_; ++i) {
        auto& link = graph[i].neigbors;
        int need_replace_loc = 0;
        while (in_edges_count[i] < min_in_degree_ && need_replace_loc < max_degree_) {
            uint32_t need_replace_id = link[need_replace_loc].id;
            bool has_connect = false;
            for (auto& neigbor : graph[need_replace_id].neigbors) {
                if (neigbor.id == i) {
                    has_connect = true;
                    break;
                }
            }
            if (replace_pos[need_replace_id] > 0 && not has_connect) {
                auto& replace_node = graph[need_replace_id].neigbors[replace_pos[need_replace_id]];
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
Odescent::prune_graph() {
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
        Vector<Node> candidates(allocator_);
        candidates.reserve(max_degree_);
        for (int i = 0; i < graph[loc].neigbors.size(); ++i) {
            bool flag = true;
            if (in_edges_count[graph[loc].neigbors[i].id] > min_in_degree_) {
                for (int j = 0; j < candidates.size(); ++j) {
                    if (get_distance(graph[loc].neigbors[i].id, candidates[j].id) * alpha_ <
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

}  // namespace vsag