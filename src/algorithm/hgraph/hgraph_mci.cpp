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

#include <algorithm>
#include <atomic>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <vector>

#include "../mci/mci_v3_builder.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "hgraph.h"
#include "impl/logger/logger.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "index_common_param.h"

namespace vsag {
namespace {

struct MCIKNNGraph {
    explicit MCIKNNGraph(Allocator* allocator)
        : neighbors(allocator), counts(allocator), allocator(allocator) {
    }

    MCIKNNGraph(const MCIKNNGraph&) = delete;
    MCIKNNGraph&
    operator=(const MCIKNNGraph&) = delete;

    MCIKNNGraph(MCIKNNGraph&& other) noexcept
        : neighbors(std::move(other.neighbors)),
          counts(std::move(other.counts)),
          raw_neighbors(other.raw_neighbors),
          raw_neighbor_count(other.raw_neighbor_count),
          row_stride(other.row_stride),
          uniform_count(other.uniform_count),
          total(other.total),
          allocator(other.allocator) {
        other.raw_neighbors = nullptr;
        other.raw_neighbor_count = 0;
    }

    MCIKNNGraph&
    operator=(MCIKNNGraph&& other) noexcept {
        if (this != &other) {
            this->ReleaseRawNeighbors();
            neighbors = std::move(other.neighbors);
            counts = std::move(other.counts);
            raw_neighbors = other.raw_neighbors;
            raw_neighbor_count = other.raw_neighbor_count;
            row_stride = other.row_stride;
            uniform_count = other.uniform_count;
            total = other.total;
            allocator = other.allocator;
            other.raw_neighbors = nullptr;
            other.raw_neighbor_count = 0;
        }
        return *this;
    }

    ~MCIKNNGraph() {
        this->ReleaseRawNeighbors();
    }

    Vector<InnerIdType> neighbors;
    Vector<uint32_t> counts;
    InnerIdType* raw_neighbors{nullptr};
    uint64_t raw_neighbor_count{0};
    uint64_t row_stride{0};
    uint64_t uniform_count{0};
    uint64_t total{0};
    Allocator* allocator{nullptr};

    template <typename Fn>
    void
    ForEachNeighbor(InnerIdType inner_id, Fn&& fn) const {
        if (inner_id >= total or row_stride == 0) {
            return;
        }
        const auto count = counts.empty() ? uniform_count : counts[inner_id];
        const auto* base = raw_neighbors != nullptr ? raw_neighbors : neighbors.data();
        const auto* row = base + static_cast<uint64_t>(inner_id) * row_stride;
        for (uint64_t rank = 0; rank < count; ++rank) {
            const auto neighbor = row[rank];
            if (neighbor >= total or neighbor == inner_id) {
                continue;
            }
            if (not fn(neighbor)) {
                break;
            }
        }
    }

    void
    AllocateRawNeighbors(uint64_t count) {
        this->ReleaseRawNeighbors();
        raw_neighbors = static_cast<InnerIdType*>(allocator->Allocate(count * sizeof(InnerIdType)));
        raw_neighbor_count = count;
    }

    void
    ReleaseRawNeighbors() {
        if (raw_neighbors != nullptr) {
            allocator->Deallocate(raw_neighbors);
            raw_neighbors = nullptr;
            raw_neighbor_count = 0;
        }
    }
};

Vector<Vector<uint64_t>>
run_local_ccr_mce(const Vector<InnerIdType>& local_nodes,
                  const Vector<uint8_t>& local_edges,
                  const std::vector<std::atomic<uint32_t>>& num_cliques_per_node,
                  uint64_t threshold,
                  uint64_t max_saved_cliques,
                  Allocator* allocator) {
    Vector<Vector<uint64_t>> saved_cliques(allocator);
    const auto local_count = local_nodes.size();
    if (local_count < threshold or max_saved_cliques == 0) {
        return saved_cliques;
    }
    auto has_local_edge = [&](uint64_t lhs, uint64_t rhs) {
        return local_edges[lhs * local_count + rhs] != 0;
    };

    Vector<uint64_t> degree(local_count, 0, allocator);
    for (uint64_t i = 0; i < local_count; ++i) {
        for (uint64_t j = 0; j < local_count; ++j) {
            if (has_local_edge(i, j)) {
                ++degree[i];
            }
        }
    }

    Vector<uint64_t> active_to_local(allocator);
    Vector<uint64_t> local_to_active(local_count, local_count, allocator);
    active_to_local.reserve(local_count);
    for (uint64_t local_id = 0; local_id < local_count; ++local_id) {
        if (degree[local_id] == 0) {
            continue;
        }
        local_to_active[local_id] = active_to_local.size();
        active_to_local.push_back(local_id);
    }
    if (active_to_local.size() < threshold) {
        return saved_cliques;
    }

    Vector<uint64_t> active_degree(allocator);
    active_degree.reserve(active_to_local.size());
    for (auto local_id : active_to_local) {
        active_degree.push_back(degree[local_id]);
    }

    Vector<uint64_t> order_to_local(allocator);
    Vector<uint64_t> local_to_order(local_count, local_count, allocator);
    Vector<uint8_t> removed(active_to_local.size(), 0, allocator);
    order_to_local.reserve(active_to_local.size());
    for (uint64_t order = 0; order < active_to_local.size(); ++order) {
        uint64_t best = active_to_local.size();
        uint64_t best_degree = std::numeric_limits<uint64_t>::max();
        for (uint64_t active_id = 0; active_id < active_to_local.size(); ++active_id) {
            const auto local_id = active_to_local[active_id];
            if (removed[active_id] == 0 and
                (active_degree[active_id] < best_degree or
                 (active_degree[active_id] == best_degree and
                  (best == active_to_local.size() or local_id < active_to_local[best])))) {
                best = active_id;
                best_degree = active_degree[active_id];
            }
        }
        if (best == active_to_local.size()) {
            break;
        }
        const auto best_local = active_to_local[best];
        removed[best] = 1;
        order_to_local.push_back(best_local);
        local_to_order[best_local] = order;
        for (uint64_t active_id = 0; active_id < active_to_local.size(); ++active_id) {
            const auto neighbor = active_to_local[active_id];
            if (removed[active_id] == 0 and has_local_edge(best_local, neighbor) and
                active_degree[active_id] > 0) {
                --active_degree[active_id];
            }
        }
    }
    if (order_to_local.size() < threshold) {
        return saved_cliques;
    }

    const auto core_count = order_to_local.size();
    Vector<uint8_t> core_edges(core_count * core_count, 0, allocator);
    auto has_core_edge = [&](uint64_t lhs, uint64_t rhs) {
        return core_edges[lhs * core_count + rhs] != 0;
    };
    auto set_core_edge = [&](uint64_t lhs, uint64_t rhs) {
        core_edges[lhs * core_count + rhs] = 1;
    };
    Vector<uint8_t> must_contain(core_count, 0, allocator);
    uint64_t remaining_must = 0;
    for (uint64_t lhs = 0; lhs < local_count; ++lhs) {
        const auto core_lhs = local_to_order[lhs];
        if (core_lhs >= core_count) {
            continue;
        }
        if (num_cliques_per_node[local_nodes[lhs]].load(std::memory_order_relaxed) == 0) {
            must_contain[core_lhs] = 1;
            ++remaining_must;
        }
        for (uint64_t rhs = lhs + 1; rhs < local_count; ++rhs) {
            if (not has_local_edge(lhs, rhs)) {
                continue;
            }
            const auto core_rhs = local_to_order[rhs];
            if (core_rhs < core_count) {
                set_core_edge(core_lhs, core_rhs);
                set_core_edge(core_rhs, core_lhs);  // NOLINT(readability-suspicious-call-argument)
            }
        }
    }
    if (remaining_must == 0) {
        return saved_cliques;
    }

    auto state_has_must = [&](const Vector<uint64_t>& current, const Vector<uint64_t>& candidates) {
        for (auto node : current) {
            if (must_contain[node] != 0) {
                return true;
            }
        }
        for (auto node : candidates) {
            if (must_contain[node] != 0) {
                return true;
            }
        }
        return false;
    };

    auto save_clique = [&](const Vector<uint64_t>& current) {
        if (current.size() < threshold or saved_cliques.size() >= max_saved_cliques) {
            return false;
        }
        bool has_must = false;
        for (auto node : current) {
            if (must_contain[node] != 0) {
                has_must = true;
                break;
            }
        }
        if (not has_must) {
            return false;
        }

        saved_cliques.push_back(Vector<uint64_t>(allocator));
        auto& saved = saved_cliques.back();
        saved.reserve(current.size());
        for (auto node : current) {
            saved.push_back(order_to_local[node]);
            if (must_contain[node] != 0) {
                must_contain[node] = 0;
                --remaining_must;
            }
        }
        return remaining_must == 0 or saved_cliques.size() >= max_saved_cliques;
    };

    std::function<void(Vector<uint64_t>&, Vector<uint64_t>&, Vector<uint64_t>&)> expand =
        [&](Vector<uint64_t>& current, Vector<uint64_t>& candidates, Vector<uint64_t>& excluded) {
            if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
                return;
            }
            if (current.size() + candidates.size() < threshold) {
                return;
            }
            if (not state_has_must(current, candidates)) {
                return;
            }
            if (candidates.empty() and excluded.empty()) {
                save_clique(current);
                return;
            }

            uint64_t pivot = core_count;
            uint64_t pivot_degree = 0;
            auto update_pivot = [&](uint64_t node) {
                uint64_t node_degree = 0;
                for (auto candidate : candidates) {
                    if (has_core_edge(node, candidate)) {
                        ++node_degree;
                    }
                }
                if (pivot == core_count or node_degree > pivot_degree) {
                    pivot = node;
                    pivot_degree = node_degree;
                }
            };
            for (auto node : candidates) {
                update_pivot(node);
            }
            for (auto node : excluded) {
                update_pivot(node);
            }

            Vector<uint64_t> branches(allocator);
            branches.reserve(candidates.size());
            for (auto node : candidates) {
                if (pivot == core_count or not has_core_edge(pivot, node)) {
                    branches.push_back(node);
                }
            }

            for (auto node : branches) {
                auto iter = std::find(candidates.begin(), candidates.end(), node);
                if (iter == candidates.end()) {
                    continue;
                }

                current.push_back(node);
                Vector<uint64_t> next_candidates(allocator);
                Vector<uint64_t> next_excluded(allocator);
                next_candidates.reserve(candidates.size());
                next_excluded.reserve(excluded.size());
                for (auto candidate : candidates) {
                    if (has_core_edge(node, candidate)) {
                        next_candidates.push_back(candidate);
                    }
                }
                for (auto excluded_node : excluded) {
                    if (has_core_edge(node, excluded_node)) {
                        next_excluded.push_back(excluded_node);
                    }
                }
                expand(current, next_candidates, next_excluded);
                current.pop_back();

                iter = std::find(candidates.begin(), candidates.end(), node);
                if (iter != candidates.end()) {
                    candidates.erase(iter);
                    excluded.push_back(node);
                }
                if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
                    return;
                }
            }
        };

    for (uint64_t root = 0; root < core_count; ++root) {
        if (remaining_must == 0 or saved_cliques.size() >= max_saved_cliques) {
            break;
        }
        Vector<uint64_t> current(allocator);
        Vector<uint64_t> candidates(allocator);
        Vector<uint64_t> excluded(allocator);
        current.push_back(root);
        for (uint64_t node = 0; node < core_count; ++node) {
            if (not has_core_edge(root, node)) {
                continue;
            }
            if (node > root) {
                candidates.push_back(node);
            } else {
                excluded.push_back(node);
            }
        }
        if (current.size() + candidates.size() < threshold) {
            continue;
        }
        if (not state_has_must(current, candidates)) {
            continue;
        }
        expand(current, candidates, excluded);
    }
    return saved_cliques;
}

}  // namespace

MCIKNNGraph
build_mci_knn_graph(const FlattenInterfacePtr& build_codes,
                    uint64_t total,
                    uint64_t mcs,
                    const std::string& knng_path,
                    float alpha,
                    MetricType metric,
                    DataTypes data_type,
                    int64_t dim,
                    int64_t extra_info_size,
                    const std::shared_ptr<SafeThreadPool>& thread_pool,
                    Allocator* allocator) {
    MCIKNNGraph graph(allocator);
    graph.total = total;
    if (total <= 1) {
        return graph;
    }

    const auto candidate_limit = std::min<uint64_t>(mcs, total - 1);
    if (not knng_path.empty()) {
        std::ifstream input(knng_path, std::ios::binary | std::ios::ate);
        CHECK_ARGUMENT(input.good(), fmt::format("failed to open knng file: {}", knng_path));
        const auto file_size = static_cast<uint64_t>(input.tellg());
        CHECK_ARGUMENT(file_size > 0, fmt::format("knng file is empty: {}", knng_path));
        CHECK_ARGUMENT(file_size % sizeof(InnerIdType) == 0,
                       fmt::format("invalid knng file size: {}", knng_path));
        const auto entry_count = file_size / sizeof(InnerIdType);
        CHECK_ARGUMENT(entry_count % total == 0,
                       fmt::format("knng entries are not divisible by total count: {}", knng_path));
        const auto file_degree = entry_count / total;
        CHECK_ARGUMENT(file_degree > 0, fmt::format("knng degree is zero: {}", knng_path));
        logger::info(
            "hgraph mci external knng load started, total={}, file_degree={}, "
            "candidate_limit={}, path={}",
            total,
            file_degree,
            candidate_limit,
            knng_path);

        graph.row_stride = file_degree;
        graph.uniform_count = std::min<uint64_t>(file_degree, candidate_limit);
        graph.AllocateRawNeighbors(entry_count);
        input.seekg(0, std::ios::beg);
        input.read(reinterpret_cast<char*>(graph.raw_neighbors),
                   static_cast<std::streamsize>(file_size));
        CHECK_ARGUMENT(input.good(), fmt::format("failed to read knng file: {}", knng_path));
        logger::info("hgraph mci external knng load finished, total={}, path={}", total, knng_path);
        return graph;
    }

    auto graph_param = std::make_shared<SparseGraphDatacellParameter>();
    graph_param->max_degree_ = candidate_limit;
    graph_param->support_delete_ = false;
    graph.row_stride = candidate_limit;
    graph.neighbors.assign(total * candidate_limit, total);
    graph.counts.assign(total, 0);

    IndexCommonParam graph_common_param;
    graph_common_param.metric_ = metric;
    graph_common_param.data_type_ = data_type;
    graph_common_param.dim_ = dim;
    graph_common_param.extra_info_size_ = extra_info_size;
    graph_common_param.thread_pool_ = thread_pool;
    graph_common_param.allocator_ = std::shared_ptr<Allocator>(allocator, [](Allocator*) {});
    auto graph_storage = GraphInterface::MakeInstance(graph_param, graph_common_param);

    auto odescent_param = std::make_shared<ODescentParameter>();
    odescent_param->max_degree = static_cast<int64_t>(candidate_limit);
    odescent_param->alpha = alpha;
    odescent_param->sample_rate = 0.2F;
    odescent_param->turn = 30;
    odescent_param->min_in_degree = 1;
    ODescent odescent_builder(odescent_param, build_codes, allocator, thread_pool.get());
    odescent_builder.Build();
    odescent_builder.SaveGraph(graph_storage);

    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        Vector<InnerIdType> neighbors(allocator);
        graph_storage->GetNeighbors(inner_id, neighbors);
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        const auto count = std::min<uint64_t>(neighbors.size(), candidate_limit);
        graph.counts[inner_id] = static_cast<uint32_t>(count);
        auto* row = graph.neighbors.data() + static_cast<uint64_t>(inner_id) * graph.row_stride;
        std::copy_n(neighbors.data(), count, row);
    }
    return graph;
}

void
AssignMCICliquesToDatacell(CliqueDataCellPtr& mci_cliques,
                           Vector<Vector<InnerIdType>>& cliques,
                           uint64_t total,
                           Allocator* allocator) {
    Vector<InnerIdType> p_maxc(allocator);
    Vector<InnerIdType> maxcs(allocator);
    Vector<Vector<InnerIdType>> node_to_clique(total, Vector<InnerIdType>(allocator), allocator);
    p_maxc.push_back(0);
    for (InnerIdType clique_id = 0; clique_id < cliques.size(); ++clique_id) {
        for (auto inner_id : cliques[clique_id]) {
            maxcs.push_back(inner_id);
            node_to_clique[inner_id].push_back(clique_id);
        }
        p_maxc.push_back(static_cast<InnerIdType>(maxcs.size()));
    }

    Vector<InnerIdType> p_node_to_cid(allocator);
    Vector<InnerIdType> node_to_cids(allocator);
    p_node_to_cid.push_back(0);
    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        auto& ids = node_to_clique[inner_id];
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        node_to_cids.insert(node_to_cids.end(), ids.begin(), ids.end());
        p_node_to_cid.push_back(static_cast<InnerIdType>(node_to_cids.size()));
    }

    mci_cliques->Assign(std::move(p_maxc),
                        std::move(maxcs),
                        std::move(p_node_to_cid),
                        std::move(node_to_cids),
                        total);
}

void
HGraph::build_mci_clique_index(const float* vectors) {
    if (not use_mci_ or mci_cliques_ == nullptr) {
        return;
    }

    const auto total = this->total_count_.load();
    if (total == 0) {
        mci_cliques_->Clear(0);
        return;
    }

    logger::info("hgraph mci knn graph build started, total={}, mcs={}, knng_path={}",
                 total,
                 mci_mcs_,
                 mci_knng_path_);
    auto graph = build_mci_knn_graph(this->basic_flatten_codes_,
                                     total,
                                     this->mci_mcs_,
                                     this->mci_knng_path_,
                                     this->mci_alpha_,
                                     this->metric_,
                                     this->data_type_,
                                     this->dim_,
                                     static_cast<int64_t>(this->extra_info_size_),
                                     this->thread_pool_,
                                     this->allocator_);
    logger::info("hgraph mci knn graph build finished, total={}, row_stride={}, candidate_count={}",
                 total,
                 graph.row_stride,
                 graph.uniform_count);

    if (vectors != nullptr and this->data_type_ == DataTypes::DATA_TYPE_FLOAT) {
        const auto graph_max_degree =
            this->bottom_graph_ == nullptr ? 32U : this->bottom_graph_->MaximumDegree();
        const auto* graph_neighbors =
            graph.raw_neighbors != nullptr ? graph.raw_neighbors : graph.neighbors.data();
        MCIV3GraphView graph_view;
        graph_view.neighbors = graph_neighbors;
        graph_view.counts = graph.counts.empty() ? nullptr : graph.counts.data();
        graph_view.total = graph.total;
        graph_view.row_stride = graph.row_stride;
        graph_view.uniform_count = graph.uniform_count;
        MCIV3BuildParams build_params;
        build_params.total = total;
        build_params.dim = this->dim_;
        build_params.candidate_limit = std::min<uint64_t>(this->mci_mcs_, total - 1);
        build_params.clique_max = this->mci_clique_max_;
        build_params.max_degree = graph_max_degree;
        build_params.alpha = this->mci_alpha_;
        build_params.thread_count = static_cast<uint64_t>(this->build_thread_count_);
        auto cliques = BuildMCICliquesV3(vectors, graph_view, build_params, this->allocator_);
        AssignMCICliquesToDatacell(this->mci_cliques_, cliques, total, this->allocator_);
        logger::info("hgraph mci v3 clique build finished, total_cliques={}", cliques.size());
        return;
    }

    logger::info("hgraph mci clique enumeration started, total={}", total);
    Vector<Vector<InnerIdType>> cliques(this->allocator_);
    std::vector<std::atomic<uint32_t>> num_cliques_per_node(total);
    for (auto& count : num_cliques_per_node) {
        count.store(0, std::memory_order_relaxed);
    }

    auto get_clique_count = [&](InnerIdType inner_id) {
        return num_cliques_per_node[inner_id].load(std::memory_order_relaxed);
    };

    auto normalize_clique = [&](const Vector<InnerIdType>& clique) {
        Vector<InnerIdType> normalized(this->allocator_);
        if (clique.empty()) {
            return normalized;
        }
        normalized.assign(clique.begin(), clique.end());
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        return normalized;
    };

    auto append_selected_clique = [&](const Vector<InnerIdType>& clique) {
        if (clique.empty()) {
            return;
        }
        cliques.push_back(Vector<InnerIdType>(this->allocator_));
        cliques.back().assign(clique.begin(), clique.end());
    };

    auto try_select_clique = [&](const Vector<InnerIdType>& clique,
                                 Vector<Vector<InnerIdType>>& output) {
        auto normalized = normalize_clique(clique);
        if (normalized.empty()) {
            return false;
        }
        bool has_uncovered = false;
        for (auto inner_id : normalized) {
            if (get_clique_count(inner_id) == 0) {
                has_uncovered = true;
                break;
            }
        }
        if (not has_uncovered) {
            return false;
        }
        for (auto inner_id : normalized) {
            num_cliques_per_node[inner_id].fetch_add(1, std::memory_order_relaxed);
        }
        output.push_back(std::move(normalized));
        return true;
    };

    const auto candidate_limit = std::min<uint64_t>(this->mci_mcs_, total - 1);
    const auto clique_min = std::max<uint64_t>(
        2, std::min<uint64_t>({this->mci_clique_max_, candidate_limit + 1, total}));
    const auto node_clique_limit = std::max<uint32_t>(3, static_cast<uint32_t>(total / 100));
    const auto graph_max_degree =
        this->bottom_graph_ == nullptr ? 32U : this->bottom_graph_->MaximumDegree();
    const auto max_saved_per_seed =
        std::min<uint64_t>(candidate_limit, static_cast<uint64_t>(graph_max_degree + 2));

    auto collect_candidates = [&](InnerIdType seed,
                                  Vector<InnerIdType>& local_nodes,
                                  Vector<float>& seed_distances) {
        local_nodes.clear();
        seed_distances.clear();
        local_nodes.push_back(seed);
        seed_distances.push_back(0.0F);
        graph.ForEachNeighbor(seed, [&](InnerIdType neighbor) {
            if (get_clique_count(neighbor) >= node_clique_limit) {
                return true;
            }
            if (std::find(local_nodes.begin(), local_nodes.end(), neighbor) != local_nodes.end()) {
                return true;
            }
            local_nodes.push_back(neighbor);
            seed_distances.push_back(
                this->basic_flatten_codes_->ComputePairVectors(seed, neighbor));
            return local_nodes.size() <= candidate_limit;
        });
        Vector<uint64_t> order(this->allocator_);
        order.reserve(local_nodes.size() - 1);
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            order.push_back(i);
        }
        std::sort(order.begin(), order.end(), [&](uint64_t lhs, uint64_t rhs) {
            if (seed_distances[lhs] != seed_distances[rhs]) {
                return seed_distances[lhs] < seed_distances[rhs];
            }
            return local_nodes[lhs] < local_nodes[rhs];
        });
        Vector<InnerIdType> sorted_nodes(this->allocator_);
        Vector<float> sorted_distances(this->allocator_);
        sorted_nodes.reserve(local_nodes.size());
        sorted_distances.reserve(seed_distances.size());
        sorted_nodes.push_back(seed);
        sorted_distances.push_back(0.0F);
        for (auto idx : order) {
            sorted_nodes.push_back(local_nodes[idx]);
            sorted_distances.push_back(seed_distances[idx]);
        }
        local_nodes.swap(sorted_nodes);
        seed_distances.swap(sorted_distances);
    };

    auto build_local_edges = [&](float now_alpha,
                                 Vector<InnerIdType>& local_nodes,
                                 Vector<float>& seed_distances,
                                 Vector<uint8_t>& local_edges) {
        const auto local_count = local_nodes.size();
        local_edges.assign(local_count * local_count, 0);
        auto set_local_edge = [&](uint64_t lhs, uint64_t rhs) {
            local_edges[lhs * local_count + rhs] = 1;
        };
        uint64_t edge_count = 0;
        if (local_nodes.size() <= 1) {
            return edge_count;
        }
        const auto distance_limit = seed_distances[1] * now_alpha;
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            if (seed_distances[i] <= distance_limit) {
                set_local_edge(0, i);
                set_local_edge(i, 0);
                ++edge_count;
            }
        }
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            for (uint64_t j = i + 1; j < local_nodes.size(); ++j) {
                if (this->basic_flatten_codes_->ComputePairVectors(
                        local_nodes[i], local_nodes[j]) <= distance_limit) {
                    set_local_edge(i, j);
                    set_local_edge(j, i);
                    ++edge_count;
                }
            }
        }
        return edge_count;
    };

    auto append_high_alpha_fallback = [&](float now_alpha,
                                          const Vector<InnerIdType>& local_nodes,
                                          Vector<Vector<InnerIdType>>& output) {
        if (now_alpha <= 100.0F or local_nodes.empty()) {
            return false;
        }
        Vector<InnerIdType> fallback(this->allocator_);
        fallback.reserve(local_nodes.size());
        fallback.push_back(local_nodes.front());
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            if (get_clique_count(local_nodes[i]) > 0) {
                fallback.push_back(local_nodes[i]);
            }
        }
        return try_select_clique(fallback, output);
    };

    auto build_ccr_saved_group = [&](const Vector<uint64_t>& local_clique,
                                     const Vector<InnerIdType>& local_nodes,
                                     const Vector<uint8_t>& local_edges) {
        Vector<InnerIdType> group(this->allocator_);
        const auto local_count = local_nodes.size();
        if (local_clique.empty() or local_count == 0) {
            return group;
        }
        auto has_edge = [&](uint64_t lhs, uint64_t rhs) {
            return local_edges[lhs * local_count + rhs] != 0;
        };
        const auto root = local_clique.front();
        Vector<uint64_t> forward_neighbors(this->allocator_);
        forward_neighbors.reserve(local_count);
        for (uint64_t local_id = 0; local_id < local_count; ++local_id) {
            if (local_id != root and has_edge(root, local_id)) {
                forward_neighbors.push_back(local_id);
            }
        }
        if (forward_neighbors.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint64_t> degree(local_count, 0, this->allocator_);
        for (uint64_t i = 0; i < forward_neighbors.size(); ++i) {
            const auto lhs = forward_neighbors[i];
            for (uint64_t j = i + 1; j < forward_neighbors.size(); ++j) {
                const auto rhs = forward_neighbors[j];
                if (has_edge(lhs, rhs)) {
                    ++degree[lhs];
                    ++degree[rhs];
                }
            }
        }

        Vector<uint8_t> removed(local_count, 0, this->allocator_);
        Vector<uint64_t> queue(this->allocator_);
        for (auto node : forward_neighbors) {
            if (degree[node] + 2 < clique_min) {
                queue.push_back(node);
            }
        }
        for (uint64_t offset = 0; offset < queue.size(); ++offset) {
            const auto node = queue[offset];
            if (removed[node] != 0) {
                continue;
            }
            removed[node] = 1;
            for (auto neighbor : forward_neighbors) {
                if (removed[neighbor] != 0 or not has_edge(node, neighbor)) {
                    continue;
                }
                if (degree[neighbor] > 0) {
                    --degree[neighbor];
                }
                if (degree[neighbor] + 2 < clique_min) {
                    queue.push_back(neighbor);
                }
            }
        }

        Vector<uint64_t> active(this->allocator_);
        active.reserve(forward_neighbors.size());
        for (auto node : forward_neighbors) {
            if (removed[node] == 0) {
                active.push_back(node);
            }
        }
        if (active.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint64_t> active_degree(local_count, 0, this->allocator_);
        for (uint64_t i = 0; i < active.size(); ++i) {
            const auto lhs = active[i];
            for (uint64_t j = i + 1; j < active.size(); ++j) {
                const auto rhs = active[j];
                if (has_edge(lhs, rhs)) {
                    ++active_degree[lhs];
                    ++active_degree[rhs];
                }
            }
        }

        Vector<uint8_t> peeled(local_count, 0, this->allocator_);
        Vector<uint64_t> peel_order(this->allocator_);
        Vector<uint64_t> core_nodes(this->allocator_);
        Vector<uint64_t> p_nodes(this->allocator_);
        peel_order.reserve(active.size());
        core_nodes.reserve(active.size());
        p_nodes.reserve(active.size());
        for (uint64_t step = 0; step < active.size(); ++step) {
            uint64_t best = local_count;
            uint64_t best_degree = std::numeric_limits<uint64_t>::max();
            for (auto node : active) {
                if (peeled[node] == 0 and (active_degree[node] < best_degree or
                                           (active_degree[node] == best_degree and node < best))) {
                    best = node;
                    best_degree = active_degree[node];
                }
            }
            if (best == local_count) {
                break;
            }
            if (best_degree + step + 1 == active.size()) {
                p_nodes.assign(peel_order.begin(), peel_order.end());
                core_nodes.push_back(best);
                peeled[best] = 1;
                for (auto node : active) {
                    if (peeled[node] == 0) {
                        core_nodes.push_back(node);
                    }
                }
                break;
            }
            peeled[best] = 1;
            peel_order.push_back(best);
            for (auto neighbor : active) {
                if (peeled[neighbor] == 0 and has_edge(best, neighbor) and
                    active_degree[neighbor] > 0) {
                    --active_degree[neighbor];
                }
            }
        }
        if (core_nodes.empty()) {
            core_nodes.assign(active.begin(), active.end());
        }
        if (core_nodes.size() + 1 < clique_min) {
            return group;
        }

        Vector<uint8_t> in_core(local_count, 0, this->allocator_);
        Vector<uint8_t> in_p(local_count, 0, this->allocator_);
        for (auto node : core_nodes) {
            in_core[node] = 1;
        }
        for (auto node : p_nodes) {
            in_p[node] = 1;
        }
        auto count_core_neighbors = [&](uint64_t node) {
            uint64_t count = 0;
            for (auto core_node : core_nodes) {
                if (has_edge(node, core_node)) {
                    ++count;
                }
            }
            return count;
        };

        bool should_save = true;
        for (auto node : p_nodes) {
            if (count_core_neighbors(node) == core_nodes.size()) {
                should_save = false;
                break;
            }
        }
        if (should_save) {
            for (auto node : forward_neighbors) {
                if (node >= root or in_core[node] != 0 or in_p[node] != 0) {
                    continue;
                }
                if (count_core_neighbors(node) == core_nodes.size()) {
                    should_save = false;
                    break;
                }
            }
        }
        if (not should_save) {
            return group;
        }

        group.reserve(1 + core_nodes.size() + p_nodes.size());
        group.push_back(local_nodes[root]);
        for (auto node : core_nodes) {
            group.push_back(local_nodes[node]);
        }
        for (auto node : p_nodes) {
            group.push_back(local_nodes[node]);
        }
        return group;
    };

    auto solve_seed = [&](InnerIdType seed,
                          float now_alpha,
                          Vector<Vector<InnerIdType>>& output,
                          Vector<InnerIdType>& local_nodes,
                          Vector<float>& seed_distances,
                          Vector<uint8_t>& local_edges) {
        collect_candidates(seed, local_nodes, seed_distances);
        if (local_nodes.size() < clique_min) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }
        const auto edge_count =
            build_local_edges(now_alpha, local_nodes, seed_distances, local_edges);
        if (edge_count < clique_min * (clique_min - 1) / 2) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }
        auto local_cliques = run_local_ccr_mce(local_nodes,
                                               local_edges,
                                               num_cliques_per_node,
                                               clique_min,
                                               max_saved_per_seed,
                                               this->allocator_);
        if (local_cliques.empty()) {
            append_high_alpha_fallback(now_alpha, local_nodes, output);
            return;
        }

        uint64_t selected = 0;
        for (const auto& local_clique : local_cliques) {
            auto clique = build_ccr_saved_group(local_clique, local_nodes, local_edges);
            if (clique.empty()) {
                clique.reserve(local_clique.size());
                for (auto local_id : local_clique) {
                    clique.push_back(local_nodes[local_id]);
                }
            }
            bool has_uncovered = false;
            for (auto node : clique) {
                if (get_clique_count(node) == 0) {
                    has_uncovered = true;
                    break;
                }
            }
            if (has_uncovered and try_select_clique(clique, output)) {
                if (++selected >= graph_max_degree) {
                    break;
                }
            }
        }
    };

    auto log_round_progress =
        [&](uint64_t round_id, float now_alpha, uint64_t scanned, uint64_t& next_progress_percent) {
            while (next_progress_percent <= 100 and
                   scanned * 100 >= total * next_progress_percent) {
                logger::info(
                    "hgraph_mci_build_round_progress round={} alpha={} scanned={} total={} "
                    "progress={}%",
                    round_id,
                    now_alpha,
                    scanned,
                    total,
                    next_progress_percent);
                next_progress_percent += 10;
            }
        };

    auto solve_serial_round = [&](uint64_t round_id, float now_alpha) {
        Vector<InnerIdType> local_nodes(this->allocator_);
        Vector<float> seed_distances(this->allocator_);
        Vector<uint8_t> local_edges(this->allocator_);
        Vector<Vector<InnerIdType>> seed_cliques(this->allocator_);
        uint64_t next_progress_percent = 10;
        for (InnerIdType seed = 0; seed < total; ++seed) {
            if (get_clique_count(seed) != 0) {
                log_round_progress(round_id, now_alpha, seed + 1, next_progress_percent);
                continue;
            }
            seed_cliques.clear();
            solve_seed(seed, now_alpha, seed_cliques, local_nodes, seed_distances, local_edges);
            for (const auto& clique : seed_cliques) {
                append_selected_clique(clique);
            }
            log_round_progress(round_id, now_alpha, seed + 1, next_progress_percent);
        }
    };

    auto solve_parallel_round = [&](uint64_t round_id, float now_alpha) {
        if (this->thread_pool_ == nullptr or this->build_thread_count_ <= 1) {
            solve_serial_round(round_id, now_alpha);
            return;
        }

        const auto thread_count = this->build_thread_count_;
        const auto batch_seed_limit = std::max<uint64_t>(thread_count, thread_count * 1024);
        constexpr uint64_t seed_grain = 32;
        Vector<InnerIdType> batch_seeds(this->allocator_);
        batch_seeds.reserve(batch_seed_limit);

        std::vector<Vector<Vector<InnerIdType>>> thread_cliques;
        thread_cliques.reserve(thread_count);
        for (uint64_t thread_id = 0; thread_id < thread_count; ++thread_id) {
            thread_cliques.emplace_back(this->allocator_);
        }

        auto worker = [&](uint64_t thread_id,
                          const Vector<InnerIdType>& seeds,
                          std::atomic<uint64_t>& seed_cursor) {
            Vector<InnerIdType> local_nodes(this->allocator_);
            Vector<float> seed_distances(this->allocator_);
            Vector<uint8_t> local_edges(this->allocator_);
            auto& output = thread_cliques[thread_id];
            while (true) {
                const auto begin = seed_cursor.fetch_add(seed_grain, std::memory_order_relaxed);
                if (begin >= seeds.size()) {
                    break;
                }
                const auto end = std::min<uint64_t>(begin + seed_grain, seeds.size());
                for (uint64_t i = begin; i < end; ++i) {
                    if (get_clique_count(seeds[i]) != 0) {
                        continue;
                    }
                    solve_seed(
                        seeds[i], now_alpha, output, local_nodes, seed_distances, local_edges);
                }
            }
        };

        InnerIdType next_seed = 0;
        uint64_t next_progress_percent = 10;
        while (next_seed < total) {
            batch_seeds.clear();
            while (next_seed < total and batch_seeds.size() < batch_seed_limit) {
                if (get_clique_count(next_seed) == 0) {
                    batch_seeds.push_back(next_seed);
                }
                ++next_seed;
            }
            if (batch_seeds.empty()) {
                continue;
            }

            for (auto& one_thread_cliques : thread_cliques) {
                one_thread_cliques.clear();
            }
            const auto active_thread_count = std::min<uint64_t>(
                thread_count, (batch_seeds.size() + seed_grain - 1) / seed_grain);
            std::atomic<uint64_t> seed_cursor{0};
            std::vector<std::future<void>> futures;
            futures.reserve(active_thread_count);
            for (uint64_t thread_id = 0; thread_id < active_thread_count; ++thread_id) {
                futures.emplace_back(this->thread_pool_->GeneralEnqueue(
                    worker, thread_id, std::cref(batch_seeds), std::ref(seed_cursor)));
            }
            for (auto& future : futures) {
                future.get();
            }
            for (const auto& one_thread_cliques : thread_cliques) {
                for (const auto& clique : one_thread_cliques) {
                    append_selected_clique(clique);
                }
            }
            log_round_progress(round_id, now_alpha, next_seed, next_progress_percent);
        }
    };

    float now_alpha = std::max(1.2F, this->mci_alpha_);
    uint64_t previous_uncovered = total;
    for (uint64_t round = 0; round < 16; ++round) {
        const auto round_id = round + 1;
        const auto cliques_before_round = cliques.size();
        logger::info("hgraph_mci_build_round_start round={} alpha={} total={} total_cliques={}",
                     round_id,
                     now_alpha,
                     total,
                     cliques.size());
        solve_parallel_round(round_id, now_alpha);

        uint64_t uncovered = 0;
        for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
            if (get_clique_count(inner_id) == 0) {
                ++uncovered;
            }
        }
        logger::info(
            "hgraph_mci_build_round round={} alpha={} uncovered={} round_cliques={} "
            "total_cliques={}",
            round_id,
            now_alpha,
            uncovered,
            cliques.size() - cliques_before_round,
            cliques.size());
        if (uncovered == 0) {
            break;
        }
        const auto previous_alpha = now_alpha;
        if (uncovered < previous_uncovered * 9 / 10) {
            now_alpha += std::max(1.2F, this->mci_alpha_);
        } else {
            now_alpha *= 2.0F;
        }
        logger::info(
            "hgraph_mci_build_alpha_update round={} previous_alpha={} next_alpha={} uncovered={} "
            "previous_uncovered={}",
            round_id,
            previous_alpha,
            now_alpha,
            uncovered,
            previous_uncovered);
        previous_uncovered = uncovered;
    }

    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        if (get_clique_count(inner_id) == 0) {
            Vector<InnerIdType> singleton(this->allocator_);
            singleton.push_back(inner_id);
            graph.ForEachNeighbor(inner_id, [&](InnerIdType neighbor) {
                if (std::find(singleton.begin(), singleton.end(), neighbor) != singleton.end()) {
                    return true;
                }
                singleton.push_back(neighbor);
                return singleton.size() < graph_max_degree;
            });
            Vector<Vector<InnerIdType>> fallback_cliques(this->allocator_);
            if (try_select_clique(singleton, fallback_cliques)) {
                append_selected_clique(fallback_cliques.front());
            }
        }
    }

    uint64_t max_membership = 0;
    uint64_t total_memberships = 0;
    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        const auto membership = static_cast<uint64_t>(get_clique_count(inner_id));
        total_memberships += membership;
        max_membership = std::max(max_membership, membership);
    }

    Vector<InnerIdType> p_maxc(this->allocator_);
    Vector<InnerIdType> maxcs(this->allocator_);
    Vector<Vector<InnerIdType>> node_to_clique(
        total, Vector<InnerIdType>(this->allocator_), this->allocator_);
    p_maxc.push_back(0);
    for (InnerIdType clique_id = 0; clique_id < cliques.size(); ++clique_id) {
        for (auto inner_id : cliques[clique_id]) {
            maxcs.push_back(inner_id);
            node_to_clique[inner_id].push_back(clique_id);
        }
        p_maxc.push_back(static_cast<InnerIdType>(maxcs.size()));
    }

    Vector<InnerIdType> p_node_to_cid(this->allocator_);
    Vector<InnerIdType> node_to_cids(this->allocator_);
    p_node_to_cid.push_back(0);
    for (InnerIdType inner_id = 0; inner_id < total; ++inner_id) {
        auto& ids = node_to_clique[inner_id];
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        node_to_cids.insert(node_to_cids.end(), ids.begin(), ids.end());
        p_node_to_cid.push_back(static_cast<InnerIdType>(node_to_cids.size()));
    }

    mci_cliques_->Assign(std::move(p_maxc),
                         std::move(maxcs),
                         std::move(p_node_to_cid),
                         std::move(node_to_cids),
                         total);
    logger::info(
        "hgraph mci clique build finished, total_cliques={}, total_memberships={}, "
        "avg_membership={}, max_membership={}",
        cliques.size(),
        total_memberships,
        static_cast<double>(total_memberships) / static_cast<double>(total),
        max_membership);
}

Vector<InnerIdType>
HGraph::find_mci_knn_for_new_node(InnerIdType new_inner_id, const void* vector) const {
    Vector<InnerIdType> knn_ids(this->allocator_);
    const auto total = this->total_count_.load();
    if (total <= 1 or this->bottom_graph_ == nullptr or this->basic_flatten_codes_ == nullptr) {
        return knn_ids;
    }

    const auto visible_total = static_cast<uint64_t>(new_inner_id) + 1;
    if (visible_total <= 1) {
        return knn_ids;
    }

    const auto k = std::min<uint64_t>(this->mci_mcs_, visible_total - 1);
    if (k == 0) {
        return knn_ids;
    }

    if (this->entry_point_id_ != INVALID_ENTRY_POINT and vector != nullptr and
        this->GetNumElements() > 0) {
        auto query = Dataset::Make();
        query->NumElements(1)
            ->Dim(static_cast<int64_t>(this->dim_))
            ->Float32Vectors(static_cast<const float*>(vector))
            ->Owner(false);

        const auto hgraph_total =
            std::min<uint64_t>(total, static_cast<uint64_t>(this->GetNumElements()));
        auto query_k = std::min<uint64_t>(k + 1, hgraph_total);
        while (query_k > 0) {
            const auto hgraph_ef = std::max<uint64_t>(query_k, 100);
            const auto hgraph_search_params = std::string(R"({"hgraph":{"ef_search":)") +
                                              std::to_string(hgraph_ef) + R"(,"use_mci":false}})";
            auto result = this->KnnSearch(
                query, static_cast<int64_t>(query_k), hgraph_search_params, nullptr);
            if (result == nullptr) {
                break;
            }

            knn_ids.clear();
            const auto* labels = result->GetIds();
            const auto count = result->GetDim();
            knn_ids.reserve(static_cast<uint64_t>(count));
            for (int64_t i = 0; i < count; ++i) {
                auto [found, inner_id] = this->label_table_->TryGetIdByLabel(labels[i]);
                if (found and inner_id < visible_total and inner_id != new_inner_id) {
                    knn_ids.push_back(inner_id);
                }
            }

            std::sort(knn_ids.begin(), knn_ids.end());
            knn_ids.erase(std::unique(knn_ids.begin(), knn_ids.end()), knn_ids.end());
            if (knn_ids.size() >= k or query_k == hgraph_total) {
                break;
            }
            const auto next_query_k = std::min<uint64_t>(query_k * 2, hgraph_total);
            if (next_query_k == query_k) {
                break;
            }
            query_k = next_query_k;
        }
    }

    std::sort(knn_ids.begin(), knn_ids.end());
    knn_ids.erase(std::unique(knn_ids.begin(), knn_ids.end()), knn_ids.end());
    if (knn_ids.size() > k) {
        Vector<std::pair<float, InnerIdType>> sorted_neighbors(this->allocator_);
        sorted_neighbors.reserve(knn_ids.size());
        for (auto neighbor : knn_ids) {
            sorted_neighbors.emplace_back(
                this->basic_flatten_codes_->ComputePairVectors(new_inner_id, neighbor), neighbor);
        }
        std::sort(
            sorted_neighbors.begin(),
            sorted_neighbors.end(),
            [](const std::pair<float, InnerIdType>& lhs, const std::pair<float, InnerIdType>& rhs) {
                if (lhs.first != rhs.first) {
                    return lhs.first < rhs.first;
                }
                return lhs.second < rhs.second;
            });
        knn_ids.clear();
        knn_ids.reserve(k);
        for (uint64_t i = 0; i < k; ++i) {
            knn_ids.push_back(sorted_neighbors[i].second);
        }
    }
    return knn_ids;
}

bool
HGraph::try_join_mci_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids) {
    if (this->mci_cliques_ == nullptr or knn_ids.empty()) {
        return false;
    }

    Vector<InnerIdType> candidate_cliques(this->allocator_);
    for (auto neighbor : knn_ids) {
        if (neighbor >= this->total_count_.load()) {
            continue;
        }
        this->mci_cliques_->CollectNodeCliqueIds(neighbor, candidate_cliques);
    }
    if (candidate_cliques.empty()) {
        return false;
    }
    std::sort(candidate_cliques.begin(), candidate_cliques.end());

    Vector<std::pair<uint64_t, InnerIdType>> targets(this->allocator_);
    const auto logical_clique_count = this->mci_cliques_->TotalLogicalCliqueCount();
    auto iter = candidate_cliques.begin();
    while (iter != candidate_cliques.end()) {
        const auto clique_id = *iter;
        auto next = iter;
        while (next != candidate_cliques.end() and *next == clique_id) {
            ++next;
        }
        const auto inter = static_cast<uint64_t>(std::distance(iter, next));
        iter = next;
        if (clique_id >= logical_clique_count) {
            continue;
        }

        const auto member_count = this->mci_cliques_->GetCliqueMemberCount(clique_id);
        if (member_count == 0) {
            continue;
        }
        const auto ratio = static_cast<float>(inter) / static_cast<float>(member_count);
        if (ratio >= this->mci_incremental_join_ratio_threshold_) {
            targets.emplace_back(inter, clique_id);
        }
    }
    if (targets.empty()) {
        return false;
    }

    std::sort(targets.begin(),
              targets.end(),
              [](const std::pair<uint64_t, InnerIdType>& lhs,
                 const std::pair<uint64_t, InnerIdType>& rhs) {
                  if (lhs.first != rhs.first) {
                      return lhs.first > rhs.first;
                  }
                  return lhs.second < rhs.second;
              });
    const auto target_count =
        std::min<uint64_t>(this->mci_incremental_added_mct_, static_cast<uint64_t>(targets.size()));
    const auto total = this->total_count_.load();
    for (uint64_t i = 0; i < target_count; ++i) {
        this->mci_cliques_->AppendNodeToClique(new_inner_id, targets[i].second, total);
    }
    return true;
}

void
HGraph::build_incremental_mci_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids) {
    if (this->mci_cliques_ == nullptr) {
        return;
    }

    Vector<InnerIdType> members(this->allocator_);
    if (knn_ids.empty()) {
        members.push_back(new_inner_id);
        this->mci_cliques_->AppendNewClique(members, this->total_count_.load());
        return;
    }

    Vector<std::pair<float, InnerIdType>> sorted_neighbors(this->allocator_);
    sorted_neighbors.reserve(knn_ids.size());
    for (auto neighbor : knn_ids) {
        sorted_neighbors.emplace_back(
            this->basic_flatten_codes_->ComputePairVectors(new_inner_id, neighbor), neighbor);
    }
    std::sort(
        sorted_neighbors.begin(),
        sorted_neighbors.end(),
        [](const std::pair<float, InnerIdType>& lhs, const std::pair<float, InnerIdType>& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

    const auto total = this->total_count_.load();
    const auto visible_total = static_cast<uint64_t>(new_inner_id) + 1;
    const auto incremental_clique_max =
        std::min<uint64_t>(this->mci_incremental_clique_max_, visible_total);
    const auto candidate_limit =
        std::min<uint64_t>({this->mci_mcs_,
                            incremental_clique_max > 0 ? incremental_clique_max - 1 : 0,
                            visible_total > 0 ? visible_total - 1 : 0});
    const auto clique_min = std::max<uint64_t>(
        2, std::min<uint64_t>({this->mci_clique_max_, candidate_limit + 1, visible_total}));
    const auto nearest_distance = sorted_neighbors.front().first;

    Vector<InnerIdType> best(this->allocator_);
    float now_alpha = std::max(1.2F, this->mci_alpha_);
    while (true) {
        const auto distance_limit = nearest_distance * now_alpha;
        Vector<InnerIdType> clique(this->allocator_);
        clique.push_back(new_inner_id);
        for (const auto& [distance, neighbor] : sorted_neighbors) {
            if (distance > distance_limit) {
                break;
            }
            bool connected = true;
            for (auto member : clique) {
                if (member == new_inner_id) {
                    continue;
                }
                if (this->basic_flatten_codes_->ComputePairVectors(member, neighbor) >
                    distance_limit) {
                    connected = false;
                    break;
                }
            }
            if (connected) {
                clique.push_back(neighbor);
                if (clique.size() >= incremental_clique_max) {
                    break;
                }
            }
        }
        if (clique.size() > best.size()) {
            best.swap(clique);
        }
        if (best.size() >= clique_min or now_alpha > 100.0F) {
            break;
        }
        now_alpha *= 2.0F;
    }

    if (best.size() < 2) {
        best.clear();
        best.push_back(new_inner_id);
        best.push_back(sorted_neighbors.front().second);
    } else if (best.size() > incremental_clique_max) {
        best.resize(incremental_clique_max);
    }
    this->mci_cliques_->AppendNewClique(best, total);
}

void
HGraph::incremental_update_mci_clique(InnerIdType new_inner_id, const void* vector) {
    if (not this->use_mci_ or this->mci_cliques_ == nullptr) {
        return;
    }
    auto knn_ids = this->find_mci_knn_for_new_node(new_inner_id, vector);
    if (knn_ids.empty()) {
        Vector<InnerIdType> singleton(this->allocator_);
        singleton.push_back(new_inner_id);
        this->mci_cliques_->AppendNewClique(singleton, this->total_count_.load());
        return;
    }
    if (not this->try_join_mci_clique(new_inner_id, knn_ids)) {
        this->build_incremental_mci_clique(new_inner_id, knn_ids);
    }
}

}  // namespace vsag
