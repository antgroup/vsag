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
#include <cmath>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <vector>

#include "../mci/mci_builder.h"
#include "datacell/graph_interface.h"
#include "datacell/sparse_graph_datacell_parameter.h"
#include "hgraph.h"
#include "impl/filter/filter_headers.h"
#include "impl/logger/logger.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/odescent/odescent_graph_parameter.h"
#include "index_common_param.h"

namespace vsag {
namespace {

constexpr uint64_t K_MCI_MIN_CLIQUE_SIZE = 2;
constexpr uint64_t K_MCI_BITSET_SEED_SAMPLE_MULTIPLIER = 64;
constexpr uint64_t K_MCI_MIN_BITSET_SEED_SAMPLES = 4096;

Vector<InnerIdType>
collect_seed_inner_ids(const FilterPtr& filter,
                       const LabelTablePtr& label_table,
                       uint64_t seed_count,
                       Allocator* allocator) {
    Vector<InnerIdType> inner_ids(allocator);
    if (filter == nullptr or label_table == nullptr or seed_count == 0) {
        return inner_ids;
    }

    const int64_t* valid_labels = nullptr;
    int64_t valid_count = 0;
    filter->GetValidIds(&valid_labels, valid_count);
    if (valid_labels == nullptr or valid_count <= 0) {
        return inner_ids;
    }

    const auto target_count = std::min<uint64_t>(seed_count, static_cast<uint64_t>(valid_count));
    inner_ids.reserve(target_count);
    UnorderedSet<InnerIdType> selected_ids(allocator);
    selected_ids.reserve(target_count);
    auto try_add_label = [&](uint64_t offset) {
        auto [found, inner_id] = label_table->TryGetIdByLabel(valid_labels[offset]);
        if (found and selected_ids.emplace(inner_id).second) {
            inner_ids.push_back(inner_id);
        }
    };
    for (uint64_t i = 0; i < target_count; ++i) {
        try_add_label(i * static_cast<uint64_t>(valid_count) / target_count);
    }
    for (uint64_t offset = 0;
         offset < static_cast<uint64_t>(valid_count) and inner_ids.size() < target_count;
         ++offset) {
        try_add_label(offset);
    }
    std::sort(inner_ids.begin(), inner_ids.end());
    inner_ids.erase(std::unique(inner_ids.begin(), inner_ids.end()), inner_ids.end());
    return inner_ids;
}

bool
has_valid_id_source(const FilterPtr& filter) {
    if (filter == nullptr) {
        return false;
    }
    const int64_t* valid_ids = nullptr;
    int64_t valid_count = 0;
    filter->GetValidIds(&valid_ids, valid_count);
    return valid_ids != nullptr and valid_count > 0;
}

bool
has_bitset_source(const FilterPtr& filter) {
    const auto bitset_filter = std::dynamic_pointer_cast<BlackListFilter>(filter);
    return bitset_filter != nullptr and bitset_filter->IsBitsetFilter();
}

Vector<InnerIdType>
collect_bitset_seed_inner_ids(const FilterPtr& inner_filter,
                              uint64_t total,
                              uint64_t seed_count,
                              Allocator* allocator) {
    Vector<InnerIdType> inner_ids(allocator);
    if (inner_filter == nullptr or total == 0 or seed_count == 0) {
        return inner_ids;
    }
    inner_ids.reserve(std::min(total, seed_count));
    const auto scaled_sample_count = seed_count > total / K_MCI_BITSET_SEED_SAMPLE_MULTIPLIER
                                         ? total
                                         : seed_count * K_MCI_BITSET_SEED_SAMPLE_MULTIPLIER;
    const auto sample_count =
        std::min(total, std::max(K_MCI_MIN_BITSET_SEED_SAMPLES, scaled_sample_count));
    const auto id_step = total / sample_count;
    const auto id_remainder = total % sample_count;
    uint64_t id = 0;
    uint64_t accumulated_remainder = 0;
    for (uint64_t sample = 0; sample < sample_count and inner_ids.size() < seed_count; ++sample) {
        const auto inner_id = static_cast<InnerIdType>(id);
        if (inner_filter->CheckValid(inner_id)) {
            inner_ids.push_back(inner_id);
        }
        id += id_step;
        accumulated_remainder += id_remainder;
        if (accumulated_remainder >= sample_count) {
            ++id;
            accumulated_remainder -= sample_count;
        }
    }
    return inner_ids;
}

float
// NOLINTNEXTLINE(readability-identifier-naming)
ExpandMCIDistanceLimit(float nearest_distance, float alpha, MetricType metric) {
    if (metric != MetricType::METRIC_TYPE_IP) {
        return nearest_distance * alpha;
    }
    const auto nearest_similarity = 1.0F - nearest_distance;
    const auto similarity_limit =
        nearest_similarity >= 0.0F ? nearest_similarity / alpha : nearest_similarity * alpha;
    return 1.0F - similarity_limit;
}

// Temporary KNN graph used during MCI companion-index construction. Rows may come from an
// external KNNG file or from an ODescent graph built over the current vectors.
// NOLINTNEXTLINE(readability-identifier-naming)
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

    // Visit valid neighbors in one row and hide whether the backing storage is raw or Vector based.
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

    // Allocate a raw neighbor buffer for an external KNNG file.
    void
    AllocateRawNeighbors(uint64_t count) {
        this->ReleaseRawNeighbors();
        raw_neighbors = static_cast<InnerIdType*>(allocator->Allocate(count * sizeof(InnerIdType)));
        raw_neighbor_count = count;
    }

    // Release only the optional raw buffer; Vector storage is managed by Vector itself.
    void
    ReleaseRawNeighbors() {
        if (raw_neighbors != nullptr) {
            allocator->Deallocate(raw_neighbors);
            raw_neighbors = nullptr;
            raw_neighbor_count = 0;
        }
    }
};

// Run local CCR-MCE around one seed neighborhood. Returned clique members are local indices into
// local_nodes; the caller maps them back to HGraph inner ids.
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

    // Keep only branches that can still cover at least one node not already assigned to a clique.
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

    // Save a qualifying local clique and mark its uncovered nodes as covered for this seed.
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

    // Bron-Kerbosch expansion with pivoting and coverage/size pruning.
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

HGraph::MCIHybridSearchResult::MCIHybridSearchResult(const HGraphSearchParameters& params,
                                                     const FilterPtr& filter)
    : valid_ratio(filter != nullptr ? filter->ValidRatio() : 1.0F),
      threshold(params.mci_hgraph_valid_ratio_threshold),
      seed_ratio(params.mci_seed_ratio) {
}

JsonType
HGraph::MCIHybridSearchResult::MakeStatistics(const SearchStatistics& stats) const {
    auto json = stats.ToJson();
    json["mci_hybrid_route"].SetString(this->route);
    json["mci_hybrid_valid_ratio"].SetFloat(this->valid_ratio);
    json["mci_hybrid_threshold"].SetFloat(this->threshold);
    json["mci_seed_count"].SetInt(static_cast<int64_t>(this->seed_count));
    json["mci_seed_ratio"].SetFloat(this->seed_ratio);
    json["mci_raw_float_csr"].SetBool(this->used_precise_float_csr);
    return json;
}

HGraph::MCIHybridSearchResult
HGraph::try_mci_search(const SearchRequest& request,
                       const HGraphSearchParameters& params,
                       const FilterPtr& inner_filter,
                       const void* query,
                       const InnerSearchParam& search_param,
                       QueryContext* ctx) const {
    MCIHybridSearchResult result(params, inner_filter);
    const bool bitset_seed_source = has_bitset_source(request.filter_);
    const bool can_use_mci = params.use_mci and this->mci_parameters_.enabled and
                             search_param.executors.empty() and
                             (has_valid_id_source(request.filter_) or bitset_seed_source) and
                             this->mci_cliques_ != nullptr and
                             this->mci_cliques_->HasCliqueIndex(this->total_count_.load());
    if (not can_use_mci) {
        return result;
    }

    result.route = "hgraph";
    if (request.mode_ == SearchMode::RANGE_SEARCH or result.valid_ratio >= result.threshold) {
        return result;
    }

    MCISearcherParam mci_param;
    const auto total_count = this->total_count_.load();
    const auto scaled_seed_count = std::ceil(std::sqrt(static_cast<double>(total_count)) *
                                             static_cast<double>(params.mci_seed_ratio));
    mci_param.seed_count = scaled_seed_count >= static_cast<double>(total_count)
                               ? total_count
                               : std::max<uint64_t>(1, static_cast<uint64_t>(scaled_seed_count));
    mci_param.hops_limit = search_param.hops_limit;

    Vector<InnerIdType> seed_inner_ids(ctx->alloc);
    if (bitset_seed_source) {
        seed_inner_ids = collect_bitset_seed_inner_ids(
            inner_filter, total_count, mci_param.seed_count, ctx->alloc);
    } else {
        std::shared_lock label_lock(this->label_lookup_mutex_);
        seed_inner_ids = collect_seed_inner_ids(
            request.filter_, this->label_table_, mci_param.seed_count, ctx->alloc);
    }
    result.seed_count = seed_inner_ids.size();
    mci_param.seed_count = result.seed_count;
    mci_param.seed_inner_ids = &seed_inner_ids;
    if (bitset_seed_source and seed_inner_ids.empty()) {
        return result;
    }

    std::shared_lock<std::shared_mutex> codes_lock(this->persistent_codes_mutex_);
    auto precise_flatten = this->get_precise_codes();
    if (precise_flatten != nullptr) {
        uint64_t precise_vector_stride = 0;
        mci_param.precise_vectors =
            precise_flatten->TryGetContiguousRawFloatData(&precise_vector_stride);
        mci_param.dim = this->dim_;
        mci_param.precise_vector_stride = precise_vector_stride;
        mci_param.metric = this->metric_;
        mci_param.used_precise_float_csr = &result.used_precise_float_csr;
    }
    result.result = this->mci_searcher_->Search(
        this->mci_cliques_, precise_flatten, query, search_param, mci_param, ctx);
    result.route = "mci";
    return result;
}

// Build the MCI KNN candidate graph from an external KNNG file when available,
// otherwise build it from the current flattened vectors with ODescent.
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

// Convert selected cliques into the compact clique-member and node-to-clique layout.
void
// NOLINTNEXTLINE(readability-identifier-naming)
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

// Build or rebuild the full MCI companion clique index for the current HGraph contents.
void
HGraph::build_mci_clique_index(const float* vectors) {
    if (not mci_parameters_.enabled or mci_cliques_ == nullptr) {
        return;
    }
    std::scoped_lock build_lock(this->mci_build_mutex_);
    std::shared_lock<std::shared_mutex> codes_lock(this->persistent_codes_mutex_);

    const auto total = this->total_count_.load();
    if (total == 0) {
        mci_cliques_->Clear(0);
        this->cal_memory_usage();
        return;
    }

    logger::info("hgraph mci knn graph build started, total={}, mcs={}, knng_path={}",
                 total,
                 mci_parameters_.mcs,
                 mci_parameters_.knng_path);
    auto graph = build_mci_knn_graph(this->basic_flatten_codes_,
                                     total,
                                     this->mci_parameters_.mcs,
                                     this->mci_parameters_.knng_path,
                                     this->mci_parameters_.alpha,
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
    const auto precise_codes = this->get_precise_codes();

    // Fast float32 path delegates clique construction to the dedicated MCI builder. Prefer
    // persistent codes because their rows always follow assigned inner IDs, including builds that
    // skipped duplicate labels in the input batch.
    uint64_t persistent_stride = 0;
    if (this->data_type_ == DataTypes::DATA_TYPE_FLOAT) {
        const auto* const persistent_vectors =
            precise_codes->TryGetContiguousRawFloatData(&persistent_stride);
        if (persistent_vectors != nullptr and persistent_stride == this->dim_) {
            vectors = persistent_vectors;
        }
    }
    if (vectors != nullptr and this->data_type_ == DataTypes::DATA_TYPE_FLOAT) {
        const auto graph_max_degree =
            this->bottom_graph_ == nullptr ? 32U : this->bottom_graph_->MaximumDegree();
        const auto* graph_neighbors =
            graph.raw_neighbors != nullptr ? graph.raw_neighbors : graph.neighbors.data();
        MCIGraphView graph_view;
        graph_view.neighbors = graph_neighbors;
        graph_view.counts = graph.counts.empty() ? nullptr : graph.counts.data();
        graph_view.total = graph.total;
        graph_view.row_stride = graph.row_stride;
        graph_view.uniform_count = graph.uniform_count;
        MCIV3BuildParams build_params;
        build_params.total = total;
        build_params.dim = this->dim_;
        build_params.candidate_limit = std::min<uint64_t>(this->mci_parameters_.mcs, total - 1);
        build_params.clique_max = this->mci_parameters_.clique_max;
        build_params.max_degree = graph_max_degree;
        build_params.alpha = this->mci_parameters_.alpha;
        build_params.thread_count = static_cast<uint64_t>(this->build_thread_count_);
        build_params.metric = this->metric_;
        auto cliques = BuildMCICliques(vectors, graph_view, build_params, this->allocator_);
        AssignMCICliquesToDatacell(this->mci_cliques_, cliques, total, this->allocator_);
        logger::info("hgraph mci v3 clique build finished, total_cliques={}", cliques.size());
        this->cal_memory_usage();
        return;
    }

    // Generic path for quantized or non-float data: enumerate cliques from local seed graphs.
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
        const auto anchor = clique.front();
        normalized.assign(clique.begin(), clique.end());
        std::sort(normalized.begin(), normalized.end());
        normalized.erase(std::unique(normalized.begin(), normalized.end()), normalized.end());
        if (normalized.size() > this->mci_parameters_.clique_max) {
            normalized.resize(this->mci_parameters_.clique_max);
            if (std::find(normalized.begin(), normalized.end(), anchor) == normalized.end()) {
                normalized.back() = anchor;
            }
        }
        return normalized;
    };

    // Append a selected clique to the final output list.
    auto append_selected_clique = [&](const Vector<InnerIdType>& clique) {
        if (clique.empty()) {
            return;
        }
        cliques.push_back(Vector<InnerIdType>(this->allocator_));
        cliques.back().assign(clique.begin(), clique.end());
    };

    // Accept a clique only if it covers at least one previously uncovered node.
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

    const auto candidate_limit = std::min<uint64_t>(this->mci_parameters_.mcs, total - 1);
    const auto clique_min = std::min<uint64_t>({K_MCI_MIN_CLIQUE_SIZE, candidate_limit + 1, total});
    const auto node_clique_limit = std::max<uint32_t>(3, static_cast<uint32_t>(total / 100));
    const auto graph_max_degree =
        this->bottom_graph_ == nullptr ? 32U : this->bottom_graph_->MaximumDegree();
    const auto max_saved_per_seed =
        std::min<uint64_t>(candidate_limit, static_cast<uint64_t>(graph_max_degree + 2));

    // Collect one seed's candidate neighborhood and sort neighbors by distance from the seed.
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
            seed_distances.push_back(precise_codes->ComputePairVectors(seed, neighbor));
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

    // Materialize the seed-local adjacency matrix using the alpha-scaled distance limit.
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
        const auto distance_limit =
            ExpandMCIDistanceLimit(seed_distances[1], now_alpha, this->metric_);
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            if (seed_distances[i] <= distance_limit) {
                set_local_edge(0, i);
                set_local_edge(i, 0);
                ++edge_count;
            }
        }
        for (uint64_t i = 1; i < local_nodes.size(); ++i) {
            for (uint64_t j = i + 1; j < local_nodes.size(); ++j) {
                if (precise_codes->ComputePairVectors(local_nodes[i], local_nodes[j]) <=
                    distance_limit) {
                    set_local_edge(i, j);
                    set_local_edge(j, i);
                    ++edge_count;
                }
            }
        }
        return edge_count;
    };

    // At very high alpha, emit a simple fallback clique so the build continues making progress.
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

    // Process a single seed through candidate collection, local edge construction, and clique save.
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
            Vector<InnerIdType> clique(this->allocator_);
            clique.reserve(local_clique.size());
            for (auto local_id : local_clique) {
                clique.push_back(local_nodes[local_id]);
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

    // Log coarse progress within a long MCI build round.
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

    // Single-threaded round executor used directly or as a fallback.
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

    // Parallel round executor: batch uncovered seeds, solve them in workers, then merge results.
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

        // Keep per-worker scratch buffers local to reduce allocation churn and data sharing.
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

    // Re-run rounds with increasing alpha until every node is covered or progress stalls out.
    float now_alpha = std::max(1.2F, this->mci_parameters_.alpha);
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
            now_alpha += std::max(1.2F, this->mci_parameters_.alpha);
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

    // Final safety pass: every node must belong to at least one MCI clique.
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
    this->cal_memory_usage();
}

// Find KNN candidates for a newly inserted node using the current HGraph search path.
Vector<InnerIdType>
HGraph::find_mci_knn_for_new_node(InnerIdType new_inner_id, const void* vector) const {
    Vector<InnerIdType> knn_ids(this->allocator_);
    const auto precise_codes = this->get_precise_codes();
    const auto total = this->total_count_.load();
    if (total <= 1 or this->bottom_graph_ == nullptr or this->basic_flatten_codes_ == nullptr) {
        return knn_ids;
    }

    const auto visible_total = static_cast<uint64_t>(new_inner_id) + 1;
    if (visible_total <= 1) {
        return knn_ids;
    }

    const auto k = std::min<uint64_t>(this->mci_parameters_.mcs, visible_total - 1);
    if (k == 0) {
        return knn_ids;
    }

    if (this->entry_point_id_ != INVALID_ENTRY_POINT and vector != nullptr and
        this->GetNumElements() > 0) {
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(static_cast<int64_t>(this->dim_))->Owner(false);
        if (this->data_type_ == DataTypes::DATA_TYPE_FLOAT) {
            query->Float32Vectors(static_cast<const float*>(vector));
        } else if (this->data_type_ == DataTypes::DATA_TYPE_INT8) {
            query->Int8Vectors(static_cast<const int8_t*>(vector));
        } else if (this->data_type_ == DataTypes::DATA_TYPE_FP16 or
                   this->data_type_ == DataTypes::DATA_TYPE_BF16) {
            query->Float16Vectors(static_cast<const uint16_t*>(vector));
        } else if (this->data_type_ == DataTypes::DATA_TYPE_SPARSE) {
            query->SparseVectors(static_cast<const SparseVector*>(vector));
        }

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
            std::shared_lock label_lock(this->label_lookup_mutex_);
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
            sorted_neighbors.emplace_back(precise_codes->ComputePairVectors(new_inner_id, neighbor),
                                          neighbor);
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

// Try to attach a new node to existing cliques that strongly overlap its KNN set.
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
        if (member_count == 0 or member_count >= this->mci_parameters_.incremental_clique_max) {
            continue;
        }
        const auto ratio = static_cast<float>(inter) / static_cast<float>(member_count);
        if (ratio >= this->mci_parameters_.incremental_join_ratio_threshold) {
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
    const auto target_count = std::min<uint64_t>(this->mci_parameters_.incremental_added_mct,
                                                 static_cast<uint64_t>(targets.size()));
    const auto total = this->total_count_.load();
    bool appended = false;
    for (uint64_t i = 0; i < target_count; ++i) {
        appended |= this->mci_cliques_->AppendNodeToClique(
            new_inner_id, targets[i].second, total, this->mci_parameters_.incremental_clique_max);
    }
    return appended;
}

// Create one new incremental clique around an inserted node when joining old cliques is not enough.
void
HGraph::build_incremental_mci_clique(InnerIdType new_inner_id, const Vector<InnerIdType>& knn_ids) {
    if (this->mci_cliques_ == nullptr) {
        return;
    }
    const auto precise_codes = this->get_precise_codes();

    Vector<InnerIdType> members(this->allocator_);
    if (knn_ids.empty()) {
        members.push_back(new_inner_id);
        this->mci_cliques_->AppendNewClique(members, this->total_count_.load());
        return;
    }

    Vector<std::pair<float, InnerIdType>> sorted_neighbors(this->allocator_);
    sorted_neighbors.reserve(knn_ids.size());
    for (auto neighbor : knn_ids) {
        sorted_neighbors.emplace_back(precise_codes->ComputePairVectors(new_inner_id, neighbor),
                                      neighbor);
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
        std::min<uint64_t>(this->mci_parameters_.incremental_clique_max, visible_total);
    const auto candidate_limit =
        std::min<uint64_t>({this->mci_parameters_.mcs,
                            incremental_clique_max > 0 ? incremental_clique_max - 1 : 0,
                            visible_total > 0 ? visible_total - 1 : 0});
    const auto clique_min =
        std::min<uint64_t>({K_MCI_MIN_CLIQUE_SIZE, candidate_limit + 1, visible_total});
    const auto nearest_distance = sorted_neighbors.front().first;

    Vector<InnerIdType> best(this->allocator_);
    float now_alpha = std::max(1.2F, this->mci_parameters_.alpha);
    while (true) {
        const auto distance_limit =
            ExpandMCIDistanceLimit(nearest_distance, now_alpha, this->metric_);
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
                if (precise_codes->ComputePairVectors(member, neighbor) > distance_limit) {
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

// Update the MCI companion index after one vector is inserted into HGraph.
void
HGraph::incremental_update_mci_clique(InnerIdType new_inner_id, const void* vector) {
    if (not this->mci_parameters_.enabled or this->mci_cliques_ == nullptr) {
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
