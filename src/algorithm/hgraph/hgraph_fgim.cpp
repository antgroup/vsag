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

#include "hgraph_fgim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>

#include "hgraph.h"
#include "impl/query_computer_pool.h"
#include "index/index_impl.h"
#include "simd/fp32_simd.h"
#include "vsag/constants.h"

namespace vsag {

std::vector<InnerIdType>
HGraphFGIM::ValidateSourcesAndGetOffsets(const Vector<const HGraph*>& source_graphs, uint64_t k) {
    CHECK_ARGUMENT(source_graphs.size() >= 2, "FGIM requires at least two source HGraphs");
    const bool valid_k = k > 0 && k <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
    CHECK_ARGUMENT(valid_k, "FGIM k must be positive and representable as int64_t");

    std::vector<InnerIdType> offsets{0};
    uint64_t total = 0;
    int64_t dimension = 0;
    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        const auto* source = source_graphs[i];
        CHECK_ARGUMENT(source != nullptr, "FGIM source HGraph must not be null");
        CHECK_ARGUMENT(std::find(source_graphs.begin(), source_graphs.begin() + i, source) ==
                           source_graphs.begin() + i,
                       "FGIM source HGraphs must be distinct");
        const auto count = source->GetNumElements();
        CHECK_ARGUMENT(count > 0, "FGIM source HGraphs must be non-empty");
        CHECK_ARGUMENT(source->GetNumberRemoved() == 0, "FGIM does not support deleted nodes");
        CHECK_ARGUMENT(source->data_type_ == DataTypes::DATA_TYPE_FLOAT,
                       "FGIM requires float32 vectors");
        CHECK_ARGUMENT(source->metric_ == MetricType::METRIC_TYPE_L2SQR, "FGIM requires L2");
        const bool valid_dimension = source->dim_ > 0 && (i == 0 || source->dim_ == dimension);
        CHECK_ARGUMENT(valid_dimension, "FGIM requires matching positive dimensions");
        dimension = source->dim_;
        CHECK_ARGUMENT(!source->support_duplicate_, "FGIM does not support duplicate grouping");
        CHECK_ARGUMENT(!source->deduplicate_storage_, "FGIM does not support deduplicate storage");
        CHECK_ARGUMENT(!source->use_conjugate_graph_,
                       "FGIM internal search does not support conjugate graph enhancement");
        CHECK_ARGUMENT(
            source->basic_flatten_codes_ != nullptr &&
                source->basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32,
            "FGIM requires fp32 base storage");
        // GetVectorByInnerId and search may select precise or raw storage instead of base.
        const bool valid_precise_storage =
            !source->has_precise_reorder() ||
            (source->high_precise_codes_ != nullptr &&
             source->high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32);
        CHECK_ARGUMENT(valid_precise_storage,
                       "FGIM requires fp32 precise storage when reorder is enabled");
        const bool valid_raw_storage =
            !source->create_new_raw_vector_ ||
            (source->raw_vector_ != nullptr &&
             source->raw_vector_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32);
        CHECK_ARGUMENT(valid_raw_storage, "FGIM requires fp32 raw vector storage");
        CHECK_ARGUMENT(
            static_cast<uint64_t>(count) <= std::numeric_limits<InnerIdType>::max() - total,
            "FGIM total node count exceeds the merged internal ID capacity");
        const bool fully_built = source->bottom_graph_ != nullptr &&
                                 source->label_table_ != nullptr &&
                                 source->bottom_graph_->TotalCount() == count &&
                                 source->basic_flatten_codes_->TotalCount() == count &&
                                 source->label_table_->GetTotalCount() == count;
        CHECK_ARGUMENT(fully_built, "FGIM requires dense, fully built source HGraphs");
        total += static_cast<uint64_t>(count);
        offsets.push_back(static_cast<InnerIdType>(total));
    }
    return offsets;
}

void
HGraphFGIM::AppendOriginalCandidates(const HGraph& source,
                                     InnerIdType local_u,
                                     InnerIdType source_offset,
                                     FGIMNeighborList& candidates) {
    Vector<InnerIdType> neighbors(source.allocator_);
    source.bottom_graph_->GetNeighbors(local_u, neighbors);
    for (InnerIdType local_v : neighbors) {
        const bool valid_neighbor = local_v < source.GetNumElements() && local_v != local_u;
        CHECK_ARGUMENT(valid_neighbor, "FGIM encountered an invalid original neighbor");
        // Both distance operands remain in the source-local internal ID space.
        // Validated fp32/L2 storage layers contain the same vectors, so base distances
        // are comparable to CrossQuery distances; bitwise equality is not assumed.
        const float distance = source.basic_flatten_codes_->ComputePairVectors(local_u, local_v);
        candidates.push_back({source_offset + local_v, distance});
    }
}

FGIMNeighborList
HGraphFGIM::CrossQuery(const HGraph& target,
                       const float* query,
                       int64_t l,
                       SearchStatistics* stats) {
    const bool valid_query = query != nullptr && l > 0 && target.GetNumElements() > 0;
    CHECK_ARGUMENT(valid_query,
                   "FGIM CrossQuery requires a query, positive L and a non-empty target");
    CHECK_ARGUMENT(!target.use_conjugate_graph_,
                   "FGIM internal search does not support conjugate graph enhancement");
    QueryComputerPool computers(query, stats);
    QueryContext ctx{.alloc = target.allocator_, .stats = stats};
    ctx.computer_pool = &computers;
    ctx.track_distance_evaluations = stats != nullptr;

    InnerSearchParam param;
    param.ep = target.entry_point_id_;
    param.topk = 1;
    param.ef = 1;
    // Reuse HGraph's routing and bottom search; no external labels or Dataset packing.
    ctx.distance_phase = DistanceEvaluationPhase::ROUTING;
    for (int64_t level = static_cast<int64_t>(target.route_graphs_.size()) - 1; level >= 0;
         --level) {
        const auto route = target.search_one_graph(query,
                                                   target.route_graphs_[level],
                                                   target.basic_flatten_codes_,
                                                   param,
                                                   VisitedListPtr{},
                                                   &ctx);
        if (!route->Empty()) {
            param.ep = route->Top().second;
        }
    }
    const int64_t topk = std::min(l, target.GetNumElements());
    // A source with fewer than L nodes cannot fill an L-sized search pool.
    param.ef = static_cast<uint64_t>(topk);
    param.topk = topk;
    param.rerank_topk = topk;
    param.consider_duplicate = target.support_duplicate_;
    // Validated sources have no deletions; no user filter is applied.
    param.is_inner_id_allowed = target.create_search_filter(nullptr, false);
    ctx.distance_phase = DistanceEvaluationPhase::APPROXIMATE;
    auto result = target.search_one_graph(
        query, target.bottom_graph_, target.basic_flatten_codes_, param, VisitedListPtr{}, &ctx);
    if (target.use_reorder_) {
        target.reorder(query, target.get_reorder_codes(), result, topk, nullptr, ctx);
    }
    while (result->Size() > static_cast<uint64_t>(topk)) {
        result->Pop();
    }
    FGIMNeighborList neighbors;
    neighbors.reserve(result->Size());
    while (!result->Empty()) {
        const auto record = result->Top();
        result->Pop();
        // Match unthresholded HGraph search: omit NaN, but preserve infinity so
        // BuildInitialKnnGraph rejects non-finite candidates instead of dropping them.
        if (!std::isnan(record.first)) {
            neighbors.push_back({record.second, record.first});
        }
    }
    // Canonicalize ties without changing the candidate set chosen by HGraph.
    std::sort(neighbors.begin(), neighbors.end(), [](const auto& a, const auto& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    return neighbors;
}

FGIMKnnGraph
HGraphFGIM::BuildInitialKnnGraph(const Vector<const HGraph*>& source_graphs,
                                 uint64_t k,
                                 uint64_t* distance_evaluations) {
    std::unique_ptr<SearchStatistics> search_statistics;
    if (distance_evaluations != nullptr) {
        search_statistics = std::make_unique<SearchStatistics>();
    }
    uint64_t original_distances = 0;
    const auto offsets = ValidateSourcesAndGetOffsets(source_graphs, k);
    const uint64_t other_count = source_graphs.size() - 1;
    // ceil(k / (m - 1)), without overflowing k + m - 2.
    const auto query_count = static_cast<int64_t>(k / other_count + (k % other_count != 0 ? 1 : 0));
    FGIMKnnGraph graph(offsets.back());
    std::vector<float> vector(source_graphs.front()->dim_);

    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        const auto& source = *source_graphs[i];
        const auto source_count = static_cast<InnerIdType>(source.GetNumElements());
        for (InnerIdType local_u = 0; local_u < source_count; ++local_u) {
            auto& candidates = graph[offsets[i] + local_u];
            AppendOriginalCandidates(source, local_u, offsets[i], candidates);
            if (distance_evaluations != nullptr) {
                original_distances += candidates.size();
            }
            source.GetVectorByInnerId(local_u, vector.data());
            for (uint64_t j = 0; j < source_graphs.size(); ++j) {
                if (j != i) {
                    const auto cross = CrossQuery(
                        *source_graphs[j], vector.data(), query_count, search_statistics.get());
                    for (const auto& neighbor : cross) {
                        // CrossQuery IDs are target-local; apply the target prefix exactly once.
                        candidates.push_back({offsets[j] + neighbor.id, neighbor.distance});
                    }
                }
            }
            for (const auto& candidate : candidates) {
                CHECK_ARGUMENT(std::isfinite(candidate.distance), "FGIM requires finite distances");
            }
            std::sort(candidates.begin(),
                      candidates.end(),
                      [](const FGIMNeighbor& a, const FGIMNeighbor& b) {
                          return a.distance < b.distance ||
                                 (a.distance == b.distance && a.id < b.id);
                      });
            if (candidates.size() > k) {
                candidates.resize(k);
                FGIMNeighborList(candidates.begin(), candidates.end()).swap(candidates);
            }
        }
    }
    if (distance_evaluations != nullptr) {
        *distance_evaluations = original_distances + search_statistics->distance_evaluations.load();
    }
    return graph;
}

namespace {

// Persistent rows are always sorted; flags move with their corresponding edges.
bool
fgim_less(const FGIMNeighbor& lhs, const FGIMNeighbor& rhs) {
    return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

void
fgim_insert(FGIMNeighborList& row, std::vector<uint8_t>& flags, FGIMNeighbor edge) {
    const auto position = std::lower_bound(row.begin(), row.end(), edge, fgim_less) - row.begin();
    row.insert(row.begin() + position, edge);
    flags.insert(flags.begin() + position, 1);
}

bool
fgim_update(FGIMNeighborList& row, std::vector<uint8_t>& flags, FGIMNeighbor edge, uint64_t k) {
    if (row.size() == k && edge.distance >= row.back().distance) {
        return false;
    }
    for (const auto& current : row) {
        if (current.id == edge.id) {
            return false;
        }
    }
    if (row.size() == k) {
        row.pop_back();
        flags.pop_back();
    }
    fgim_insert(row, flags, edge);
    return true;
}

void
fgim_unique(std::vector<InnerIdType>& ids) {
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

// Algorithm 4. Replacement preserves degree and never removes a sole incoming edge.
uint64_t
fgim_repair_indegree(FGIMKnnGraph& graph, std::vector<std::vector<uint8_t>>& flags) {
    std::vector<uint64_t> indegree(graph.size(), 0);
    std::vector<std::vector<InnerIdType>> protected_edges(graph.size());
    for (const auto& row : graph) {
        for (const auto& edge : row) {
            ++indegree[edge.id];
        }
    }
    uint64_t repairs = 0;
    for (InnerIdType u = 0; u < graph.size(); ++u) {
        if (indegree[u] != 0) {
            continue;
        }
        // The source row cannot change: self loops are forbidden.
        for (const auto& edge : graph[u]) {
            auto& target = graph[edge.id];
            auto& target_flags = flags[edge.id];
            auto& protected_ids = protected_edges[edge.id];
            for (uint64_t i = target.size(); i > 0; --i) {
                const auto removed = target[i - 1].id;
                if (indegree[removed] <= 1 ||
                    std::find(protected_ids.begin(), protected_ids.end(), removed) !=
                        protected_ids.end()) {
                    continue;
                }
                --indegree[removed];
                ++indegree[u];
                target.erase(target.begin() + static_cast<int64_t>(i - 1));
                target_flags.erase(target_flags.begin() + static_cast<int64_t>(i - 1));
                fgim_insert(target, target_flags, {u, edge.distance});
                protected_ids.push_back(u);
                ++repairs;
                break;
            }
            if (indegree[u] != 0) {
                break;
            }
        }
    }
    return repairs;
}

}  // namespace

FGIMRefinementStatistics
HGraphFGIM::RefineKnnGraph(const Vector<const HGraph*>& source_graphs,
                           FGIMKnnGraph& graph,
                           uint64_t k,
                           uint64_t max_iterations) {
    const auto offsets = ValidateSourcesAndGetOffsets(source_graphs, k);
    ValidateKnnGraph(graph, k, offsets.back());
    FGIMRefinementStatistics stats;
    if (max_iterations == 0) {
        return stats;
    }

    const uint64_t dimension = source_graphs.front()->dim_;
    CHECK_ARGUMENT(graph.size() <= std::numeric_limits<uint64_t>::max() / dimension,
                   "FGIM refinement vector size overflow");
    // Decode once, not for every candidate pair. Reuse the FP32 storage distance kernel.
    Vector<float> vectors(source_graphs.front()->allocator_);
    vectors.resize(graph.size() * dimension);
    std::vector<uint64_t> source_ids(graph.size());
    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        for (InnerIdType local = 0; local < source_graphs[i]->GetNumElements(); ++local) {
            const uint64_t merged = offsets[i] + local;
            source_graphs[i]->GetVectorByInnerId(local, vectors.data() + merged * dimension);
            source_ids[merged] = i;
        }
    }
    std::vector<std::vector<uint8_t>> flags(graph.size());
    std::vector<std::vector<InnerIdType>> reverse_new(graph.size());
    std::vector<std::vector<InnerIdType>> reverse_old(graph.size());
    for (uint64_t u = 0; u < graph.size(); ++u) {
        flags[u].assign(graph[u].size(), 1);
    }
    std::vector<uint8_t> visited(graph.size());
    std::vector<InnerIdType> stack;
    std::vector<InnerIdType> new_ids;
    std::vector<InnerIdType> old_ids;
    for (uint64_t iteration = 0; iteration < max_iterations; ++iteration) {
        std::fill(visited.begin(), visited.end(), 0);
        const auto updates_before = stats.neighbor_updates;
        for (InnerIdType root = 0; root < graph.size(); ++root) {
            if (visited[root] != 0) {
                continue;
            }
            visited[root] = 1;
            stack.push_back(root);
            while (!stack.empty()) {
                const auto u = stack.back();
                stack.pop_back();
                new_ids.clear();
                old_ids.clear();
                for (uint64_t i = 0; i < graph[u].size(); ++i) {
                    const auto v = graph[u][i].id;
                    if (flags[u][i] != 0) {
                        new_ids.push_back(v);
                        reverse_new[v].push_back(u);
                        flags[u][i] = 0;
                    } else {
                        old_ids.push_back(v);
                        reverse_old[v].push_back(u);
                    }
                    if (visited[v] == 0) {
                        visited[v] = 1;
                        stack.push_back(v);
                    }
                }
                new_ids.insert(new_ids.end(), reverse_new[u].begin(), reverse_new[u].end());
                old_ids.insert(old_ids.end(), reverse_old[u].begin(), reverse_old[u].end());
                reverse_new[u].clear();
                reverse_old[u].clear();
                fgim_unique(new_ids);
                fgim_unique(old_ids);
                // A vertex seen as both new and old is processed only as new.
                old_ids.erase(std::remove_if(old_ids.begin(),
                                             old_ids.end(),
                                             [&](InnerIdType v) {
                                                 return std::binary_search(
                                                     new_ids.begin(), new_ids.end(), v);
                                             }),
                              old_ids.end());
                const auto compare = [&](InnerIdType a, InnerIdType b, bool both_new) {
                    if (a == b) {
                        return;
                    }
                    if (source_ids[a] == source_ids[b]) {
                        ++stats.same_source_pairs_skipped;
                        return;
                    }
                    if (both_new) {
                        ++stats.new_new_pairs;
                    } else {
                        ++stats.new_old_pairs;
                    }
                    const float distance =
                        FP32ComputeL2Sqr(vectors.data() + uint64_t{a} * dimension,
                                         vectors.data() + uint64_t{b} * dimension,
                                         dimension);
                    ++stats.distance_evaluations;
                    CHECK_ARGUMENT(std::isfinite(distance),
                                   "FGIM refinement encountered a non-finite distance");
                    if (fgim_update(graph[a], flags[a], {b, distance}, k)) {
                        ++stats.neighbor_updates;
                    }
                    if (fgim_update(graph[b], flags[b], {a, distance}, k)) {
                        ++stats.neighbor_updates;
                    }
                };
                for (uint64_t i = 0; i < new_ids.size(); ++i) {
                    for (uint64_t j = i + 1; j < new_ids.size(); ++j) {
                        compare(new_ids[i], new_ids[j], true);
                    }
                    for (const auto old_id : old_ids) {
                        compare(new_ids[i], old_id, false);
                    }
                }
            }
        }
        const auto repairs = fgim_repair_indegree(graph, flags);
        stats.indegree_repairs += repairs;
        ++stats.iterations;
        const bool pending_new = std::any_of(
            reverse_new.begin(), reverse_new.end(), [](const auto& row) { return !row.empty(); });
        // Pending reverse-new edges must get their next visit before convergence.
        if (stats.neighbor_updates == updates_before && repairs == 0 && !pending_new) {
            break;
        }
    }
    return stats;
}

void
HGraphFGIM::ValidateKnnGraph(const FGIMKnnGraph& graph, uint64_t k, uint64_t expected_count) {
    CHECK_ARGUMENT(k > 0, "FGIM degree must be positive");
    CHECK_ARGUMENT(graph.size() == expected_count, "FGIM node count mismatch");
    std::vector<InnerIdType> ids;
    for (InnerIdType u = 0; u < graph.size(); ++u) {
        const auto& row = graph[u];
        CHECK_ARGUMENT(row.size() <= k, "FGIM degree exceeds k");
        CHECK_ARGUMENT(std::is_sorted(row.begin(), row.end(), fgim_less),
                       "FGIM requires sorted neighbor lists");
        ids.clear();
        for (const auto& edge : row) {
            const bool valid_condition_1 = edge.id < graph.size() && edge.id != u &&
                                           std::isfinite(edge.distance) && edge.distance >= 0;
            CHECK_ARGUMENT(valid_condition_1, "FGIM encountered an invalid neighbor");
            ids.push_back(edge.id);
        }
        std::sort(ids.begin(), ids.end());
        CHECK_ARGUMENT(std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
                       "FGIM encountered duplicate neighbors");
    }
}

FGIMKnnGraph
HGraphFGIM::KnnGraphToPG(const Vector<const HGraph*>& source_graphs,
                         const FGIMKnnGraph& graph,
                         uint64_t k,
                         uint64_t* distance_evaluations) {
    const auto offsets = ValidateSourcesAndGetOffsets(source_graphs, k);
    ValidateKnnGraph(graph, k, offsets.back());
    uint64_t count = 0;
    const uint64_t dimension = source_graphs.front()->dim_;
    CHECK_ARGUMENT(graph.size() <= std::numeric_limits<uint64_t>::max() / dimension,
                   "FGIM reconstruction vector size overflow");
    Vector<float> vectors(source_graphs.front()->allocator_);
    vectors.resize(graph.size() * dimension);
    for (uint64_t i = 0; i < source_graphs.size(); ++i) {
        for (InnerIdType u = 0; u < source_graphs[i]->GetNumElements(); ++u) {
            source_graphs[i]->GetVectorByInnerId(
                u, vectors.data() + (uint64_t{offsets[i]} + u) * dimension);
        }
    }
    std::vector<uint64_t> indegree(graph.size(), 0);
    for (const auto& row : graph) {
        for (const auto& edge : row) {
            ++indegree[edge.id];
        }
    }
    FGIMKnnGraph selected(graph.size());
    for (InnerIdType u = 0; u < graph.size(); ++u) {
        auto& retained = selected[u];
        for (const auto& edge : graph[u]) {
            bool keep = true;
            if (indegree[edge.id] > 1) {
                for (const auto& previous : retained) {
                    const float distance =
                        FP32ComputeL2Sqr(vectors.data() + uint64_t{edge.id} * dimension,
                                         vectors.data() + uint64_t{previous.id} * dimension,
                                         dimension);
                    ++count;
                    CHECK_ARGUMENT(std::isfinite(distance),
                                   "FGIM reconstruction encountered a non-finite distance");
                    if (!(edge.distance < distance)) {
                        keep = false;
                        break;
                    }
                }
            }
            if (keep) {
                retained.push_back(edge);
            } else {
                // Update the recorder so later removals cannot delete the last incoming edge.
                --indegree[edge.id];
            }
        }
    }

    // Reverse edges come from a fixed pruned snapshot, not from partially augmented rows.
    auto result = selected;
    for (InnerIdType u = 0; u < selected.size(); ++u) {
        for (const auto& edge : selected[u]) {
            result[edge.id].push_back({u, edge.distance});
        }
    }
    for (auto& row : result) {
        // Deduplicate by ID first; retain the smaller stored distance if roundoff differs.
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.id < rhs.id || (lhs.id == rhs.id && lhs.distance < rhs.distance);
        });
        row.erase(std::unique(row.begin(),
                              row.end(),
                              [](const auto& lhs, const auto& rhs) { return lhs.id == rhs.id; }),
                  row.end());
        std::sort(row.begin(), row.end(), fgim_less);
        if (row.size() > k) {
            row.resize(k);
        }
    }
    if (distance_evaluations != nullptr) {
        *distance_evaluations = count;
    }
    return result;
}

void
HGraphFGIM::WriteBottomGraph(HGraph& destination, const FGIMKnnGraph& graph, uint64_t k) {
    ValidateKnnGraph(graph, k, destination.GetNumElements());
    const bool valid_condition_2 = destination.bottom_graph_ != nullptr &&
                                   destination.bottom_graph_->MaximumDegree() >= k &&
                                   destination.bottom_graph_->MaxCapacity() >= graph.size();
    CHECK_ARGUMENT(valid_condition_2, "FGIM destination bottom graph capacity is insufficient");
    const bool valid_condition_3 = destination.basic_flatten_codes_ != nullptr &&
                                   destination.basic_flatten_codes_->TotalCount() == graph.size() &&
                                   destination.label_table_->GetTotalCount() == graph.size() &&
                                   destination.GetNumberRemoved() == 0;
    CHECK_ARGUMENT(valid_condition_3, "FGIM destination storage is not prepared");
    Vector<InnerIdType> neighbors(destination.allocator_);
    neighbors.reserve(k);
    for (InnerIdType u = 0; u < graph.size(); ++u) {
        neighbors.clear();
        for (const auto& edge : graph[u]) {
            neighbors.push_back(edge.id);
        }
        destination.bottom_graph_->InsertNeighborsById(u, neighbors);
    }
    destination.bottom_graph_->SetTotalCount(graph.size());
}

void
HGraphFGIM::RebuildUpperLayers(HGraph& destination) {
    const bool valid_condition_4 =
        destination.GetNumElements() > 0 && destination.GetNumberRemoved() == 0 &&
        !destination.deduplicate_storage_ && destination.data_type_ == DataTypes::DATA_TYPE_FLOAT &&
        destination.metric_ == MetricType::METRIC_TYPE_L2SQR &&
        destination.basic_flatten_codes_ != nullptr &&
        destination.basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32;
    CHECK_ARGUMENT(valid_condition_4,
                   "FGIM upper rebuild requires a non-empty float32/L2/fp32 destination");
    const bool valid_condition_5 =
        destination.bottom_graph_->TotalCount() == destination.GetNumElements() &&
        destination.basic_flatten_codes_->TotalCount() == destination.GetNumElements();
    CHECK_ARGUMENT(valid_condition_5,
                   "FGIM upper rebuild requires prepared bottom graph and vectors");
    CHECK_ARGUMENT(
        destination.ef_construct_ <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
        "FGIM upper construction width exceeds the search limit");
    destination.route_graphs_.clear();
    destination.entry_point_id_ = 0;
    destination.level_generator_.seed(2021);
    std::vector<float> vector(destination.dim_);
    const auto codes = destination.has_precise_reorder() && !destination.build_by_base_
                           ? destination.high_precise_codes_
                           : destination.basic_flatten_codes_;
    for (InnerIdType u = 0; u < destination.GetNumElements(); ++u) {
        // VSAG uses -1 for bottom-only nodes and 0 for the first route graph.
        const int level = destination.get_random_level() - 1;
        if (level < 0) {
            continue;
        }
        destination.GetVectorByInnerId(u, vector.data());
        const int previous_height = static_cast<int>(destination.route_graphs_.size());
        InnerSearchParam param;
        param.ep = destination.entry_point_id_;
        param.ef = 1;
        param.topk = 1;
        const auto computer = destination.make_build_computer(vector.data(), u);
        for (int layer = previous_height - 1; layer > level; --layer) {
            const auto result = destination.search_graph_for_build(
                vector.data(), destination.route_graphs_[layer], codes, param, computer);
            if (!result->Empty()) {
                param.ep = result->Top().second;
            }
        }
        destination.ensure_route_graphs_for_level(level);
        param.ef = destination.ef_construct_;
        param.topk = static_cast<int64_t>(destination.ef_construct_);
        // Reuse native layer search, heuristic selection and bidirectional connection.
        destination.publish_unique_to_route_graphs(vector.data(), level, u, param, codes);
        if (level >= previous_height) {
            destination.entry_point_id_ = u;
        }
    }
}

void
HGraphFGIM::Merge(HGraph& destination,
                  const std::vector<MergeUnit>& merge_units,
                  uint64_t max_iterations,
                  FGIMMergeStatistics* statistics) {
    if (statistics != nullptr) {
        *statistics = {};
    }
    if (merge_units.empty()) {
        return;
    }
    CHECK_ARGUMENT(!destination.immutable_.load(), "FGIM cannot merge into an immutable HGraph");
    std::unique_lock<std::shared_mutex> add_lock(destination.add_mutex_);
    std::unique_lock<std::shared_mutex> global_lock(destination.global_mutex_);
    Vector<const HGraph*> sources(destination.allocator_);
    if (destination.GetNumElements() != 0) {
        sources.push_back(&destination);
    }
    for (const auto& unit : merge_units) {
        const auto wrapper = std::dynamic_pointer_cast<IndexImpl<HGraph>>(unit.index);
        CHECK_ARGUMENT(wrapper != nullptr, "FGIM MergeUnit must contain an HGraph");
        const auto source = std::dynamic_pointer_cast<HGraph>(wrapper->GetInnerIndex());
        const bool valid_condition_6 = source != nullptr && source.get() != &destination;
        CHECK_ARGUMENT(valid_condition_6, "FGIM cannot merge a destination into itself");
        sources.push_back(source.get());
    }
    const auto k = destination.bottom_graph_->MaximumDegree();
    const auto offsets = ValidateSourcesAndGetOffsets(sources, k);
    const auto validate_metadata = [&](const HGraph& index) {
        const bool valid_condition_7 =
            index.data_type_ == DataTypes::DATA_TYPE_FLOAT &&
            index.metric_ == MetricType::METRIC_TYPE_L2SQR && index.dim_ == sources.front()->dim_ &&
            index.basic_flatten_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32 &&
            (!index.has_precise_reorder() ||
             index.high_precise_codes_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) &&
            (!index.create_new_raw_vector_ ||
             index.raw_vector_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32);
        CHECK_ARGUMENT(valid_condition_7,
                       "FGIM destination and sources require homogeneous float32/L2/fp32 storage");
        const bool valid_condition_8 = !index.deduplicate_storage_ && !index.support_duplicate_ &&
                                       index.GetNumberRemoved() == 0 &&
                                       !index.use_conjugate_graph_ &&
                                       !index.mci_parameters_.enabled && !index.has_attribute_ &&
                                       index.extra_info_size_ == 0 && !index.persist_source_id_;
        CHECK_ARGUMENT(valid_condition_8,
                       "FGIM does not support deletion, duplicate tracking or auxiliary metadata");
    };
    validate_metadata(destination);
    for (const auto* source : sources) {
        validate_metadata(*source);
    }

    // Evaluate callbacks once, before changing storage; only public integration handles labels.
    std::vector<LabelType> labels;
    labels.reserve(offsets.back());
    const uint64_t original_count = destination.GetNumElements();
    for (InnerIdType u = 0; u < original_count; ++u) {
        labels.push_back(destination.label_table_->GetLabelById(u));
    }
    for (const auto& unit : merge_units) {
        const auto wrapper = std::dynamic_pointer_cast<IndexImpl<HGraph>>(unit.index);
        const auto source = std::dynamic_pointer_cast<HGraph>(wrapper->GetInnerIndex());
        for (InnerIdType u = 0; u < source->GetNumElements(); ++u) {
            const auto original_label = source->label_table_->GetLabelById(u);
            const auto mapped = unit.id_map_func ? unit.id_map_func(original_label)
                                                 : std::make_tuple(true, original_label);
            CHECK_ARGUMENT(std::get<0>(mapped),
                           "FGIM does not support filtering nodes through id_map_func");
            labels.push_back(std::get<1>(mapped));
        }
    }
    auto sorted_labels = labels;
    std::sort(sorted_labels.begin(), sorted_labels.end());
    CHECK_ARGUMENT(
        std::adjacent_find(sorted_labels.begin(), sorted_labels.end()) == sorted_labels.end(),
        "FGIM merged external labels must be unique");
    sorted_labels.clear();
    sorted_labels.shrink_to_fit();

    using Clock = std::chrono::steady_clock;
    auto checkpoint = statistics != nullptr ? Clock::now() : Clock::time_point{};
    const auto lap = [&]() {
        if (statistics == nullptr) {
            return 0.0;
        }
        const auto now = Clock::now();
        const auto seconds = std::chrono::duration<double>(now - checkpoint).count();
        checkpoint = now;
        return seconds;
    };
    auto graph = BuildInitialKnnGraph(
        sources, k, statistics != nullptr ? &statistics->initial_distance_evaluations : nullptr);
    if (statistics != nullptr) {
        statistics->initial_seconds = lap();
    }
    const auto refinement = RefineKnnGraph(sources, graph, k, max_iterations);
    if (statistics != nullptr) {
        statistics->refinement = refinement;
        statistics->refinement_seconds = lap();
    }
    graph = KnnGraphToPG(
        sources,
        graph,
        k,
        statistics != nullptr ? &statistics->reconstruction_distance_evaluations : nullptr);
    if (statistics != nullptr) {
        statistics->reconstruction_seconds = lap();
    }

    // All source-dependent graph work finishes before mutating a non-empty destination.
    // resize acquires global_mutex_ itself; add_mutex_ still excludes concurrent insertion.
    global_lock.unlock();
    destination.resize(offsets.back());
    global_lock.lock();
    std::vector<float> vector(destination.dim_);
    for (uint64_t i = 0; i < sources.size(); ++i) {
        const auto& source = *sources[i];
        if (&source == &destination) {
            continue;
        }
        for (InnerIdType local = 0; local < source.GetNumElements(); ++local) {
            const auto merged = offsets[i] + local;
            source.GetVectorByInnerId(local, vector.data());
            destination.insert_persistent_codes_unlocked(vector.data(), merged);
            destination.label_table_->Insert(merged, labels[merged]);
        }
    }
    destination.total_count_.store(offsets.back());
    if (statistics != nullptr) {
        statistics->storage_seconds = lap();
    }
    WriteBottomGraph(destination, graph, k);
    if (statistics != nullptr) {
        statistics->bottom_write_seconds = lap();
    }
    RebuildUpperLayers(destination);
    if (statistics != nullptr) {
        statistics->upper_seconds = lap();
    }
}

}  // namespace vsag
