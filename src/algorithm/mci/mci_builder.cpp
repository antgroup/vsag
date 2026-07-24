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

#include "mci_builder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "impl/logger/logger.h"
#include "mci_runner.h"
#include "simd/fp32_simd.h"

namespace vsag {
namespace {

constexpr uint64_t K_V3_SCHEDULE_CHUNK = 64;
constexpr uint64_t K_MCI_MIN_CLIQUE_SIZE = 50;

// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIV3Candidate {
    InnerIdType id{0};
    float distance{0.0F};
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIV3Edge {
    InnerIdType u{0};
    InnerIdType v{0};
    float dis{0.0F};

    bool
    operator<(const MCIV3Edge& other) const {
        return dis < other.dis;
    }
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct MCIV3ThreadTiming {
    double candidate_collect{0.0};
    double query_distance{0.0};
    double pair_distance{0.0};
    double edge_sort{0.0};
    double mce{0.0};
    double choose{0.0};
};

double
// NOLINTNEXTLINE(readability-identifier-naming)
SecondsSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

float
// NOLINTNEXTLINE(readability-identifier-naming)
ExpandDistanceLimit(float nearest_distance, float alpha, MetricType metric) {
    if (metric != MetricType::METRIC_TYPE_IP) {
        return nearest_distance * alpha;
    }
    const auto nearest_similarity = 1.0F - nearest_distance;
    const auto similarity_limit =
        nearest_similarity >= 0.0F ? nearest_similarity / alpha : nearest_similarity * alpha;
    return 1.0F - similarity_limit;
}

const float*
// NOLINTNEXTLINE(readability-identifier-naming)
VectorAt(const float* vectors, uint64_t dim, InnerIdType id) {
    return vectors + static_cast<uint64_t>(id) * dim;
}

const InnerIdType*
// NOLINTNEXTLINE(readability-identifier-naming)
GraphRow(const MCIGraphView& graph, InnerIdType id) {
    return graph.neighbors + static_cast<uint64_t>(id) * graph.row_stride;
}

uint64_t
// NOLINTNEXTLINE(readability-identifier-naming)
GraphRowCount(const MCIGraphView& graph, InnerIdType id) {
    if (graph.counts != nullptr) {
        return graph.counts[id];
    }
    return graph.uniform_count;
}

MCIV3ThreadTiming
// NOLINTNEXTLINE(readability-identifier-naming)
SumTiming(const Vector<MCIV3ThreadTiming>& timings) {
    MCIV3ThreadTiming total;
    for (const auto& item : timings) {
        total.candidate_collect += item.candidate_collect;
        total.query_distance += item.query_distance;
        total.pair_distance += item.pair_distance;
        total.edge_sort += item.edge_sort;
        total.mce += item.mce;
        total.choose += item.choose;
    }
    return total;
}

}  // namespace

Vector<Vector<InnerIdType>>
BuildMCICliques(const float* vectors,
                const MCIGraphView& graph,
                const MCIV3BuildParams& params,
                Allocator* allocator) {
    Vector<Vector<InnerIdType>> result(allocator);
    if (vectors == nullptr or graph.neighbors == nullptr or params.total == 0 or params.dim == 0 or
        graph.row_stride == 0) {
        return result;
    }

    const auto total = params.total;
    const auto dim = params.dim;
    const auto candidate_limit =
        std::min<uint64_t>({params.candidate_limit, graph.row_stride, total - 1});
    const auto clique_threshold =
        std::min<uint64_t>({K_MCI_MIN_CLIQUE_SIZE, candidate_limit + 1, total});
    const auto node_clique_limit = std::max<int>(3, static_cast<int>(total / 100));
    const auto thread_count = std::max<uint64_t>(1, std::min<uint64_t>(params.thread_count, total));
    const auto max_saved_cliques =
        std::min<uint64_t>(candidate_limit, static_cast<uint64_t>(params.max_degree + 2));

    logger::info(
        "mci v3 clique build started, total={}, dim={}, candidate_limit={}, clique_threshold={}, "
        "node_clique_limit={}, max_saved_cliques={}, threads={}",
        total,
        dim,
        candidate_limit,
        clique_threshold,
        node_clique_limit,
        max_saved_cliques,
        thread_count);

    Vector<float> l2_norms(total, allocator);
    double max_norm_error = 0.0;
    {
        std::atomic<uint64_t> next_id{0};
        std::vector<std::future<void>> workers;
        workers.reserve(thread_count);
        std::mutex norm_mutex;
        for (uint64_t tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back(std::async(std::launch::async, [&, tid]() {
                double local_max = 0.0;
                while (true) {
                    const auto start = next_id.fetch_add(K_V3_SCHEDULE_CHUNK);
                    if (start >= total) {
                        break;
                    }
                    const auto end = std::min<uint64_t>(total, start + K_V3_SCHEDULE_CHUNK);
                    for (uint64_t id = start; id < end; ++id) {
                        const float norm = FP32ComputeIP(
                            VectorAt(vectors, dim, id), VectorAt(vectors, dim, id), dim);
                        l2_norms[id] = norm;
                        local_max = std::max(local_max, std::abs(static_cast<double>(norm) - 1.0));
                    }
                }
                std::lock_guard<std::mutex> lock(norm_mutex);
                max_norm_error = std::max(max_norm_error, local_max);
            }));
        }
        for (auto& worker : workers) {
            worker.get();
        }
    }
    const bool unit_norm = max_norm_error < 1e-3;
    logger::info("mci v3 float continuous path enabled, unit_norm={}, max_norm_error={:.6f}",
                 unit_norm,
                 max_norm_error);

    auto distance_from_dot = [&](InnerIdType lhs, InnerIdType rhs, float dot) {
        if (params.metric == MetricType::METRIC_TYPE_L2SQR) {
            return FP32ComputeL2Sqr(VectorAt(vectors, dim, lhs), VectorAt(vectors, dim, rhs), dim);
        }
        if (params.metric == MetricType::METRIC_TYPE_IP) {
            return 1.0F - dot;
        }
        const auto norm_product = l2_norms[lhs] * l2_norms[rhs];
        if (norm_product <= 0.0F) {
            return 1.0F;
        }
        return 1.0F - dot / std::sqrt(norm_product);
    };
    auto compute_distance = [&](InnerIdType lhs, InnerIdType rhs) {
        if (params.metric == MetricType::METRIC_TYPE_L2SQR) {
            return FP32ComputeL2Sqr(VectorAt(vectors, dim, lhs), VectorAt(vectors, dim, rhs), dim);
        }
        return distance_from_dot(
            lhs, rhs, FP32ComputeIP(VectorAt(vectors, dim, lhs), VectorAt(vectors, dim, rhs), dim));
    };
    const auto compute_distance_batch = params.metric == MetricType::METRIC_TYPE_L2SQR
                                            ? FP32ComputeL2SqrBatch4
                                            : FP32ComputeIPBatch4;

    std::vector<std::atomic<int>> num_cliques_per_node(total);
    for (auto& count : num_cliques_per_node) {
        count.store(0, std::memory_order_relaxed);
    }
    Vector<Vector<Vector<InnerIdType>>> selected_by_thread(allocator);
    selected_by_thread.reserve(thread_count);
    for (uint64_t tid = 0; tid < thread_count; ++tid) {
        selected_by_thread.emplace_back(allocator);
    }
    Vector<MCIV3ThreadTiming> total_timings(thread_count, allocator);
    std::atomic<uint64_t> total_clique_count{0};
    std::atomic<uint64_t> clique_containing_seed_count{0};

    float now_alpha = params.alpha;
    uint64_t previous_uncovered = total;
    uint64_t uncovered = total;
    uint64_t round = 0;
    const auto build_start = std::chrono::steady_clock::now();

    do {
        ++round;
        logger::info("mci v3 round started, round={}, alpha={:.4f}, uncovered={}",
                     round,
                     now_alpha,
                     uncovered);

        std::atomic<uint64_t> next_seed{0};
        uint64_t next_progress = std::max<uint64_t>(1, total / 10);
        const uint64_t progress_step = std::max<uint64_t>(1, total / 10);
        std::mutex progress_mutex;
        std::atomic<uint64_t> sum_candidate_size{0};
        std::atomic<uint64_t> candidate_count{0};
        std::atomic<uint64_t> sum_edge_size{0};
        std::atomic<uint64_t> sum_clique_count{0};
        Vector<MCIV3ThreadTiming> round_timings(thread_count, allocator);
        std::vector<std::future<void>> workers;
        workers.reserve(thread_count);

        for (uint64_t tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back(std::async(std::launch::async, [&, tid]() {
                std::vector<MCIV3Edge> edges;
                edges.reserve(candidate_limit * (candidate_limit + 1) / 2);
                Vector<MCIV3Candidate> candidates(candidate_limit, allocator);
                Vector<const float*> candidate_vectors(candidate_limit, allocator);
                std::vector<std::vector<InnerIdType>> local_max_cliques(max_saved_cliques);
                for (auto& clique : local_max_cliques) {
                    clique.reserve(candidate_limit + 1);
                }
                mci::ccrmce_runner<MCIV3Edge, InnerIdType> mce_runner;
                mce_runner.reserve(static_cast<uint32_t>(candidate_limit + 1));
                auto append_selected_clique = [&](const auto& clique) {
                    selected_by_thread[tid].emplace_back(allocator);
                    selected_by_thread[tid].back().assign(clique.begin(), clique.end());
                    if (selected_by_thread[tid].back().size() > params.clique_max) {
                        selected_by_thread[tid].back().resize(params.clique_max);
                    }
                };

                auto solve_seed = [&](InnerIdType seed) {
                    const auto* query = VectorAt(vectors, dim, seed);
                    edges.clear();
                    uint64_t candidate_size = 0;

                    auto stage_start = std::chrono::steady_clock::now();
                    const auto* row = GraphRow(graph, seed);
                    const auto row_count =
                        std::min<uint64_t>(GraphRowCount(graph, seed), candidate_limit);
                    for (uint64_t rank = 0; rank < row_count; ++rank) {
                        const auto neighbor = row[rank];
                        if (neighbor >= total or neighbor == seed or
                            num_cliques_per_node[neighbor].load(std::memory_order_relaxed) >=
                                static_cast<int>(node_clique_limit)) {
                            continue;
                        }
                        bool duplicate = false;
                        for (uint64_t i = 0; i < candidate_size; ++i) {
                            if (candidates[i].id == neighbor) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) {
                            continue;
                        }
                        candidates[candidate_size].id = neighbor;
                        candidate_vectors[candidate_size] = VectorAt(vectors, dim, neighbor);
                        ++candidate_size;
                    }
                    round_timings[tid].candidate_collect += SecondsSince(stage_start);
                    sum_candidate_size.fetch_add(candidate_size, std::memory_order_relaxed);
                    candidate_count.fetch_add(1, std::memory_order_relaxed);

                    if (candidate_size + 1 < clique_threshold) {
                        if (now_alpha > 100.0F) {
                            Vector<InnerIdType> fallback(allocator);
                            fallback.reserve(candidate_size + 1);
                            fallback.push_back(seed);
                            for (uint64_t i = 0; i < candidate_size; ++i) {
                                const auto id = candidates[i].id;
                                if (num_cliques_per_node[id].load(std::memory_order_relaxed) > 0) {
                                    fallback.push_back(id);
                                }
                            }
                            num_cliques_per_node[seed].fetch_add(1, std::memory_order_relaxed);
                            total_clique_count.fetch_add(1, std::memory_order_relaxed);
                            append_selected_clique(fallback);
                            clique_containing_seed_count.fetch_add(1, std::memory_order_relaxed);
                        }
                        return;
                    }

                    stage_start = std::chrono::steady_clock::now();
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        const auto candidate_id = candidates[i].id;
                        candidates[i].distance = compute_distance(candidate_id, seed);
                    }
                    std::sort(candidates.begin(),
                              candidates.begin() + static_cast<int64_t>(candidate_size),
                              [](const MCIV3Candidate& lhs, const MCIV3Candidate& rhs) {
                                  return lhs.distance < rhs.distance;
                              });
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        candidate_vectors[i] = VectorAt(vectors, dim, candidates[i].id);
                    }
                    round_timings[tid].query_distance += SecondsSince(stage_start);

                    const float distance_limit =
                        ExpandDistanceLimit(candidates[0].distance, now_alpha, params.metric);
                    stage_start = std::chrono::steady_clock::now();
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        if (candidates[i].distance <= distance_limit) {
                            edges.push_back(
                                MCIV3Edge{seed, candidates[i].id, candidates[i].distance});
                        }
                    }
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        const auto lhs_id = candidates[i].id;
                        const auto* lhs_vector = candidate_vectors[i];
                        uint64_t j = i + 1;
                        for (; j + 3 < candidate_size; j += 4) {
                            float distances[4]{0.0F, 0.0F, 0.0F, 0.0F};
                            compute_distance_batch(lhs_vector,
                                                   dim,
                                                   candidate_vectors[j],
                                                   candidate_vectors[j + 1],
                                                   candidate_vectors[j + 2],
                                                   candidate_vectors[j + 3],
                                                   distances[0],
                                                   distances[1],
                                                   distances[2],
                                                   distances[3]);
                            for (uint64_t batch = 0; batch < 4; ++batch) {
                                const auto rhs_id = candidates[j + batch].id;
                                const float distance =
                                    params.metric == MetricType::METRIC_TYPE_L2SQR
                                        ? distances[batch]
                                        : distance_from_dot(lhs_id, rhs_id, distances[batch]);
                                if (distance <= distance_limit) {
                                    edges.push_back(MCIV3Edge{lhs_id, rhs_id, distance});
                                }
                            }
                        }
                        for (; j < candidate_size; ++j) {
                            const auto rhs_id = candidates[j].id;
                            const float distance = compute_distance(lhs_id, rhs_id);
                            if (distance <= distance_limit) {
                                edges.push_back(MCIV3Edge{lhs_id, rhs_id, distance});
                            }
                        }
                    }
                    round_timings[tid].pair_distance += SecondsSince(stage_start);
                    sum_edge_size.fetch_add(edges.size(), std::memory_order_relaxed);

                    stage_start = std::chrono::steady_clock::now();
                    std::sort(edges.begin(), edges.end());
                    round_timings[tid].edge_sort += SecondsSince(stage_start);

                    if (edges.size() < clique_threshold * (clique_threshold - 1) / 2 and
                        now_alpha > 100.0F) {
                        Vector<InnerIdType> fallback(allocator);
                        fallback.reserve(candidate_size + 1);
                        fallback.push_back(seed);
                        for (uint64_t i = 0; i < candidate_size; ++i) {
                            const auto id = candidates[i].id;
                            if (num_cliques_per_node[id].load(std::memory_order_relaxed) > 0) {
                                fallback.push_back(id);
                                num_cliques_per_node[id].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        num_cliques_per_node[seed].fetch_add(1, std::memory_order_relaxed);
                        total_clique_count.fetch_add(1, std::memory_order_relaxed);
                        append_selected_clique(fallback);
                        clique_containing_seed_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    stage_start = std::chrono::steady_clock::now();
                    const InnerIdType local_clique_count =
                        mce_runner.run(edges,
                                       local_max_cliques,
                                       static_cast<InnerIdType>(clique_threshold),
                                       num_cliques_per_node,
                                       static_cast<InnerIdType>(max_saved_cliques));
                    round_timings[tid].mce += SecondsSince(stage_start);
                    sum_clique_count.fetch_add(local_clique_count, std::memory_order_relaxed);

                    if (local_clique_count == 0 and now_alpha > 100.0F) {
                        Vector<InnerIdType> fallback(allocator);
                        fallback.reserve(candidate_size + 1);
                        fallback.push_back(seed);
                        for (uint64_t i = 0; i < candidate_size; ++i) {
                            const auto id = candidates[i].id;
                            if (num_cliques_per_node[id].load(std::memory_order_relaxed) > 0) {
                                fallback.push_back(id);
                                num_cliques_per_node[id].fetch_add(1, std::memory_order_relaxed);
                            }
                        }
                        num_cliques_per_node[seed].fetch_add(1, std::memory_order_relaxed);
                        total_clique_count.fetch_add(1, std::memory_order_relaxed);
                        append_selected_clique(fallback);
                        clique_containing_seed_count.fetch_add(1, std::memory_order_relaxed);
                        return;
                    }

                    stage_start = std::chrono::steady_clock::now();
                    uint64_t chosen_count = 0;
                    for (uint64_t i = 0; i < local_clique_count; ++i) {
                        auto& clique = local_max_cliques[i];
                        if (clique.size() > params.clique_max) {
                            const bool contains_seed =
                                std::find(clique.begin(), clique.end(), seed) != clique.end();
                            clique.resize(params.clique_max);
                            if (contains_seed and
                                std::find(clique.begin(), clique.end(), seed) == clique.end()) {
                                clique.back() = seed;
                            }
                        }
                        bool has_uncovered_node = false;
                        bool contains_seed = false;
                        for (auto id : clique) {
                            if (num_cliques_per_node[id].load(std::memory_order_relaxed) < 1) {
                                has_uncovered_node = true;
                            }
                            if (id == seed) {
                                contains_seed = true;
                            }
                        }
                        if (not has_uncovered_node) {
                            continue;
                        }
                        for (auto id : clique) {
                            num_cliques_per_node[id].fetch_add(1, std::memory_order_relaxed);
                        }
                        total_clique_count.fetch_add(1, std::memory_order_relaxed);
                        if (contains_seed) {
                            clique_containing_seed_count.fetch_add(1, std::memory_order_relaxed);
                        }
                        append_selected_clique(clique);
                        ++chosen_count;
                        if (chosen_count > params.max_degree) {
                            break;
                        }
                    }
                    round_timings[tid].choose += SecondsSince(stage_start);
                };

                while (true) {
                    const auto start = next_seed.fetch_add(K_V3_SCHEDULE_CHUNK);
                    if (start >= total) {
                        break;
                    }
                    const auto end = std::min<uint64_t>(total, start + K_V3_SCHEDULE_CHUNK);
                    {
                        std::lock_guard<std::mutex> lock(progress_mutex);
                        while (end >= next_progress and next_progress <= total) {
                            logger::info("mci v3 round progress, round={}, scanned={}/{}",
                                         round,
                                         next_progress,
                                         total);
                            next_progress += progress_step;
                        }
                    }
                    for (uint64_t raw_seed = start; raw_seed < end; ++raw_seed) {
                        const auto seed = static_cast<InnerIdType>(raw_seed);
                        if (num_cliques_per_node[seed].load(std::memory_order_relaxed) > 0) {
                            continue;
                        }
                        solve_seed(seed);
                    }
                }
            }));
        }
        for (auto& worker : workers) {
            worker.get();
        }

        uncovered = 0;
        for (uint64_t id = 0; id < total; ++id) {
            if (num_cliques_per_node[id].load(std::memory_order_relaxed) == 0) {
                ++uncovered;
            }
        }

        for (uint64_t tid = 0; tid < thread_count; ++tid) {
            total_timings[tid].candidate_collect += round_timings[tid].candidate_collect;
            total_timings[tid].query_distance += round_timings[tid].query_distance;
            total_timings[tid].pair_distance += round_timings[tid].pair_distance;
            total_timings[tid].edge_sort += round_timings[tid].edge_sort;
            total_timings[tid].mce += round_timings[tid].mce;
            total_timings[tid].choose += round_timings[tid].choose;
        }
        const auto timing = SumTiming(round_timings);
        const auto scanned_candidates = std::max<uint64_t>(1, candidate_count.load());
        logger::info(
            "mci v3 round finished, round={}, alpha={:.4f}, uncovered={}, total_cliques={}, "
            "avg_candidates={:.2f}, avg_edges={:.2f}, avg_mce_cliques={:.2f}, "
            "candidate_s={:.3f}, query_s={:.3f}, pair_s={:.3f}, sort_s={:.3f}, mce_s={:.3f}, "
            "choose_s={:.3f}",
            round,
            now_alpha,
            uncovered,
            total_clique_count.load(),
            static_cast<double>(sum_candidate_size.load()) /
                static_cast<double>(scanned_candidates),
            static_cast<double>(sum_edge_size.load()) / static_cast<double>(scanned_candidates),
            static_cast<double>(sum_clique_count.load()) / static_cast<double>(scanned_candidates),
            timing.candidate_collect,
            timing.query_distance,
            timing.pair_distance,
            timing.edge_sort,
            timing.mce,
            timing.choose);

        if (uncovered < static_cast<uint64_t>(0.9 * static_cast<double>(previous_uncovered))) {
            now_alpha += params.alpha;
        } else {
            now_alpha *= 2.0F;
        }
        previous_uncovered = uncovered;
        if (now_alpha > 1000.0F) {
            break;
        }
    } while (uncovered > 0);

    const auto timing = SumTiming(total_timings);
    logger::info(
        "mci v3 clique build finished, total_cliques={}, seed_cliques={}, uncovered={}, "
        "wall_s={:.3f}, candidate_s={:.3f}, query_s={:.3f}, pair_s={:.3f}, sort_s={:.3f}, "
        "mce_s={:.3f}, choose_s={:.3f}",
        total_clique_count.load(),
        clique_containing_seed_count.load(),
        uncovered,
        SecondsSince(build_start),
        timing.candidate_collect,
        timing.query_distance,
        timing.pair_distance,
        timing.edge_sort,
        timing.mce,
        timing.choose);

    result.reserve(total_clique_count.load());
    for (auto& thread_cliques : selected_by_thread) {
        for (auto& clique : thread_cliques) {
            if (clique.empty()) {
                continue;
            }
            result.push_back(Vector<InnerIdType>(allocator));
            result.back().assign(clique.begin(), clique.end());
        }
    }
    return result;
}

}  // namespace vsag
