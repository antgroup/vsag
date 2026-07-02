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

#include "mci_v3_builder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "impl/logger/logger.h"
#include "mci_v3_runner.h"

namespace vsag {
namespace {

constexpr uint64_t K_V3_SCHEDULE_CHUNK = 64;

struct MCIV3Candidate {
    InnerIdType id{0};
    float distance{0.0F};
};

struct MCIV3Edge {
    InnerIdType u{0};
    InnerIdType v{0};
    float dis{0.0F};

    bool
    operator<(const MCIV3Edge& other) const {
        return dis < other.dis;
    }
};

struct MCIV3ThreadTiming {
    double candidate_collect{0.0};
    double query_distance{0.0};
    double pair_distance{0.0};
    double edge_sort{0.0};
    double mce{0.0};
    double choose{0.0};
};

double
SecondsSince(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

#if defined(__AVX__)
float
ReduceM256(__m256 value) {
    __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 low = _mm256_castps256_ps128(value);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

float
DotProduct(const float* lhs, const float* rhs, uint64_t dim) {
#if defined(__AVX__)
    __m256 sum = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 x = _mm256_loadu_ps(lhs + i);
        const __m256 y = _mm256_loadu_ps(rhs + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(x, y));
    }
    float result = ReduceM256(sum);
    for (; i < dim; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
#else
    float result = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        result += lhs[i] * rhs[i];
    }
    return result;
#endif
}

void
DotProductBatch4(const float* lhs,
                 const float* rhs0,
                 const float* rhs1,
                 const float* rhs2,
                 const float* rhs3,
                 uint64_t dim,
                 float dots[4]) {
#if defined(__AVX__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();
    uint64_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 x = _mm256_loadu_ps(lhs + i);
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(x, _mm256_loadu_ps(rhs0 + i)));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(x, _mm256_loadu_ps(rhs1 + i)));
        sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(x, _mm256_loadu_ps(rhs2 + i)));
        sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(x, _mm256_loadu_ps(rhs3 + i)));
    }
    dots[0] = ReduceM256(sum0);
    dots[1] = ReduceM256(sum1);
    dots[2] = ReduceM256(sum2);
    dots[3] = ReduceM256(sum3);
    for (; i < dim; ++i) {
        const float x = lhs[i];
        dots[0] += x * rhs0[i];
        dots[1] += x * rhs1[i];
        dots[2] += x * rhs2[i];
        dots[3] += x * rhs3[i];
    }
#else
    dots[0] = DotProduct(lhs, rhs0, dim);
    dots[1] = DotProduct(lhs, rhs1, dim);
    dots[2] = DotProduct(lhs, rhs2, dim);
    dots[3] = DotProduct(lhs, rhs3, dim);
#endif
}

const float*
VectorAt(const float* vectors, uint64_t dim, InnerIdType id) {
    return vectors + static_cast<uint64_t>(id) * dim;
}

const InnerIdType*
GraphRow(const MCIV3GraphView& graph, InnerIdType id) {
    return graph.neighbors + static_cast<uint64_t>(id) * graph.row_stride;
}

uint64_t
GraphRowCount(const MCIV3GraphView& graph, InnerIdType id) {
    if (graph.counts != nullptr) {
        return graph.counts[id];
    }
    return graph.uniform_count;
}

MCIV3ThreadTiming
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
BuildMCICliquesV3(const float* vectors,
                  const MCIV3GraphView& graph,
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
        std::max<uint64_t>(2, std::min<uint64_t>({params.clique_max, candidate_limit + 1, total}));
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
        Vector<std::thread> workers(allocator);
        std::mutex norm_mutex;
        for (uint64_t tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back([&, tid]() {
                double local_max = 0.0;
                while (true) {
                    const auto start = next_id.fetch_add(K_V3_SCHEDULE_CHUNK);
                    if (start >= total) {
                        break;
                    }
                    const auto end = std::min<uint64_t>(total, start + K_V3_SCHEDULE_CHUNK);
                    for (uint64_t id = start; id < end; ++id) {
                        const float norm =
                            DotProduct(VectorAt(vectors, dim, id), VectorAt(vectors, dim, id), dim);
                        l2_norms[id] = norm;
                        local_max = std::max(local_max, std::abs(static_cast<double>(norm) - 1.0));
                    }
                }
                std::lock_guard<std::mutex> lock(norm_mutex);
                max_norm_error = std::max(max_norm_error, local_max);
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }
    const bool unit_norm = max_norm_error < 1e-3;
    logger::info("mci v3 float continuous path enabled, unit_norm={}, max_norm_error={:.6f}",
                 unit_norm,
                 max_norm_error);

    auto distance_from_dot = [&](InnerIdType lhs, InnerIdType rhs, float dot) {
        const float distance =
            unit_norm ? 2.0F - 2.0F * dot : l2_norms[lhs] + l2_norms[rhs] - 2.0F * dot;
        return distance > 0.0F ? distance : 0.0F;
    };

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
        Vector<std::thread> workers(allocator);

        for (uint64_t tid = 0; tid < thread_count; ++tid) {
            workers.emplace_back([&, tid]() {
                std::vector<MCIV3Edge> edges;
                edges.reserve(candidate_limit * (candidate_limit + 1) / 2);
                Vector<MCIV3Candidate> candidates(candidate_limit, allocator);
                Vector<const float*> candidate_vectors(candidate_limit, allocator);
                std::vector<std::vector<InnerIdType>> local_max_cliques(max_saved_cliques);
                for (auto& clique : local_max_cliques) {
                    clique.reserve(candidate_limit + 1);
                }
                mci_v3::ccrmceRunner<MCIV3Edge, InnerIdType> mce_runner;
                mce_runner.reserve(static_cast<uint32_t>(candidate_limit + 1));
                auto append_selected_clique = [&](const auto& clique) {
                    selected_by_thread[tid].emplace_back(allocator);
                    selected_by_thread[tid].back().assign(clique.begin(), clique.end());
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
                        const float dot = DotProduct(candidate_vectors[i], query, dim);
                        candidates[i].distance = distance_from_dot(candidate_id, seed, dot);
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

                    const float distance_limit = candidates[0].distance * now_alpha;
                    stage_start = std::chrono::steady_clock::now();
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        if (candidates[i].distance < distance_limit) {
                            edges.push_back(
                                MCIV3Edge{seed, candidates[i].id, candidates[i].distance});
                        }
                    }
                    for (uint64_t i = 0; i < candidate_size; ++i) {
                        const auto lhs_id = candidates[i].id;
                        const auto* lhs_vector = candidate_vectors[i];
                        uint64_t j = i + 1;
                        for (; j + 3 < candidate_size; j += 4) {
                            float dots[4];
                            DotProductBatch4(lhs_vector,
                                             candidate_vectors[j],
                                             candidate_vectors[j + 1],
                                             candidate_vectors[j + 2],
                                             candidate_vectors[j + 3],
                                             dim,
                                             dots);
                            for (uint64_t batch = 0; batch < 4; ++batch) {
                                const auto rhs_id = candidates[j + batch].id;
                                const float distance =
                                    distance_from_dot(lhs_id, rhs_id, dots[batch]);
                                if (distance <= distance_limit) {
                                    edges.push_back(MCIV3Edge{lhs_id, rhs_id, distance});
                                }
                            }
                        }
                        for (; j < candidate_size; ++j) {
                            const auto rhs_id = candidates[j].id;
                            const float distance = distance_from_dot(
                                lhs_id, rhs_id, DotProduct(lhs_vector, candidate_vectors[j], dim));
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
            });
        }
        for (auto& worker : workers) {
            worker.join();
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
