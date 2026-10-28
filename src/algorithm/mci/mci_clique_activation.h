// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <future>
#include <numeric>
#include <vector>

#include "mci_builder.h"

namespace vsag {

// Reorder outgoing incidences only. Members and clique IDs remain unchanged. The first
// greedy part covers the input KNN edges already represented by these cliques; the tail
// prefers a larger fraction of near edges. No new distance evaluations are required.
inline void
rank_mci_clique_activation(const MCIGraphView& graph,
                           const Vector<Vector<InnerIdType>>& cliques,
                           Vector<Vector<InnerIdType>>& rows,
                           uint64_t thread_count) {
    if (graph.neighbors == nullptr or graph.total == 0 or graph.row_stride == 0) {
        return;
    }
    const auto worker_count = std::max<uint64_t>(1, std::min(thread_count, graph.total));
    std::atomic<uint64_t> next{0};
    std::vector<std::future<void>> workers;
    for (uint64_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back(std::async(std::launch::async, [&]() {
            std::vector<uint32_t> marks(graph.total, 0);
            std::vector<uint64_t> ranks(graph.total);
            uint32_t epoch = 0;
            std::vector<uint64_t> masks;
            std::vector<uint64_t> covered;
            std::vector<uint64_t> local;
            std::vector<uint64_t> tail;
            std::vector<uint8_t> picked;
            std::vector<InnerIdType> ordered;
            while (true) {
                const auto u = next.fetch_add(1, std::memory_order_relaxed);
                if (u >= graph.total) {
                    break;
                }
                auto& ids = rows[u];
                if (ids.size() <= 32) {
                    continue;
                }
                if (++epoch == 0) {
                    std::fill(marks.begin(), marks.end(), 0);
                    epoch = 1;
                }
                const auto count = std::min<uint64_t>(
                    graph.counts == nullptr ? graph.uniform_count : graph.counts[u],
                    graph.row_stride);
                uint64_t targets = 0;
                for (uint64_t i = 0; i < count; ++i) {
                    const auto v = graph.neighbors[u * graph.row_stride + i];
                    if (v < graph.total and v != u and marks[v] != epoch) {
                        marks[v] = epoch;
                        ranks[v] = targets++;
                    }
                }
                const auto words = (targets + 63) / 64;
                masks.assign(ids.size() * words, 0);
                covered.assign(words, 0);
                local.assign(ids.size(), 0);
                picked.assign(ids.size(), 0);
                ordered.clear();
                for (uint64_t i = 0; i < ids.size(); ++i) {
                    for (auto v : cliques[ids[i]]) {
                        if (v < graph.total and marks[v] == epoch) {
                            masks[i * words + ranks[v] / 64] |= uint64_t{1} << (ranks[v] % 64);
                        }
                    }
                    for (uint64_t w = 0; w < words; ++w) {
                        local[i] += __builtin_popcountll(masks[i * words + w]);
                    }
                }
                while (true) {
                    uint64_t best = ids.size();
                    uint64_t best_gain = 0;
                    for (uint64_t i = 0; i < ids.size(); ++i) {
                        if (picked[i] != 0) {
                            continue;
                        }
                        uint64_t gain = 0;
                        for (uint64_t w = 0; w < words; ++w) {
                            gain += __builtin_popcountll(masks[i * words + w] & ~covered[w]);
                        }
                        if (gain > best_gain or
                            (gain != 0 and gain == best_gain and
                             cliques[ids[i]].size() < cliques[ids[best]].size())) {
                            best = i;
                            best_gain = gain;
                        }
                    }
                    if (best_gain == 0) {
                        break;
                    }
                    picked[best] = 1;
                    ordered.push_back(ids[best]);
                    for (uint64_t w = 0; w < words; ++w) {
                        covered[w] |= masks[best * words + w];
                    }
                }
                tail.clear();
                for (uint64_t i = 0; i < ids.size(); ++i) {
                    if (picked[i] == 0) {
                        tail.push_back(i);
                    }
                }
                std::sort(tail.begin(), tail.end(), [&](auto a, auto b) {
                    const auto lhs = local[a] * cliques[ids[b]].size();
                    const auto rhs = local[b] * cliques[ids[a]].size();
                    return lhs != rhs ? lhs > rhs : ids[a] < ids[b];
                });
                for (auto i : tail) {
                    ordered.push_back(ids[i]);
                }
                std::copy(ordered.begin(), ordered.end(), ids.begin());
            }
        }));
    }
    for (auto& worker : workers) {
        worker.get();
    }
}

}  // namespace vsag
