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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <vector>

#include "mci_builder.h"
#include "mci_runner.h"

namespace vsag {

struct MCILocalBuildStats {
    double candidate_collect{0.0};
    double query_distance{0.0};
    double pair_distance{0.0};
    double edge_sort{0.0};
    double mce{0.0};
    double choose{0.0};
    uint64_t candidates{0};
    uint64_t edges{0};
    uint64_t cliques{0};
};

inline float
next_mci_alpha(float alpha, float initial_alpha, uint64_t uncovered, uint64_t previous_uncovered) {
    return uncovered < static_cast<uint64_t>(0.9 * static_cast<double>(previous_uncovered))
               ? alpha + initial_alpha
               : alpha * 2.0F;
}

/// Reusable seed-local part of BuildMCICliques. IDs index the supplied coverage counters;
/// incremental callers may remap their neighborhood to compact local IDs. The caller owns
/// round scheduling and output storage. Distances can use either SIMD or stored vector codes.
class MCILocalCliqueBuilder {
public:
    MCILocalCliqueBuilder(const MCIV3BuildParams& params, Allocator* allocator)
        : params_(params),
          candidate_limit_(
              std::min(params.candidate_limit, params.total > 0 ? params.total - 1 : 0)),
          threshold_(std::max<uint64_t>(
              2, std::min<uint64_t>({params.clique_max, candidate_limit_ + 1, params.total}))),
          max_saved_(std::min<uint64_t>(candidate_limit_, params.max_degree + 2)),
          candidates_(allocator),
          local_cliques_(max_saved_) {
        candidates_.reserve(candidate_limit_);
        edges_.reserve(candidate_limit_ * (candidate_limit_ + 1) / 2);
        for (auto& clique : local_cliques_) {
            clique.reserve(candidate_limit_ + 1);
        }
        runner_.reserve(static_cast<uint32_t>(candidate_limit_ + 1));
    }

    template <typename Distance, typename BatchDistance, typename Emit>
    MCILocalBuildStats
    Build(InnerIdType seed,
          const InnerIdType* neighbors,
          uint64_t neighbor_count,
          float alpha,
          std::vector<std::atomic<int>>& coverage,
          Distance distance,
          BatchDistance batch_distance,
          Emit emit) {
        MCILocalBuildStats stats;
        candidates_.clear();
        edges_.clear();
        auto start = std::chrono::steady_clock::now();
        const auto node_limit = std::max<uint64_t>(3, params_.total / 100);
        for (uint64_t i = 0; i < std::min(neighbor_count, candidate_limit_); ++i) {
            const auto id = neighbors[i];
            if (id >= params_.total or id >= coverage.size() or id == seed or
                coverage[id].load(std::memory_order_relaxed) >= static_cast<int>(node_limit)) {
                continue;
            }
            if (std::none_of(candidates_.begin(), candidates_.end(), [id](const auto& candidate) {
                    return candidate.id == id;
                })) {
                candidates_.push_back({id, 0.0F});
            }
        }
        stats.candidates = candidates_.size();
        stats.candidate_collect = elapsed(start);

        auto append = [&](auto clique) {
            const bool contains_seed =
                std::find(clique.begin(), clique.end(), seed) != clique.end();
            if (clique.size() > params_.clique_max) {
                clique.resize(params_.clique_max);
                if (contains_seed and
                    std::find(clique.begin(), clique.end(), seed) == clique.end()) {
                    clique.back() = seed;
                }
            }
            if (std::none_of(clique.begin(), clique.end(), [&](auto id) {
                    return coverage[id].load(std::memory_order_relaxed) == 0;
                })) {
                return false;
            }
            emit(clique);
            // Count only memberships that were actually emitted, including capped fallbacks.
            for (auto id : clique) {
                coverage[id].fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        };
        auto fallback = [&]() {
            std::vector<InnerIdType> clique{seed};
            for (const auto& candidate : candidates_) {
                if (coverage[candidate.id].load(std::memory_order_relaxed) > 0) {
                    clique.push_back(candidate.id);
                }
            }
            append(std::move(clique));
        };
        if (candidates_.size() + 1 < threshold_) {
            if (alpha > FALLBACK_ALPHA_THRESHOLD) {
                fallback();
            }
            return stats;
        }

        start = std::chrono::steady_clock::now();
        for (auto& candidate : candidates_) {
            candidate.distance = distance(candidate.id, seed);
        }
        std::sort(candidates_.begin(), candidates_.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.distance < rhs.distance;
        });
        stats.query_distance = elapsed(start);
        const auto limit = distance_limit(candidates_.front().distance, alpha, params_.metric);
        start = std::chrono::steady_clock::now();
        for (const auto& candidate : candidates_) {
            if (candidate.distance < limit) {
                edges_.push_back({seed, candidate.id, candidate.distance});
            }
        }
        for (uint64_t i = 0; i < candidates_.size(); ++i) {
            const auto lhs = candidates_[i].id;
            uint64_t j = i + 1;
            for (; j + 3 < candidates_.size(); j += 4) {
                float values[4]{};
                batch_distance(lhs,
                               candidates_[j].id,
                               candidates_[j + 1].id,
                               candidates_[j + 2].id,
                               candidates_[j + 3].id,
                               values);
                for (uint64_t b = 0; b < 4; ++b) {
                    if (values[b] <= limit) {
                        edges_.push_back({lhs, candidates_[j + b].id, values[b]});
                    }
                }
            }
            for (; j < candidates_.size(); ++j) {
                const auto rhs = candidates_[j].id;
                const auto value = distance(lhs, rhs);
                if (value <= limit) {
                    edges_.push_back({lhs, rhs, value});
                }
            }
        }
        stats.edges = edges_.size();
        stats.pair_distance = elapsed(start);
        start = std::chrono::steady_clock::now();
        std::sort(edges_.begin(), edges_.end());
        stats.edge_sort = elapsed(start);
        if (edges_.size() < threshold_ * (threshold_ - 1) / 2 and
            alpha > FALLBACK_ALPHA_THRESHOLD) {
            fallback();
            return stats;
        }
        start = std::chrono::steady_clock::now();
        stats.cliques = runner_.run(edges_,
                                    local_cliques_,
                                    static_cast<InnerIdType>(threshold_),
                                    coverage,
                                    static_cast<InnerIdType>(max_saved_));
        stats.mce = elapsed(start);
        if (stats.cliques == 0 and alpha > FALLBACK_ALPHA_THRESHOLD) {
            fallback();
            return stats;
        }
        start = std::chrono::steady_clock::now();
        uint64_t chosen = 0;
        for (uint64_t i = 0; i < stats.cliques; ++i) {
            if (append(local_cliques_[i]) and ++chosen > params_.max_degree) {
                break;
            }
        }
        stats.choose = elapsed(start);
        return stats;
    }

private:
    // Above this alpha, failed local enumeration may relax constraints to cover the seed.
    static constexpr float FALLBACK_ALPHA_THRESHOLD = 100.0F;

    struct Candidate {
        InnerIdType id;
        float distance;
    };
    struct Edge {
        InnerIdType u;
        InnerIdType v;
        float dis;
        bool
        operator<(const Edge& other) const {
            return dis < other.dis;
        }
    };

    static double
    elapsed(const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    static float
    distance_limit(float nearest, float alpha, MetricType metric) {
        if (metric != MetricType::METRIC_TYPE_IP) {
            return nearest * alpha;
        }
        const auto similarity = 1.0F - nearest;
        // IP distance is 1 - dot, without normalization or a [0, 2] bound. For negative
        // similarity, multiplication relaxes the bound as alpha grows; clamping would
        // reject legitimate negative-IP neighbors. E.g. nearest=2, alpha=2 gives limit=3,
        // which still excludes a pair at distance 4.
        return 1.0F - (similarity >= 0.0F ? similarity / alpha : similarity * alpha);
    }

    MCIV3BuildParams params_;
    uint64_t candidate_limit_;
    uint64_t threshold_;
    uint64_t max_saved_;
    Vector<Candidate> candidates_;
    std::vector<Edge> edges_;
    std::vector<std::vector<InnerIdType>> local_cliques_;
    mci::ccrmce_runner<Edge, InnerIdType> runner_;
};

}  // namespace vsag
