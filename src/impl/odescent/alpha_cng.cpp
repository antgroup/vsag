// Copyright 2026 the vsag project
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
#include <cmath>
#include <numeric>

#include "impl/adaptive_pruning.h"
#include "odescent_graph_builder.h"
#include "simd/fp32_simd.h"

namespace vsag {
namespace {

constexpr uint32_t kBeamWidth = 60;
constexpr uint32_t kCandidateLimit = 500;

uint32_t
centroid_entry(const Vector<const float*>& rows, uint64_t dim, Allocator* allocator) {
    Vector<double> sum(dim, 0.0, allocator);
    for (const auto* row : rows) {
        for (uint64_t j = 0; j < dim; ++j) {
            sum[j] += row[j];
        }
    }
    Vector<float> centroid(dim, allocator);
    for (uint64_t j = 0; j < dim; ++j) {
        centroid[j] = static_cast<float>(sum[j] / rows.size());
    }
    float best = std::numeric_limits<float>::max();
    uint32_t entry = 0;
    for (uint32_t id = 0; id < rows.size(); ++id) {
        float d = FP32ComputeL2Sqr(rows[id], centroid.data(), dim);
        if (d < best) {
            best = d;
            entry = id;
        }
    }
    return entry;
}

// Retain the closest C *evaluated* candidates, including points that missed the L-wide beam.
void
collect_candidates(const Vector<Linklist>& seed,
                   const Vector<const float*>& rows,
                   uint64_t dim,
                   uint32_t query,
                   uint32_t entry,
                   Vector<uint32_t>& visited,
                   Vector<Node>& beam,
                   Vector<Node>& candidates) {
    beam.clear();
    candidates.clear();
    auto visit = [&](uint32_t id) {
        if (visited[id] == query) {
            return;
        }
        visited[id] = query;
        Node item(id, FP32ComputeL2Sqr(rows[query], rows[id], dim));
        if (id != query) {
            candidates.push_back(item);
        }
        if (beam.size() == kBeamWidth and not(item < beam.back())) {
            return;
        }
        auto pos = std::lower_bound(beam.begin(), beam.end(), item);
        beam.insert(pos, item);
        if (beam.size() > kBeamWidth) {
            beam.pop_back();
        }
    };
    visit(entry);
    for (;;) {
        auto next = std::find_if(beam.begin(), beam.end(), [](const Node& n) { return not n.old; });
        if (next == beam.end()) {
            break;
        }
        next->old = true;
        const auto id = next->id;
        for (const auto& neighbor : seed[id].neighbors) {
            visit(neighbor.id);
        }
    }
    if (candidates.size() > kCandidateLimit) {
        std::nth_element(
            candidates.begin(), candidates.begin() + kCandidateLimit, candidates.end());
        candidates.resize(kCandidateLimit);
    }
    for (auto& candidate : candidates) {
        candidate.distance = std::sqrt(candidate.distance);
    }
    std::sort(candidates.begin(), candidates.end());
}

// Use local positions as IDs so ties after subtracting tau retain the original distance order.
void
adaptive_prune(const Vector<Node>& sorted,
               const Vector<const float*>& rows,
               uint64_t dim,
               uint32_t degree,
               float alpha,
               const AdaptivePruningParameter& pruning,
               float tau,
               Vector<Node>& result,
               Allocator* allocator) {
    auto selected = select_alpha_cng_edges(
        sorted.size(),
        degree,
        alpha,
        pruning,
        tau,
        [&](uint32_t i) { return sorted[i].distance; },
        [&](uint32_t i, uint32_t j) {
            return FP32ComputeL2Sqr(rows[sorted[i].id], rows[sorted[j].id], dim);
        },
        allocator);
    result.clear();
    for (const auto& candidate : selected) {
        result.push_back(sorted[candidate.second]);
    }
}

// The last vertex expanded by each DFS has no tree children. Its last edge can be replaced
// without disconnecting an already reached vertex, even when the storage degree is full.
void
repair_reachability(Vector<Linklist>& graph,
                    uint32_t entry,
                    uint32_t degree,
                    Allocator* allocator) {
    Vector<uint8_t> seen(graph.size(), 0, allocator);
    Vector<uint32_t> stack(allocator);
    uint32_t leaf = entry;
    auto traverse = [&](uint32_t root) {
        seen[root] = 1;
        stack.push_back(root);
        while (not stack.empty()) {
            leaf = stack.back();
            stack.pop_back();
            for (const auto& n : graph[leaf].neighbors) {
                if (not seen[n.id]) {
                    seen[n.id] = 1;
                    stack.push_back(n.id);
                }
            }
        }
    };
    traverse(entry);
    for (uint32_t id = 0; id < graph.size(); ++id) {
        if (not seen[id]) {
            auto& edges = graph[leaf].neighbors;
            if (edges.size() == degree) {
                edges.pop_back();
            }
            edges.emplace_back(id, 0.0F);
            traverse(id);
        }
    }
}

}  // namespace

void
ODescent::BuildAlphaCng(const Vector<InnerIdType>& ids,
                        const Vector<const float*>& rows,
                        uint64_t dim,
                        uint32_t max_degree,
                        float alpha,
                        float adjust_step,
                        float tau,
                        InnerIdType route_entry) {
    CHECK_ARGUMENT(not pruning_ and not ids.empty() and ids.size() == rows.size(),
                   "alpha_cng requires an unpruned K-NN seed and matching rows");
    const AdaptivePruningParameter pruning{.enabled = true, .adjust_step = adjust_step};
    pruning.Validate(alpha);
    Build(ids);
    if (ids.size() == 1) {
        return;
    }
    const auto entry = centroid_entry(rows, dim, allocator_);
    Vector<Linklist> forward(ids.size(), Linklist(allocator_), allocator_);
    parallelize_task([&](int64_t begin, int64_t end) {
        Vector<uint32_t> visited(ids.size(), std::numeric_limits<uint32_t>::max(), allocator_);
        Vector<Node> candidates(allocator_), beam(allocator_);
        beam.reserve(kBeamWidth + 1);
        for (int64_t i = begin; i < end; ++i) {
            collect_candidates(graph_, rows, dim, i, entry, visited, beam, candidates);
            adaptive_prune(candidates,
                           rows,
                           dim,
                           max_degree,
                           alpha,
                           pruning,
                           tau,
                           forward[i].neighbors,
                           allocator_);
        }
    });
    graph_.swap(forward);
    Vector<Linklist>(allocator_).swap(forward);

    // Store all incoming edges in CSR order, then prune each overflowing union exactly once.
    Vector<uint64_t> offsets(ids.size() + 1, 0, allocator_);
    for (const auto& row : graph_) {
        for (const auto& n : row.neighbors) {
            ++offsets[n.id + 1];
        }
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    auto cursor = offsets;
    Vector<Node> incoming(offsets.back(), allocator_);
    for (uint32_t id = 0; id < ids.size(); ++id) {
        for (const auto& n : graph_[id].neighbors) {
            incoming[cursor[n.id]++] = Node(id, n.distance);
        }
    }
    parallelize_task([&](int64_t begin, int64_t end) {
        Vector<Node> merged(allocator_);
        for (int64_t i = begin; i < end; ++i) {
            auto& edges = graph_[i].neighbors;
            merged.assign(edges.begin(), edges.end());
            merged.insert(
                merged.end(), incoming.begin() + offsets[i], incoming.begin() + offsets[i + 1]);
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            if (merged.size() > max_degree) {
                adaptive_prune(
                    merged, rows, dim, max_degree, alpha, pruning, tau, edges, allocator_);
            } else {
                edges.assign(merged.begin(), merged.end());
            }
        }
    });
    auto it = std::find(ids.begin(), ids.end(), route_entry);
    repair_reachability(graph_, it == ids.end() ? 0 : it - ids.begin(), max_degree, allocator_);
}

}  // namespace vsag
