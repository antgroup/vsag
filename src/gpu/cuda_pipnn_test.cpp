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
#include <cstdint>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_backend.h"
#include "unittest.h"

// The device-resident PiPNN build, checked against a brute-force nearest-neighbour
// reference. Without the backend, or without a card, only the refusal cases run.

using namespace vsag::gpu;

namespace {

constexpr uint64_t kBudget = 1ULL << 30;

/// Gaussian blobs, so that leaves hold real neighbourhoods.
std::vector<float>
ClusteredRows(uint64_t rows, int32_t dim, uint32_t clusters, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> noise(0.0F, 1.0F);
    std::uniform_real_distribution<float> spread(-20.0F, 20.0F);
    std::vector<float> centers(static_cast<uint64_t>(clusters) * dim);
    for (auto& v : centers) {
        v = spread(gen);
    }
    std::vector<float> out(rows * static_cast<uint64_t>(dim));
    for (uint64_t i = 0; i < rows; ++i) {
        const uint64_t c = gen() % clusters;
        for (int32_t d = 0; d < dim; ++d) {
            out[i * dim + d] = centers[c * dim + d] + noise(gen);
        }
    }
    return out;
}

CudaPiPNNConfig
SmallConfig() {
    CudaPiPNNConfig config;
    config.max_leaf_size = 256;
    config.min_leaf_size = 16;
    config.leader_sample_rate = 0.005F;
    config.fanout_l1 = 4;
    config.fanout_l2 = 2;
    config.leaf_neighbor_count = 5;
    config.max_degree = 24;
    // Occlusion off: every distinct direction is kept, which is what a nearest
    // neighbour recall measures.
    config.alpha = 1e6F;
    return config;
}

struct Graph {
    std::vector<uint32_t> neighbors;
    std::vector<uint32_t> degrees;
    bool built{false};
};

Graph
BuildGraph(const std::vector<float>& rows,
           int32_t dim,
           const CudaPiPNNConfig& config,
           uint64_t budget) {
    Graph graph;
    const uint64_t count = rows.size() / dim;
    graph.neighbors.assign(count * config.max_degree, 0xFFFFFFFFU);
    graph.degrees.assign(count, 0xFFFFFFFFU);
    graph.built = CudaBuildPiPNNGraph(
        rows.data(), count, dim, config, graph.neighbors.data(), graph.degrees.data(), budget);
    return graph;
}

#ifdef VSAG_ENABLE_CUDA
void
CheckWellFormed(const Graph& graph, uint64_t count, const CudaPiPNNConfig& config) {
    for (uint64_t u = 0; u < count; ++u) {
        const uint32_t degree = graph.degrees[u];
        REQUIRE(degree <= config.max_degree);
        std::unordered_set<uint32_t> seen;
        for (uint32_t j = 0; j < degree; ++j) {
            const uint32_t v = graph.neighbors[u * config.max_degree + j];
            REQUIRE(v < count);
            REQUIRE(v != u);
            REQUIRE(seen.insert(v).second);
        }
    }
}

/// Mean fraction of each sampled row's true k nearest neighbours the graph lists.
float
NeighborRecall(const std::vector<float>& rows,
               int32_t dim,
               const Graph& graph,
               const CudaPiPNNConfig& config,
               uint32_t k) {
    const uint64_t count = rows.size() / dim;
    const uint64_t step = std::max<uint64_t>(1, count / 200);
    uint64_t hits = 0;
    uint64_t total = 0;
    std::vector<std::pair<float, uint32_t>> dists(count);
    for (uint64_t u = 0; u < count; u += step) {
        for (uint64_t v = 0; v < count; ++v) {
            float d = 0.0F;
            for (int32_t j = 0; j < dim; ++j) {
                const float diff = rows[u * dim + j] - rows[v * dim + j];
                d += diff * diff;
            }
            dists[v] = {u == v ? 1e30F : d, static_cast<uint32_t>(v)};
        }
        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
        std::unordered_set<uint32_t> listed(
            graph.neighbors.begin() + u * config.max_degree,
            graph.neighbors.begin() + u * config.max_degree + graph.degrees[u]);
        for (uint32_t i = 0; i < k; ++i) {
            hits += listed.count(dists[i].second);
        }
        total += k;
    }
    return static_cast<float>(hits) / static_cast<float>(total);
}

bool
NoDevice() {
    return not CudaAvailable();
}
#endif  // VSAG_ENABLE_CUDA

}  // namespace

// Holds in every build: with no device the build declines and writes nothing.
TEST_CASE("PiPNN build: a refusal leaves the outputs alone", "[ut][gpu_pipnn]") {
    const int32_t dim = 16;
    const auto rows = ClusteredRows(512, dim, 4, 1);
    auto config = SmallConfig();

    SECTION("an unsupported metric") {
        config.distance_type = DistanceType::InnerProduct;
        const auto graph = BuildGraph(rows, dim, config, kBudget);
        REQUIRE_FALSE(graph.built);
        REQUIRE(graph.degrees[0] == 0xFFFFFFFFU);
    }

    SECTION("an empty budget") {
        const auto graph = BuildGraph(rows, dim, config, 0);
        REQUIRE_FALSE(graph.built);
        REQUIRE(graph.degrees[0] == 0xFFFFFFFFU);
    }

    SECTION("out-of-range parameters") {
        config.max_degree = 0;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.leaf_neighbor_count = 0;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.hash_plane_count = 13;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.distance_type = DistanceType::InnerProduct;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.alpha = 0.9F;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.reservoir_size = 129;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
        config = SmallConfig();
        config.max_leaf_size = 8;
        config.min_leaf_size = 16;
        REQUIRE_FALSE(BuildGraph(rows, dim, config, kBudget).built);
    }

    SECTION("missing buffers and degenerate shapes") {
        uint32_t neighbors[4] = {};
        uint32_t degrees[2] = {};
        REQUIRE_FALSE(CudaBuildPiPNNGraph(nullptr, 2, 16, config, neighbors, degrees, kBudget));
        REQUIRE_FALSE(
            CudaBuildPiPNNGraph(rows.data(), 1, dim, config, neighbors, degrees, kBudget));
        REQUIRE_FALSE(CudaBuildPiPNNGraph(rows.data(), 2, 0, config, neighbors, degrees, kBudget));
    }
}

#ifdef VSAG_ENABLE_CUDA

TEST_CASE("PiPNN build: the graph is well formed and finds the nearest neighbours",
          "[ut][gpu_pipnn]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    // 37 is neither a multiple of 4 nor of 16, so the padding is exercised.
    const int32_t dim = 37;
    const uint64_t count = 12000;
    const auto rows = ClusteredRows(count, dim, 40, 3);
    const auto config = SmallConfig();

    const auto graph = BuildGraph(rows, dim, config, kBudget);
    REQUIRE(graph.built);
    CheckWellFormed(graph, count, config);
    REQUIRE(NeighborRecall(rows, dim, graph, config, 10) >= 0.75F);
}

TEST_CASE("PiPNN build: a small budget sorts in slabs and still agrees", "[ut][gpu_pipnn]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 32;
    const uint64_t count = 12000;
    const auto rows = ClusteredRows(count, dim, 40, 5);
    const auto config = SmallConfig();

    const auto whole = BuildGraph(rows, dim, config, kBudget);
    REQUIRE(whole.built);

    // Past the fixed scratch margin but short of the candidate edges (about 23 MiB),
    // so the edges can only be sorted a slab at a time.
    const auto slabs = BuildGraph(rows, dim, config, 80ULL << 20);
    REQUIRE(slabs.built);
    CheckWellFormed(slabs, count, config);
    REQUIRE(NeighborRecall(rows, dim, slabs, config, 10) >= 0.75F);
    // The pruning is keyed on the vertex, so how the edges were batched is invisible.
    REQUIRE(slabs.degrees == whole.degrees);
}

TEST_CASE("PiPNN build: the occlusion rule thins the candidate list", "[ut][gpu_pipnn]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 32;
    const uint64_t count = 12000;
    const auto rows = ClusteredRows(count, dim, 40, 7);
    auto config = SmallConfig();

    const auto open = BuildGraph(rows, dim, config, kBudget);
    REQUIRE(open.built);

    config.alpha = 1.0F;
    const auto strict = BuildGraph(rows, dim, config, kBudget);
    REQUIRE(strict.built);
    CheckWellFormed(strict, count, config);

    config.alpha = 1.2F;
    const auto relaxed = BuildGraph(rows, dim, config, kBudget);
    REQUIRE(relaxed.built);
    CheckWellFormed(relaxed, count, config);

    uint64_t open_edges = 0;
    uint64_t strict_edges = 0;
    uint64_t relaxed_edges = 0;
    for (uint64_t u = 0; u < count; ++u) {
        REQUIRE(strict.degrees[u] <= open.degrees[u]);
        REQUIRE(relaxed.degrees[u] <= open.degrees[u]);
        // The nearest candidate is never occluded, so a vertex with any candidate keeps one.
        REQUIRE((strict.degrees[u] > 0) == (open.degrees[u] > 0));
        // Nearest first: the head of the list does not change with alpha.
        if (open.degrees[u] > 0) {
            REQUIRE(strict.neighbors[u * config.max_degree] ==
                    open.neighbors[u * config.max_degree]);
        }
        open_edges += open.degrees[u];
        strict_edges += strict.degrees[u];
        relaxed_edges += relaxed.degrees[u];
    }
    REQUIRE(strict_edges < relaxed_edges);
    REQUIRE(relaxed_edges < open_edges);
}

TEST_CASE("PiPNN build: a budget the dataset cannot fit is refused", "[ut][gpu_pipnn]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 32;
    const auto rows = ClusteredRows(4000, dim, 10, 9);
    const auto graph = BuildGraph(rows, dim, SmallConfig(), 1ULL << 16);
    REQUIRE_FALSE(graph.built);
    REQUIRE(graph.degrees[0] == 0xFFFFFFFFU);
}

TEST_CASE("PiPNN build: tiny inputs and duplicate rows stay valid", "[ut][gpu_pipnn]") {
    if (NoDevice()) {
        SKIP("no CUDA device on this machine");
    }
    const int32_t dim = 8;
    auto config = SmallConfig();

    SECTION("fewer rows than a leaf") {
        const auto rows = ClusteredRows(10, dim, 2, 11);
        const auto graph = BuildGraph(rows, dim, config, kBudget);
        REQUIRE(graph.built);
        CheckWellFormed(graph, 10, config);
    }

    SECTION("identical rows") {
        const std::vector<float> rows(300 * dim, 1.5F);
        const auto graph = BuildGraph(rows, dim, config, kBudget);
        REQUIRE(graph.built);
        CheckWellFormed(graph, 300, config);
    }
}

#endif
