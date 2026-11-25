// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pipnn_graph_builder.h"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "datacell/graph_interface.h"
#include "datacell/graph_interface_parameter.h"
#include "half_vector.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "pipnn_build_cache.h"
#include "pipnn_incremental.h"
#include "simd/bf16_simd.h"
#include "simd/fp32_simd.h"
#include "simd/pipnn_filter.h"
#include "simd/pipnn_ip_tile.h"
#include "unittest.h"
#include "utils/half_input.h"
#include "vsag_exception.h"

namespace {

class RejectSecondSubmissionPool : public vsag::ThreadPool {
public:
    std::promise<void> rejected;
    std::shared_ptr<std::packaged_task<void()>> pending;

    std::future<void>
    Enqueue(std::function<void()> task) override {
        if (pending != nullptr) {
            rejected.set_value();
            throw std::runtime_error("injected PiPNN submission failure");
        }
        pending = std::make_shared<std::packaged_task<void()>>(std::move(task));
        return pending->get_future();
    }

    void
    WaitUntilEmpty() override {
    }

    void
    SetQueueSizeLimit(uint64_t) override {
    }

    void
    SetPoolSize(uint64_t) override {
    }
};

vsag::IndexCommonParam
MakeCommonParam(uint64_t dimensions,
                vsag::MetricType metric = vsag::MetricType::METRIC_TYPE_L2SQR) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = static_cast<int64_t>(dimensions);
    common_param.metric_ = metric;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

std::vector<float>
MakeVectors(uint64_t count, uint64_t dimensions) {
    std::vector<float> vectors(count * dimensions);
    for (uint64_t point = 0; point < count; ++point) {
        for (uint64_t dim = 0; dim < dimensions; ++dim) {
            vectors[point * dimensions + dim] = std::sin(static_cast<float>(point * 17 + dim * 3)) +
                                                static_cast<float>(point) * 0.01F;
        }
    }
    return vectors;
}

vsag::Vector<const float*>
MakeRows(const std::vector<float>& vectors,
         const vsag::Vector<vsag::InnerIdType>& ids,
         uint64_t dimensions,
         vsag::Allocator* allocator) {
    vsag::Vector<const float*> rows(allocator);
    rows.reserve(ids.size());
    for (const auto id : ids) {
        rows.emplace_back(vectors.data() + static_cast<uint64_t>(id) * dimensions);
    }
    return rows;
}

vsag::GraphInterfacePtr
MakeGraph(const vsag::IndexCommonParam& common_param,
          uint64_t count,
          uint64_t max_degree,
          bool support_duplicate = false) {
    const auto graph_json = vsag::JsonType::Parse(fmt::format(
        R"({{"io_params": {{"type": "block_memory_io"}}, "max_degree": {}, "init_capacity": {}, "support_duplicate": {}}})",
        max_degree,
        std::max<uint64_t>(count, 1),
        support_duplicate));
    const auto graph_param = vsag::GraphInterfaceParameter::GetGraphParameterByJson(
        vsag::GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, graph_json);
    auto graph = vsag::GraphInterface::MakeInstance(graph_param, common_param);
    graph->Resize(static_cast<vsag::InnerIdType>(std::max<uint64_t>(count, 1)));
    return graph;
}

void
RequireGraphInvariants(const vsag::GraphInterfacePtr& graph,
                       const vsag::Vector<vsag::InnerIdType>& ids,
                       uint64_t max_degree,
                       vsag::Allocator* allocator,
                       bool require_neighbors = true) {
    std::set<vsag::InnerIdType> valid(ids.begin(), ids.end());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> neighbors(allocator);
        graph->GetNeighbors(id, neighbors);
        REQUIRE(neighbors.size() <= max_degree);
        REQUIRE(std::set<vsag::InnerIdType>(neighbors.begin(), neighbors.end()).size() ==
                neighbors.size());
        for (const auto neighbor : neighbors) {
            REQUIRE(neighbor != id);
            REQUIRE(valid.find(neighbor) != valid.end());
        }
        if (require_neighbors && ids.size() > 1) {
            REQUIRE_FALSE(neighbors.empty());
        }
    }
}

}  // namespace

TEST_CASE("PiPNN drains accepted workers when task submission fails",
          "[ut][pipnn][submission_failure]") {
    auto common = MakeCommonParam(4);
    auto graph = MakeGraph(common, 512, 16);
    auto vectors = MakeVectors(512, 4);
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    for (vsag::InnerIdType id = 0; id < 512; ++id) {
        ids.emplace_back(id);
    }
    auto rows = MakeRows(vectors, ids, 4, common.allocator_.get());
    auto pool = std::make_shared<RejectSecondSubmissionPool>();
    auto rejected = pool->rejected.get_future();
    vsag::SafeThreadPool safe_pool(pool);
    vsag::PiPNNGraphBuilder builder({}, 4, common.metric_, common.allocator_.get(), &safe_pool, 2);
    auto build = std::async(std::launch::async, [&]() {
        try {
            builder.Build(graph, ids, rows);
        } catch (const std::runtime_error& error) {
            return std::string(error.what());
        }
        return std::string{};
    });
    rejected.wait();
    // Keep the accepted task pending: Build must not unwind its captured state yet.
    const bool returned_early =
        build.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready;
    if (not returned_early) {
        (*pool->pending)();
    }
    // On the buggy path, discard the callback instead of executing dangling references.
    CHECK(build.get() == "injected PiPNN submission failure");
    CHECK_FALSE(returned_early);
}

TEST_CASE("PiPNN graph builder preserves adjacency invariants and determinism", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 8;
    constexpr uint64_t count = 40;
    constexpr uint64_t max_degree = 8;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);

    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; id += 2) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 6;
    parameter.min_leaf_size = 2;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {2, 1};
    parameter.reservoir_size = 16;

    auto first = MakeGraph(common_param, count, max_degree);
    auto second = MakeGraph(common_param, count, max_degree);
    vsag::PiPNNGraphBuilder builder(
        parameter, dimensions, common_param.metric_, common_param.allocator_.get());
    builder.Build(first, ids, rows);
    builder.Build(second, ids, rows);

    RequireGraphInvariants(first, ids, max_degree, common_param.allocator_.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> first_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> second_neighbors(common_param.allocator_.get());
        first->GetNeighbors(id, first_neighbors);
        second->GetNeighbors(id, second_neighbors);
        REQUIRE(first_neighbors == second_neighbors);
    }
}

TEST_CASE("PiPNN graph builder writes an empty row for one point", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 4;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(1, dimensions);
    auto graph = MakeGraph(common_param, 1, 4);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    ids.emplace_back(0);
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());

    vsag::PiPNNGraphBuilder({}, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(graph, ids, rows);

    vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors.empty());
}

TEST_CASE("PiPNN adaptive zero step preserves original adjacency", "[ut][pipnn][pipnn_adaptive]") {
    const uint64_t dimensions = GENERATE(8, 960);
    constexpr uint64_t count = 80;
    auto common = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    for (uint32_t id = 0; id < count; ++id) {
        ids.emplace_back(id);
    }
    const auto rows = MakeRows(vectors, ids, dimensions, common.allocator_.get());
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.alpha = 1.2F;
    parameter.max_leaf_size = 12;
    parameter.min_leaf_size = 4;
    auto baseline = MakeGraph(common, count, 8);
    auto zero_step = MakeGraph(common, count, 8);
    auto adaptive = MakeGraph(common, count, 8);
    vsag::PiPNNGraphBuilder(parameter, dimensions, common.metric_, common.allocator_.get())
        .Build(baseline, ids, rows);
    parameter.adaptive_pruning = true;
    parameter.adaptive_pruning_adjust_step = 0;
    vsag::PiPNNGraphBuilder(parameter, dimensions, common.metric_, common.allocator_.get())
        .Build(zero_step, ids, rows);
    parameter.adaptive_pruning_adjust_step = 0.06F;
    vsag::PiPNNGraphBuilder(parameter, dimensions, common.metric_, common.allocator_.get())
        .Build(adaptive, ids, rows);
    RequireGraphInvariants(adaptive, ids, 8, common.allocator_.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> first(common.allocator_.get()),
            second(common.allocator_.get());
        baseline->GetNeighbors(id, first);
        zero_step->GetNeighbors(id, second);
        REQUIRE(first == second);
    }
    REQUIRE_THROWS(vsag::PiPNNGraphBuilder(
        parameter, dimensions, vsag::MetricType::METRIC_TYPE_IP, common.allocator_.get()));
}

TEST_CASE("PiPNN graph builder runs overlapping leaves in parallel",
          "[ut][pipnn][pipnn_adaptive]") {
    const bool adaptive_pruning = GENERATE(false, true);
    constexpr uint64_t dimensions = 8;
    constexpr uint64_t count = 96;
    constexpr uint64_t max_degree = 8;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto serial = MakeGraph(common_param, count, max_degree);
    auto first = MakeGraph(common_param, count, max_degree);
    auto second = MakeGraph(common_param, count, max_degree);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 12;
    parameter.min_leaf_size = 3;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {3, 2};
    parameter.reservoir_size = 16;
    parameter.alpha = 1.2F;
    parameter.adaptive_pruning = adaptive_pruning;
    vsag::PiPNNGraphBuilder(
        parameter, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(serial, ids, rows);
    auto thread_pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    thread_pool->SetPoolSize(4);
    vsag::PiPNNGraphBuilder builder(parameter,
                                    dimensions,
                                    common_param.metric_,
                                    common_param.allocator_.get(),
                                    thread_pool.get(),
                                    4);
    builder.Build(first, ids, rows);
    builder.Build(second, ids, rows);

    RequireGraphInvariants(first, ids, max_degree, common_param.allocator_.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> serial_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> first_neighbors(common_param.allocator_.get());
        vsag::Vector<vsag::InnerIdType> second_neighbors(common_param.allocator_.get());
        serial->GetNeighbors(id, serial_neighbors);
        first->GetNeighbors(id, first_neighbors);
        second->GetNeighbors(id, second_neighbors);
        REQUIRE(serial_neighbors == first_neighbors);
        REQUIRE(first_neighbors == second_neighbors);
    }
}

TEST_CASE("PiPNN keeps identical vectors reachable across fallback leaves", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 4;
    constexpr uint64_t max_leaf_size = 8;
    constexpr uint64_t count = max_leaf_size + 1;
    constexpr uint64_t max_degree = 4;
    auto common_param = MakeCommonParam(dimensions);
    std::vector<float> vectors(count * dimensions, 1.0F);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, max_degree);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = max_leaf_size;
    parameter.min_leaf_size = 2;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {2, 1};
    parameter.hash_plane_count = 3;
    parameter.reservoir_size = 8;
    vsag::PiPNNGraphBuilder(
        parameter, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(graph, ids, rows);

    RequireGraphInvariants(graph, ids, max_degree, common_param.allocator_.get());
    std::set<vsag::InnerIdType> visited{ids.front()};
    std::vector<vsag::InnerIdType> pending{ids.front()};
    for (uint64_t cursor = 0; cursor < pending.size(); ++cursor) {
        vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
        graph->GetNeighbors(pending[cursor], neighbors);
        for (const auto neighbor : neighbors) {
            if (visited.emplace(neighbor).second) {
                pending.emplace_back(neighbor);
            }
        }
    }
    REQUIRE(visited.size() == count);
}

TEST_CASE("PiPNN terminates partitioning for a dominant near-duplicate cluster", "[ut][pipnn]") {
    constexpr uint64_t dimensions = 4;
    constexpr uint64_t count = 2049;
    constexpr uint64_t max_degree = 8;
    auto common_param = MakeCommonParam(dimensions);
    std::vector<float> vectors(count * dimensions, 1.0F);
    for (uint64_t point = 0; point < count; ++point) {
        const float perturbation = static_cast<float>(point % 97) * 1e-7F;
        vectors[point * dimensions + point % dimensions] += perturbation;
    }
    for (uint64_t point = count * 9 / 10; point < count; ++point) {
        vectors[point * dimensions] += static_cast<float>(point) * 0.01F;
    }

    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, max_degree);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 32;
    parameter.min_leaf_size = 2;
    parameter.leader_sample_rate = 0.001F;
    parameter.fanout = {2, 1};
    parameter.reservoir_size = 16;
    vsag::PiPNNGraphBuilder(
        parameter, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(graph, ids, rows);

    RequireGraphInvariants(graph, ids, max_degree, common_param.allocator_.get());
}

TEST_CASE("PiPNN registers exact duplicate rows instead of building duplicate vertices",
          "[ut][pipnn][duplicate]") {
    constexpr uint64_t dimensions = 4;
    constexpr uint64_t count = 6;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    std::copy(vectors.begin() + dimensions,
              vectors.begin() + 2 * dimensions,
              vectors.begin() + 4 * dimensions);
    std::copy(vectors.begin() + dimensions,
              vectors.begin() + 2 * dimensions,
              vectors.begin() + 5 * dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, 4, true);

    vsag::PiPNNGraphBuilder({}, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(graph, ids, rows);

    REQUIRE(graph->GetGroupId(4) == 1);
    REQUIRE(graph->GetGroupId(5) == 1);
    auto duplicates = graph->GetDuplicateIds(1);
    std::sort(duplicates.begin(), duplicates.end());
    REQUIRE(duplicates == std::vector<vsag::InnerIdType>{4, 5});
    REQUIRE(graph->TotalCount() == 4);
}

TEST_CASE("PiPNN graph builder validates structural parameters", "[ut][pipnn]") {
    vsag::PiPNNGraphBuilderParameter parameter;
    REQUIRE_NOTHROW(parameter.Validate(32));

    SECTION("leaf size below two") {
        parameter.max_leaf_size = 1;
        parameter.min_leaf_size = 1;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero min leaf size") {
        parameter.min_leaf_size = 0;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("min leaf size exceeds max") {
        parameter.min_leaf_size = parameter.max_leaf_size + 1;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("empty fanout") {
        parameter.fanout.clear();
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero leaf neighbor count") {
        parameter.leaf_neighbor_count = 0;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("hash count uses the coincident marker bit") {
        parameter.hash_plane_count = 16;
        REQUIRE_THROWS(parameter.Validate(32));
    }
    SECTION("zero graph degree") {
        REQUIRE_THROWS(parameter.Validate(0));
    }
}

TEST_CASE("PiPNN graph builder applies all supported metrics", "[ut][pipnn][metric]") {
    constexpr uint64_t dimensions = 2;
    constexpr uint64_t count = 3;
    const std::vector<float> vectors = {1.0F, 0.0F, 2.0F, 0.0F, 0.8F, 0.6F};

    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    auto common_param = MakeCommonParam(dimensions, metric);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    ids = {0, 1, 2};
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, 1);

    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.leaf_neighbor_count = count - 1;
    vsag::PiPNNGraphBuilder(parameter, dimensions, metric, common_param.allocator_.get())
        .Build(graph, ids, rows);

    vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors.size() == 1);
    const auto expected = metric == vsag::MetricType::METRIC_TYPE_L2SQR ? 2 : 1;
    REQUIRE(neighbors.front() == expected);
}

TEST_CASE("PiPNN builds a connected L2 graph with a sane average degree", "[ut][pipnn][l2]") {
    // Guards the squared-L2 branch of `distance_from_dot`, which converts a dot product with
    // `norms[lhs] + norms[rhs] - 2 * dot`. If that term is lost (observed with optimized builds
    // on some toolchains, where the `2.0F` constant did not survive the BLAS calls it was live
    // across), every pair distance collapses to `norms[lhs] + norms[rhs]`, leaf candidates
    // degenerate to id-adjacent ties, and the graph shatters into tiny components with an
    // average degree of exactly one. Keep this check structural so it also fails in optimized
    // builds, not only in debug.
    constexpr uint64_t dimensions = 128;
    constexpr uint64_t count = 600;
    constexpr uint64_t max_degree = 32;
    auto common_param = MakeCommonParam(dimensions);
    auto vectors = MakeVectors(count, dimensions);
    vsag::Vector<vsag::InnerIdType> ids(common_param.allocator_.get());
    ids.reserve(count);
    for (uint64_t id = 0; id < count; ++id) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(id));
    }
    auto rows = MakeRows(vectors, ids, dimensions, common_param.allocator_.get());
    auto graph = MakeGraph(common_param, count, max_degree);
    vsag::PiPNNGraphBuilder({}, dimensions, common_param.metric_, common_param.allocator_.get())
        .Build(graph, ids, rows);

    uint64_t degree_sum = 0;
    uint64_t isolated = 0;
    for (const auto id : ids) {
        const auto size = graph->GetNeighborSize(id);
        degree_sum += size;
        isolated += size == 0 ? 1 : 0;
    }
    REQUIRE(isolated == 0);
    REQUIRE(degree_sum >= 2 * count);

    std::vector<char> visited(count, 0);
    std::vector<vsag::InnerIdType> stack{ids.front()};
    visited[ids.front()] = 1;
    uint64_t reached = 0;
    while (not stack.empty()) {
        const auto current = stack.back();
        stack.pop_back();
        ++reached;
        vsag::Vector<vsag::InnerIdType> neighbors(common_param.allocator_.get());
        graph->GetNeighbors(current, neighbors);
        for (const auto neighbor : neighbors) {
            if (visited[neighbor] == 0) {
                visited[neighbor] = 1;
                stack.emplace_back(neighbor);
            }
        }
    }
    REQUIRE(reached == count);
}

TEST_CASE("PiPNN pruning scratch does not retain neighbors from another source",
          "[ut][pipnn][pipnn_adaptive]") {
    const uint64_t count = GENERATE(1, 2, 65, 129);
    const bool adaptive = GENERATE(false, true);
    const uint64_t threads = GENERATE(1, 4);
    CAPTURE(count, adaptive, threads);
    auto common = MakeCommonParam(4);
    std::vector<float> vectors(count * 4, 1.0F);
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    vsag::Vector<const float*> rows(common.allocator_.get());
    for (uint64_t point = 0; point < count; ++point) {
        ids.emplace_back(static_cast<vsag::InnerIdType>((count - point) * 3));
        rows.emplace_back(vectors.data() + point * 4);
    }
    auto graph = MakeGraph(common, count * 3 + 1, 8);
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 256;
    parameter.leaf_neighbor_count = 255;
    parameter.reservoir_size = 256;
    parameter.adaptive_pruning = adaptive;
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(threads);
    vsag::PiPNNGraphBuilder(
        parameter, 4, common.metric_, common.allocator_.get(), pool.get(), threads)
        .Build(graph, ids, rows);
    for (const auto source : ids) {
        vsag::Vector<vsag::InnerIdType> expected(common.allocator_.get());
        for (const auto target : ids) {
            if (target != source) {
                expected.emplace_back(target);
            }
        }
        // Coincident points have zero distance, so robust pruning retains the first
        // max_degree candidates in distance / ID-gap / ID order.
        std::sort(expected.begin(), expected.end(), [source](auto lhs, auto rhs) {
            const auto gap = [source](auto target) {
                return source > target ? source - target : target - source;
            };
            return gap(lhs) != gap(rhs) ? gap(lhs) < gap(rhs) : lhs < rhs;
        });
        expected.resize(std::min<uint64_t>(8, expected.size()));
        vsag::Vector<vsag::InnerIdType> actual(common.allocator_.get());
        graph->GetNeighbors(source, actual);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("PiPNN small leaf candidates match exhaustive directed selection", "[ut][pipnn]") {
    const auto metric =
        GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR, vsag::MetricType::METRIC_TYPE_IP);
    // Exercise partial/full 32-candidate prefilter blocks, including the K=5 seed offset.
    const uint64_t count =
        GENERATE(2, 6, 7, 17, 18, 31, 32, 33, 36, 37, 38, 63, 64, 65, 96, 127, 128, 129);
    const uint64_t requested = GENERATE(1, 2, 3, 4, 5, 10, 15, 16, 17);
    const bool coincident = GENERATE(false, true);
    CAPTURE(metric, count, requested, coincident);
    auto common = MakeCommonParam(1, metric);
    auto* allocator = common.allocator_.get();
    std::vector<float> values(count);
    vsag::Vector<vsag::InnerIdType> ids(allocator);
    vsag::Vector<const float*> rows(allocator);
    for (uint64_t i = 0; i < count; ++i) {
        values[i] = coincident ? 1.0F : static_cast<float>(static_cast<int>(i * 7 % 23) - 11);
        ids.emplace_back(static_cast<vsag::InnerIdType>((count - i) * 3));
        rows.emplace_back(&values[i]);
    }
    const auto distance = [&](uint64_t lhs, uint64_t rhs) {
        const float delta = values[lhs] - values[rhs];
        return metric == vsag::MetricType::METRIC_TYPE_L2SQR ? delta * delta
                                                             : 1.0F - values[lhs] * values[rhs];
    };
    const auto gap = [&](uint64_t source, uint64_t target) {
        return ids[source] > ids[target] ? ids[source] - ids[target] : ids[target] - ids[source];
    };
    // Exhaustive directed full sorting is independent of the symmetric TopK scan.
    std::vector<std::vector<bool>> proposed(count, std::vector<bool>(count));
    for (uint64_t source = 0; source < count; ++source) {
        std::vector<uint64_t> targets;
        for (uint64_t target = 0; target < count; ++target) {
            if (source != target) {
                targets.emplace_back(target);
            }
        }
        std::sort(targets.begin(), targets.end(), [&](auto lhs, auto rhs) {
            return std::make_tuple(distance(source, lhs), gap(source, lhs), ids[lhs]) <
                   std::make_tuple(distance(source, rhs), gap(source, rhs), ids[rhs]);
        });
        for (uint64_t selected = 0; selected < std::min(requested, count - 1); ++selected) {
            const auto target = targets[selected];
            proposed[source][target] = proposed[target][source] = true;
        }
    }
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 256;
    parameter.min_leaf_size = 1;
    parameter.leaf_neighbor_count = requested;
    parameter.reservoir_size = count;
    parameter.alpha = 1.2F;
    auto graph = MakeGraph(common, count * 3 + 1, 8);
    vsag::PiPNNGraphBuilder(parameter, 1, metric, allocator).Build(graph, ids, rows);
    for (uint64_t source = 0; source < count; ++source) {
        const auto quantized = [&](uint64_t target) {
            return vsag::generic::BF16ToFloat(vsag::generic::FloatToBF16(distance(source, target)));
        };
        // In one dimension, all non-coincident relative projections form exactly
        // two hash buckets (left/right), independent of hyperplane signs. Equal
        // coordinates carry the coincident flag and are retained individually.
        std::vector<uint64_t> candidates;
        for (const int direction : {-1, 0, 1}) {
            std::vector<uint64_t> bucket;
            for (uint64_t target = 0; target < count; ++target) {
                const int side =
                    (values[target] > values[source]) - (values[target] < values[source]);
                if (proposed[source][target] && side == direction) {
                    bucket.emplace_back(target);
                }
            }
            if (direction != 0 && !bucket.empty()) {
                const auto best =
                    *std::min_element(bucket.begin(), bucket.end(), [&](auto a, auto b) {
                        return std::make_pair(quantized(a), ids[a]) <
                               std::make_pair(quantized(b), ids[b]);
                    });
                bucket = {best};
            }
            candidates.insert(candidates.end(), bucket.begin(), bucket.end());
        }
        std::sort(candidates.begin(), candidates.end(), [&](auto lhs, auto rhs) {
            return std::make_tuple(quantized(lhs), gap(source, lhs), ids[lhs]) <
                   std::make_tuple(quantized(rhs), gap(source, rhs), ids[rhs]);
        });
        std::vector<uint64_t> selected;
        vsag::Vector<vsag::InnerIdType> expected(allocator), actual(allocator);
        for (const auto candidate : candidates) {
            if (std::none_of(selected.begin(), selected.end(), [&](auto neighbor) {
                    return parameter.alpha * distance(neighbor, candidate) < quantized(candidate);
                })) {
                selected.emplace_back(candidate);
                expected.emplace_back(ids[candidate]);
                if (expected.size() == 8) {
                    break;
                }
            }
        }
        graph->GetNeighbors(ids[source], actual);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("PiPNN partition TopK preserves serial and parallel graphs", "[ut][pipnn]") {
    const uint64_t count = GENERATE(512, 4096);
    const uint64_t fanout = GENERATE(1, 2, 10, 17);
    CAPTURE(count, fanout);
    constexpr uint64_t dimensions = 8;
    auto common = MakeCommonParam(dimensions);
    auto* allocator = common.allocator_.get();
    std::vector<float> vectors((count * 2 + 1) * dimensions);
    std::mt19937 random(123);
    for (auto& value : vectors) {
        value = static_cast<float>(static_cast<int>(random() % 64) - 32);
    }
    vsag::Vector<vsag::InnerIdType> ids(allocator);
    for (uint64_t point = 0; point < count; ++point) {
        ids.emplace_back(static_cast<vsag::InnerIdType>((count - point) * 2));
    }
    auto rows = MakeRows(vectors, ids, dimensions, allocator);
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = 256;
    parameter.min_leaf_size = 1;
    parameter.leader_sample_rate = 0.125F;
    parameter.fanout = {fanout, 1};
    parameter.reservoir_size = 16;
    auto serial = MakeGraph(common, count * 2 + 1, 8);
    auto parallel = MakeGraph(common, count * 2 + 1, 8);
    vsag::PiPNNGraphBuilder(parameter, dimensions, common.metric_, allocator)
        .Build(serial, ids, rows);
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(12);
    vsag::PiPNNGraphBuilder(parameter, dimensions, common.metric_, allocator, pool.get(), 12)
        .Build(parallel, ids, rows);
    // min_leaf_size=1 intentionally permits singleton leaves with empty rows.
    RequireGraphInvariants(parallel, ids, 8, allocator, false);
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> expected(allocator), actual(allocator);
        serial->GetNeighbors(id, expected);
        parallel->GetNeighbors(id, actual);
        REQUIRE(actual == expected);
    }
}

namespace {
uint16_t
EncodeHalf(float value, vsag::pipnn_half::Format format) {
    return format == vsag::pipnn_half::Format::FP16 ? vsag::generic::FloatToFP16(value)
                                                    : vsag::generic::FloatToBF16(value);
}

// Independent reference: generic::FP16ToFloat does not decode IEEE subnormals.
double
ReferenceHalf(uint16_t value, vsag::pipnn_half::Format format) {
    if (format == vsag::pipnn_half::Format::FP16) {
        return vsag::half_input::FP16ToFloat(value);
    }
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float decoded;
    std::memcpy(&decoded, &bits, sizeof(decoded));
    return decoded;
}

uint32_t
FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void
RequireHalfDot(float actual,
               const uint16_t* a,
               const uint16_t* b,
               uint64_t dim,
               vsag::pipnn_half::Format format) {
    double expected = 0.0, magnitude = 0.0;
    for (uint64_t k = 0; k < dim; ++k) {
        const double product = ReferenceHalf(a[k], format) * ReferenceHalf(b[k], format);
        expected += product;
        magnitude += std::abs(product);
    }
    // SIMD/FMA reduction order need not match double scalar accumulation.
    REQUIRE(std::abs(static_cast<double>(actual) - expected) <= 2e-5 * std::max(1.0, magnitude));
}
}  // namespace

TEST_CASE("PiPNN half decode preserves IEEE bits including exceptional values",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const std::vector<std::pair<uint16_t, uint32_t>> fp16 = {{0x0000, 0x00000000},
                                                             {0x8000, 0x80000000},
                                                             {0x0001, 0x33800000},
                                                             {0x8001, 0xb3800000},
                                                             {0x03ff, 0x387fc000},
                                                             {0x0400, 0x38800000},
                                                             {0x3c00, 0x3f800000},
                                                             {0x7bff, 0x477fe000},
                                                             {0x7c00, 0x7f800000},
                                                             {0xfc00, 0xff800000},
                                                             {0x7e01, 0x7fc02000},
                                                             {0xfe01, 0xffc02000},
                                                             {0x7c01, 0x7f802000}};
    for (const auto& [input, expected] : fp16) {
        CAPTURE(input);
        REQUIRE(FloatBits(vsag::pipnn_half::Decode(input, Format::FP16)) == expected);
    }
    for (const uint16_t input : {0x0000,
                                 0x8000,
                                 0x0001,
                                 0x8001,
                                 0x007f,
                                 0x0080,
                                 0x3f80,
                                 0x7f7f,
                                 0x7f80,
                                 0xff80,
                                 0x7fc1,
                                 0xffc1,
                                 0x7f81}) {
        CAPTURE(input);
        REQUIRE(FloatBits(vsag::pipnn_half::Decode(input, Format::BF16)) ==
                (static_cast<uint32_t>(input) << 16));
    }
}

TEST_CASE("PiPNN bulk half decode matches scalar bits for every raw pattern",
          "[ut][pipnn][half][decode_row]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    constexpr uint64_t count = 65536;
    std::vector<uint16_t> input(count + 1);
    std::vector<float> actual(count + 2, -123.0F);
    std::vector<float> scalar(count + 2, -123.0F);
    // Offset by one element to exercise non-SIMD-aligned input and output.
    for (uint64_t k = 0; k < count; ++k) {
        input[k + 1] = static_cast<uint16_t>(k);
    }
    vsag::pipnn_half::DecodeRow(input.data() + 1, actual.data() + 1, count, format);
    if (format == Format::FP16) {
        vsag::pipnn_half::detail::DecodeRowScalar<Format::FP16>(
            input.data() + 1, scalar.data() + 1, count);
    } else {
        vsag::pipnn_half::detail::DecodeRowScalar<Format::BF16>(
            input.data() + 1, scalar.data() + 1, count);
    }
    for (uint64_t k = 0; k < count; ++k) {
        CAPTURE(k, static_cast<int>(format));
        const auto expected = FloatBits(vsag::pipnn_half::Decode(input[k + 1], format));
        REQUIRE(FloatBits(actual[k + 1]) == expected);
        REQUIRE(FloatBits(scalar[k + 1]) == expected);
    }
    REQUIRE(actual.front() == -123.0F);
    REQUIRE(actual.back() == -123.0F);
    REQUIRE(scalar.front() == -123.0F);
    REQUIRE(scalar.back() == -123.0F);
}

TEST_CASE("PiPNN bulk half decode respects unaligned row bounds and every SIMD tail",
          "[ut][pipnn][half][decode_row]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t offset = GENERATE(1, 3, 7);
    // Sizes straddle multiple vector boundaries, with every possible tail length.
    for (uint64_t dim = 0; dim <= 33; ++dim) {
        CAPTURE(dim, offset, static_cast<int>(format));
        // Input ends exactly at the row end: ASan can detect tail overreads.
        std::vector<uint16_t> input(offset + dim);
        std::vector<float> output(offset + dim + 1, -123.0F);
        const uint16_t patterns[] = {0x0000,
                                     0x8000,
                                     0x0001,
                                     0x8001,
                                     0x03ff,
                                     0x0400,
                                     0x7bff,
                                     0x7c00,
                                     0xfc00,
                                     0x7c01,
                                     0xfc01,
                                     0x7e01,
                                     0x7f80,
                                     0xff80,
                                     0x7f81,
                                     0xff81,
                                     0x007f,
                                     0x0080};
        // Rotate exceptional values through each vector and scalar-tail position.
        for (uint64_t rotation = 0; rotation < 18; ++rotation) {
            CAPTURE(rotation);
            for (uint64_t k = 0; k < dim; ++k) {
                input[offset + k] = patterns[(k + rotation) % 18];
            }
            vsag::pipnn_half::DecodeRow(input.data() + offset, output.data() + offset, dim, format);
            for (uint64_t k = 0; k < dim; ++k) {
                REQUIRE(FloatBits(output[offset + k]) ==
                        FloatBits(vsag::pipnn_half::Decode(input[offset + k], format)));
            }
            for (uint64_t k = 0; k < offset; ++k) {
                REQUIRE(output[k] == -123.0F);
            }
            REQUIRE(output.back() == -123.0F);
        }
    }
    vsag::pipnn_half::DecodeRow(nullptr, nullptr, 0, format);
}

TEST_CASE("PiPNN half decode preserves sNaN bits without raising invalid",
          "[ut][pipnn][half][decode_row]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t dim = GENERATE(1, 7, 8, 9, 15, 16, 17);
    const uint16_t snan = format == Format::FP16 ? 0x7c01 : 0x7f81;
    const uint16_t negative_snan = format == Format::FP16 ? 0xfc01 : 0xff81;
    for (uint64_t position = 0; position < dim; ++position) {
        CAPTURE(dim, position, static_cast<int>(format));
        std::vector<uint16_t> input(dim + 1, 0);
        std::vector<float> scalar(dim + 2, -123.0F);
        std::vector<float> dispatched(dim + 2, -123.0F);
        input[position + 1] = position % 2 == 0 ? snan : negative_snan;
        std::fenv_t saved;
        const int held = std::feholdexcept(&saved);
        REQUIRE(held == 0);
        if (format == Format::FP16) {
            vsag::pipnn_half::detail::DecodeRowScalar<Format::FP16>(
                input.data() + 1, scalar.data() + 1, dim);
        } else {
            vsag::pipnn_half::detail::DecodeRowScalar<Format::BF16>(
                input.data() + 1, scalar.data() + 1, dim);
        }
        const int scalar_invalid = std::fetestexcept(FE_INVALID);
        std::feclearexcept(FE_ALL_EXCEPT);
        vsag::pipnn_half::DecodeRow(input.data() + 1, dispatched.data() + 1, dim, format);
        const int dispatch_invalid = std::fetestexcept(FE_INVALID);
        const int restored = std::fesetenv(&saved);
        REQUIRE(restored == 0);
        REQUIRE(scalar_invalid == 0);
        REQUIRE(dispatch_invalid == 0);
        for (uint64_t k = 0; k < dim; ++k) {
            REQUIRE(FloatBits(dispatched[k + 1]) == FloatBits(scalar[k + 1]));
        }
        REQUIRE(FloatBits(dispatched[position + 1]) ==
                (format == Format::FP16
                     ? (static_cast<uint32_t>(input[position + 1] & 0x8000U) << 16) | 0x7f802000U
                     : static_cast<uint32_t>(input[position + 1]) << 16));
        REQUIRE(dispatched.front() == -123.0F);
        REQUIRE(dispatched.back() == -123.0F);
    }
}

TEST_CASE("PiPNN half preparation scalar and dispatch agree at vector boundaries",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t dim = GENERATE(0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33);
    CAPTURE(dim, static_cast<int>(format));
    std::vector<uint16_t> input(dim + 1);
    std::vector<float> plane(dim + 1);
    for (uint64_t k = 0; k < dim; ++k) {
        input[k + 1] = EncodeHalf(static_cast<float>(static_cast<int>(k % 9) - 4) / 4.0F, format);
        plane[k + 1] = static_cast<float>(static_cast<int>(k % 7) - 3) / 8.0F;
    }
    for (const bool square : {false, true}) {
        CAPTURE(square);
        const float* b = square ? nullptr : plane.data() + 1;
        const float scalar =
            vsag::pipnn_half::detail::PreparationScalar(input.data() + 1, b, dim, format);
        const float dispatched = vsag::pipnn_half::Preparation(input.data() + 1, b, dim, format);
        REQUIRE(FloatBits(dispatched) == FloatBits(scalar));
    }
}

TEST_CASE("PiPNN preparation matches reference for varied finite inputs",
          "[ut][pipnn][half][preparation]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    std::mt19937 rng(0x19ac723u);
    // Non-power-of-two FP32 planes and mixed signs exercise rounding and cancellation.
    std::uniform_real_distribution<float> half_values(-32.0F, 32.0F);
    std::uniform_real_distribution<float> plane_values(-8.0F, 8.0F);
    for (uint64_t dim = 0; dim <= 127; ++dim) {
        std::vector<uint16_t> input(dim + 1);
        std::vector<float> plane(dim + 1);
        for (uint64_t i = 0; i < dim; ++i) {
            input[i + 1] = EncodeHalf(half_values(rng), format);
            plane[i + 1] = plane_values(rng);
        }
        for (const bool square : {false, true}) {
            CAPTURE(dim, square, static_cast<int>(format));
            const float* b = square ? nullptr : plane.data() + 1;
            double reference = 0.0;
            double magnitude = 0.0;
            for (uint64_t i = 0; i < dim; ++i) {
                const double value = ReferenceHalf(input[i + 1], format);
                const double product = value * (square ? value : static_cast<double>(b[i]));
                reference += product;
                magnitude += std::abs(product);
            }
            // Absolute products give a useful bound even when projections cancel.
            // Allow for rounded products and up to dim additions in either reduction.
            const double tolerance = (dim + 2) * std::numeric_limits<float>::epsilon() * magnitude;
            const float scalar =
                vsag::pipnn_half::detail::PreparationScalar(input.data() + 1, b, dim, format);
            const float dispatched =
                vsag::pipnn_half::Preparation(input.data() + 1, b, dim, format);
            CAPTURE(reference, magnitude, tolerance, scalar, dispatched);
            REQUIRE(std::abs(static_cast<double>(scalar) - reference) <= tolerance);
            REQUIRE(std::abs(static_cast<double>(dispatched) - reference) <= tolerance);
        }
    }
}

TEST_CASE("PiPNN half vector arithmetic handles unaligned rows and SIMD tails",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t dim = GENERATE(0, 1, 3, 4, 5, 7, 8, 9, 17, 127, 512);
    const uint64_t m = GENERATE(1, 3, 5);
    const uint64_t n = GENERATE(1, 3, 5);
    CAPTURE(format, dim, m, n);
    const uint64_t as = dim + 3, bs = dim + 5;
    // One-element offsets exercise unaligned loads; nonzero padding poisons overreads.
    std::vector<uint16_t> a(1 + m * as, EncodeHalf(13.0F, format));
    std::vector<uint16_t> b(1 + n * bs, EncodeHalf(-11.0F, format));
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t k = 0; k < dim; ++k) {
            a[1 + i * as + k] = EncodeHalf(
                static_cast<float>(static_cast<int>((i * 13 + k * 7) % 41) - 20) / 13.0F, format);
        }
    }
    for (uint64_t j = 0; j < n; ++j) {
        for (uint64_t k = 0; k < dim; ++k) {
            b[1 + j * bs + k] = EncodeHalf(
                static_cast<float>(static_cast<int>((j * 17 + k * 11) % 47) - 23) / 17.0F, format);
        }
    }
    const auto original_a = a, original_b = b;
    for (uint64_t i = 0; i < m; ++i) {
        for (uint64_t j = 0; j < n; ++j) {
            const auto* ar = a.data() + 1 + i * as;
            const auto* br = b.data() + 1 + j * bs;
            RequireHalfDot(vsag::pipnn_half::Dot(ar, br, dim, format), ar, br, dim, format);
        }
        const auto* ar = a.data() + 1 + i * as;
        RequireHalfDot(
            vsag::pipnn_half::Preparation(ar, nullptr, dim, format), ar, ar, dim, format);
    }
    REQUIRE(a == original_a);
    REQUIRE(b == original_b);
    REQUIRE(vsag::pipnn_half::Dot(nullptr, nullptr, 0, format) == 0.0F);
    REQUIRE(vsag::pipnn_half::Preparation(nullptr, nullptr, 0, format) == 0.0F);
}

TEST_CASE("PiPNN mixed preparation retains unquantized FP32 projection planes",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t dim = GENERATE(0, 1, 3, 4, 5, 7, 8, 9, 17, 127, 512);
    CAPTURE(format, dim);
    std::vector<uint16_t> input(dim + 1, EncodeHalf(1.0F, format));
    std::vector<float> plane(dim + 1, 1.00313F);
    double expected = 0.0, quantized = 0.0;
    for (uint64_t k = 0; k < dim; ++k) {
        expected += ReferenceHalf(input[k], format) * static_cast<double>(plane[k]);
        quantized +=
            ReferenceHalf(input[k], format) * ReferenceHalf(EncodeHalf(plane[k], format), format);
    }
    const float actual = vsag::pipnn_half::Preparation(input.data(), plane.data(), dim, format);
    REQUIRE(std::abs(static_cast<double>(actual) - expected) <= 2e-6 * std::max(1.0, expected));
    if (dim != 0) {
        REQUIRE(std::abs(expected - quantized) > 1e-4 * dim);
        REQUIRE(std::abs(static_cast<double>(actual) - expected) <
                std::abs(static_cast<double>(actual) - quantized));
    }
}

TEST_CASE("PiPNN borrowed half builder preserves metric nearest neighbors",
          "[ut][pipnn][half][metric]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    auto common = MakeCommonParam(2, metric);
    common.data_type_ =
        format == Format::FP16 ? vsag::DataTypes::DATA_TYPE_FP16 : vsag::DataTypes::DATA_TYPE_BF16;
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    ids = {11, 3, 19};
    // Row positions match the subset sequence, not these sparse nonmonotonic IDs.
    const std::vector<float> values = {1.0F, 0.0F, 2.0F, 0.0F, 0.8F, 0.6F};
    std::vector<uint16_t> codes;
    for (const auto value : values) {
        codes.emplace_back(EncodeHalf(value, format));
    }
    vsag::Vector<const uint16_t*> rows(common.allocator_.get());
    for (uint64_t i = 0; i < ids.size(); ++i) {
        rows.emplace_back(codes.data() + i * 2);
    }
    auto graph = MakeGraph(common, 20, 1);
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.leaf_neighbor_count = 2;
    vsag::PiPNNGraphBuilder(parameter, 2, metric, common.allocator_.get())
        .Build(graph, ids, rows, common.data_type_);
    RequireGraphInvariants(graph, ids, 1, common.allocator_.get());
    vsag::Vector<vsag::InnerIdType> neighbors(common.allocator_.get());
    graph->GetNeighbors(ids.front(), neighbors);
    REQUIRE(neighbors.front() == (metric == vsag::MetricType::METRIC_TYPE_L2SQR ? ids[2] : ids[1]));
}

TEST_CASE("PiPNN borrowed half builder covers small and normal leaves with subset rows",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    const uint64_t requested = GENERATE(5, 17);
    const bool partition = GENERATE(false, true);
    CAPTURE(format, metric, requested, partition);
    constexpr uint64_t dim = 17, count = 65, max_degree = 12;
    auto common = MakeCommonParam(dim, metric);
    auto* allocator = common.allocator_.get();
    common.data_type_ =
        format == Format::FP16 ? vsag::DataTypes::DATA_TYPE_FP16 : vsag::DataTypes::DATA_TYPE_BF16;
    const auto input = MakeVectors(count + 7, dim);
    std::vector<uint16_t> codes((count + 7) * (dim + 3), EncodeHalf(23.0F, format));
    std::vector<float> decoded(count * dim);
    vsag::Vector<vsag::InnerIdType> ids(allocator);
    vsag::Vector<const uint16_t*> half_rows(allocator);
    vsag::Vector<const float*> float_rows(allocator);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t storage_row = (i * 23 + 5) % (count + 7);
        ids.emplace_back(static_cast<vsag::InnerIdType>((count - i) * 3));
        auto* row = codes.data() + storage_row * (dim + 3) + 1;
        for (uint64_t k = 0; k < dim; ++k) {
            row[k] = EncodeHalf(input[storage_row * dim + k], format);
            decoded[i * dim + k] = static_cast<float>(ReferenceHalf(row[k], format));
        }
        half_rows.emplace_back(row);
        float_rows.emplace_back(decoded.data() + i * dim);
    }
    const auto original_codes = codes;
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.max_leaf_size = partition ? 24 : 128;
    parameter.min_leaf_size = 4;
    parameter.leader_sample_rate = 0.25F;
    parameter.fanout = {2, 1};
    parameter.leaf_neighbor_count = requested;
    parameter.reservoir_size = 64;
    parameter.alpha = 1.2F;
    auto native = MakeGraph(common, count * 3 + 1, max_degree);
    auto reference = MakeGraph(common, count * 3 + 1, max_degree);
    vsag::PiPNNGraphBuilder builder(parameter, dim, metric, allocator);
    builder.Build(native, ids, half_rows, common.data_type_);
    builder.Build(reference, ids, float_rows);
    RequireGraphInvariants(native, ids, max_degree, allocator);
    REQUIRE(codes == original_codes);
    const auto distance = [&](uint64_t lhs, uint64_t rhs) {
        double dot = 0.0, l2 = 0.0, an = 0.0, bn = 0.0;
        for (uint64_t k = 0; k < dim; ++k) {
            const double a = decoded[lhs * dim + k], b = decoded[rhs * dim + k];
            dot += a * b;
            l2 += (a - b) * (a - b);
            an += a * a;
            bn += b * b;
        }
        if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            return l2;
        }
        return metric == vsag::MetricType::METRIC_TYPE_COSINE ? 1.0 - dot / std::sqrt(an * bn)
                                                              : 1.0 - dot;
    };
    const auto quality = [&](const vsag::GraphInterfacePtr& graph) {
        double total = 0.0;
        for (uint64_t i = 0; i < count; ++i) {
            vsag::Vector<vsag::InnerIdType> neighbors(allocator);
            graph->GetNeighbors(ids[i], neighbors);
            REQUIRE_FALSE(neighbors.empty());
            double best = std::numeric_limits<double>::max();
            for (const auto neighbor : neighbors) {
                const auto position = std::find(ids.begin(), ids.end(), neighbor);
                REQUIRE(position != ids.end());
                best = std::min(best, distance(i, static_cast<uint64_t>(position - ids.begin())));
            }
            total += best;
        }
        return total / count;
    };
    // BLAS and native SIMD reducers can choose different near-tied candidates.
    // Compare geometric quality rather than exact floating-point adjacency.
    const double expected = quality(reference);
    REQUIRE(quality(native) <= expected + 0.1 * (1.0 + std::abs(expected)));
}

TEST_CASE("PiPNN borrowed half duplicate mapping uses first bitwise representative",
          "[ut][pipnn][half][duplicate]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    constexpr uint64_t dim = 9;
    auto common = MakeCommonParam(dim);
    common.data_type_ =
        format == Format::FP16 ? vsag::DataTypes::DATA_TYPE_FP16 : vsag::DataTypes::DATA_TYPE_BF16;
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    ids = {17, 3, 29, 8, 23, 11};
    const auto input = MakeVectors(ids.size(), dim);
    std::vector<uint16_t> codes(ids.size() * dim);
    for (uint64_t i = 0; i < codes.size(); ++i) {
        codes[i] = EncodeHalf(input[i], format);
    }
    std::copy_n(codes.data() + dim, dim, codes.data() + 4 * dim);
    std::copy_n(codes.data() + dim, dim, codes.data() + 5 * dim);
    // Numerically identical rows with opposite signed-zero bits are NOT exact
    // code duplicates. Deduplication must compare original half bytes.
    codes[2 * dim] = 0x0000;
    std::copy_n(codes.data() + 2 * dim, dim, codes.data() + 3 * dim);
    codes[3 * dim] = 0x8000;
    vsag::Vector<const uint16_t*> rows(common.allocator_.get());
    for (uint64_t i = 0; i < ids.size(); ++i) {
        rows.emplace_back(codes.data() + i * dim);
    }
    auto graph = MakeGraph(common, 30, 4, true);
    vsag::PiPNNGraphBuilder({}, dim, common.metric_, common.allocator_.get())
        .Build(graph, ids, rows, common.data_type_);
    REQUIRE(graph->GetGroupId(ids[4]) == ids[1]);
    REQUIRE(graph->GetGroupId(ids[5]) == ids[1]);
    auto duplicates = graph->GetDuplicateIds(ids[1]);
    std::sort(duplicates.begin(), duplicates.end());
    REQUIRE(duplicates == std::vector<vsag::InnerIdType>{11, 23});
    // TotalCount is the highest populated internal ID plus one, not a live-node count.
    REQUIRE(graph->TotalCount() == 30);
    REQUIRE(graph->GetGroupId(ids[2]) != graph->GetGroupId(ids[3]));
    vsag::Vector<vsag::InnerIdType> representatives(common.allocator_.get());
    representatives = {17, 3, 29, 8};
    RequireGraphInvariants(graph, representatives, 4, common.allocator_.get());
}

TEST_CASE("PiPNN half arithmetic decodes finite subnormal operands in vector and tail lanes",
          "[ut][pipnn][half]") {
    using vsag::pipnn_half::Format;
    const auto format = GENERATE(Format::FP16, Format::BF16);
    const uint64_t dim = GENERATE(1, 7, 8, 9, 17);
    CAPTURE(format, dim);
    // Scaling by the largest finite code makes a lost subnormal observable with
    // an absolute tolerance, without overflowing any FP32 product or reduction.
    std::vector<uint16_t> a(dim, 0x0001);
    std::vector<uint16_t> b(dim, format == Format::FP16 ? 0x7bff : 0x7f7f);
    const double expected = dim * ReferenceHalf(a[0], format) * ReferenceHalf(b[0], format);
    const auto require_close = [&](float actual) {
        REQUIRE(std::abs(static_cast<double>(actual) - expected) <= expected * 2e-5);
    };
    require_close(vsag::pipnn_half::Dot(a.data(), b.data(), dim, format));
    // The projection plane stays FP32, including large values outside half range.
    std::vector<float> plane(dim, static_cast<float>(ReferenceHalf(b[0], format)));
    require_close(vsag::pipnn_half::Preparation(a.data(), plane.data(), dim, format));
}

TEST_CASE("PiPNN typed Build validation parity", "[ut][pipnn][typed_build]") {
    const auto dtype = GENERATE(vsag::DataTypes::DATA_TYPE_FLOAT,
                                vsag::DataTypes::DATA_TYPE_FP16,
                                vsag::DataTypes::DATA_TYPE_BF16);
    const bool support_duplicate = GENERATE(false, true);
    const bool ordered_ids = GENERATE(false, true);
    CAPTURE(dtype, support_duplicate, ordered_ids);
    constexpr uint64_t dim = 4;
    auto common = MakeCommonParam(dim);
    common.data_type_ = dtype;
    auto* allocator = common.allocator_.get();
    vsag::PiPNNGraphBuilder builder({}, dim, common.metric_, allocator);
    // Exact integer values make the same rows representable in all three dtypes.
    const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4};
    std::vector<uint16_t> codes(values.size());
    if (dtype != vsag::DataTypes::DATA_TYPE_FLOAT) {
        const auto format = dtype == vsag::DataTypes::DATA_TYPE_FP16
                                ? vsag::pipnn_half::Format::FP16
                                : vsag::pipnn_half::Format::BF16;
        for (uint64_t i = 0; i < values.size(); ++i) {
            codes[i] = EncodeHalf(values[i], format);
        }
    }
    vsag::Vector<vsag::InnerIdType> ids(allocator);
    if (ordered_ids) {
        ids = {3, 11, 17};
    } else {
        ids = {17, 3, 11};
    }
    vsag::Vector<const float*> float_rows(allocator);
    vsag::Vector<const uint16_t*> half_rows(allocator);
    for (uint64_t i = 0; i < ids.size(); ++i) {
        float_rows.emplace_back(values.data() + i * dim);
        half_rows.emplace_back(codes.data() + i * dim);
    }
    const auto build = [&](const vsag::GraphInterfacePtr& graph) {
        if (dtype == vsag::DataTypes::DATA_TYPE_FLOAT) {
            builder.Build(graph, ids, float_rows);
        } else {
            builder.Build(graph, ids, half_rows, dtype);
        }
    };
    const auto require_error = [&](const vsag::GraphInterfacePtr& graph,
                                   const std::string& message) {
        bool threw = false;
        try {
            build(graph);
        } catch (const vsag::VsagException& error) {
            threw = true;
            REQUIRE(error.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
            REQUIRE(std::string(error.what()).find(message) != std::string::npos);
        }
        REQUIRE(threw);
    };
    auto graph = MakeGraph(common, 18, 4, support_duplicate);
    SECTION("null graph") {
        require_error(nullptr, "PiPNN graph must not be null");
    }
    SECTION("row count mismatch") {
        float_rows.pop_back();
        half_rows.pop_back();
        require_error(graph, "PiPNN IDs and rows must have equal sizes");
    }
    SECTION("duplicate IDs") {
        ids[1] = ids[0];
        require_error(graph, "PiPNN input IDs must be unique");
    }
    SECTION("null row") {
        float_rows[1] = nullptr;
        half_rows[1] = nullptr;
        require_error(graph, "PiPNN input rows must not be null");
    }
    SECTION("ID outside capacity") {
        ids[1] = graph->MaxCapacity();
        require_error(graph, "PiPNN input ID is outside the graph capacity");
    }
    SECTION("noncontiguous positional rows and exact duplicates") {
        build(graph);
        if (support_duplicate) {
            // The tracker canonicalizes a duplicate cycle to its minimum ID,
            // independently of the first positional row used to build the graph.
            const auto group_id = std::min(ids[0], ids[2]);
            REQUIRE(graph->GetGroupId(ids[0]) == group_id);
            REQUIRE(graph->GetGroupId(ids[2]) == group_id);
            REQUIRE(graph->GetGroupId(ids[1]) == ids[1]);
            REQUIRE(graph->GetDuplicateIds(ids[0]) == std::vector<vsag::InnerIdType>{ids[2]});
            REQUIRE(graph->GetDuplicateIds(ids[2]) == std::vector<vsag::InnerIdType>{ids[0]});
            vsag::Vector<vsag::InnerIdType> representatives(allocator);
            representatives = {ids[0], ids[1]};
            RequireGraphInvariants(graph, representatives, 4, allocator);
        } else {
            RequireGraphInvariants(graph, ids, 4, allocator);
        }
    }
    SECTION("empty input") {
        ids.clear();
        float_rows.clear();
        half_rows.clear();
        REQUIRE_NOTHROW(build(graph));
    }
}

TEST_CASE("PiPNN half Build rejects unsupported dtype before graph validation",
          "[ut][pipnn][typed_build]") {
    auto common = MakeCommonParam(4);
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    vsag::Vector<const uint16_t*> rows(common.allocator_.get());
    vsag::PiPNNGraphBuilder builder({}, 4, common.metric_, common.allocator_.get());
    const auto dtype = GENERATE(vsag::DataTypes::DATA_TYPE_FLOAT,
                                vsag::DataTypes::DATA_TYPE_INT8,
                                vsag::DataTypes::DATA_TYPE_SPARSE);
    bool threw = false;
    try {
        builder.Build(nullptr, ids, rows, dtype);
    } catch (const vsag::VsagException& error) {
        threw = true;
        REQUIRE(error.error_.type == vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(std::string(error.what()).find("PiPNN native half rows require FP16 or BF16") !=
                std::string::npos);
    }
    REQUIRE(threw);
}

TEST_CASE("PiPNN final graph reuse matches full pruning with stable graph IDs",
          "[ut][pipnn_cache]") {
    const auto mode = GENERATE(0, 1, 2, 3);
    const auto metric = mode == 1   ? vsag::MetricType::METRIC_TYPE_IP
                        : mode == 2 ? vsag::MetricType::METRIC_TYPE_COSINE
                                    : vsag::MetricType::METRIC_TYPE_L2SQR;
    auto common = MakeCommonParam(8, metric);
    auto* allocator = common.allocator_.get();
    auto vectors = MakeVectors(160, 8);
    std::mt19937 generator(42);
    std::normal_distribution<float> distribution;
    for (auto& value : vectors) {
        value = distribution(generator);
    }
    vsag::Vector<vsag::InnerIdType> original_ids(allocator);
    std::vector<std::string> original_sources;
    for (uint32_t id = 0; id < 128; ++id) {
        original_ids.push_back(id);
        original_sources.push_back("source-" + std::to_string(id));
    }
    auto original_rows = MakeRows(vectors, original_ids, 8, allocator);
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.adaptive_pruning = mode == 3;
    parameter.reservoir_size = 32;
    parameter.leaf_neighbor_count = 16;
    parameter.min_leaf_size = 1;
    vsag::PiPNNGraphBuilder builder(parameter, 8, metric, allocator);
    auto original = MakeGraph(common, 160, 8);
    std::shared_ptr<vsag::PiPNNBuildCache> snapshot;
    builder.Build(original, original_ids, original_rows, &original_sources, nullptr, &snapshot);
    REQUIRE(snapshot != nullptr);
    REQUIRE(snapshot->targets.get_allocator().allocator_ == allocator);
    REQUIRE(snapshot->distances.get_allocator().allocator_ == allocator);
    REQUIRE(snapshot->fingerprints.get_allocator().allocator_ == allocator);
    REQUIRE(snapshot->source_ids.get_allocator().allocator_ == allocator);
    vsag::Vector<vsag::InnerIdType> ids(allocator);
    std::vector<std::string> sources;
    for (uint32_t id = 0; id < 128; ++id) {
        if (id % 20 != 0) {
            ids.push_back(id);
        }
    }
    for (uint32_t id = 128; ids.size() < 128; ++id) {
        ids.push_back(id);
    }
    for (const auto id : ids) {
        sources.push_back("source-" + std::to_string(id));
    }
    auto rows = MakeRows(vectors, ids, 8, allocator);
    auto reused = MakeGraph(common, 160, 8);
    auto reference = MakeGraph(common, 160, 8);
    std::shared_ptr<vsag::PiPNNBuildCache> result;
    builder.Build(reused, ids, rows, &sources, snapshot.get(), &result);
    if (parameter.adaptive_pruning) {
        REQUIRE(result->reused_graph_rows == 0);
    } else {
        REQUIRE(result->reused_graph_rows > 0);
    }
    snapshot->reuse_final_graph = false;
    builder.Build(reference, ids, rows, &sources, snapshot.get());
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> actual(allocator), expected(allocator);
        reused->GetNeighbors(id, actual);
        reference->GetNeighbors(id, expected);
        REQUIRE(actual == expected);
    }
    vsag::Vector<vsag::InnerIdType> remapped(allocator);
    for (uint32_t i = 0; i < ids.size(); ++i) {
        remapped.push_back(static_cast<uint32_t>(ids.size()) - i - 1);
    }
    auto remapped_graph = MakeGraph(common, 160, 8);
    auto remapped_reference = MakeGraph(common, 160, 8);
    std::shared_ptr<vsag::PiPNNBuildCache> third;
    builder.Build(remapped_graph, remapped, rows, &sources, result.get(), &third);
    result->reuse_final_graph = false;
    builder.Build(remapped_reference, remapped, rows, &sources, result.get());
    for (const auto id : remapped) {
        vsag::Vector<vsag::InnerIdType> actual(allocator), expected(allocator);
        remapped_graph->GetNeighbors(id, actual);
        remapped_reference->GetNeighbors(id, expected);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("PiPNN Add matches brute force on its persisted overlapping leaves", "[ut][pipnn][add]") {
    constexpr uint64_t dim = 8, total = 220;
    auto common = MakeCommonParam(dim);
    auto* alloc = common.allocator_.get();
    const bool sparse = GENERATE(false, true);
    const auto physical = [sparse](uint32_t i) -> uint32_t {
        return sparse ? 2 * (total - 1 - i) + 1 : i;
    };
    auto graph = MakeGraph(common, 2 * total, 8);
    bool appended = false, replaced = false, rejected = false, filtered = false;
    vsag::PiPNNGraphBuilderParameter p;
    p.max_leaf_size = 40;
    p.min_leaf_size = 10;
    p.fanout = {2, 2};
    p.leader_sample_rate = 0.12F;
    p.reservoir_size = GENERATE(8U, 16U);
    p.hash_plane_count = 8;
    p.alpha = 1.2F;
    auto values = MakeVectors(total, dim);
    std::mt19937 generator(291);
    for (auto& x : values) x = static_cast<float>(generator() % 6);
    auto distance = [&](uint32_t a, uint32_t b) {
        float d = 0;
        for (uint64_t j = 0; j < dim; ++j) {
            const auto diff = values[a * dim + j] - values[b * dim + j];
            d += diff * diff;
        }
        return d;
    };
    vsag::PiPNNIncrementalState state;
    // Parallel leaf replay can deliver both ends of the same pair from different leaves.
    // Compare every graph row with the exhaustive oracle for both schedules.
    const auto workers = GENERATE(1U, 4U);
    CAPTURE(workers);
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(workers);
    vsag::PiPNNGraphBuilder builder(p, dim, common.metric_, alloc, pool.get(), workers);
    auto hash = [&](uint32_t u, uint32_t v) {
        unsigned bits = 0;
        for (uint64_t j = 0; j < p.hash_plane_count; ++j) {
            bits |= (state.projections.Row(v)[j] >= state.projections.Row(u)[j]) << j;
        }
        return bits;
    };
    std::vector<std::vector<uint32_t>> prior(total);
    for (auto end : {139U, 140U, 179U, 220U}) {
        auto start = static_cast<uint32_t>(state.ids.size());
        vsag::Vector<vsag::InnerIdType> ids(end - start, 0, alloc);
        vsag::Vector<const float*> rows(alloc);
        for (auto i = start; i < end; ++i) {
            ids[i - start] = physical(i);
            rows.push_back(values.data() + i * dim);
        }
        builder.Update(graph, ids, rows, state);
        std::vector<std::set<uint32_t>> candidates(end);
        auto order = [&](uint32_t u, uint32_t v) {
            return std::make_tuple(
                distance(u, v),
                std::abs(static_cast<int>(physical(u)) - static_cast<int>(physical(v))),
                physical(v));
        };
        for (const auto& leaf : state.leaves) {
            REQUIRE(leaf.points.size() <= p.max_leaf_size);
            for (auto u : leaf.points) {
                auto neighbors = leaf.points;
                neighbors.erase(std::find(neighbors.begin(), neighbors.end(), u));
                std::sort(neighbors.begin(), neighbors.end(), [&](auto a, auto b) {
                    return order(u, a) < order(u, b);
                });
                neighbors.resize(std::min<uint64_t>(p.leaf_neighbor_count, neighbors.size()));
                for (auto v : neighbors) {
                    candidates[u].insert(v);
                    candidates[v].insert(u);
                }
            }
        }
        for (uint32_t u = 0; u < end; ++u) {
            if (u < start) {
                auto& eligible = candidates[u];
                eligible.erase(eligible.begin(), eligible.lower_bound(start));
                eligible.insert(prior[u].begin(), prior[u].end());
            }
            std::map<unsigned, uint32_t> best;
            for (auto v : candidates[u]) {
                auto key = hash(u, v);
                auto it = best.find(key);
                if (it == best.end() ||
                    std::make_pair(distance(u, v), physical(v)) <
                        std::make_pair(distance(u, it->second), physical(it->second)))
                    best[key] = v;
            }
            std::vector<uint32_t> pool;
            for (auto [h, v] : best) pool.push_back(v);
            std::sort(pool.begin(), pool.end(), [&](auto a, auto b) {
                return std::make_pair(distance(u, a), physical(a)) <
                       std::make_pair(distance(u, b), physical(b));
            });
            pool.resize(std::min<uint64_t>(pool.size(), p.reservoir_size));
            std::vector<uint32_t> expected;
            if (u < start) {
                expected = prior[u];
                for (auto v : candidates[u]) {
                    filtered |= v >= start && std::find(pool.begin(), pool.end(), v) == pool.end();
                }
                for (auto v : pool) {
                    if (v < start)
                        continue;
                    if (expected.size() < 8) {
                        expected.push_back(v);
                        appended = true;
                        continue;
                    }
                    auto priority = [&](uint32_t w) {
                        return std::make_tuple(std::popcount(hash(u, w) ^ hash(u, v)),
                                               -distance(u, w),
                                               -static_cast<int>(physical(w)));
                    };
                    auto rival =
                        std::min_element(expected.begin(), expected.end(), [&](auto a, auto b) {
                            return priority(a) < priority(b);
                        });
                    if (distance(u, v) < distance(u, *rival)) {
                        *rival = v;
                        replaced = true;
                    } else {
                        rejected = true;
                    }
                }
            } else {
                std::sort(pool.begin(), pool.end(), [&](auto a, auto b) {
                    return order(u, a) < order(u, b);
                });
                for (auto v : pool) {
                    if (std::none_of(expected.begin(), expected.end(), [&](auto w) {
                            return p.alpha * distance(w, v) < distance(u, v);
                        }))
                        expected.push_back(v);
                    if (expected.size() == 8)
                        break;
                }
            }
            prior[u] = expected;
            vsag::Vector<vsag::InnerIdType> actual(alloc), expected_ids(alloc);
            graph->GetNeighbors(physical(u), actual);
            for (auto v : expected) expected_ids.push_back(physical(v));
            CAPTURE(start, end, u, sparse);
            REQUIRE(actual == expected_ids);
        }
    }
    REQUIRE(appended);
    REQUIRE(replaced);
    REQUIRE(rejected);
    REQUIRE(filtered);
}

TEST_CASE("PiPNN old-point HashPrune uses only current edges and new candidates",
          "[ut][pipnn][add]") {
    auto common = MakeCommonParam(1);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 5, 3);
    vsag::PiPNNGraphBuilder builder({}, 1, common.metric_, alloc);
    vsag::PiPNNIncrementalState state;
    const float x[] = {0, 1, -3, 2, 4};
    vsag::Vector<vsag::InnerIdType> ids(alloc), neighbors(alloc);
    vsag::Vector<const float*> rows(alloc);
    ids = {0, 1, 2};
    rows = {x, x + 1, x + 2};
    builder.Update(graph, ids, rows, state);

    // Point 1 stays in the leaf cache and is closer than 3 in the same
    // relative-hash bucket. Only point 2 is currently an outgoing neighbor.
    neighbors = {2};
    graph->InsertNeighborsById(0, neighbors);
    ids = {3};
    rows = {x + 3};
    builder.Update(graph, ids, rows, state);
    graph->GetNeighbors(0, neighbors);
    vsag::Vector<vsag::InnerIdType> expected(alloc);
    expected = {2, 3};
    REQUIRE(neighbors == expected);

    // Point 3 now occupies that bucket and blocks the farther point 4,
    // even though there is still a free outgoing slot.
    ids = {4};
    rows = {x + 4};
    builder.Update(graph, ids, rows, state);
    graph->GetNeighbors(0, neighbors);
    REQUIRE(neighbors == expected);
}

TEST_CASE("PiPNN old edge survives a tie in the reservoir distance key", "[ut][pipnn][add]") {
    auto common = MakeCommonParam(1);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 20, 1);
    vsag::PiPNNGraphBuilder builder({}, 1, common.metric_, alloc);
    vsag::PiPNNIncrementalState state;
    const float points[] = {0.0F, 1.0004F, 1.0F, 0.5F};
    vsag::Vector<vsag::InnerIdType> ids(alloc), neighbors(alloc);
    vsag::Vector<const float*> rows(alloc);
    ids = {17, 9};
    rows = {points, points + 1};
    builder.Update(graph, ids, rows, state);
    // ID 3 beats 9 in HashPrune's same-bucket distance tie. The incumbent edge to 9
    // still wins the separate one-to-one competition on the equal BF16 keys.
    ids = {3};
    rows = {points + 2};
    builder.Update(graph, ids, rows, state);
    graph->GetNeighbors(17, neighbors);
    REQUIRE(neighbors.size() == 1);
    REQUIRE(neighbors.front() == 9);
    ids = {14};
    rows = {points + 3};
    builder.Update(graph, ids, rows, state);
    graph->GetNeighbors(17, neighbors);
    REQUIRE(neighbors.size() == 1);
    REQUIRE(neighbors.front() == 14);
}

TEST_CASE("PiPNN old-edge competition follows the index metric", "[ut][pipnn][add]") {
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_COSINE,
                                 vsag::MetricType::METRIC_TYPE_IP);
    auto common = MakeCommonParam(2, metric);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 20, 1);
    vsag::PiPNNGraphBuilder builder({}, 2, metric, alloc);
    vsag::PiPNNIncrementalState state;
    const float points[][2] = {{1, 0}, {0, 1}, {1, 0.125F}};
    vsag::Vector<vsag::InnerIdType> ids(alloc), neighbors(alloc);
    vsag::Vector<const float*> rows(alloc);
    ids = {17, 9};
    rows = {points[0], points[1]};
    builder.Update(graph, ids, rows, state);
    ids = {3};
    rows = {points[2]};
    builder.Update(graph, ids, rows, state);
    graph->GetNeighbors(17, neighbors);
    REQUIRE(neighbors.size() == 1);
    REQUIRE(neighbors.front() == 3);
}

TEST_CASE("PiPNN Add keeps old FP32 rows in place and writes the version 1 payload",
          "[ut][pipnn][add]") {
    constexpr uint32_t dim = 33, total = 83;
    auto common = MakeCommonParam(dim);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, total, 8);
    vsag::PiPNNIncrementalState state;
    vsag::PiPNNGraphBuilderParameter parameter;
    vsag::PiPNNGraphBuilder builder(parameter, dim, common.metric_, alloc);
    std::mt19937 generator(54);
    std::vector<float> values(total * dim);
    for (auto& v : values) v = static_cast<float>(generator() % 8);
    std::vector<const float*> prior;
    for (uint32_t end : {7U, 8U, 39U, total}) {
        vsag::Vector<vsag::InnerIdType> ids(alloc);
        vsag::Vector<const float*> input(alloc);
        for (uint32_t i = prior.size(); i < end; ++i) {
            ids.push_back(i);
            input.push_back(values.data() + i * dim);
        }
        builder.Update(graph, ids, input, state);
        for (uint32_t i = 0; i < prior.size(); ++i) REQUIRE(state.GetVector(i) == prior[i]);
        for (uint32_t i = prior.size(); i < end; ++i) prior.push_back(state.GetVector(i));

        REQUIRE(state.leaves.size() == 1);
        const auto& leaf = state.leaves.front();
        const auto k = leaf.nearest.Width();
        for (uint32_t u = 0; u < end; ++u) {
            std::vector<std::tuple<float, uint32_t, uint32_t>> expected;
            for (uint32_t v = 0; v < end; ++v) {
                if (u == v)
                    continue;
                float distance = 0;
                for (uint32_t d = 0; d < dim; ++d) {
                    const auto delta = values[u * dim + d] - values[v * dim + d];
                    distance += delta * delta;
                }
                expected.emplace_back(distance, std::max(u, v) - std::min(u, v), v);
            }
            std::sort(expected.begin(), expected.end());
            for (uint32_t j = 0; j < k; ++j) {
                const auto& actual = leaf.nearest.Row(u)[j];
                REQUIRE(actual.target == std::get<2>(expected[j]));
                REQUIRE(actual.distance == std::get<0>(expected[j]));
            }
        }
    }

    std::stringstream wire;
    vsag::IOStreamWriter writer(wire);
    state.Serialize(writer);
    vsag::IOStreamReader reader(wire);
    std::vector<uint32_t> legacy_ids;
    std::vector<float> legacy_rows;
    vsag::StreamReader::ReadVector(reader, legacy_ids);
    vsag::StreamReader::ReadVector(reader, legacy_rows);
    REQUIRE(legacy_ids == state.ids);
    REQUIRE(legacy_rows == values);

    wire.seekg(0);
    vsag::IOStreamReader input(wire);
    vsag::PiPNNIncrementalState restored;
    restored.Deserialize(
        input, total, dim, parameter.hash_plane_count, graph->MaxCapacity(), alloc);
    const auto* first = restored.GetVector(0);
    restored.ReserveVectors(total * 3, dim, alloc);
    REQUIRE(first == restored.GetVector(0));
    for (uint32_t i = 0; i < total; ++i) {
        REQUIRE(std::memcmp(values.data() + i * dim, restored.GetVector(i), dim * sizeof(float)) ==
                0);
    }
    std::stringstream output;
    vsag::IOStreamWriter restored_writer(output);
    restored.Serialize(restored_writer);
    REQUIRE(output.str() == wire.str());
}

TEST_CASE("PiPNN appends sketches in place and validates sparse IDs after reload",
          "[ut][pipnn][add]") {
    constexpr uint32_t dim = 7, total = 97, capacity = 223;
    auto common = MakeCommonParam(dim);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, capacity, 8);
    vsag::PiPNNGraphBuilderParameter p;
    vsag::PiPNNGraphBuilder builder(p, dim, common.metric_, alloc);
    vsag::PiPNNIncrementalState initial, restored;
    auto vectors = MakeVectors(total, dim);
    auto append = [&](auto& state, uint32_t end) {
        vsag::Vector<uint32_t> ids(alloc);
        vsag::Vector<const float*> rows(alloc);
        for (uint32_t i = state.ids.size(); i < end; ++i) {
            ids.push_back((i * 31) % capacity);
            rows.push_back(vectors.data() + i * dim);
        }
        builder.Update(graph, ids, rows, state);
    };
    auto serialize = [](const auto& state) {
        std::stringstream stream;
        vsag::IOStreamWriter writer(stream);
        state.Serialize(writer);
        return stream.str();
    };
    auto load = [&](const std::string& wire, auto& state) {
        std::stringstream stream(wire);
        vsag::IOStreamReader reader(stream);
        state.Deserialize(reader, total, dim, p.hash_plane_count, capacity, alloc);
    };
    append(initial, 17);
    const auto snapshot = serialize(initial);
    load(snapshot, restored);
    REQUIRE(serialize(restored) == snapshot);
    const auto planes = restored.planes;
    const auto* old_vector = restored.Rows()[3];
    const auto* old_projection = restored.projections.Row(3);
    const auto* old_norm = restored.norms.Row(3);
    const auto expected_norm = *old_norm;
    std::vector<float> expected_projection(old_projection, old_projection + p.hash_plane_count);
    for (uint32_t end : {18U, 34U, total}) {
        append(restored, end);
        REQUIRE(restored.Rows()[3] == old_vector);
        REQUIRE(restored.projections.Row(3) == old_projection);
        REQUIRE(restored.norms.Row(3) == old_norm);
        REQUIRE(*old_norm == expected_norm);
        REQUIRE(std::equal(expected_projection.begin(), expected_projection.end(), old_projection));
        REQUIRE(restored.planes == planes);
        for (uint32_t i = 0; i < end; ++i) REQUIRE(restored.LocalId(restored.ids[i]) == i);
    }
    const auto before = serialize(restored);
    auto reject = [&](std::initializer_list<uint32_t> values, bool null_row = false) {
        vsag::Vector<uint32_t> ids(values, alloc);
        vsag::Vector<const float*> rows(ids.size(), null_row ? nullptr : vectors.data(), alloc);
        REQUIRE_THROWS(builder.Update(graph, ids, rows, restored));
        REQUIRE(serialize(restored) == before);
    };
    reject({31});  // A historical sparse ID.
    reject({222, 222});
    reject({capacity});
    reject({222}, true);
    for (const auto invalid : {uint32_t{0}, capacity}) {
        auto damaged = snapshot;
        std::memcpy(
            damaged.data() + sizeof(uint64_t) + sizeof(uint32_t), &invalid, sizeof(invalid));
        vsag::PiPNNIncrementalState state;
        REQUIRE_THROWS(load(damaged, state));
    }
}

TEST_CASE("PiPNN old row tiles match the dispatched FP32 kernels bit for bit", "[ut][pipnn][add]") {
    const uint64_t dim = GENERATE(959U, 960U, 961U);
    const uint64_t m = GENERATE(3U, 6U, 8U);
    const uint64_t b = GENERATE(1U, 3U, 4U, 7U, 8U);
    const auto n = m + b;
    auto values = MakeVectors(n, dim);
    std::vector<const float*> rows(n);
    std::vector<uint32_t> members(n);
    for (uint64_t i = 0; i < n; ++i) {
        rows[i] = values.data() + i * dim;
        members[i] = static_cast<uint32_t>(n - 1 - i);
    }
    constexpr float sentinel = -123456.0F;
    std::vector<float> dots(n * b, sentinel), expected(n * b, sentinel);
    const auto processed =
        vsag::ComputePiPNNOldRowTiles(rows.data(), members.data(), m, b, dim, dots.data());
    const bool use_tiles = dim == 960 && vsag::FP32ComputeIP == vsag::avx512::FP32ComputeIP &&
                           vsag::FP32ComputeIPBatch4 == vsag::avx512::FP32ComputeIPBatch4;
    REQUIRE(processed == (use_tiles ? m - m % 4 : 0));
    for (uint64_t i = 0; i < processed; ++i) {
        auto* old = rows[members[i]];
        uint64_t j = 0;
        for (; j + 4 <= b; j += 4) {
            float scores[4]{};
            vsag::FP32ComputeIPBatch4(old,
                                      dim,
                                      rows[members[m + j]],
                                      rows[members[m + j + 1]],
                                      rows[members[m + j + 2]],
                                      rows[members[m + j + 3]],
                                      scores[0],
                                      scores[1],
                                      scores[2],
                                      scores[3]);
            for (int t = 0; t < 4; ++t) expected[(j + t) * n + i] = scores[t];
        }
        for (; j < b; ++j)
            expected[j * n + i] = vsag::FP32ComputeIP(old, rows[members[m + j]], dim);
    }
    REQUIRE(std::memcmp(dots.data(), expected.data(), dots.size() * sizeof(float)) == 0);
}

TEST_CASE("PiPNN dispatched relative hashes preserve both directions and coincidence",
          "[ut][pipnn]") {
    const auto hash = vsag::pipnn_filter::GetKernels().hashes;
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (uint64_t planes = 1; planes <= 15; ++planes) {
        const uint16_t active = static_cast<uint16_t>((1U << planes) - 1);
        std::vector<float> a(planes, -0.0F), b(planes, +0.0F);
        a.front() = b.front() = inf;
        const auto equal = hash(a.data(), b.data(), planes);
        REQUIRE(equal.forward == (active | vsag::pipnn_filter::COINCIDENT_HASH_FLAG));
        REQUIRE(equal.reverse == equal.forward);
        for (uint64_t i = 0; i < planes; ++i) {
            CAPTURE(planes, i);
            const auto bit = static_cast<uint16_t>(1U << i);
            a[i] = -inf;
            b[i] = inf;
            auto different = hash(a.data(), b.data(), planes);
            REQUIRE(different.forward == active);
            REQUIRE(different.reverse == (active ^ bit));
            b[i] = nan;
            auto unordered = hash(a.data(), b.data(), planes);
            REQUIRE(unordered.forward == (active ^ bit));
            REQUIRE(unordered.reverse == (active ^ bit));
            b[i] = a[i];
        }
    }
}

TEST_CASE("PiPNN vector kth filter includes ties and masks the last partial group", "[ut][pipnn]") {
    const auto select = vsag::pipnn_filter::GetKernels().kth;
    if (!select) {
        SUCCEED("The generic Add path uses row-wise comparisons");
        return;
    }
    for (uint64_t n : {0U, 1U, 15U, 16U, 17U, 31U, 32U, 33U}) {
        std::vector<float> d(n), limit(n);
        std::vector<uint32_t> rows(n + 1, UINT32_MAX), expected;
        for (uint32_t i = 0; i < n; ++i) {
            limit[i] = i % 5 == 0 ? std::numeric_limits<float>::infinity() : -0.0F;
            d[i] = i % 5 == 0 ? limit[i] : static_cast<float>(i % 3) - 1.0F;
            if (d[i] <= limit[i])
                expected.push_back(i);
        }
        const auto count = select(d.data(), limit.data(), n, rows.data());
        REQUIRE(count == expected.size());
        REQUIRE(std::equal(expected.begin(), expected.end(), rows.begin()));
        REQUIRE(rows[count] == UINT32_MAX);
    }
}

TEST_CASE("PiPNN old and new leaf rows retain exact ranks across vector filter batches",
          "[ut][pipnn][add]") {
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    auto common = MakeCommonParam(1, metric);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 64, 8);
    vsag::PiPNNGraphBuilderParameter p;
    p.min_leaf_size = 5;
    p.leaf_neighbor_count = 3;
    vsag::PiPNNGraphBuilder builder(p, 1, metric, alloc);
    vsag::PiPNNIncrementalState state;
    std::array<float, 41> values{};
    for (uint32_t i = 0; i < values.size(); ++i) values[i] = static_cast<int>(i * 7 % 5) - 2;
    for (const uint32_t end : {15U, 16U, 33U, 41U}) {
        vsag::Vector<vsag::InnerIdType> ids(alloc);
        vsag::Vector<const float*> rows(alloc);
        for (uint32_t i = state.ids.size(); i < end; ++i) {
            ids.push_back(2 + i * 17 % 59);
            rows.push_back(&values[i]);
        }
        builder.Update(graph, ids, rows, state);
        REQUIRE(state.leaves.size() == 1);
        const auto& leaf = state.leaves.front();
        for (uint32_t i = 0; i < end; ++i) {
            const auto u = leaf.points[i];
            std::vector<std::tuple<float, uint32_t, uint32_t>> expected;
            for (uint32_t v = 0; v < end; ++v) {
                if (v == u)
                    continue;
                const float dot = values[u] * values[v];
                float d = 1 - dot;
                if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    const float delta = values[u] - values[v];
                    d = delta * delta;
                } else if (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                    d = 1 - ((dot > 0) - (dot < 0));
                }
                const auto gap =
                    std::max(state.ids[u], state.ids[v]) - std::min(state.ids[u], state.ids[v]);
                expected.emplace_back(d, gap, state.ids[v]);
            }
            std::sort(expected.begin(), expected.end());
            for (uint32_t j = 0; j < p.leaf_neighbor_count; ++j) {
                const auto& actual = leaf.nearest.Row(i)[j];
                REQUIRE(actual.distance == std::get<0>(expected[j]));
                REQUIRE(state.ids[actual.target] == std::get<2>(expected[j]));
            }
        }
    }
}

TEST_CASE("PiPNN ordinary leaf Add keeps ranked rows across ties and width changes",
          "[ut][pipnn][add]") {
    constexpr uint32_t total = 12;
    const std::array<uint32_t, total> physical{40, 80, 90, 56, 60, 62, 70, 41, 39, 10, 38, 42};
    const std::array<float, total> values{0, 1, 1, 1, -1, 1, 1, 1, -1, 10, -1, 0};
    auto common = MakeCommonParam(1);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 96, 8);
    vsag::PiPNNGraphBuilderParameter p;
    p.min_leaf_size = 7;
    p.leaf_neighbor_count = GENERATE(1U, 2U);
    vsag::PiPNNGraphBuilder builder(p, 1, common.metric_, alloc);
    vsag::PiPNNIncrementalState state;
    for (const uint32_t end : {2U, 5U, 7U, 9U, total}) {
        vsag::Vector<vsag::InnerIdType> ids(alloc);
        vsag::Vector<const float*> rows(alloc);
        for (uint32_t i = state.ids.size(); i < end; ++i) {
            ids.push_back(physical[i]);
            rows.push_back(&values[i]);
        }
        builder.Update(graph, ids, rows, state);
        REQUIRE(state.leaves.size() == 1);
        const auto& leaf = state.leaves.front();
        const auto k = std::min<uint64_t>(end - 1, end < 7 ? 4 : p.leaf_neighbor_count);
        REQUIRE(leaf.nearest.Width() == k);
        for (uint32_t i = 0; i < end; ++i) {
            auto u = leaf.points[i];
            std::vector<std::tuple<float, uint32_t, uint32_t>> expected;
            for (uint32_t v = 0; v < end; ++v) {
                if (u == v) {
                    continue;
                }
                const auto delta = values[u] - values[v];
                const auto gap =
                    std::max(physical[u], physical[v]) - std::min(physical[u], physical[v]);
                expected.emplace_back(delta * delta, gap, physical[v]);
            }
            std::sort(expected.begin(), expected.end());
            for (uint32_t j = 0; j < k; ++j) {
                const auto& actual = leaf.nearest.Row(i)[j];
                REQUIRE(actual.distance == std::get<0>(expected[j]));
                REQUIRE(physical.at(actual.target) == std::get<2>(expected[j]));
            }
        }
    }
}

TEST_CASE("PiPNN parallel Add preserves routes across overlapping and replaced leaves",
          "[ut][pipnn][add]") {
    constexpr uint32_t dim = 9, total = 701;
    auto common = MakeCommonParam(dim);
    auto* alloc = common.allocator_.get();
    vsag::PiPNNGraphBuilderParameter p;
    p.max_leaf_size = 32;
    p.min_leaf_size = 12;
    p.leader_sample_rate = 0.12F;
    p.fanout = {2, 2};
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(4);
    vsag::PiPNNGraphBuilder serial(p, dim, common.metric_, alloc);
    vsag::PiPNNGraphBuilder parallel(p, dim, common.metric_, alloc, pool.get(), 4);
    vsag::PiPNNIncrementalState reference, actual;
    auto reference_graph = MakeGraph(common, total, 8);
    auto actual_graph = MakeGraph(common, total, 8);
    std::mt19937 random(49);
    std::vector<float> values(total * dim);
    for (auto& x : values) x = static_cast<float>(random() % 16);
    // Start with a terminal root. Subsequent Adds traverse its replacement
    // and deeper replacements, with singleton and uneven four-way batches.
    bool saw_overlapping_split_parents = false;
    const float* cached_root = nullptr;
    for (const auto end : {20U, 139U, 140U, 663U, total}) {
        vsag::Vector<vsag::InnerIdType> ids(alloc);
        vsag::Vector<const float*> rows(alloc);
        for (uint32_t i = reference.ids.size(); i < end; ++i) {
            ids.push_back(i);
            rows.push_back(values.data() + i * dim);
        }
        const auto old_memberships = reference.memberships;
        serial.Update(reference_graph, ids, rows, reference);
        parallel.Update(actual_graph, ids, rows, actual);
        if (cached_root)
            REQUIRE(actual.root_leader_vectors.data() == cached_root);
        if (!actual.root_leader_vectors.empty())
            cached_root = actual.root_leader_vectors.data();
        for (const auto& hits : old_memberships) {
            const auto replaced = std::count_if(hits.begin(), hits.end(), [&](uint32_t leaf) {
                return !actual.leaves.at(leaf).replacement.children.empty();
            });
            saw_overlapping_split_parents |= replaced >= 2;
        }
        REQUIRE(actual.memberships == reference.memberships);
        std::stringstream a, b;
        vsag::IOStreamWriter wa(a), wb(b);
        actual.Serialize(wa);
        reference.Serialize(wb);
        REQUIRE(a.str() == b.str());
        if (end == 663) {
            REQUIRE(cached_root != nullptr);
            vsag::IOStreamReader reader(a);
            actual.Deserialize(
                reader, end, dim, p.hash_plane_count, actual_graph->MaxCapacity(), alloc);
            REQUIRE(actual.root_leader_vectors.empty());
            cached_root = nullptr;
            for (uint64_t leaf = 0; leaf < actual.leaves.size(); ++leaf) {
                REQUIRE(actual.leaves[leaf].depth == reference.leaves[leaf].depth);
            }
        }
        vsag::Vector<vsag::InnerIdType> x(alloc), y(alloc);
        for (uint32_t i = 0; i < end; ++i) {
            actual_graph->GetNeighbors(i, x);
            reference_graph->GetNeighbors(i, y);
            REQUIRE(x == y);
        }
    }
    REQUIRE(actual.leaves.at(actual.root.leaf).replacement.children.size() > 1);
    REQUIRE(saw_overlapping_split_parents);
    REQUIRE(std::any_of(actual.memberships.begin(), actual.memberships.end(), [](const auto& hits) {
        return hits.size() > 1;
    }));
}

TEST_CASE("PiPNN Add groups both directions by local ID for singleton and bulk batches",
          "[ut][pipnn][add]") {
    constexpr uint32_t base = 1024, additions = 256, total = base + additions;
    const auto physical = [](uint32_t local) { return 2 * (total - local) - 1; };
    auto common = MakeCommonParam(2);
    auto* alloc = common.allocator_.get();
    auto graph = MakeGraph(common, 2 * total, 8);
    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(4);
    vsag::PiPNNGraphBuilderParameter p;
    p.max_leaf_size = total;
    p.min_leaf_size = 1;
    p.leaf_neighbor_count = 1;
    vsag::PiPNNGraphBuilder builder(p, 2, common.metric_, alloc, pool.get(), 4);
    vsag::PiPNNIncrementalState state;
    std::vector<std::array<float, 2>> values(total);
    for (uint32_t i = 0; i < total; ++i) {
        const auto group = i < base ? i / 2 : i - base;
        const float x = 8 * (group % 32);
        const float y = 8 * (group / 32);
        values[i] = {x + (i >= base ? 0.25F : 2.0F * (i % 2)), y};
    }
    // Old points form isolated pairs. Each insertion lies strictly between the two
    // members of a distinct pair, so no other old row should receive a candidate.
    for (const uint32_t end : {base, base + 1, total}) {
        vsag::Vector<vsag::InnerIdType> ids(alloc);
        vsag::Vector<const float*> rows(alloc);
        for (uint32_t i = state.ids.size(); i < end; ++i) {
            ids.push_back(physical(i));
            rows.push_back(values[i].data());
        }
        builder.Update(graph, ids, rows, state);
        vsag::Vector<vsag::InnerIdType> neighbors(alloc);
        for (uint32_t old = 0; old < base; ++old) {
            graph->GetNeighbors(physical(old), neighbors);
            const auto newcomer = base + old / 2;
            if (newcomer < end) {
                REQUIRE(std::find(neighbors.begin(), neighbors.end(), physical(newcomer)) !=
                        neighbors.end());
            } else {
                REQUIRE(neighbors.size() == 1);
                REQUIRE(neighbors.front() == physical(old ^ 1U));
            }
        }
        for (uint32_t i = base; i < end; ++i) {
            graph->GetNeighbors(physical(i), neighbors);
            REQUIRE(std::find(neighbors.begin(), neighbors.end(), physical(2 * (i - base))) !=
                    neighbors.end());
        }
    }
}

TEST_CASE("PiPNN leaf top-k append keeps existing rows in place and writes packed snapshots",
          "[ut][pipnn][add]") {
    constexpr uint32_t width = 3;
    std::vector<vsag::PiPNNLocalNeighbor> flat;
    for (uint32_t i = 0; i < 5 * width; ++i) {
        flat.push_back({i, static_cast<float>(i)});
    }
    auto packed = flat;
    vsag::PiPNNLeafTopK cache(5, std::move(packed));
    std::vector<const vsag::PiPNNLocalNeighbor*> addresses;
    for (uint32_t i = 0; i < 5; ++i) {
        addresses.push_back(cache.Row(i).data());
    }
    for (const uint32_t batch : {1U, 2U, 11U, 1U}) {
        const auto old = addresses.size();
        cache.AppendRows(batch);
        for (uint32_t i = 0; i < old; ++i) {
            REQUIRE(cache.Row(i).data() == addresses[i]);
        }
        for (uint32_t i = old; i < old + batch; ++i) {
            auto row = cache.Row(i);
            addresses.push_back(row.data());
            for (uint32_t j = 0; j < width; ++j) {
                const auto id = i * width + j;
                row[j] = {id, static_cast<float>(id)};
                flat.push_back(row[j]);
            }
        }
        // An incumbent in the original block and another in the first append block can change.
        for (uint32_t i : {1U, 5U}) {
            cache.Row(i)[1].distance += 0.5F;
            flat[i * width + 1].distance += 0.5F;
        }
        std::stringstream actual, expected;
        vsag::IOStreamWriter actual_writer(actual), expected_writer(expected);
        cache.Write(actual_writer);
        vsag::StreamWriter::WriteVector(expected_writer, flat);
        REQUIRE(actual.str() == expected.str());
    }
}
