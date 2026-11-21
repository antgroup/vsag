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
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <set>
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
#include "simd/bf16_simd.h"
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
    const uint64_t count = GENERATE(2, 6, 7, 17, 18, 63, 64, 65, 96, 127, 128, 129);
    const uint64_t requested = GENERATE(1, 5, 16, 17);
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
