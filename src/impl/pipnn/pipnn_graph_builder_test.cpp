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
#include <chrono>
#include <cmath>
#include <future>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "data_type.h"
#include "datacell/graph_interface.h"
#include "datacell/graph_interface_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "row_accessor.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "unittest.h"

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

// Build-time comparison: RowAccessor (fp16 on-the-fly conversion) vs legacy
// Vector<const float*> (pre-converted fp32).  Both paths build from the same
// fp32-represented vectors so the comparison isolates the conversion overhead.

TEST_CASE("PiPNN RowAccessor perf is comparable to fp32 rows path", "[ut][pipnn][pipnn_perf]") {
    constexpr uint64_t N = 10000;
    constexpr uint64_t DIM = 64;
    constexpr uint64_t MAX_DEGREE = 16;
    constexpr uint64_t POINTS_PER_LEAF = 128;
    constexpr uint64_t FANOUT = 8;
    using Clock = std::chrono::high_resolution_clock;

    std::vector<float> fp32_vecs(N * DIM);
    std::vector<uint16_t> fp16_vecs(N * DIM);
    for (uint64_t i = 0; i < N * DIM; ++i) {
        float v = std::sin(static_cast<float>(i)) * 0.5F + static_cast<float>(i % 997) * 0.0001F;
        fp32_vecs[i] = v;
        fp16_vecs[i] = vsag::generic::FloatToFP16(v);
    }

    auto common = MakeCommonParam(DIM, vsag::MetricType::METRIC_TYPE_L2SQR);

    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    ids.reserve(N);
    for (uint64_t i = 0; i < N; ++i) {
        ids.emplace_back(static_cast<vsag::InnerIdType>(i));
    }

    vsag::PiPNNGraphBuilderParameter parameter;
    // max_degree is validated via Validate();
    parameter.max_leaf_size = POINTS_PER_LEAF;
    parameter.fanout = {FANOUT, 1};
    parameter.reservoir_size = 16;

    // Legacy fp32 rows path.
    auto rows_fp32 = MakeRows(fp32_vecs, ids, DIM, common.allocator_.get());
    auto graph_fp32 = MakeGraph(common, N, MAX_DEGREE);
    auto t0 = Clock::now();
    vsag::PiPNNGraphBuilder(parameter, DIM, common.metric_, common.allocator_.get())
        .Build(graph_fp32, ids, rows_fp32);
    auto t1 = Clock::now();

    // RowAccessor fp16.
    vsag::RowAccessor accessor;
    accessor.data = fp16_vecs.data();
    accessor.data_type_ = vsag::DataTypes::DATA_TYPE_FP16;
    accessor.count_ = N;
    accessor.dim_ = DIM;
    auto graph_fp16 = MakeGraph(common, N, MAX_DEGREE);
    auto t2 = Clock::now();
    vsag::PiPNNGraphBuilder(parameter, DIM, common.metric_, common.allocator_.get())
        .Build(graph_fp16, ids, accessor);
    auto t3 = Clock::now();

    // RowAccessor fp32 for overhead measurement.
    vsag::RowAccessor accessor_fp32;
    accessor_fp32.data = fp32_vecs.data();
    accessor_fp32.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    accessor_fp32.count_ = N;
    accessor_fp32.dim_ = DIM;
    auto graph_fp32ra = MakeGraph(common, N, MAX_DEGREE);
    auto t4 = Clock::now();
    vsag::PiPNNGraphBuilder(parameter, DIM, common.metric_, common.allocator_.get())
        .Build(graph_fp32ra, ids, accessor_fp32);
    auto t5 = Clock::now();

    auto fp32_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto fp16_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    auto fp32_ra_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    WARN("PiPNN build @" << N << "x" << DIM << " fp32-rows=" << fp32_ms << "ms  fp16-ra=" << fp16_ms
                         << "ms  fp32-ra=" << fp32_ra_ms << "ms");

    // RowAccessor fp32 overhead must be <=15%.
    auto ra_overhead = fp32_ra_ms / std::max(1.0, fp32_ms);
    WARN("RowAccessor overhead (fp32): " << ra_overhead << "x");
    REQUIRE(ra_overhead <= 1.15);

    // fp16 must not exceed 3x.
    auto fp16_ratio = fp16_ms / std::max(1.0, fp32_ms);
    WARN("fp16 / fp32 ratio: " << fp16_ratio << "x");
    REQUIRE(fp16_ratio <= 3.0);

    // fp16 graph: verify every vertex has at least one neighbor.
    int isolated16 = 0;
    int isolated32 = 0;
    for (uint64_t i = 0; i < N; ++i) {
        vsag::Vector<vsag::InnerIdType> nbr32(common.allocator_.get()),
            nbr16(common.allocator_.get());
        graph_fp32->GetNeighbors(static_cast<vsag::InnerIdType>(i), nbr32);
        graph_fp16->GetNeighbors(static_cast<vsag::InnerIdType>(i), nbr16);
        REQUIRE(nbr32.size() > 0);
        REQUIRE(nbr16.size() > 0);
        isolated16 += nbr16.empty() ? 1 : 0;
        isolated32 += nbr32.empty() ? 1 : 0;
    }
    WARN("fp32 isolated=" << isolated32 << " fp16 isolated=" << isolated16);
    REQUIRE(isolated32 == 0);
    REQUIRE(isolated16 == 0);
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
    constexpr uint64_t dimensions = 8;
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

TEST_CASE("PiPNN half accessor matches represented float rows for subsets",
          "[ut][pipnn][half_accessor_regression]") {
    const bool bf16 = GENERATE(false, true);
    const auto metric = GENERATE(vsag::MetricType::METRIC_TYPE_L2SQR,
                                 vsag::MetricType::METRIC_TYPE_IP,
                                 vsag::MetricType::METRIC_TYPE_COSINE);
    constexpr uint64_t count = 96;
    constexpr uint64_t dim = 16;
    auto common = MakeCommonParam(dim, metric);
    auto source = MakeVectors(count, dim);
    std::vector<uint16_t> half(count * dim);
    std::vector<float> represented(count * dim);
    for (uint64_t i = 0; i < count * dim; ++i) {
        half[i] =
            bf16 ? vsag::generic::FloatToBF16(source[i]) : vsag::generic::FloatToFP16(source[i]);
        represented[i] =
            bf16 ? vsag::generic::BF16ToFloat(half[i]) : vsag::generic::FP16ToFloat(half[i]);
    }
    // Non-contiguous inner IDs and reversed source mapping exercise both mappings.
    std::vector<uint64_t> remap(count);
    std::iota(remap.rbegin(), remap.rend(), uint64_t{0});
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    vsag::Vector<const float*> rows(common.allocator_.get());
    for (uint64_t id = 1; id < count; id += 3) {
        ids.push_back(static_cast<vsag::InnerIdType>(id));
        rows.push_back(represented.data() + remap[id] * dim);
    }
    vsag::RowAccessor accessor;
    accessor.data = half.data();
    accessor.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    accessor.count_ = count;
    accessor.dim_ = dim;
    accessor.remap = remap.data();
    vsag::PiPNNGraphBuilderParameter parameter;
    parameter.min_leaf_size = 8;
    parameter.max_leaf_size = 16;
    parameter.reservoir_size = 16;
    auto expected = MakeGraph(common, count, 8);
    auto actual = MakeGraph(common, count, 8);
    vsag::PiPNNGraphBuilder builder(parameter, dim, metric, common.allocator_.get());
    builder.Build(expected, ids, rows);
    builder.Build(actual, ids, accessor);
    for (const auto id : ids) {
        vsag::Vector<vsag::InnerIdType> expected_neighbors(common.allocator_.get());
        vsag::Vector<vsag::InnerIdType> actual_neighbors(common.allocator_.get());
        expected->GetNeighbors(id, expected_neighbors);
        actual->GetNeighbors(id, actual_neighbors);
        REQUIRE(actual_neighbors == expected_neighbors);
    }
}

TEST_CASE("PiPNN accessor preserves duplicate groups and validates original IDs",
          "[ut][pipnn][half_accessor_regression]") {
    const bool bf16 = GENERATE(false, true);
    constexpr uint64_t count = 12;
    constexpr uint64_t dim = 4;
    auto common = MakeCommonParam(dim, vsag::MetricType::METRIC_TYPE_L2SQR);
    std::vector<uint16_t> half(count * dim);
    std::vector<float> represented(count * dim);
    std::vector<uint64_t> remap(count);
    std::iota(remap.rbegin(), remap.rend(), uint64_t{0});
    for (uint64_t id = 0; id < count; ++id) {
        const float value = (id == 7 || id == 2) ? 1.0F : static_cast<float>(id + 1);
        for (uint64_t d = 0; d < dim; ++d) {
            const auto offset = remap[id] * dim + d;
            half[offset] =
                bf16 ? vsag::generic::FloatToBF16(value) : vsag::generic::FloatToFP16(value);
            represented[offset] = value;
        }
    }
    vsag::Vector<vsag::InnerIdType> ids(common.allocator_.get());
    vsag::Vector<const float*> rows(common.allocator_.get());
    for (const auto id : {7U, 2U, 9U, 4U}) {
        ids.push_back(id);
        rows.push_back(represented.data() + remap[id] * dim);
    }
    vsag::RowAccessor accessor;
    accessor.data = half.data();
    accessor.data_type_ = bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16;
    accessor.count_ = count;
    accessor.dim_ = dim;
    accessor.remap = remap.data();
    vsag::PiPNNGraphBuilderParameter parameter;
    auto expected = MakeGraph(common, count, 8, true);
    auto actual = MakeGraph(common, count, 8, true);
    vsag::PiPNNGraphBuilder builder(
        parameter, dim, vsag::MetricType::METRIC_TYPE_L2SQR, common.allocator_.get());
    builder.Build(expected, ids, rows);
    builder.Build(actual, ids, accessor);
    REQUIRE(actual->GetGroupId(2) == 2);
    REQUIRE(actual->GetGroupId(2) == expected->GetGroupId(2));
    REQUIRE(actual->GetDuplicateIds(7) == expected->GetDuplicateIds(7));
    REQUIRE(actual->TotalCount() == expected->TotalCount());
    accessor.count_ = 4;  // Enough rows by count, but not by sparse input IDs.
    REQUIRE_THROWS(builder.Build(MakeGraph(common, count, 8), ids, accessor));
    accessor.count_ = count;
    accessor.dim_ = dim + 1;
    REQUIRE_THROWS(builder.Build(MakeGraph(common, count, 8), ids, accessor));
    accessor.dim_ = dim;
    accessor.data = nullptr;
    REQUIRE_THROWS(builder.Build(MakeGraph(common, count, 8), ids, accessor));
}
