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

#include "query_computer_pool.h"

#include <fmt/format.h>

#include <atomic>
#include <memory>
#include <vector>

#include "datacell/flatten_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/common/io_parameter.h"
#include "io/memory_io/memory_io.h"
#include "layout/fixed_layout.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/quantizer_parameter.h"
#include "unittest.h"

namespace {

using TestQuantizer = vsag::FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>;
using TestFlatten = vsag::FlattenDataCell<TestQuantizer, vsag::FixedLayout<vsag::MemoryIO>>;

class CountingFlattenDataCell final : public TestFlatten {
public:
    using TestFlatten::TestFlatten;

    vsag::ComputerInterfacePtr
    FactoryComputer(const void* query) override {
        factory_count.fetch_add(1, std::memory_order_relaxed);
        return TestFlatten::FactoryComputer(query);
    }

    std::atomic<uint64_t> factory_count{0};
};

std::shared_ptr<CountingFlattenDataCell>
MakeCountingFlatten(uint64_t dim) {
    constexpr const char* kParamTemplate = R"({{"type": "{}"}})";
    auto quantizer_param = vsag::QuantizerParameter::GetQuantizerParameterByJson(
        vsag::JsonType::Parse(fmt::format(kParamTemplate, "fp32")));
    auto io_param = vsag::IOParameter::GetIOParameterByJson(
        vsag::JsonType::Parse(fmt::format(kParamTemplate, "memory_io")));

    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::IndexCommonParam common;
    common.dim_ = dim;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common.allocator_ = allocator;

    auto flatten = std::make_shared<CountingFlattenDataCell>(quantizer_param, io_param, common);
    flatten->SetQuantizer(std::make_shared<TestQuantizer>(dim, allocator.get()));
    flatten->SetIO(std::make_shared<vsag::MemoryIO>(allocator.get()));
    return flatten;
}

}  // namespace

TEST_CASE("QueryComputerPool reuses computers by cell identity", "[ut][query_computer_pool]") {
    constexpr uint64_t kDim = 8;
    std::vector<float> query_a(kDim, 1.0F);
    std::vector<float> query_b(kDim, 2.0F);

    SECTION("same cell is prepared once") {
        auto cell = MakeCountingFlatten(kDim);
        vsag::SearchStatistics stats;
        vsag::QueryComputerPool pool(query_a.data(), &stats);

        auto first = pool.Acquire(cell, query_a.data());
        auto second = pool.Acquire(cell, query_a.data());

        REQUIRE(first.computer.get() == second.computer.get());
        REQUIRE(first.owner.get() == cell.get());
        REQUIRE_NOTHROW(first.Validate(cell, query_a.data()));
        REQUIRE(pool.Size() == 1);
        REQUIRE(cell->factory_count.load(std::memory_order_relaxed) == 1);
        REQUIRE(stats.query_computer_count.load(std::memory_order_relaxed) == 1);
    }

    SECTION("different cells with the same model shape stay isolated") {
        auto first_cell = MakeCountingFlatten(kDim);
        auto second_cell = MakeCountingFlatten(kDim);
        vsag::SearchStatistics stats;
        vsag::QueryComputerPool pool(query_a.data(), &stats);

        auto first = pool.Acquire(first_cell, query_a.data());
        auto second = pool.Acquire(second_cell, query_a.data());

        REQUIRE(first.computer.get() != second.computer.get());
        REQUIRE(pool.Size() == 2);
        REQUIRE(first_cell->factory_count.load(std::memory_order_relaxed) == 1);
        REQUIRE(second_cell->factory_count.load(std::memory_order_relaxed) == 1);
        REQUIRE(stats.query_computer_count.load(std::memory_order_relaxed) == 2);
        REQUIRE_THROWS_AS(first.Validate(second_cell, query_a.data()), vsag::VsagException);
        REQUIRE_THROWS_AS(second.Validate(first_cell, query_a.data()), vsag::VsagException);
    }

    SECTION("a pool rejects a different query") {
        auto cell = MakeCountingFlatten(kDim);
        vsag::SearchStatistics stats;
        vsag::QueryComputerPool pool(query_a.data(), &stats);

        REQUIRE_THROWS_AS(pool.Acquire(cell, query_b.data()), vsag::VsagException);
        REQUIRE(pool.Size() == 0);
        REQUIRE(cell->factory_count.load(std::memory_order_relaxed) == 0);
        REQUIRE(stats.query_computer_count.load(std::memory_order_relaxed) == 0);
    }

    SECTION("the pool owns the cell for the computer lifetime") {
        auto cell = MakeCountingFlatten(kDim);
        std::weak_ptr<CountingFlattenDataCell> weak_cell = cell;
        auto pool = std::make_unique<vsag::QueryComputerPool>(query_a.data());
        {
            auto lease = pool->Acquire(cell, query_a.data());
            REQUIRE_NOTHROW(lease.Validate(cell, query_a.data()));
        }

        cell.reset();
        REQUIRE_FALSE(weak_cell.expired());
        pool.reset();
        REQUIRE(weak_cell.expired());
    }

    SECTION("the fixed-capacity pool rejects a fourth cell before construction") {
        auto first_cell = MakeCountingFlatten(kDim);
        auto second_cell = MakeCountingFlatten(kDim);
        auto third_cell = MakeCountingFlatten(kDim);
        auto fourth_cell = MakeCountingFlatten(kDim);
        vsag::SearchStatistics stats;
        vsag::QueryComputerPool pool(query_a.data(), &stats);

        static_cast<void>(pool.Acquire(first_cell, query_a.data()));
        static_cast<void>(pool.Acquire(second_cell, query_a.data()));
        static_cast<void>(pool.Acquire(third_cell, query_a.data()));

        REQUIRE_THROWS_AS(pool.Acquire(fourth_cell, query_a.data()), vsag::VsagException);
        REQUIRE(pool.Size() == 3);
        REQUIRE(fourth_cell->factory_count.load(std::memory_order_relaxed) == 0);
        REQUIRE(stats.query_computer_count.load(std::memory_order_relaxed) == 3);
    }

    SECTION("fallback without a pool records and owns a valid computer") {
        auto cell = MakeCountingFlatten(kDim);
        std::weak_ptr<CountingFlattenDataCell> weak_cell = cell;
        vsag::SearchStatistics stats;
        vsag::QueryContext ctx{.stats = &stats};

        {
            auto lease = vsag::AcquireQueryComputer(cell, query_a.data(), &ctx);
            REQUIRE(ctx.computer_pool == nullptr);
            REQUIRE_NOTHROW(lease.Validate(cell, query_a.data()));
            REQUIRE(lease.computer != nullptr);
            REQUIRE(cell->factory_count.load(std::memory_order_relaxed) == 1);
            REQUIRE(stats.query_computer_count.load(std::memory_order_relaxed) == 1);

            cell.reset();
            REQUIRE_FALSE(weak_cell.expired());
        }
        REQUIRE(weak_cell.expired());
    }
}
