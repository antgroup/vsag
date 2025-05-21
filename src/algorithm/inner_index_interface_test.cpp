
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

#include "inner_index_interface.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "brute_force.h"
#include "hgraph.h"
#include "safe_allocator.h"

using namespace vsag;

TEST_CASE("Fast Create Index", "[ut][InnerIndexInterface]") {
    IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;

    SECTION("HGraph created with minimal parameters") {
        std::string index_fast_str = "hgraph|100|fp16";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<HGraph*>(index.get()) != nullptr);
    }

    SECTION("HGraph created with optional parameters") {
        std::string index_fast_str = "hgraph|100|sq8|fp32";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<HGraph*>(index.get()) != nullptr);
    }

    SECTION("BruteForce created") {
        std::string index_fast_str = "brute_force|fp32";
        auto index = InnerIndexInterface::FastCreateIndex(index_fast_str, common_param);
        REQUIRE(index != nullptr);
        REQUIRE(dynamic_cast<BruteForce*>(index.get()) != nullptr);
    }

    SECTION("Unsupported index type returns null") {
        std::string index_fast_str = "UNKNOWN|other";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }

    SECTION("Invalid parameter count for HGraph (too few)") {
        std::string index_fast_str = "hgraph|100";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }

    SECTION("Invalid parameter count for BruteForce (too few)") {
        std::string index_fast_str = "bruteforce";
        REQUIRE_THROWS(InnerIndexInterface::FastCreateIndex(index_fast_str, common_param));
    }
}
