
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

#include "iterator_filter.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "vsag/iterator_context.h"

using namespace vsag;

TEST_CASE("Iterator context", "[ut][hnsw]") {
    auto allocator = std::make_shared<DefaultAllocator>();
    vsag::IteratorFilterContext filter_context = IteratorFilterContext();
    uint32_t max_size = 1000;
    int64_t ef_search = 200;
    const int64_t num_elements = 100;
    int64_t dim = 1;
    auto [base_ids, base_distance] = fixtures::generate_ids_and_vectors(num_elements, dim);
    uint32_t res_top_id = 0;
    float res_top_dis = 0.0;

    SECTION("class IteratorContext") {
        auto res = filter_context.init(max_size, ef_search, allocator.get());
        REQUIRE(res.has_value());
        if (filter_context.IsFirstUsed()) {
            filter_context.SetOFFFirstUsed();
        }
        REQUIRE(filter_context.Empty() == true);
        for (int64_t i = 0; i < num_elements; i++) {
            filter_context.AddDiscardNode(base_distance[i], base_ids[i]);
            if (i % 2 == 0) {
                filter_context.SetPoint(base_ids[i]);
            }
        }
        REQUIRE(filter_context.Empty() == false);
        REQUIRE(filter_context.GetDiscardElementNum() == num_elements);
        uint32_t res_top_id1 = filter_context.GetTopID();
        float res_top_dis1 = filter_context.GetTopDist();
        filter_context.PopDiscard();
        uint32_t res_top_id2 = filter_context.GetTopID();
        float res_top_dis2 = filter_context.GetTopDist();
        REQUIRE(res_top_dis1 >= res_top_dis2);
        REQUIRE(filter_context.GetDiscardElementNum() == num_elements - 1);
        REQUIRE(filter_context.CheckPoint(55));
    }
}
