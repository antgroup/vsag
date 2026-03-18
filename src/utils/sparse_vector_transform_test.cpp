
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

#include "sparse_vector_transform.h"

#include <catch2/catch_test_macros.hpp>

#include "impl/allocator/default_allocator.h"

using namespace vsag;

TEST_CASE("SparseVectorTransform Basic", "[ut][SparseVectorTransform]") {
    auto allocator = std::make_shared<DefaultAllocator>();

    SECTION("sort sparse vector by value descending") {
        std::vector<uint32_t> ids{1, 2, 3};
        std::vector<float> vals{0.5F, 1.2F, 0.7F};
        SparseVector sparse{static_cast<uint32_t>(ids.size()), ids.data(), vals.data()};

        Vector<std::pair<uint32_t, float>> sorted(allocator.get());
        sort_sparse_vector(sparse, sorted);
        REQUIRE(sorted.size() == 3);
        REQUIRE(sorted[0].first == 2);
        REQUIRE(sorted[1].first == 3);
        REQUIRE(sorted[2].first == 1);
    }

    SECTION("subset positive and negative cases") {
        std::vector<uint32_t> ids1{1, 2};
        std::vector<float> vals1{0.1F, 0.2F};
        SparseVector sv1{static_cast<uint32_t>(ids1.size()), ids1.data(), vals1.data()};

        std::vector<uint32_t> ids2{1, 2, 3};
        std::vector<float> vals2{0.1F, 0.2F, 0.3F};
        SparseVector sv2{static_cast<uint32_t>(ids2.size()), ids2.data(), vals2.data()};
        REQUIRE(is_subset_of_sparse_vector(sv1, sv2));

        std::vector<uint32_t> ids3{1, 2};
        std::vector<float> vals3{0.1F, 0.25F};
        SparseVector sv3{static_cast<uint32_t>(ids3.size()), ids3.data(), vals3.data()};
        REQUIRE_FALSE(is_subset_of_sparse_vector(sv3, sv2));

        std::vector<uint32_t> ids4{1, 4};
        std::vector<float> vals4{0.1F, 0.2F};
        SparseVector sv4{static_cast<uint32_t>(ids4.size()), ids4.data(), vals4.data()};
        REQUIRE_FALSE(is_subset_of_sparse_vector(sv4, sv2));
    }

    SECTION("empty sparse vector is subset") {
        SparseVector empty{};
        std::vector<uint32_t> ids{1};
        std::vector<float> vals{0.1F};
        SparseVector non_empty{static_cast<uint32_t>(ids.size()), ids.data(), vals.data()};
        REQUIRE(is_subset_of_sparse_vector(empty, non_empty));
    }
}
