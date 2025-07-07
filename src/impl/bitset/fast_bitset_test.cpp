
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

#include "fast_bitset.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "fixtures.h"
#include "utils/util_functions.h"

using namespace vsag;

std::pair<FastBitsetPtr, std::vector<int>>
GetRandomBitset(Allocator* allocator, int64_t max_size, int64_t max_element) {
    auto values = select_k_numbers(max_element, max_size);
    auto bitset = std::make_shared<FastBitset>(allocator);
    for (auto& v : values) {
        bitset->Set(v, true);
    }
    return std::make_pair(bitset, values);
}

std::unordered_set<int>
GetIntersection(const std::vector<int>& a, const std::vector<int>& b) {
    std::unordered_set<int> setA(a.begin(), a.end());
    std::unordered_set<int> result;

    for (int val : b) {
        if (setA.count(val)) {
            result.insert(val);
        }
    }

    return result;
}

std::unordered_set<int>
GetIntersection(const std::unordered_set<int>& set) {
    return set;
}

template <typename... Args>
std::unordered_set<int>
GetIntersection(const std::unordered_set<int>& set, const std::vector<int>& vec, Args... args) {
    std::unordered_set<int> result;
    for (auto v : vec) {
        if (set.count(v)) {
            result.insert(v);
        }
    }
    return GetIntersection(result, args...);
}

template <typename... Args>
std::unordered_set<int>
GetIntersection(const std::vector<int>& vec1, const std::vector<int>& vec2, Args... args) {
    auto result = GetIntersection(vec1, vec2);
    return GetIntersection(result, args...);
}

TEST_CASE("FastBitset And operations", "[ut][FastBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("and basic operator with ptr") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset1->And(bitset2);
        auto values = GetIntersection(values1, values2);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("and basic operator with ref") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset1->And(*bitset2);
        auto values = GetIntersection(values1, values2);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("and basic operator with vector") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset3, values3] = GetRandomBitset(allocator.get(), 100, 10000);
        std::vector<ComputableBitsetPtr> bitsets;
        bitsets.emplace_back(bitset2);
        bitsets.emplace_back(bitset3);
        bitset1->And(bitsets);
        auto values = GetIntersection(values1, values2, values3);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values3) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("and basic operator with not") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset2->Not();
        std::unordered_set<int> values_origin(values2.begin(), values2.end());
        std::unordered_set<int> values;
        for (int i = 0; i < 10000; ++i) {
            if (values_origin.find(i) == values_origin.end()) {
                REQUIRE(bitset2->Test(i) == true);
            } else {
                REQUIRE(bitset2->Test(i) == false);
            }
        }
        for (auto& v : values1) {
            if (values_origin.count(v) == 0) {
                values.insert(v);
            }
        }

        bitset1->And(bitset2);

        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }
}

std::unordered_set<int>
GetUnion(const std::vector<int>& values1, const std::vector<int>& values2) {
    // get the union of values1 and values2
    std::unordered_set<int> result(values2.begin(), values2.end());

    for (int value : values1) {
        result.insert(value);
    }
    return result;
}

std::unordered_set<int>
GetUnion(const std::unordered_set<int>& set) {
    return set;
}

template <typename... Args>
std::unordered_set<int>
GetUnion(const std::unordered_set<int>& set, const std::vector<int>& vec, Args... args) {
    std::unordered_set<int> result = set;
    for (auto v : vec) {
        result.insert(v);
    }
    return GetUnion(result, args...);
}

template <typename... Args>
std::unordered_set<int>
GetUnion(const std::vector<int>& vec1, const std::vector<int>& vec2, Args... args) {
    auto result = GetUnion(vec1, vec2);
    return GetUnion(result, args...);
}

TEST_CASE("FastBitset Or operations", "[ut][FastBitset]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();

    SECTION("or basic operator with ptr") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset1->Or(bitset2);
        auto values = GetUnion(values1, values2);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("or basic operator with ref") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset1->Or(*bitset2);
        auto values = GetUnion(values1, values2);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("and basic operator with vector") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset3, values3] = GetRandomBitset(allocator.get(), 100, 10000);
        std::vector<ComputableBitsetPtr> bitsets;
        bitsets.emplace_back(bitset2);
        bitsets.emplace_back(bitset3);
        bitset1->Or(bitsets);
        auto values = GetUnion(values1, values2, values3);
        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values3) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }

    SECTION("and basic operator with not") {
        auto [bitset1, values1] = GetRandomBitset(allocator.get(), 100, 10000);
        auto [bitset2, values2] = GetRandomBitset(allocator.get(), 100, 10000);
        bitset2->Not();
        std::unordered_set<int> values_origin(values2.begin(), values2.end());
        std::unordered_set<int> values;
        for (int i = 0; i < 10000; ++i) {
            if (values_origin.find(i) == values_origin.end()) {
                REQUIRE(bitset2->Test(i) == true);
                values.insert(i);
            } else {
                REQUIRE(bitset2->Test(i) == false);
            }
        }
        for (auto& v : values1) {
            values.insert(v);
        }

        bitset1->Or(bitset2);

        for (auto& v : values1) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
        for (auto& v : values2) {
            if (values.find(v) == values.end()) {
                REQUIRE(bitset1->Test(v) == false);
            } else {
                REQUIRE(bitset1->Test(v) == true);
            }
        }
    }
}
