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

#include "allocator_wrapper.h"

#include <catch2/catch_test_macros.hpp>
#include <vector>

#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("AllocatorWrapper Basic Test", "[ut][AllocatorWrapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AllocatorWrapper<int> wrapper(allocator.get());

    SECTION("Allocate and Deallocate") {
        int* ptr = wrapper.allocate(10);
        REQUIRE(ptr != nullptr);

        for (int i = 0; i < 10; ++i) {
            ptr[i] = i;
        }

        for (int i = 0; i < 10; ++i) {
            REQUIRE(ptr[i] == i);
        }

        wrapper.deallocate(ptr, 10);
    }

    SECTION("Construct and Destroy") {
        int* ptr = wrapper.allocate(1);
        wrapper.construct(ptr, 42);
        REQUIRE(*ptr == 42);
        wrapper.destroy(ptr);
        wrapper.deallocate(ptr, 1);
    }
}

TEST_CASE("AllocatorWrapper STL Container Test", "[ut][AllocatorWrapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AllocatorWrapper<int> wrapper(allocator.get());

    SECTION("Vector with AllocatorWrapper") {
        std::vector<int, AllocatorWrapper<int>> vec(wrapper);
        vec.push_back(1);
        vec.push_back(2);
        vec.push_back(3);

        REQUIRE(vec.size() == 3);
        REQUIRE(vec[0] == 1);
        REQUIRE(vec[1] == 2);
        REQUIRE(vec[2] == 3);
    }
}

TEST_CASE("AllocatorWrapper Equality Test", "[ut][AllocatorWrapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AllocatorWrapper<int> wrapper1(allocator.get());
    AllocatorWrapper<int> wrapper2(allocator.get());
    AllocatorWrapper<int> wrapper3(nullptr);

    REQUIRE(wrapper1 == wrapper2);
    REQUIRE_FALSE(wrapper1 == wrapper3);
    REQUIRE(wrapper1 != wrapper3);
}

TEST_CASE("AllocatorWrapper Rebind Test", "[ut][AllocatorWrapper]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    AllocatorWrapper<int> wrapper_int(allocator.get());

    typename AllocatorWrapper<int>::rebind<double>::other wrapper_double(wrapper_int);
    double* ptr = wrapper_double.allocate(5);
    REQUIRE(ptr != nullptr);
    wrapper_double.construct(ptr, 3.14);
    REQUIRE(*ptr == 3.14);
    wrapper_double.destroy(ptr);
    wrapper_double.deallocate(ptr, 5);
}
