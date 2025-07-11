
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

#include "safe_allocator.h"

#include <catch2/catch_test_macros.hpp>

#include "allocator_wrapper.h"

TEST_CASE("SafeAllocator Basic Test", "[ut][SafeAllocator]") {
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    REQUIRE(allocator->Name() == "DefaultAllocator_safewrapper");
    auto allocator2 = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::AllocatorWrapper<int> allocator_wrapper1(allocator.get());
    vsag::AllocatorWrapper<int> allocator_wrapper2(allocator.get());
    vsag::AllocatorWrapper<int> allocator_wrapper3(allocator2.get());
    REQUIRE(allocator_wrapper1 == allocator_wrapper2);
    REQUIRE(allocator_wrapper1 != allocator_wrapper3);
}
