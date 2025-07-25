
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

#include "memory_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("MemoryIO Read and Write", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<MemoryIO>(allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("MemoryIO Serialize and Deserialize", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto wio = std::make_unique<MemoryIO>(allocator.get());
    auto rio = std::make_unique<MemoryIO>(allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}
