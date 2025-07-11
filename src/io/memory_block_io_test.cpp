
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

#include "memory_block_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

auto block_memory_io_block_sizes = {1023, 4096, 123123, 1024 * 1024};

TEST_CASE("MemoryBlockIO Read and Write Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto block_size : block_memory_io_block_sizes) {
        auto io = std::make_unique<MemoryBlockIO>(allocator.get(), block_size);
        TestBasicReadWrite(*io);
    }
}

TEST_CASE("MemoryBlockIO Serialize and Deserialize Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto block_size : block_memory_io_block_sizes) {
        auto wio = std::make_unique<MemoryBlockIO>(allocator.get(), block_size);
        auto rio = std::make_unique<MemoryBlockIO>(allocator.get(), block_size);
        TestSerializeAndDeserialize(*wio, *rio);
    }
}
