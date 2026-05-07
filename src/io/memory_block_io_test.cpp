
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

#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

auto block_memory_io_block_sizes = {1023, 4096, 123123, 1024 * 1024};

TEST_CASE("MemoryBlockIO Read and Write Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto block_size : block_memory_io_block_sizes) {
        auto io = std::make_unique<MemoryBlockIO>(block_size, allocator.get());
        TestBasicReadWrite(*io);
    }
}

TEST_CASE("MemoryBlockIO Serialize and Deserialize Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (auto block_size : block_memory_io_block_sizes) {
        auto wio = std::make_unique<MemoryBlockIO>(block_size, allocator.get());
        auto rio = std::make_unique<MemoryBlockIO>(block_size, allocator.get());
        TestSerializeAndDeserialize(*wio, *rio);
    }
}

TEST_CASE("MemoryBlockIO Shrink Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto block_size = 4096;
    auto io = std::make_unique<MemoryBlockIO>(block_size, allocator.get());

    std::vector<uint8_t> data(10000, 0xAB);
    io->Write(data.data(), data.size(), 0);
    REQUIRE(io->size_ == data.size());

    io->Shrink(5000);
    REQUIRE(io->size_ == 5000);

    std::vector<uint8_t> read_data(5000);
    io->Read(5000, 0, read_data.data());
    REQUIRE(memcmp(read_data.data(), data.data(), 5000) == 0);

    io->Shrink(1000);
    REQUIRE(io->size_ == 1000);

    io->Shrink(2000);
    REQUIRE(io->size_ == 1000);
}
