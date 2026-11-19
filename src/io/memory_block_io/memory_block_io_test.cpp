
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

#include "io/memory_block_io/memory_block_io.h"

#include <memory>

#include "impl/allocator/safe_allocator.h"
#include "io/common/io_contract_test.h"
#include "unittest.h"

using namespace vsag;

auto block_memory_io_block_sizes = {1023, 4096, 123123, 1024 * 1024};

TEST_CASE("MemoryBlockIO Read and Write Test", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    REQUIRE_THROWS_AS(MemoryBlockIO(0, allocator.get()), VsagException);
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
    REQUIRE(io->Size() == data.size());

    io->Shrink(5000);
    REQUIRE(io->Size() == 5000);

    std::vector<uint8_t> read_data(5000);
    io->Read(5000, 0, read_data.data());
    REQUIRE(memcmp(read_data.data(), data.data(), 5000) == 0);

    io->Shrink(1000);
    REQUIRE(io->Size() == 1000);

    io->Shrink(2000);
    REQUIRE(io->Size() == 1000);
}

TEST_CASE("MemoryBlockIO blocks start on a cache line boundary", "[ut][MemoryBlockIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    for (const uint64_t block_size : {1024ULL, 4096ULL, 1024ULL * 1024ULL}) {
        constexpr uint64_t block_count = 3;
        auto io = std::make_unique<MemoryBlockIO>(block_size, allocator.get());
        std::vector<uint8_t> data(block_count * block_size);
        for (uint64_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<uint8_t>(i % 251);
        }
        io->Write(data.data(), data.size(), 0);

        for (uint64_t block = 0; block < block_count; ++block) {
            CAPTURE(block_size, block);
            // A read inside one block borrows the stored bytes, so its address is the block start.
            auto lease = io->Acquire(block * block_size, 64);
            REQUIRE(lease);
            const auto address = reinterpret_cast<uintptr_t>(lease.Data());
            REQUIRE(address % BlockMemoryBackend::kBlockAlignment == 0);
            REQUIRE(memcmp(lease.Data(), data.data() + block * block_size, 64) == 0);
        }

        // Dropping blocks and growing again releases and aligns the right pointers.
        io->Shrink(block_size);
        io->Write(data.data(), data.size(), 0);
        std::vector<uint8_t> read_data(data.size());
        io->Read(read_data.size(), 0, read_data.data());
        REQUIRE(read_data == data);
        auto lease = io->Acquire((block_count - 1) * block_size, 64);
        REQUIRE(lease);
        const auto address = reinterpret_cast<uintptr_t>(lease.Data());
        REQUIRE(address % BlockMemoryBackend::kBlockAlignment == 0);
    }
}
