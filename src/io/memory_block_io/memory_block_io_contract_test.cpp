// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/memory_block_io/memory_block_io.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

class BlockCountingAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "BlockCountingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        if (fail_allocations_) {
            return nullptr;
        }
        allocations_.fetch_add(1, std::memory_order_relaxed);
        return std::malloc(size);
    }

    void
    Deallocate(void* data) override {
        deallocations_.fetch_add(1, std::memory_order_relaxed);
        std::free(data);
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        return std::realloc(data, size);
    }

    void
    Reset() {
        allocations_.store(0, std::memory_order_relaxed);
        deallocations_.store(0, std::memory_order_relaxed);
    }

    void
    SetFailAllocations(bool fail) {
        fail_allocations_ = fail;
    }

    [[nodiscard]] uint64_t
    Allocations() const {
        return allocations_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Deallocations() const {
        return deallocations_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> deallocations_{0};
    bool fail_allocations_{false};
};

void
RequireSameBlockContent(const MemoryBlockIO& compatibility, const MemoryBlockIO& canonical) {
    REQUIRE(compatibility.Size() == canonical.Size());
    std::vector<uint8_t> compatibility_data(compatibility.Size());
    std::vector<uint8_t> canonical_data(canonical.Size());
    REQUIRE(compatibility.Read(compatibility.Size(), 0, compatibility_data.data()));
    REQUIRE(canonical.ReadAt(0, canonical.Size(), canonical_data.data()));
    REQUIRE(compatibility_data == canonical_data);
}

}  // namespace

TEST_CASE("MemoryBlockIO lease ownership and batch contract", "[ut][MemoryBlockIO]") {
    BlockCountingAllocator allocator;
    MemoryBlockIO io(64, &allocator);
    std::array<uint8_t, 192> source{};
    for (uint64_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i * 17);
    }
    io.WriteAt(0, source.data(), source.size());

    allocator.Reset();
    {
        auto lease = io.Acquire(8, 32);
        REQUIRE(lease);
        REQUIRE(lease.Size() == 32);
        REQUIRE(std::memcmp(lease.Data(), source.data() + 8, 32) == 0);
        REQUIRE(allocator.Allocations() == 0);
    }
    REQUIRE(allocator.Deallocations() == 0);

    {
        auto lease = io.Acquire(48, 32);
        REQUIRE(lease);
        REQUIRE(std::memcmp(lease.Data(), source.data() + 48, 32) == 0);
        REQUIRE(allocator.Allocations() == 1);
    }
    REQUIRE(allocator.Deallocations() == 1);

    bool need_release = false;
    const uint8_t* borrowed = io.Read(16, 16, need_release);
    REQUIRE(borrowed != nullptr);
    REQUIRE_FALSE(need_release);
    const uint8_t* owned = io.Read(32, 48, need_release);
    REQUIRE(owned != nullptr);
    REQUIRE(need_release);
    io.Release(owned);

    std::array<uint8_t, 12> first{};
    std::array<uint8_t, 20> second{};
    std::array<ReadRequest, 2> requests{{
        {first.data(), 4, first.size()},
        {second.data(), 58, second.size()},
    }};
    REQUIRE(io.ReadMany(requests.data(), requests.size()));
    REQUIRE(std::memcmp(first.data(), source.data() + 4, first.size()) == 0);
    REQUIRE(std::memcmp(second.data(), source.data() + 58, second.size()) == 0);

    allocator.SetFailAllocations(true);
    REQUIRE_THROWS_AS(io.Acquire(48, 32), VsagException);

    need_release = true;
    const uint8_t* failed_legacy_read = io.Read(32, 48, need_release);
    REQUIRE(failed_legacy_read == nullptr);
    REQUIRE_FALSE(need_release);
    allocator.SetFailAllocations(false);
}

TEST_CASE("MemoryBlockIO randomized differential", "[ut][MemoryBlockIO][differential]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MemoryBlockIO compatibility(256, allocator.get());
    MemoryBlockIO canonical(256, allocator.get());
    std::mt19937_64 random(20260822);

    for (uint64_t step = 0; step < 1200; ++step) {
        uint64_t operation = random() % 4;
        if (operation <= 1) {
            uint64_t offset =
                compatibility.Size() == 0 ? 0 : random() % (compatibility.Size() + 129);
            uint64_t size = random() % 193;
            std::vector<uint8_t> data(size);
            for (auto& value : data) {
                value = static_cast<uint8_t>(random());
            }
            compatibility.Write(data.data(), size, offset);
            canonical.WriteAt(offset, data.data(), size);
        } else if (operation == 2) {
            uint64_t size = random() % 4096;
            compatibility.Resize(size);
            canonical.Resize(size);
        } else {
            uint64_t size = compatibility.Size() == 0 ? 0 : random() % (compatibility.Size() + 1);
            compatibility.Shrink(size);
            canonical.Shrink(size);
        }
        RequireSameBlockContent(compatibility, canonical);
    }
}

TEST_CASE("MemoryBlockIO serialization is cross compatible", "[ut][MemoryBlockIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint8_t> data(4097);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 29);
    }

    MemoryBlockIO compatibility_source(256, allocator.get());
    MemoryBlockIO canonical_destination(256, allocator.get());
    compatibility_source.Write(data.data(), data.size(), 0);
    std::stringstream old_stream;
    IOStreamWriter old_writer(old_stream);
    compatibility_source.Serialize(old_writer);
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireSameBlockContent(compatibility_source, canonical_destination);

    MemoryBlockIO canonical_source(256, allocator.get());
    MemoryBlockIO compatibility_destination(256, allocator.get());
    canonical_source.WriteAt(0, data.data(), data.size());
    std::stringstream new_stream;
    IOStreamWriter new_writer(new_stream);
    canonical_source.Serialize(new_writer);
    IOStreamReader new_reader(new_stream);
    compatibility_destination.Deserialize(new_reader);
    RequireSameBlockContent(compatibility_destination, canonical_source);
}

}  // namespace vsag
