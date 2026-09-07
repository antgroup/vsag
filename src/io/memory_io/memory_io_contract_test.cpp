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
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "io/backend/heap_region.h"
#include "io/memory_io/memory_io.h"
#include "storage/serialization.h"
#include "unittest.h"

namespace vsag {
namespace {

class MemoryIOFailAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "MemoryIOFailAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        if (fail_allocate_) {
            return nullptr;
        }
        return std::malloc(size);
    }

    void
    Deallocate(void* data) override {
        std::free(data);
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        ++reallocate_count_;
        if (fail_reallocate_) {
            return nullptr;
        }
        return std::realloc(data, size);
    }

    bool fail_allocate_{false};
    bool fail_reallocate_{false};
    uint64_t reallocate_count_{0};
};

class MemoryIOStableAllocator : public Allocator {
public:
    explicit MemoryIOStableAllocator(uint64_t capacity) : storage_(capacity) {
    }

    std::string
    Name() override {
        return "MemoryIOStableAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        return size <= storage_.size() ? storage_.data() : nullptr;
    }

    void
    Deallocate(void*) override {
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        return data == storage_.data() and size <= storage_.size() ? data : nullptr;
    }

private:
    std::vector<uint8_t> storage_;
};

void
RequireSameContent(const MemoryIO& compatibility, const MemoryIO& canonical) {
    REQUIRE(compatibility.Size() == canonical.Size());
    if (compatibility.Size() == 0) {
        return;
    }
    std::vector<uint8_t> compatibility_data(compatibility.Size());
    std::vector<uint8_t> canonical_data(canonical.Size());
    REQUIRE(compatibility.Read(compatibility.Size(), 0, compatibility_data.data()));
    REQUIRE(canonical.ReadAt(0, canonical.Size(), canonical_data.data()));
    REQUIRE(compatibility_data == canonical_data);
}

}  // namespace

TEST_CASE("HeapRegion tracks its minimum physical allocation", "[ut][MemoryIO][HeapRegion]") {
    MemoryIOFailAllocator allocator;
    HeapRegion region(&allocator);

    REQUIRE(region.Capacity() == 1);
    REQUIRE(allocator.reallocate_count_ == 0);

    region.EnsureCapacity(1);
    REQUIRE(allocator.reallocate_count_ == 0);

    region.EnsureCapacity(8);
    REQUIRE(region.Capacity() == 8);
    REQUIRE(allocator.reallocate_count_ == 1);

    region.ShrinkPhysical(0);
    REQUIRE(region.Capacity() == 1);
    REQUIRE(allocator.reallocate_count_ == 2);

    region.EnsureCapacity(1);
    REQUIRE(allocator.reallocate_count_ == 2);
}

TEST_CASE("MemoryIO canonical zero acquire is empty while compatibility read is borrowed",
          "[ut][MemoryIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MemoryIO io(allocator.get());

    REQUIRE_FALSE(io.Acquire(0, 0));
    bool need_release = true;
    const uint8_t* data = io.Read(0, 0, need_release);
    REQUIRE(data != nullptr);
    REQUIRE_FALSE(need_release);
}

TEST_CASE("MemoryIO contract", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MemoryIO io(allocator.get());

    REQUIRE(MemoryIO::InMemory);
    REQUIRE_FALSE(MemoryIO::SkipDeserialize);
    REQUIRE(io.Size() == 0);
    REQUIRE(io.ReadAt(0, 0, nullptr));
    REQUIRE_FALSE(io.ReadAt(1, 0, nullptr));
    REQUIRE_FALSE(io.ReadAt(0, 1, nullptr));
    REQUIRE(io.ReadMany(nullptr, 0));

    std::array<uint8_t, 8> source{0, 1, 2, 3, 4, 5, 6, 7};
    io.WriteAt(4, source.data(), source.size());
    REQUIRE(io.Size() == 12);

    std::array<uint8_t, 4> first{};
    std::array<uint8_t, 3> second{};
    std::array<ReadRequest, 3> requests{{
        {first.data(), 4, first.size()},
        {nullptr, 12, 0},
        {second.data(), 9, second.size()},
    }};
    REQUIRE(io.ReadMany(requests.data(), requests.size()));
    REQUIRE((first == std::array<uint8_t, 4>{0, 1, 2, 3}));
    REQUIRE((second == std::array<uint8_t, 3>{5, 6, 7}));

    auto operation = io.SubmitReads(requests.data(), requests.size());
    REQUIRE(operation.Poll());
    REQUIRE(operation.Wait());

    auto lease = io.Acquire(5, 4);
    REQUIRE(lease);
    REQUIRE(lease.Size() == 4);
    REQUIRE(std::memcmp(lease.Data(), source.data() + 1, 4) == 0);
    REQUIRE_FALSE(io.Acquire(12, 1));

    bool need_release = true;
    const uint8_t* compatibility_data = io.Read(4, 5, need_release);
    REQUIRE(compatibility_data == lease.Data());
    REQUIRE_FALSE(need_release);
    io.Release(compatibility_data);

    REQUIRE_THROWS_AS(io.WriteAt(std::numeric_limits<uint64_t>::max(), source.data(), 2),
                      VsagException);
}

TEST_CASE("MemoryIO compatibility and scatter batch adapters", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MemoryIO io(allocator.get());
    std::array<uint8_t, 16> source{};
    for (uint64_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>(i);
    }
    io.Write(source.data(), source.size(), 0);

    std::array<uint64_t, 3> offsets{7, 0, 12};
    std::array<uint64_t, 3> sizes{3, 4, 4};
    std::array<uint8_t, 11> contiguous{};
    REQUIRE(io.MultiRead(contiguous.data(), sizes.data(), offsets.data(), offsets.size()));
    REQUIRE((contiguous == std::array<uint8_t, 11>{7, 8, 9, 0, 1, 2, 3, 12, 13, 14, 15}));

    std::array<uint8_t, 4> a{};
    std::array<uint8_t, 3> b{};
    std::array<ReadRequest, 2> scatter{{{a.data(), 8, 4}, {b.data(), 1, 3}}};
    REQUIRE(io.ReadMany(scatter.data(), scatter.size()));
    REQUIRE((a == std::array<uint8_t, 4>{8, 9, 10, 11}));
    REQUIRE((b == std::array<uint8_t, 3>{1, 2, 3}));
}

TEST_CASE("MemoryIO randomized differential", "[ut][MemoryIO][differential]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    MemoryIO compatibility(allocator.get());
    MemoryIO canonical(allocator.get());
    std::mt19937_64 random(20260821);

    for (uint64_t step = 0; step < 2000; ++step) {
        uint64_t operation = random() % 5;
        if (operation <= 1) {
            uint64_t offset = random() % 8192;
            uint64_t size = random() % 257;
            if (offset > compatibility.Size()) {
                std::vector<uint8_t> gap(offset - compatibility.Size(), 0);
                compatibility.Write(gap.data(), gap.size(), compatibility.Size());
                canonical.WriteAt(canonical.Size(), gap.data(), gap.size());
            }
            std::vector<uint8_t> data(size);
            for (auto& value : data) {
                value = static_cast<uint8_t>(random());
            }
            compatibility.Write(data.data(), size, offset);
            canonical.WriteAt(offset, data.data(), size);
        } else if (operation == 2) {
            uint64_t size = random() % 8192;
            uint64_t old_size = compatibility.Size();
            compatibility.Resize(size);
            canonical.Resize(size);
            if (size > old_size) {
                std::vector<uint8_t> initialized(size - old_size, 0);
                compatibility.Write(initialized.data(), initialized.size(), old_size);
                canonical.WriteAt(old_size, initialized.data(), initialized.size());
            }
        } else if (operation == 3) {
            uint64_t size = compatibility.Size() == 0 ? 0 : random() % (compatibility.Size() + 1);
            compatibility.Shrink(size);
            canonical.Shrink(size);
        } else if (compatibility.Size() > 0) {
            uint64_t offset = random() % compatibility.Size();
            uint64_t size = random() % (compatibility.Size() - offset + 1);
            std::vector<uint8_t> compatibility_data(size);
            std::vector<uint8_t> canonical_data(size);
            REQUIRE(compatibility.Read(size, offset, compatibility_data.data()));
            REQUIRE(canonical.ReadAt(offset, size, canonical_data.data()));
            REQUIRE(compatibility_data == canonical_data);
        }
        RequireSameContent(compatibility, canonical);
    }
}

TEST_CASE("MemoryIO serialization is cross compatible", "[ut][MemoryIO][compatibility]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<uint8_t> data(4097);
    for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>(i * 31);
    }

    MemoryIO compatibility_source(allocator.get());
    MemoryIO canonical_destination(allocator.get());
    compatibility_source.Write(data.data(), data.size(), 0);
    std::stringstream old_stream;
    IOStreamWriter old_writer(old_stream);
    compatibility_source.Serialize(old_writer);
    IOStreamReader old_reader(old_stream);
    canonical_destination.Deserialize(old_reader);
    RequireSameContent(compatibility_source, canonical_destination);

    MemoryIO canonical_source(allocator.get());
    MemoryIO compatibility_destination(allocator.get());
    canonical_source.Write(data.data(), data.size(), 0);
    std::stringstream new_stream;
    IOStreamWriter new_writer(new_stream);
    canonical_source.Serialize(new_writer);
    IOStreamReader new_reader(new_stream);
    compatibility_destination.Deserialize(new_reader);
    RequireSameContent(compatibility_destination, canonical_source);
}

TEST_CASE("MemoryIO allocation failures preserve data", "[ut][MemoryIO]") {
    MemoryIOFailAllocator allocator;
    allocator.fail_allocate_ = true;
    REQUIRE_THROWS_AS(MemoryIO(&allocator), VsagException);

    allocator.fail_allocate_ = false;
    MemoryIO io(&allocator);
    std::array<uint8_t, 4> source{1, 2, 3, 4};
    io.WriteAt(0, source.data(), source.size());
    allocator.fail_reallocate_ = true;
    REQUIRE_THROWS_AS(io.WriteAt(1000, source.data(), source.size()), VsagException);

    std::array<uint8_t, 4> result{};
    REQUIRE(io.ReadAt(0, result.size(), result.data()));
    REQUIRE(result == source);
}

TEST_CASE("MemoryIO publishes initialized append bytes to concurrent readers",
          "[ut][MemoryIO][concurrency]") {
    constexpr uint64_t byte_count = 16 * 1024;
    MemoryIOStableAllocator allocator(byte_count);
    MemoryIO io(&allocator);

    std::atomic<bool> success{true};
    std::thread reader([&]() {
        uint64_t observed = 0;
        while (observed < byte_count) {
            const uint64_t published = io.Size();
            while (observed < published) {
                uint8_t value = 0;
                if (not io.ReadAt(observed, 1, &value) or value != static_cast<uint8_t>(observed)) {
                    success.store(false, std::memory_order_relaxed);
                    return;
                }
                ++observed;
            }
            std::this_thread::yield();
        }
    });

    for (uint64_t offset = 0; offset < byte_count; ++offset) {
        const uint8_t value = static_cast<uint8_t>(offset);
        io.WriteAt(offset, &value, 1);
    }
    reader.join();

    REQUIRE(success.load(std::memory_order_relaxed));
    REQUIRE(io.Size() == byte_count);
}

}  // namespace vsag
