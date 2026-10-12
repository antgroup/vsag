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

#include "aligned_block_layout.h"

#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

namespace vsag {
namespace {
// Deliberately break malloc's alignment and allow allocation failure after a valid reserve.
class UnalignedBlockAllocator : public Allocator {
public:
    void*
    Allocate(uint64_t size) override {
        if (fail or (size >= 4096 and block_allocations_until_failure-- == 0)) {
            throw std::bad_alloc();
        }
        allocated_bytes += size;
        auto* raw = static_cast<uint8_t*>(base_->Allocate(size + 80));
        const auto aligned = (reinterpret_cast<uintptr_t>(raw) + 63) & ~uintptr_t{63};
        auto* result = reinterpret_cast<uint8_t*>(aligned) + 16;
        std::memcpy(result - 16, &raw, sizeof(raw));
        std::memcpy(result - 8, &size, sizeof(size));
        return result;
    }

    void
    Deallocate(void* pointer) override {
        if (pointer != nullptr) {
            uint8_t* raw = nullptr;
            std::memcpy(&raw, static_cast<uint8_t*>(pointer) - 16, sizeof(raw));
            base_->Deallocate(raw);
        }
    }

    void*
    Reallocate(void* pointer, uint64_t size) override {
        auto* result = Allocate(size);
        if (pointer != nullptr) {
            uint64_t old_size = 0;
            std::memcpy(&old_size, static_cast<uint8_t*>(pointer) - 8, sizeof(old_size));
            std::memcpy(result, pointer, std::min(size, old_size));
            Deallocate(pointer);
        }
        return result;
    }

    std::string
    Name() override {
        return "unaligned_block_test";
    }

    bool fail{false};
    uint64_t allocated_bytes{0};
    uint64_t block_allocations_until_failure{std::numeric_limits<uint64_t>::max()};

private:
    std::shared_ptr<Allocator> base_{SafeAllocator::FactoryDefaultAllocator()};
};
}  // namespace

TEST_CASE("Aligned blocks preserve records across growth and wire layouts",
          "[ut][AlignedBlockLayout]") {
    UnalignedBlockAllocator allocator;
    constexpr uint64_t stride = 1792;
    AlignedBlockLayout layout(stride, &allocator, 4096);
    REQUIRE(layout.RecordsPerBlock() == 2);
    layout.Reserve(1);
    auto* first = layout.GetMutableRecord(0);
    std::memset(first, 0xAB, stride);
    for (uint64_t count = 2; count <= 19; ++count) {
        layout.Reserve(count);
        REQUIRE(layout.GetRecord(0) == first);
        REQUIRE(first[stride - 1] == 0xAB);
        auto* record = layout.GetMutableRecord(count - 1);
        REQUIRE(reinterpret_cast<uintptr_t>(record) % 64 == 0);
        for (uint64_t j = 0; j < stride; ++j) {
            REQUIRE(record[j] == 0);
        }
        std::memset(record, count - 1, stride);
    }
    layout.Reserve(0);
    REQUIRE(layout.GetRecord(0) == first);
    REQUIRE(allocator.allocated_bytes < 19 * stride * 2);

    // Packed wire bytes remain independent of physical block size and alignment padding.
    std::stringstream stream;
    IOStreamWriter writer(stream);
    layout.WriteRecords(writer, 19);
    const auto payload = stream.str();
    REQUIRE(payload.size() == 19 * stride);
    for (uint64_t id = 0; id < 19; ++id) {
        REQUIRE(std::memcmp(payload.data() + id * stride, layout.GetRecord(id), stride) == 0);
    }
    AlignedBlockLayout restored(stride, &allocator, 8192);
    restored.Reserve(19);
    IOStreamReader reader(stream);
    restored.ReadRecords(reader, 19);
    for (uint64_t id = 0; id < 19; ++id) {
        REQUIRE(std::memcmp(restored.GetRecord(id), layout.GetRecord(id), stride) == 0);
    }

    allocator.fail = true;
    REQUIRE_THROWS(layout.Reserve(100));
    REQUIRE(layout.GetRecord(0) == first);
    REQUIRE(first[stride - 1] == 0xAB);
    allocator.fail = false;
    // Fail after one new payload block has been allocated. Retry must reuse it
    // safely, while all previously published records remain readable.
    allocator.block_allocations_until_failure = 1;
    REQUIRE_THROWS(layout.Reserve(100));
    REQUIRE(layout.GetRecord(0) == first);
    REQUIRE(first[stride - 1] == 0xAB);
    allocator.block_allocations_until_failure = std::numeric_limits<uint64_t>::max();
    REQUIRE_NOTHROW(layout.Reserve(100));
    REQUIRE(layout.GetRecord(0) == first);
}

TEST_CASE("Aligned blocks validate sizes and support large records and parallel writes",
          "[ut][AlignedBlockLayout]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    REQUIRE_THROWS(AlignedBlockLayout(0, allocator.get(), 4096));
    REQUIRE_THROWS(AlignedBlockLayout(65, allocator.get(), 4096));
    REQUIRE_THROWS(AlignedBlockLayout(64, allocator.get(), 0));
    REQUIRE_THROWS(AlignedBlockLayout(uint64_t{1} << 63, allocator.get(), 4096));
    AlignedBlockLayout layout(8192, allocator.get(), 4096);
    REQUIRE(layout.RecordsPerBlock() == 1);
    REQUIRE_THROWS(layout.Reserve(std::numeric_limits<uint64_t>::max()));
    layout.Reserve(32);
    std::vector<std::thread> workers;
    for (uint64_t t = 0; t < 4; ++t) {
        workers.emplace_back([&, t] {
            for (uint64_t id = t; id < 32; id += 4) {
                std::memset(layout.GetMutableRecord(id), id, 8192);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    for (uint64_t id = 0; id < 32; ++id) {
        const auto* record = layout.GetRecord(id);
        REQUIRE(reinterpret_cast<uintptr_t>(record) % 64 == 0);
        REQUIRE(record[0] == id);
        REQUIRE(record[8191] == id);
    }
}
}  // namespace vsag
