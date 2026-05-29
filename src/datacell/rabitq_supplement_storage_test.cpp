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

#include "rabitq_supplement_storage.h"

#include <cstring>
#include <numeric>
#include <sstream>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/memory_io.h"
#include "io/memory_io_parameter.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

namespace {

constexpr uint64_t kCodeSize = 24;
constexpr uint64_t kInitialCapacity = 8;

IndexCommonParam
MakeCommonParam(const std::shared_ptr<Allocator>& allocator) {
    IndexCommonParam common_param;
    common_param.allocator_ = allocator;
    common_param.dim_ = 64;
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    return common_param;
}

std::shared_ptr<RaBitQSupplementStorage<MemoryIO>>
MakeStorage(const IndexCommonParam& common_param, uint64_t code_size = kCodeSize) {
    auto io_param = std::make_shared<MemoryIOParameter>();
    auto storage = std::make_shared<RaBitQSupplementStorage<MemoryIO>>(io_param, common_param);
    storage->SetCodeSize(code_size);
    return storage;
}

}  // namespace

TEST_CASE("RaBitQSupplementStorage CodeSize", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);
    REQUIRE(storage->GetCodeSize() == kCodeSize);
    REQUIRE(RaBitQSupplementStorage<MemoryIO>::InMemory == MemoryIO::InMemory);

    storage->SetCodeSize(48);
    REQUIRE(storage->GetCodeSize() == 48);
}

TEST_CASE("RaBitQSupplementStorage Resize And Shrink", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);

    storage->Resize(kInitialCapacity);
    REQUIRE(storage->GetMemoryUsage() >= 0);

    storage->Shrink(kInitialCapacity / 2);
    REQUIRE(storage->GetMemoryUsage() >= 0);
}

TEST_CASE("RaBitQSupplementStorage Write And Read", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);

    storage->Resize(kInitialCapacity);

    std::vector<std::vector<uint8_t>> records(kInitialCapacity, std::vector<uint8_t>(kCodeSize, 0));
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        for (uint64_t b = 0; b < kCodeSize; ++b) {
            records[id][b] = static_cast<uint8_t>((id * 17U + b * 3U) & 0xFFU);
        }
        storage->Write(records[id].data(), static_cast<InnerIdType>(id));
    }

    // Copy-Read overload.
    std::vector<uint8_t> buffer(kCodeSize, 0);
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        std::fill(buffer.begin(), buffer.end(), 0);
        REQUIRE(storage->Read(static_cast<InnerIdType>(id), buffer.data()));
        REQUIRE(std::memcmp(buffer.data(), records[id].data(), kCodeSize) == 0);
    }

    // Direct-Read + Release overload.
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        bool need_release = false;
        const uint8_t* ptr = storage->Read(static_cast<InnerIdType>(id), need_release);
        REQUIRE(ptr != nullptr);
        REQUIRE(std::memcmp(ptr, records[id].data(), kCodeSize) == 0);
        if (need_release) {
            storage->Release(ptr);
        }
        // Releasing nullptr must be a no-op.
        storage->Release(nullptr);
    }
}

TEST_CASE("RaBitQSupplementStorage Prefetch And MultiRead", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);

    storage->Resize(kInitialCapacity);

    std::vector<uint8_t> record(kCodeSize, 0);
    std::iota(record.begin(), record.end(), static_cast<uint8_t>(2));
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        storage->Write(record.data(), static_cast<InnerIdType>(id));
    }

    // Prefetch should not throw and the bytes argument is clamped to code size.
    storage->Prefetch(0, kCodeSize);
    storage->Prefetch(static_cast<InnerIdType>(kInitialCapacity - 1), kCodeSize * 8);

    constexpr uint64_t kBatch = 4;
    std::vector<uint8_t> dst(kCodeSize * kBatch, 0);
    std::vector<uint64_t> sizes(kBatch, kCodeSize);
    std::vector<uint64_t> offsets(kBatch, 0);
    for (uint64_t i = 0; i < kBatch; ++i) {
        offsets[i] = i * kCodeSize;
    }
    storage->MultiRead(dst.data(), sizes.data(), offsets.data(), kBatch);
    for (uint64_t i = 0; i < kBatch; ++i) {
        REQUIRE(std::memcmp(dst.data() + i * kCodeSize, record.data(), kCodeSize) == 0);
    }
}

TEST_CASE("RaBitQSupplementStorage Serialize And Deserialize", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);

    storage->Resize(kInitialCapacity);
    std::vector<uint8_t> record(kCodeSize, 0);
    for (uint64_t b = 0; b < kCodeSize; ++b) {
        record[b] = static_cast<uint8_t>(b * 5U + 11U);
    }
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        storage->Write(record.data(), static_cast<InnerIdType>(id));
    }

    std::stringstream ss;
    IOStreamWriter writer(ss);
    storage->Serialize(writer);

    auto other = MakeStorage(common_param);
    other->Resize(kInitialCapacity);
    IOStreamReader reader(ss);
    other->Deserialize(reader);

    std::vector<uint8_t> buffer(kCodeSize, 0);
    for (uint64_t id = 0; id < kInitialCapacity; ++id) {
        std::fill(buffer.begin(), buffer.end(), 0);
        REQUIRE(other->Read(static_cast<InnerIdType>(id), buffer.data()));
        REQUIRE(std::memcmp(buffer.data(), record.data(), kCodeSize) == 0);
    }
}

TEST_CASE("RaBitQSupplementStorage InitIO", "[ut][RaBitQSupplementStorage]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common_param = MakeCommonParam(allocator);
    auto storage = MakeStorage(common_param);

    auto io_param = std::make_shared<MemoryIOParameter>();
    storage->InitIO(io_param);
    REQUIRE(storage->GetCodeSize() == kCodeSize);
}
