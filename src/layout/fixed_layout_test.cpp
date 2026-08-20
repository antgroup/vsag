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

#include "layout/fixed_layout.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "io/memory_io/memory_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"

using namespace vsag;

namespace {

class NonMemoryReviewIO : public BasicIO<NonMemoryReviewIO> {
public:
    static constexpr bool InMemory = false;
    static constexpr bool SkipDeserialize = false;

    NonMemoryReviewIO(const IOParamPtr&, const IndexCommonParam& common_param)
        : BasicIO<NonMemoryReviewIO>(common_param.allocator_.get()) {
    }

    void
    WriteImpl(const uint8_t*, uint64_t size, uint64_t offset) {
        this->size_ = std::max(this->size_, offset + size);
    }

    bool
    ReadImpl(uint64_t, uint64_t, uint8_t*) const {
        return false;
    }

    [[nodiscard]] const uint8_t*
    DirectReadImpl(uint64_t, uint64_t, bool& need_release) const {
        need_release = false;
        return nullptr;
    }
};

IndexCommonParam
MakeCommonParam() {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

}  // namespace

TEST_CASE("FixedLayout maps ids to fixed records", "[ut][FixedLayout]") {
    auto common_param = MakeCommonParam();
    FixedLayout<MemoryIO> layout(4, nullptr, common_param);
    layout.Resize(4);

    const std::array<uint8_t, 4> first{1, 2, 3, 4};
    const std::array<uint8_t, 4> second{5, 6, 7, 8};
    const std::array<uint8_t, 4> replacement{9, 10, 11, 12};
    layout.Write(1, first.data());
    layout.Write(3, second.data());

    std::array<uint8_t, 4> output{};
    REQUIRE(layout.Read(1, output.data()));
    REQUIRE(output == first);
    REQUIRE(layout.Read(3, output.data()));
    REQUIRE(output == second);

    layout.Write(1, replacement.data());
    REQUIRE(layout.Read(1, output.data()));
    REQUIRE(output == replacement);
}

TEST_CASE("FixedLayout supports ranges and id batch reads", "[ut][FixedLayout]") {
    auto common_param = MakeCommonParam();
    FixedLayout<MemoryIO> layout(2, nullptr, common_param);
    layout.Resize(4);

    const std::array<uint8_t, 8> records{1, 2, 3, 4, 5, 6, 7, 8};
    layout.WriteRange(0, records.data(), 4);

    bool need_release = true;
    const auto* range = layout.ReadRange(1, 2, need_release);
    REQUIRE(range != nullptr);
    REQUIRE_FALSE(need_release);
    REQUIRE(std::memcmp(range, records.data() + 2, 4) == 0);

    const std::array<InnerIdType, 4> ids{3, 1, 3, 0};
    std::array<uint8_t, 8> batch{};
    REQUIRE(layout.MultiRead(ids.data(), ids.size(), batch.data(), common_param.allocator_.get()));
    const std::array<uint8_t, 8> expected{7, 8, 3, 4, 7, 8, 1, 2};
    REQUIRE(batch == expected);

    std::array<uint8_t, 1> empty_batch{};
    REQUIRE(layout.MultiRead(nullptr, 0, empty_batch.data(), common_param.allocator_.get()));

    REQUIRE(layout.TryGetContiguousData() != nullptr);
    REQUIRE(std::memcmp(layout.TryGetContiguousData(), records.data(), records.size()) == 0);
}

TEST_CASE("FixedLayout preserves io serialization bytes", "[ut][FixedLayout]") {
    auto common_param = MakeCommonParam();
    FixedLayout<MemoryIO> source(3, nullptr, common_param);
    source.Resize(2);
    const std::array<uint8_t, 6> records{1, 3, 5, 7, 9, 11};
    source.WriteRange(0, records.data(), 2);

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source.Serialize(writer);
    stream.seekg(0, std::ios::beg);

    FixedLayout<MemoryIO> restored(3, nullptr, common_param);
    IOStreamReader reader(stream);
    restored.Deserialize(reader);

    bool need_release = false;
    const auto* restored_records = restored.ReadRange(0, 2, need_release);
    REQUIRE(restored_records != nullptr);
    REQUIRE_FALSE(need_release);
    REQUIRE(std::memcmp(restored_records, records.data(), records.size()) == 0);
}

TEST_CASE("FixedLayout rejects overflowing byte ranges", "[ut][FixedLayout]") {
    auto common_param = MakeCommonParam();
    FixedLayout<MemoryIO> layout(std::numeric_limits<uint64_t>::max(), nullptr, common_param);
    REQUIRE_THROWS_AS(layout.Resize(2), VsagException);
}

TEST_CASE("FixedLayout handles non-memory accounting and failed moves", "[ut][FixedLayout]") {
    auto common_param = MakeCommonParam();
    FixedLayout<NonMemoryReviewIO> layout(4, nullptr, common_param);
    layout.Resize(2);

    REQUIRE(layout.GetMemoryUsage() == 0);
    REQUIRE_THROWS_AS(layout.Move(0, 1), VsagException);
}
