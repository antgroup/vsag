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

#include "layout/variable_record_layout.h"

#include <array>
#include <cstring>
#include <limits>

#include "impl/allocator/safe_allocator.h"
#include "io/memory_block_io/memory_block_io.h"
#include "io/memory_io/memory_io.h"
#include "unittest.h"
#include "vsag/options.h"

namespace vsag {

TEST_CASE("VariableRecordLayout maps ids to appended records", "[ut][VariableRecordLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto location_io = std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(),
                                                       common_param.allocator_.get());
    auto payload_io = std::make_shared<MemoryIO>(IOParamPtr{}, common_param);
    VariableRecordLayout<OffsetAndLengthLocationPolicy, MemoryBlockIO, MemoryIO> layout(location_io,
                                                                                        payload_io);
    layout.ResizeLocations(4);

    const std::array<uint8_t, 3> first{1, 3, 5};
    const std::array<uint8_t, 5> second{2, 4, 6, 8, 10};
    layout.Write(2, first.data(), first.size());
    layout.Write(0, second.data(), second.size());

    const auto first_location = layout.ReadLocation(2);
    const auto second_location = layout.ReadLocation(0);
    REQUIRE(first_location.offset == 0);
    REQUIRE(first_location.length == first.size());
    REQUIRE(second_location.offset == first.size());
    REQUIRE(second_location.length == second.size());
    REQUIRE(layout.GetNextOffset() == first.size() + second.size());

    bool need_release = true;
    const auto* record = layout.Read(0, need_release);
    REQUIRE(record != nullptr);
    REQUIRE_FALSE(need_release);
    REQUIRE(std::memcmp(record, second.data(), second.size()) == 0);

    const std::array locations{second_location, first_location};
    std::array<uint8_t, 8> batch{};
    REQUIRE(layout.MultiRead(
        locations.data(), locations.size(), batch.data(), common_param.allocator_.get()));
    const std::array<uint8_t, 8> expected{2, 4, 6, 8, 10, 1, 3, 5};
    REQUIRE(batch == expected);
}

TEST_CASE("HeaderLengthLocationPolicy rejects an unset element width",
          "[ut][VariableRecordLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    ByteRangeLayout<MemoryIO> payload(nullptr, common_param);
    HeaderLengthLocationPolicy policy;
    REQUIRE_THROWS_AS(policy.GetLength(0, payload), VsagException);

    auto location_io = std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(),
                                                       common_param.allocator_.get());
    auto payload_io = std::make_shared<MemoryIO>(IOParamPtr{}, common_param);
    VariableRecordLayout<HeaderLengthLocationPolicy, MemoryBlockIO, MemoryIO> layout(location_io,
                                                                                     payload_io);
    REQUIRE_FALSE(layout.IsValidLocation(0));
    const HeaderLengthLocationPolicy::Entry location = 0;
    uint8_t output = 0;
    REQUIRE_FALSE(layout.MultiRead(&location, 1, &output, common_param.allocator_.get()));
}

TEST_CASE("VariableRecordLayout validates a location before writing payload",
          "[ut][VariableRecordLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto location_io = std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(),
                                                       common_param.allocator_.get());
    auto payload_io = std::make_shared<MemoryIO>(IOParamPtr{}, common_param);
    VariableRecordLayout<OffsetAndLengthLocationPolicy, MemoryBlockIO, MemoryIO> layout(location_io,
                                                                                        payload_io);
    layout.ResizeLocations(1);

    const uint8_t data = 0;
    const auto payload_size = layout.Payload().GetByteSize();
    const auto invalid_length = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    REQUIRE_THROWS_AS(layout.Write(0, &data, invalid_length), VsagException);
    REQUIRE(layout.Payload().GetByteSize() == payload_size);
    REQUIRE(layout.GetNextOffset() == 0);
}

TEST_CASE("VariableRecordLayout rejects out-of-bounds locations", "[ut][VariableRecordLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto location_io = std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(),
                                                       common_param.allocator_.get());
    auto payload_io = std::make_shared<MemoryIO>(IOParamPtr{}, common_param);
    VariableRecordLayout<OffsetAndLengthLocationPolicy, MemoryBlockIO, MemoryIO> layout(location_io,
                                                                                        payload_io);
    layout.ReservePayload(8);

    const OffsetAndLengthLocationPolicy::Entry overflow{std::numeric_limits<uint64_t>::max(), 2};
    const OffsetAndLengthLocationPolicy::Entry out_of_bounds{7, 2};
    REQUIRE_FALSE(layout.IsValidLocation(overflow));
    REQUIRE_FALSE(layout.IsValidLocation(out_of_bounds));

    bool need_release = true;
    REQUIRE(layout.Read(overflow, need_release) == nullptr);
    REQUIRE_FALSE(need_release);
    need_release = true;
    REQUIRE(layout.Read(out_of_bounds, need_release) == nullptr);
    REQUIRE_FALSE(need_release);
}

TEST_CASE("VariableRecordLayout payload reserve is grow-only", "[ut][VariableRecordLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    auto location_io = std::make_shared<MemoryBlockIO>(Options::Instance().block_size_limit(),
                                                       common_param.allocator_.get());
    auto payload_io = std::make_shared<MemoryIO>(IOParamPtr{}, common_param);
    VariableRecordLayout<OffsetAndLengthLocationPolicy, MemoryBlockIO, MemoryIO> layout(location_io,
                                                                                        payload_io);

    layout.ReservePayload(16);
    layout.ReservePayload(8);
    REQUIRE(layout.Payload().GetByteSize() == 16);
}

}  // namespace vsag
