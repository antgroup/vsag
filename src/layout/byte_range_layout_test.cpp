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

#include "layout/byte_range_layout.h"

#include <array>
#include <cstring>
#include <sstream>

#include "impl/allocator/safe_allocator.h"
#include "io/memory_io/memory_io.h"
#include "unittest.h"

namespace vsag {

TEST_CASE("ByteRangeLayout addresses opaque byte ranges", "[ut][ByteRangeLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    ByteRangeLayout<MemoryIO> layout(nullptr, common_param);
    layout.Resize(16);

    const std::array<uint8_t, 4> first{1, 2, 3, 4};
    const std::array<uint8_t, 3> second{7, 8, 9};
    layout.Write(2, first.data(), first.size());
    layout.Write(10, second.data(), second.size());

    std::array<uint8_t, 4> output{};
    REQUIRE(layout.Read(2, output.size(), output.data()));
    REQUIRE(output == first);

    std::array<uint64_t, 2> offsets{10, 2};
    std::array<uint64_t, 2> lengths{3, 4};
    std::array<uint8_t, 7> batch{};
    REQUIRE(layout.MultiRead(offsets.data(), lengths.data(), offsets.size(), batch.data()));
    const std::array<uint8_t, 7> expected{7, 8, 9, 1, 2, 3, 4};
    REQUIRE(batch == expected);
}

TEST_CASE("ByteRangeLayout preserves underlying IO serialization", "[ut][ByteRangeLayout]") {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    ByteRangeLayout<MemoryIO> source(nullptr, common_param);
    const std::array<uint8_t, 5> bytes{2, 4, 6, 8, 10};
    source.Resize(bytes.size());
    source.Write(0, bytes.data(), bytes.size());

    std::stringstream stream;
    IOStreamWriter writer(stream);
    source.Serialize(writer);
    stream.seekg(0, std::ios::beg);

    ByteRangeLayout<MemoryIO> restored(nullptr, common_param);
    IOStreamReader reader(stream);
    restored.Deserialize(reader);
    bool need_release = true;
    const auto* result = restored.Read(0, bytes.size(), need_release);
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(need_release);
    REQUIRE(std::memcmp(result, bytes.data(), bytes.size()) == 0);
    REQUIRE(restored.GetMemoryUsage() >= bytes.size());
}

}  // namespace vsag
