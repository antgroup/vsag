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

#include "elias_fano_stream.h"

#include <algorithm>
#include <vector>

#include "unittest.h"

namespace vsag {

TEST_CASE("EliasFanoStream round trips ordered integers", "[ut][EliasFanoStream]") {
    const std::vector<std::vector<uint32_t>> cases{
        {0},
        {0, 0, 1, 1, 2},
        {3, 5, 8, 13},
        {1, 31, 32, 63, 64, 1'000, 99'999},
    };
    const std::vector<uint32_t> universes{1, 3, 16, 100'000};

    for (uint32_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& values = cases[case_index];
        const uint32_t universe = universes[case_index];
        const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
        std::vector<uint8_t> codes(layout.SizeInBytes());
        EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());
        EliasFanoStreamReader reader(codes.data(), values.size(), layout);
        for (const uint32_t value : values) {
            REQUIRE(reader.Read() == value);
        }
    }
}

TEST_CASE("EliasFanoStream reads ordered integers in batches", "[ut][EliasFanoStream]") {
    std::vector<uint32_t> values;
    for (uint32_t index = 0; index < 73; ++index) {
        values.push_back(index * index + index / 3);
    }
    const uint32_t universe = values.back() + 1;
    const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
    std::vector<uint8_t> codes(layout.SizeInBytes());
    EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());

    EliasFanoStreamReader reader(codes.data(), values.size(), layout);
    std::vector<uint32_t> decoded;
    uint32_t batch[EliasFanoStreamReader::MAX_BATCH_SIZE];
    while (const uint32_t count = reader.ReadBatch(batch, EliasFanoStreamReader::MAX_BATCH_SIZE)) {
        decoded.insert(decoded.end(), batch, batch + count);
    }
    REQUIRE(decoded == values);
    REQUIRE(reader.ReadBatch(batch, EliasFanoStreamReader::MAX_BATCH_SIZE) == 0);
}

TEST_CASE("EliasFanoStream batch reader skips empty high words", "[ut][EliasFanoStream]") {
    std::vector<uint32_t> values(73, 0);
    const uint32_t universe = 1U << 20;
    values.back() = universe - 1;
    const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
    std::vector<uint8_t> codes(layout.SizeInBytes());
    EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());

    EliasFanoStreamReader reader(codes.data(), values.size(), layout);
    std::vector<uint32_t> decoded;
    uint32_t batch[EliasFanoStreamReader::MAX_BATCH_SIZE];
    REQUIRE(reader.ReadBatch(nullptr, 0) == 0);
    while (const uint32_t count = reader.ReadBatch(batch, EliasFanoStreamReader::MAX_BATCH_SIZE)) {
        decoded.insert(decoded.end(), batch, batch + count);
    }
    REQUIRE(decoded == values);
}

TEST_CASE("EliasFanoStream supports mixed scalar and batch reads", "[ut][EliasFanoStream]") {
    const std::vector<uint32_t> values{0, 1, 1, 7, 31, 32, 100, 255, 1'024, 8'191, 65'535};
    const uint32_t universe = values.back() + 1;
    const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
    std::vector<uint8_t> codes(layout.SizeInBytes());
    EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());

    EliasFanoStreamReader reader(codes.data(), values.size(), layout);
    REQUIRE(reader.Read() == values[0]);
    uint32_t batch[EliasFanoStreamReader::MAX_BATCH_SIZE];
    REQUIRE(reader.ReadBatch(batch, 3) == 3);
    REQUIRE(std::vector<uint32_t>(batch, batch + 3) ==
            std::vector<uint32_t>(values.begin() + 1, values.begin() + 4));
    REQUIRE(reader.Read() == values[4]);
    REQUIRE(reader.ReadBatch(batch, EliasFanoStreamReader::MAX_BATCH_SIZE) == 6);
    REQUIRE(std::vector<uint32_t>(batch, batch + 6) ==
            std::vector<uint32_t>(values.begin() + 5, values.end()));
    REQUIRE(reader.ReadBatch(nullptr, 0) == 0);
    REQUIRE_THROWS(reader.ReadBatch(batch, EliasFanoStreamReader::MAX_BATCH_SIZE + 1));
}

TEST_CASE("EliasFanoSeekReader finds ordinal ranges", "[ut][EliasFanoStream]") {
    const std::vector<std::vector<uint32_t>> cases{
        {0, 0, 1, 1, 2, 3, 3},
        {3, 5, 8, 13},
        {0, 1, 7, 31, 32, 100, 255, 511, 1'023},
    };
    const std::vector<uint32_t> universes{4, 16, 1'024};

    for (uint32_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& values = cases[case_index];
        const uint32_t universe = universes[case_index];
        const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
        std::vector<uint8_t> codes(layout.SizeInBytes());
        EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());

        EliasFanoSeekReader reader(codes.data(), values.size(), layout);
        for (uint32_t target = 0; target < universe; ++target) {
            const auto expected = std::equal_range(values.begin(), values.end(), target);
            const auto actual = reader.FindEqualRange(target);
            REQUIRE(actual.begin == static_cast<uint32_t>(expected.first - values.begin()));
            REQUIRE(actual.end == static_cast<uint32_t>(expected.second - values.begin()));
        }
    }
}

TEST_CASE("EliasFanoSeekReader skips high words and validates ordering", "[ut][EliasFanoStream]") {
    std::vector<uint32_t> values(73, 0);
    const uint32_t universe = 1U << 20;
    values.back() = universe - 1;
    const auto layout = EliasFanoStream::GetLayout(values.size(), universe);
    std::vector<uint8_t> codes(layout.SizeInBytes());
    EliasFanoStream::Encode(values.data(), values.size(), universe, layout, codes.data());

    EliasFanoSeekReader reader(codes.data(), values.size(), layout);
    REQUIRE(reader.FindEqualRange(0).begin == 0);
    REQUIRE(reader.FindEqualRange(0).end == values.size() - 1);
    const auto missing = reader.FindEqualRange(universe / 2);
    REQUIRE(missing.begin == missing.end);
    const auto last = reader.FindEqualRange(universe - 1);
    REQUIRE(last.begin == values.size() - 1);
    REQUIRE(last.end == values.size());

    EliasFanoSeekReader unordered_reader(codes.data(), values.size(), layout);
    REQUIRE_NOTHROW(unordered_reader.FindEqualRange(10));
    REQUIRE_THROWS(unordered_reader.FindEqualRange(9));
}

TEST_CASE("EliasFanoStream validates its input", "[ut][EliasFanoStream]") {
    std::vector<uint32_t> unordered{2, 1};
    const auto layout = EliasFanoStream::GetLayout(unordered.size(), 3);
    std::vector<uint8_t> codes(layout.SizeInBytes());
    REQUIRE_THROWS(EliasFanoStream::Encode(unordered.data(), unordered.size(), 3, codes.data()));

    std::vector<uint32_t> out_of_range{0, 3};
    REQUIRE_THROWS(
        EliasFanoStream::Encode(out_of_range.data(), out_of_range.size(), 3, codes.data()));
}

}  // namespace vsag
