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

#pragma once

#include <cstdint>

namespace vsag {

struct EliasFanoStreamLayout {
    uint32_t low_bits_width{0};
    uint64_t low_bits_bytes{0};
    uint64_t high_bits_count{0};
    uint64_t high_bits_bytes{0};

    [[nodiscard]] uint64_t
    SizeInBytes() const {
        return low_bits_bytes + high_bits_bytes;
    }
};

struct EliasFanoOrdinalRange {
    uint32_t begin{0};
    uint32_t end{0};
};

class EliasFanoStream {
public:
    [[nodiscard]] static EliasFanoStreamLayout
    GetLayout(uint32_t count, uint32_t universe);

    static void
    Encode(const uint32_t* values, uint32_t count, uint32_t universe, uint8_t* codes);

    static void
    Encode(const uint32_t* values,
           uint32_t count,
           uint32_t universe,
           const EliasFanoStreamLayout& layout,
           uint8_t* codes);
};

class EliasFanoStreamReader {
public:
    static constexpr uint32_t MAX_BATCH_SIZE = 8;

    EliasFanoStreamReader(const uint8_t* codes, uint32_t count, uint32_t universe);

    EliasFanoStreamReader(const uint8_t* codes,
                          uint32_t count,
                          const EliasFanoStreamLayout& layout);

    [[nodiscard]] uint32_t
    Read();

    [[nodiscard]] uint32_t
    ReadBatch(uint32_t* values, uint32_t max_count);

private:
    [[nodiscard]] uint32_t
    ReadLowBits();

    [[nodiscard]] uint64_t
    ReadHighBits();

    void
    RefillHighBits();

private:
    const uint8_t* low_cursor_{nullptr};
    const uint8_t* high_cursor_{nullptr};
    const uint8_t* high_end_{nullptr};
    uint64_t low_buffer_{0};
    uint64_t high_buffer_{0};
    uint64_t high_base_position_{0};
    uint32_t low_available_bits_{0};
    uint32_t high_available_bits_{0};
    uint32_t low_bits_width_{0};
    uint32_t low_mask_{0};
    uint32_t count_{0};
    uint32_t index_{0};
};

class EliasFanoSeekReader {
public:
    EliasFanoSeekReader(const uint8_t* codes, uint32_t count, const EliasFanoStreamLayout& layout);

    [[nodiscard]] EliasFanoOrdinalRange
    FindEqualRange(uint32_t target);

private:
    void
    AdvanceToZeroCount(uint64_t target);

    void
    ConsumeHighBits(uint32_t count);

    void
    RefillHighBits();

    [[nodiscard]] uint32_t
    ReadLowBitsAt(uint32_t index) const;

    [[nodiscard]] uint32_t
    LowerBoundLow(uint32_t begin, uint32_t end, uint32_t target) const;

    [[nodiscard]] uint32_t
    UpperBoundLow(uint32_t begin, uint32_t end, uint32_t target) const;

private:
    const uint8_t* low_bits_{nullptr};
    const uint8_t* low_end_{nullptr};
    const uint8_t* high_cursor_{nullptr};
    uint64_t high_buffer_{0};
    uint64_t high_remaining_bits_{0};
    uint64_t zero_count_{0};
    uint32_t high_available_bits_{0};
    uint32_t low_bits_width_{0};
    uint32_t low_mask_{0};
    uint32_t ordinal_{0};
    uint32_t count_{0};
    uint32_t cached_high_{0};
    EliasFanoOrdinalRange cached_range_;
    uint32_t last_target_{0};
    bool has_cached_high_{false};
    bool has_last_target_{false};
};

}  // namespace vsag
