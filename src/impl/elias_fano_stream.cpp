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
#include <limits>

#include "common.h"

namespace vsag {
namespace {

uint32_t
floor_log2(uint32_t value) {
    uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

void
store_packed(uint8_t* bytes, uint64_t index, uint32_t bits, uint32_t value) {
    const uint64_t bit_offset = index * bits;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        if ((value & (1U << bit)) != 0) {
            const uint64_t target_bit = bit_offset + bit;
            bytes[target_bit / 8] |= static_cast<uint8_t>(1U << (target_bit % 8));
        }
    }
}

uint32_t
count_trailing_zeros(uint64_t value) {
#ifdef __GNUC__
    return static_cast<uint32_t>(__builtin_ctzll(value));
#else
    uint32_t result = 0;
    while ((value & 1U) == 0) {
        value >>= 1;
        ++result;
    }
    return result;
#endif
}

uint32_t
count_ones(uint64_t value) {
#ifdef __GNUC__
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t result = 0;
    while (value != 0) {
        value &= value - 1;
        ++result;
    }
    return result;
#endif
}

uint64_t
get_low_mask(uint32_t bits) {
    if (bits == 0) {
        return 0;
    }
    if (bits == 64) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (1ULL << bits) - 1ULL;
}

uint64_t
load_little_endian(const uint8_t* cursor, const uint8_t* end, uint32_t max_bytes) {
    const uint64_t remaining = static_cast<uint64_t>(end - cursor);
    const uint32_t bytes = static_cast<uint32_t>(std::min<uint64_t>(remaining, max_bytes));
    uint64_t result = 0;
    for (uint32_t byte = 0; byte < bytes; ++byte) {
        result |= static_cast<uint64_t>(cursor[byte]) << (byte * 8);
    }
    return result;
}

}  // namespace

EliasFanoStreamLayout
EliasFanoStream::GetLayout(uint32_t count, uint32_t universe) {
    EliasFanoStreamLayout layout;
    if (count == 0) {
        return layout;
    }
    CHECK_ARGUMENT(universe > 0, "Elias-Fano universe must be positive");
    const uint32_t quotient = universe / count;
    layout.low_bits_width = quotient <= 1 ? 0 : floor_log2(quotient);
    layout.low_bits_bytes = (static_cast<uint64_t>(count) * layout.low_bits_width + 7) / 8;
    layout.high_bits_count =
        ((static_cast<uint64_t>(universe) - 1) >> layout.low_bits_width) + count + 1;
    layout.high_bits_bytes = (layout.high_bits_count + 7) / 8;
    return layout;
}

void
EliasFanoStream::Encode(const uint32_t* values, uint32_t count, uint32_t universe, uint8_t* codes) {
    Encode(values, count, universe, GetLayout(count, universe), codes);
}

void
EliasFanoStream::Encode(const uint32_t* values,
                        uint32_t count,
                        uint32_t universe,
                        const EliasFanoStreamLayout& layout,
                        uint8_t* codes) {
    if (count == 0) {
        return;
    }
    CHECK_ARGUMENT(values != nullptr, "Elias-Fano values are null");
    CHECK_ARGUMENT(codes != nullptr, "Elias-Fano output codes are null");
    std::fill_n(codes, layout.SizeInBytes(), 0);
    uint32_t previous = 0;
    const uint32_t low_mask =
        layout.low_bits_width == 0
            ? 0
            : (layout.low_bits_width == 32 ? std::numeric_limits<uint32_t>::max()
                                           : ((1U << layout.low_bits_width) - 1U));
    auto* high_bits = codes + layout.low_bits_bytes;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t value = values[index];
        CHECK_ARGUMENT(value < universe, "Elias-Fano value exceeds its universe");
        CHECK_ARGUMENT(index == 0 || previous <= value, "Elias-Fano values must be ordered");
        if (layout.low_bits_width != 0) {
            store_packed(codes, index, layout.low_bits_width, value & low_mask);
        }
        const uint64_t high = static_cast<uint64_t>(value) >> layout.low_bits_width;
        const uint64_t position = high + index;
        high_bits[position / 8] |= static_cast<uint8_t>(1U << (position % 8));
        previous = value;
    }
}

EliasFanoStreamReader::EliasFanoStreamReader(const uint8_t* codes,
                                             uint32_t count,
                                             uint32_t universe)
    : EliasFanoStreamReader(codes, count, EliasFanoStream::GetLayout(count, universe)) {
}

EliasFanoStreamReader::EliasFanoStreamReader(const uint8_t* codes,
                                             uint32_t count,
                                             const EliasFanoStreamLayout& layout)
    : count_(count) {
    if (count == 0) {
        return;
    }
    CHECK_ARGUMENT(codes != nullptr, "Elias-Fano codes are null");
    low_cursor_ = codes;
    high_cursor_ = codes + layout.low_bits_bytes;
    high_end_ = high_cursor_ + layout.high_bits_bytes;
    low_bits_width_ = layout.low_bits_width;
    low_mask_ = low_bits_width_ == 32 ? std::numeric_limits<uint32_t>::max()
                                      : ((1U << low_bits_width_) - 1U);
}

uint32_t
EliasFanoStreamReader::ReadLowBits() {
    if (low_bits_width_ == 0) {
        return 0;
    }
    while (low_available_bits_ < low_bits_width_) {
        low_buffer_ |= static_cast<uint64_t>(*low_cursor_++) << low_available_bits_;
        low_available_bits_ += 8;
    }
    const uint32_t result = static_cast<uint32_t>(low_buffer_) & low_mask_;
    low_buffer_ >>= low_bits_width_;
    low_available_bits_ -= low_bits_width_;
    return result;
}

void
EliasFanoStreamReader::RefillHighBits() {
    while (high_buffer_ == 0) {
        high_base_position_ += high_available_bits_;
        CHECK_ARGUMENT(high_cursor_ < high_end_, "Elias-Fano high bits are invalid");
        const uint64_t remaining = static_cast<uint64_t>(high_end_ - high_cursor_);
        const uint32_t bytes =
            static_cast<uint32_t>(std::min<uint64_t>(remaining, sizeof(uint64_t)));
        high_buffer_ = 0;
        for (uint32_t byte = 0; byte < bytes; ++byte) {
            high_buffer_ |= static_cast<uint64_t>(high_cursor_[byte]) << (byte * 8);
        }
        high_cursor_ += bytes;
        high_available_bits_ = bytes * 8;
    }
}

uint64_t
EliasFanoStreamReader::ReadHighBits() {
    RefillHighBits();
    const uint32_t high_bit = count_trailing_zeros(high_buffer_);
    high_buffer_ &= high_buffer_ - 1U;
    const uint64_t position = high_base_position_ + high_bit;
    CHECK_ARGUMENT(position >= index_, "Elias-Fano high bits are invalid");
    return position - index_;
}

uint32_t
EliasFanoStreamReader::Read() {
    CHECK_ARGUMENT(index_ < count_, "Elias-Fano reader is exhausted");
    const uint64_t high = ReadHighBits();
    const uint32_t low = ReadLowBits();
    ++index_;
    return static_cast<uint32_t>((high << low_bits_width_) | low);
}

uint32_t
EliasFanoStreamReader::ReadBatch(uint32_t* values, uint32_t max_count) {
    CHECK_ARGUMENT(max_count <= MAX_BATCH_SIZE, "Elias-Fano batch size exceeds its limit");
    const uint32_t batch_count = std::min(max_count, count_ - index_);
    if (batch_count == 0) {
        return 0;
    }
    CHECK_ARGUMENT(values != nullptr, "Elias-Fano batch output is null");

    const uint32_t first_index = index_;
    for (uint32_t offset = 0; offset < batch_count; ++offset) {
        const uint64_t high = ReadHighBits();
        values[offset] = static_cast<uint32_t>(high);
        ++index_;
    }
    index_ = first_index;
    for (uint32_t offset = 0; offset < batch_count; ++offset) {
        const uint32_t low = ReadLowBits();
        values[offset] =
            static_cast<uint32_t>((static_cast<uint64_t>(values[offset]) << low_bits_width_) | low);
        ++index_;
    }
    return batch_count;
}

EliasFanoSeekReader::EliasFanoSeekReader(const uint8_t* codes,
                                         uint32_t count,
                                         const EliasFanoStreamLayout& layout)
    : high_remaining_bits_(layout.high_bits_count),
      low_bits_width_(layout.low_bits_width),
      low_mask_(static_cast<uint32_t>(get_low_mask(layout.low_bits_width))),
      count_(count) {
    if (count == 0) {
        return;
    }
    CHECK_ARGUMENT(codes != nullptr, "Elias-Fano codes are null");
    low_bits_ = codes;
    low_end_ = codes + layout.low_bits_bytes;
    high_cursor_ = low_end_;
}

void
EliasFanoSeekReader::RefillHighBits() {
    CHECK_ARGUMENT(high_available_bits_ == 0, "Elias-Fano high bits are not exhausted");
    CHECK_ARGUMENT(high_remaining_bits_ != 0, "Elias-Fano high bits are invalid");
    const uint32_t bits =
        static_cast<uint32_t>(std::min<uint64_t>(high_remaining_bits_, sizeof(uint64_t) * 8));
    const uint32_t bytes = (bits + 7) / 8;
    high_buffer_ = load_little_endian(high_cursor_, high_cursor_ + bytes, bytes);
    high_buffer_ &= get_low_mask(bits);
    high_cursor_ += bytes;
    high_available_bits_ = bits;
    high_remaining_bits_ -= bits;
}

void
EliasFanoSeekReader::ConsumeHighBits(uint32_t count) {
    CHECK_ARGUMENT(count <= high_available_bits_, "Elias-Fano high bits are invalid");
    const uint64_t consumed = high_buffer_ & get_low_mask(count);
    const uint32_t ones = count_ones(consumed);
    ordinal_ += ones;
    zero_count_ += count - ones;
    if (count == 64) {
        high_buffer_ = 0;
    } else {
        high_buffer_ >>= count;
    }
    high_available_bits_ -= count;
}

void
EliasFanoSeekReader::AdvanceToZeroCount(uint64_t target) {
    CHECK_ARGUMENT(target >= zero_count_, "Elias-Fano seek target must be ordered");
    while (zero_count_ < target) {
        if (high_available_bits_ == 0) {
            RefillHighBits();
        }
        const uint64_t valid_mask = get_low_mask(high_available_bits_);
        const uint64_t zero_mask = (~high_buffer_) & valid_mask;
        const uint32_t available_zeros = count_ones(zero_mask);
        const uint64_t required_zeros = target - zero_count_;
        if (available_zeros < required_zeros) {
            ConsumeHighBits(high_available_bits_);
            continue;
        }

        uint64_t remaining_zeros = zero_mask;
        for (uint64_t rank = 1; rank < required_zeros; ++rank) {
            remaining_zeros &= remaining_zeros - 1;
        }
        const uint32_t zero_position = count_trailing_zeros(remaining_zeros);
        ConsumeHighBits(zero_position + 1);
    }
}

uint32_t
EliasFanoSeekReader::ReadLowBitsAt(uint32_t index) const {
    if (low_bits_width_ == 0) {
        return 0;
    }
    const uint64_t bit_offset = static_cast<uint64_t>(index) * low_bits_width_;
    const auto* cursor = low_bits_ + bit_offset / 8;
    const uint32_t shift = static_cast<uint32_t>(bit_offset % 8);
    const uint64_t word = load_little_endian(cursor, low_end_, sizeof(uint64_t));
    return static_cast<uint32_t>((word >> shift) & low_mask_);
}

uint32_t
EliasFanoSeekReader::LowerBoundLow(uint32_t begin, uint32_t end, uint32_t target) const {
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2;
        if (ReadLowBitsAt(middle) < target) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

uint32_t
EliasFanoSeekReader::UpperBoundLow(uint32_t begin, uint32_t end, uint32_t target) const {
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2;
        if (ReadLowBitsAt(middle) <= target) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

EliasFanoOrdinalRange
EliasFanoSeekReader::FindEqualRange(uint32_t target) {
    CHECK_ARGUMENT(!has_last_target_ || last_target_ <= target,
                   "Elias-Fano seek target must be ordered");
    has_last_target_ = true;
    last_target_ = target;
    if (count_ == 0) {
        return {};
    }

    const uint32_t high = target >> low_bits_width_;
    if (!has_cached_high_ || cached_high_ != high) {
        AdvanceToZeroCount(high);
        cached_range_.begin = ordinal_;
        AdvanceToZeroCount(static_cast<uint64_t>(high) + 1);
        cached_range_.end = ordinal_;
        cached_high_ = high;
        has_cached_high_ = true;
    }

    const uint32_t low = target & low_mask_;
    const uint32_t begin = LowerBoundLow(cached_range_.begin, cached_range_.end, low);
    if (begin == cached_range_.end || ReadLowBitsAt(begin) != low) {
        return {begin, begin};
    }
    return {begin, UpperBoundLow(begin, cached_range_.end, low)};
}

}  // namespace vsag
