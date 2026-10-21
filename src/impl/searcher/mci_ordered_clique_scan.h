// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "impl/searcher/mci_clique_counts.h"

namespace vsag {

// The hierarchy stores one bit per active clique and one bit per nonempty word above it.
// Epoch-tagged words avoid clearing the bitmap on every query. Counts remain exact under removal.
template <typename Marker = uint32_t>
class MCIOrderedCliqueCounts {
public:
    static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

    void
    Reset(uint64_t size) {
        counts_.Reset(size);
        size_ = size;
        minimum_ = kNone;
        maximum_ = 0;
        ++marker_;
        if (marker_ == 0) {
            for (auto& level : words_) {
                std::fill(level.begin(), level.end(), Word{});
            }
            marker_ = 1;
        }
        levels_ = 0;
        do {
            size = (size + 63) / 64;
            if (words_[levels_].size() < size) {
                words_[levels_].resize(size);
            }
            ++levels_;
        } while (size > 1);
    }

    bool
    Add(uint64_t id) {
        if (not counts_.Add(id)) {
            return false;
        }
        const bool was_empty = minimum_ == kNone;
        minimum_ = std::min(minimum_, static_cast<uint32_t>(id));
        maximum_ = was_empty ? id : std::max<uint32_t>(maximum_, id);
        for (uint32_t level = 0; level < levels_; ++level) {
            auto& word = words_[level][id >> 6U];
            if (word.marker != marker_) {
                word.marker = marker_;
                word.bits = 0;
            }
            const auto before = word.bits;
            word.bits |= uint64_t{1} << (id & 63U);
            if (before != 0) {
                break;
            }
            id >>= 6U;
        }
        return true;
    }

    bool
    Remove(uint64_t id) {
        if (not counts_.Remove(id)) {
            return false;
        }
        if (counts_.Get(id) != 0) {
            return true;
        }
        auto index = id;
        for (uint32_t level = 0; level < levels_; ++level) {
            auto& word = words_[level][index >> 6U];
            word.bits &= ~(uint64_t{1} << (index & 63U));
            if (word.bits != 0) {
                break;
            }
            index >>= 6U;
        }
        if (id == minimum_) {
            minimum_ = Next(id + 1);
        }
        if (id == maximum_) {
            maximum_ = id == 0 ? 0 : Previous(id - 1);
            if (maximum_ == kNone) {
                maximum_ = 0;
            }
        }
        return true;
    }

    [[nodiscard]] uint32_t
    Get(uint64_t id) const {
        return counts_.Get(id);
    }

    [[nodiscard]] bool
    Contains(uint32_t id) const {
        return id < size_ and ContainsKnownValid(id);
    }

    // Only for IDs read from the immutable, validated CSR.
    [[nodiscard]] bool
    ContainsKnownValid(uint32_t id) const {
        const auto& word = words_[0][id >> 6U];
        return word.marker == marker_ and ((word.bits >> (id & 63U)) & 1U) != 0;
    }

    [[nodiscard]] uint32_t
    Minimum() const {
        return minimum_;
    }

    [[nodiscard]] uint32_t
    Maximum() const {
        return maximum_;
    }

    [[nodiscard]] uint32_t
    Next(uint64_t id) const {
        if (id >= size_) {
            return kNone;
        }
        for (uint32_t level = 0; level < levels_; ++level) {
            const auto word_id = id >> 6U;
            const auto bits = Bits(level, word_id) & (~uint64_t{0} << (id & 63U));
            if (bits != 0) {
                id = word_id * 64 + __builtin_ctzll(bits);
                for (uint32_t lower = level; lower > 0; --lower) {
                    id = id * 64 + __builtin_ctzll(Bits(lower - 1, id));
                }
                return static_cast<uint32_t>(id);
            }
            id = word_id + 1;
        }
        return kNone;
    }

    [[nodiscard]] uint32_t
    Previous(uint64_t id) const {
        if (size_ == 0) {
            return kNone;
        }
        id = std::min(id, size_ - 1);
        for (uint32_t level = 0; level < levels_; ++level) {
            const auto word_id = id >> 6U;
            const auto bits = Bits(level, word_id) & (~uint64_t{0} >> (63U - (id & 63U)));
            if (bits != 0) {
                id = word_id * 64 + 63U - __builtin_clzll(bits);
                for (uint32_t lower = level; lower > 0; --lower) {
                    id = id * 64 + 63U - __builtin_clzll(Bits(lower - 1, id));
                }
                return static_cast<uint32_t>(id);
            }
            if (word_id == 0) {
                return kNone;
            }
            id = word_id - 1;
        }
        return kNone;
    }

private:
    struct Word {
        uint64_t bits{0};
        Marker marker{0};
    };

    uint64_t
    Bits(uint32_t level, uint64_t word) const {
        if (word >= words_[level].size() or words_[level][word].marker != marker_) {
            return 0;
        }
        return words_[level][word].bits;
    }

    MCICliqueCounts<Marker> counts_;
    std::array<std::vector<Word>, 6> words_;
    uint64_t size_{0};
    Marker marker_{0};
    uint32_t levels_{0};
    uint32_t minimum_{kNone};
    uint32_t maximum_{0};
};

// The immutable inverse CSR rows must have strictly increasing clique IDs. Sequence 1 is the
// lexicographic row order, sequence 2 orders its row positions by maximum clique ID. A suffix
// minimum on sequence 2 finds the earliest sequence-1 position whose maximum reaches a bound.
class MCIOrderedValidRows {
public:
    struct Row {
        uint32_t id;
        uint32_t begin;
        uint32_t end;
        uint32_t minimum;
        uint32_t maximum;
    };
    static constexpr uint64_t kBlock = 32;

    void
    Build(const uint32_t* offsets,
          const uint32_t* cliques,
          uint64_t total,
          const uint32_t* valid,
          uint64_t count) {
        rows.clear();
        rows.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            const auto id = valid[i];
            if (id < total and offsets[id] < offsets[id + 1]) {
                const auto begin = offsets[id];
                const auto end = offsets[id + 1];
                rows.push_back(Row{id, begin, end, cliques[begin], cliques[end - 1]});
            }
        }
        RadixSort(rows, row_scratch_, [](const auto& row) { return row.minimum; });
        auto compare_tail = [cliques](const Row& a, const Row& b) {
            const auto* ab = cliques + a.begin + 1;
            const auto* ae = cliques + a.end;
            const auto* bb = cliques + b.begin + 1;
            const auto* be = cliques + b.end;
            if (std::lexicographical_compare(ab, ae, bb, be)) {
                return true;
            }
            if (std::lexicographical_compare(bb, be, ab, ae)) {
                return false;
            }
            return a.id < b.id;
        };
        for (uint64_t begin = 0; begin < rows.size();) {
            auto end = begin + 1;
            while (end < rows.size() and rows[end].minimum == rows[begin].minimum) {
                ++end;
            }
            if (end - begin > 1) {
                std::sort(rows.begin() + begin, rows.begin() + end, compare_tail);
            }
            begin = end;
        }
        by_max.resize(rows.size());
        for (uint64_t i = 0; i < rows.size(); ++i) {
            by_max[i] = static_cast<uint32_t>(i);
        }
        // Stable radix order preserves sequence-1 position as the tie breaker.
        RadixSort(by_max, max_scratch_, [this](auto index) { return rows[index].maximum; });
        suffix_first.resize(rows.size());
        auto first = static_cast<uint32_t>(rows.size());
        for (uint64_t i = rows.size(); i > 0; --i) {
            first = std::min(first, by_max[i - 1]);
            suffix_first[i - 1] = first;
        }
        block_max.resize((rows.size() + kBlock - 1) / kBlock);
        std::fill(block_max.begin(), block_max.end(), 0);
        for (uint64_t i = 0; i < rows.size(); ++i) {
            block_max[i / kBlock] = std::max(block_max[i / kBlock], rows[i].maximum);
        }
    }

    [[nodiscard]] uint64_t
    Start(uint32_t lower) const {
        const auto it =
            std::lower_bound(by_max.begin(), by_max.end(), lower, [this](auto row, auto bound) {
                return rows[row].maximum < bound;
            });
        return it == by_max.end() ? rows.size() : suffix_first[it - by_max.begin()];
    }

    std::vector<Row> rows;
    std::vector<uint32_t> by_max;
    std::vector<uint32_t> suffix_first;
    std::vector<uint32_t> block_max;

private:
    template <typename Value, typename Key>
    static void
    RadixSort(std::vector<Value>& values, std::vector<Value>& scratch, Key key) {
        const auto count = values.size();
        if (count < 2) {
            return;
        }
        if (scratch.size() < count) {
            scratch.resize(count);
        }
        uint32_t combined = 0;
        for (const auto& value : values) {
            combined |= key(value);
        }
        auto* source = values.data();
        auto* target = scratch.data();
        constexpr uint32_t kBits = 11;
        constexpr uint32_t kBuckets = 1U << kBits;
        for (uint32_t shift = 0; shift < 32 and (combined >> shift) != 0; shift += kBits) {
            std::array<uint32_t, kBuckets> offsets{};
            for (uint64_t i = 0; i < count; ++i) {
                ++offsets[(key(source[i]) >> shift) & (kBuckets - 1)];
            }
            uint32_t next = 0;
            for (auto& offset : offsets) {
                const auto size = offset;
                offset = next;
                next += size;
            }
            for (uint64_t i = 0; i < count; ++i) {
                target[offsets[(key(source[i]) >> shift) & (kBuckets - 1)]++] = source[i];
            }
            std::swap(source, target);
        }
        if (source != values.data()) {
            std::copy_n(source, count, values.data());
        }
    }

    std::vector<Row> row_scratch_;
    std::vector<uint32_t> max_scratch_;
};

}  // namespace vsag
