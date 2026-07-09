
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

#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>

namespace vsag {

// A grow-only, per-slot visibility flag map backed by atomic words.
//
// Each slot owns one bit. The intended use is a "publish after fully written"
// pattern: a producer writes a slot's payload and then calls Mark(slot)
// (release), while a consumer calls IsReady(slot) (acquire) before reading the
// payload. The release/acquire pair establishes a happens-before edge so the
// consumer that observes a set bit is guaranteed to see the fully written
// payload.
//
// Bits are only ever set, never cleared (other than via Resize/the ctor), so
// the consumer side is free of TOCTOU races: observing "ready" once means ready
// forever.
//
// Concurrency contract:
//   - Mark() and IsReady() are safe to call concurrently with each other.
//   - Resize() reallocates the backing storage and is NOT safe to run
//     concurrently with Mark()/IsReady(); the caller must serialise Resize()
//     against all Mark()/IsReady() via an external lock. In HGraph, resize()
//     takes global_mutex_ exclusively; add-side call sites coordinate resize
//     and slot publication with their own synchronization.
class AtomicVisibilityBitmap {
public:
    AtomicVisibilityBitmap() = default;

    /**
     * @brief Grow the bitmap so that at least `capacity` slots are addressable.
     *
     * Existing bits are preserved; newly added bits start cleared. Shrinking is
     * a no-op. Must not run concurrently with Mark()/IsReady().
     */
    void
    Resize(uint64_t capacity) {
        uint64_t new_words = (capacity + BITS_PER_WORD - 1) / BITS_PER_WORD;
        if (new_words <= word_count_) {
            capacity_ = std::max(capacity_, capacity);
            return;
        }
        auto new_data = std::make_unique<std::atomic<uint64_t>[]>(new_words);
        for (uint64_t i = 0; i < word_count_; ++i) {
            const auto word = data_[i].load(std::memory_order_acquire);
            new_data[i].store(word, std::memory_order_release);
        }
        for (uint64_t i = word_count_; i < new_words; ++i) {
            new_data[i].store(0, std::memory_order_relaxed);
        }
        data_ = std::move(new_data);
        word_count_ = new_words;
        capacity_ = capacity;
    }

    /**
     * @brief Publish slot `pos`: set its bit with release ordering.
     *
     * Call this only after the slot's payload has been fully written.
     */
    void
    Mark(uint64_t pos) {
        assert(pos < capacity_);
        data_[pos / BITS_PER_WORD].fetch_or(bit_mask(pos), std::memory_order_release);
    }

    /**
     * @brief Check whether slot `pos` has been published, with acquire ordering.
     *
     * A return of true guarantees the payload written before the matching
     * Mark() is visible to the caller.
     */
    [[nodiscard]] bool
    IsReady(uint64_t pos) const {
        assert(pos < capacity_);
        const auto word = data_[pos / BITS_PER_WORD].load(std::memory_order_acquire);
        return (word & bit_mask(pos)) != 0;
    }

    /**
     * @brief Publish all slots in [0, count) at once (release ordering).
     *
     * Used to backfill slots populated through bulk paths (deserialize / merge)
     * that bypass the per-slot Mark() call.
     */
    void
    MarkRange(uint64_t count) {
        assert(count <= capacity_);
        if (count == 0) {
            return;
        }

        const auto full_words = count / BITS_PER_WORD;
        for (uint64_t word = 0; word < full_words; ++word) {
            data_[word].store(0xFFFFFFFFFFFFFFFFULL, std::memory_order_release);
        }

        const auto remaining_bits = count % BITS_PER_WORD;
        if (remaining_bits != 0) {
            const auto remainder_word = count / BITS_PER_WORD;
            data_[remainder_word].fetch_or((static_cast<uint64_t>(1) << remaining_bits) - 1,
                                           std::memory_order_release);
        } else {
            const auto last_word = (count - 1) / BITS_PER_WORD;
            data_[last_word].fetch_or(0xFFFFFFFFFFFFFFFFULL, std::memory_order_release);
        }
    }

private:
    static constexpr uint64_t BITS_PER_WORD = 64;

    static uint64_t
    bit_mask(uint64_t pos) {
        return static_cast<uint64_t>(1) << (pos % BITS_PER_WORD);
    }

    std::unique_ptr<std::atomic<uint64_t>[]> data_{nullptr};
    uint64_t word_count_{0};
    uint64_t capacity_{0};
};

}  // namespace vsag
