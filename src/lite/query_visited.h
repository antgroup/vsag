// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>

namespace vsag::lite::detail {

// Query-owned bitmap: same word/mask semantics as native VisitedList::TestAndSet,
// without a persistent pool, epoch tags or additional Full dependencies.
template <bool packed>
class QueryVisited {
public:
    explicit QueryVisited(uint64_t count)
        : words_(packed ? count / 64 + static_cast<uint64_t>(count % 64 != 0) : count, 0) {
    }

    // Slots are already bounded by the index/topology, as with the old byte array.
    [[nodiscard]] bool
    TestAndSet(uint64_t slot) {
        if constexpr (not packed) {
            if (words_[slot] != 0) {
                return true;
            }
            words_[slot] = 1;
            return false;
        } else {
            auto& word = words_[slot / 64];
            const uint64_t mask = uint64_t{1} << (slot % 64);
            if ((word & mask) != 0) {
                return true;
            }
            word |= mask;
            return false;
        }
    }

    // Logical word payload, not allocator overhead or process RSS.
    [[nodiscard]] uint64_t
    PayloadBytes() const {
        return words_.size() * sizeof(Word);
    }

private:
    using Word = std::conditional_t<packed, uint64_t, uint8_t>;
    std::vector<Word> words_;
};

}  // namespace vsag::lite::detail
