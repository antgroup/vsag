// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vsag {

// One reusable workspace per query worker. A count is the number of current candidates covering
// a clique, not the number of times the clique has been expanded. Reset never scans all cliques
// except on marker wraparound or initial allocation/growth.
template <typename Marker = uint32_t>
class MCICliqueCounts {
public:
    void
    Reset(uint64_t size) {
        if (slots_.size() < size) {
            slots_.resize(size);
        }
        size_ = size;
        ++marker_;
        if (marker_ == 0) {
            std::fill(slots_.begin(), slots_.end(), Slot{});
            marker_ = 1;
        }
    }

    // Returns true only for a zero-to-one transition.
    bool
    Add(uint64_t id) {
        if (id >= size_) {
            return false;
        }
        auto& slot = slots_[id];
        if (slot.marker != marker_) {
            slot.marker = marker_;
            slot.count = 0;
        }
        return slot.count++ == 0;
    }

    bool
    Remove(uint64_t id) {
        if (id >= size_ or slots_[id].marker != marker_ or slots_[id].count == 0) {
            return false;
        }
        --slots_[id].count;
        return true;
    }

    [[nodiscard]] uint32_t
    Get(uint64_t id) const {
        return id < size_ and slots_[id].marker == marker_ ? slots_[id].count : 0;
    }

private:
    struct Slot {
        Marker marker{0};
        uint32_t count{0};
    };
    std::vector<Slot> slots_;
    uint64_t size_{0};
    Marker marker_{0};
};

}  // namespace vsag
