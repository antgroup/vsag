// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace vsag {

// Reused on one query worker. Reset advances the query marker without clearing the array.
// Only allocation/growth and marker wraparound initialize storage.
template <typename Marker = uint32_t>
class MCIEpochVisitMarks {
    static_assert(std::is_integral_v<Marker> and std::is_unsigned_v<Marker>);

public:
    void
    Reset(uint64_t size) {
        if (marks_.size() < size) {
            marks_.resize(size, 0);
        }
        ++marker_;
        if (marker_ == 0) {
            std::memset(marks_.data(), 0, marks_.size() * sizeof(Marker));
            marker_ = 1;
        }
    }

    [[nodiscard]] bool
    Get(uint64_t id) const {
        return id < marks_.size() and marks_[id] == marker_;
    }

    void
    Set(uint64_t id) {
        if (id >= marks_.size()) {
            marks_.resize(id + 1, 0);
        }
        marks_[id] = marker_;
    }

    [[nodiscard]] bool
    TryVisit(uint64_t id) {
        if (id >= marks_.size()) {
            return false;
        }
        auto& value = marks_[id];
        if (value == marker_) {
            return false;
        }
        value = marker_;
        return true;
    }

private:
    std::vector<Marker> marks_;
    Marker marker_{0};
};

using MCIEpochMarks = MCIEpochVisitMarks<uint32_t>;

}  // namespace vsag
