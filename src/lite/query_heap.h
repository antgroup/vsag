// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

namespace vsag::lite::detail {

// Same ranking/admission as priority_queue; owned data can be consumed once routing ends.
// Native DistanceHeap::GetData likewise exposes retained candidates without popping them.
template <typename T, typename Compare>
class QueryHeap : public std::priority_queue<T, std::vector<T>, Compare> {
    using Base = std::priority_queue<T, std::vector<T>, Compare>;

public:
    using Base::Base;

    // Caller admits only values better than the nonempty heap's current root.
    void
    ReplaceTop(T value) {
        const uint64_t count = this->c.size();
        uint64_t hole = 0;
        while (hole < count / 2) {
            uint64_t child = 2 * hole + 1;
            if (child + 1 < count and this->comp(this->c[child], this->c[child + 1])) {
                ++child;
            }
            if (not this->comp(value, this->c[child])) {
                break;
            }
            this->c[hole] = std::move(this->c[child]);
            hole = child;
        }
        this->c[hole] = std::move(value);
    }

    [[nodiscard]] const std::vector<T>&
    GetData() const {
        return this->c;
    }

    [[nodiscard]] std::vector<T>
    TakeData() {
        return std::move(this->c);
    }
};

}  // namespace vsag::lite::detail
