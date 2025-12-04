
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

#include <atomic>
#include <cstdint>

namespace vsag {

template <typename T, uint64_t N>
class SPSCQueue {
    static_assert(N && !(N & (N - 1)), "N must be power of 2");

    alignas(64) std::atomic<uint64_t> write_idx{0};
    alignas(64) std::atomic<uint64_t> read_idx{0};
    alignas(64) T buffer[N];

public:
    inline bool
    Push(const T& item) {
        auto current_write = write_idx.load(std::memory_order_relaxed);
        auto next_write = current_write + 1;
        if (next_write - read_idx.load(std::memory_order_acquire) >= N) {
            return false;  // full
        }

        buffer[current_write & (N - 1)] = item;
        write_idx.store(next_write, std::memory_order_release);
        return true;
    }

    inline bool
    Push(T&& item) {
        auto current_write = write_idx.load(std::memory_order_relaxed);
        auto next_write = current_write + 1;
        if (next_write - read_idx.load(std::memory_order_acquire) >= N) {
            return false;  // full
        }

        buffer[current_write & (N - 1)] = std::move(item);
        write_idx.store(next_write, std::memory_order_release);
        return true;
    }

    inline bool
    Pop(T& out) {
        auto current_read = read_idx.load(std::memory_order_relaxed);
        if (current_read == write_idx.load(std::memory_order_acquire)) {
            return false;  // empty
        }

        out = buffer[current_read & (N - 1)];
        read_idx.store(current_read + 1, std::memory_order_release);
        return true;
    }
};
}  // namespace vsag
