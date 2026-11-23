// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>

namespace vsag {

// Invocation-local diagnostics. Never shared by concurrent distance workers.
struct SparseQueryProfile {
    uint64_t read_ns{0};
    uint64_t compute_ns{0};
    uint64_t read_bytes{0};
    uint64_t read_ranges{0};
    const char* io_strategy{"unavailable"};
};

class QueryProfileTimer {
public:
    using Clock = std::chrono::steady_clock;

    explicit QueryProfileTimer(uint64_t* elapsed)
        : elapsed_(elapsed), start_(elapsed ? Clock::now() : Clock::time_point{}) {
    }

    ~QueryProfileTimer() {
        if (elapsed_) {
            *elapsed_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_)
                    .count());
        }
    }

    QueryProfileTimer(const QueryProfileTimer&) = delete;
    QueryProfileTimer&
    operator=(const QueryProfileTimer&) = delete;

private:
    uint64_t* elapsed_;
    Clock::time_point start_;
};

}  // namespace vsag
