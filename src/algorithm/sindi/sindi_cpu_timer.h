// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ctime>

namespace vsag {

struct SindiCpuTime {
    uint64_t nanoseconds{0};
    bool available{false};
};

// Two thread-CPU clock reads per participating thread, only when explicitly enabled.
class SindiCpuTimer {
public:
    SindiCpuTimer(bool enabled, SindiCpuTime& result) : result_(result) {
        running_ = enabled and Read(begin_);
    }

    ~SindiCpuTimer() {
        Stop();
    }

    void
    Stop() {
        if (running_) {
            uint64_t end = 0;
            result_.available = Read(end) and end >= begin_;
            if (result_.available) {
                result_.nanoseconds = end - begin_;
            }
            running_ = false;
        }
    }

private:
    static bool
    Read(uint64_t& value) {
#ifdef CLOCK_THREAD_CPUTIME_ID
        timespec time{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time) == 0) {
            value = uint64_t(time.tv_sec) * 1000000000ULL + uint64_t(time.tv_nsec);
            return true;
        }
#endif
        return false;
    }

    SindiCpuTime& result_;
    uint64_t begin_{0};
    bool running_{false};
};

}  // namespace vsag
