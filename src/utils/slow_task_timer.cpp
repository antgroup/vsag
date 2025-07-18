
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

#include "slow_task_timer.h"

#include "logger.h"

namespace vsag {
SlowTaskTimer::SlowTaskTimer(std::string n, int64_t log_threshold_ms)
    : name(std::move(n)), threshold(log_threshold_ms) {
    start = std::chrono::steady_clock::now();
}

SlowTaskTimer::~SlowTaskTimer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (duration.count() > static_cast<double>(threshold)) {
        if (duration.count() >= 1000) {
            logger::info("{0} cost {1:.3f}s", name, duration.count() / 1000);
        } else {
            logger::info("{0} cost {1:.3f}ms", name, duration.count());
        }
    }
}
}  // namespace vsag
