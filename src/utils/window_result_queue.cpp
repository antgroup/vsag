
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

#include "window_result_queue.h"

namespace vsag {

static constexpr int64_t DEFAULT_WATCH_WINDOW_SIZE = 20;

WindowResultQueue::WindowResultQueue() {
    queue_.resize(DEFAULT_WATCH_WINDOW_SIZE);
}

void
WindowResultQueue::Push(float value) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    uint64_t pos = count_.load(std::memory_order_relaxed);
    uint64_t window_size = queue_.size();
    queue_[pos % window_size] = value;
    count_.fetch_add(1, std::memory_order_relaxed);
}

float
WindowResultQueue::GetAvgResult() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    uint64_t statistic_num =
        std::min<uint64_t>(count_.load(std::memory_order_relaxed), queue_.size());
    if (statistic_num == 0) {
        return 0.0F;
    }
    float result = 0;
    for (uint64_t i = 0; i < statistic_num; i++) {
        result += queue_[i];
    }
    return result / static_cast<float>(statistic_num);
}

}  // namespace vsag
