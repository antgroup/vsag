
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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "monitor.h"

namespace vsag::eval {

class MemoryPeakMonitor : public Monitor {
public:
    class MemoryReadResult {
    public:
        bool available{false};
        uint64_t bytes{0};
        std::string error{};
    };

    using MemoryReader = std::function<MemoryReadResult()>;

    explicit MemoryPeakMonitor(std::string name);

    MemoryPeakMonitor(std::string name,
                      MemoryReader reader,
                      std::chrono::milliseconds sample_interval);

    ~MemoryPeakMonitor() override;

    void
    Start() override;

    void
    Stop() override;

    JsonType
    GetResult() override;

    void
    Record(void* input = nullptr) override;

private:
    MemoryReadResult
    read_memory() const;

    void
    sample_memory();

    void
    sampling_loop();

    void
    stop_sampling();

    std::string
    metric_name(const std::string& metric) const;

private:
    MemoryReader reader_;
    std::chrono::milliseconds sample_interval_;
    std::string process_name_;

    mutable std::mutex state_mutex_;
    std::condition_variable stop_condition_;
    std::thread sampling_thread_;
    bool running_{false};
    bool worker_ready_{false};

    bool available_{false};
    uint64_t baseline_bytes_{0};
    uint64_t absolute_peak_bytes_{0};
    uint64_t sample_count_{0};
    uint64_t failure_count_{0};
    std::string last_error_{"memory monitor has not been started"};
};

}  // namespace vsag::eval
