
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
#include <thread>
#include <unordered_map>

#include "monitor.h"
namespace vsag::eval {

class LatencyMonitor : public Monitor {
public:
    explicit LatencyMonitor(uint64_t max_record_counts = 0);

    ~LatencyMonitor() override = default;

    void
    Start() override;

    void
    Stop() override;

    JsonType
    GetResult() override;

    void
    Record(void* input) override;

    void
    SetMetrics(std::string metric);

private:
    void
    cal_and_set_result(const std::string& metric, JsonType& result);

    double
    cal_qps();

    double
    cal_avg_latency();

    double
    cal_latency_rate(double rate);

private:
    std::vector<double> latency_records_;

    using Clock = std::chrono::high_resolution_clock;
    std::unordered_map<std::thread::id, decltype(Clock::now())> cur_time_;

    std::vector<std::string> metrics_;
};

}  // namespace vsag::eval
