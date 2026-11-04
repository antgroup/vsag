
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

#include <cstdint>
#include <vector>

#include "monitor.h"
namespace vsag::eval {

struct LatencyTimingBatch {
    std::vector<double> latency_ms;
    uint64_t successful_query_count{0};
    double wall_time_seconds{0.0};
};

class LatencyMonitor : public Monitor {
public:
    LatencyMonitor();

    ~LatencyMonitor() override = default;

    void
    Start() override;

    void
    Stop() override;

    JsonType
    GetResult() override;

    void
    SetTimingBatch(LatencyTimingBatch timing_batch);

    void
    SetMetrics(std::string metric);

private:
    void
    cal_and_set_result(const std::string& metric, JsonType& result);

    double
    cal_qps() const;

    double
    cal_avg_latency() const;

    double
    cal_latency_rate(double rate);

private:
    std::vector<double> latency_records_;

    uint64_t successful_query_count_{0};

    double wall_time_seconds_{0.0};

    std::vector<std::string> metrics_;
};

}  // namespace vsag::eval
