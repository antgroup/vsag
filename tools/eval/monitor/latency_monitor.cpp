
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

#include "latency_monitor.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace vsag::eval {

LatencyMonitor::LatencyMonitor() : Monitor("latency_monitor") {
}

void
LatencyMonitor::Start() {
    this->latency_records_.clear();
    this->successful_query_count_ = 0;
    this->wall_time_seconds_ = 0.0;
}
void
LatencyMonitor::Stop() {
}

Monitor::JsonType
LatencyMonitor::GetResult() {
    JsonType result;
    for (auto& metric : metrics_) {
        this->cal_and_set_result(metric, result);
    }
    result["measurement_method"]["latency"] = "steady_clock_around_knn_search";
    result["measurement_method"]["qps"] = "successful_queries_per_wall_time";
    result["measurement_sample_count"] = static_cast<uint64_t>(this->latency_records_.size());
    result["measurement_successful_query_count"] = this->successful_query_count_;
    result["measurement_duration(s)"] = this->wall_time_seconds_;
    return result;
}

void
LatencyMonitor::SetTimingBatch(LatencyTimingBatch timing_batch) {
    this->latency_records_ = std::move(timing_batch.latency_ms);
    this->latency_records_.erase(
        std::remove_if(
            this->latency_records_.begin(),
            this->latency_records_.end(),
            [](double latency_ms) { return not std::isfinite(latency_ms) or latency_ms < 0; }),
        this->latency_records_.end());
    this->successful_query_count_ = timing_batch.successful_query_count;
    this->wall_time_seconds_ =
        std::isfinite(timing_batch.wall_time_seconds) and timing_batch.wall_time_seconds > 0.0
            ? timing_batch.wall_time_seconds
            : 0.0;
}

void
LatencyMonitor::SetMetrics(std::string metric) {
    this->metrics_.emplace_back(std::move(metric));
}
void
LatencyMonitor::cal_and_set_result(const std::string& metric, Monitor::JsonType& result) {
    if (metric == "qps") {
        auto val = this->cal_qps();
        result["qps"] = val;
    } else if (metric == "avg_latency") {
        auto val = this->cal_avg_latency();
        result["latency_avg(ms)"] = val;
    } else if (metric == "percent_latency") {
        std::vector<double> percents = {50, 80, 90, 95, 99};
        for (auto& percent : percents) {
            auto val = this->cal_latency_rate(percent * 0.01);
            result["latency_detail(ms)"]["p" + std::to_string(int(percent))] = val;
        }
    }
}

double
LatencyMonitor::cal_qps() const {
    if (this->successful_query_count_ == 0 or this->wall_time_seconds_ <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(this->successful_query_count_) / this->wall_time_seconds_;
}

double
LatencyMonitor::cal_avg_latency() const {
    if (this->latency_records_.empty()) {
        return 0.0;
    }
    double total_time_cost =
        std::accumulate(this->latency_records_.begin(), this->latency_records_.end(), double(0));
    return total_time_cost / static_cast<double>(this->latency_records_.size());
}

double
LatencyMonitor::cal_latency_rate(double rate) {
    if (this->latency_records_.empty()) {
        return 0.0;
    }
    std::sort(this->latency_records_.begin(), this->latency_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->latency_records_.size() - 1));
    return latency_records_[pos];
}
}  // namespace vsag::eval
