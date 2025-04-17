
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

#include <mutex>

namespace vsag::eval {

LatencyMonitor::LatencyMonitor(uint64_t max_record_counts) : Monitor("latency_monitor") {
    if (max_record_counts > 0) {
        this->latency_records_.reserve(max_record_counts);
    }
}

void
LatencyMonitor::Start() {
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
    return result;
}
void
LatencyMonitor::Record(void* input) {
    std::lock_guard<std::mutex> lock(record_mutex_);
    std::thread::id thread_id = std::this_thread::get_id();
    if (cur_time_.find(thread_id) == cur_time_.end()) {
        cur_time_[thread_id] = Clock::now();
        return;
    }
    auto end_time = Clock::now();
    double duration =
        std::chrono::duration<double, std::milli>(end_time - cur_time_[thread_id]).count();
    this->latency_records_.emplace_back(duration);
    this->cur_time_[thread_id] = Clock::now();
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
LatencyMonitor::cal_qps() {
    double total_time_cost =
        std::accumulate(this->latency_records_.begin(), this->latency_records_.end(), double(0));
    auto thread_num = cur_time_.size();
    auto query_num = latency_records_.size();
    return static_cast<double>(query_num) * thread_num * 1000.0 / total_time_cost;
}

double
LatencyMonitor::cal_avg_latency() {
    double total_time_cost =
        std::accumulate(this->latency_records_.begin(), this->latency_records_.end(), double(0));
    auto query_num = latency_records_.size();
    return total_time_cost / static_cast<double>(query_num);
}
double
LatencyMonitor::cal_latency_rate(double rate) {
    std::sort(this->latency_records_.begin(), this->latency_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->latency_records_.size() - 1));
    return latency_records_[pos];
}
}  // namespace vsag::eval
