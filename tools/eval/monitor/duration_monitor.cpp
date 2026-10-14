
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

#include "duration_monitor.h"

namespace vsag::eval {

namespace {

double
TimevalSeconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1'000'000.0;
}

}  // namespace

DurationMonitor::DurationMonitor() : Monitor("duration_monitor") {
}

void
DurationMonitor::Start() {
    getrusage(RUSAGE_SELF, &start_usage_);
    cur_time_ = Clock::now();
}
void
DurationMonitor::Stop() {
    auto end_time = Clock::now();
    struct rusage end_usage {};
    getrusage(RUSAGE_SELF, &end_usage);
    this->duration_ = std::chrono::duration<double>(end_time - cur_time_).count();
    this->user_cpu_duration_ =
        TimevalSeconds(end_usage.ru_utime) - TimevalSeconds(start_usage_.ru_utime);
    this->system_cpu_duration_ =
        TimevalSeconds(end_usage.ru_stime) - TimevalSeconds(start_usage_.ru_stime);
}
Monitor::JsonType
DurationMonitor::GetResult() {
    JsonType result;
    result["duration(s)"] = this->duration_;
    result["user_cpu_duration(s)"] = this->user_cpu_duration_;
    result["system_cpu_duration(s)"] = this->system_cpu_duration_;
    const auto cpu_duration = this->user_cpu_duration_ + this->system_cpu_duration_;
    result["cpu_duration(s)"] = cpu_duration;
    result["average_cpu_cores"] = this->duration_ > 0 ? cpu_duration / this->duration_ : 0.0;
    return result;
}

}  // namespace vsag::eval
