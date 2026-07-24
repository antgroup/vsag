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

#include "search_pass_runner.h"

#include <algorithm>
#include <stdexcept>

namespace vsag::eval {

namespace {

class MonitorPassGuard {
public:
    explicit MonitorPassGuard(const MonitorPtr& monitor) : monitor_(monitor) {
    }

    ~MonitorPassGuard() {
        if (started_) {
            try {
                monitor_->Stop();
            } catch (...) {
            }
        }
    }

    void
    Start() {
        monitor_->Start();
        started_ = true;
    }

    void
    Stop() {
        monitor_->Stop();
        started_ = false;
    }

private:
    const MonitorPtr& monitor_;
    bool started_{false};
};

void
run_monitored_pass(const MonitorPtr& monitor,
                   const MonitorPtr& memory_monitor,
                   const SearchPassRunner::SearchPass& search_pass) {
    MonitorPassGuard monitor_guard(monitor);
    monitor_guard.Start();

    if (memory_monitor != nullptr) {
        MonitorPassGuard memory_guard(memory_monitor);
        memory_guard.Start();
        search_pass(monitor);
        memory_guard.Stop();
    } else {
        search_pass(monitor);
    }

    monitor_guard.Stop();
}

}  // namespace

void
SearchPassRunner::Run(const std::vector<MonitorPtr>& monitors,
                      const MonitorPtr& latency_monitor,
                      const MonitorPtr& memory_monitor,
                      const SearchPass& search_pass,
                      const StatisticsPass& statistics_pass) {
    if (not search_pass) {
        throw std::invalid_argument("search pass callback is required");
    }
    if (not statistics_pass) {
        throw std::invalid_argument("statistics pass callback is required");
    }
    if (std::any_of(monitors.begin(), monitors.end(), [](const MonitorPtr& monitor) {
            return monitor == nullptr;
        })) {
        throw std::invalid_argument("search pass monitor must not be null");
    }
    if (latency_monitor != nullptr && latency_monitor == memory_monitor) {
        throw std::invalid_argument("latency and memory monitors must be distinct");
    }

    const bool has_latency_pass =
        latency_monitor != nullptr &&
        std::find(monitors.begin(), monitors.end(), latency_monitor) != monitors.end();
    bool ran_query_pass = false;

    if (memory_monitor != nullptr && not has_latency_pass) {
        MonitorPassGuard memory_guard(memory_monitor);
        memory_guard.Start();
        search_pass(nullptr);
        ran_query_pass = true;
        memory_guard.Stop();
    }

    for (const auto& monitor : monitors) {
        const auto shared_memory_monitor =
            monitor == latency_monitor ? memory_monitor : MonitorPtr{nullptr};
        run_monitored_pass(monitor, shared_memory_monitor, search_pass);
        ran_query_pass = true;
    }

    if (ran_query_pass) {
        statistics_pass();
    }
}

}  // namespace vsag::eval
