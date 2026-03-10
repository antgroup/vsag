
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

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common.h"

namespace vsag::eval {

class HttpServerMonitor {
public:
    using TimePoint = std::chrono::steady_clock::time_point;

    explicit HttpServerMonitor(int port = 8080);
    ~HttpServerMonitor();

    // Server control
    void
    Start();

    void
    Stop();

    bool
    IsRunning() const {
        return running_;
    }
    int
    GetPort() const {
        return port_;
    }

    // Progress tracking
    void
    SetTotalCases(int total);

    void
    SetCurrentCase(const std::string& name);

    void
    SetCaseStatus(const std::string& status);  // "pending", "running", "completed", "failed"

    void
    MarkCaseCompleted();

    // Metrics update - called by EvalCase during execution
    void
    UpdateMetrics(const JsonType& metrics);

    void
    UpdateProgress(float percent_complete);  // 0.0 - 100.0

    // Static files
    void
    SetWebRoot(const std::string& path);

private:
    void
    setup_routes();

    void
    run_server();

    JsonType
    get_status_json() const;

    std::string
    get_index_html() const;

private:
    // HTTP server
    std::unique_ptr<httplib::Server> server_;
    std::thread server_thread_;
    int port_;
    std::atomic<bool> running_{false};
    std::string web_root_;

    // State data (protected by mutex)
    mutable std::mutex state_mutex_;
    int total_cases_{0};
    int completed_cases_{0};
    std::string current_case_;
    std::string case_status_{"idle"};  // idle, running, completed, failed
    float current_progress_{0.0F};     // 0.0 - 100.0
    JsonType current_metrics_;
    TimePoint start_time_;
    TimePoint case_start_time_;
    std::vector<JsonType> case_history_;  // Completed cases data
};

}  // namespace vsag::eval
