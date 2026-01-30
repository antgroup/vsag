
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

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

namespace vsag::eval {

class ProgressBar {
public:
    ProgressBar(const ProgressBar&) = delete;
    ProgressBar&
    operator=(const ProgressBar&) = delete;
    ProgressBar(ProgressBar&&) = delete;
    ProgressBar&
    operator=(ProgressBar&&) = delete;

    ProgressBar(std::string name, uint64_t total, uint64_t report_interval_ms = 1000)
        : name_(std::move(name)),
          total_(total),
          report_interval_ms_(report_interval_ms),
          running_(false),
          current_(0) {
    }

    ~ProgressBar() {
        Finish();
    }

    void
    Update(uint64_t step = 1) {
        current_.fetch_add(step, std::memory_order_relaxed);
    }

    void
    Start() {
        if (running_) {
            return;
        }
        running_ = true;
        start_time_ = std::chrono::steady_clock::now();
        reporter_thread_ = std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(report_interval_ms_));
                if (!running_) {
                    break;
                }
                Print();
            }
        });
    }

    void
    Finish() {
        if (!running_) {
            return;
        }
        running_ = false;
        if (reporter_thread_.joinable()) {
            reporter_thread_.join();
        }
        Print(true);
        std::cout << std::endl;
    }

private:
    void
    Print(bool force_end = false) {
        std::lock_guard<std::mutex> lock(cout_mutex_);
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        uint64_t cur = current_.load(std::memory_order_relaxed);

        std::stringstream ss;
        if (total_ > 0) {
            double percent = (double)cur / total_ * 100.0;
            if (percent > 100.0) {
                percent = 100.0;
            }
            ss << "\r"
               << "[" << name_ << "] " << cur << "/" << total_ << " (" << std::fixed
               << std::setprecision(1) << percent << "%) "
               << "Elapsed: " << elapsed << "s";
        } else {
            ss << "\r"
               << "[" << name_ << "] "
               << "Elapsed: " << elapsed << "s";
        }
        std::cout << ss.str() << std::flush;
    }

    std::string name_;
    uint64_t total_;
    uint64_t report_interval_ms_;
    std::atomic<uint64_t> current_;
    std::atomic<bool> running_;
    std::thread reporter_thread_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    inline static std::mutex cout_mutex_;
};

}  // namespace vsag::eval
