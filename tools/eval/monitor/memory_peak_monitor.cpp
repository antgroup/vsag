
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

#include "memory_peak_monitor.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace vsag::eval {

namespace {

constexpr std::chrono::milliseconds K_DEFAULT_SAMPLE_INTERVAL{5};

MemoryPeakMonitor::MemoryReadResult
read_process_memory() {
#if defined(__linux__)
    errno = 0;
    std::ifstream statm("/proc/self/statm");
    if (not statm.is_open()) {
        const auto error_number = errno;
        std::string error = "failed to open /proc/self/statm";
        if (error_number != 0) {
            error += ": ";
            error += std::strerror(error_number);
        }
        return {false, 0, std::move(error)};
    }

    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    if (not(statm >> total_pages >> resident_pages)) {
        return {false, 0, "failed to parse resident pages from /proc/self/statm"};
    }

    errno = 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        const auto error_number = errno;
        std::string error = "sysconf(_SC_PAGESIZE) failed";
        if (error_number != 0) {
            error += ": ";
            error += std::strerror(error_number);
        }
        return {false, 0, std::move(error)};
    }

    const auto page_size_bytes = static_cast<uint64_t>(page_size);
    if (resident_pages > std::numeric_limits<uint64_t>::max() / page_size_bytes) {
        return {false, 0, "resident memory byte count overflows uint64_t"};
    }
    return {true, resident_pages * page_size_bytes, {}};
#elif defined(__APPLE__)
    mach_task_basic_info_data_t memory_info{};
    mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
    const auto status = task_info(mach_task_self(),
                                  MACH_TASK_BASIC_INFO,
                                  reinterpret_cast<task_info_t>(&memory_info),
                                  &info_count);
    if (status != KERN_SUCCESS) {
        return {false, 0, "task_info failed with status " + std::to_string(status)};
    }
    return {true, static_cast<uint64_t>(memory_info.resident_size), {}};
#else
    return {false, 0, "resident memory sampling is unavailable on this platform"};
#endif
}

std::string
format_bytes(uint64_t bytes) {
    constexpr std::array<const char*, 5> k_units = {"B", "KB", "MB", "GB", "TB"};
    auto size = static_cast<long double>(bytes);
    uint64_t unit_index = 0;
    while (size >= 1024.0L && unit_index + 1 < k_units.size()) {
        size /= 1024.0L;
        ++unit_index;
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << size << " " << k_units[unit_index];
    return stream.str();
}

}  // namespace

MemoryPeakMonitor::MemoryPeakMonitor(std::string name)
    : MemoryPeakMonitor(std::move(name), read_process_memory, K_DEFAULT_SAMPLE_INTERVAL) {
}

MemoryPeakMonitor::MemoryPeakMonitor(std::string name,
                                     MemoryReader reader,
                                     std::chrono::milliseconds sample_interval)
    : Monitor("memory_peak_monitor"),
      reader_(std::move(reader)),
      sample_interval_(std::max(sample_interval, std::chrono::milliseconds{1})),
      process_name_(std::move(name)) {
}

MemoryPeakMonitor::~MemoryPeakMonitor() {
    try {
        stop_sampling();
    } catch (...) {
    }
}

MemoryPeakMonitor::MemoryReadResult
MemoryPeakMonitor::read_memory() const {
    if (not reader_) {
        return {false, 0, "memory reader is not configured"};
    }

    try {
        auto result = reader_();
        if (not result.available && result.error.empty()) {
            result.error = "memory reader reported no data";
        }
        return result;
    } catch (const std::exception& exception) {
        return {false, 0, "memory reader failed: " + std::string(exception.what())};
    } catch (...) {
        return {false, 0, "memory reader failed with an unknown error"};
    }
}

void
MemoryPeakMonitor::Start() {
    Stop();

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        available_ = false;
        baseline_bytes_ = 0;
        absolute_peak_bytes_ = 0;
        sample_count_ = 0;
        failure_count_ = 0;
        last_error_.clear();
        running_ = true;
        worker_ready_ = false;
    }

    try {
        sampling_thread_ = std::thread(&MemoryPeakMonitor::sampling_loop, this);
    } catch (const std::exception& exception) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        running_ = false;
        failure_count_ = 1;
        last_error_ = "failed to start memory sampler: " + std::string(exception.what());
        return;
    }

    bool baseline_available = false;
    {
        std::unique_lock<std::mutex> lock(state_mutex_);
        stop_condition_.wait(lock, [this]() { return worker_ready_; });
        baseline_available = available_;
    }
    if (not baseline_available && sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
}

void
MemoryPeakMonitor::Stop() {
    stop_sampling();
}

void
MemoryPeakMonitor::stop_sampling() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (not running_) {
            return;
        }
        running_ = false;
    }

    stop_condition_.notify_all();
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
}

Monitor::JsonType
MemoryPeakMonitor::GetResult() {
    bool available = false;
    uint64_t baseline_bytes = 0;
    uint64_t absolute_peak_bytes = 0;
    uint64_t sample_count = 0;
    uint64_t failure_count = 0;
    std::string last_error;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        available = available_;
        baseline_bytes = baseline_bytes_;
        absolute_peak_bytes = absolute_peak_bytes_;
        sample_count = sample_count_;
        failure_count = failure_count_;
        last_error = last_error_;
    }

    const uint64_t delta_bytes =
        absolute_peak_bytes >= baseline_bytes ? absolute_peak_bytes - baseline_bytes : 0;

    JsonType result;
    result[metric_name("memory_peak")] = available ? format_bytes(delta_bytes) : "N/A";
    result[metric_name("memory_rss_baseline_bytes")] = baseline_bytes;
    result[metric_name("memory_rss_peak_bytes")] = absolute_peak_bytes;
    result[metric_name("memory_peak_delta_bytes")] = delta_bytes;
    result[metric_name("memory_peak_sample_count")] = sample_count;
    result[metric_name("memory_peak_failed_sample_count")] = failure_count;
    result[metric_name("memory_peak_available")] = available;
    result[metric_name("memory_peak_error")] = last_error;
    return result;
}

void
MemoryPeakMonitor::Record(void* /*input*/) {
}

void
MemoryPeakMonitor::sample_memory() {
    const auto sample = read_memory();
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (sample.available) {
        absolute_peak_bytes_ = std::max(absolute_peak_bytes_, sample.bytes);
        ++sample_count_;
        return;
    }

    ++failure_count_;
    last_error_ = sample.error;
}

void
MemoryPeakMonitor::sampling_loop() {
    const auto baseline = read_memory();
    std::unique_lock<std::mutex> lock(state_mutex_);
    available_ = baseline.available;
    baseline_bytes_ = baseline.available ? baseline.bytes : 0;
    absolute_peak_bytes_ = baseline_bytes_;
    sample_count_ = baseline.available ? 1 : 0;
    failure_count_ = baseline.available ? 0 : 1;
    last_error_ = baseline.error;
    running_ = baseline.available;
    worker_ready_ = true;
    stop_condition_.notify_all();
    if (not running_) {
        return;
    }
    while (
        not stop_condition_.wait_for(lock, sample_interval_, [this]() { return not running_; })) {
        lock.unlock();
        sample_memory();
        lock.lock();
    }
    lock.unlock();
    sample_memory();
}

std::string
MemoryPeakMonitor::metric_name(const std::string& metric) const {
    return metric + "(" + process_name_ + ")";
}

}  // namespace vsag::eval
