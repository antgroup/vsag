
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

#include <mutex>
#include <sstream>

namespace vsag::eval {

static std::string
GetProcFileName(pid_t pid) {
    return "/proc/" + std::to_string(pid) + "/statm";
}

MemoryPeakMonitor::MemoryPeakMonitor(const std::string& name)
    : Monitor("memory_peak_monitor"), process_name_(name) {
    this->pid_ = getpid();
    this->infile_.open(GetProcFileName(pid_));
}

MemoryPeakMonitor::~MemoryPeakMonitor() {
    this->Stop();
}

void
MemoryPeakMonitor::sample() {
    std::lock_guard<std::mutex> lock(record_mutex_);
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    const bool sampled = static_cast<bool>(this->infile_ >> total_pages >> resident_pages);
    this->infile_.clear();
    this->infile_.seekg(0, std::ios::beg);
    if (sampled and resident_pages <= total_pages and max_memory_ < resident_pages) {
        max_memory_ = resident_pages;
    }
}

void
MemoryPeakMonitor::Start() {
    this->sample();
    init_memory_ = max_memory_;
    {
        std::lock_guard<std::mutex> lock(sampling_mutex_);
        sampling_ = true;
    }
    sampling_thread_ = std::thread([this]() {
        std::unique_lock<std::mutex> lock(sampling_mutex_);
        while (sampling_) {
            if (sampling_condition_.wait_for(
                    lock, std::chrono::milliseconds(10), [this]() { return not sampling_; })) {
                break;
            }
            lock.unlock();
            this->sample();
            lock.lock();
        }
    });
}
void
MemoryPeakMonitor::Stop() {
    {
        std::lock_guard<std::mutex> lock(sampling_mutex_);
        if (not sampling_) {
            return;
        }
        sampling_ = false;
    }
    sampling_condition_.notify_one();
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
    this->sample();
}
Monitor::JsonType
MemoryPeakMonitor::GetResult() {
    JsonType result;
    std::vector<std::string> metrics = {"B", "KB", "MB", "GB", "TB"};
    const auto page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const auto increment_pages = max_memory_ > init_memory_ ? max_memory_ - init_memory_ : 0;
    auto size = static_cast<float>(increment_pages * page_size);
    size_t i = 0;
    while (size >= 1024.0F && i < metrics.size() - 1) {
        size /= 1024;
        i++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size;
    result["memory_peak(" + process_name_ + ")"] = oss.str() + " " + metrics[i];
    result["memory_rss_before_build(B)"] = init_memory_ * page_size;
    result["memory_rss_peak_build(B)"] = max_memory_ * page_size;
    result["memory_rss_increment_build(B)"] = increment_pages * page_size;
    result["memory_rss_sample_interval_ms"] = 10;
    return result;
}
void
MemoryPeakMonitor::Record(void* input) {
    this->sample();
}

}  // namespace vsag::eval
