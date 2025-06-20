
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

namespace vsag::eval {

static std::string
GetProcFileName(pid_t pid) {
    return "/proc/" + std::to_string(pid) + "/statm";
}

MemoryPeakMonitor::MemoryPeakMonitor(const std::string& name)
    : Monitor("memory_peak_monitor"), process_name_(name) {
    this->pid_ = getpid();
    this->infile_.open(GetProcFileName(pid_));
    uint64_t val1, val2;
    this->infile_ >> val1 >> val2;
    this->infile_.clear();
    this->infile_.seekg(0, std::ios::beg);
    init_memory_ = val2;
}

void
MemoryPeakMonitor::Start() {
}
void
MemoryPeakMonitor::Stop() {
}
Monitor::JsonType
MemoryPeakMonitor::GetResult() {
    JsonType result;
    std::vector<std::string> metrics = {"B", "KB", "MB", "GB", "TB"};
    auto size =
        static_cast<float>((this->max_memory_ - this->init_memory_) * sysconf(_SC_PAGESIZE));
    size_t i = 0;
    while (size >= 1024.0F && i < metrics.size() - 1) {
        size /= 1024;
        i++;
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size;
    result["memory_peak(" + process_name_ + ")"] = oss.str() + " " + metrics[i];
    return result;
}
void
MemoryPeakMonitor::Record(void* input) {
    std::lock_guard<std::mutex> lock(record_mutex_);

    uint64_t val1, val2;
    this->infile_ >> val1 >> val2;
    this->infile_.clear();
    this->infile_.seekg(0, std::ios::beg);
    if (max_memory_ < val2) {
        max_memory_ = val2;
    }
}

}  // namespace vsag::eval
