
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

DurationMonitor::DurationMonitor() : Monitor("duration_monitor") {
}

void
DurationMonitor::Start() {
    cur_time_ = Clock::now();
}
void
DurationMonitor::Stop() {
    auto end_time = Clock::now();
    this->duration_ = std::chrono::duration<double>(end_time - cur_time_).count();
}
Monitor::JsonType
DurationMonitor::GetResult() {
    JsonType result;
    result["duration(s)"] = this->duration_;
    return result;
}

}  // namespace vsag::eval
