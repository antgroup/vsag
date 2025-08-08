
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

#include "timer.h"

#include <numeric>

namespace vsag {
Timer::Timer(double* ref) : ref_(ref) {
    start = std::chrono::steady_clock::now();
    this->threshold_ = std::numeric_limits<double>::max();
}

Timer::Timer(double& ref) : Timer(&ref){};

Timer::Timer() : Timer(nullptr){};

double
Timer::Record() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    return duration.count();
}

bool
Timer::CheckOvertime() {
    if (threshold_ == std::numeric_limits<double>::max()) {
        return false;
    }
    double duration = Record();
    return duration > threshold_;
}

void
Timer::SetThreshold(double threshold) {
    threshold_ = threshold;
}

Timer::~Timer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (ref_ != nullptr) {
        *ref_ = duration.count();
    }
}
}  // namespace vsag
