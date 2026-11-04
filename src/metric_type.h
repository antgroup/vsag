
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

#include "vsag/constants.h"

namespace vsag {
enum class MetricType {
    METRIC_TYPE_L2SQR = 0,
    METRIC_TYPE_IP = 1,
    METRIC_TYPE_COSINE = 2,
    // popcount(a XOR b) over bit-packed binary vectors; only valid with DATA_TYPE_BINARY.
    METRIC_TYPE_HAMMING = 3,
};

constexpr const char*
ToString(MetricType metric) noexcept {
    switch (metric) {
        case MetricType::METRIC_TYPE_L2SQR:
            return METRIC_L2;
        case MetricType::METRIC_TYPE_IP:
            return METRIC_IP;
        case MetricType::METRIC_TYPE_COSINE:
            return METRIC_COSINE;
        case MetricType::METRIC_TYPE_HAMMING:
            return METRIC_HAMMING;
    }
    return "unknown";
}

}  // namespace vsag
