
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

#include <algorithm>
#include <random>

#include "flatten_interface.h"

namespace vsag {
class FlattenInterfaceTest {
public:
    FlattenInterfaceTest(FlattenInterfacePtr flatten, MetricType metric)
        : flatten_(flatten), metric_(metric){};

    void
    BasicTest(int64_t dim, uint64_t base_count, float error = 1e-5f);

    void
    TestSerializeAndDeserialize(int64_t dim, FlattenInterfacePtr other, float error = 1e-5f);

public:
    FlattenInterfacePtr flatten_{nullptr};

    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
};
}  // namespace vsag
