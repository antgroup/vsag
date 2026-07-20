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

#include <cstdint>

#include "basic_types.h"
#include "metric_type.h"
#include "typing.h"

namespace vsag {

struct MCIGraphView {
    const InnerIdType* neighbors{nullptr};
    const uint32_t* counts{nullptr};
    uint64_t total{0};
    uint64_t row_stride{0};
    uint64_t uniform_count{0};
};

struct MCIV3BuildParams {
    uint64_t total{0};
    uint64_t dim{0};
    uint64_t candidate_limit{0};
    uint64_t clique_max{50};
    uint64_t max_degree{32};
    float alpha{1.2F};
    uint64_t thread_count{1};
    MetricType metric{MetricType::METRIC_TYPE_L2SQR};
};

Vector<Vector<InnerIdType>>
BuildMCICliques(const float* vectors,
                const MCIGraphView& graph,
                const MCIV3BuildParams& params,
                Allocator* allocator);

}  // namespace vsag
