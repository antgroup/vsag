
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

#include "graph_interface.h"

namespace vsag {
class GraphInterfaceTest {
public:
    explicit GraphInterfaceTest(GraphInterfacePtr graph, const bool require_sorted = false)
        : graph_(std::move(graph)), require_sorted_(require_sorted) {
    }

    void
    BasicTest(uint64_t max_id, uint64_t count, const GraphInterfacePtr& other, bool test_delete);

public:
    GraphInterfacePtr graph_{nullptr};
    bool require_sorted_{false};
};
}  // namespace vsag
