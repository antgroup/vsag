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

#include <string>
#include <vector>

#include "basic_types.h"
#include "common.h"

namespace vsag {

class HGraph;
class Dataset;
using DatasetPtr = std::shared_ptr<Dataset>;

class HGraphBuilder {
public:
    static void
    Train(HGraph& graph, const DatasetPtr& base);

    static std::vector<int64_t>
    BuildByODescent(HGraph& graph, const DatasetPtr& data);

    static void
    AddOnePoint(HGraph& graph, const void* data, int level, InnerIdType inner_id);

    static bool
    GraphAddOne(HGraph& graph, const void* data, int level, InnerIdType inner_id);

    static void
    Resize(HGraph& graph, uint64_t new_size);

    static void
    ELPOptimize(HGraph& graph);
};

}  // namespace vsag
