
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

#include "merge_index.h"

#include "index/hnsw.h"

namespace vsag {

void
extract_data_and_graph(const std::vector<std::shared_ptr<Index>>& indexes,
                       const DatasetPtr& dataset,
                       Vector<Vector<uint32_t>>& graph) {
    for (const auto& index : indexes) {
        auto stat_string = index->GetStats();
        auto stats = JsonType::parse(stat_string);
        std::string index_name = stats[STATSTIC_INDEX_NAME];
        if (index_name == INDEX_HNSW) {
            auto hnsw = std::dynamic_pointer_cast<HNSW>(index);
            hnsw->ExtractDataAndGraph(dataset, graph);
        }
    }
}
}  // namespace vsag
