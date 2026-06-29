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

#include "pyramid_build_cache.h"

namespace vsag {

PyramidBuildCache::PyramidBuildCache(Allocator* allocator)
    : allocator_(allocator), graph_caches_(allocator_) {
}

void
PyramidBuildCache::Serialize(StreamWriter& writer) const {
    uint64_t graph_count = graph_caches_.size();
    StreamWriter::WriteObj(writer, graph_count);
    for (const auto& [graph_key, graph_cache] : graph_caches_) {
        StreamWriter::WriteString(writer, graph_key);
        graph_cache->Serialize(writer);
    }
}

void
PyramidBuildCache::Deserialize(StreamReader& reader) {
    uint64_t graph_count = 0;
    StreamReader::ReadObj(reader, graph_count);
    graph_caches_.clear();
    for (uint64_t i = 0; i < graph_count; ++i) {
        auto graph_key = StreamReader::ReadString(reader);
        auto graph_cache = std::make_unique<BuildCache>(allocator_);
        graph_cache->Deserialize(reader);
        graph_caches_.emplace(std::move(graph_key), std::move(graph_cache));
    }
}

BuildCache*
PyramidBuildCache::GetGraphCache(const std::string& hierarchy_name,
                                 const std::string& node_path) const {
    auto key = MakeGraphKey(hierarchy_name, node_path);
    auto it = graph_caches_.find(key);
    if (it == graph_caches_.end()) {
        return nullptr;
    }
    return it->second.get();
}

BuildCache&
PyramidBuildCache::CreateGraphCache(const std::string& hierarchy_name,
                                    const std::string& node_path) {
    auto key = MakeGraphKey(hierarchy_name, node_path);
    auto it = graph_caches_.find(key);
    if (it == graph_caches_.end()) {
        auto graph_cache = std::make_unique<BuildCache>(allocator_);
        auto result = graph_caches_.emplace(std::move(key), std::move(graph_cache));
        it = result.first;
    }
    return *it->second;
}

std::string
PyramidBuildCache::MakeGraphKey(const std::string& hierarchy_name, const std::string& node_path) {
    return std::to_string(hierarchy_name.size()) + ":" + hierarchy_name + node_path;
}

}  // namespace vsag
