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

#include "vsag_exception.h"

namespace vsag {
namespace {

uint64_t
remaining_bytes(StreamReader& reader) {
    const auto cursor = reader.GetCursor();
    const auto length = reader.Length();
    if (cursor > length) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted Pyramid build cache cursor");
    }
    return length - cursor;
}

std::string
read_string(StreamReader& reader) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    if (size > remaining_bytes(reader)) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "corrupted Pyramid build cache string length");
    }
    std::vector<char> buffer(size);
    reader.Read(buffer.data(), size);
    return {buffer.data(), size};
}

}  // namespace

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
    if (graph_count > remaining_bytes(reader) / (sizeof(uint64_t) * 2)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted Pyramid build cache graph count");
    }
    graph_caches_.clear();
    for (uint64_t i = 0; i < graph_count; ++i) {
        auto graph_key = read_string(reader);
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
    // The length prefix makes the hierarchy-name/path boundary unambiguous for arbitrary strings.
    return std::to_string(hierarchy_name.size()) + ":" + hierarchy_name + node_path;
}

}  // namespace vsag
