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

#include <limits>

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
    : allocator_(allocator), graph_caches_(allocator_), group_owners_(allocator_) {
}

void
PyramidBuildCache::Serialize(StreamWriter& writer) const {
    uint64_t graph_count = graph_caches_.size();
    // Impossible legacy graph count; BuildCache (and HGraph) remain unchanged.
    const bool has_groups = not group_owners_.empty();
    if (has_groups) {
        StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());
        StreamWriter::WriteObj(writer, uint64_t{1});
    }
    StreamWriter::WriteObj(writer, graph_count);
    for (const auto& [graph_key, graph_cache] : graph_caches_) {
        StreamWriter::WriteString(writer, graph_key);
        graph_cache->Serialize(writer);
        if (not has_groups) {
            continue;
        }
        auto owners = group_owners_.find(graph_key);
        const uint64_t count = owners == group_owners_.end() ? 0 : owners->second.size();
        StreamWriter::WriteObj(writer, count);
        if (owners != group_owners_.end()) {
            for (const auto owner : owners->second) {
                StreamWriter::WriteObj(writer, owner);
            }
        }
    }
}

void
PyramidBuildCache::Deserialize(StreamReader& reader) {
    uint64_t graph_count = 0;
    StreamReader::ReadObj(reader, graph_count);
    const bool has_groups = graph_count == std::numeric_limits<uint64_t>::max();
    if (has_groups) {
        uint64_t version = 0;
        StreamReader::ReadObj(reader, version);
        if (version != 1) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "unknown Pyramid build-cache payload version (got " +
                                    std::to_string(version) + ", expected 1)");
        }
        StreamReader::ReadObj(reader, graph_count);
    }
    if (graph_count > remaining_bytes(reader) / (sizeof(uint64_t) * 2)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted Pyramid build cache graph count");
    }
    UnorderedMap<std::string, std::unique_ptr<BuildCache>> graphs(allocator_);
    UnorderedMap<std::string, GroupOwners> groups(allocator_);
    for (uint64_t i = 0; i < graph_count; ++i) {
        auto graph_key = read_string(reader);
        auto graph_cache = std::make_unique<BuildCache>(allocator_);
        graph_cache->Deserialize(reader);
        if (has_groups) {
            uint64_t count = 0;
            StreamReader::ReadObj(reader, count);
            if (count > remaining_bytes(reader) / sizeof(InnerIdType) ||
                (count != 0 && count != graph_cache->source_ids_.size())) {
                throw VsagException(ErrorType::INVALID_BINARY, "invalid Pyramid group count");
            }
            GroupOwners owners(count);
            for (auto& owner : owners) {
                StreamReader::ReadObj(reader, owner);
                if (owner >= count) {
                    throw VsagException(ErrorType::INVALID_BINARY, "invalid Pyramid group owner");
                }
            }
            for (const auto owner : owners) {
                if (owners[owner] != owner) {
                    throw VsagException(ErrorType::INVALID_BINARY, "noncanonical Pyramid group");
                }
            }
            if (count != 0) {
                groups.emplace(graph_key, std::move(owners));
            }
        }
        if (not graphs.emplace(std::move(graph_key), std::move(graph_cache)).second) {
            throw VsagException(ErrorType::INVALID_BINARY, "duplicate Pyramid graph key");
        }
    }
    graph_caches_.swap(graphs);
    group_owners_.swap(groups);
}

const PyramidBuildCache::GroupOwners*
PyramidBuildCache::GetGroupOwners(const std::string& hierarchy_name,
                                  const std::string& node_path) const {
    auto it = group_owners_.find(MakeGraphKey(hierarchy_name, node_path));
    return it == group_owners_.end() ? nullptr : &it->second;
}

void
PyramidBuildCache::SetGroupOwners(const std::string& hierarchy_name,
                                  const std::string& node_path,
                                  GroupOwners owners) {
    const auto* graph = GetGraphCache(hierarchy_name, node_path);
    if (graph == nullptr || owners.size() != graph->source_ids_.size()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Pyramid group metadata must cover the graph source IDs");
    }
    for (const auto owner : owners) {
        if (owner >= owners.size()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "Pyramid group owner out of range");
        }
    }
    for (const auto owner : owners) {
        if (owners[owner] != owner) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Pyramid group owner is not canonical");
        }
    }
    group_owners_.insert_or_assign(MakeGraphKey(hierarchy_name, node_path), std::move(owners));
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

uint64_t
PyramidBuildCache::CountMatchedSourceIds(const std::string* source_ids, uint64_t count) const {
    UnorderedSet<std::string> cached_source_ids(allocator_);
    cached_source_ids.reserve(count);
    for (const auto& [key, graph_cache] : graph_caches_) {
        if (graph_cache == nullptr) {
            continue;
        }
        for (const auto& source_id : graph_cache->source_ids_) {
            cached_source_ids.emplace(source_id);
        }
    }
    uint64_t matched = 0;
    for (uint64_t index = 0; index < count; ++index) {
        if (cached_source_ids.find(source_ids[index]) != cached_source_ids.end()) {
            ++matched;
        }
    }
    return matched;
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
