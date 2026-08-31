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

#include "build_cache.h"

#include "impl/allocator/default_allocator.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

uint64_t
remaining_bytes(StreamReader& reader) {
    const auto cursor = reader.GetCursor();
    const auto length = reader.Length();
    if (cursor > length) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache cursor");
    }
    return length - cursor;
}

void
require_remaining(StreamReader& reader, uint64_t size, const char* field) {
    if (size > remaining_bytes(reader)) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("corrupted build cache {} length", field));
    }
}

std::string
read_string(StreamReader& reader) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    require_remaining(reader, size, "string");
    std::vector<char> buffer(size);
    reader.Read(buffer.data(), size);
    return {buffer.data(), size};
}

void
read_neighbors(StreamReader& reader, Vector<InnerIdType>& neighbors) {
    uint64_t size = 0;
    StreamReader::ReadObj(reader, size);
    if (size > remaining_bytes(reader) / sizeof(InnerIdType)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache neighbor count");
    }
    neighbors.resize(size);
    reader.Read(reinterpret_cast<char*>(neighbors.data()), size * sizeof(InnerIdType));
}

}  // namespace

BuildCache::BuildCache(Allocator* allocator)
    : allocator_(allocator), source_ids_(allocator_), neighbors_(allocator_) {
}

void
BuildCache::Serialize(StreamWriter& writer) const {
    uint64_t source_ids_size = source_ids_.size();
    StreamWriter::WriteObj(writer, source_ids_size);
    Vector<InnerIdType> empty(allocator_);
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        const auto& source_id = source_ids_[i];
        StreamWriter::WriteString(writer, source_id);
        auto it = neighbors_.find(source_id);
        if (it != neighbors_.end()) {
            StreamWriter::WriteVector(writer, it->second);
        } else {
            StreamWriter::WriteVector(writer, empty);
        }
    }
}

void
BuildCache::Deserialize(StreamReader& reader) {
    uint64_t source_ids_size = 0;
    StreamReader::ReadObj(reader, source_ids_size);
    if (source_ids_size > remaining_bytes(reader) / (sizeof(uint64_t) * 2)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache source-id count");
    }
    source_ids_.clear();
    source_ids_.reserve(source_ids_size);
    neighbors_.clear();
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        auto source_id = read_string(reader);
        source_ids_.push_back(source_id);
        Vector<InnerIdType> neighbors(allocator_);
        read_neighbors(reader, neighbors);
        if (!neighbors.empty()) {
            neighbors_.emplace(std::move(source_id), std::move(neighbors));
        }
    }
}

std::vector<std::string>
BuildCache::GetNeighbors(const std::string& source_id) const {
    std::vector<std::string> result;
    auto it = neighbors_.find(source_id);
    if (it == neighbors_.end()) {
        return result;
    }
    const auto& inner_ids = it->second;
    result.reserve(inner_ids.empty() ? 0 : inner_ids.size() - 1);
    for (uint64_t i = 1; i < inner_ids.size(); ++i) {
        const auto& inner_id = inner_ids[i];
        if (static_cast<uint64_t>(inner_id) < source_ids_.size()) {
            result.push_back(source_ids_[inner_id]);
        }
    }
    return result;
}

}  // namespace vsag
