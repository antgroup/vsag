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

#include <algorithm>
#include <limits>
#include <type_traits>

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
    if (pipnn != nullptr) {
        StreamWriter::WriteObj(writer, std::numeric_limits<uint64_t>::max());
        StreamWriter::WriteObj(writer, uint64_t{2});
    }
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
    if (pipnn != nullptr) {
        StreamWriter::WriteString(writer, pipnn->policy);
        StreamWriter::WriteObj(writer, static_cast<uint64_t>(pipnn->source_ids.size()));
        for (const auto& id : pipnn->source_ids) {
            StreamWriter::WriteString(writer, id);
        }
        StreamWriter::WriteVector(writer, pipnn->fingerprints);
        StreamWriter::WriteVector(writer, pipnn->offsets);
        StreamWriter::WriteVector(writer, pipnn->targets);
        StreamWriter::WriteVector(writer, pipnn->distances);
        StreamWriter::WriteVector(writer, pipnn->hashes);
        StreamWriter::WriteVector(writer, pipnn->inner_ids);
        StreamWriter::WriteVector(writer, pipnn->graph_offsets);
        StreamWriter::WriteVector(writer, pipnn->graph_targets);
        StreamWriter::WriteVector(writer, pipnn->graph_distances);
    }
}

void
BuildCache::Deserialize(StreamReader& reader) {
    BuildCache result(allocator_);
    uint64_t source_ids_size = 0;
    StreamReader::ReadObj(reader, source_ids_size);
    uint64_t version = 0;
    const bool versioned = source_ids_size == std::numeric_limits<uint64_t>::max();
    if (versioned) {
        StreamReader::ReadObj(reader, version);
        if (version != 1 and version != 2) {
            throw VsagException(ErrorType::INVALID_BINARY, "unsupported PiPNN cache version");
        }
        StreamReader::ReadObj(reader, source_ids_size);
    }
    if (source_ids_size > remaining_bytes(reader) / (sizeof(uint64_t) * 2)) {
        throw VsagException(ErrorType::INVALID_BINARY, "corrupted build cache source-id count");
    }
    result.source_ids_.reserve(source_ids_size);
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        auto source_id = read_string(reader);
        result.source_ids_.push_back(source_id);
        Vector<InnerIdType> neighbors(allocator_);
        read_neighbors(reader, neighbors);
        if (!neighbors.empty()) {
            result.neighbors_.emplace(std::move(source_id), std::move(neighbors));
        }
    }
    if (versioned) {
        auto snapshot = std::make_shared<PiPNNBuildCache>();
        snapshot->policy = read_string(reader);
        uint64_t count = 0;
        StreamReader::ReadObj(reader, count);
        if (count > remaining_bytes(reader) / sizeof(uint64_t)) {
            throw VsagException(ErrorType::INVALID_BINARY, "corrupted PiPNN source count");
        }
        UnorderedSet<std::string> seen(allocator_);
        for (uint64_t i = 0; i < count; ++i) {
            auto id = read_string(reader);
            if (id.empty() or not seen.emplace(id).second) {
                throw VsagException(ErrorType::INVALID_BINARY, "invalid PiPNN source ID");
            }
            snapshot->source_ids.push_back(std::move(id));
        }
        auto read_vector = [&](auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            uint64_t size = 0;
            StreamReader::ReadObj(reader, size);
            if (size > remaining_bytes(reader) / sizeof(Value)) {
                throw VsagException(ErrorType::INVALID_BINARY, "corrupted PiPNN vector length");
            }
            values.resize(size);
            reader.Read(reinterpret_cast<char*>(values.data()), size * sizeof(Value));
        };
        read_vector(snapshot->fingerprints);
        read_vector(snapshot->offsets);
        read_vector(snapshot->targets);
        read_vector(snapshot->distances);
        if (snapshot->fingerprints.size() != count or snapshot->offsets.size() != count + 1 or
            snapshot->offsets.front() != 0 or
            snapshot->offsets.back() != snapshot->targets.size() or
            snapshot->targets.size() != snapshot->distances.size() or
            not std::is_sorted(snapshot->offsets.begin(), snapshot->offsets.end())) {
            throw VsagException(ErrorType::INVALID_BINARY, "invalid PiPNN cache offsets");
        }
        for (const auto target : snapshot->targets) {
            if (target >= count) {
                throw VsagException(ErrorType::INVALID_BINARY, "invalid PiPNN cache target");
            }
        }
        if (version == 2) {
            read_vector(snapshot->hashes);
            read_vector(snapshot->inner_ids);
            read_vector(snapshot->graph_offsets);
            read_vector(snapshot->graph_targets);
            read_vector(snapshot->graph_distances);
            if (snapshot->hashes.size() != snapshot->targets.size() or
                snapshot->inner_ids.size() != count or
                snapshot->graph_offsets.size() != count + 1 or
                snapshot->graph_offsets.front() != 0 or
                snapshot->graph_offsets.back() != snapshot->graph_targets.size() or
                snapshot->graph_distances.size() != snapshot->graph_targets.size() or
                not std::is_sorted(snapshot->graph_offsets.begin(),
                                   snapshot->graph_offsets.end())) {
                throw VsagException(ErrorType::INVALID_BINARY, "invalid PiPNN graph cache offsets");
            }
            for (const auto target : snapshot->graph_targets) {
                if (target >= count) {
                    throw VsagException(ErrorType::INVALID_BINARY,
                                        "invalid PiPNN graph cache target");
                }
            }
            for (uint64_t row = 0; row < count; ++row) {
                const auto begin =
                    snapshot->targets.begin() + static_cast<int64_t>(snapshot->offsets[row]);
                const auto end =
                    snapshot->targets.begin() + static_cast<int64_t>(snapshot->offsets[row + 1]);
                if (std::adjacent_find(begin, end, std::greater_equal<InnerIdType>{}) != end) {
                    throw VsagException(ErrorType::INVALID_BINARY, "unsorted PiPNN candidate row");
                }
            }
        }
        result.pipnn = std::move(snapshot);
    }
    source_ids_.swap(result.source_ids_);
    neighbors_.swap(result.neighbors_);
    pipnn = std::move(result.pipnn);
}

const Vector<InnerIdType>*
BuildCache::FindNeighborInnerIds(const std::string& source_id) const {
    auto it = neighbors_.find(source_id);
    if (it == neighbors_.end()) {
        return nullptr;
    }
    return &it->second;
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
