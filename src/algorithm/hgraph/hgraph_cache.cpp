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

#include "hgraph_cache.h"

#include "impl/allocator/default_allocator.h"

namespace vsag {

HGraphCache::HGraphCache(Allocator* allocator)
    : allocator_(allocator), source_ids_(allocator_), neighbors_(allocator_) {
}

void
HGraphCache::Serialize(StreamWriter& writer) const {
    uint64_t source_ids_size = source_ids_.size();
    StreamWriter::WriteObj(writer, source_ids_size);
    Vector<InnerIdType> empty(allocator_);
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        const auto& source_id = source_ids_[i];
        StreamWriter::WriteString(writer, source_id);
        // Write neighbors for this source_id if exists, otherwise write empty vector
        auto it = neighbors_.find(source_id);
        if (it != neighbors_.end()) {
            StreamWriter::WriteVector(writer, it->second);
        } else {
            StreamWriter::WriteVector(writer, empty);
        }
    }
}

void
HGraphCache::Deserialize(StreamReader& reader) {
    uint64_t source_ids_size = 0;
    StreamReader::ReadObj(reader, source_ids_size);
    source_ids_.clear();
    source_ids_.reserve(source_ids_size);
    neighbors_.clear();
    for (uint64_t i = 0; i < source_ids_size; ++i) {
        std::string source_id = StreamReader::ReadString(reader);
        source_ids_.push_back(source_id);
        Vector<InnerIdType> neighbors(allocator_);
        StreamReader::ReadVector(reader, neighbors);
        if (!neighbors.empty()) {
            neighbors_.emplace(std::move(source_id), std::move(neighbors));
        }
    }
}

std::vector<std::string>
HGraphCache::GetNeighbors(const std::string& source_id) const {
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
