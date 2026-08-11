
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

#include "term_id_mapper.h"

#include <fmt/format.h>

#include "common.h"
#include "vsag_exception.h"

namespace vsag {

TermIdMapper::TermIdMapper(uint32_t term_id_limit, Allocator* allocator)
    : orig_to_compact_(allocator),
      compact_to_orig_(allocator),
      term_id_limit_(term_id_limit),
      allocator_(allocator) {
}

uint32_t
TermIdMapper::Map(uint32_t orig_id) {
    auto it = orig_to_compact_.find(orig_id);
    if (it != orig_to_compact_.end()) {
        return it->second;
    }
    if (next_id_ >= term_id_limit_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("term id mapper is full: next_id ({}) >= term_id_limit ({})",
                        next_id_,
                        term_id_limit_));
    }
    uint32_t compact_id = next_id_++;
    orig_to_compact_[orig_id] = compact_id;
    compact_to_orig_.push_back(orig_id);
    return compact_id;
}

std::optional<uint32_t>
TermIdMapper::TryMap(uint32_t orig_id) const {
    auto it = orig_to_compact_.find(orig_id);
    if (it != orig_to_compact_.end()) {
        return it->second;
    }
    return std::nullopt;
}

uint32_t
TermIdMapper::ReverseMap(uint32_t compact_id) const {
    if (compact_id >= next_id_) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("compact_id ({}) >= mapper size ({})", compact_id, next_id_));
    }
    return compact_to_orig_[compact_id];
}

void
TermIdMapper::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, next_id_);
    StreamWriter::WriteObj(writer, term_id_limit_);
    StreamWriter::WriteVector(writer, compact_to_orig_);
}

void
TermIdMapper::Deserialize(StreamReader& reader) {
    uint32_t next_id = 0;
    uint32_t term_id_limit = 0;
    Vector<uint32_t> compact_to_orig(allocator_);
    StreamReader::ReadObj(reader, next_id);
    StreamReader::ReadObj(reader, term_id_limit);
    StreamReader::ReadVector(reader, compact_to_orig);

    CHECK_ARGUMENT(term_id_limit == term_id_limit_, "term id mapper limit does not match index");
    CHECK_ARGUMENT(next_id <= term_id_limit_, "term id mapper size exceeds configured limit");
    CHECK_ARGUMENT(next_id == compact_to_orig.size(),
                   "term id mapper size does not match reverse table");

    // Rebuild the forward map from the deserialized reverse map.
    UnorderedMap<uint32_t, uint32_t> orig_to_compact(allocator_);
    orig_to_compact.reserve(compact_to_orig.size());
    for (uint32_t i = 0; i < compact_to_orig.size(); ++i) {
        const auto inserted = orig_to_compact.emplace(compact_to_orig[i], i).second;
        CHECK_ARGUMENT(inserted, "term id mapper reverse table contains duplicate terms");
    }
    next_id_ = next_id;
    orig_to_compact_ = std::move(orig_to_compact);
    compact_to_orig_ = std::move(compact_to_orig);
}

}  // namespace vsag
