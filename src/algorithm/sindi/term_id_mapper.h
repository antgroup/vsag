
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

#include <cstdint>
#include <optional>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class TermIdMapper {
public:
    TermIdMapper(uint32_t term_id_limit, Allocator* allocator);

    // Map an original term ID to a compact ID.
    // First occurrence gets the next sequential ID.
    // Throws VsagException if the compact ID space is exhausted.
    uint32_t
    Map(uint32_t orig_id);

    // Try to map an original term ID to a compact ID.
    // Returns nullopt if the term has not been seen before (does not allocate).
    [[nodiscard]] std::optional<uint32_t>
    TryMap(uint32_t orig_id) const;

    // Reverse map a compact ID back to the original term ID.
    [[nodiscard]] uint32_t
    ReverseMap(uint32_t compact_id) const;

    [[nodiscard]] uint32_t
    Size() const {
        return next_id_;
    }

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

private:
    UnorderedMap<uint32_t, uint32_t> orig_to_compact_;
    Vector<uint32_t> compact_to_orig_;
    uint32_t next_id_{0};
    uint32_t term_id_limit_{0};
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
