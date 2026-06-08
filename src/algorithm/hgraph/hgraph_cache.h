
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

#include "storage/serialization.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class HGraphCache {
public:
    explicit HGraphCache(Allocator* allocator);

    ~HGraphCache() = default;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    std::vector<std::string>
    GetNeighbors(const std::string& source_id) const;

public:
    Allocator* const allocator_;

    // mapping from inner_id to source_id string
    Vector<std::string> source_ids_;

    // mapping from source_id to neighbor inner_ids,
    // neighbors_[source_id][0] is self inner_id,
    // neighbors_[source_id][1...] are neighbor inner_ids
    UnorderedMap<std::string, Vector<InnerIdType>> neighbors_;
};
}  // namespace vsag
