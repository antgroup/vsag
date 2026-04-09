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

#include <mutex>
#include <shared_mutex>

#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class ReverseEdge {
public:
    explicit ReverseEdge(Allocator* allocator) : allocator_(allocator), reverse_edges_(allocator) {
    }

    void
    AddReverseEdge(InnerIdType from, InnerIdType to);

    void
    RemoveReverseEdge(InnerIdType from, InnerIdType to);

    void
    GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const;

    void
    ClearIncomingNeighbors(InnerIdType id);

    void
    Clear();

    void
    Resize(InnerIdType new_size);

    int64_t
    GetMemoryUsage() const;

private:
    Allocator* const allocator_;
    UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> reverse_edges_;
    mutable std::shared_mutex mutex_{};
};

}  // namespace vsag
