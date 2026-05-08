
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

#include "graph_interface.h"

#include "compressed_graph_datacell.h"
#include "graph_datacell.h"
#include "io/io_headers.h"
#include "sparse_graph_datacell.h"

namespace vsag {

void
GraphInterface::UpdateReverseEdges(InnerIdType id,
                                   const Vector<InnerIdType>& old_neighbors,
                                   const Vector<InnerIdType>& new_neighbors) {
    if (reverse_edges_) {
        UnorderedSet<InnerIdType> old_set(allocator_);
        UnorderedSet<InnerIdType> new_set(allocator_);
        for (const auto& n : old_neighbors) {
            old_set.insert(n);
        }
        for (const auto& n : new_neighbors) {
            new_set.insert(n);
        }
        for (const auto& old_n : old_neighbors) {
            if (new_set.find(old_n) == new_set.end()) {
                reverse_edges_->RemoveReverseEdge(id, old_n);
            }
        }
        for (const auto& new_n : new_neighbors) {
            if (old_set.find(new_n) == old_set.end()) {
                reverse_edges_->AddReverseEdge(id, new_n);
            }
        }
    }
}

}  // namespace vsag
