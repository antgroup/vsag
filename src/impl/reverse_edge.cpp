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

#include "reverse_edge.h"

namespace vsag {

void
ReverseEdge::AddReverseEdge(InnerIdType from, InnerIdType to) {
    std::unique_lock<std::shared_mutex> wlock(mutex_);
    auto [it, inserted] =
        reverse_edges_.emplace(to, std::make_unique<Vector<InnerIdType>>(allocator_));
    const auto& incoming = it->second;
    for (const auto& id : *incoming) {
        if (id == from) {
            return;
        }
    }
    it->second->push_back(from);
}

void
ReverseEdge::RemoveReverseEdge(InnerIdType from, InnerIdType to) {
    std::unique_lock<std::shared_mutex> wlock(mutex_);
    auto it = reverse_edges_.find(to);
    if (it == reverse_edges_.end()) {
        return;
    }
    const auto& incoming = it->second;
    for (auto iter = incoming->begin(); iter != incoming->end(); ++iter) {
        if (*iter == from) {
            it->second->erase(iter);
            return;
        }
    }
}

void
ReverseEdge::GetIncomingNeighbors(InnerIdType id, Vector<InnerIdType>& neighbors) const {
    std::shared_lock<std::shared_mutex> rlock(mutex_);
    auto it = reverse_edges_.find(id);
    if (it == reverse_edges_.end()) {
        neighbors.clear();
        return;
    }
    neighbors.assign(it->second->begin(), it->second->end());
}

void
ReverseEdge::ClearIncomingNeighbors(InnerIdType id) {
    std::unique_lock<std::shared_mutex> wlock(mutex_);
    auto it = reverse_edges_.find(id);
    if (it != reverse_edges_.end()) {
        it->second->clear();
    }
}

void
ReverseEdge::Clear() {
    std::unique_lock<std::shared_mutex> wlock(mutex_);
    reverse_edges_.clear();
}

int64_t
ReverseEdge::GetMemoryUsage() const {
    std::shared_lock<std::shared_mutex> rlock(mutex_);
    int64_t usage = sizeof(ReverseEdge);
    for (const auto& [id, neighbors] : reverse_edges_) {
        usage += sizeof(id);
        usage += sizeof(std::unique_ptr<Vector<InnerIdType>>);
        if (neighbors) {
            usage += static_cast<int64_t>(neighbors->capacity() * sizeof(InnerIdType));
        }
    }
    return usage;
}

}  // namespace vsag
