
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

#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "datacell/code_slot_map.h"
#include "hgraph.h"

namespace vsag {

// Private implementation shared only by HGraph translation units. Slot bindings keep their
// own shared ownership when flatten adapters outlive this state during destruction.
class HGraph::StorageState {
public:
    StorageState(bool enabled, Allocator* allocator)
        : code_slot_map_(enabled ? std::make_shared<CodeSlotMap>(allocator) : nullptr),
          deduplicate_storage_(enabled) {
    }

    [[nodiscard]] std::shared_lock<std::shared_mutex>
    AcquireReadLock(std::shared_mutex& global_mutex) const {
        if (not physical_code_resize_pending_.load(std::memory_order_acquire)) {
            return std::shared_lock<std::shared_mutex>(global_mutex);
        }
        std::scoped_lock resize_lock(physical_code_resize_mutex_);
        return std::shared_lock<std::shared_mutex>(global_mutex);
    }

    static void
    InsertCodes(HGraph& index, const void* data, CodeSlotIdType code_slot_id);

    void
    EnsurePhysicalCapacity(HGraph& index, CodeSlotIdType required_capacity);

    void
    EnsurePhysicalCapacityUnlocked(HGraph& index, CodeSlotIdType required_capacity);

    std::shared_ptr<CodeSlotMap> code_slot_map_;
    mutable std::shared_mutex storage_mutation_mutex_;
    mutable std::mutex physical_code_resize_mutex_;
    std::atomic<bool> physical_code_resize_pending_{false};
    std::atomic<bool> pipnn_initial_add_pending_{false};
    std::atomic<CodeSlotIdType> physical_code_capacity_{0};
    const bool deduplicate_storage_;
};

}  // namespace vsag
