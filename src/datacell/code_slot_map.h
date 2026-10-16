// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <atomic>
#include <shared_mutex>

#include "basic_types.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/allocator.h"

namespace vsag {

using CodeSlotIdType = InnerIdType;

// Maps logical inner IDs to physical slots in shared code storage. Resolve* translates logical
// IDs to physical slot IDs, while PublishSlot atomically binds a logical ID to a slot.
class CodeSlotMap {
public:
    // Capacity changes must be externally synchronized with Resolve, PublishSlot, and
    // AllocateSlot. Slot bindings themselves are published and read atomically.
    explicit CodeSlotMap(Allocator* allocator);

    ~CodeSlotMap();

    CodeSlotIdType
    AllocateSlot();

    void
    PublishSlot(InnerIdType inner_id, CodeSlotIdType code_slot_id);

    [[nodiscard]] CodeSlotIdType
    Resolve(InnerIdType inner_id) const;

    void
    ResolvePair(InnerIdType inner_id1,
                InnerIdType inner_id2,
                CodeSlotIdType& code_slot_id1,
                CodeSlotIdType& code_slot_id2) const;

    void
    ResolveBatch(const InnerIdType* inner_ids,
                 InnerIdType count,
                 CodeSlotIdType* code_slot_ids) const;

    void
    ReserveLogicalSize(InnerIdType new_size);

    [[nodiscard]] CodeSlotIdType
    PhysicalCount() const;

    [[nodiscard]] InnerIdType
    PublishedLogicalCount() const;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

private:
    void
    EnsureLogicalSize(InnerIdType new_size);

    void
    ReleaseSlots();

    Allocator* allocator_{nullptr};
    std::atomic<CodeSlotIdType>* inner_to_slot_{nullptr};
    InnerIdType logical_capacity_{0};
    std::atomic<CodeSlotIdType> physical_count_{0};
    std::atomic<InnerIdType> published_logical_count_{0};
    mutable std::shared_mutex mutex_;
};

}  // namespace vsag
