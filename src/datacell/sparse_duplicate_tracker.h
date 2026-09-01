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

#include <shared_mutex>

#include "duplicate_interface.h"
#include "impl/allocator/allocator_wrapper.h"
#include "typing.h"

namespace vsag {

class SparseDuplicateTracker : public DuplicateInterface {
public:
    explicit SparseDuplicateTracker(Allocator* allocator);

    void
    SetDuplicateId(InnerIdType group_id, InnerIdType duplicate_id) override;

    auto
    GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> override;

    [[nodiscard]] auto
    GetGroupId(InnerIdType id) const -> InnerIdType override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

    void
    DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) override;

    void
    Resize(InnerIdType new_size) override {
        (void)new_size;
    }

private:
    Allocator* allocator_;
    UnorderedMap<InnerIdType, InnerIdType> next_ids_;
    mutable std::shared_mutex mutex_;
    size_t duplicate_count_{0};
    bool has_deserialized_{false};
};

}  // namespace vsag
