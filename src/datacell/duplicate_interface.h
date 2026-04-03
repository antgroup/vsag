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

#include <memory>
#include <vector>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(DuplicateInterface);

class DuplicateInterface {
public:
    virtual ~DuplicateInterface() = default;

    virtual void
    SetDuplicateId(InnerIdType original_id, InnerIdType duplicate_id) = 0;

    virtual auto
    GetDuplicateIds(InnerIdType id) const -> std::vector<InnerIdType> = 0;

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    virtual void
    DeserializeFromLegacyFormat(StreamReader& reader, size_t total_size) = 0;

    virtual void
    Resize(InnerIdType new_size) = 0;
};

using DuplicateTrackerPtr = std::shared_ptr<DuplicateInterface>;

}  // namespace vsag
