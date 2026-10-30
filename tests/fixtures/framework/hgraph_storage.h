
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

#include <algorithm>
#include <sstream>

#include "datacell/code_slot_map.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "unittest.h"
#include "vsag/index.h"

namespace fixtures {

inline void
require_code_storage_counts(const vsag::IndexPtr& index,
                            vsag::InnerIdType logical,
                            vsag::InnerIdType physical) {
    REQUIRE(index->GetNumElements() == logical);
    std::stringstream stream;
    REQUIRE(index->SerializeStreaming(stream).has_value());
    stream.seekg(0);
    const auto metadata = vsag::Index::GetStreamingMetadata(stream);
    REQUIRE(metadata.has_value());
    const auto& blocks = metadata.value().blocks;
    const auto tag = static_cast<uint32_t>(vsag::StreamSerializationTag::CODE_SLOT_MAP);
    const auto block = std::find_if(
        blocks.begin(), blocks.end(), [tag](const auto& item) { return item.tag == tag; });
    REQUIRE(block != blocks.end());
    stream.clear();
    stream.seekg(0);
    vsag::IOStreamReader reader(stream);
    reader.Seek(block->payload_offset);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    vsag::CodeSlotMap mapping(allocator.get());
    mapping.Deserialize(reader);
    REQUIRE(mapping.PublishedLogicalCount() == logical);
    REQUIRE(mapping.PhysicalCount() == physical);
}

}  // namespace fixtures
