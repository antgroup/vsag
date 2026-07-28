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
#include <memory>

#include "code_slot_map.h"
#include "flatten_interface.h"

namespace vsag {

FlattenInterfacePtr
MakeCodeSlotFlattenAdapter(FlattenInterfacePtr base,
                           std::shared_ptr<const CodeSlotMap> mapping,
                           Allocator* allocator,
                           const std::atomic<uint64_t>* logical_total_count);

// Physical storage operations must bypass the logical-id adapter.
FlattenInterfacePtr
GetCodeSlotPhysicalFlatten(const FlattenInterfacePtr& flatten);

void
InsertVectorToCodeSlot(const FlattenInterfacePtr& flatten,
                       const void* vector,
                       CodeSlotIdType code_slot_id);

}  // namespace vsag
