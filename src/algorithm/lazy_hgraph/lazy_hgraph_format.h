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

#include <cstdint>

namespace vsag {

// LazyHGraph::Serialize writes these two fields, then the phase byte, before the serialized
// active index. The file therefore ends with the footer of the active index.
constexpr uint64_t LAZY_HGRAPH_MAGIC = 0x4C415A5948475246ULL;  // "LAZYHGRF"
constexpr uint64_t LAZY_HGRAPH_VERSION = 1;

}  // namespace vsag
