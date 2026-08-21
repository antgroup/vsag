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
#include <string>
#include <vector>

#include "json_types.h"

namespace vsag {

/// Metadata key under which the chunked layout is stored in the footer.
constexpr const char* CHUNKED_LAYOUT_KEY = "chunked_layout";

/// Physical extent of one compressed frame inside the body.
struct ChunkRecord {
    uint64_t offset{0};
    uint64_t compressed_size{0};
};

/// Frame granularity of one component inside the body, stored in JSON as a
/// string tag; unknown tags are rejected at parse time, so readers older
/// than a format extension fail fast instead of misreading the body.
enum class ComponentGranularity : uint8_t {
    Whole,  // single frame holding the entire component
    Byte,   // head (plain) / io data (frame per chunk) / tail (plain); io
            // split at fixed byte boundaries (JSON tag: "chunked")
};

/// Layout of a single serialized component inside the body.
struct ComponentLayout {
    std::string name;
    ComponentGranularity granularity{ComponentGranularity::Whole};

    // Whole granularity only: single frame extent and its logical size
    uint64_t offset{0};
    uint64_t compressed_size{0};
    uint64_t logical_size{0};

    // Byte granularity only
    uint64_t head_offset{0};
    uint64_t head_size{0};
    uint64_t io_size{0};
    std::vector<ChunkRecord> chunks;
    uint64_t tail_offset{0};
    uint64_t tail_size{0};
};

/// In-memory form of the "chunked_layout" footer metadata key: the codec
/// name, the logical chunk granularity and the per-component physical
/// layout, recorded while writing the chunked format.
class ChunkedLayout {
public:
    static constexpr int64_t VERSION = 1;

    ChunkedLayout() = default;

    [[nodiscard]] JsonType
    ToJson() const;

    static ChunkedLayout
    FromJson(const JsonType& json);

    /// FromJson only checks the JSON shape; Validate enforces the semantic
    /// and physical invariants a tampered footer could otherwise violate:
    /// component names are unique, every frame extent fits in [0, body_end)
    /// without overflowing, no two frames overlap, and for an uncompressed
    /// codec each recorded physical size equals its logical size. Throws
    /// VsagException(INVALID_BINARY) on the first violation.
    void
    Validate(uint64_t body_end) const;

    [[nodiscard]] const ComponentLayout*
    FindComponent(const std::string& name) const;

public:
    std::string codec_{"none"};
    uint64_t chunk_size_{0};
    std::vector<ComponentLayout> components_;
};

}  // namespace vsag
