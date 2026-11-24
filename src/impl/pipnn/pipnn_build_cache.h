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

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include "json_wrapper.h"
#include "typing.h"

namespace vsag {

// Self-contained PiPNN snapshot. Candidate and adjacency IDs are local indices.
// Original graph IDs are retained only for validating pruning tie ordering.
// Source fingerprints refer to the original FP32 input, independent of index quantization.
struct PiPNNBuildCache {
    explicit PiPNNBuildCache(Allocator* allocator)
        : source_ids(allocator),
          fingerprints(allocator),
          offsets(allocator),
          targets(allocator),
          distances(allocator),
          hashes(allocator),
          inner_ids(allocator),
          graph_offsets(allocator),
          graph_targets(allocator),
          graph_distances(allocator),
          reuse_mask(allocator) {
    }

    std::string policy;
    Vector<std::string> source_ids;
    Vector<uint64_t> fingerprints;
    Vector<uint64_t> offsets;
    Vector<InnerIdType> targets;
    Vector<uint16_t> distances;
    Vector<uint16_t> hashes;
    Vector<InnerIdType> inner_ids;
    Vector<uint64_t> graph_offsets;
    Vector<InnerIdType> graph_targets;
    Vector<uint16_t> graph_distances;
    // Internal comparison switch; never serialized or exposed as an index parameter.
    bool reuse_final_graph{true};
    uint64_t reused_graph_rows{0};
    uint64_t certificate_rows{0};
    uint64_t reused_rows{0};  // Diagnostics for the last build, not serialized.
    Vector<uint8_t> reuse_mask;
    std::array<double, 6> phase_seconds{};

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        auto buffer_bytes = [](const auto& values) {
            using Value = typename std::decay_t<decltype(values)>::value_type;
            return static_cast<uint64_t>(values.capacity()) * sizeof(Value);
        };
        auto string_bytes = [](const std::string& value) -> uint64_t {
            const auto data = reinterpret_cast<std::uintptr_t>(value.data());
            const auto object = reinterpret_cast<std::uintptr_t>(&value);
            // Small strings already occupy space inside the string object/vector slot.
            return data >= object and data - object < sizeof(value) ? 0 : value.capacity() + 1;
        };
        uint64_t bytes = sizeof(*this) + string_bytes(policy);
        bytes += buffer_bytes(source_ids) + buffer_bytes(fingerprints) + buffer_bytes(offsets);
        bytes += buffer_bytes(targets) + buffer_bytes(distances) + buffer_bytes(hashes);
        bytes += buffer_bytes(inner_ids) + buffer_bytes(graph_offsets);
        bytes += buffer_bytes(graph_targets) + buffer_bytes(graph_distances);
        bytes += buffer_bytes(reuse_mask);
        for (const auto& source : source_ids) {
            bytes += string_bytes(source);
        }
        return bytes;
    }

    [[nodiscard]] JsonType
    GetBuildStats() const {
        constexpr std::array<const char*, 6> phases = {
            "prepare", "cache_match", "partition", "leaf_and_merge", "graph", "capture"};
        JsonType result;
        for (uint64_t i = 0; i < phases.size(); ++i) {
            result["phase_seconds"][phases[i]].SetFloat(static_cast<float>(phase_seconds[i]));
        }
        result["reused_rows"].SetUint64(reused_rows);
        result["reused_graph_rows"].SetUint64(reused_graph_rows);
        result["certificate_rows"].SetUint64(certificate_rows);
        return result;
    }
};

}  // namespace vsag
