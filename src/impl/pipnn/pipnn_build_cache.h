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
#include <vector>

#include "json_wrapper.h"
#include "typing.h"

namespace vsag {

// Self-contained PiPNN snapshot. Candidate and adjacency IDs are local indices.
// Original graph IDs are retained only for validating pruning tie ordering.
// Source fingerprints refer to the original FP32 input, independent of index quantization.
struct PiPNNBuildCache {
    std::string policy;
    std::vector<std::string> source_ids;
    std::vector<uint64_t> fingerprints;
    std::vector<uint64_t> offsets;
    std::vector<InnerIdType> targets;
    std::vector<uint16_t> distances;
    std::vector<uint16_t> hashes;
    std::vector<InnerIdType> inner_ids;
    std::vector<uint64_t> graph_offsets;
    std::vector<InnerIdType> graph_targets;
    std::vector<uint16_t> graph_distances;
    // Internal comparison switch; never serialized or exposed as an index parameter.
    bool reuse_final_graph{true};
    uint64_t reused_graph_rows{0};
    uint64_t certificate_rows{0};
    uint64_t reused_rows{0};  // Diagnostics for the last build, not serialized.
    std::vector<uint8_t> reuse_mask;
    std::array<double, 6> phase_seconds{};

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
