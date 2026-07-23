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

#include "typing.h"
#include "vsag/constants.h"

namespace vsag {

enum class StreamSerializationTag : uint32_t {
    SECTION_END = 0,
    LABEL_TABLE = 1,
    BASE_CODES = 2,
    ATTRIBUTE_FILTER = 3,
    BOTTOM_GRAPH = 4,
    HIGH_PRECISION_CODES = 5,
    ROUTE_GRAPHS = 6,
    EXTRA_INFO = 7,
    RAW_VECTOR = 8,
    IVF_BUCKET = 9,
    IVF_PARTITION_STRATEGY = 10,
    SINDI_WINDOWS = 11,
    SINDI_RERANK_INDEX = 12,
    SINDI_TERM_ID_MAPPER = 13,
    PYRAMID_HIERARCHIES = 14,
    CODE_SLOT_MAP = 15,
};

inline const char*
StreamSerializationTagName(uint32_t tag) {
    switch (static_cast<StreamSerializationTag>(tag)) {
        case StreamSerializationTag::SECTION_END:
            return "section_end";
        case StreamSerializationTag::LABEL_TABLE:
            return "label_table";
        case StreamSerializationTag::BASE_CODES:
            return "base_codes";
        case StreamSerializationTag::ATTRIBUTE_FILTER:
            return "attribute_filter";
        case StreamSerializationTag::BOTTOM_GRAPH:
            return "bottom_graph";
        case StreamSerializationTag::HIGH_PRECISION_CODES:
            return "high_precision_codes";
        case StreamSerializationTag::ROUTE_GRAPHS:
            return "route_graphs";
        case StreamSerializationTag::EXTRA_INFO:
            return "extra_info";
        case StreamSerializationTag::RAW_VECTOR:
            return "raw_vector";
        case StreamSerializationTag::IVF_BUCKET:
            return "ivf_bucket";
        case StreamSerializationTag::IVF_PARTITION_STRATEGY:
            return "ivf_partition_strategy";
        case StreamSerializationTag::SINDI_WINDOWS:
            return "sindi_windows";
        case StreamSerializationTag::SINDI_RERANK_INDEX:
            return "sindi_rerank_index";
        case StreamSerializationTag::SINDI_TERM_ID_MAPPER:
            return "sindi_term_id_mapper";
        case StreamSerializationTag::PYRAMID_HIERARCHIES:
            return "pyramid_hierarchies";
        case StreamSerializationTag::CODE_SLOT_MAP:
            return "code_slot_map";
    }
    return "unknown";
}

inline bool
StreamSerializationTagCritical(uint32_t tag) {
    switch (static_cast<StreamSerializationTag>(tag)) {
        case StreamSerializationTag::SECTION_END:
            return false;
        case StreamSerializationTag::LABEL_TABLE:
        case StreamSerializationTag::BASE_CODES:
        case StreamSerializationTag::BOTTOM_GRAPH:
        case StreamSerializationTag::ROUTE_GRAPHS:
        case StreamSerializationTag::IVF_BUCKET:
        case StreamSerializationTag::IVF_PARTITION_STRATEGY:
        case StreamSerializationTag::HIGH_PRECISION_CODES:
        case StreamSerializationTag::SINDI_WINDOWS:
        case StreamSerializationTag::SINDI_RERANK_INDEX:
        case StreamSerializationTag::SINDI_TERM_ID_MAPPER:
        case StreamSerializationTag::PYRAMID_HIERARCHIES:
        case StreamSerializationTag::CODE_SLOT_MAP:
            return true;
        case StreamSerializationTag::ATTRIBUTE_FILTER:
        case StreamSerializationTag::EXTRA_INFO:
        case StreamSerializationTag::RAW_VECTOR:
            return false;
    }
    return false;
}

inline constexpr uint32_t kStreamSerializationBlockVersionV1 = 1;

inline uint32_t
StreamSerializationBlockCurrentVersion(uint32_t tag) {
    switch (static_cast<StreamSerializationTag>(tag)) {
        case StreamSerializationTag::SECTION_END:
        case StreamSerializationTag::LABEL_TABLE:
        case StreamSerializationTag::BASE_CODES:
        case StreamSerializationTag::ATTRIBUTE_FILTER:
        case StreamSerializationTag::BOTTOM_GRAPH:
        case StreamSerializationTag::HIGH_PRECISION_CODES:
        case StreamSerializationTag::ROUTE_GRAPHS:
        case StreamSerializationTag::EXTRA_INFO:
        case StreamSerializationTag::RAW_VECTOR:
        case StreamSerializationTag::IVF_BUCKET:
        case StreamSerializationTag::IVF_PARTITION_STRATEGY:
        case StreamSerializationTag::SINDI_WINDOWS:
        case StreamSerializationTag::SINDI_RERANK_INDEX:
        case StreamSerializationTag::SINDI_TERM_ID_MAPPER:
        case StreamSerializationTag::PYRAMID_HIERARCHIES:
        case StreamSerializationTag::CODE_SLOT_MAP:
            return kStreamSerializationBlockVersionV1;
    }
    return kStreamSerializationBlockVersionV1;
}

inline bool
StreamSerializationBlockVersionSupported(uint32_t tag, uint32_t version) {
    return version == StreamSerializationBlockCurrentVersion(tag);
}

inline void
AppendStreamingManifestBlock(
    JsonType& manifest, uint32_t tag, uint32_t version, bool critical, uint64_t payload_size = 0) {
    JsonType block;
    block["tag"].SetInt(tag);
    block["name"].SetString(StreamSerializationTagName(tag));
    block["version"].SetInt(version);
    block["critical"].SetBool(critical);
    block["payload_size"].SetUint64(payload_size);
    manifest.AppendJson(block);
}

}  // namespace vsag
