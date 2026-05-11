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

#include <string>

#include "common.h"
#include "json_types.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {

class HGraph;

class HGraphSerializer {
public:
    static void
    Serialize(const HGraph& graph, StreamWriter& writer);

    static void
    Deserialize(HGraph& graph, StreamReader& reader);

    static auto
    SerializeBasicInfo(const HGraph& graph) -> JsonType;

    static void
    DeserializeBasicInfo(HGraph& graph, const JsonType& jsonify_basic_info);

    static void
    SerializeLabelInfo(const HGraph& graph, StreamWriter& writer);

    static void
    DeserializeLabelInfo(const HGraph& graph, StreamReader& reader);

    static void
    SerializeBasicInfoV0_14(const HGraph& graph, StreamWriter& writer);

    static void
    DeserializeBasicInfoV0_14(HGraph& graph, StreamReader& reader);

    static auto
    GetMemoryUsageDetail(const HGraph& graph) -> std::string;
};

}  // namespace vsag
