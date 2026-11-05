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

#include "lazy_hgraph_parameter.h"

#include <fmt/format.h>

#include "impl/logger/logger.h"
#include "vsag/constants.h"

namespace vsag {

namespace {
const char* const LAZY_HGRAPH_TRANSITION_THRESHOLD = "transition_threshold";
const char* const LAZY_HGRAPH_GRAPH = "hgraph";
const char* const LAZY_HGRAPH_FLAT = "flat";
}  // namespace

void
LazyHGraphParameter::FromJson(const JsonType& json) {
    InnerIndexParameter::FromJson(json);
    CHECK_ARGUMENT(
        json.Contains(LAZY_HGRAPH_TRANSITION_THRESHOLD),
        fmt::format("lazy_hgraph parameters must contain {}", LAZY_HGRAPH_TRANSITION_THRESHOLD));
    CHECK_ARGUMENT(json.Contains(LAZY_HGRAPH_GRAPH),
                   fmt::format("lazy_hgraph parameters must contain {}", LAZY_HGRAPH_GRAPH));
    CHECK_ARGUMENT(json.Contains(LAZY_HGRAPH_FLAT),
                   fmt::format("lazy_hgraph parameters must contain {}", LAZY_HGRAPH_FLAT));

    const auto threshold_json = json[LAZY_HGRAPH_TRANSITION_THRESHOLD];
    CHECK_ARGUMENT(threshold_json.IsNumberInteger(),
                   "lazy_hgraph transition_threshold must be an integer");
    if (threshold_json.IsNumberUnsigned()) {
        this->transition_threshold = threshold_json.GetUint64();
        CHECK_ARGUMENT(this->transition_threshold > 0,
                       "lazy_hgraph transition_threshold must be positive");
    } else {
        const auto threshold = threshold_json.GetInt();
        CHECK_ARGUMENT(threshold > 0, "lazy_hgraph transition_threshold must be positive");
        this->transition_threshold = static_cast<uint64_t>(threshold);
    }
    this->graph_param = std::make_shared<HGraphParameter>();
    this->graph_param->FromJson(json[LAZY_HGRAPH_GRAPH]);
    this->flat_param = std::make_shared<BruteForceParameter>();
    this->flat_param->FromJson(json[LAZY_HGRAPH_FLAT]);
}

JsonType
LazyHGraphParameter::ToJson() const {
    JsonType json = InnerIndexParameter::ToJson();
    json["type"].SetString(INDEX_LAZY_HGRAPH);
    json[LAZY_HGRAPH_TRANSITION_THRESHOLD].SetUint64(this->transition_threshold);
    json[LAZY_HGRAPH_GRAPH].SetJson(this->graph_param->ToJson());
    json[LAZY_HGRAPH_FLAT].SetJson(this->flat_param->ToJson());
    return json;
}

bool
LazyHGraphParameter::CheckCompatibility(const ParamPtr& other) const {
    if (not InnerIndexParameter::CheckCompatibility(other)) {
        return false;
    }
    auto lazy_param = std::dynamic_pointer_cast<LazyHGraphParameter>(other);
    if (not lazy_param) {
        logger::error(
            "LazyHGraphParameter::CheckCompatibility: "
            "other parameter is not a LazyHGraphParameter");
        return false;
    }
    return this->transition_threshold == lazy_param->transition_threshold and
           this->flat_param->CheckCompatibility(lazy_param->flat_param) and
           this->graph_param->CheckCompatibility(lazy_param->graph_param);
}

}  // namespace vsag
