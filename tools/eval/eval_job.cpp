
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

#include "./eval_job.h"

#include "./common.h"

namespace vsag::eval {

exporter
exporter::Load(YAML::Node& node) {
    exporter ret;
    ret.format = check_and_get_value(node, "format");
    ret.to = check_and_get_value(node, "to");
    if (not node["vars"].IsDefined()) {
        return ret;
    }

    // optional fields
    ret.vars = check_and_get_map(node, "vars");

    return ret;
}

}  // namespace vsag::eval
