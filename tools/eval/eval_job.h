
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

#include <yaml-cpp/yaml.h>

#include <string>
#include <vector>

#include "./eval_config.h"

namespace vsag::eval {

struct exporter {
    static exporter
    Load(YAML::Node&);

    std::string format{"json"};  // json, or text
    std::string to{"stdout"};    // stdout, or /path/to/file
};

// a eval_job contains multiple eval cases
struct eval_job {
    using eval_case = YAML::Node;
    using name2case = std::pair<std::string, eval_case>;

    std::vector<exporter> exporters;
    std::vector<name2case> cases;
};

}  // namespace vsag::eval
