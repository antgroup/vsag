
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

#include <optional>
#include <string>
#include <vector>

namespace vsag::eval {

struct exporter {
    static exporter
    Load(YAML::Node&);

    std::string format{"json"};  // json, text/table, line_protocol
    std::string to{"stdout"};    // stdout, file://path/to/file or influxdb://endpoint
    std::unordered_map<std::string, std::string> vars;  // environment variables, like cookies
};

// a eval_job contains multiple eval cases
struct eval_job {
    using eval_case = YAML::Node;
    using name2case = std::pair<std::string, eval_case>;

    std::vector<name2case> cases;

    // global options
    std::vector<exporter> exporters;
    std::optional<int32_t> num_threads_building;
    std::optional<int32_t> num_threads_searching;
};

}  // namespace vsag::eval
