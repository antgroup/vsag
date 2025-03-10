
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

namespace vsag::eval {

constexpr static const char* DENSE_VECTORS = "dense_vectors";
constexpr static const char* SPARSE_VECTORS = "sparse_vectors";

template <class T = std::string>
T
check_exist_and_get_value(const YAML::Node& node, const std::string& key) {
    if (not node[key].IsDefined()) {
        throw std::invalid_argument(key + " is not in config file");
    }
    return node[key].as<T>();
};

template <class T = std::string>
T
check_and_get_value(const YAML::Node& node, const std::string& key) {
    if (node[key].IsDefined()) {
        return node[key].as<T>();
    }

    return T();
};

}  // namespace vsag::eval
