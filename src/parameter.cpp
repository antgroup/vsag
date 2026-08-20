// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "parameter.h"

#include <algorithm>
#include <functional>
#include <nlohmann/json.hpp>
#include <utility>

namespace vsag {

namespace {

std::string
escape_json_pointer_token(const std::string& token) {
    std::string escaped;
    escaped.reserve(token.size());
    for (const auto ch : token) {
        if (ch == '~') {
            escaped.append("~0");
        } else if (ch == '/') {
            escaped.append("~1");
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

std::string
append_json_pointer_token(const std::string& path, const std::string& token) {
    return path + "/" + escape_json_pointer_token(token);
}

}  // namespace

CompatibilityReport
Parameter::CollectCompatibilityIssues(const ParamPtr& other) const {
    CompatibilityReport report;
    if (other == nullptr) {
        report.issues.push_back({"", "other parameter is null"});
        return report;
    }

    const auto left = this->ToJson();
    const auto right = other->ToJson();
    const auto* left_json = left.GetInnerJson();
    const auto* right_json = right.GetInnerJson();
    using CompatibilityCollector =
        std::function<void(const nlohmann::json&, const nlohmann::json&, const std::string&)>;
    CompatibilityCollector collect =
        [&](const nlohmann::json& lhs, const nlohmann::json& rhs, const std::string& path) {
            if (lhs.type() != rhs.type()) {
                report.issues.push_back({path, "JSON value types differ"});
                return;
            }
            if (lhs.is_object()) {
                for (const auto& [key, value] : lhs.items()) {
                    auto child_path = append_json_pointer_token(path, key);
                    if (not rhs.contains(key)) {
                        report.issues.push_back({child_path, "field is missing"});
                    } else {
                        collect(value, rhs.at(key), child_path);
                    }
                }
                for (const auto& [key, value] : rhs.items()) {
                    (void)value;
                    if (not lhs.contains(key)) {
                        auto child_path = append_json_pointer_token(path, key);
                        report.issues.push_back({std::move(child_path), "unexpected field"});
                    }
                }
                return;
            }
            if (lhs.is_array()) {
                if (lhs.size() != rhs.size()) {
                    report.issues.push_back({path, "array sizes differ"});
                }
                const auto common_size = std::min(lhs.size(), rhs.size());
                for (uint64_t i = 0; i < common_size; ++i) {
                    collect(lhs[i], rhs[i], append_json_pointer_token(path, std::to_string(i)));
                }
                return;
            }
            if (lhs != rhs) {
                report.issues.push_back({path, "values differ"});
            }
        };
    collect(*left_json, *right_json, "");
    return report;
}

}  // namespace vsag
