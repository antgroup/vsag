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

#include "eval_config.h"

#include <limits>
#include <stdexcept>

namespace vsag::eval {

namespace {

[[noreturn]] void
invalid_field(const std::string& key, const char* expected) {
    throw std::invalid_argument("eval config field '" + key + "' must be " + expected);
}

void
read_string(const nlohmann::json& json, const std::string& key, std::string& value) {
    if (not json.contains(key)) {
        return;
    }
    const auto& field = json.at(key);
    if (field.is_string()) {
        value = field.get<std::string>();
    } else if (field.is_object()) {
        value = field.dump();
    } else {
        invalid_field(key, "a string or a JSON object");
    }
}

void
read_bool(const nlohmann::json& json, const std::string& key, bool& value) {
    if (not json.contains(key)) {
        return;
    }
    const auto& field = json.at(key);
    if (not field.is_boolean()) {
        invalid_field(key, "a boolean");
    }
    value = field.get<bool>();
}

void
read_int32(const nlohmann::json& json, const std::string& key, int32_t& value) {
    if (not json.contains(key)) {
        return;
    }
    const auto& field = json.at(key);
    if (not field.is_number_integer() ||
        field.get<int64_t>() < std::numeric_limits<int32_t>::min() ||
        field.get<int64_t>() > std::numeric_limits<int32_t>::max()) {
        invalid_field(key, "a 32-bit integer");
    }
    value = static_cast<int32_t>(field.get<int64_t>());
}

void
read_float(const nlohmann::json& json, const std::string& key, float& value) {
    if (not json.contains(key)) {
        return;
    }
    const auto& field = json.at(key);
    if (not field.is_number()) {
        invalid_field(key, "a number");
    }
    value = field.get<float>();
}

void
read_uint64(const nlohmann::json& json, const std::string& key, uint64_t& value) {
    if (not json.contains(key)) {
        return;
    }
    const auto& field = json.at(key);
    if (not field.is_number_unsigned() &&
        not(field.is_number_integer() && field.get<int64_t>() >= 0)) {
        invalid_field(key, "a non-negative integer");
    }
    value = field.get<uint64_t>();
}

}  // namespace

EvalConfig
EvalConfig::FromJson(const nlohmann::json& json) {
    if (not json.is_object()) {
        throw std::invalid_argument("eval config must be a JSON object");
    }

    static const char* const known_keys[] = {"dataset_path",
                                             "action_type",
                                             "index_name",
                                             "build_param",
                                             "index_path",
                                             "search_param",
                                             "search_mode",
                                             "top_k",
                                             "radius",
                                             "search_query_count",
                                             "delete_index_after_search",
                                             "num_threads_building",
                                             "num_threads_searching",
                                             "enable_recall",
                                             "enable_percent_recall",
                                             "enable_qps",
                                             "enable_tps",
                                             "enable_memory",
                                             "enable_latency",
                                             "enable_percent_latency",
                                             "use_id_based_recall"};
    for (const auto& item : json.items()) {
        bool known = false;
        for (const auto* key : known_keys) {
            known = known || item.key() == key;
        }
        if (not known) {
            throw std::invalid_argument("unknown eval config field '" + item.key() + "'");
        }
    }

    EvalConfig config;
    read_string(json, "dataset_path", config.dataset_path);
    read_string(json, "action_type", config.action_type);
    read_string(json, "index_name", config.index_name);
    read_string(json, "build_param", config.build_param);
    read_string(json, "index_path", config.index_path);
    read_string(json, "search_param", config.search_param);
    read_string(json, "search_mode", config.search_mode);
    read_int32(json, "top_k", config.top_k);
    read_float(json, "radius", config.radius);
    read_uint64(json, "search_query_count", config.search_query_count);
    read_bool(json, "delete_index_after_search", config.delete_index_after_search);
    read_int32(json, "num_threads_building", config.num_threads_building);
    read_int32(json, "num_threads_searching", config.num_threads_searching);
    read_bool(json, "enable_recall", config.enable_recall);
    read_bool(json, "enable_percent_recall", config.enable_percent_recall);
    read_bool(json, "enable_qps", config.enable_qps);
    read_bool(json, "enable_tps", config.enable_tps);
    read_bool(json, "enable_memory", config.enable_memory);
    read_bool(json, "enable_latency", config.enable_latency);
    read_bool(json, "enable_percent_latency", config.enable_percent_latency);
    read_bool(json, "use_id_based_recall", config.use_id_based_recall);
    return config;
}

}  // namespace vsag::eval
