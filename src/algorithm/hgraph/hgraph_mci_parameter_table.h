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

#include "common.h"
#include "fmt/format.h"
#include "hgraph_parameter.h"
#include "impl/logger/logger.h"
#include "inner_string_params.h"
#include "json_types.h"
#include "vsag/constants.h"

namespace vsag {

// Why this table exists
// ---------------------
// HGraphMCIParameters has thirteen JSON-visible fields. Before this table the same thirteen
// (key, member) pairs had to be repeated by hand in five places: presence detection, FromJson,
// ToJson, CheckCompatibility and the external-key whitelist in hgraph_param_mapping.cpp. Adding a
// parameter therefore meant touching five lists and staying consistent across all of them, which
// is exactly the kind of change an automated reader cannot verify mechanically.
//
// kHGraphMCIFields is now the only place where a key is paired with a member. Every consumer
// iterates it, so a new parameter is one entry here plus an optional bounds check in
// ValidateHGraphMCIParameters(). The generic read/write/compare helpers below are the only code
// that touches the members, which keeps the per-field logic to zero.

// Selects which member pointer of HGraphMCIField is meaningful, and therefore how a value is
// read from JSON, written back to JSON and compared between two parameter objects.
enum class HGraphMCIFieldType : uint8_t {
    BOOLEAN,
    UINT64,
    FLOAT,
    STRING,
};

// One row of kHGraphMCIFields.
// Exactly one of the four member pointers is set; `type` says which one.
struct HGraphMCIField {
    const char* key;   // JSON key, also used in parse error messages
    const char* name;  // C++ member name, used in compatibility logs
    HGraphMCIFieldType type;
    // UINT64 only: message suffix when a negative JSON integer is rejected.
    // Matches the exact text ValidateHGraphMCIParameters() uses for the zero-bound check so
    // that the same human-readable error is produced regardless of whether the rejection
    // happens at read time (negative) or validation time (zero).
    const char* negative_message{nullptr};
    bool HGraphMCIParameters::* bool_member{nullptr};
    uint64_t HGraphMCIParameters::* uint64_member{nullptr};
    float HGraphMCIParameters::* float_member{nullptr};
    std::string HGraphMCIParameters::* string_member{nullptr};
    // Internal build-side keys are accepted from serialized JSON but not from user config, so
    // that a user cannot point a build at a graph file the build host cannot see.
    bool user_facing{true};
    // STRING only: an empty value carries no information and is omitted on write.
    bool skip_when_empty{false};
};

// The single source of truth for the HGraphMCIParameters <-> JSON mapping.
// Order matches HGraphMCIParameters declaration order.
//
// Not constexpr: the keys are the public `const char* const` constants from vsag/constants.h, which
// are constant-initialized but not declared constexpr. They are still initialized before any
// dynamic initializer runs, so reading them here is safe.
// std::to_array deduces the element count so that adding or removing a row cannot leave a stale
// trailing 13 that a compiler would not diagnose.
inline const auto kHGraphMCIFields = std::to_array<HGraphMCIField>({
    {.key = HGRAPH_MCI_CLIQUE_ACTIVATION_ORDER,
     .name = "clique_activation_order",
     .type = HGraphMCIFieldType::BOOLEAN,
     .bool_member = &HGraphMCIParameters::clique_activation_order},
    {.key = HGRAPH_MCI_MCS,
     .name = "mcs",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_mcs must be positive",
     .uint64_member = &HGraphMCIParameters::mcs},
    {.key = HGRAPH_MCI_CLIQUE_MAX,
     .name = "clique_max",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_clique_max must be positive",
     .uint64_member = &HGraphMCIParameters::clique_max},
    {.key = HGRAPH_MCI_ALPHA,
     .name = "alpha",
     .type = HGraphMCIFieldType::FLOAT,
     .float_member = &HGraphMCIParameters::alpha},
    {.key = HGRAPH_MCI_KNNG_SOURCE,
     .name = "knng_source",
     .type = HGraphMCIFieldType::STRING,
     .string_member = &HGraphMCIParameters::knng_source},
    {.key = HGRAPH_MCI_INCREMENTAL_JOIN_RATIO_THRESHOLD_KEY,
     .name = "incremental_join_ratio_threshold",
     .type = HGraphMCIFieldType::FLOAT,
     .float_member = &HGraphMCIParameters::incremental_join_ratio_threshold},
    {.key = HGRAPH_MCI_INCREMENTAL_DEGREE_MIN_KEY,
     .name = "incremental_degree_min",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_incremental_degree_min must be positive",
     .uint64_member = &HGraphMCIParameters::incremental_degree_min},
    {.key = HGRAPH_MCI_INCREMENTAL_DEGREE_N_DIVISOR_KEY,
     .name = "incremental_degree_n_divisor",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_incremental_degree_n_divisor must be positive",
     .uint64_member = &HGraphMCIParameters::incremental_degree_n_divisor},
    {.key = HGRAPH_MCI_INCREMENTAL_DEGREE_MCS_DIVISOR_KEY,
     .name = "incremental_degree_mcs_divisor",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_incremental_degree_mcs_divisor must be positive",
     .uint64_member = &HGraphMCIParameters::incremental_degree_mcs_divisor},
    {.key = HGRAPH_MCI_INCREMENTAL_CLIQUE_MAX_KEY,
     .name = "incremental_clique_max",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_incremental_clique_max must be >= 2",
     .uint64_member = &HGraphMCIParameters::incremental_clique_max},
    {.key = HGRAPH_MCI_DELETE_CLIQUE_SIZE_THRESHOLD_KEY,
     .name = "delete_clique_size_threshold",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_delete_clique_size_threshold must be positive",
     .uint64_member = &HGraphMCIParameters::delete_clique_size_threshold},
    {.key = HGRAPH_MCI_DELETE_NODE_MCT_THRESHOLD_KEY,
     .name = "delete_node_mct_threshold",
     .type = HGraphMCIFieldType::UINT64,
     .negative_message = "hgraph mci_delete_node_mct_threshold must be positive",
     .uint64_member = &HGraphMCIParameters::delete_node_mct_threshold},
    {.key = HGRAPH_MCI_KNNG_PATH_KEY,
     .name = "knng_path",
     .type = HGraphMCIFieldType::STRING,
     .string_member = &HGraphMCIParameters::knng_path,
     .user_facing = false,
     .skip_when_empty = true},
});

// True when any MCI field key is present in `json`, which is how an otherwise plain hgraph config
// opts into MCI without spelling out use_mci.
inline bool
HGraphMCIFieldsPresent(const JsonType& json) {
    for (const auto& field : kHGraphMCIFields) {
        if (json.Contains(field.key)) {
            return true;
        }
    }
    return false;
}

// True when `key` is an MCI parameter a user is allowed to set through index config.
inline bool
IsUserFacingHGraphMCIKey(const std::string& key) {
    for (const auto& field : kHGraphMCIFields) {
        if (field.user_facing and key == field.key) {
            return true;
        }
    }
    return false;
}

// Copies one field out of `json` into `params` when the key is present.
// A UINT64 field is read through the signed accessor first so that a negative JSON value is
// rejected instead of silently wrapping to a huge positive value.
inline void
ReadHGraphMCIField(HGraphMCIParameters& params, const JsonType& json, const HGraphMCIField& field) {
    if (not json.Contains(field.key)) {
        return;
    }
    switch (field.type) {
        case HGraphMCIFieldType::BOOLEAN:
            params.*field.bool_member = json[field.key].GetBool();
            return;
        case HGraphMCIFieldType::UINT64: {
            const auto signed_value = json[field.key].GetInt();
            const char* message = field.negative_message ? field.negative_message
                                                         : "hgraph mci field must be non-negative";
            CHECK_ARGUMENT(signed_value >= 0, fmt::format("{}", message));
            params.*field.uint64_member = static_cast<uint64_t>(signed_value);
            return;
        }
        case HGraphMCIFieldType::FLOAT:
            params.*field.float_member = json[field.key].GetFloat();
            return;
        case HGraphMCIFieldType::STRING:
            params.*field.string_member = json[field.key].GetString();
            return;
    }
}

// Writes one field into `json`.
inline void
WriteHGraphMCIField(const HGraphMCIParameters& params,
                    JsonType& json,
                    const HGraphMCIField& field) {
    switch (field.type) {
        case HGraphMCIFieldType::BOOLEAN:
            json[field.key].SetBool(params.*field.bool_member);
            return;
        case HGraphMCIFieldType::UINT64:
            json[field.key].SetInt(static_cast<int64_t>(params.*field.uint64_member));
            return;
        case HGraphMCIFieldType::FLOAT:
            json[field.key].SetFloat(params.*field.float_member);
            return;
        case HGraphMCIFieldType::STRING:
            if (field.skip_when_empty and (params.*field.string_member).empty()) {
                return;
            }
            json[field.key].SetString(params.*field.string_member);
            return;
    }
}

// True when both parameter objects agree on one field.
inline bool
HGraphMCIFieldEqual(const HGraphMCIParameters& lhs,
                    const HGraphMCIParameters& rhs,
                    const HGraphMCIField& field) {
    switch (field.type) {
        case HGraphMCIFieldType::BOOLEAN:
            return lhs.*field.bool_member == rhs.*field.bool_member;
        case HGraphMCIFieldType::UINT64:
            return lhs.*field.uint64_member == rhs.*field.uint64_member;
        case HGraphMCIFieldType::FLOAT:
            return lhs.*field.float_member == rhs.*field.float_member;
        case HGraphMCIFieldType::STRING:
            return lhs.*field.string_member == rhs.*field.string_member;
    }
    return true;
}

// Rejects parameter combinations that are syntactically well-formed but semantically impossible.
// These ranges are deliberately kept next to the table rather than folded into ReadHGraphMCIField,
// because they describe relationships between fields and not the JSON encoding of one field.
inline void
ValidateHGraphMCIParameters(const HGraphMCIParameters& params) {
    CHECK_ARGUMENT(params.mcs > 0, "hgraph mci_mcs must be positive");
    CHECK_ARGUMENT(params.clique_max > 0, "hgraph mci_clique_max must be positive");
    CHECK_ARGUMENT(params.alpha >= 1.0F, "hgraph mci_alpha must be >= 1.0");
    CHECK_ARGUMENT(params.knng_source == HGRAPH_MCI_KNNG_SOURCE_HGRAPH or
                       params.knng_source == HGRAPH_MCI_KNNG_SOURCE_ODESCENT,
                   fmt::format("hgraph mci_knng_source must be '{}' or '{}'",
                               HGRAPH_MCI_KNNG_SOURCE_HGRAPH,
                               HGRAPH_MCI_KNNG_SOURCE_ODESCENT));
    // Comparison written as a positive range check so that NaN fails it.
    CHECK_ARGUMENT(params.incremental_join_ratio_threshold >= 0.0F and
                       params.incremental_join_ratio_threshold <= 1.0F,  // NOLINT
                   "hgraph mci_incremental_join_ratio_threshold must be in range [0, 1]");
    CHECK_ARGUMENT(params.incremental_degree_min > 0,
                   "hgraph mci_incremental_degree_min must be positive");
    CHECK_ARGUMENT(params.incremental_degree_n_divisor > 0,
                   "hgraph mci_incremental_degree_n_divisor must be positive");
    CHECK_ARGUMENT(params.incremental_degree_mcs_divisor > 0,
                   "hgraph mci_incremental_degree_mcs_divisor must be positive");
    CHECK_ARGUMENT(params.incremental_clique_max >= 2,
                   "hgraph mci_incremental_clique_max must be >= 2");
    CHECK_ARGUMENT(params.delete_clique_size_threshold > 0,
                   "hgraph mci_delete_clique_size_threshold must be positive");
    CHECK_ARGUMENT(params.delete_node_mct_threshold > 0,
                   "hgraph mci_delete_node_mct_threshold must be positive");
}

// Serialized indexes must be reopened with the exact MCI configuration that produced them; a
// partial match would reinterpret an on-disk clique layout. `enabled` is compared here as well,
// because it decides whether the other thirteen fields are meaningful at all.
inline bool
HGraphMCIParametersCompatible(const HGraphMCIParameters& lhs, const HGraphMCIParameters& rhs) {
    if (lhs.enabled != rhs.enabled) {
        logger::error("mci_parameters.enabled mismatch");
        return false;
    }
    for (const auto& field : kHGraphMCIFields) {
        if (not HGraphMCIFieldEqual(lhs, rhs, field)) {
            logger::error(fmt::format("mci_parameters.{} mismatch", field.name));
            return false;
        }
    }
    return true;
}

}  // namespace vsag