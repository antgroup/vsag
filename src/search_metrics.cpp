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

#include "vsag/search_metrics.h"

#include <algorithm>
#include <type_traits>
#include <utility>

#include "search_metrics_names.h"
#include "typing.h"

namespace vsag {

MetricsSchema::MetricsSchema(std::vector<MetricsField> fields) : fields_(std::move(fields)) {
}

bool
MetricsSchema::Contains(std::string_view key) const {
    // The baseline has 19 fields; a linear scan avoids a second owning lookup container.
    return std::any_of(fields_.begin(), fields_.end(), [&key](const auto& field) {
        return std::string_view(field.name) == key;
    });
}

const std::vector<MetricsField>&
MetricsSchema::Fields() const {
    return fields_;
}

const MetricsSchema&
MetricsSchema::Base() {
    static const MetricsSchema schema({
        {"is_timeout", "Whether search timed out", MetricValueType::BOOL},
        {"dist_cmp", "Legacy distance comparison count", MetricValueType::UINT32},
        {"hops", "Legacy graph hop count", MetricValueType::UINT32},
        {"io_cnt", "IO operation count", MetricValueType::UINT32},
        {"io_time_ms", "IO time in milliseconds", MetricValueType::UINT32},
        {"reorder_distance_count", "Reorder distance count", MetricValueType::UINT32},
        {"reorder_candidate_count", "Reorder candidate count", MetricValueType::UINT32},
        {"reorder_lower_bound_probe_count",
         "Reorder lower-bound probe count",
         MetricValueType::UINT32},
        {"rabitq_filter_count", "RaBitQ filter count", MetricValueType::UINT32},
        {"rabitq_full_count", "RaBitQ full distance count", MetricValueType::UINT32},
        {"rabitq_filter_fallback_full_count",
         "RaBitQ filter fallback full count",
         MetricValueType::UINT32},
        {"rabitq_reorder_hint_full_count",
         "RaBitQ reorder hint full count",
         MetricValueType::UINT32},
        {"rabitq_reorder_fallback_full_count",
         "RaBitQ reorder fallback full count",
         MetricValueType::UINT32},
        {"query_computer_count", "Query computer count", MetricValueType::UINT32},
        {"parallel_search_fallback_count",
         "Parallel search fallback count",
         MetricValueType::UINT32},
        {"distance_evaluations", "Total distance evaluations", MetricValueType::UINT64},
        {"distance_evaluations_by_phase",
         "Distance evaluations by search phase",
         MetricValueType::OBJECT},
        {"distance_evaluations_by_backend",
         "Distance evaluations by backend",
         MetricValueType::OBJECT},
        {"complete", "Whether accounting is complete", MetricValueType::BOOL},
    });
    return schema;
}

std::string
SearchResultMetrics::Dump() const {
    JsonType json;
    // Serialize UINT32 schema fields through the unsigned 64-bit JSON setter so values through
    // UINT32_MAX cannot be interpreted as signed while the typed struct and schema remain UINT32.
    json["is_timeout"].SetBool(is_timeout);
    json["dist_cmp"].SetUint64(dist_cmp);
    json["hops"].SetUint64(hops);
    json["io_cnt"].SetUint64(io_cnt);
    json["io_time_ms"].SetUint64(io_time_ms);
    json["reorder_distance_count"].SetUint64(reorder_distance_count);
    json["reorder_candidate_count"].SetUint64(reorder_candidate_count);
    json["reorder_lower_bound_probe_count"].SetUint64(reorder_lower_bound_probe_count);
    json["rabitq_filter_count"].SetUint64(rabitq_filter_count);
    json["rabitq_full_count"].SetUint64(rabitq_full_count);
    json["rabitq_filter_fallback_full_count"].SetUint64(rabitq_filter_fallback_full_count);
    json["rabitq_reorder_hint_full_count"].SetUint64(rabitq_reorder_hint_full_count);
    json["rabitq_reorder_fallback_full_count"].SetUint64(rabitq_reorder_fallback_full_count);
    json["query_computer_count"].SetUint64(query_computer_count);
    json["parallel_search_fallback_count"].SetUint64(parallel_search_fallback_count);
    json["distance_evaluations"].SetUint64(distance_evaluations);
    for (uint64_t i = 0; i < distance_evaluations_by_phase.size(); ++i) {
        json["distance_evaluations_by_phase"]
            [DistanceEvaluationPhaseName(static_cast<DistanceEvaluationPhase>(i))]
                .SetUint64(distance_evaluations_by_phase[i]);
    }
    for (uint64_t i = 0; i < distance_evaluations_by_backend.size(); ++i) {
        json["distance_evaluations_by_backend"]
            [DistanceEvaluationBackendName(static_cast<DistanceEvaluationBackend>(i))]
                .SetUint64(distance_evaluations_by_backend[i]);
    }
    json["complete"].SetBool(complete);
    for (const auto& extension : extensions_) {
        const auto& key = extension.first;
        const auto& value = extension.second;
        std::visit(
            [&json, &key](const auto& typed_value) {
                using ValueType = std::decay_t<decltype(typed_value)>;
                if constexpr (std::is_same_v<ValueType, uint32_t>) {
                    json[key].SetUint64(static_cast<uint64_t>(typed_value));
                } else if constexpr (std::is_same_v<ValueType, uint64_t>) {
                    json[key].SetUint64(typed_value);
                } else if constexpr (std::is_same_v<ValueType, bool>) {
                    json[key].SetBool(typed_value);
                } else if constexpr (std::is_same_v<ValueType, double>) {
                    json[key].SetDouble(typed_value);
                } else if constexpr (std::is_same_v<ValueType, std::string>) {
                    json[key].SetString(typed_value);
                } else {
                    static_assert(std::is_same_v<ValueType, void>, "unhandled MetricValue type");
                }
            },
            value);
    }
    return json.Dump();
}

}  // namespace vsag
