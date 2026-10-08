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
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace vsag {

/** Search phase in which a distance evaluation was recorded.
 * APPROXIMATE denotes the main candidate-search phase, including exact BruteForce scans;
 * it does not describe the numerical accuracy of a distance evaluation.
 */
enum class DistanceEvaluationPhase : uint8_t { ROUTING = 0, APPROXIMATE, RERANK, COUNT };

/** Distance-computation backend used for a recorded evaluation. */
enum class DistanceEvaluationBackend : uint8_t {
    FP32 = 0,
    FP16,
    BF16,
    INT8,
    SQ8,
    SQ4,
    SQ8_UNIFORM,
    SQ4_UNIFORM,
    PQ,
    PQ_FASTSCAN,
    RABITQ,
    BINARY,
    SPARSE_FP32,
    SPARSE_FP16,
    SPARSE_SQ8,
    UNKNOWN,
    COUNT,
};

/** Public value categories used by the search-statistics schema. */
enum class MetricValueType : uint8_t { UINT32, UINT64, BOOL, DOUBLE, OBJECT, STRING };

/** Supported value types for index-specific statistics extensions. */
using MetricValue = std::variant<uint32_t, uint64_t, bool, double, std::string>;

/** Description of one stable statistics field. */
struct MetricsField {
    std::string name;
    std::string description;
    MetricValueType type;
};

/** Read-only description of the stable search-statistics field set. */
class MetricsSchema {
public:
    [[nodiscard]] bool
    Contains(std::string_view key) const;

    [[nodiscard]] const std::vector<MetricsField>&
    Fields() const;

    static const MetricsSchema&
    Base();

private:
    explicit MetricsSchema(std::vector<MetricsField> fields);

    std::vector<MetricsField> fields_;
};

/**
 * Detached value snapshot returned with a completed search.
 *
 * The result owns all values and may be copied or modified by the caller. It does not reference live
 * query counters. Index-specific fields are
 * stored in extensions; names from MetricsSchema::Base() are reserved and cannot override baseline
 * JSON fields.
 */
class SearchMetrics;

struct SearchResultMetrics {
    SearchResultMetrics() = default;
    SearchResultMetrics(const SearchResultMetrics&) = default;
    SearchResultMetrics(SearchResultMetrics&&) noexcept = default;
    SearchResultMetrics&
    operator=(const SearchResultMetrics&) = default;
    SearchResultMetrics&
    operator=(SearchResultMetrics&&) noexcept = default;

    bool is_timeout{false};
    uint32_t dist_cmp{0};
    uint32_t hops{0};
    uint32_t io_cnt{0};
    uint32_t io_time_ms{0};
    uint32_t reorder_distance_count{0};
    uint32_t reorder_candidate_count{0};
    uint32_t reorder_lower_bound_probe_count{0};
    uint32_t rabitq_filter_count{0};
    uint32_t rabitq_full_count{0};
    uint32_t rabitq_filter_fallback_full_count{0};
    uint32_t rabitq_reorder_hint_full_count{0};
    uint32_t rabitq_reorder_fallback_full_count{0};
    uint32_t query_computer_count{0};
    uint32_t parallel_search_fallback_count{0};
    uint64_t distance_evaluations{0};
    std::array<uint64_t, static_cast<uint8_t>(DistanceEvaluationPhase::COUNT)>
        distance_evaluations_by_phase{};
    std::array<uint64_t, static_cast<uint8_t>(DistanceEvaluationBackend::COUNT)>
        distance_evaluations_by_backend{};
    // Diagnostic distance-accounting status, not search success, recall, or timeout status.
    // Unknown backends and saturated 64-bit distance increments mark collection incomplete.
    // Legacy uint32_t counters retain their existing update/overflow semantics.
    bool complete{true};

    [[nodiscard]] const std::map<std::string, MetricValue>&
    Extensions() const {
        return extensions_;
    }

    [[nodiscard]] uint64_t
    DistanceEvaluations(DistanceEvaluationPhase phase) const {
        const auto index = static_cast<uint8_t>(phase);
        return index < distance_evaluations_by_phase.size() ? distance_evaluations_by_phase[index]
                                                            : 0;
    }

    [[nodiscard]] uint64_t
    DistanceEvaluations(DistanceEvaluationBackend backend) const {
        const auto index = static_cast<uint8_t>(backend);
        return index < distance_evaluations_by_backend.size()
                   ? distance_evaluations_by_backend[index]
                   : 0;
    }

    [[nodiscard]] std::string
    Dump() const;

private:
    friend class SearchMetrics;
    std::map<std::string, MetricValue> extensions_;
};

}  // namespace vsag
