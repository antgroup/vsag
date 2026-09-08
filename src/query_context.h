
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <string>

#include "metric_type.h"
#include "search_metrics_names.h"
#include "typing.h"
#include "vsag/allocator.h"
#include "vsag/search_metrics.h"

namespace vsag {

class SearchStatistics;
class ReasoningContext;
class QueryComputerPool;
template <typename QuantTmpl, MetricType metric>
class TransformQuantizer;

struct QueryContext {
    Allocator* alloc = nullptr;
    SearchStatistics* stats = nullptr;
    ReasoningContext* reasoning_ctx = nullptr;
    float rabitq_error_rate = std::numeric_limits<float>::quiet_NaN();
    DistanceEvaluationPhase distance_phase = DistanceEvaluationPhase::APPROXIMATE;
    bool track_distance_evaluations = true;
    bool enable_rabitq_reorder = true;
    QueryComputerPool* computer_pool = nullptr;
};

class ScopedDistancePhase {
public:
    ScopedDistancePhase(QueryContext& context, DistanceEvaluationPhase phase)
        : context_(context), previous_phase_(context.distance_phase) {
        context_.distance_phase = phase;
    }

    ~ScopedDistancePhase() {
        context_.distance_phase = previous_phase_;
    }

    ScopedDistancePhase(const ScopedDistancePhase&) = delete;
    ScopedDistancePhase&
    operator=(const ScopedDistancePhase&) = delete;

private:
    QueryContext& context_;
    DistanceEvaluationPhase previous_phase_;
};

class SearchStatistics {
public:
    using DistancePhase = DistanceEvaluationPhase;

    static const char*
    PhaseName(DistancePhase phase) {
        return DistanceEvaluationPhaseName(phase);
    }

    static DistanceEvaluationBackend
    BackendFromName(const std::string& name) {
        // Quantizer names may be decorated (for example, QUANTIZATION_ADAPTER_sq8_uniform), so
        // exact matching is insufficient. Keep overlapping families in most-specific-first order.
        if (name.find("sparse") != std::string::npos) {
            if (name.find("sq8") != std::string::npos)
                return DistanceEvaluationBackend::SPARSE_SQ8;
            if (name.find("fp16") != std::string::npos)
                return DistanceEvaluationBackend::SPARSE_FP16;
            return DistanceEvaluationBackend::SPARSE_FP32;
        }
        if (name.find("rabitq") != std::string::npos)
            return DistanceEvaluationBackend::RABITQ;
        if (name.find("pq_fastscan") != std::string::npos || name.find("pqfs") != std::string::npos)
            return DistanceEvaluationBackend::PQ_FASTSCAN;
        if (name.find("pq") != std::string::npos)
            return DistanceEvaluationBackend::PQ;
        if (name.find("sq8_uniform") != std::string::npos)
            return DistanceEvaluationBackend::SQ8_UNIFORM;
        if (name.find("sq4_uniform") != std::string::npos)
            return DistanceEvaluationBackend::SQ4_UNIFORM;
        if (name.find("sq8") != std::string::npos)
            return DistanceEvaluationBackend::SQ8;
        if (name.find("sq4") != std::string::npos)
            return DistanceEvaluationBackend::SQ4;
        if (name.find("bf16") != std::string::npos)
            return DistanceEvaluationBackend::BF16;
        if (name.find("fp16") != std::string::npos)
            return DistanceEvaluationBackend::FP16;
        if (name.find("int8") != std::string::npos)
            return DistanceEvaluationBackend::INT8;
        if (name.find("binary") != std::string::npos)
            return DistanceEvaluationBackend::BINARY;
        if (name.find("fp32") != std::string::npos)
            return DistanceEvaluationBackend::FP32;
        return DistanceEvaluationBackend::UNKNOWN;
    }

    static const char*
    BackendName(DistanceEvaluationBackend backend) {
        return DistanceEvaluationBackendName(backend);
    }

    static bool
    SaturatingAdd(std::atomic<uint64_t>& value, uint64_t amount) {
        return SaturatingAddAccepted(value, amount) < amount;
    }

    static uint64_t
    SaturatingAddAccepted(std::atomic<uint64_t>& value, uint64_t amount) {
        if (amount == 0) {
            return 0;
        }
        auto current = value.load(std::memory_order_relaxed);
        while (true) {
            const auto accepted = std::min(amount, std::numeric_limits<uint64_t>::max() - current);
            const auto next = current + accepted;
            if (value.compare_exchange_weak(
                    current, next, std::memory_order_relaxed, std::memory_order_relaxed))
                return accepted;
        }
    }

    void
    AddDistance(DistancePhase phase, DistanceEvaluationBackend backend, uint64_t count = 1) {
        if (count == 0) {
            return;
        }
        const auto accepted = SaturatingAddAccepted(distance_evaluations, count);
        bool overflowed = accepted < count;
        overflowed =
            SaturatingAdd(distance_evaluations_by_phase[static_cast<size_t>(phase)], accepted) or
            overflowed;
        auto backend_index = static_cast<uint8_t>(backend);
        overflowed =
            SaturatingAdd(distance_evaluations_by_backend[backend_index], accepted) or overflowed;
        if (overflowed) {
            complete.store(false, std::memory_order_relaxed);
        }
        if (backend == DistanceEvaluationBackend::UNKNOWN) {
            complete.store(false, std::memory_order_relaxed);
        }
    }

    void
    AddDistance(DistancePhase phase, const char* backend, uint64_t count = 1) {
        AddDistance(phase, BackendFromName(backend), count);
    }

    void
    AddDistance(DistancePhase phase, const std::string& backend, uint64_t count = 1) {
        AddDistance(phase, BackendFromName(backend), count);
    }

    [[nodiscard]] JsonType
    ToJson() const {
        JsonType j;
        j["is_timeout"].SetBool(is_timeout.load(std::memory_order_relaxed));
        j["dist_cmp"].SetUint64(dist_cmp.load(std::memory_order_relaxed));
        j["hops"].SetUint64(hops.load(std::memory_order_relaxed));
        j["io_cnt"].SetUint64(io_cnt.load(std::memory_order_relaxed));
        j["io_time_ms"].SetUint64(io_time_ms.load(std::memory_order_relaxed));
        j["reorder_distance_count"].SetUint64(
            reorder_distance_count.load(std::memory_order_relaxed));
        j["reorder_candidate_count"].SetUint64(
            reorder_candidate_count.load(std::memory_order_relaxed));
        j["reorder_lower_bound_probe_count"].SetUint64(
            reorder_lower_bound_probe_count.load(std::memory_order_relaxed));
        j["rabitq_filter_count"].SetUint64(rabitq_filter_count.load(std::memory_order_relaxed));
        j["rabitq_full_count"].SetUint64(rabitq_full_count.load(std::memory_order_relaxed));
        j["rabitq_filter_fallback_full_count"].SetUint64(
            rabitq_filter_fallback_full_count.load(std::memory_order_relaxed));
        j["rabitq_reorder_hint_full_count"].SetUint64(
            rabitq_reorder_hint_full_count.load(std::memory_order_relaxed));
        j["rabitq_reorder_fallback_full_count"].SetUint64(
            rabitq_reorder_fallback_full_count.load(std::memory_order_relaxed));
        j["query_computer_count"].SetUint64(query_computer_count.load(std::memory_order_relaxed));
        j["parallel_search_fallback_count"].SetUint64(
            parallel_search_fallback_count.load(std::memory_order_relaxed));
        j["distance_evaluations"].SetUint64(distance_evaluations.load(std::memory_order_relaxed));
        for (uint64_t i = 0; i < distance_evaluations_by_phase.size(); ++i) {
            j["distance_evaluations_by_phase"]
             [DistanceEvaluationPhaseName(static_cast<DistanceEvaluationPhase>(i))]
                 .SetUint64(distance_evaluations_by_phase[i].load(std::memory_order_relaxed));
        }
        for (uint64_t i = 0; i < distance_evaluations_by_backend.size(); ++i) {
            const auto backend = static_cast<DistanceEvaluationBackend>(i);
            j["distance_evaluations_by_backend"][DistanceEvaluationBackendName(backend)].SetUint64(
                distance_evaluations_by_backend[i].load(std::memory_order_relaxed));
        }
        j["complete"].SetBool(complete.load(std::memory_order_relaxed));
        return j;
    }

    [[nodiscard]] std::string
    Dump() const {
        return ToJson().Dump();
    }

public:
    std::atomic<bool> is_timeout{false};
    std::atomic<uint32_t> dist_cmp{0};
    std::atomic<uint32_t> hops{0};
    std::atomic<uint32_t> io_cnt{0};
    std::atomic<uint32_t> io_time_ms{0};
    std::atomic<uint32_t> reorder_distance_count{0};
    std::atomic<uint32_t> reorder_candidate_count{0};
    std::atomic<uint32_t> reorder_lower_bound_probe_count{0};
    std::atomic<uint32_t> rabitq_filter_count{0};
    std::atomic<uint32_t> rabitq_full_count{0};
    std::atomic<uint32_t> rabitq_filter_fallback_full_count{0};
    std::atomic<uint32_t> rabitq_reorder_hint_full_count{0};
    std::atomic<uint32_t> rabitq_reorder_fallback_full_count{0};
    std::atomic<uint32_t> query_computer_count{0};
    std::atomic<uint32_t> parallel_search_fallback_count{0};
    std::atomic<uint64_t> distance_evaluations{0};
    std::array<std::atomic<uint64_t>, static_cast<uint8_t>(DistanceEvaluationPhase::COUNT)>
        distance_evaluations_by_phase{};
    std::array<std::atomic<uint64_t>, static_cast<uint8_t>(DistanceEvaluationBackend::COUNT)>
        distance_evaluations_by_backend{};
    // Multi-vector (SIMQ) fine-grained statistics
    std::atomic<uint32_t> mv_io_time_ms{0};
    std::atomic<uint32_t> mv_compute_time_ms{0};
    std::atomic<uint32_t> mv_candidate_count{0};
    std::atomic<uint64_t> mv_io_bytes{0};
    std::atomic<bool> complete{true};
};

class SearchMetrics {
public:
    SearchStatistics base;

    /**
     * Add a query-specific field without replacing established values.
     * Returns false for a baseline-schema collision or a duplicate extension key; on failure, the
     * existing metric value is unchanged.
     */
    [[nodiscard]] bool
    AddExtension(std::string key, MetricValue value) {
        if (MetricsSchema::Base().Contains(key)) {
            return false;
        }
        return extensions_.emplace(std::move(key), std::move(value)).second;
    }

    /**
     * Consume the query-local collector after search workers have completed.
     * Relaxed loads are sufficient for diagnostics after worker synchronization; callers must not
     * snapshot concurrently with active writers or assume a transactional cross-counter view.
     */
    [[nodiscard]] SearchResultMetrics
    Snapshot() && {
        SearchResultMetrics result;
        result.is_timeout = base.is_timeout.load(std::memory_order_relaxed);
        result.dist_cmp = base.dist_cmp.load(std::memory_order_relaxed);
        result.hops = base.hops.load(std::memory_order_relaxed);
        result.io_cnt = base.io_cnt.load(std::memory_order_relaxed);
        result.io_time_ms = base.io_time_ms.load(std::memory_order_relaxed);
        result.reorder_distance_count = base.reorder_distance_count.load(std::memory_order_relaxed);
        result.reorder_candidate_count =
            base.reorder_candidate_count.load(std::memory_order_relaxed);
        result.reorder_lower_bound_probe_count =
            base.reorder_lower_bound_probe_count.load(std::memory_order_relaxed);
        result.rabitq_filter_count = base.rabitq_filter_count.load(std::memory_order_relaxed);
        result.rabitq_full_count = base.rabitq_full_count.load(std::memory_order_relaxed);
        result.rabitq_filter_fallback_full_count =
            base.rabitq_filter_fallback_full_count.load(std::memory_order_relaxed);
        result.rabitq_reorder_hint_full_count =
            base.rabitq_reorder_hint_full_count.load(std::memory_order_relaxed);
        result.rabitq_reorder_fallback_full_count =
            base.rabitq_reorder_fallback_full_count.load(std::memory_order_relaxed);
        result.query_computer_count = base.query_computer_count.load(std::memory_order_relaxed);
        result.parallel_search_fallback_count =
            base.parallel_search_fallback_count.load(std::memory_order_relaxed);
        result.distance_evaluations = base.distance_evaluations.load(std::memory_order_relaxed);
        for (uint64_t i = 0; i < result.distance_evaluations_by_phase.size(); ++i) {
            result.distance_evaluations_by_phase[i] =
                base.distance_evaluations_by_phase[i].load(std::memory_order_relaxed);
        }
        for (uint64_t i = 0; i < result.distance_evaluations_by_backend.size(); ++i) {
            result.distance_evaluations_by_backend[i] =
                base.distance_evaluations_by_backend[i].load(std::memory_order_relaxed);
        }
        result.complete = base.complete.load(std::memory_order_relaxed);
        result.extensions_ = std::move(extensions_);
        return result;
    }

private:
    std::map<std::string, MetricValue> extensions_;
};

template <typename QuantTmpl>
struct QuantizerDistanceBackend {
    static DistanceEvaluationBackend
    Get(const QuantTmpl& quantizer) {
        return SearchStatistics::BackendFromName(quantizer.Name());
    }
};

template <typename QuantTmpl, MetricType metric>
struct QuantizerDistanceBackend<TransformQuantizer<QuantTmpl, metric>> {
    static DistanceEvaluationBackend
    Get(const TransformQuantizer<QuantTmpl, metric>& quantizer) {
        return SearchStatistics::BackendFromName(quantizer.quantizer_->Name());
    }
};

inline Allocator*
select_query_allocator(QueryContext* ctx, Allocator* index_allocator) {
    if (ctx != nullptr and ctx->alloc != nullptr) {
        // use the query specified memory allocator
        return ctx->alloc;
    }

    // use the index allocator
    return index_allocator;
}

inline Allocator*
select_query_allocator(Allocator* query_allocator, Allocator* index_allocator) {
    if (query_allocator != nullptr) {
        // use the query specified memory allocator
        return query_allocator;
    }

    // use the index allocator
    return index_allocator;
}

}  // namespace vsag
