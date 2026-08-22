
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
#include <string>

#include "metric_type.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

enum class DistanceEvaluationPhase : uint8_t { ROUTING = 0, APPROXIMATE = 1, RERANK = 2 };

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
};

class SearchStatistics;
class ReasoningContext;
template <typename QuantTmpl, MetricType metric>
class TransformQuantizer;

struct QueryContext {
    Allocator* alloc = nullptr;
    SearchStatistics* stats = nullptr;
    ReasoningContext* reasoning_ctx = nullptr;
    float rabitq_error_rate = std::numeric_limits<float>::quiet_NaN();
    DistanceEvaluationPhase distance_phase = DistanceEvaluationPhase::APPROXIMATE;
    bool track_distance_evaluations = true;
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
        switch (phase) {
            case DistancePhase::ROUTING:
                return "routing";
            case DistancePhase::APPROXIMATE:
                return "approximate";
            case DistancePhase::RERANK:
                return "rerank";
            default:
                return "approximate";
        }
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
        static constexpr const char* names[] = {"fp32",
                                                "fp16",
                                                "bf16",
                                                "int8",
                                                "sq8",
                                                "sq4",
                                                "sq8_uniform",
                                                "sq4_uniform",
                                                "pq",
                                                "pq_fastscan",
                                                "rabitq",
                                                "binary",
                                                "sparse_fp32",
                                                "sparse_fp16",
                                                "sparse_sq8",
                                                "unknown"};
        return names[static_cast<uint8_t>(backend)];
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

    // Same clamping semantics as SaturatingAddAccepted, but with plain
    // load/add/store: for a thread-private counter the locked RMW is pure
    // overhead (~20-45 cycles vs ~3).
    static uint64_t
    PlainSaturatingAddAccepted(std::atomic<uint64_t>& value, uint64_t amount) {
        if (amount == 0) {
            return 0;
        }
        const auto current = value.load(std::memory_order_relaxed);
        const auto accepted = std::min(amount, std::numeric_limits<uint64_t>::max() - current);
        value.store(current + accepted, std::memory_order_relaxed);
        return accepted;
    }

    void
    AddDistance(DistancePhase phase, DistanceEvaluationBackend backend, uint64_t count = 1) {
        if (count == 0) {
            return;
        }
        auto backend_index = static_cast<uint8_t>(backend);
        bool overflowed = false;
        if (!parallel_.load(std::memory_order_relaxed)) {
            // Thread-private statistics: skip the locked read-modify-writes.
            const auto accepted =
                PlainSaturatingAddAccepted(distance_evaluations, count);
            overflowed = accepted < count;
            overflowed = PlainSaturatingAddAccepted(
                             distance_evaluations_by_phase[static_cast<size_t>(phase)], accepted) <
                             accepted or
                         overflowed;
            overflowed = PlainSaturatingAddAccepted(
                             distance_evaluations_by_backend[backend_index], accepted) < accepted or
                         overflowed;
        } else {
            const auto accepted = SaturatingAddAccepted(distance_evaluations, count);
            overflowed = accepted < count;
            overflowed = SaturatingAdd(
                             distance_evaluations_by_phase[static_cast<size_t>(phase)], accepted) or
                         overflowed;
            overflowed =
                SaturatingAdd(distance_evaluations_by_backend[backend_index], accepted) or
                overflowed;
        }
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
        j["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
        j["hops"].SetInt(hops.load(std::memory_order_relaxed));
        j["io_cnt"].SetInt(io_cnt.load(std::memory_order_relaxed));
        j["io_time_ms"].SetInt(io_time_ms.load(std::memory_order_relaxed));
        j["reorder_distance_count"].SetInt(reorder_distance_count.load(std::memory_order_relaxed));
        j["reorder_lower_bound_probe_count"].SetInt(
            reorder_lower_bound_probe_count.load(std::memory_order_relaxed));
        j["rabitq_filter_count"].SetInt(rabitq_filter_count.load(std::memory_order_relaxed));
        j["rabitq_full_count"].SetInt(rabitq_full_count.load(std::memory_order_relaxed));
        j["rabitq_filter_fallback_full_count"].SetInt(
            rabitq_filter_fallback_full_count.load(std::memory_order_relaxed));
        j["rabitq_reorder_hint_full_count"].SetInt(
            rabitq_reorder_hint_full_count.load(std::memory_order_relaxed));
        j["rabitq_reorder_fallback_full_count"].SetInt(
            rabitq_reorder_fallback_full_count.load(std::memory_order_relaxed));
        j["distance_evaluations"].SetUint64(distance_evaluations.load(std::memory_order_relaxed));
        for (size_t i = 0; i < 3; ++i) {
            j["distance_evaluations_by_phase"][PhaseName(static_cast<DistancePhase>(i))].SetUint64(
                distance_evaluations_by_phase[i].load(std::memory_order_relaxed));
        }
        for (size_t i = 0; i < distance_evaluations_by_backend.size(); ++i) {
            const auto backend = static_cast<DistanceEvaluationBackend>(i);
            j["distance_evaluations_by_backend"][BackendName(backend)].SetUint64(
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
    // True while multiple threads mutate this statistics object concurrently
    // (parallel search); selects the locked RMW path in AddDistance. Most
    // searches are thread-private and take the plain path.
    std::atomic<bool> parallel_{false};

    std::atomic<bool> is_timeout{false};
    std::atomic<uint32_t> dist_cmp{0};
    std::atomic<uint32_t> hops{0};
    std::atomic<uint32_t> io_cnt{0};
    std::atomic<uint32_t> io_time_ms{0};
    std::atomic<uint32_t> reorder_distance_count{0};
    std::atomic<uint32_t> reorder_lower_bound_probe_count{0};
    std::atomic<uint32_t> rabitq_filter_count{0};
    std::atomic<uint32_t> rabitq_full_count{0};
    std::atomic<uint32_t> rabitq_filter_fallback_full_count{0};
    std::atomic<uint32_t> rabitq_reorder_hint_full_count{0};
    std::atomic<uint32_t> rabitq_reorder_fallback_full_count{0};
    std::atomic<uint64_t> distance_evaluations{0};
    std::array<std::atomic<uint64_t>, 3> distance_evaluations_by_phase{};
    std::array<std::atomic<uint64_t>, 16> distance_evaluations_by_backend{};
    std::atomic<bool> complete{true};
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
