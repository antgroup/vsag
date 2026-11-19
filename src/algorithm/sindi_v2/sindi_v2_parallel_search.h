// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
#include <string>

#include "algorithm/sindi_v2/sindi_v2_parameter.h"
#include "datacell/sindi_search_term_datacell.h"
#include "impl/searcher/sindi_cpu_timer.h"
#include "impl/searcher/sindi_parallel_search.h"
#include "query_context.h"

namespace vsag {

class SearchExecutionContext;

struct SindiV2SearchProfile {
    using Clock = std::chrono::steady_clock;
    bool enabled{false};
    bool diagnose{false};
    bool profile_cpu{false};
    SindiCpuTime caller_cpu;
    uint64_t worker_cpu_ns{0};
    bool worker_cpu_available{true};
    bool parallel{false};
    uint32_t requested{1};
    uint32_t granted{1};
    double candidate_ratio{2};
    uint64_t partitions{0};
    uint64_t capacity{0};
    uint64_t effective_candidates{0};
    uint64_t filter_slots{0};
    uint64_t attempted_windows{0};
    uint64_t clear_ns{0};
    uint64_t filter_ns{0};
    uint64_t filter_calls{0};
    uint64_t admission_ns{0};
    uint64_t slot_prepare_ns{0};
    uint64_t routed_windows{0};
    uint64_t route_digest{14695981039346656037ULL};
    std::array<uint64_t, 17> boundaries{};
    uint64_t merge_ns{0};
    uint64_t scratch_bytes{0};
    std::array<uint64_t, 16> quotas{};
    std::array<SindiPartitionStatistics, 16> partition_statistics{};

    void
    CapturePartitions(const SindiParallelState& state) noexcept;
    const char* fallback{"disabled"};
    const char* stop{"complete"};
    Clock::time_point start{Clock::now()};
    Clock::time_point deadline{Clock::time_point::max()};
    uint64_t prepare_ns{0};
    uint64_t batch_ns{0};
    uint64_t caller_compute_ns{0};
    uint64_t wait_ns{0};
    uint64_t compute_sum_ns{0};
    uint64_t compute_max_ns{0};
    std::array<uint64_t, 6> compute_histogram{};
    uint32_t slowest_window{0};
    uint64_t slowest_window_postings{0};
    uint64_t consume_ns{0};
    uint64_t rerank_ns{0};
    uint64_t output_ns{0};
    uint64_t computed_windows{0};
    uint64_t consumed_windows{0};
    uint64_t candidate_count{0};
    uint64_t query_terms{0};
    uint64_t scan_postings{0};
    bool scan_postings_available{false};
    uint64_t candidate_digest{14695981039346656037ULL};
    uint64_t result_digest{14695981039346656037ULL};
    SparseQueryProfile rerank;

    void
    CheckDeadline();
    void
    RecordCompute(uint32_t window, uint64_t elapsed_ns, uint64_t postings);
    std::string
    Dump(const SearchStatistics& statistics) const;
    void
    DigestCandidates(const MaxHeap& heap);
    void
    DigestResults(const DatasetPtr& result);
};

SindiV2SearchProfile
make_sindi_v2_profile(const SINDIV2SearchParameter& parameters,
                      SindiV2SearchProfile::Clock::time_point start);

// Errors cannot carry a Dataset. Emit the same bounded, payload-free diagnostics
// when profiling is enabled, while preserving the original exception.
class SindiV2FailureDiagnostics {
public:
    SindiV2FailureDiagnostics(SindiV2SearchProfile& profile, SearchStatistics& statistics)
        : profile_(profile), statistics_(statistics), exceptions_(std::uncaught_exceptions()) {
    }
    ~SindiV2FailureDiagnostics();

private:
    SindiV2SearchProfile& profile_;
    SearchStatistics& statistics_;
    int exceptions_;
};

}  // namespace vsag
