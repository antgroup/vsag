// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sindi_v2_parallel_search.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "impl/logger/logger.h"
#include "impl/searcher/search_execution_context.h"
#include "vsag_exception.h"

namespace vsag {
namespace {
void
hash_word(uint64_t& hash, uint64_t word) {
    for (uint32_t i = 0; i < 8; ++i) {
        hash = (hash ^ (word & 255)) * 1099511628211ULL;
        word >>= 8;
    }
}
}  // namespace

SindiV2SearchProfile
make_sindi_v2_profile(const SINDIV2SearchParameter& parameters,
                      SindiV2SearchProfile::Clock::time_point start) {
    SindiV2SearchProfile profile;
    profile.start = start;
    profile.enabled = parameters.profile_search;
    profile.diagnose = parameters.diagnose_search;
    profile.profile_cpu = parameters.profile_cpu;
    profile.requested = parameters.window_parallelism;
    profile.candidate_ratio = parameters.parallel_n_candidate_ratio;
    if (parameters.enable_time_record && profile.requested > 1) {
        const auto timeout = parameters.timeout_ms;
        CHECK_ARGUMENT(std::isfinite(timeout), "timeout_ms must be finite");
        const auto maximum = std::chrono::duration<double, std::milli>(
                                 SindiV2SearchProfile::Clock::time_point::max() - start)
                                 .count();
        if (timeout <= 0) {
            profile.deadline = start;
        } else if (timeout < maximum) {
            profile.deadline =
                start + std::chrono::duration_cast<SindiV2SearchProfile::Clock::duration>(
                            std::chrono::duration<double, std::milli>(timeout));
        }
    }
    return profile;
}

void
SindiV2SearchProfile::CheckDeadline() {
    if (parallel && deadline != Clock::time_point::max() && Clock::now() >= deadline) {
        stop = "deadline";
        throw VsagException(ErrorType::INTERNAL_ERROR, "SINDI_V2 parallel deadline exceeded");
    }
}

void
SindiV2SearchProfile::RecordCompute(uint32_t window, uint64_t elapsed_ns, uint64_t postings) {
    if (!enabled) {
        return;
    }
    constexpr std::array<uint64_t, 5> bounds{50000, 200000, 1000000, 5000000, 20000000};
    const auto bucket = std::upper_bound(bounds.begin(), bounds.end(), elapsed_ns) - bounds.begin();
    ++compute_histogram[bucket];
    compute_sum_ns += elapsed_ns;
    if (elapsed_ns >= compute_max_ns) {
        compute_max_ns = elapsed_ns;
        slowest_window = window;
        slowest_window_postings = postings;
    }
}

SindiV2FailureDiagnostics::~SindiV2FailureDiagnostics() {
    if (std::uncaught_exceptions() <= exceptions_ || !profile_.enabled) {
        return;
    }
    try {
        const bool timeout = std::string(profile_.stop) == "deadline";
        profile_.stop = timeout ? "deadline" : "error";
        if (timeout) {
            statistics_.is_timeout.store(true, std::memory_order_relaxed);
        }
        logger::error("SINDI_V2 execution failure " + profile_.Dump(statistics_));
    } catch (...) {  // NOLINT(bugprone-empty-catch): preserve the original query exception.
        // Diagnostic allocation/logging must never replace the original failure.
    }
}

void
SindiV2SearchProfile::DigestCandidates(const MaxHeap& heap) {
    candidate_count = heap.size();
    if (!diagnose) {
        return;
    }
    auto copy = heap;
    while (!copy.empty()) {
        uint32_t bits;
        const auto candidate = copy.top();
        std::memcpy(&bits, &candidate.first, sizeof(bits));
        hash_word(candidate_digest, bits);
        hash_word(candidate_digest, candidate.second);
        copy.pop();
    }
}

void
SindiV2SearchProfile::DigestResults(const DatasetPtr& result) {
    if (!diagnose) {
        return;
    }
    hash_word(result_digest, static_cast<uint64_t>(result->GetDim()));
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        uint32_t bits;
        const auto distance = result->GetDistances()[i];
        std::memcpy(&bits, &distance, sizeof(bits));
        hash_word(result_digest, bits);
        hash_word(result_digest, static_cast<uint64_t>(result->GetIds()[i]));
    }
}

void
SindiV2SearchProfile::CapturePartitions(const SindiParallelState& state) noexcept {
    granted = state.granted;
    filter_slots = state.lease ? state.lease->evaluators.size() : 0;
    partitions = state.plan.quotas.size();
    capacity = state.plan.capacity;
    effective_candidates = state.plan.candidates;
    for (uint64_t i = 0; i < state.plan.boundaries.size() and i < boundaries.size(); ++i) {
        boundaries[i] = state.plan.boundaries[i];
    }
    worker_cpu_ns = state.worker_cpu_ns;
    worker_cpu_available = state.worker_cpu_available;
    for (uint64_t i = 0; i < partitions and i < quotas.size(); ++i) {
        quotas[i] = state.plan.quotas[i];
    }
    if (const auto* diagnostics = state.diagnostics) {
        wait_ns = diagnostics->wait_ns;
        merge_ns = diagnostics->merge_ns;
        admission_ns = diagnostics->admission_ns;
        slot_prepare_ns = diagnostics->slot_prepare_ns;
        for (uint64_t i = 0; i < diagnostics->partitions.size() and i < quotas.size(); ++i) {
            partition_statistics[i] = diagnostics->partitions[i];
        }
        for (const auto& slot : diagnostics->slots) {
            computed_windows += slot.computed;
            attempted_windows += slot.attempted;
            consumed_windows += slot.consumed;
            compute_sum_ns += slot.compute_ns;
            consume_ns += slot.consume_ns;
            clear_ns += slot.clear_ns;
            filter_ns += slot.filter_ns;
            filter_calls += slot.filter_calls;
            scan_postings += slot.postings;
            for (uint64_t i = 0; i < compute_histogram.size(); ++i) {
                compute_histogram[i] += slot.histogram[i];
            }
            if (slot.compute_max_ns >= compute_max_ns) {
                compute_max_ns = slot.compute_max_ns;
                slowest_window = slot.slowest_window;
                slowest_window_postings = slot.slowest_postings;
            }
        }
    }
    switch (state.stop.load()) {
        case SindiParallelState::Stop::CALLBACK_LIMIT:
            stop = "callback_limit";
            break;
        case SindiParallelState::Stop::ERROR:
            stop = "error";
            break;
        case SindiParallelState::Stop::DEADLINE:
            stop = "deadline";
            break;
        case SindiParallelState::Stop::NONE:
            break;
    }
}

std::string
SindiV2SearchProfile::Dump(const SearchStatistics& statistics) const {
    auto json = statistics.ToJson();
    const auto us = [&](const char* key, uint64_t ns) {
        json[key].SetDouble(static_cast<double>(ns) / 1000.0);
    };
    json["sindi_v2_strategy"].SetString(parallel ? "partition_search" : "serial");
    json["sindi_v2_fallback"].SetString(fallback);
    json["sindi_v2_stop_reason"].SetString(stop);
    json["sindi_v2_execution_complete"].SetBool(std::string(stop) == "complete");
    json["sindi_v2_requested_slots"].SetUint64(requested);
    json["sindi_v2_granted_slots"].SetUint64(granted);
    if (parallel) {
        json["sindi_v2_partitions"].SetUint64(partitions);
        json["sindi_v2_candidate_ratio"].SetDouble(candidate_ratio);
        json["sindi_v2_candidate_capacity"].SetUint64(capacity);
        json["sindi_v2_n_candidate"].SetUint64(effective_candidates);
        json["sindi_v2_filter_slots"].SetUint64(filter_slots);
        json["sindi_v2_routed_windows"].SetUint64(routed_windows);
        if (diagnose) {
            json["sindi_v2_route_digest"].SetString(std::to_string(route_digest));
        }
        json["sindi_v2_windows_attempted"].SetUint64(attempted_windows);
        for (uint64_t i = 0; i < partitions and i < quotas.size(); ++i) {
            const auto prefix = "sindi_v2_partition_" + std::to_string(i);
            json[prefix + "_quota"].SetUint64(quotas[i]);
            json[prefix + "_begin_offset"].SetUint64(boundaries[i]);
            json[prefix + "_end_offset"].SetUint64(boundaries[i + 1]);
            json[prefix + "_windows"].SetUint64(partition_statistics[i].windows);
            json[prefix + "_candidates"].SetUint64(partition_statistics[i].candidates);
            if (enabled) {
                us((prefix + "_work_us").c_str(), partition_statistics[i].wall_ns);
            }
        }
    }
    json["sindi_v2_windows_computed"].SetUint64(computed_windows);
    json["sindi_v2_windows_consumed"].SetUint64(consumed_windows);
    json["sindi_v2_candidates"].SetUint64(candidate_count);
    json["sindi_v2_profile_enabled"].SetBool(enabled);
    json["sindi_v2_cpu_profile_enabled"].SetBool(profile_cpu);
    if (profile_cpu) {
        const bool compute_available = worker_cpu_available;
        json["sindi_v2_caller_cpu_available"].SetBool(caller_cpu.available);
        json["sindi_v2_worker_cpu_available"].SetBool(compute_available);
        if (caller_cpu.available) {
            us("sindi_v2_caller_cpu_us", caller_cpu.nanoseconds);
        }
        if (compute_available) {
            us("sindi_v2_worker_cpu_us", worker_cpu_ns);
        }
    }
    if (enabled) {
        us("sindi_v2_prepare_us", prepare_ns);
        us(parallel ? "sindi_v2_partition_wall_us" : "sindi_v2_batch_wall_us", batch_ns);
        if (!parallel) {
            us("sindi_v2_caller_compute_us", caller_compute_ns);
        }
        if (parallel) {
            us("sindi_v2_clear_sum_us", clear_ns);
            us("sindi_v2_filter_sum_us", filter_ns);
            json["sindi_v2_filter_calls"].SetUint64(filter_calls);
            json["sindi_v2_scratch_bytes"].SetUint64(scratch_bytes);
            us("sindi_v2_merge_us", merge_ns);
            us("sindi_v2_admission_us", admission_ns);
            us("sindi_v2_slot_prepare_us", slot_prepare_ns);
        }
        us("sindi_v2_join_wait_us", wait_ns);
        us("sindi_v2_compute_sum_us", compute_sum_ns);
        us("sindi_v2_compute_max_us", compute_max_ns);
        constexpr std::array<const char*, 6> buckets{
            "lt_50us", "50_200us", "200_1000us", "1000_5000us", "5000_20000us", "ge_20000us"};
        for (uint64_t i = 0; i < buckets.size(); ++i) {
            json[std::string("sindi_v2_window_compute_") + buckets[i]].SetUint64(
                compute_histogram[i]);
        }
        json["sindi_v2_slowest_window"].SetUint64(slowest_window);
        if (scan_postings_available) {
            json["sindi_v2_slowest_window_postings"].SetUint64(slowest_window_postings);
        }
        us(parallel ? "sindi_v2_consume_sum_us" : "sindi_v2_consume_us", consume_ns);
        us("sindi_v2_rerank_us", rerank_ns);
        us("sindi_v2_output_us", output_ns);
        const auto total = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
        us("sindi_v2_total_us", total);
        const auto accounted =
            prepare_ns + batch_ns + (parallel ? 0 : consume_ns) + rerank_ns + output_ns;
        us("sindi_v2_other_us", total > accounted ? total - accounted : 0);
        us("sindi_v2_rerank_read_us", rerank.read_ns);
        us("sindi_v2_rerank_compute_us", rerank.compute_ns);
        json["sindi_v2_rerank_logical_bytes"].SetUint64(rerank.read_bytes);
        json["sindi_v2_rerank_read_ranges"].SetUint64(rerank.read_ranges);
        json["sindi_v2_rerank_io_strategy"].SetString(rerank.io_strategy);
        json["sindi_v2_query_terms"].SetUint64(query_terms);
        if (scan_postings_available) {
            json["sindi_v2_scanned_postings"].SetUint64(scan_postings);
        }
    }
    if (diagnose) {
        json["sindi_v2_candidate_digest"].SetString(std::to_string(candidate_digest));
        json["sindi_v2_result_digest"].SetString(std::to_string(result_digest));
    }
    return json.Dump();
}

}  // namespace vsag
