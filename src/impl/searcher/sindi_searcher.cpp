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

#include "sindi_searcher.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>

#include "datacell/flatten_interface.h"
#include "datacell/immutable_sindi_term_datacell.h"
#include "datacell/mutable_sindi_term_datacell.h"
#include "vsag_exception.h"

namespace vsag {
namespace {
using Clock = std::chrono::steady_clock;

double
elapsed_us(Clock::time_point begin) {
    return std::chrono::duration<double, std::micro>(Clock::now() - begin).count();
}

struct alignas(64) AtomicRange {
    std::atomic<uint64_t> next{0};
    uint64_t end{0};
};

template <typename Prefix>
void
partition_by_prefix(std::vector<uint64_t>& bounds, uint64_t count, const Prefix& prefix) {
    const uint64_t workers = bounds.size() - 1;
    const auto total = prefix(count);
    if (total == 0) {
        return;
    }
    for (uint64_t i = 1; i < workers; ++i) {
        const auto minimum = bounds[i - 1] + 1;
        const auto maximum = count - (workers - i);
        const auto target = total / workers * i + (total % workers) * i / workers;
        uint64_t low = minimum, high = maximum + 1;
        while (low < high) {
            const auto middle = low + (high - low) / 2;
            if (prefix(middle) < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        auto cut = std::min<uint64_t>(low, maximum);
        if (cut > minimum and prefix(cut) >= target and target >= prefix(cut - 1) and
            target - prefix(cut - 1) < prefix(cut) - target) {
            --cut;
        }
        bounds[i] = cut;
    }
}

// Join every submitted task even if submission or a worker fails. No in-pool barriers:
// logical workers may outnumber pool threads, including a single-thread pool.
template <typename Work>
void
run_query_workers(uint32_t count,
                  const SafeThreadPoolPtr& pool,
                  const Work& work,
                  bool caller_participates = true) {
    std::vector<std::future<void>> futures;
    const auto submitted = count - static_cast<uint32_t>(caller_participates);
    futures.reserve(submitted);
    std::exception_ptr error;
    try {
        for (uint32_t me = 0; me < submitted; ++me) {
            futures.push_back(pool->GeneralEnqueue(work, me));
        }
        if (caller_participates) {
            work(count - 1);
        }
    } catch (...) {
        error = std::current_exception();
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (error == nullptr) {
                error = std::current_exception();
            }
        }
    }
    if (error != nullptr) {
        std::rethrow_exception(error);
    }
}

struct WindowQueue {
    std::mutex mutex;
    std::deque<uint32_t> windows;
};
}  // namespace

SindiQueryStageProfile&
SindiLastQueryProfile() {
    thread_local SindiQueryStageProfile profile;
    return profile;
}

SindiRerankProfile&
SindiLastRerankProfile() {
    thread_local SindiRerankProfile profile;
    return profile;
}

const char*
SindiSearcher::SchedulerName(SindiWindowScheduler scheduler) {
    switch (scheduler) {
        case SindiWindowScheduler::COUNT_DEQUE:
            return "count_deque";
        case SindiWindowScheduler::COUNT_ATOMIC:
            return "count_atomic";
        case SindiWindowScheduler::COST_ATOMIC:
            return "cost_atomic";
    }
    return "unknown";
}

SindiSearchOptions
SindiSearchOptions::Parse(const JsonType& json) {
    SindiSearchOptions options;
    const auto param = json[INDEX_SINDI];
    if (param.Contains("window_scheduler")) {
        const auto name = param["window_scheduler"].GetString();
        if (name == "count_deque") {
            options.scheduler = SindiWindowScheduler::COUNT_DEQUE;
        } else if (name == "count_atomic") {
            options.scheduler = SindiWindowScheduler::COUNT_ATOMIC;
        } else {
            CHECK_ARGUMENT(name == "cost_atomic", "unknown SINDI window_scheduler: " + name);
            options.scheduler = SindiWindowScheduler::COST_ATOMIC;
        }
    }
    return options;
}

void
SindiSearchOptions::WriteTo(JsonType& json) const {
    if (scheduler != SindiWindowScheduler::COUNT_DEQUE) {
        json[INDEX_SINDI]["window_scheduler"].SetString(SindiSearcher::SchedulerName(scheduler));
    }
}

std::vector<uint64_t>
SindiSearcher::Partition(const std::vector<uint64_t>& costs, uint32_t workers, bool weighted) {
    CHECK_ARGUMENT(workers > 0, "window worker count must be positive");
    const uint64_t count = costs.size();
    std::vector<uint64_t> bounds(workers + 1, 0);
    const auto per_worker = (count + workers - 1) / workers;
    for (uint32_t i = 1; i <= workers; ++i) {
        bounds[i] = std::min<uint64_t>(i * per_worker, count);
    }
    if (not weighted or count < workers) {
        return bounds;
    }
    std::vector<uint64_t> prefix(count + 1, 0);
    std::partial_sum(costs.begin(), costs.end(), prefix.begin() + 1);
    partition_by_prefix(bounds, count, [&](uint64_t position) { return prefix[position]; });
    bounds.back() = count;
    return bounds;
}

uint64_t
SindiSearcher::EstimateWindowCost(const SindiSearchTermDataCell& datacell,
                                  const SparseTermComputer& computer,
                                  uint32_t window_id,
                                  bool full_term_scan) {
    const auto* compact = dynamic_cast<const ImmutableSindiTermDataCell*>(&datacell);
    const auto* mutable_cell = dynamic_cast<const MutableSindiTermDataCell*>(&datacell);
    CHECK_ARGUMENT(compact != nullptr or mutable_cell != nullptr,
                   "cost_atomic requires an in-memory SINDI DataCell");
    uint64_t cost = 0;
    for (uint32_t i = 0; i < computer.pruned_len_; ++i) {
        const auto term = computer.sorted_query_[i].first;
        uint32_t postings = 0;
        if (compact != nullptr) {
            const auto& window = compact->GetWindows().at(window_id);
            uint64_t local = term;
            if (not window.sorted_global_terms.empty()) {
                const auto found = std::lower_bound(
                    window.sorted_global_terms.begin(), window.sorted_global_terms.end(), term);
                if (found == window.sorted_global_terms.end() or *found != term) {
                    continue;
                }
                local = found - window.sorted_global_terms.begin();
            }
            if (local + 1 < window.offsets.size()) {
                postings = window.offsets[local + 1] - window.offsets[local];
            }
        } else {
            const auto& window = mutable_cell->GetWindow(window_id);
            if (term < window.term_sizes_.size()) {
                postings = window.term_sizes_[term];
            }
        }
        cost += full_term_scan ? postings : computer.GetTermScanCount(postings);
    }
    return cost;
}

bool
SindiSearcher::MergeHeaps(std::vector<SindiWindowHeap>& heaps,
                          MaxHeap& heap,
                          uint64_t limit,
                          Allocator* allocator) {
    if (limit == 0) {
        return true;
    }
    auto legacy_merge = [&]() {
        for (auto& local_heap : heaps) {
            while (not local_heap.empty()) {
                const auto candidate = local_heap.top();
                if (heap.size() < limit or candidate < heap.top()) {
                    heap.push(candidate);
                    if (heap.size() > limit) {
                        heap.pop();
                    }
                }
                local_heap.pop();
            }
        }
    };
    if (not heap.empty()) {
        legacy_merge();
        return false;
    }
    uint64_t total = 0;
    for (const auto& local_heap : heaps) {
        total += local_heap.size();
    }
    Vector<std::pair<float, InnerIdType>> candidates(allocator);
    candidates.reserve(total);
    for (const auto& local_heap : heaps) {
        for (const auto& candidate : local_heap.Candidates()) {
            // Inspect the exponent bits: -Ofast can optimize away std::isfinite checks.
            uint32_t bits = 0;
            std::memcpy(&bits, &candidate.first, sizeof(bits));
            if ((bits & 0x7F800000U) == 0x7F800000U) {
                legacy_merge();
                return false;
            }
            candidates.push_back(candidate);
        }
    }
    const auto selected = std::min(total, limit);
    const CompareByFirst compare;
    if (selected < total) {
        std::nth_element(
            candidates.begin(), candidates.begin() + selected, candidates.end(), compare);
    }
    if (selected > 0 and selected < total) {
        const auto worst =
            std::max_element(candidates.begin(), candidates.begin() + selected, compare);
        if (worst->first == candidates[selected].first) {
            legacy_merge();
            return false;
        }
    }
    candidates.resize(selected);
    heap = MaxHeap(compare, std::move(candidates));
    return true;
}

Vector<float>
SindiSearcher::RerankDistances(FlattenInterface& datacell,
                               const SparseVector& query,
                               const Vector<InnerIdType>& ids,
                               uint32_t workers,
                               const SafeThreadPoolPtr& pool,
                               Allocator* allocator,
                               SearchStatistics* statistics,
                               bool stage_profile,
                               SindiRerankScheduler scheduler) {
    auto& profile = SindiLastRerankProfile();
    profile = {};
    CHECK_ARGUMENT(workers > 0, "rerank worker count must be positive");
    Vector<float> distances(ids.size(), allocator);
    if (ids.empty()) {
        return distances;
    }
    const auto count =
        static_cast<uint32_t>(std::min<uint64_t>(pool == nullptr ? 1 : workers, ids.size()));
    const bool atomic = scheduler != SindiRerankScheduler::COUNT_STATIC and count > 1;
    const bool weighted = scheduler == SindiRerankScheduler::NNZ_ATOMIC and count > 1;
    const uint64_t estimated_count = weighted or stage_profile ? ids.size() : 0;
    std::vector<uint64_t> costs(estimated_count, 0);
    std::vector<uint64_t> local_prefix(estimated_count, 0);
    std::vector<uint64_t> block_offsets(count + 1, 0);
    if (stage_profile) {
        profile.workers.resize(count);
    }
    if (weighted or stage_profile) {
        const auto begin = stage_profile ? Clock::now() : Clock::time_point{};
        run_query_workers(count, pool, [&](uint32_t me) {
            const auto worker_begin = stage_profile ? Clock::now() : Clock::time_point{};
            const auto first = ids.size() * me / count;
            const auto last = ids.size() * (me + 1) / count;
            uint64_t total = 0;
            for (uint64_t i = first; i < last; ++i) {
                // Sparse FP32/FP16 codes start with uint32_t NNZ; RAII releases copied codes.
                const auto codes = datacell.AcquireCodesById(ids[i]);
                uint32_t nonzeros = 0;
                std::memcpy(&nonzeros, codes.Data(), sizeof(nonzeros));
                costs[i] = nonzeros;
                local_prefix[i] = total;
                total += nonzeros;
            }
            block_offsets[me + 1] = total;
            if (stage_profile) {
                profile.workers[me].cost_estimation_us = elapsed_us(worker_begin);
                profile.workers[me].cost_start_delay_us =
                    std::chrono::duration<double, std::micro>(worker_begin - begin).count();
            }
        });
        if (stage_profile) {
            // Critical-path wall time includes dispatch and joining the parallel estimation.
            profile.cost_estimation_us = elapsed_us(begin);
        }
    }
    const auto partition_begin = stage_profile ? Clock::now() : Clock::time_point{};
    // Only p block totals are combined on the caller; no serial O(candidate count) scan.
    std::partial_sum(block_offsets.begin(), block_offsets.end(), block_offsets.begin());
    const auto prefix = [&](uint64_t position) -> uint64_t {
        if (position == ids.size()) {
            return block_offsets.back();
        }
        // Invert first=floor(N*block/p), including uneven block sizes.
        const auto block = ((position + 1) * count - 1) / ids.size();
        return block_offsets[block] + local_prefix[position];
    };
    std::vector<uint64_t> bounds(count + 1, 0);
    for (uint32_t i = 0; i <= count; ++i) {
        bounds[i] = ids.size() * i / count;
    }
    if (weighted) {
        partition_by_prefix(bounds, ids.size(), prefix);
    }
    auto ranges = std::make_unique<AtomicRange[]>(atomic ? count : 0);
    for (uint32_t i = 0; atomic and i < count; ++i) {
        ranges[i].next.store(bounds[i], std::memory_order_relaxed);
        ranges[i].end = bounds[i + 1];
    }
    if (stage_profile) {
        profile.boundaries = bounds;
        profile.initial_costs.resize(count);
        profile.workers.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            profile.initial_costs[i] = prefix(bounds[i + 1]) - prefix(bounds[i]);
        }
        profile.partition_us = elapsed_us(partition_begin);
    }
    std::vector<std::unique_ptr<SearchStatistics>> local_statistics;
    if (statistics != nullptr) {
        local_statistics.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            local_statistics.emplace_back(std::make_unique<SearchStatistics>());
        }
    }
    const auto dispatch_begin = stage_profile ? Clock::now() : Clock::time_point{};
    auto run = [&](uint32_t me) {
        const auto begin = stage_profile ? Clock::now() : Clock::time_point{};
        SindiRerankWorkerProfile local =
            stage_profile ? profile.workers[me] : SindiRerankWorkerProfile{};
        auto computer = datacell.FactoryComputer(&query);
        QueryContext context{.alloc = allocator,
                             .stats = statistics != nullptr ? local_statistics[me].get() : nullptr,
                             .distance_phase = DistanceEvaluationPhase::RERANK};
        if (not atomic) {
            datacell.Query(distances.data() + bounds[me],
                           computer,
                           ids.data() + bounds[me],
                           static_cast<InnerIdType>(bounds[me + 1] - bounds[me]),
                           &context);
            if (stage_profile) {
                local.candidates = bounds[me + 1] - bounds[me];
                local.nonzeros = profile.initial_costs[me];
            }
        } else {
            for (uint32_t offset = 0; offset < count; ++offset) {
                const auto victim = (me + offset) % count;
                if (stage_profile and offset > 0) {
                    ++local.steal_attempts;
                }
                for (;;) {
                    const auto position =
                        ranges[victim].next.fetch_add(1, std::memory_order_relaxed);
                    if (stage_profile) {
                        ++local.cursor_fetches;
                    }
                    if (position >= ranges[victim].end) {
                        break;
                    }
                    datacell.Query(
                        distances.data() + position, computer, ids.data() + position, 1, &context);
                    if (stage_profile) {
                        ++local.candidates;
                        local.nonzeros += costs[position];
                        local.stolen_candidates += offset > 0;
                    }
                }
            }
        }
        if (stage_profile) {
            local.start_delay_us =
                std::chrono::duration<double, std::micro>(begin - dispatch_begin).count();
            local.elapsed_us = elapsed_us(begin);
            profile.workers[me] = local;
        }
    };
    run_query_workers(count, pool, run);
    if (stage_profile) {
        profile.dispatch_join_us = elapsed_us(dispatch_begin);
    }
    if (statistics != nullptr) {
        SindiStageTimer timer(stage_profile, SindiLastQueryProfile().statistics_counter_us);
        std::array<uint64_t, static_cast<uint64_t>(DistanceEvaluationBackend::COUNT)> counts{};
        for (const auto& local : local_statistics) {
            for (uint64_t backend = 0; backend < counts.size(); ++backend) {
                counts[backend] +=
                    local->distance_evaluations_by_backend[backend].load(std::memory_order_relaxed);
            }
            if (not local->complete.load(std::memory_order_relaxed)) {
                statistics->complete.store(false, std::memory_order_relaxed);
            }
        }
        for (uint64_t backend = 0; backend < counts.size(); ++backend) {
            statistics->AddDistance(DistanceEvaluationPhase::RERANK,
                                    static_cast<DistanceEvaluationBackend>(backend),
                                    counts[backend]);
        }
    }
    return distances;
}

void
SindiSearcher::SearchWindows(const SindiWindowSearchRequest& req,
                             const SindiSearchOptions& options,
                             SindiWindowSearchProfile& profile,
                             MaxHeap& heap) {
    const auto count = static_cast<uint32_t>(req.search.parallel_search_thread_count);
    const bool deque_mode = options.scheduler == SindiWindowScheduler::COUNT_DEQUE;
    const bool weighted = options.scheduler == SindiWindowScheduler::COST_ATOMIC;
    profile.ran = true;
    std::vector<uint64_t> costs(req.windows.size(), 0);
    auto phase_start = Clock::now();
    if (weighted) {
        for (uint64_t i = 0; i < req.windows.size(); ++i) {
            const auto window = req.windows[i];
            costs[i] = EstimateWindowCost(
                *req.datacell,
                *req.computer,
                window,
                req.metadata_filter.RequiresFullTermScan(req.route, window, req.window_size));
        }
    }
    if (options.profile) {
        profile.cost_estimation_us = weighted ? elapsed_us(phase_start) : 0;
    }
    phase_start = Clock::now();
    auto bounds = Partition(costs, count, weighted);
    std::vector<SindiWindowHeap> heaps;
    heaps.reserve(count);
    auto ranges = std::make_unique<AtomicRange[]>(deque_mode ? 0 : count);
    std::vector<std::unique_ptr<WindowQueue>> queues;
    if (deque_mode) {
        queues.reserve(count);
    }
    for (uint32_t i = 0; i < count; ++i) {
        heaps.emplace_back(req.allocator);
        if (not deque_mode) {
            ranges[i].next.store(bounds[i], std::memory_order_relaxed);
            ranges[i].end = bounds[i + 1];
        }
        if (deque_mode) {
            queues.emplace_back(std::make_unique<WindowQueue>());
            for (auto pos = bounds[i]; pos < bounds[i + 1]; ++pos) {
                queues.back()->windows.push_back(req.windows[pos]);
            }
        }
    }
    if (options.profile) {
        profile.boundaries = bounds;
        profile.initial_costs.resize(count, 0);
        profile.workers.resize(count);
        for (uint32_t i = 0; i < count; ++i) {
            profile.initial_costs[i] = std::accumulate(
                costs.begin() + bounds[i], costs.begin() + bounds[i + 1], uint64_t{0});
        }
        profile.partition_us = elapsed_us(phase_start);
    }
    const auto dispatch_start = Clock::now();
    auto run = [&](uint32_t me) {
        const auto start = Clock::now();
        SindiWindowWorkerProfile local;
        if (options.profile) {
            local.start_delay_us =
                std::chrono::duration<double, std::micro>(start - dispatch_start).count();
        }
        auto computer = std::make_shared<SparseTermComputer>(*req.computer);
        Vector<float> dists(req.window_size, 0.0F, req.allocator);
        SindiQueryContext context(req.allocator);
        auto param = req.search;
        param.ef = req.worker_candidates;
        uint32_t range_offset = 0;
        uint64_t claimed_pos = 0;
        auto pop = [&](uint32_t victim, uint32_t& window) {
            if (options.profile and victim != me) {
                ++local.steal_attempts;
            }
            if (not deque_mode) {
                const auto pos = ranges[victim].next.fetch_add(1, std::memory_order_relaxed);
                if (pos >= ranges[victim].end) {
                    return false;
                }
                claimed_pos = pos;
                window = req.windows[pos];
                return true;
            }
            auto& queue = *queues[victim];
            std::unique_lock<std::mutex> lock(queue.mutex, std::defer_lock);
            if (victim == me) {
                lock.lock();
            } else if (not lock.try_lock()) {
                return false;
            }
            if (queue.windows.empty()) {
                return false;
            }
            if (victim == me) {
                window = queue.windows.front();
                queue.windows.pop_front();
            } else {
                window = queue.windows.back();
                queue.windows.pop_back();
            }
            return true;
        };
        auto take = [&](uint32_t& window, uint32_t& victim) {
            if (deque_mode) {
                for (uint32_t offset = 0; offset < count; ++offset) {
                    victim = (me + offset) % count;
                    if (pop(victim, window)) {
                        return true;
                    }
                }
            } else {
                while (range_offset < count) {
                    victim = (me + range_offset) % count;
                    if (pop(victim, window)) {
                        return true;
                    }
                    ++range_offset;
                }
            }
            return false;
        };
        uint32_t window = 0;
        uint32_t victim = 0;
        while (take(window, victim)) {
            if (options.profile) {
                ++local.windows;
                local.stolen_windows += victim != me;
                if (weighted) {
                    local.estimated_cost += costs[claimed_pos];
                }
            }
            const auto begin_id = window * req.window_size;
            computer->SetTermPruneEnabled(
                not req.metadata_filter.RequiresFullTermScan(req.route, window, req.window_size));
            req.datacell->QueryWindow(dists.data(), window, computer, req.use_term_lists, context);
            if (req.use_term_lists) {
                req.datacell->InsertHeapByWindow(dists.data(),
                                                 window,
                                                 computer,
                                                 heaps[me],
                                                 param,
                                                 begin_id,
                                                 KNN_SEARCH,
                                                 false,
                                                 context);
            } else {
                const auto docs =
                    static_cast<uint32_t>(std::min(req.window_size, req.element_count - begin_id));
                req.datacell->InsertHeapByDists(
                    dists.data(), docs, heaps[me], param, begin_id, KNN_SEARCH, false);
            }
        }
        if (req.statistics != nullptr and context.has_untracked_approximate_evaluations) {
            req.statistics->complete.store(false, std::memory_order_relaxed);
        }
        if (options.profile) {
            local.elapsed_us = elapsed_us(start);
            profile.workers[me] = local;
        }
    };
    run_query_workers(count, req.pool, run, false);
    if (options.profile) {
        profile.dispatch_join_us = elapsed_us(dispatch_start);
    }
    phase_start = Clock::now();
    profile.bulk_merge_used = MergeHeaps(heaps, heap, req.search.ef, req.allocator);
    if (options.profile) {
        profile.merge_us = elapsed_us(phase_start);
    }
}

std::string
SindiSearcher::DumpStatistics(const SearchStatistics& statistics,
                              const SindiSearchOptions& options,
                              const SindiWindowSearchProfile& profile) {
    if (not options.profile) {
        return statistics.Dump();
    }
    auto json = nlohmann::json::parse(statistics.Dump());
    auto& out = json["sindi_scheduler"];
    out = {{"requested", SchedulerName(options.scheduler)},
           {"used", profile.ran ? SchedulerName(options.scheduler) : "serial"},
           {"cost_estimation_us", profile.cost_estimation_us},
           {"partition_us", profile.partition_us},
           {"dispatch_join_us", profile.dispatch_join_us},
           {"merge_us", profile.merge_us},
           {"bulk_merge_used", profile.bulk_merge_used},
           {"boundaries", profile.boundaries},
           {"initial_costs", profile.initial_costs},
           {"workers", nlohmann::json::array()}};
    for (uint64_t i = 0; i < profile.workers.size(); ++i) {
        const auto& worker = profile.workers[i];
        out["workers"].push_back({{"worker", i},
                                  {"windows", worker.windows},
                                  {"stolen_windows", worker.stolen_windows},
                                  {"steal_attempts", worker.steal_attempts},
                                  {"estimated_cost", worker.estimated_cost},
                                  {"start_delay_us", worker.start_delay_us},
                                  {"elapsed_us", worker.elapsed_us}});
    }
    return json.dump();
}

}  // namespace vsag
