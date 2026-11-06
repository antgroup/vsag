// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sindi_parallel_search.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>

#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/search_execution_context.h"
#include "sindi_cpu_timer.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

uint64_t
resource_limit(const char* name, uint64_t fallback) {
    const auto* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    uint64_t parsed = 0;
    const auto* end = value + std::strlen(value);
    const auto result = std::from_chars(value, end, parsed);
    CHECK_ARGUMENT(result.ec == std::errc{} and result.ptr == end,
                   std::string("invalid ") + name + ": expected unsigned decimal digits");
    return parsed;
}

struct Reservation {
    explicit Reservation(SearchExecutionContext& resources) : owner(resources) {
        std::lock_guard<std::mutex> guard(owner.mutex);
        ++owner.callers;
    }
    ~Reservation() {
        std::lock_guard<std::mutex> guard(owner.mutex);
        owner.busy -= extra;
        --owner.callers;
    }
    SearchExecutionContext& owner;
    uint32_t extra{0};
};

class BudgetedFilter final : public Filter {
public:
    BudgetedFilter(FilterPtr delegate, SindiParallelState& state)
        : delegate_(std::move(delegate)), state_(state) {
    }
    bool
    CheckValid(int64_t id) const override {
        // Each evaluator is slot-exclusive. Poll the clock at most once per 32 callbacks;
        // stop propagation and callback permits remain exact on every callback.
        const bool stopped = (++checks_ % 32 == 0) ? state_.Stopped()
                                                   : state_.stop.load(std::memory_order_relaxed) !=
                                                         SindiParallelState::Stop::NONE;
        if (stopped or not state_.ReserveCallback()) {
            return false;
        }
        const auto allowed = delegate_->CheckValid(id);
        // Keep the last permitted result, then stop even if no later score
        // would cause another filter callback to be attempted.
        if (state_.callback_limit != 0 and
            state_.callbacks.load(std::memory_order_relaxed) >= state_.callback_limit) {
            state_.RequestStop(SindiParallelState::Stop::CALLBACK_LIMIT);
        }
        return allowed;
    }

private:
    FilterPtr delegate_;
    SindiParallelState& state_;
    mutable uint32_t checks_{0};
};

struct Partition {
    Partition(uint64_t begin, uint64_t end, uint64_t capacity, Allocator* allocator)
        : cursor(begin),
          end(end),
          capacity(capacity),
          heap(CompareByFirst{true}, Vector<std::pair<float, InnerIdType>>(allocator)) {
    }
    uint64_t cursor;
    uint64_t end;
    uint64_t capacity;
    MaxHeap heap;
};

struct Slot {
    explicit Slot(const SindiPartitionSearchRequest& request, FilterPtr filter)
        : allocator(SafeAllocator::FactoryDefaultAllocator()),
          computer(std::make_shared<SparseTermComputer>(*request.computer, allocator.get())),
          dists(request.window_size, 0.0F, allocator.get()),
          context(allocator.get()),
          filter(std::move(filter)) {
    }
    std::shared_ptr<Allocator> allocator;
    SparseTermComputerPtr computer;
    Vector<float> dists;
    SindiQueryContext context;
    FilterPtr filter;
    SindiCpuTime cpu;
};

}  // namespace

std::string
SindiExecutionStatistics(JsonType json,
                         const SindiParallelState* parallel,
                         const SindiCpuTime& caller_cpu,
                         bool profile_cpu,
                         uint32_t requested_slots) {
    if (not parallel and not profile_cpu and requested_slots <= 1) {
        return json.Dump();
    }
    if (parallel) {
        json["sindi_strategy"].SetString("accel_v1");
        json["sindi_requested_slots"].SetUint64(parallel->requested);
        json["sindi_resources_ready"].SetBool(parallel->resources != nullptr);
        json["sindi_granted_slots"].SetUint64(parallel->granted);
        json["sindi_filter_slots"].SetUint64(parallel->filter_slots);
        json["sindi_cpu_slots"].SetUint64(parallel->resources ? parallel->resources->slots : 0);
        json["sindi_worker_limit"].SetUint64(parallel->resources ? parallel->resources->limit : 0);
        json["sindi_partitions"].SetUint64(parallel->plan.quotas.size());
        json["sindi_candidate_capacity"].SetUint64(parallel->plan.capacity);
        json["sindi_n_candidate"].SetUint64(parallel->plan.candidates);
        json["sindi_windows_processed"].SetUint64(parallel->completed_windows.load());
        json["sindi_execution_complete"].SetBool(parallel->stop.load() ==
                                                 SindiParallelState::Stop::NONE);
        switch (parallel->stop.load()) {
            case SindiParallelState::Stop::NONE:
                json["sindi_stop_reason"].SetString("none");
                break;
            case SindiParallelState::Stop::CALLBACK_LIMIT:
                json["sindi_stop_reason"].SetString("callback_limit");
                break;
            case SindiParallelState::Stop::DEADLINE:
                json["sindi_stop_reason"].SetString("deadline");
                break;
            case SindiParallelState::Stop::ERROR:
                json["sindi_stop_reason"].SetString("error");
                break;
        }
    } else {
        json["sindi_strategy"].SetString("serial");
        if (requested_slots > 1) {
            json["sindi_requested_slots"].SetUint64(requested_slots);
            json["sindi_granted_slots"].SetUint64(1);
            json["sindi_resources_ready"].SetBool(false);
            json["sindi_worker_limit"].SetUint64(0);
        }
    }
    const bool cpu_complete =
        profile_cpu and caller_cpu.available and (not parallel or parallel->worker_cpu_available);
    json["sindi_cpu_time_enabled"].SetBool(profile_cpu);
    json["sindi_cpu_time_complete"].SetBool(cpu_complete);
    if (cpu_complete) {
        const auto worker_ns = parallel ? parallel->worker_cpu_ns : 0;
        json["sindi_search_cpu_us"].SetDouble(double(caller_cpu.nanoseconds + worker_ns) / 1000);
        json["sindi_worker_cpu_us"].SetDouble(double(worker_ns) / 1000);
    }
    return json.Dump();
}

SindiParallelPlan
SindiParallelPlan::Make(uint64_t windows, uint64_t candidates, double ratio) {
    uint64_t ratio_bits = 0;
    std::memcpy(&ratio_bits, &ratio, sizeof(ratio_bits));
    CHECK_ARGUMENT(candidates > 0 and ratio >= 1.0 and
                       (ratio_bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL,
                   "invalid parallel candidate capacity");
    const auto maximum = resource_limit("VSAG_SINDI_MAX_CANDIDATES", 1048576);
    CHECK_ARGUMENT(maximum > 0 and maximum <= std::numeric_limits<uint32_t>::max(),
                   "VSAG_SINDI_MAX_CANDIDATES must fit a positive uint32");
    CHECK_ARGUMENT(candidates <= maximum and ratio <= double(maximum) / double(candidates),
                   "parallel candidate capacity exceeds deployment limit");
    SindiParallelPlan plan;
    plan.candidates = candidates;
    plan.capacity = static_cast<uint64_t>(std::ceil(ratio * double(candidates)));
    CHECK_ARGUMENT(plan.capacity <= maximum, "parallel candidate capacity exceeds limit");
    const auto groups = std::min({uint64_t{16}, windows, candidates});
    plan.boundaries.push_back(0);
    for (uint64_t i = 0; i < groups; ++i) {
        plan.boundaries.push_back(windows / groups * (i + 1) +
                                  (windows % groups) * (i + 1) / groups);
        plan.quotas.push_back(plan.capacity / groups + (i < plan.capacity % groups));
    }
    return plan;
}

void
SindiParallelState::RequestStop(Stop reason) {
    auto expected = Stop::NONE;
    stop.compare_exchange_strong(expected, reason, std::memory_order_relaxed);
}

bool
SindiParallelState::Stopped() {
    if (deadline != Clock::time_point::max() and Clock::now() >= deadline) {
        RequestStop(Stop::DEADLINE);
        // A deadline must remain visible even after callback budget exhaustion.
        return true;
    }
    return stop.load(std::memory_order_relaxed) != Stop::NONE;
}

bool
SindiParallelState::ReserveCallback() {
    if (callback_limit == 0) {
        return true;
    }
    auto used = callbacks.load(std::memory_order_relaxed);
    while (used < callback_limit) {
        if (callbacks.compare_exchange_weak(used, used + 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    RequestStop(Stop::CALLBACK_LIMIT);
    return false;
}

void
SearchSindiPartitions(const SindiPartitionSearchRequest& request,
                      SindiParallelState& state,
                      MaxHeap& output) {
    state.plan = SindiParallelPlan::Make(request.windows.size(), request.search.ef, state.ratio);
    const uint64_t groups = state.plan.quotas.size();
    if (groups == 0) {
        return;
    }
    CHECK_ARGUMENT(state.resources != nullptr,
                   "SINDI partition executor requires prepared resources");
    auto& shared = *state.resources;
    Reservation reservation(shared);
    auto maximum = static_cast<uint32_t>(
        std::min({uint64_t(state.requested), groups, uint64_t(state.execution_limit)}));
    auto& lease = state.lease;
    const auto* provider =
        state.user_filter ? dynamic_cast<const ConcurrentFilterProvider*>(state.user_filter.get())
                          : nullptr;
    CHECK_ARGUMENT(not state.user_filter or provider != nullptr,
                   "SINDI parallel search requires independent filters");
    for (;;) {
        if (state.user_filter) {
            lease = provider->TryLease(maximum);
            CHECK_ARGUMENT(lease != nullptr and not lease->evaluators.empty() and
                               lease->evaluators.size() <= maximum,
                           "SINDI parallel filter resources unavailable");
            maximum = static_cast<uint32_t>(lease->evaluators.size());
            if (state.filter_slots == 0)
                state.filter_slots = maximum;
        }
        reservation.extra = shared.TryReserve(maximum - 1);
        state.granted = reservation.extra + 1;
        if (state.granted == maximum || not lease)
            break;
        maximum = state.granted;
        {
            std::lock_guard<std::mutex> guard(shared.mutex);
            shared.busy -= reservation.extra;
            reservation.extra = 0;
        }
        lease.reset();
    }
    std::vector<std::unique_ptr<Slot>> slots;
    for (uint32_t i = 0; i < state.granted; ++i) {
        FilterPtr filter;
        if (lease) {
            CHECK_ARGUMENT(lease->evaluators[i] != nullptr, "null parallel evaluator");
            filter = std::make_shared<BudgetedFilter>(lease->evaluators[i], state);
        }
        slots.emplace_back(std::make_unique<Slot>(request, state.map_filter(filter)));
    }
    auto heap_allocator = SafeAllocator::FactoryDefaultAllocator();
    std::vector<Partition> partitions;
    std::deque<uint32_t> ready;
    for (uint32_t i = 0; i < groups; ++i) {
        partitions.emplace_back(state.plan.boundaries[i],
                                state.plan.boundaries[i + 1],
                                state.plan.quotas[i],
                                heap_allocator.get());
        ready.push_back(i);
    }
    std::mutex queue_mutex;
    auto run = [&](uint32_t id) {
        // Slot zero is already included in the calling thread's CPU timer.
        SindiCpuTimer cpu_timer(state.profile_cpu and id != 0, slots[id]->cpu);
        try {
            auto& slot = *slots[id];
            while (not state.Stopped()) {
                uint32_t shard;
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (ready.empty()) {
                        break;
                    }
                    shard = ready.front();
                    ready.pop_front();
                }
                auto& part = partitions[shard];
                auto param = request.search;
                param.ef = part.capacity;
                param.is_inner_id_allowed = slot.filter;
                const auto end = std::min(part.end, part.cursor + 4);
                while (part.cursor < end and not state.Stopped()) {
                    const auto window = request.windows[part.cursor];
                    const auto offset = window * request.window_size;
                    slot.computer->SetTermPruneEnabled(
                        not request.metadata_filter.RequiresFullTermScan(
                            request.route, window, request.window_size));
                    request.datacell->QueryWindow(slot.dists.data(),
                                                  window,
                                                  slot.computer,
                                                  request.use_term_lists,
                                                  slot.context);
                    if (request.use_term_lists) {
                        request.datacell->InsertHeapByWindow(slot.dists.data(),
                                                             window,
                                                             slot.computer,
                                                             part.heap,
                                                             param,
                                                             offset,
                                                             KNN_SEARCH,
                                                             slot.filter != nullptr,
                                                             slot.context);
                    } else {
                        const auto docs = static_cast<uint32_t>(
                            std::min(request.window_size, request.element_count - offset));
                        request.datacell->InsertHeapByDists(slot.dists.data(),
                                                            docs,
                                                            part.heap,
                                                            param,
                                                            offset,
                                                            KNN_SEARCH,
                                                            slot.filter != nullptr);
                    }
                    ++part.cursor;
                    state.completed_windows.fetch_add(1, std::memory_order_relaxed);
                }
                if (part.cursor < part.end and not state.Stopped()) {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    ready.push_back(shard);
                }
            }
        } catch (...) {
            state.RequestStop(SindiParallelState::Stop::ERROR);
            throw;
        }
    };
    std::vector<std::future<void>> futures;
    futures.reserve(state.granted - 1);
    std::exception_ptr failure;
    try {
        for (uint32_t i = 1; i < state.granted; ++i) {
            futures.push_back(shared.pool->GeneralEnqueue(run, i));
        }
        run(0);
    } catch (...) {
        state.RequestStop(SindiParallelState::Stop::ERROR);
        failure = std::current_exception();
    }
    for (auto& future : futures) {
        try {
            future.get();
        } catch (...) {
            if (failure == nullptr) {
                failure = std::current_exception();
            }
        }
    }
    if (failure) {
        std::rethrow_exception(failure);
    }
    for (uint32_t i = 0; i < state.granted; ++i) {
        if (request.statistics != nullptr and
            slots[i]->context.has_untracked_approximate_evaluations) {
            request.statistics->complete.store(false, std::memory_order_relaxed);
        }
        if (state.profile_cpu and i != 0) {
            state.worker_cpu_available = state.worker_cpu_available and slots[i]->cpu.available;
            state.worker_cpu_ns += slots[i]->cpu.nanoseconds;
        }
    }
    if (SindiParallelState::Clock::now() >= state.deadline) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "SINDI parallel search deadline exceeded after drain");
    }
    SindiCandidateHeap merged(CompareByFirst{true}, Vector<SindiCandidate>(request.allocator));
    for (auto& part : partitions) {
        while (not part.heap.empty()) {
            const auto candidate = part.heap.top();
            SindiMergeCandidate(merged, candidate, request.search.ef);
            part.heap.pop();
        }
    }
    output = std::move(merged);
}

}  // namespace vsag
