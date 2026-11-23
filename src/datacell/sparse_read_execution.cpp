// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "sparse_read_execution.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <future>
#include <vector>

#include "impl/searcher/search_execution_context.h"
#include "impl/searcher/sindi_cpu_timer.h"
#include "sparse_query_profile.h"
#if HAVE_LIBAIO
#include "io/async_io/aio_context.h"
#endif

namespace vsag {
bool
CanParallelizeSparseReads(const SparseReadExecution& execution, uint64_t count) {
    if (execution.requested <= 1 or count == 0) {
        return false;
    }
#if HAVE_LIBAIO
    return count > IOContext::DEFAULT_REQUEST_COUNT and execution.resources != nullptr and
           execution.resources->pool and not execution.resources->pool->RequiresInlineExecution();
#else
    return false;
#endif
}

void
ReadSparseRanges(SparseReadExecution& execution,
                 uint64_t count,
                 const std::function<void(uint64_t, uint64_t)>& read) {
    execution.granted = 1;
    if (not CanParallelizeSparseReads(execution, count)) {
        execution.tasks = count != 0 ? 1U : 0U;
        if (count != 0) {
            QueryProfileTimer timer(execution.profile_search ? &execution.read_sum_ns : nullptr);
            read(0, count);
        }
        return;
    }
    uint32_t maximum = 1;
#if HAVE_LIBAIO
    const auto batches = count / IOContext::DEFAULT_REQUEST_COUNT +
                         static_cast<uint64_t>(count % IOContext::DEFAULT_REQUEST_COUNT != 0);
    maximum = static_cast<uint32_t>(std::min<uint64_t>(execution.requested, batches));
#endif
    auto* resources = execution.resources;
    SearchExecutionReservation reservation(*resources);
    {
        QueryProfileTimer timer(execution.profile_search ? &execution.admission_ns : nullptr);
        reservation.extra = resources->TryReserve(maximum - 1);
    }
    const auto slots = reservation.extra + 1;
    execution.granted = slots;
    execution.tasks = slots;
    std::vector<uint64_t> read_times(slots, 0);
    std::vector<SindiCpuTime> cpu(slots);
    std::atomic<bool> failed{false};
    const auto run = [&](uint32_t slot) {
        SindiCpuTimer cpu_timer(execution.profile_cpu and slot != 0, cpu[slot]);
        try {
            if (failed.load(std::memory_order_relaxed)) {
                return;
            }
            const auto first = (count / slots) * slot + std::min<uint64_t>(count % slots, slot);
            const auto length = count / slots + static_cast<uint64_t>(slot < count % slots);
            QueryProfileTimer timer(execution.profile_search ? &read_times[slot] : nullptr);
            read(first, length);
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            throw;
        }
    };
    std::vector<std::future<void>> futures;
    futures.reserve(slots - 1);
    std::exception_ptr failure;
    try {
        for (uint32_t slot = 1; slot < slots; ++slot) {
            futures.push_back(resources->pool->GeneralEnqueue(run, slot));
        }
        run(0);
    } catch (...) {
        failure = std::current_exception();
        failed.store(true, std::memory_order_relaxed);
    }
    {
        QueryProfileTimer timer(execution.profile_search ? &execution.wait_ns : nullptr);
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (failure == nullptr) {
                    failure = std::current_exception();
                }
            }
        }
    }
    for (uint32_t slot = 0; slot < slots; ++slot) {
        execution.read_sum_ns += read_times[slot];
        if (execution.profile_cpu and slot != 0) {
            execution.worker_cpu_available &= cpu[slot].available;
            execution.worker_cpu_ns += cpu[slot].nanoseconds;
        }
    }
    if (failure != nullptr) {
        std::rethrow_exception(failure);
    }
}
}  // namespace vsag
