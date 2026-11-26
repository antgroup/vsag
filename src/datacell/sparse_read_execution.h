// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <functional>

namespace vsag {
class SearchExecutionContext;

// Invocation-owned state. Resources are borrowed until all read tasks have drained.
// The caller owns the global range plan and disjoint destination buffers.
struct SparseReadExecution {
    SearchExecutionContext* resources{nullptr};
    uint32_t requested{1};
    bool profile_search{false};
    bool profile_cpu{false};
    uint32_t granted{1};
    uint64_t tasks{0};
    uint64_t admission_ns{0};
    uint64_t wait_ns{0};
    uint64_t read_sum_ns{0};
    uint64_t worker_cpu_ns{0};
    bool worker_cpu_available{true};
};

// Cheap eligibility check, before constructing a read callback on the single-batch path.
bool
CanParallelizeSparseReads(const SparseReadExecution& execution, uint64_t count);

// Preserve globally merged ranges. Parallelism only partitions the range list;
// it never rebuilds candidate-local plans or changes distance evaluation order.
void
ReadSparseRanges(SparseReadExecution& execution,
                 uint64_t count,
                 const std::function<void(uint64_t, uint64_t)>& read);
}  // namespace vsag
