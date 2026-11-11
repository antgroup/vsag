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
#include <cstdint>

/// Sizing for the CUDA backend, free of CUDA so it is testable without a device.
/// Each entry point asks for a plan and refuses what it says does not fit.
///
/// Each Plan* orders `k` and `dim` as the entry point it serves does: k first
/// for PlanAssign, dim first for PlanSeed and PlanAccumulate. Both are small
/// integers, so check rather than assume. WorthOffloading serves all three and
/// takes k first.

namespace vsag::gpu {

/// Below this many row * centroid * dimension products, launch and transfer cost
/// more than the device saves.
///
/// Measured against the host path over k in 64..8192 at dim 128, with
/// n = max(65536, 64 * k) as IVF samples: the device lost at k = 256 (0.72x) and
/// won at k = 512 (1.25x). This is set to the winning configuration's own work
/// product, 65536 * 512 * 128, which is exactly 2^32, so nothing smaller than a
/// size the sweep measured a win at is offloaded. Where the ratio actually
/// crosses 1.0 is somewhere between those two and was not measured. Both points
/// are pinned in gpu_plan_test.cpp. The crossover moves with the host's
/// CPU-to-GPU ratio, which is what `gpu_min_work_threshold` overrides.
constexpr uint64_t kMinWorkForGpu = 1ULL << 32;

/// Centroid slice size when the full set cannot stay resident.
constexpr uint64_t kStreamedCentroidChunk = 8192;

/// Streams the assignment pass alternates chunks between, so a transfer overlaps
/// the previous chunk's compute. Every per-chunk buffer is allocated once per
/// stream, so the budget divides by this too; the sizing and the chunk loop must
/// agree on it.
///
/// The overlap pays for that halving: on 400000 rows of 128 into 4096 centroids,
/// the reference shape for the timings in cuda_raii.h too, two streams took
/// 0.097 s against 0.114 s on one, which had twice the chunk size out of the same
/// budget. Every timing quoted for this shape is one run, so the tenths of a
/// millisecond between them carry nothing.
constexpr uint64_t kAssignStreams = 2;

/// Threads per warp. `warpSize` is a device-only builtin, so host code sizing a
/// launch needs its own name for it, and one name keeps the kernels agreeing.
constexpr int32_t kWarpWidth = 32;

/// Tiles the seeding pass splits the dataset into: a draw walks the tile sums
/// plus the one tile it lands in.
constexpr int32_t kInitBlocks = 1024;

/// Shared memory ceiling for the seeding draw, kept clear of the 48 KiB limit.
/// This counts the draw's own tiles; the 4 KiB left over is what covers the few
/// bytes kpp_select declares statically. Staging one tile there also caps the dataset at about 11 million rows; above
/// that seeding refuses and the host takes it, which needs more than 170k
/// buckets to reach.
constexpr uint64_t kMaxInitShared = 44U << 10;

/// Working-set ceiling for the chunked paths. A chunk-size sweep was fastest near
/// 1.1 GiB and slower at 10.5 GiB, so a larger share of the card buys nothing.
constexpr uint64_t kChunkedBudgetCap = 2ULL << 30;

/// Which ceiling applies: the caller's when it named one, kChunkedBudgetCap
/// otherwise. Both callers of the chunked passes ask this, so the rule is here
/// rather than in each of them.
inline uint64_t
EffectiveChunkedCap(uint64_t requested) {
    return requested > 0 ? requested : kChunkedBudgetCap;
}

/// Smallest chunk worth issuing. A budget that cannot hold this many rows, or the
/// whole input when it is smaller, is refused rather than overspent.
constexpr uint64_t kMinChunkRows = 1024;

/// True when a problem of this shape is worth the fixed cost of offloading;
/// `min_work` of 0 selects kMinWorkForGpu.
///
/// Every pass asks this, so the knob turns the backend off for a build rather
/// than for one pass. Callers ask it before binding a device, because binding
/// creates a context and takes memory on the card.
inline bool
WorthOffloading(uint64_t rows, uint64_t k, int32_t dim, uint64_t min_work) {
    if (rows == 0 or k == 0 or dim <= 0) {
        return false;
    }
    const uint64_t floor = min_work > 0 ? min_work : kMinWorkForGpu;
    // `rows * k >= floor / dim` says the same thing and reads better, but the
    // product can wrap for a large enough corpus and centroid count. Comparing
    // rows against the ceiling of the quotient is the same predicate without a
    // product: `floor / dim` is at most the threshold itself, which FromJson caps
    // at INT64_MAX, and `k` at INT32_MAX, so the sum below has room.
    const uint64_t per_row = floor / static_cast<uint64_t>(dim);
    // A dim above the threshold leaves per_row 0 and the comparison always true.
    // That is the answer and not a gap in it: one row and one centroid already do
    // `dim` work, which is more than the threshold asks for. Said here because the
    // integer division is what produces it.
    return rows >= (per_row + k - 1) / k;
}

/// How one nearest-centroid pass gets cut up.
struct AssignPlan {
    bool offload{false};  ///< false: the caller must use the CPU path
    bool centroids_resident{false};
    uint64_t centroid_slots{0};  ///< streamed centroids need one slice per stream
    uint64_t k_chunk{0};
    uint64_t n_chunk{0};
    uint64_t centroid_bytes{0};  ///< centroid slices and their row norms
};

/// Device bytes the assignment pass holds per query row, across both streams: the
/// row, its distance to every centroid in the slice, and the best distance with
/// its centroid and the row's squared norm.
///
/// Each stream also pins a host mirror of the row plus its two results, not added
/// up here because it cannot exceed what is: `dim + 2` floats against the
/// `dim + k_chunk + 3` below. So the one budget bounds the pinned host memory too.
///
/// Named, not written out at each use, so a buffer added to the pass cannot be
/// counted in one place and missed in another.
inline uint64_t
AssignBytesPerRow(uint64_t k_chunk, int32_t dim) {
    const uint64_t per_stream = (static_cast<uint64_t>(dim) + k_chunk) * sizeof(float) +
                                2 * sizeof(float) + sizeof(int32_t);
    return kAssignStreams * per_stream;
}

inline AssignPlan
PlanAssign(
    uint64_t query_count, uint64_t k, int32_t dim, uint64_t budget_bytes, uint64_t min_work) {
    AssignPlan plan;
    // Only this plan names a budget of 0, because what follows divides by what is
    // left of it. PlanSeed and PlanAccumulate reach the same answer through the
    // check they already make, their resident bytes and their accumulators each
    // exceeding a budget of 0, so a guard in either would be unreachable;
    // gpu_plan_test.cpp pins the outcome for all three rather than the guard.
    if (budget_bytes == 0 or not WorthOffloading(query_count, k, dim, min_work)) {
        return plan;
    }

    const uint64_t cent_bytes = k * static_cast<uint64_t>(dim) * sizeof(float);
    // Resident centroids are shared; streamed ones need a slice per stream so the
    // uploads cannot clobber one another.
    plan.centroids_resident = cent_bytes * 2 < budget_bytes;
    plan.k_chunk = plan.centroids_resident ? k : std::min<uint64_t>(k, kStreamedCentroidChunk);
    plan.centroid_slots = plan.centroids_resident ? 1 : kAssignStreams;
    // A slice carries its rows and one squared norm each, resident for the pass.
    plan.centroid_bytes =
        plan.centroid_slots * plan.k_chunk * (static_cast<uint64_t>(dim) + 1) * sizeof(float);
    if (plan.centroid_bytes >= budget_bytes) {
        return plan;
    }

    const uint64_t remain = budget_bytes - plan.centroid_bytes;
    // Too tight for a chunk worth launching: refuse rather than overspend.
    const uint64_t rows_that_fit = remain / AssignBytesPerRow(plan.k_chunk, dim);
    const uint64_t floor_rows = std::min<uint64_t>(query_count, kMinChunkRows);
    if (rows_that_fit < floor_rows) {
        return plan;
    }
    plan.n_chunk = std::min<uint64_t>(query_count, rows_that_fit);
    plan.offload = true;
    return plan;
}

/// Uniforms the seeding pass indexes for k centroids. Named for the same reason
/// as AssignBytesPerRow: the caller fills the buffer and the pass indexes it, so
/// the size belongs in one place. cuda_backend.h gives which index does what.
inline uint64_t
SeedUniformCount(uint32_t k) {
    return 2ULL * static_cast<uint64_t>(k);
}

/// How the k-means++ seeding pass gets laid out.
struct SeedPlan {
    bool offload{false};
    int32_t blocks{0};
    uint64_t rows_per_block{0};
    uint64_t shared_bytes{0};
    uint64_t device_bytes{0};  ///< dataset, running minima, tile sums and picks
};

inline SeedPlan
PlanSeed(uint64_t count, int32_t dim, uint32_t k, uint64_t budget_bytes, uint64_t min_work) {
    SeedPlan plan;
    if (static_cast<uint64_t>(k) > count or not WorthOffloading(count, k, dim, min_work)) {
        return plan;
    }
    const uint64_t wanted_tiles = std::min<uint64_t>(static_cast<uint64_t>(kInitBlocks), count);
    plan.rows_per_block = (count + wanted_tiles - 1) / wanted_tiles;
    // Deriving the count back from the size keeps every tile non-empty: rounding
    // the size up can push the last tiles of a fixed count past the end.
    plan.blocks = static_cast<int32_t>((count + plan.rows_per_block - 1) / plan.rows_per_block);
    plan.shared_bytes = (static_cast<uint64_t>(plan.blocks) + plan.rows_per_block) * sizeof(float);
    if (plan.shared_bytes > kMaxInitShared) {
        return plan;
    }
    // The dataset stays resident: every one of the k steps reads all of it.
    plan.device_bytes = count * static_cast<uint64_t>(dim) * sizeof(float) + count * sizeof(float) +
                        static_cast<uint64_t>(plan.blocks) * sizeof(float) +
                        static_cast<uint64_t>(k) * sizeof(uint64_t);
    if (plan.device_bytes > budget_bytes) {
        return plan;
    }
    plan.offload = true;
    return plan;
}

/// How the centroid accumulation pass gets cut up.
struct AccumulatePlan {
    bool offload{false};
    uint64_t n_chunk{0};
    uint64_t accumulator_bytes{0};
};

/// Device bytes the accumulation pass holds per point: the point itself and its
/// label. Named for the same reason as AssignBytesPerRow.
inline uint64_t
AccumulateBytesPerRow(int32_t dim) {
    return static_cast<uint64_t>(dim) * sizeof(float) + sizeof(int32_t);
}

inline AccumulatePlan
PlanAccumulate(uint64_t count, int32_t dim, uint32_t k, uint64_t budget_bytes, uint64_t min_work) {
    AccumulatePlan plan;
    if (not WorthOffloading(count, k, dim, min_work)) {
        return plan;
    }
    const uint64_t sum_bytes =
        static_cast<uint64_t>(k) * static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t cnt_bytes = static_cast<uint64_t>(k) * sizeof(int32_t);
    plan.accumulator_bytes = sum_bytes + cnt_bytes;
    if (plan.accumulator_bytes >= budget_bytes) {
        return plan;
    }
    const uint64_t remain = budget_bytes - plan.accumulator_bytes;
    const uint64_t rows_that_fit = remain / AccumulateBytesPerRow(dim);
    const uint64_t floor_rows = std::min<uint64_t>(count, kMinChunkRows);
    if (rows_that_fit < floor_rows) {
        return plan;
    }
    plan.n_chunk = std::min<uint64_t>(count, rows_that_fit);
    plan.offload = true;
    return plan;
}

/// Share of free device memory a build may take, leaving the rest to allocator
/// fragmentation, context growth and anything else on the card. 0 means no cap.
inline uint64_t
CapBudget(uint64_t free_bytes, uint64_t cap_bytes) {
    const uint64_t usable = static_cast<uint64_t>(static_cast<double>(free_bytes) * 0.8);
    return cap_bytes > 0 ? std::min<uint64_t>(usable, cap_bytes) : usable;
}

}  // namespace vsag::gpu
