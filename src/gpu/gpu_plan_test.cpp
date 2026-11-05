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

#include "gpu_plan.h"

#include "unittest.h"

// The sizing decisions carry no CUDA, so they are covered on any machine. What
// the kernels do with a plan needs a device and is not tested here.

using namespace vsag::gpu;

namespace {
constexpr uint64_t kGiB = 1ULL << 30;

/// Every byte a pass holds on the device for the whole of it. The per-row part
/// comes from the plan's own definition, so a buffer added to the pass cannot be
/// counted there and missed here.
uint64_t
AssignWorkingSet(const AssignPlan& plan, int32_t dim) {
    return plan.centroid_bytes + plan.n_chunk * AssignBytesPerRow(plan.k_chunk, dim);
}
}  // namespace

TEST_CASE("PlanAssign refuses degenerate input", "[ut][gpu_plan]") {
    const uint64_t big = 1ULL << 20;
    REQUIRE_FALSE(PlanAssign(0, 4096, 128, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(big, 0, 128, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(big, 4096, 0, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanAssign(big, 4096, -1, 2 * kGiB, 0).offload);
    // A zero budget is what CudaSuggestedBudget reports with no usable device.
    REQUIRE_FALSE(PlanAssign(big, 4096, 128, 0, 0).offload);
}

TEST_CASE("PlanAssign keeps small problems on the CPU", "[ut][gpu_plan]") {
    // The floor is on rows * k * dim, so a small problem is refused however the
    // smallness is distributed.
    REQUIRE_FALSE(PlanAssign(1024, 16, 128, 2 * kGiB, 0).offload);
    REQUIRE(PlanAssign(262144, 4096, 128, 2 * kGiB, 0).offload);

    // Raising the threshold pushes a problem back to the CPU, lowering it pulls
    // one onto the device. Both directions matter: the first is how a user opts
    // out, the second is how they opt in on faster hardware.
    const uint64_t work = 262144ULL * 4096ULL * 128ULL;
    REQUIRE_FALSE(PlanAssign(262144, 4096, 128, 2 * kGiB, work * 2).offload);
    REQUIRE(PlanAssign(1024, 16, 128, 2 * kGiB, 1).offload);
}

TEST_CASE("The floor matches the measured crossover", "[ut][gpu_plan]") {
    // Pins the two configurations the calibration sweep straddles, so a later
    // edit to kMinWorkForGpu has to face the measurement. Training sets follow
    // IVF's own n = max(65536, 64 * k).
    //
    // k=256 measured 0.72x (the device lost by 28%) and must stay on the CPU.
    REQUIRE_FALSE(WorthOffloading(65536, 256, 128, 0));
    // k=512 measured 1.25x and must reach the device. Its work product is exactly
    // kMinWorkForGpu, so this sits on the boundary: raising the constant at all
    // fails here, and lowering it fails the assertion above once it passes k=256.
    REQUIRE(WorthOffloading(65536, 512, 128, 0));
}

TEST_CASE("Each plan reads k and dim in its entry point's order", "[ut][gpu_plan]") {
    // k and dim are both small integers, so a swapped pair compiles. What keeps
    // the call sites honest is that each plan takes them in the order its entry
    // point does; these three assertions fail if any of the orders is changed.
    //
    // None of the swaps is caught by a refusal: every one of them still fits its
    // budget and reports offload. What catches them is that the plan then
    // describes a different problem, so each case pins a number that moves.
    //
    // PlanAssign takes k first. Read the other way round the call describes 128
    // centroids at 4096 dimensions, which stays resident just the same, but
    // slices the centroids 128 wide instead of 4096.
    const auto assign = PlanAssign(262144, 4096, 128, 2 * kGiB, 1);
    REQUIRE(assign.offload);
    REQUIRE(assign.centroids_resident);
    REQUIRE(assign.k_chunk == 4096);

    // Seeding and accumulation take dim first. 1024 centroids over 262144 rows of
    // 128 dimensions is what fits here; swapped, the resident set is the same
    // rows at 1024 dimensions, eight times larger, which 19 GiB still holds. The
    // equality below is what refuses it.
    const auto seed = PlanSeed(262144, 128, 1024, 19 * kGiB, 1);
    REQUIRE(seed.offload);
    REQUIRE(seed.device_bytes == 262144ULL * 128 * sizeof(float) + 262144ULL * sizeof(float) +
                                     static_cast<uint64_t>(seed.blocks) * sizeof(float) +
                                     1024ULL * sizeof(uint64_t));

    const auto accumulate = PlanAccumulate(262144, 128, 1024, 2 * kGiB, 1);
    REQUIRE(accumulate.offload);
    REQUIRE(accumulate.accumulator_bytes ==
            1024ULL * 128 * sizeof(float) + 1024ULL * sizeof(int32_t));
}

TEST_CASE("One threshold governs all three passes", "[ut][gpu_plan]") {
    // A user who turns the backend off for a build expects every pass to honour
    // that, not just the one that happens to read the knob.
    const uint64_t huge = 1ULL << 62;
    REQUIRE_FALSE(PlanAssign(262144, 4096, 128, 2 * kGiB, huge).offload);
    REQUIRE_FALSE(PlanSeed(262144, 128, 4096, 19 * kGiB, huge).offload);
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 4096, 2 * kGiB, huge).offload);
}

TEST_CASE("PlanAssign keeps centroids resident only when they fit", "[ut][gpu_plan]") {
    // 4096 centroids of 128 dimensions is 2 MiB, well inside 2 GiB, so they stay
    // and both streams share the one copy.
    const auto resident = PlanAssign(262144, 4096, 128, 2 * kGiB, 0);
    REQUIRE(resident.offload);
    REQUIRE(resident.centroids_resident);
    REQUIRE(resident.centroid_slots == 1);
    REQUIRE(resident.k_chunk == 4096);

    // 200k centroids of 128 dimensions are 97 MiB, which a 128 MiB budget cannot
    // hold alongside a chunk of distances, so they come a slice at a time and each
    // stream gets its own. The whole working set still has to fit.
    const uint64_t tight = 128ULL << 20;
    const auto streamed = PlanAssign(2560000, 200000, 128, tight, 0);
    REQUIRE(streamed.offload);
    REQUIRE_FALSE(streamed.centroids_resident);
    REQUIRE(streamed.centroid_slots == kAssignStreams);
    REQUIRE(streamed.k_chunk == kStreamedCentroidChunk);
    REQUIRE(AssignWorkingSet(streamed, 128) <= tight);
}

TEST_CASE("PlanAssign working set stays within the budget", "[ut][gpu_plan]") {
    // The whole working set, centroids included, must fit what the plan was
    // given. Counting only the per-row part would miss the slices that stay
    // resident for the entire pass.
    for (const uint64_t budget : {kGiB / 4, kGiB, 2 * kGiB, 8 * kGiB}) {
        for (const uint64_t k : {1024ULL, 4096ULL, 40000ULL}) {
            const auto plan = PlanAssign(2560000, k, 128, budget, 0);
            if (not plan.offload) {
                continue;
            }
            REQUIRE(AssignWorkingSet(plan, 128) <= budget);
            REQUIRE(plan.n_chunk >= std::min<uint64_t>(2560000, kMinChunkRows));
            REQUIRE(plan.n_chunk <= 2560000);
        }
    }
}

TEST_CASE("PlanAssign refuses a budget it cannot honour", "[ut][gpu_plan]") {
    // 4096 centroids of 128 dimensions need 2 MiB resident and every
    // 1024-row chunk another 33 MiB. Under a 1 MiB budget the pass must refuse:
    // a floor that pushes the working set past the budget is a reason to refuse,
    // not to overspend.
    const auto plan = PlanAssign(2560000, 4096, 128, 1ULL << 20, 0);
    REQUIRE_FALSE(plan.offload);
}

TEST_CASE("PlanAssign refuses when the slices alone exceed the budget", "[ut][gpu_plan]") {
    // Two slices of 8192 centroids at 128 dimensions are 8.1 MiB, which an
    // 8 MiB budget cannot hold before a single query row is transferred. The
    // pass refuses rather than allocating past what it was given.
    const auto plan = PlanAssign(2560000, 40000, 128, 8ULL << 20, 0);
    REQUIRE_FALSE(plan.offload);
}

TEST_CASE("PlanAssign never asks for more rows than it has", "[ut][gpu_plan]") {
    // A budget far larger than the problem must still chunk at the row count,
    // not above it.
    const auto plan = PlanAssign(262144, 4096, 128, 64 * kGiB, 0);
    REQUIRE(plan.offload);
    REQUIRE(plan.n_chunk == 262144);
}

TEST_CASE("PlanSeed refuses what cannot stay resident", "[ut][gpu_plan]") {
    // The seeding pass reads the whole dataset once per centroid, so it keeps
    // it resident: 1,048,576 points of 960 dimensions is 3.75 GiB, which does
    // not fit a 2 GiB budget but does fit what a 24 GiB card reports free.
    REQUIRE_FALSE(PlanSeed(1048576, 960, 1024, 2 * kGiB, 0).offload);
    REQUIRE(PlanSeed(1048576, 960, 1024, 19 * kGiB, 0).offload);
}

TEST_CASE("PlanSeed refuses degenerate input", "[ut][gpu_plan]") {
    REQUIRE_FALSE(PlanSeed(0, 128, 8, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanSeed(1024, 0, 8, 2 * kGiB, 0).offload);
    REQUIRE_FALSE(PlanSeed(1024, 128, 0, 2 * kGiB, 0).offload);
    // More centroids than points has no k-means++ answer.
    REQUIRE_FALSE(PlanSeed(64, 128, 128, 2 * kGiB, 0).offload);
}

TEST_CASE("PlanSeed tiles are non-empty and cover the dataset", "[ut][gpu_plan]") {
    // The selection kernel reduces over [begin, end) of one tile. A tile that
    // starts past the end would give an empty, and in unsigned arithmetic an
    // enormous, range. Counts that are not multiples of the tile count are the
    // case that exposes it: IVF reaches 96000 at buckets_count=1500.
    for (const uint64_t count :
         {65536ULL, 96000ULL, 96001ULL, 100003ULL, 262144ULL, 999983ULL, 1048576ULL}) {
        const auto plan = PlanSeed(count, 128, 512, 19 * kGiB, 0);
        REQUIRE(plan.offload);
        REQUIRE(plan.blocks > 0);
        // Every point lands in a tile.
        REQUIRE(static_cast<uint64_t>(plan.blocks) * plan.rows_per_block >= count);
        // No tile begins past the end.
        REQUIRE((static_cast<uint64_t>(plan.blocks) - 1) * plan.rows_per_block < count);
        REQUIRE(plan.shared_bytes <= kMaxInitShared);
    }
    // Fewer points than tiles gives one point per tile rather than empty ones.
    // The threshold has to be lowered to reach the layout at this size.
    const auto tiny = PlanSeed(64, 128, 8, 2 * kGiB, 1);
    REQUIRE(tiny.offload);
    REQUIRE(tiny.blocks == 64);
    REQUIRE(tiny.rows_per_block == 1);
}

TEST_CASE("PlanAccumulate refuses when the accumulators do not fit", "[ut][gpu_plan]") {
    // 40000 centroids of 128 dimensions is 19.5 MiB of sums and 19.7 MiB with
    // their counts; a 16 MiB budget cannot hold that, let alone any points.
    REQUIRE_FALSE(PlanAccumulate(2560000, 128, 40000, 16ULL << 20, 0).offload);
    REQUIRE(PlanAccumulate(2560000, 128, 40000, kGiB, 0).offload);
}

TEST_CASE("PlanAccumulate chunks stay within the budget", "[ut][gpu_plan]") {
    for (const uint64_t budget : {kGiB / 8, kGiB, 2 * kGiB}) {
        const auto plan = PlanAccumulate(2560000, 128, 4096, budget, 0);
        REQUIRE(plan.offload);
        // From the plan's own definition, so a buffer added to the pass cannot be
        // counted there and missed here.
        REQUIRE(plan.accumulator_bytes + plan.n_chunk * AccumulateBytesPerRow(128) <= budget);
        REQUIRE(plan.n_chunk <= 2560000);
    }
}

TEST_CASE("CapBudget leaves headroom and honours the ceiling", "[ut][gpu_plan]") {
    // Never all of what is free: the context grows and the allocator fragments.
    REQUIRE(CapBudget(10 * kGiB, 0) < 10 * kGiB);
    REQUIRE(CapBudget(10 * kGiB, 2 * kGiB) == 2 * kGiB);
    // A ceiling above the share does not raise it.
    REQUIRE(CapBudget(kGiB, 8 * kGiB) < kGiB);
    REQUIRE(CapBudget(0, 2 * kGiB) == 0);
}

TEST_CASE("A zero budget keeps every path on the CPU", "[ut][gpu_plan]") {
    // What CudaSuggestedBudget returns when there is no usable device.
    REQUIRE_FALSE(PlanAssign(262144, 4096, 128, 0, 0).offload);
    REQUIRE_FALSE(PlanSeed(262144, 128, 4096, 0, 0).offload);
    REQUIRE_FALSE(PlanAccumulate(262144, 128, 4096, 0, 0).offload);
}

TEST_CASE("EffectiveChunkedCap prefers what the caller asked for", "[ut][gpu_plan]") {
    // The rule both callers of the chunked passes apply, in one place so a change
    // to the default reaches both.
    REQUIRE(EffectiveChunkedCap(0) == kChunkedBudgetCap);
    REQUIRE(EffectiveChunkedCap(1) == 1);
    REQUIRE(EffectiveChunkedCap(64 * kGiB) == 64 * kGiB);
}

TEST_CASE("Every plan refuses a budget of zero", "[ut][gpu_plan]") {
    // 0 is what a caller with no device reports, so each pass has to refuse it.
    // PlanAssign says so at the top; the other two reach the same answer through a
    // check they already make, PlanSeed because its resident bytes exceed any
    // budget of 0 and PlanAccumulate because its accumulators do, which is why an
    // explicit guard in either would be unreachable. Pinned here as an outcome so
    // it holds whichever check enforces it.
    constexpr uint64_t count = 262144;
    REQUIRE_FALSE(PlanAssign(count, 4096, 128, 0, 0).offload);
    REQUIRE_FALSE(PlanSeed(count, 128, 1024, 0, 0).offload);
    REQUIRE_FALSE(PlanAccumulate(count, 128, 4096, 0, 0).offload);

    // Again with the work threshold forced to 1, so the refusal is the budget's
    // doing rather than the threshold's.
    REQUIRE_FALSE(PlanAssign(count, 4096, 128, 0, 1).offload);
    REQUIRE_FALSE(PlanSeed(count, 128, 1024, 0, 1).offload);
    REQUIRE_FALSE(PlanAccumulate(count, 128, 4096, 0, 1).offload);
}
