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

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"
#include "gpu_plan.h"

// Centroid accumulation: each cluster's sum and point count, which the caller
// divides to get the next iteration's centroids. One scatter-add over the points
// replaces a per-task k x dim buffer on the host.

namespace vsag::gpu {

namespace {

/// The scatter-add block is one warp wide by this many warps.
constexpr int32_t kAccumulateWarps = 8;

/// One warp per point, threads along dim so the reads coalesce. Only about
/// count/k rows compete for the same accumulator, so the atomics rarely contend.
__global__ void
accumulate_centroids(const float* __restrict__ data,
                     const int32_t* __restrict__ labels,
                     uint64_t rows,
                     int dim,
                     uint32_t k,
                     float* __restrict__ sums,
                     int32_t* __restrict__ counts) {
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int32_t lab = labels[row];
    // Skipped rather than clamped or refused, because the host path does the same:
    // its own accumulation is guarded by `label >= 0 && label < k` and a row
    // outside that adds nothing. Refusing here would hand the caller a fallback
    // that drops the identical row, after the work.
    if (lab < 0 or static_cast<uint32_t>(lab) >= k) {
        return;
    }
    const float* p = data + row * static_cast<uint64_t>(dim);
    float* dst = sums + static_cast<uint64_t>(lab) * static_cast<uint64_t>(dim);
    for (int j = static_cast<int>(threadIdx.x); j < dim; j += static_cast<int>(blockDim.x)) {
        atomicAdd(dst + j, p[j]);
    }
    if (threadIdx.x == 0) {
        atomicAdd(counts + lab, 1);
    }
}

}  // namespace

bool
CudaAccumulateCentroids(const float* datas,
                        uint64_t count,
                        int32_t dim,
                        const int32_t* labels,
                        uint32_t k,
                        float* sums,
                        int32_t* counts,
                        uint64_t budget_bytes,
                        uint64_t min_work) {
    if (datas == nullptr or labels == nullptr or sums == nullptr or counts == nullptr or
        count == 0 or k == 0 or dim <= 0) {
        return false;
    }
    const AccumulatePlan plan = PlanAccumulate(count, dim, k, budget_bytes, min_work);
    if (not plan.offload or not CudaAvailable()) {
        return false;
    }
    ClearPendingError();

    const uint64_t sum_bytes =
        static_cast<uint64_t>(k) * static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t cnt_bytes = static_cast<uint64_t>(k) * sizeof(int32_t);

    DeviceArray<float> device_sums;
    DeviceArray<int32_t> device_counts;
    DeviceArray<float> chunk_data;
    DeviceArray<int32_t> chunk_labels;
    if (not device_sums.Alloc(static_cast<uint64_t>(k) * dim) or not device_counts.Alloc(k) or
        not chunk_data.Alloc(plan.n_chunk * static_cast<uint64_t>(dim)) or
        not chunk_labels.Alloc(plan.n_chunk)) {
        return false;
    }
    if (cudaMemset(device_sums.Get(), 0, sum_bytes) != cudaSuccess or
        cudaMemset(device_counts.Get(), 0, cnt_bytes) != cudaSuccess) {
        return false;
    }

    // Copied straight out of the caller's buffer, not through a pinned one. The
    // data starts in the caller's own pages, so a pinned buffer would have to be
    // filled by a host copy first, and on a 1014 MiB chunk that copy plus the
    // faster transfer took 189 ms against 110 ms for copying directly, before
    // counting the 336 ms cudaHostAlloc needs once per k-means iteration.
    //
    // All on the default stream, so an upload cannot overtake the kernel still
    // reading what it writes into. Overlapping the two needs a second chunk
    // buffer, not only a second stream, and has almost nothing to hide: on
    // 2000000 rows of 128 in four chunks the uploads alone took 87.2 ms against
    // 1.5 ms for all four launches, and a two-buffer two-stream version came to
    // 87.5 ms. Fastest of nine interleaved runs each. Below roughly four million
    // rows at this dim the loop runs once anyway, kChunkedBudgetCap holding the
    // whole input in one chunk.
    const dim3 threads(kWarpWidth, kAccumulateWarps);
    for (uint64_t off = 0; off < count; off += plan.n_chunk) {
        const uint64_t cur = std::min<uint64_t>(plan.n_chunk, count - off);
        if (cudaMemcpy(chunk_data.Get(),
                       datas + off * static_cast<uint64_t>(dim),
                       cur * static_cast<uint64_t>(dim) * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess or
            cudaMemcpy(
                chunk_labels.Get(), labels + off, cur * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            return false;
        }
        accumulate_centroids<<<static_cast<int>((cur + threads.y - 1) / threads.y), threads>>>(
            chunk_data.Get(),
            chunk_labels.Get(),
            cur,
            dim,
            k,
            device_sums.Get(),
            device_counts.Get());
    }
    // The default stream is the only one this pass uses, and waiting on it alone
    // leaves whatever else the process has on the card out of it.
    if (cudaStreamSynchronize(nullptr) != cudaSuccess or cudaGetLastError() != cudaSuccess) {
        return false;
    }

    // Staged and handed over together: the caller's fallback adds onto `sums`, so
    // a half-written `sums` with a false return would be counted twice.
    //
    // Synchronous and unpinned in this direction too, for a different reason than
    // the uploads: it follows a full stream sync, so there is nothing left to
    // overlap with, and `k * dim` floats plus `k` ints is a fraction of one chunk.
    std::vector<float> staged_sums(static_cast<uint64_t>(k) * dim);
    std::vector<int32_t> staged_counts(k);
    if (cudaMemcpy(staged_sums.data(), device_sums.Get(), sum_bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess or
        cudaMemcpy(staged_counts.data(), device_counts.Get(), cnt_bytes, cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        return false;
    }
    std::copy(staged_sums.begin(), staged_sums.end(), sums);
    std::copy(staged_counts.begin(), staged_counts.end(), counts);
    return true;
}

}  // namespace vsag::gpu
