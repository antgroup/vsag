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
#include <cfloat>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"
#include "gpu_plan.h"

// k-means++ seeding.
//
// Each of the k steps picks a row with probability proportional to its squared
// distance to the nearest centroid so far. The dataset and the running minima
// stay on the device, so a step is two launches and no host round trip: one sweep
// folds the new centroid into the minima and reduces per tile, then a single block
// walks the tile sums to draw the next row.

namespace vsag::gpu {

namespace {

constexpr int kThreads = 256;
/// The seeding sweep runs one warp wide by this many warps.
constexpr int32_t kInitWarps = 8;

__global__ void
fill_float(float* __restrict__ p, uint64_t n, float v) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i] = v;
    }
}

/// Folds the distance to the new centroid into min_dist and reduces the weights
/// into one partial sum per tile. The centroid is read out of the resident dataset
/// via sel_prev, so consecutive steps need no host round trip.
__global__ void
kpp_update_min(const float* __restrict__ data,
               uint64_t count,
               int dim,
               const uint64_t* __restrict__ sel_prev,
               uint64_t rows_per_block,
               float* __restrict__ min_dist,
               float* __restrict__ block_sum) {
    const uint64_t begin = static_cast<uint64_t>(blockIdx.x) * rows_per_block;
    uint64_t end = begin + rows_per_block;
    if (end > count) {
        end = count;
    }
    const float* centroid = data + (*sel_prev) * static_cast<uint64_t>(dim);
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);

    // The tile sums stay float while kpp_select's scan over them is double. The
    // warps stride the tile, so each of these holds about rows_per_block /
    // kInitWarps terms, and that split is what bounds the error: at the largest
    // shape seeding accepts, 11000000 rows over 1024 tiles, the tile sums land
    // 3.1e-9 from the exact total against 2.5e-10 for double partials, and the
    // worst shift of a normalised prefix is 3.8e-9, so a draw lands elsewhere with
    // about that probability. Widening these would also widen block_sum and halve
    // the rows kMaxInitShared admits.
    float acc = 0.F;
    for (uint64_t r = begin + static_cast<uint64_t>(warp); r < end;
         r += static_cast<uint64_t>(blockDim.y)) {
        const float* p = data + r * static_cast<uint64_t>(dim);
        float d = 0.F;
        for (int j = lane; j < dim; j += static_cast<int>(blockDim.x)) {
            const float t = p[j] - centroid[j];
            d += t * t;
        }
        // The shuffle is a warp primitive and the mask names every lane, so this
        // reduction is the one place the hardware width is the right number. It
        // sums the whole row only because the block is launched one warp wide.
        for (int off = warpSize / 2; off > 0; off >>= 1) {
            d += __shfl_down_sync(0xffffffffU, d, off);
        }
        if (lane == 0) {
            float m = min_dist[r];
            if (d < m) {
                m = d;
                min_dist[r] = m;
            }
            acc += m;
        }
    }

    __shared__ float s[kInitWarps];
    if (lane == 0) {
        s[warp] = acc;
    }
    __syncthreads();
    if (warp == 0 and lane == 0) {
        float v = 0.F;
        for (int i = 0; i < static_cast<int>(blockDim.y); ++i) {
            v += s[i];
        }
        block_sum[blockIdx.x] = v;
    }
}

/// Single-block weighted draw. Both walks are over shared memory, so the
/// sequential prefix scan costs far less than the sweep above.
__global__ void
kpp_select(const float* __restrict__ block_sum,
           const float* __restrict__ min_dist,
           uint64_t count,
           uint64_t rows_per_block,
           int32_t nblocks,
           float u_pick,
           float u_fallback,
           uint64_t* __restrict__ sel_out) {
    extern __shared__ float sh[];
    float* s_blk = sh;
    float* s_row = sh + nblocks;

    // Thread 0 is the only one that writes or reads these two, so they are its
    // own rather than shared, which leaves s_block as the one value that crosses
    // threads and the barriers below as being about it and s_blk.
    double total = 0.0;
    double prefix = 0.0;
    __shared__ int32_t s_block;

    for (int32_t b = static_cast<int32_t>(threadIdx.x); b < nblocks;
         b += static_cast<int32_t>(blockDim.x)) {
        s_blk[b] = block_sum[b];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int32_t b = 0; b < nblocks; ++b) {
            total += static_cast<double>(s_blk[b]);
        }
        s_block = -1;
        if (total > 0.0) {
            const double threshold = static_cast<double>(u_pick) * total;
            double cum = 0.0;
            int32_t b = 0;
            for (; b < nblocks - 1; ++b) {
                const double next = cum + static_cast<double>(s_blk[b]);
                if (next >= threshold) {
                    break;
                }
                cum = next;
            }
            s_block = b;
            prefix = cum;
        }
    }
    __syncthreads();

    if (s_block < 0) {
        // Every remaining point coincides with a centroid: uniform pick, as the
        // CPU routine does.
        if (threadIdx.x == 0) {
            uint64_t idx =
                static_cast<uint64_t>(static_cast<double>(u_fallback) * static_cast<double>(count));
            *sel_out = idx >= count ? count - 1 : idx;
        }
        return;
    }

    // PlanSeed derives the count from the size, so begin is inside the dataset.
    const uint64_t begin = static_cast<uint64_t>(s_block) * rows_per_block;
    const uint64_t end = min(begin + rows_per_block, count);
    const uint64_t len = end - begin;
    for (uint64_t r = threadIdx.x; r < len; r += blockDim.x) {
        s_row[r] = min_dist[begin + r];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Double again, and for the same reason as the tile scan above: it bounds
        // this walk's own error rather than recovering what the float `min_dist`
        // and tile sums already cost, which kpp_update_min measures.
        const double threshold = static_cast<double>(u_pick) * total;
        double cum = prefix;
        uint64_t idx = end - 1;
        for (uint64_t r = 0; r < len; ++r) {
            cum += static_cast<double>(s_row[r]);
            if (cum >= threshold) {
                idx = begin + r;
                break;
            }
        }
        *sel_out = idx;
    }
}

}  // namespace

bool
CudaKMeansPlusPlusInit(const float* datas,
                       uint64_t count,
                       int32_t dim,
                       uint32_t k,
                       const float* uniforms,
                       float* centroids_out,
                       uint64_t budget_bytes,
                       uint64_t min_work) {
    if (datas == nullptr or uniforms == nullptr or centroids_out == nullptr or count == 0 or
        k == 0 or dim <= 0 or static_cast<uint64_t>(k) > count) {
        return false;
    }
    const SeedPlan plan = PlanSeed(count, dim, k, budget_bytes, min_work);
    if (not plan.offload or not CudaAvailable()) {
        return false;
    }
    ClearPendingError();

    const uint64_t data_bytes = count * static_cast<uint64_t>(dim) * sizeof(float);
    DeviceArray<float> data;
    DeviceArray<float> min_dist;
    DeviceArray<float> block_sum;
    DeviceArray<uint64_t> selected;
    if (not data.Alloc(count * static_cast<uint64_t>(dim)) or not min_dist.Alloc(count) or
        not block_sum.Alloc(static_cast<uint64_t>(plan.blocks)) or not selected.Alloc(k)) {
        return false;
    }
    if (cudaMemcpy(data.Get(), datas, data_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }

    std::vector<uint64_t> picks(k, 0);
    fill_float<<<static_cast<int>((count + kThreads - 1) / kThreads), kThreads>>>(
        min_dist.Get(), count, FLT_MAX);

    // Clamped, not refused: uniforms are documented as [0, 1), and this folds the
    // closed edge a caller may hand over. Unlike the readback below, nothing has
    // gone wrong here.
    uint64_t first =
        static_cast<uint64_t>(static_cast<double>(uniforms[0]) * static_cast<double>(count));
    if (first >= count) {
        first = count - 1;
    }
    picks[0] = first;
    if (cudaMemcpy(selected.Get(), picks.data(), sizeof(uint64_t), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }

    const dim3 threads(kWarpWidth, kInitWarps);
    for (uint32_t c = 1; c < k; ++c) {
        kpp_update_min<<<plan.blocks, threads>>>(data.Get(),
                                                 count,
                                                 dim,
                                                 selected.Get() + (c - 1),
                                                 plan.rows_per_block,
                                                 min_dist.Get(),
                                                 block_sum.Get());
        kpp_select<<<1, kThreads, plan.shared_bytes>>>(block_sum.Get(),
                                                       min_dist.Get(),
                                                       count,
                                                       plan.rows_per_block,
                                                       plan.blocks,
                                                       uniforms[2 * c],
                                                       uniforms[2 * c + 1],
                                                       selected.Get() + c);
    }
    if (cudaGetLastError() != cudaSuccess or cudaMemcpy(picks.data(),
                                                        selected.Get(),
                                                        static_cast<uint64_t>(k) * sizeof(uint64_t),
                                                        cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }

    // Copied from the caller's buffer; the device copy only keeps sweeps local.
    //
    // A pick outside the dataset cannot come from a correct kpp_select, which
    // clamps every branch, so this bounds a value that crossed the device
    // boundary. It refuses rather than substituting a row, so the caller gets the
    // host's k-means++ instead of centroids this function does not claim to
    // produce, and it checks every pick before the first write, because false has
    // to leave the caller's buffer as it found it.
    for (uint32_t c = 0; c < k; ++c) {
        if (picks[c] >= count) {
            return false;
        }
    }
    for (uint32_t c = 0; c < k; ++c) {
        std::copy(datas + picks[c] * static_cast<uint64_t>(dim),
                  datas + (picks[c] + 1) * static_cast<uint64_t>(dim),
                  centroids_out + static_cast<uint64_t>(c) * static_cast<uint64_t>(dim));
    }
    return true;
}

}  // namespace vsag::gpu
