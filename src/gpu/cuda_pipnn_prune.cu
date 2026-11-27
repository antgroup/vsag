// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Stage 4 of the device-resident PiPNN build: candidate edges sorted on
// (source, bucket, distance) are reduced by one warp per vertex to the closest edge of
// every hash bucket, the closest `reservoir` of those, and finally the ones an
// alpha-relaxed occlusion rule leaves, up to `max_degree`.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda_pipnn_internal.h"

namespace vsag::gpu::pipnn {

namespace {

constexpr uint64_t kPruneSmem = 48 * 1024;

__global__ void
vertex_boundary_kernel(const uint64_t* keys,
                       uint64_t n,
                       uint32_t count,
                       uint32_t slab_begin,
                       uint32_t* begin,
                       uint32_t* end) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        uint32_t u = static_cast<uint32_t>(keys[i] >> 32);
        if (u >= count) {
            continue;
        }
        if (i == 0 or static_cast<uint32_t>(keys[i - 1] >> 32) != u) {
            begin[u - slab_begin] = static_cast<uint32_t>(i);
        }
        if (i + 1 == n or static_cast<uint32_t>(keys[i + 1] >> 32) != u) {
            end[u - slab_begin] = static_cast<uint32_t>(i + 1);
        }
    }
}

__device__ __forceinline__ int
row_dot(const int* a, const int* b, uint32_t words) {
    int dot = 0;
    for (uint32_t w = 0; w < words; w += 4) {
        const int4 x = *reinterpret_cast<const int4*>(a + w);
        const int4 y = *reinterpret_cast<const int4*>(b + w);
        dot = __dp4a(x.x, y.x, dot);
        dot = __dp4a(x.y, y.y, dot);
        dot = __dp4a(x.z, y.z, dot);
        dot = __dp4a(x.w, y.w, dot);
    }
    return dot;
}

struct PruneArgs {
    const uint64_t* keys;
    const uint32_t* vals;
    const uint32_t* begin;
    const uint32_t* end;
    const int8_t* q;
    const int32_t* norms;
    uint32_t dim_words;
    uint32_t slab_begin;
    uint32_t slab_count;
    uint32_t max_degree;
    uint32_t reservoir;
    float alpha;
    uint32_t warp_ints;
    bool stage_rows;
    uint32_t* out_neighbors;
    uint32_t* out_degrees;
};

constexpr uint32_t kFullMask = 0xFFFFFFFFU;
constexpr uint32_t kRowPad = 4;

// One warp per vertex. Edges arrive ordered by (bucket, distance), so the first
// non-self edge of each bucket is that direction's shortest. The lanes find those
// leaders together and keep the nearest `reservoir` of them sorted in shared memory.
// The candidates are then walked nearest first and one is dropped when an already
// kept neighbour is closer to it than the alpha-relaxed distance to the vertex;
// at most `max_degree` survive. Each lane checks one kept neighbour at a time.
__global__ void
hash_prune_reduce_kernel(const PruneArgs args) {
    extern __shared__ __align__(16) int s_prune[];
    const uint32_t lane = threadIdx.x & 31U;
    const uint32_t v = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (v >= args.slab_count) {
        return;
    }
    const uint32_t words = args.dim_words;
    const uint32_t row_stride = words + kRowPad;
    const uint32_t reservoir = args.reservoir;
    int* warp_mem = s_prune + (threadIdx.x >> 5) * args.warp_ints;
    int* rows = warp_mem;
    const uint32_t staged_ints = args.stage_rows ? reservoir * row_stride : 0;
    uint32_t* cd = reinterpret_cast<uint32_t*>(warp_mem + staged_ints);
    uint32_t* cv = cd + reservoir;
    int* cn = reinterpret_cast<int*>(cv + reservoir);
    uint32_t* kp = reinterpret_cast<uint32_t*>(cn + reservoir);

    const uint32_t u = args.slab_begin + v;
    const uint32_t first = args.begin[v];
    const uint32_t last = args.end[v];

    uint32_t count = 0;
    for (uint32_t base = first; base < last; base += 32) {
        const uint32_t i = base + lane;
        bool leader = false;
        uint32_t dist = 0;
        uint32_t neighbor = 0;
        if (i < last) {
            neighbor = args.vals[i];
            if (neighbor != u) {
                const uint64_t key = args.keys[i];
                const uint32_t bucket = static_cast<uint32_t>(key >> 20) & 0xFFFU;
                dist = static_cast<uint32_t>(key) & 0xFFFFFU;
                leader = true;
                for (uint32_t j = i; j > first; --j) {
                    const uint64_t prev = args.keys[j - 1];
                    if ((static_cast<uint32_t>(prev >> 20) & 0xFFFU) != bucket) {
                        break;
                    }
                    if (args.vals[j - 1] != u) {
                        leader = false;
                        break;
                    }
                }
            }
        }
        uint32_t mask = __ballot_sync(kFullMask, leader);
        while (mask != 0) {
            const uint32_t src = __ffs(mask) - 1U;
            mask &= mask - 1U;
            const uint32_t d = __shfl_sync(kFullMask, dist, src);
            const uint32_t x = __shfl_sync(kFullMask, neighbor, src);
            if (count == reservoir and d >= cd[reservoir - 1]) {
                continue;
            }
            uint32_t pos = 0;
            for (uint32_t c = 0; c < count; c += 32) {
                const uint32_t s = c + lane;
                pos += __popc(__ballot_sync(kFullMask, s < count and cd[s] <= d));
            }
            const uint32_t new_count = min(count + 1U, reservoir);
            uint32_t moved_d[kMaxDegree / 32];
            uint32_t moved_v[kMaxDegree / 32];
#pragma unroll
            for (uint32_t m = 0; m < kMaxDegree / 32; ++m) {
                const uint32_t s = m * 32U + lane;
                if (s < new_count and s > pos) {
                    moved_d[m] = cd[s - 1];
                    moved_v[m] = cv[s - 1];
                }
            }
            __syncwarp();
#pragma unroll
            for (uint32_t m = 0; m < kMaxDegree / 32; ++m) {
                const uint32_t s = m * 32U + lane;
                if (s < new_count) {
                    if (s == pos) {
                        cd[s] = d;
                        cv[s] = x;
                    } else if (s > pos) {
                        cd[s] = moved_d[m];
                        cv[s] = moved_v[m];
                    }
                }
            }
            __syncwarp();
            count = new_count;
        }
    }

    const int* all_rows = reinterpret_cast<const int*>(args.q);
    for (uint32_t r = lane; r < count; r += 32) {
        cn[r] = args.norms[cv[r]];
    }
    if (args.stage_rows) {
        const uint32_t chunks = words / 4U;
        for (uint32_t idx = lane; idx < count * chunks; idx += 32) {
            const uint32_t r = idx / chunks;
            const uint32_t c = idx - r * chunks;
            const int4 value = *reinterpret_cast<const int4*>(
                all_rows + static_cast<uint64_t>(cv[r]) * words + c * 4U);
            *reinterpret_cast<int4*>(rows + r * row_stride + c * 4U) = value;
        }
    }
    __syncwarp();

    auto row_of = [&](uint32_t r) -> const int* {
        return args.stage_rows ? rows + r * row_stride
                               : all_rows + static_cast<uint64_t>(cv[r]) * words;
    };

    uint32_t kept = 0;
    for (uint32_t i = 0; i < count and kept < args.max_degree; ++i) {
        const float d_uc = __uint_as_float(cd[i] << 11);
        const int* row_i = row_of(i);
        bool occluded = false;
        for (uint32_t j = lane; j < kept; j += 32) {
            const uint32_t p = kp[j];
            const float d_pi =
                static_cast<float>(cn[p] + cn[i] - 2 * row_dot(row_of(p), row_i, words));
            occluded = occluded or args.alpha * d_pi < d_uc;
        }
        if (not __any_sync(kFullMask, occluded)) {
            if (lane == 0) {
                kp[kept] = i;
            }
            ++kept;
            __syncwarp();
        }
    }

    uint32_t* out = args.out_neighbors + static_cast<uint64_t>(u) * args.max_degree;
    for (uint32_t j = lane; j < kept; j += 32) {
        out[j] = cv[kp[j]];
    }
    if (lane == 0) {
        args.out_degrees[u] = kept;
    }
}

}  // namespace

bool
PruneSlab(const PruneJob& job) {
    const uint32_t slab_n = job.slab_end - job.slab_begin;
    VSAG_CUDA_TRY(cudaMemset(job.begin, 0, slab_n * sizeof(uint32_t)));
    VSAG_CUDA_TRY(cudaMemset(job.end, 0, slab_n * sizeof(uint32_t)));
    if (job.edges > 0) {
        vertex_boundary_kernel<<<grid_for(job.edges), kBlockThreads>>>(
            job.keys, job.edges, job.vertex_count, job.slab_begin, job.begin, job.end);
    }
    PruneArgs prune{};
    prune.keys = job.keys;
    prune.vals = job.vals;
    prune.begin = job.begin;
    prune.end = job.end;
    prune.q = job.q;
    prune.norms = job.norms;
    prune.dim_words = job.dim_words;
    prune.slab_begin = job.slab_begin;
    prune.slab_count = slab_n;
    prune.max_degree = job.max_degree;
    prune.reservoir = job.reservoir;
    prune.alpha = job.alpha;
    prune.out_neighbors = job.out_neighbors;
    prune.out_degrees = job.out_degrees;
    const uint32_t row_stride = job.dim_words + kRowPad;
    const uint64_t head = (3ULL * job.reservoir + job.max_degree) * sizeof(int);
    const uint64_t staged = head + static_cast<uint64_t>(job.reservoir) * row_stride * sizeof(int);
    uint32_t warps = static_cast<uint32_t>(std::min<uint64_t>(8, kPruneSmem / staged));
    prune.stage_rows = warps > 0;
    uint64_t per_warp = staged;
    if (warps == 0) {
        warps = 8;
        per_warp = head;
    }
    per_warp = (per_warp + 15) / 16 * 16;
    prune.warp_ints = static_cast<uint32_t>(per_warp / sizeof(int));
    hash_prune_reduce_kernel<<<(slab_n + warps - 1) / warps, warps * 32, per_warp * warps>>>(prune);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace vsag::gpu::pipnn
