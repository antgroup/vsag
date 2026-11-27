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

// Scoring shared by the partition and pick stages: one fused kernel scores INT8 row
// tiles against their candidates with dp4a and keeps each row's top-K in registers. In
// assign mode it picks the nearest leaders; in pick mode it emits leaf-neighbour edges
// with folded sign-hash buckets, and no distance matrix is ever stored.

#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "cuda_pipnn_internal.h"

namespace vsag::gpu::pipnn {

namespace {

constexpr uint32_t kStripeWords = 16;
constexpr uint32_t kStripeStride = 20;
constexpr uint32_t kNoValue = 0xFFFFFFFFU;

// ---------------------------------------------------------------------------
// Fused score-tile kernel
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool
less_pair(int d1, int i1, int d2, int i2) {
    return d1 < d2 or (d1 == d2 and i1 < i2);
}

template <int KC>
__device__ __forceinline__ void
insert_top(int d, int j, int (&best_d)[KC], int (&best_i)[KC]) {
    if (not less_pair(d, j, best_d[KC - 1], best_i[KC - 1])) {
        return;
    }
    int cd = d;
    int ci = j;
#pragma unroll
    for (int p = 0; p < KC; ++p) {
        if (less_pair(cd, ci, best_d[p], best_i[p])) {
            int td = best_d[p];
            int ti = best_i[p];
            best_d[p] = cd;
            best_i[p] = ci;
            cd = td;
            ci = ti;
        }
    }
}

// Four consecutive lanes own one query row and interleaved candidate columns; two
// xor shuffles leave every lane of the group holding the row's complete top-KC.
template <int KC>
__device__ __forceinline__ void
merge_four_lanes(int (&best_d)[KC], int (&best_i)[KC]) {
#pragma unroll
    for (int mask = 1; mask <= 2; mask <<= 1) {
        int other_d[KC];
        int other_i[KC];
#pragma unroll
        for (int s = 0; s < KC; ++s) {
            other_d[s] = __shfl_xor_sync(0xFFFFFFFFU, best_d[s], mask);
            other_i[s] = __shfl_xor_sync(0xFFFFFFFFU, best_i[s], mask);
        }
#pragma unroll
        for (int s = 0; s < KC; ++s) {
            if (other_d[s] != INT_MAX) {
                insert_top<KC>(other_d[s], other_i[s], best_d, best_i);
            }
        }
    }
}

// Folds the sign of (b - a), summed over the coordinates that share a bit, into
// `bits` bits. Coordinate c feeds bit c % bits, so with 12 bits the words of a row
// cycle through three groups of four accumulators and nothing leaves registers.
// A bit is set in the result when its sum is positive and in `reverse` when it is
// negative, so a bit with no signal is clear in both and the bucket of the edge
// (a, b) is the one the edge (b, a) sees from the other end.
__device__ __forceinline__ uint32_t
fold_hash(const int* a, const int* b, uint32_t words, uint32_t bits, uint32_t& reverse) {
    uint32_t hash = 0;
    reverse = 0;
    if (bits == 12) {
        int acc[12];
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            acc[i] = 0;
        }
        for (uint32_t w = 0; w < words; w += 3) {
#pragma unroll
            for (int g = 0; g < 3; ++g) {
                if (w + g < words) {
                    int aw = a[w + g];
                    int bw = b[w + g];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        acc[4 * g + i] += static_cast<int>(static_cast<int8_t>(bw >> (8 * i))) -
                                          static_cast<int>(static_cast<int8_t>(aw >> (8 * i)));
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 12; ++i) {
            hash |= (acc[i] > 0 ? 1U : 0U) << i;
            reverse |= (acc[i] < 0 ? 1U : 0U) << i;
        }
        return hash;
    }
    int acc[kMaxHashBits];
    for (uint32_t i = 0; i < kMaxHashBits; ++i) {
        acc[i] = 0;
    }
    for (uint32_t w = 0; w < words; ++w) {
        int aw = a[w];
        int bw = b[w];
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t bit = (4 * w + i) % bits;
            acc[bit] += static_cast<int>(static_cast<int8_t>(bw >> (8 * i))) -
                        static_cast<int>(static_cast<int8_t>(aw >> (8 * i)));
        }
    }
    for (uint32_t i = 0; i < bits; ++i) {
        hash |= (acc[i] > 0 ? 1U : 0U) << i;
        reverse |= (acc[i] < 0 ? 1U : 0U) << i;
    }
    return hash;
}

__device__ __forceinline__ uint64_t
make_key(uint32_t u, uint32_t bucket, int dist) {
    uint32_t dist_bits = (__float_as_uint(static_cast<float>(dist)) >> 11) & 0xFFFFFU;
    return (static_cast<uint64_t>(u) << 32) | (static_cast<uint64_t>(bucket) << 20) | dist_bits;
}

// kPick = false: every member of a segment is scored against the segment's candidate
// leaders and the nearest `out_k` candidate indices are stored.
// kPick = true: a leaf is scored against itself and the nearest `out_k` other members
// of every row become two edges, one in each direction.
template <int KC, bool kPick>
__global__ void
__launch_bounds__(kBlockThreads) score_topk_kernel(const ScoreArgs args) {
    __shared__ __align__(16) int s_q[kTileRows][kStripeStride];
    __shared__ __align__(16) int s_db[kTileRows][kStripeStride];
    __shared__ int s_db_norm[kTileRows];
    __shared__ uint32_t s_db_id[kTileRows];

    const uint32_t tid = threadIdx.x;
    const uint32_t row = tid >> 2;
    const uint32_t lane = tid & 3U;

    const uint32_t seg = args.tile_seg[blockIdx.x];
    const uint32_t row0 = args.tile_row0[blockIdx.x];
    const uint32_t m_begin = args.member_off[seg];
    const uint32_t m_count = args.member_off[seg + 1] - m_begin;
    const uint32_t c_begin = args.cand_off[seg];
    const uint32_t c_count = args.cand_off[seg + 1] - c_begin;

    const uint32_t seg_row = row0 + row;
    const bool row_valid = seg_row < m_count;
    const uint32_t pos = m_begin + seg_row;
    const uint32_t qid = row_valid ? (args.q_ids != nullptr ? args.q_ids[pos] : pos) : 0U;
    const int qnorm = args.norms[qid];

    int best_d[KC];
    int best_i[KC];
#pragma unroll
    for (int s = 0; s < KC; ++s) {
        best_d[s] = INT_MAX;
        best_i[s] = INT_MAX;
    }

    const uint32_t dim_words = args.dim_words;
    const uint32_t part = lane * 4U;
    const int8_t* q_row_ptr = args.q + static_cast<uint64_t>(qid) * args.dim_pad;

    for (uint32_t t0 = 0; t0 < c_count; t0 += kTileRows) {
        __syncthreads();
        if (tid < kTileRows) {
            uint32_t j = t0 + tid;
            if (j < c_count) {
                uint32_t id = args.c_ids[c_begin + j];
                s_db_id[tid] = id;
                s_db_norm[tid] = args.norms[id];
            } else {
                s_db_id[tid] = kNoValue;
                s_db_norm[tid] = 0;
            }
        }
        __syncthreads();

        int acc[16];
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            acc[c] = 0;
        }
        const uint32_t db_id = s_db_id[row];
        const int8_t* db_row_ptr =
            args.q + static_cast<uint64_t>(db_id == kNoValue ? 0U : db_id) * args.dim_pad;

        for (uint32_t w0 = 0; w0 < dim_words; w0 += kStripeWords) {
            const uint32_t w = w0 + part;
            int4 qv = make_int4(0, 0, 0, 0);
            int4 dv = make_int4(0, 0, 0, 0);
            if (w < dim_words) {
                qv = *reinterpret_cast<const int4*>(q_row_ptr + 4U * w);
                if (db_id != kNoValue) {
                    dv = *reinterpret_cast<const int4*>(db_row_ptr + 4U * w);
                }
            }
            *reinterpret_cast<int4*>(&s_q[row][part]) = qv;
            *reinterpret_cast<int4*>(&s_db[row][part]) = dv;
            __syncthreads();

            int4 a[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                a[k] = *reinterpret_cast<const int4*>(&s_q[row][4 * k]);
            }
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const int4* p = reinterpret_cast<const int4*>(&s_db[lane + 4 * c][0]);
                int sum = acc[c];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    int4 b = p[k];
                    sum = __dp4a(a[k].x, b.x, sum);
                    sum = __dp4a(a[k].y, b.y, sum);
                    sum = __dp4a(a[k].z, b.z, sum);
                    sum = __dp4a(a[k].w, b.w, sum);
                }
                acc[c] = sum;
            }
            __syncthreads();
        }

        if (row_valid) {
#pragma unroll
            for (int c = 0; c < 16; ++c) {
                const uint32_t jj = lane + 4U * c;
                const uint32_t j = t0 + jj;
                if (j < c_count) {
                    const uint32_t cid = s_db_id[jj];
                    if (not kPick or cid != qid) {
                        int d = qnorm + s_db_norm[jj] - 2 * acc[c];
                        insert_top<KC>(d, static_cast<int>(j), best_d, best_i);
                    }
                }
            }
        }
    }

    merge_four_lanes<KC>(best_d, best_i);

    if (not row_valid) {
        return;
    }

    if constexpr (not kPick) {
#pragma unroll
        for (int s = 0; s < KC; ++s) {
            if (static_cast<uint32_t>(s) < args.out_k and (static_cast<uint32_t>(s) & 3U) == lane) {
                args.out_assign[static_cast<uint64_t>(pos) * args.out_k + s] =
                    best_i[s] == INT_MAX ? kNoValue : c_begin + static_cast<uint32_t>(best_i[s]);
            }
        }
    } else {
        const uint64_t sentinel = static_cast<uint64_t>(args.count) << 32;
#pragma unroll
        for (int s = 0; s < KC; ++s) {
            if (static_cast<uint32_t>(s) < args.out_k and (static_cast<uint32_t>(s) & 3U) == lane) {
                const uint64_t slot =
                    (static_cast<uint64_t>(pos) - args.pos_base) * 2U * args.out_k + 2U * s;
                uint64_t fwd_key = sentinel;
                uint64_t rev_key = sentinel;
                uint32_t v = 0;
                if (best_i[s] != INT_MAX) {
                    v = args.c_ids[c_begin + static_cast<uint32_t>(best_i[s])];
                    const int* a = reinterpret_cast<const int*>(q_row_ptr);
                    const int* b =
                        reinterpret_cast<const int*>(args.q) + static_cast<uint64_t>(v) * dim_words;
                    uint32_t reverse_bucket = 0;
                    const uint32_t bucket =
                        fold_hash(a, b, dim_words, args.hash_bits, reverse_bucket);
                    if (qid >= args.slab_begin and qid < args.slab_end) {
                        fwd_key = make_key(qid, bucket, best_d[s]);
                    }
                    if (v >= args.slab_begin and v < args.slab_end) {
                        rev_key = make_key(v, reverse_bucket, best_d[s]);
                    }
                }
                args.out_keys[slot] = fwd_key;
                args.out_vals[slot] = v;
                args.out_keys[slot + 1] = rev_key;
                args.out_vals[slot + 1] = qid;
            }
        }
    }
}

__global__ void
edge_flag_kernel(const uint64_t* keys, uint64_t n, uint32_t count, uint8_t* flags) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        flags[i] = (keys[i] >> 32) < count ? 1 : 0;
    }
}

}  // namespace

template <bool kPick>
bool
launch_score(const ScoreArgs& args, uint32_t tiles) {
    if (tiles == 0) {
        return true;
    }
    const uint32_t k = args.out_k;
    if (k <= 2) {
        score_topk_kernel<2, kPick><<<tiles, kBlockThreads>>>(args);
    } else if (k <= 4) {
        score_topk_kernel<4, kPick><<<tiles, kBlockThreads>>>(args);
    } else if (k <= 6) {
        score_topk_kernel<6, kPick><<<tiles, kBlockThreads>>>(args);
    } else if (k <= 8) {
        score_topk_kernel<8, kPick><<<tiles, kBlockThreads>>>(args);
    } else if (k <= 12) {
        score_topk_kernel<12, kPick><<<tiles, kBlockThreads>>>(args);
    } else {
        score_topk_kernel<16, kPick><<<tiles, kBlockThreads>>>(args);
    }
    return cudaGetLastError() == cudaSuccess;
}

template bool
launch_score<false>(const ScoreArgs& args, uint32_t tiles);

template bool
launch_score<true>(const ScoreArgs& args, uint32_t tiles);

bool
FlagEdges(const uint64_t* keys, uint64_t n, uint32_t count, uint8_t* flags) {
    edge_flag_kernel<<<grid_for(n), kBlockThreads>>>(keys, n, count, flags);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace vsag::gpu::pipnn
