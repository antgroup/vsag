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

#include "pipnn_simd_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "simd/bf16_simd.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && \
    ((defined(__GNUC__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 14))
#define VSAG_PIPNN_COMPILE_AVX_VNNI 1
#else
#define VSAG_PIPNN_COMPILE_AVX_VNNI 0
#endif

#if defined(__aarch64__) &&                                                    \
    (defined(__ARM_FEATURE_DOTPROD) || (defined(__GNUC__) && __GNUC__ >= 9) || \
     (defined(__clang__) && __clang_major__ >= 11))
#define VSAG_PIPNN_COMPILE_NEON_DOTPROD 1
#else
#define VSAG_PIPNN_COMPILE_NEON_DOTPROD 0
#endif

namespace vsag::pipnn_simd {

namespace {

constexpr uint32_t FLIP_U32 = 0x80808080U;
constexpr float SENTINEL = std::numeric_limits<float>::max();

void*
alloc_aligned64(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const size_t aligned_bytes = (bytes + 63ULL) & ~63ULL;
    void* ptr = std::aligned_alloc(64, aligned_bytes);
    if (ptr == nullptr) {
        throw std::bad_alloc();
    }
    return ptr;
}

inline bool
is_finite_f32(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return (bits & 0x7F800000U) != 0x7F800000U;
}

inline float
sanitize_distance(float distance) {
    if (not is_finite_f32(distance)) {
        return std::numeric_limits<float>::infinity();
    }
    return distance == 0.0F ? 0.0F : distance;
}

inline uint16_t
float_to_order_key(float value) {
    const uint16_t bits = generic::FloatToBF16(sanitize_distance(value));
    if ((bits & 0x8000U) != 0) {
        return static_cast<uint16_t>(~bits);
    }
    return static_cast<uint16_t>(bits ^ 0x8000U);
}

inline uint32_t
id_gap(uint32_t a, uint32_t b) {
    return a >= b ? (a - b) : (b - a);
}

inline bool
better_leaf(float lhs_dist, uint32_t src_id, uint32_t lhs_id, float rhs_dist, uint32_t rhs_id) {
    if (lhs_dist != rhs_dist) {
        return lhs_dist < rhs_dist;
    }
    const uint32_t lhs_gap = id_gap(src_id, lhs_id);
    const uint32_t rhs_gap = id_gap(src_id, rhs_id);
    if (lhs_gap != rhs_gap) {
        return lhs_gap < rhs_gap;
    }
    return lhs_id < rhs_id;
}

inline bool
better_leader(float lhs_dist, uint32_t lhs_id, float rhs_dist, uint32_t rhs_id) {
    if (lhs_dist != rhs_dist) {
        return lhs_dist < rhs_dist;
    }
    return lhs_id < rhs_id;
}

inline void
update_leaf_topk(uint32_t point_idx,
                 uint32_t cand_local_idx,
                 float cand_dist,
                 uint32_t k,
                 const uint32_t* global_ids,
                 VnniScratch& scratch) {
    float* dists = scratch.topk_dists + static_cast<size_t>(point_idx) * k;
    uint32_t* idxs = scratch.topk_idx + static_cast<size_t>(point_idx) * k;
    uint16_t cnt = scratch.topk_count[point_idx];

    if (cnt < k) {
        dists[cnt] = cand_dist;
        idxs[cnt] = cand_local_idx;
        if (cnt == 0) {
            scratch.topk_farthest_pos[point_idx] = 0;
        } else {
            const uint16_t fpos = scratch.topk_farthest_pos[point_idx];
            const float fdist = dists[fpos];
            if (cand_dist > fdist or
                (cand_dist == fdist and better_leaf(fdist,
                                                    global_ids[point_idx],
                                                    global_ids[idxs[fpos]],
                                                    cand_dist,
                                                    global_ids[cand_local_idx]))) {
                scratch.topk_farthest_pos[point_idx] = cnt;
            }
        }
        ++cnt;
        scratch.topk_count[point_idx] = cnt;
        if (cnt == k) {
            scratch.farthest_dist[point_idx] = dists[scratch.topk_farthest_pos[point_idx]];
        }
        return;
    }

    const uint16_t fpos = scratch.topk_farthest_pos[point_idx];
    const float fdist = dists[fpos];
    if (cand_dist > fdist) {
        return;
    }
    if (cand_dist == fdist and not better_leaf(cand_dist,
                                               global_ids[point_idx],
                                               global_ids[cand_local_idx],
                                               fdist,
                                               global_ids[idxs[fpos]])) {
        return;
    }
    dists[fpos] = cand_dist;
    idxs[fpos] = cand_local_idx;

    uint16_t new_fpos = 0;
    float max_d = dists[0];
    for (uint16_t s = 1; s < cnt; ++s) {
        const float sd = dists[s];
        if (sd > max_d) {
            new_fpos = s;
            max_d = sd;
        } else if (sd == max_d) [[unlikely]] {
            if (better_leaf(max_d,
                            global_ids[point_idx],
                            global_ids[idxs[new_fpos]],
                            sd,
                            global_ids[idxs[s]])) {
                new_fpos = s;
            }
        }
    }
    scratch.topk_farthest_pos[point_idx] = new_fpos;
    scratch.farthest_dist[point_idx] = max_d;
}

inline void
update_leader_topk(uint32_t point_idx,
                   uint32_t leader_idx,
                   float cand_dist,
                   uint32_t fanout,
                   const uint32_t* leader_ids,
                   VnniScratch& scratch) {
    float* dists = scratch.topk_dists + static_cast<size_t>(point_idx) * fanout;
    uint32_t* idxs = scratch.topk_idx + static_cast<size_t>(point_idx) * fanout;
    uint16_t cnt = scratch.topk_count[point_idx];

    if (cnt < fanout) {
        dists[cnt] = cand_dist;
        idxs[cnt] = leader_idx;
        if (cnt == 0) {
            scratch.topk_farthest_pos[point_idx] = 0;
        } else {
            const uint16_t fpos = scratch.topk_farthest_pos[point_idx];
            const float fdist = dists[fpos];
            if (cand_dist > fdist or
                (cand_dist == fdist and leader_ids[leader_idx] > leader_ids[idxs[fpos]])) {
                scratch.topk_farthest_pos[point_idx] = cnt;
            }
        }
        ++cnt;
        scratch.topk_count[point_idx] = cnt;
        if (cnt == fanout) {
            scratch.farthest_dist[point_idx] = dists[scratch.topk_farthest_pos[point_idx]];
        }
        return;
    }

    const uint16_t fpos = scratch.topk_farthest_pos[point_idx];
    const float fdist = dists[fpos];
    if (cand_dist > fdist) {
        return;
    }
    if (cand_dist == fdist and leader_ids[leader_idx] >= leader_ids[idxs[fpos]]) {
        return;
    }
    dists[fpos] = cand_dist;
    idxs[fpos] = leader_idx;

    uint16_t new_fpos = 0;
    float max_d = dists[0];
    for (uint16_t s = 1; s < cnt; ++s) {
        const float sd = dists[s];
        if (sd > max_d) {
            new_fpos = s;
            max_d = sd;
        } else if (sd == max_d) [[unlikely]] {
            if (leader_ids[idxs[s]] > leader_ids[idxs[new_fpos]]) {
                new_fpos = s;
            }
        }
    }
    scratch.topk_farthest_pos[point_idx] = new_fpos;
    scratch.farthest_dist[point_idx] = max_d;
}

inline void
sort_leaf_topk_row(uint32_t point_idx,
                   uint32_t k,
                   const uint32_t* global_ids,
                   VnniScratch& scratch) {
    const uint16_t cnt = scratch.topk_count[point_idx];
    if (cnt <= 1) {
        return;
    }
    float* dists = scratch.topk_dists + static_cast<size_t>(point_idx) * k;
    uint32_t* idxs = scratch.topk_idx + static_cast<size_t>(point_idx) * k;
    const uint32_t src_id = global_ids[point_idx];
    for (uint16_t i = 1; i < cnt; ++i) {
        const float kd = dists[i];
        const uint32_t ki = idxs[i];
        const uint32_t kid = global_ids[ki];
        uint16_t j = i;
        while (j > 0 and better_leaf(kd, src_id, kid, dists[j - 1], global_ids[idxs[j - 1]])) {
            dists[j] = dists[j - 1];
            idxs[j] = idxs[j - 1];
            --j;
        }
        dists[j] = kd;
        idxs[j] = ki;
    }
}

inline void
write_sorted_assignments(uint64_t stripe_size,
                         uint32_t fanout,
                         const VnniScratch& leader_scratch,
                         const VnniScratch& worker_scratch,
                         uint32_t* out_assignments) {
    for (uint64_t i = 0; i < stripe_size; ++i) {
        const uint32_t* idxs = worker_scratch.topk_idx + i * fanout;
        const float* dists = worker_scratch.topk_dists + i * fanout;
        const uint32_t cnt = worker_scratch.topk_count[i];
        if (cnt <= 64) {
            uint32_t order[64];
            for (uint32_t s = 0; s < cnt; ++s) {
                order[s] = s;
            }
            std::sort(order, order + cnt, [&](uint32_t a, uint32_t b) {
                return better_leader(dists[a],
                                     leader_scratch.global_ids[idxs[a]],
                                     dists[b],
                                     leader_scratch.global_ids[idxs[b]]);
            });
            for (uint32_t s = 0; s < cnt; ++s) {
                out_assignments[i * fanout + s] = idxs[order[s]];
            }
        } else {
            std::vector<uint32_t> order(cnt);
            for (uint32_t s = 0; s < cnt; ++s) {
                order[s] = s;
            }
            std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
                return better_leader(dists[a],
                                     leader_scratch.global_ids[idxs[a]],
                                     dists[b],
                                     leader_scratch.global_ids[idxs[b]]);
            });
            for (uint32_t s = 0; s < cnt; ++s) {
                out_assignments[i * fanout + s] = idxs[order[s]];
            }
        }
    }
}

inline float
finalize_scalar_distance(float dot, float norm_a, float norm_b, MetricKind metric) {
    if (metric == MetricKind::L2_SQR) {
        const float d = norm_a + norm_b - dot - dot;
        return d > 0.0F ? d : 0.0F;
    }
    if (metric == MetricKind::IP) {
        const float d = 1.0F - dot;
        return d == 0.0F ? 0.0F : d;
    }
    const float sim = std::clamp(dot, -1.0F, 1.0F);
    const float d = 1.0F - sim;
    return d == 0.0F ? 0.0F : d;
}

inline float
finalize_scalar_distance_fp32(float dot, float norm_a, float norm_b, MetricKind metric) {
    if (metric == MetricKind::L2_SQR) {
        const float sum = norm_a + norm_b;
        const float d = sanitize_distance(sum - dot - dot);
        return d > 0.0F ? d : 0.0F;
    }
    if (metric == MetricKind::IP) {
        return sanitize_distance(1.0F - dot);
    }
    const float sim = std::clamp(dot * norm_a * norm_b, -1.0F, 1.0F);
    return sanitize_distance(1.0F - sim);
}

inline float
sq8_dot_from_signed_acc(
    int32_t acc, float step_a, float min_a, float su_a, float step_b, float c_b, float sum_x_b) {
    return (step_a * step_b) * static_cast<float>(acc) + su_a * c_b + min_a * sum_x_b;
}

inline float
sq8_dot_from_unsigned_acc(
    uint32_t acc, float step_a, float min_a, float su_a, float step_b, float min_b, float sum_x_b) {
    return (step_a * step_b) * static_cast<float>(acc) + su_a * min_b + min_a * sum_x_b;
}

inline void
finalize_quantize_meta(uint64_t dim,
                       float min_val,
                       float step,
                       uint64_t sum_u,
                       uint64_t sum_u2,
                       MetricKind metric,
                       QMeta& out_meta) {
    const auto d_step = static_cast<double>(step);
    const auto d_min = static_cast<double>(min_val);
    const double d_su = d_step * static_cast<double>(sum_u);
    out_meta.step = step;
    out_meta.min_val = min_val;
    out_meta.su = static_cast<float>(d_su);
    if (metric == MetricKind::L2_SQR) {
        const double norm_hat = static_cast<double>(dim) * d_min * d_min + 2.0 * d_min * d_su +
                                d_step * d_step * static_cast<double>(sum_u2);
        out_meta.norm = static_cast<float>(norm_hat > 0.0 ? norm_hat : 0.0);
    } else {
        out_meta.norm = 0.0F;
    }
}

__attribute__((noinline)) void
flush_leaf_mask_hits(uint64_t q_idx,
                     uint64_t panel_base,
                     uint32_t neighbor_count,
                     uint32_t q_mask,
                     uint32_t db_mask,
                     const float* tmp_d,
                     VnniScratch& scratch) {
    uint32_t m = q_mask;
    while (m != 0) {
        const auto lane = static_cast<uint32_t>(__builtin_ctz(m));
        m &= (m - 1U);
        update_leaf_topk(static_cast<uint32_t>(q_idx),
                         static_cast<uint32_t>(panel_base + lane),
                         tmp_d[lane],
                         neighbor_count,
                         scratch.global_ids,
                         scratch);
    }
    m = db_mask;
    while (m != 0) {
        const auto lane = static_cast<uint32_t>(__builtin_ctz(m));
        m &= (m - 1U);
        update_leaf_topk(static_cast<uint32_t>(panel_base + lane),
                         static_cast<uint32_t>(q_idx),
                         tmp_d[lane],
                         neighbor_count,
                         scratch.global_ids,
                         scratch);
    }
}

__attribute__((noinline)) void
flush_leader_mask_hits(uint64_t qi,
                       uint64_t panel_base,
                       uint32_t fanout,
                       uint32_t mask,
                       const float* tmp_d,
                       const VnniScratch& leader_scratch,
                       VnniScratch& worker_scratch) {
    uint32_t m = mask;
    while (m != 0) {
        const auto lane = static_cast<uint32_t>(__builtin_ctz(m));
        m &= (m - 1U);
        update_leader_topk(static_cast<uint32_t>(qi),
                           static_cast<uint32_t>(panel_base + lane),
                           tmp_d[lane],
                           fanout,
                           leader_scratch.global_ids,
                           worker_scratch);
    }
}

// =============================================================================
// x86-64 AVX-512BW / AVX-512 VNNI Kernels
// =============================================================================
#if defined(__x86_64__) || defined(_M_X64)

__attribute__((always_inline)) inline void
prefetch_vec_u8(const uint8_t* ptr, uint64_t dim_pad4) {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
    if (dim_pad4 > 64) {
        _mm_prefetch(reinterpret_cast<const char*>(ptr + 64), _MM_HINT_T0);
    }
    if (dim_pad4 > 128) {
        _mm_prefetch(reinterpret_cast<const char*>(ptr + 128), _MM_HINT_T0);
    }
}

__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq"))) inline void
copy_vec_u8_avx512(const uint8_t* src, uint8_t* dst, uint64_t bytes) {
    uint64_t i = 0;
    for (; i + 64 <= bytes; i += 64) {
        _mm512_storeu_si512(reinterpret_cast<void*>(dst + i),
                            _mm512_loadu_si512(reinterpret_cast<const void*>(src + i)));
    }
    if (i < bytes) {
        std::memcpy(dst + i, src + i, bytes - i);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint16_t
relative_hash16_avx512(const float* sketch_u,
                       const float* sketch_v,
                       uint64_t planes,
                       uint16_t coincident_flag) {
    const __m512 su = _mm512_load_ps(sketch_u);
    const __m512 sv = _mm512_load_ps(sketch_v);
    const auto plane_mask = static_cast<uint16_t>((1U << planes) - 1U);
    uint16_t ge_mask = static_cast<uint16_t>(_mm512_cmp_ps_mask(sv, su, _CMP_GE_OQ)) & plane_mask;
    const uint16_t eq_mask =
        static_cast<uint16_t>(_mm512_cmp_ps_mask(sv, su, _CMP_EQ_OQ)) & plane_mask;
    if (eq_mask == plane_mask) [[unlikely]] {
        ge_mask |= coincident_flag;
    }
    return ge_mask;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint16_t
relative_hash16_u16_avx512(const uint16_t* sketch_u,
                           const uint16_t* sketch_v,
                           uint64_t planes,
                           uint16_t coincident_flag) {
    const __m256i su = _mm256_load_si256(reinterpret_cast<const __m256i*>(sketch_u));
    const __m256i sv = _mm256_load_si256(reinterpret_cast<const __m256i*>(sketch_v));
    const auto plane_mask = static_cast<uint16_t>((1U << planes) - 1U);
    uint16_t ge_mask = static_cast<uint16_t>(_mm256_cmpge_epu16_mask(sv, su)) & plane_mask;
    const uint16_t eq_mask = static_cast<uint16_t>(_mm256_cmpeq_epu16_mask(sv, su)) & plane_mask;
    if (eq_mask == plane_mask) [[unlikely]] {
        ge_mask |= coincident_flag;
    }
    return ge_mask;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint32_t
group_and_hash_leaf_edges_fp32_avx512(const uint32_t* leaf_points,
                                      uint64_t point_count,
                                      uint32_t neighbor_count,
                                      const uint32_t* topk_idx,
                                      const float* topk_dists,
                                      const uint16_t* topk_count,
                                      const float* global_sketches_f32,
                                      uint64_t hash_planes,
                                      uint16_t coincident_flag,
                                      VnniScratch& scratch,
                                      uint32_t* offsets,
                                      uint32_t* g_targets,
                                      uint16_t* g_dist_keys,
                                      uint16_t* g_hashes) {
    scratch.EnsureLeafSketchesF32(point_count * 16);
    float* leaf_sk = scratch.leaf_sketches_f32;
    for (uint64_t i = 0; i < point_count; ++i) {
        if (i + 8 < point_count) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             global_sketches_f32 + static_cast<uint64_t>(leaf_points[i + 8]) * 16),
                         _MM_HINT_T0);
        }
        const __m512 sk =
            _mm512_load_ps(global_sketches_f32 + static_cast<uint64_t>(leaf_points[i]) * 16);
        _mm512_store_ps(leaf_sk + i * 16, sk);
    }

    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;

    const auto plane_mask = static_cast<uint16_t>((1U << hash_planes) - 1U);
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const __m512 su = _mm512_load_ps(leaf_sk + src * 16);
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const __m512 sv = _mm512_load_ps(leaf_sk + static_cast<size_t>(dst_local) * 16);
            uint16_t fwd_h =
                static_cast<uint16_t>(_mm512_cmp_ps_mask(sv, su, _CMP_GE_OQ)) & plane_mask;
            uint16_t rev_h =
                static_cast<uint16_t>(_mm512_cmp_ps_mask(su, sv, _CMP_GE_OQ)) & plane_mask;
            const uint16_t eq =
                static_cast<uint16_t>(_mm512_cmp_ps_mask(sv, su, _CMP_EQ_OQ)) & plane_mask;
            if (eq == plane_mask) [[unlikely]] {
                fwd_h |= coincident_flag;
                rev_h |= coincident_flag;
            }
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint32_t
group_and_hash_leaf_edges_u16_avx512(const uint32_t* leaf_points,
                                     uint64_t point_count,
                                     uint32_t neighbor_count,
                                     const uint32_t* topk_idx,
                                     const float* topk_dists,
                                     const uint16_t* topk_count,
                                     const uint16_t* global_sketches_u16,
                                     uint64_t hash_planes,
                                     uint16_t coincident_flag,
                                     VnniScratch& scratch,
                                     uint32_t* offsets,
                                     uint32_t* g_targets,
                                     uint16_t* g_dist_keys,
                                     uint16_t* g_hashes) {
    scratch.EnsureLeafSketchesU16(point_count * 16);
    uint16_t* leaf_sk = scratch.leaf_sketches_u16;
    for (uint64_t i = 0; i < point_count; ++i) {
        if (i + 8 < point_count) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             global_sketches_u16 + static_cast<uint64_t>(leaf_points[i + 8]) * 16),
                         _MM_HINT_T0);
        }
        const __m256i sk = _mm256_load_si256(reinterpret_cast<const __m256i*>(
            global_sketches_u16 + static_cast<uint64_t>(leaf_points[i]) * 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(leaf_sk + i * 16), sk);
    }

    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;

    const auto plane_mask = static_cast<uint16_t>((1U << hash_planes) - 1U);
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const __m256i su = _mm256_load_si256(reinterpret_cast<const __m256i*>(leaf_sk + src * 16));
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const __m256i sv = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(leaf_sk + static_cast<size_t>(dst_local) * 16));
            uint16_t fwd_h = static_cast<uint16_t>(_mm256_cmpge_epu16_mask(sv, su)) & plane_mask;
            uint16_t rev_h = static_cast<uint16_t>(_mm256_cmpge_epu16_mask(su, sv)) & plane_mask;
            const uint16_t eq = static_cast<uint16_t>(_mm256_cmpeq_epu16_mask(sv, su)) & plane_mask;
            if (eq == plane_mask) [[unlikely]] {
                fwd_h |= coincident_flag;
                rev_h |= coincident_flag;
            }
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

template <MetricKind metric_kind>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) void
select_leaf_topk_from_dots_fp32_avx512_impl(const uint32_t* leaf_points,
                                            uint64_t point_count,
                                            const float* dot_matrix,
                                            const float* global_norms,
                                            const uint32_t* global_ids_map,
                                            bool ids_is_identity,
                                            uint32_t neighbor_count,
                                            VnniScratch& scratch) {
    const uint64_t padded_pts = ((point_count + 15ULL) & ~15ULL);
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, neighbor_count);

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t pid = leaf_points[i];
        scratch.norms[i] = (global_norms != nullptr) ? global_norms[pid] : 0.0F;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.farthest_dist[i] = SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }
    for (uint64_t i = point_count; i < padded_pts; ++i) {
        scratch.norms[i] = 0.0F;
        scratch.global_ids[i] = 0;
    }

    const __m512 v_zero = _mm512_setzero_ps();
    const __m512 v_one = _mm512_set1_ps(1.0F);
    const __m512 v_neg_one = _mm512_set1_ps(-1.0F);
    const __m512 v_inf = _mm512_set1_ps(SENTINEL);

    for (uint64_t src = 0; src < point_count; ++src) {
        const float* row = dot_matrix + src * point_count;
        const float n_src = scratch.norms[src];
        const __m512 v_nsrc = _mm512_set1_ps(n_src);
        __m512 v_far = _mm512_set1_ps(SENTINEL);

        uint64_t j = 0;
        for (; j + 16 <= point_count; j += 16) {
            const __m512 v_dot = _mm512_loadu_ps(row + j);
            __m512 v_d;
            if constexpr (metric_kind == MetricKind::L2_SQR) {
                const __m512 v_sum = _mm512_add_ps(v_nsrc, _mm512_load_ps(scratch.norms + j));
                v_d = _mm512_max_ps(v_zero, _mm512_sub_ps(_mm512_sub_ps(v_sum, v_dot), v_dot));
            } else if constexpr (metric_kind == MetricKind::IP) {
                v_d = _mm512_sub_ps(v_one, v_dot);
            } else {
                const __m512 v_mul =
                    _mm512_mul_ps(_mm512_mul_ps(v_dot, v_nsrc), _mm512_load_ps(scratch.norms + j));
                const __m512 v_sim = _mm512_min_ps(v_one, _mm512_max_ps(v_neg_one, v_mul));
                v_d = _mm512_sub_ps(v_one, v_sim);
            }
            if (src >= j and src < j + 16) {
                const auto self_m = static_cast<__mmask16>(1U << (src - j));
                v_d = _mm512_mask_mov_ps(v_d, self_m, v_inf);
            }
            uint32_t mask = (scratch.topk_count[src] < neighbor_count or
                             not is_finite_f32(scratch.farthest_dist[src]))
                                ? 0xFFFFU
                                : static_cast<uint32_t>(_mm512_cmp_ps_mask(v_d, v_far, _CMP_LE_OQ));
            if (mask != 0) {
                alignas(64) float tmp_d[16];
                _mm512_store_ps(tmp_d, v_d);
                while (mask != 0) {
                    const auto lane = static_cast<uint32_t>(__builtin_ctz(mask));
                    mask &= (mask - 1U);
                    const uint64_t cand = j + lane;
                    if (cand == src) {
                        continue;
                    }
                    update_leaf_topk(static_cast<uint32_t>(src),
                                     static_cast<uint32_t>(cand),
                                     sanitize_distance(tmp_d[lane]),
                                     neighbor_count,
                                     scratch.global_ids,
                                     scratch);
                }
                v_far = _mm512_set1_ps(scratch.farthest_dist[src]);
            }
        }
        for (; j < point_count; ++j) {
            if (j == src) {
                continue;
            }
            const float d =
                finalize_scalar_distance_fp32(row[j], n_src, scratch.norms[j], metric_kind);
            if (scratch.topk_count[src] < neighbor_count or d <= scratch.farthest_dist[src] or
                not is_finite_f32(scratch.farthest_dist[src])) {
                update_leaf_topk(static_cast<uint32_t>(src),
                                 static_cast<uint32_t>(j),
                                 d,
                                 neighbor_count,
                                 scratch.global_ids,
                                 scratch);
            }
        }
        sort_leaf_topk_row(static_cast<uint32_t>(src), neighbor_count, scratch.global_ids, scratch);
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint16_t
find_hash_u16_avx512(const uint16_t* hashes, uint16_t deg, uint16_t target) {
    const __m512i t = _mm512_set1_epi16(static_cast<int16_t>(target));
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 32U <= deg; i = static_cast<uint16_t>(i + 32U)) {
        const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(hashes + i));
        const __mmask32 m = _mm512_cmpeq_epi16_mask(v, t);
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m)));
        }
    }
    if (i < deg) {
        const uint32_t rem = deg - i;
        const __mmask32 valid = (rem == 32U) ? 0xFFFFFFFFU : ((1U << rem) - 1U);
        const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(hashes + i));
        const __mmask32 m = _mm512_cmpeq_epi16_mask(v, t) & valid;
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m)));
        }
    }
    return deg;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint16_t
find_id_u32_avx512(const uint32_t* ids, uint16_t deg, uint32_t target) {
    const __m512i t = _mm512_set1_epi32(static_cast<int32_t>(target));
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 16U <= deg; i = static_cast<uint16_t>(i + 16U)) {
        const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(ids + i));
        const __mmask16 m = _mm512_cmpeq_epi32_mask(v, t);
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m)));
        }
    }
    if (i < deg) {
        const uint32_t rem = deg - i;
        const auto valid = static_cast<__mmask16>((1U << rem) - 1U);
        const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(ids + i));
        const __mmask16 m = _mm512_cmpeq_epi32_mask(v, t) & valid;
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m)));
        }
    }
    return deg;
}

__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq"))) inline uint16_t
horizontal_max_u16_avx512(__m512i v) {
    const __m256i lo256 = _mm512_castsi512_si256(v);
    const __m256i hi256 = _mm512_extracti64x4_epi64(v, 1);
    const __m256i m256 = _mm256_max_epu16(lo256, hi256);
    const __m128i lo128 = _mm256_castsi256_si128(m256);
    const __m128i hi128 = _mm256_extracti128_si256(m256, 1);
    const __m128i m128 = _mm_max_epu16(lo128, hi128);
    const __m128i inv = _mm_xor_si128(m128, _mm_set1_epi16(-1));
    const __m128i minpos = _mm_minpos_epu16(inv);
    return static_cast<uint16_t>(~static_cast<uint32_t>(_mm_cvtsi128_si32(minpos)) & 0xFFFFU);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) uint16_t
find_farthest_slot_u16_avx512(const uint16_t* dists,
                              const uint16_t* hashes,
                              const uint32_t* neighbors,
                              const uint32_t* ids,
                              bool ids_is_identity,
                              uint16_t deg,
                              uint16_t& out_farthest_dist) {
    if (deg <= 64) {
        const __mmask32 m0_valid =
            (deg >= 32) ? 0xFFFFFFFFU : static_cast<__mmask32>((1U << deg) - 1U);
        const __m512i v0 = _mm512_maskz_loadu_epi16(m0_valid, reinterpret_cast<const void*>(dists));
        __m512i vmax = v0;
        __m512i v1 = _mm512_setzero_si512();
        __mmask32 m1_valid = 0;
        if (deg > 32) {
            const uint32_t rem = deg - 32U;
            m1_valid = (rem == 32U) ? 0xFFFFFFFFU : static_cast<__mmask32>((1U << rem) - 1U);
            v1 = _mm512_maskz_loadu_epi16(m1_valid, reinterpret_cast<const void*>(dists + 32));
            vmax = _mm512_max_epu16(v0, v1);
        }
        const uint16_t max_d = horizontal_max_u16_avx512(vmax);
        out_farthest_dist = max_d;
        const __m512i vt = _mm512_set1_epi16(static_cast<int16_t>(max_d));
        const auto eq0 = static_cast<uint64_t>(_mm512_cmpeq_epi16_mask(v0, vt) & m0_valid);
        const uint64_t eq1 =
            (deg > 32) ? static_cast<uint64_t>(_mm512_cmpeq_epi16_mask(v1, vt) & m1_valid) : 0ULL;
        uint64_t eq_mask = eq0 | (eq1 << 32);
        if ((eq_mask & (eq_mask - 1ULL)) == 0) [[likely]] {
            return static_cast<uint16_t>(__builtin_ctzll(eq_mask));
        }
        auto best_idx = static_cast<uint16_t>(__builtin_ctzll(eq_mask));
        eq_mask &= (eq_mask - 1ULL);
        uint32_t far_id = ids_is_identity ? neighbors[best_idx] : ids[neighbors[best_idx]];
        uint16_t far_hash = hashes[best_idx];
        while (eq_mask != 0) {
            const auto idx = static_cast<uint16_t>(__builtin_ctzll(eq_mask));
            eq_mask &= (eq_mask - 1ULL);
            const uint32_t cid = ids_is_identity ? neighbors[idx] : ids[neighbors[idx]];
            const uint16_t chash = hashes[idx];
            if (cid > far_id or (cid == far_id and chash > far_hash)) {
                best_idx = idx;
                far_id = cid;
                far_hash = chash;
            }
        }
        return best_idx;
    }

    __m512i vmax = _mm512_setzero_si512();
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 32U <= deg; i = static_cast<uint16_t>(i + 32U)) {
        vmax = _mm512_max_epu16(vmax, _mm512_load_si512(reinterpret_cast<const void*>(dists + i)));
    }
    if (i < deg) {
        const uint32_t rem = deg - i;
        const auto valid = static_cast<__mmask32>((1U << rem) - 1U);
        vmax = _mm512_max_epu16(
            vmax, _mm512_maskz_loadu_epi16(valid, reinterpret_cast<const void*>(dists + i)));
    }
    const uint16_t max_d = horizontal_max_u16_avx512(vmax);
    out_farthest_dist = max_d;
    const __m512i vt = _mm512_set1_epi16(static_cast<int16_t>(max_d));

    uint16_t best_idx = 0;
    bool have_best = false;
    uint32_t far_id = 0;
    uint16_t far_hash = 0;
    for (uint16_t base = 0; base < deg; base = static_cast<uint16_t>(base + 32U)) {
        const uint32_t rem = std::min<uint32_t>(32U, deg - base);
        const __mmask32 valid =
            (rem == 32U) ? 0xFFFFFFFFU : static_cast<__mmask32>((1U << rem) - 1U);
        const __m512i v = _mm512_load_si512(reinterpret_cast<const void*>(dists + base));
        uint32_t m = _mm512_cmpeq_epi16_mask(v, vt) & valid;
        while (m != 0) {
            const auto idx = static_cast<uint16_t>(base + static_cast<uint16_t>(__builtin_ctz(m)));
            m &= (m - 1U);
            const uint32_t cid = ids_is_identity ? neighbors[idx] : ids[neighbors[idx]];
            const uint16_t chash = hashes[idx];
            if (not have_best or cid > far_id or (cid == far_id and chash > far_hash)) {
                best_idx = idx;
                far_id = cid;
                far_hash = chash;
                have_best = true;
            }
        }
    }
    return best_idx;
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) void
quantize_row_sq8_local_avx512(const float* src,
                              uint64_t dim,
                              uint64_t stride,
                              float pre_scale,
                              MetricKind metric,
                              uint8_t* dst_u8,
                              QMeta& out_meta) {
    if (dim == 0 or not is_finite_f32(pre_scale)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const __m512 vpre = _mm512_set1_ps(pre_scale);
    __m512 vmin = _mm512_set1_ps(SENTINEL);
    __m512 vmax = _mm512_set1_ps(-SENTINEL);
    __mmask16 nan_mask = 0;
    uint64_t k = 0;
    for (; k + 16 <= dim; k += 16) {
        const __m512 v = _mm512_mul_ps(_mm512_loadu_ps(src + k), vpre);
        nan_mask = static_cast<__mmask16>(nan_mask | _mm512_cmp_ps_mask(v, v, _CMP_UNORD_Q));
        vmin = _mm512_min_ps(vmin, v);
        vmax = _mm512_max_ps(vmax, v);
    }
    bool has_non_finite = (nan_mask != 0);
    float min_val = (k > 0) ? _mm512_reduce_min_ps(vmin) : (src[0] * pre_scale);
    float max_val = (k > 0) ? _mm512_reduce_max_ps(vmax) : (src[0] * pre_scale);
    for (; k < dim and not has_non_finite; ++k) {
        const float v = src[k] * pre_scale;
        if (not is_finite_f32(v)) {
            has_non_finite = true;
            break;
        }
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }
    if (has_non_finite or not is_finite_f32(min_val) or not is_finite_f32(max_val)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    const auto step = static_cast<float>(range / 255.0);
    if (range <= 0.0 or step <= 0.0F or not is_finite_f32(step)) {
        std::memset(dst_u8, 0, stride);
        finalize_quantize_meta(dim, min_val, 0.0F, 0, 0, metric, out_meta);
        return;
    }

    const auto inv_step = static_cast<float>(255.0 / range);
    const __m512 vmin_v = _mm512_set1_ps(min_val);
    const __m512 vinv = _mm512_set1_ps(inv_step);
    const __m512 vzero = _mm512_setzero_ps();
    const __m512 v255 = _mm512_set1_ps(255.0F);
    __m512i vsum = _mm512_setzero_si512();
    __m512i vsum2 = _mm512_setzero_si512();

    k = 0;
    for (; k + 16 <= dim; k += 16) {
        __m512 v = _mm512_sub_ps(_mm512_mul_ps(_mm512_loadu_ps(src + k), vpre), vmin_v);
        v = _mm512_mul_ps(v, vinv);
        v = _mm512_min_ps(v255, _mm512_max_ps(vzero, v));
        const __m512i vi = _mm512_cvtps_epi32(v);
        vsum = _mm512_add_epi32(vsum, vi);
        vsum2 = _mm512_add_epi32(vsum2, _mm512_mullo_epi32(vi, vi));
        const __m128i vb = _mm512_cvtusepi32_epi8(vi);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_u8 + k), vb);
    }
    alignas(64) uint32_t s1_lanes[16];
    alignas(64) uint32_t s2_lanes[16];
    _mm512_store_si512(reinterpret_cast<void*>(s1_lanes), vsum);
    _mm512_store_si512(reinterpret_cast<void*>(s2_lanes), vsum2);
    uint64_t sum_u = 0;
    uint64_t sum_u2 = 0;
    for (uint32_t l = 0; l < 16; ++l) {
        sum_u += s1_lanes[l];
        sum_u2 += s2_lanes[l];
    }
    for (; k < dim; ++k) {
        const float scaled = std::clamp((src[k] * pre_scale - min_val) * inv_step, 0.0F, 255.0F);
        const auto q = static_cast<uint32_t>(std::nearbyint(scaled));
        dst_u8[k] = static_cast<uint8_t>(q);
        sum_u += q;
        sum_u2 += static_cast<uint64_t>(q) * static_cast<uint64_t>(q);
    }
    if (dim < stride) {
        std::memset(dst_u8 + dim, 0, stride - dim);
    }
    finalize_quantize_meta(dim, min_val, step, sum_u, sum_u2, metric, out_meta);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) float
quantized_pair_distance_avx512_vnni(const uint8_t* qa_u8,
                                    const QMeta& ma,
                                    const uint8_t* qb_u8,
                                    const QMeta& mb,
                                    uint64_t dim,
                                    uint64_t dim_pad4,
                                    MetricKind metric) {
    const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
    __m512i acc = _mm512_setzero_si512();
    uint64_t k = 0;
    for (; k + 64 <= dim_pad4; k += 64) {
        const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(qa_u8 + k));
        const __m512i vb =
            _mm512_xor_si512(_mm512_loadu_si512(reinterpret_cast<const void*>(qb_u8 + k)), flip);
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    int32_t sum = _mm512_reduce_add_epi32(acc);
    for (; k < dim_pad4; ++k) {
        const auto sb = static_cast<int8_t>(qb_u8[k] ^ 0x80U);
        sum += static_cast<int32_t>(qa_u8[k]) * static_cast<int32_t>(sb);
    }
    const float c_b = mb.min_val + 128.0F * mb.step;
    const float sum_x_b = static_cast<float>(dim) * mb.min_val + mb.su;
    const float dot =
        sq8_dot_from_signed_acc(sum, ma.step, ma.min_val, ma.su, mb.step, c_b, sum_x_b);
    return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
}

template <MetricKind metric_kind>
__attribute__((always_inline,
               target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline __m512
compute_panel_distance_avx512_t(__m512i acc_i32,
                                __m512 v_c_b,
                                __m512 v_sum_x_b,
                                __m512 v_step_b,
                                __m512 v_norm_b,
                                float step_q,
                                float min_q,
                                float su_q,
                                float norm_q) {
    __m512 dot =
        _mm512_mul_ps(_mm512_cvtepi32_ps(acc_i32), _mm512_mul_ps(_mm512_set1_ps(step_q), v_step_b));
    dot = _mm512_fmadd_ps(_mm512_set1_ps(su_q), v_c_b, dot);
    dot = _mm512_fmadd_ps(_mm512_set1_ps(min_q), v_sum_x_b, dot);
    if constexpr (metric_kind == MetricKind::L2_SQR) {
        const __m512 sum = _mm512_add_ps(_mm512_set1_ps(norm_q), v_norm_b);
        const __m512 d = _mm512_fnmadd_ps(_mm512_set1_ps(2.0F), dot, sum);
        return _mm512_max_ps(_mm512_setzero_ps(), d);
    } else if constexpr (metric_kind == MetricKind::IP) {
        return _mm512_sub_ps(_mm512_set1_ps(1.0F), dot);
    } else {
        const __m512 sim =
            _mm512_min_ps(_mm512_set1_ps(1.0F), _mm512_max_ps(_mm512_set1_ps(-1.0F), dot));
        return _mm512_sub_ps(_mm512_set1_ps(1.0F), sim);
    }
}

// Pure register-only branchless AVX-512 per-lane Top-K insertion (zero memory ops!).
template <uint32_t k_val>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert_reg(__m512 (&d)[k_val], __m512i (&idx)[k_val], __m512 dv, __m512i iv) {
    const __mmask16 lt_last = _mm512_cmp_ps_mask(dv, d[k_val - 1], _CMP_LT_OQ);
    if (lt_last == 0) {
        return;
    }
    if constexpr (k_val == 1) {
        d[0] = _mm512_mask_mov_ps(d[0], lt_last, dv);
        idx[0] = _mm512_mask_mov_epi32(idx[0], lt_last, iv);
    } else if constexpr (k_val == 2) {
        const __mmask16 lt0 = _mm512_cmp_ps_mask(dv, d[0], _CMP_LT_OQ);
        d[1] = _mm512_mask_mov_ps(d[1], lt_last, _mm512_mask_mov_ps(dv, lt0, d[0]));
        idx[1] = _mm512_mask_mov_epi32(idx[1], lt_last, _mm512_mask_mov_epi32(iv, lt0, idx[0]));
        d[0] = _mm512_mask_mov_ps(d[0], lt0, dv);
        idx[0] = _mm512_mask_mov_epi32(idx[0], lt0, iv);
    } else {
        __mmask16 lt[k_val];
        for (uint32_t s = 0; s + 1 < k_val; ++s) {
            lt[s] = _mm512_cmp_ps_mask(dv, d[s], _CMP_LT_OQ);
        }
        lt[k_val - 1] = lt_last;
        for (uint32_t s = k_val - 1; s > 0; --s) {
            d[s] = _mm512_mask_mov_ps(d[s], lt[s], _mm512_mask_mov_ps(dv, lt[s - 1], d[s - 1]));
            idx[s] = _mm512_mask_mov_epi32(
                idx[s], lt[s], _mm512_mask_mov_epi32(iv, lt[s - 1], idx[s - 1]));
        }
        d[0] = _mm512_mask_mov_ps(d[0], lt[0], dv);
        idx[0] = _mm512_mask_mov_epi32(idx[0], lt[0], iv);
    }
}

template <uint32_t k_val>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert_const(float* __restrict d_base,
                       uint32_t* __restrict i_base,
                       __m512 dv,
                       __m512i iv) {
    const __m512 d_last = _mm512_load_ps(d_base + static_cast<size_t>(k_val - 1) * 16);
    if (_mm512_cmp_ps_mask(dv, d_last, _CMP_LT_OQ) == 0) {
        return;
    }
    __m512 d[k_val];
    __m512i idx[k_val];
    for (uint32_t s = 0; s + 1 < k_val; ++s) {
        d[s] = _mm512_load_ps(d_base + static_cast<size_t>(s) * 16);
        idx[s] =
            _mm512_load_si512(reinterpret_cast<const void*>(i_base + static_cast<size_t>(s) * 16));
    }
    d[k_val - 1] = d_last;
    idx[k_val - 1] = _mm512_load_si512(
        reinterpret_cast<const void*>(i_base + static_cast<size_t>(k_val - 1) * 16));
    simd_topk_insert_reg<k_val>(d, idx, dv, iv);
    for (uint32_t s = 0; s < k_val; ++s) {
        _mm512_store_ps(d_base + static_cast<size_t>(s) * 16, d[s]);
        _mm512_store_si512(reinterpret_cast<void*>(i_base + static_cast<size_t>(s) * 16), idx[s]);
    }
}

__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert_dyn(
    float* __restrict d_base, uint32_t* __restrict i_base, uint32_t k, __m512 dv, __m512i iv) {
    const __m512 d_last = _mm512_load_ps(d_base + static_cast<size_t>(k - 1) * 16);
    const __mmask16 lt_last = _mm512_cmp_ps_mask(dv, d_last, _CMP_LT_OQ);
    if (lt_last == 0) {
        return;
    }
    __m512 d[8];
    __m512i idx[8];
    __mmask16 lt[8];
    for (uint32_t s = 0; s + 1 < k; ++s) {
        d[s] = _mm512_load_ps(d_base + static_cast<size_t>(s) * 16);
        idx[s] =
            _mm512_load_si512(reinterpret_cast<const void*>(i_base + static_cast<size_t>(s) * 16));
        lt[s] = _mm512_cmp_ps_mask(dv, d[s], _CMP_LT_OQ);
    }
    d[k - 1] = d_last;
    idx[k - 1] =
        _mm512_load_si512(reinterpret_cast<const void*>(i_base + static_cast<size_t>(k - 1) * 16));
    lt[k - 1] = lt_last;

    for (uint32_t s = k - 1; s > 0; --s) {
        const __m512 new_d =
            _mm512_mask_mov_ps(d[s], lt[s], _mm512_mask_mov_ps(dv, lt[s - 1], d[s - 1]));
        const __m512i new_i =
            _mm512_mask_mov_epi32(idx[s], lt[s], _mm512_mask_mov_epi32(iv, lt[s - 1], idx[s - 1]));
        _mm512_store_ps(d_base + static_cast<size_t>(s) * 16, new_d);
        _mm512_store_si512(reinterpret_cast<void*>(i_base + static_cast<size_t>(s) * 16), new_i);
    }
    _mm512_store_ps(d_base, _mm512_mask_mov_ps(d[0], lt[0], dv));
    _mm512_store_si512(reinterpret_cast<void*>(i_base), _mm512_mask_mov_epi32(idx[0], lt[0], iv));
}

template <uint32_t k_const>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert(
    uint32_t k, float* __restrict d_base, uint32_t* __restrict i_base, __m512 dv, __m512i iv) {
    if constexpr (k_const > 0) {
        (void)k;
        simd_topk_insert_const<k_const>(d_base, i_base, dv, iv);
    } else {
        simd_topk_insert_dyn(d_base, i_base, k, dv, iv);
    }
}

template <uint32_t k_const>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert_batch4(uint32_t k,
                        float* __restrict d_base,
                        uint32_t* __restrict i_base,
                        __m512 dv0,
                        __m512 dv1,
                        __m512 dv2,
                        __m512 dv3,
                        __m512i iv0,
                        __m512i iv1,
                        __m512i iv2,
                        __m512i iv3) {
    if constexpr (k_const > 0) {
        (void)k;
        const __m512 d_last = _mm512_load_ps(d_base + static_cast<size_t>(k_const - 1) * 16);
        const __m512 d_min = _mm512_min_ps(_mm512_min_ps(dv0, dv1), _mm512_min_ps(dv2, dv3));
        if (_mm512_cmp_ps_mask(d_min, d_last, _CMP_LT_OQ) == 0) {
            return;
        }
        __m512 d[k_const];
        __m512i idx[k_const];
        for (uint32_t s = 0; s + 1 < k_const; ++s) {
            d[s] = _mm512_load_ps(d_base + static_cast<size_t>(s) * 16);
            idx[s] = _mm512_load_si512(
                reinterpret_cast<const void*>(i_base + static_cast<size_t>(s) * 16));
        }
        d[k_const - 1] = d_last;
        idx[k_const - 1] = _mm512_load_si512(
            reinterpret_cast<const void*>(i_base + static_cast<size_t>(k_const - 1) * 16));
        simd_topk_insert_reg<k_const>(d, idx, dv0, iv0);
        simd_topk_insert_reg<k_const>(d, idx, dv1, iv1);
        simd_topk_insert_reg<k_const>(d, idx, dv2, iv2);
        simd_topk_insert_reg<k_const>(d, idx, dv3, iv3);
        for (uint32_t s = 0; s < k_const; ++s) {
            _mm512_store_ps(d_base + static_cast<size_t>(s) * 16, d[s]);
            _mm512_store_si512(reinterpret_cast<void*>(i_base + static_cast<size_t>(s) * 16),
                               idx[s]);
        }
    } else {
        simd_topk_insert_dyn(d_base, i_base, k, dv0, iv0);
        simd_topk_insert_dyn(d_base, i_base, k, dv1, iv1);
        simd_topk_insert_dyn(d_base, i_base, k, dv2, iv2);
        simd_topk_insert_dyn(d_base, i_base, k, dv3, iv3);
    }
}

template <uint32_t k_const>
__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
simd_topk_insert_batch2(uint32_t k,
                        float* __restrict d_base,
                        uint32_t* __restrict i_base,
                        __m512 dv0,
                        __m512 dv1,
                        __m512i iv0,
                        __m512i iv1) {
    if constexpr (k_const > 0) {
        (void)k;
        const __m512 d_last = _mm512_load_ps(d_base + static_cast<size_t>(k_const - 1) * 16);
        const __m512 d_min = _mm512_min_ps(dv0, dv1);
        if (_mm512_cmp_ps_mask(d_min, d_last, _CMP_LT_OQ) == 0) {
            return;
        }
        __m512 d[k_const];
        __m512i idx[k_const];
        for (uint32_t s = 0; s + 1 < k_const; ++s) {
            d[s] = _mm512_load_ps(d_base + static_cast<size_t>(s) * 16);
            idx[s] = _mm512_load_si512(
                reinterpret_cast<const void*>(i_base + static_cast<size_t>(s) * 16));
        }
        d[k_const - 1] = d_last;
        idx[k_const - 1] = _mm512_load_si512(
            reinterpret_cast<const void*>(i_base + static_cast<size_t>(k_const - 1) * 16));
        simd_topk_insert_reg<k_const>(d, idx, dv0, iv0);
        simd_topk_insert_reg<k_const>(d, idx, dv1, iv1);
        for (uint32_t s = 0; s < k_const; ++s) {
            _mm512_store_ps(d_base + static_cast<size_t>(s) * 16, d[s]);
            _mm512_store_si512(reinterpret_cast<void*>(i_base + static_cast<size_t>(s) * 16),
                               idx[s]);
        }
    } else {
        simd_topk_insert_dyn(d_base, i_base, k, dv0, iv0);
        simd_topk_insert_dyn(d_base, i_base, k, dv1, iv1);
    }
}

__attribute__((always_inline,
               target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline uint16_t
horizontal_reduce_simd_topk(const float* __restrict q_d,
                            const uint32_t* __restrict q_i,
                            uint32_t k,
                            float* __restrict out_d,
                            uint32_t* __restrict out_i) {
    alignas(64) float cur_d_arr[16];
    alignas(64) uint32_t cur_i_arr[16];
    uint8_t ptr[16] = {};
    __m512 cur_d = _mm512_load_ps(q_d);
    _mm512_store_ps(cur_d_arr, cur_d);
    _mm512_store_si512(reinterpret_cast<void*>(cur_i_arr),
                       _mm512_load_si512(reinterpret_cast<const void*>(q_i)));

    uint16_t out_cnt = 0;
    for (uint32_t s = 0; s < k; ++s) {
        const float best_d = _mm512_reduce_min_ps(cur_d);
        if (best_d >= SENTINEL) {
            break;
        }
        const __mmask16 eq = _mm512_cmp_ps_mask(cur_d, _mm512_set1_ps(best_d), _CMP_EQ_OQ);
        const auto lane = static_cast<uint32_t>(__builtin_ctz(eq));
        out_d[out_cnt] = best_d;
        out_i[out_cnt] = cur_i_arr[lane];
        ++out_cnt;

        const auto ns = static_cast<uint8_t>(ptr[lane] + 1U);
        ptr[lane] = ns;
        if (ns < k) {
            cur_d_arr[lane] = q_d[static_cast<size_t>(ns) * 16 + lane];
            cur_i_arr[lane] = q_i[static_cast<size_t>(ns) * 16 + lane];
        } else {
            cur_d_arr[lane] = SENTINEL;
        }
        cur_d = _mm512_load_ps(cur_d_arr);
    }
    return out_cnt;
}

__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
check_and_scatter_leaf_masked_avx512(uint64_t q_idx,
                                     uint64_t panel_base,
                                     uint64_t point_count,
                                     uint32_t neighbor_count,
                                     __m512 d_vec,
                                     VnniScratch& scratch) {
    const uint64_t lo = (q_idx >= panel_base) ? (q_idx - panel_base + 1) : 0;
    const uint64_t hi = std::min<uint64_t>(16, point_count - panel_base);
    if (lo >= hi) {
        return;
    }
    const auto valid_mask = static_cast<uint16_t>(((1U << hi) - 1U) & ~((1U << lo) - 1U));
    const __mmask16 q_mask =
        _mm512_cmp_ps_mask(d_vec, _mm512_set1_ps(scratch.farthest_dist[q_idx]), _CMP_LE_OQ) &
        valid_mask;
    const __mmask16 db_mask =
        _mm512_cmp_ps_mask(d_vec, _mm512_load_ps(scratch.farthest_dist + panel_base), _CMP_LE_OQ) &
        valid_mask;
    if ((q_mask | db_mask) != 0) {
        alignas(64) float tmp_d[16];
        _mm512_store_ps(tmp_d, d_vec);
        flush_leaf_mask_hits(q_idx, panel_base, neighbor_count, q_mask, db_mask, tmp_d, scratch);
    }
}

__attribute__((always_inline, target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) inline void
check_and_update_leader_avx512(uint64_t qi,
                               uint64_t panel_base,
                               uint16_t valid_mask,
                               uint32_t fanout,
                               __m512 d_vec,
                               const VnniScratch& leader_scratch,
                               VnniScratch& worker_scratch) {
    const __mmask16 mask =
        _mm512_cmp_ps_mask(d_vec, _mm512_set1_ps(worker_scratch.farthest_dist[qi]), _CMP_LE_OQ) &
        valid_mask;
    if (mask != 0) {
        alignas(64) float tmp_d[16];
        _mm512_store_ps(tmp_d, d_vec);
        flush_leader_mask_hits(qi, panel_base, fanout, mask, tmp_d, leader_scratch, worker_scratch);
    }
}

// =============================================================================
// Batched (4 x 2-panel) AVX-512 VNNI RobustPrune (asymmetric sq8_local)
// =============================================================================
template <MetricKind metric_kind>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) uint32_t
quantized_robust_prune_panels_avx512_vnni_impl(const uint8_t* q_data_u8,
                                               const QMeta* q_meta,
                                               uint64_t dim,
                                               uint64_t dim_pad4,
                                               uint64_t q_stride,
                                               const uint32_t* ordered_cand_ids,
                                               const float* ordered_cand_dists,
                                               uint32_t num_cands,
                                               uint32_t max_degree,
                                               float alpha,
                                               uint32_t* out_selected) {
    if (num_cands == 0 or max_degree == 0) {
        return 0;
    }

    const auto dim_f = static_cast<float>(dim);
    const uint64_t total_tiles = dim_pad4 / 4;
    const uint32_t max_panels = (max_degree + 15U) / 16U;
    const uint32_t padded_surv = max_panels * 16U;
    const size_t panel_bytes = total_tiles * 64;

    thread_local VnniScratch prune_scratch;
    prune_scratch.EnsurePanels(static_cast<size_t>(max_panels) * panel_bytes);
    prune_scratch.EnsureMeta(padded_surv);

    for (uint32_t i = 0; i < padded_surv; ++i) {
        prune_scratch.steps[i] = 0.0F;
        prune_scratch.c_vals[i] = 0.0F;
        prune_scratch.sum_x[i] = 0.0F;
        prune_scratch.norms[i] = 0.0F;
    }

    auto insert_survivor = [&](uint32_t surv_id, const uint8_t* surv_u8, uint32_t slot) {
        out_selected[slot] = surv_id;
        const uint32_t p = slot / 16U;
        const uint32_t lane = slot % 16U;
        const QMeta& m = q_meta[surv_id];
        prune_scratch.steps[slot] = m.step;
        prune_scratch.c_vals[slot] = m.min_val + 128.0F * m.step;
        prune_scratch.sum_x[slot] = dim_f * m.min_val + m.su;
        prune_scratch.norms[slot] = m.norm;
        auto* panel_u32 = reinterpret_cast<uint32_t*>(prune_scratch.panel_buf +
                                                      static_cast<size_t>(p) * panel_bytes);
        const auto* s_u32 = reinterpret_cast<const uint32_t*>(surv_u8);
        for (uint64_t t = 0; t < total_tiles; ++t) {
            panel_u32[t * 16 + lane] = s_u32[t] ^ FLIP_U32;
        }
    };

    for (uint32_t k = 0; k < 8U and k < num_cands; ++k) {
        const uint32_t pf_id = ordered_cand_ids[k];
        prefetch_vec_u8(q_data_u8 + static_cast<size_t>(pf_id) * q_stride, dim_pad4);
        _mm_prefetch(reinterpret_cast<const char*>(q_meta + pf_id), _MM_HINT_T0);
    }

    uint32_t selected_count = 0;
    {
        const uint32_t c0 = ordered_cand_ids[0];
        insert_survivor(c0, q_data_u8 + static_cast<size_t>(c0) * q_stride, 0);
        selected_count = 1;
        if (selected_count == max_degree) {
            return selected_count;
        }
    }

    const __m512 v_alpha = _mm512_set1_ps(alpha);

    uint32_t c = 1;
    while (c < num_cands) {
        const uint32_t batch = std::min<uint32_t>(4U, num_cands - c);
        for (uint32_t k = 0; k < 4U and c + 8U + k < num_cands; ++k) {
            const uint32_t pf_id = ordered_cand_ids[c + 8U + k];
            prefetch_vec_u8(q_data_u8 + static_cast<size_t>(pf_id) * q_stride, dim_pad4);
            _mm_prefetch(reinterpret_cast<const char*>(q_meta + pf_id), _MM_HINT_T0);
        }

        uint32_t cands[4];
        float src_dists[4];
        QMeta c_meta[4];
        const uint8_t* c_u8[4];
        const int32_t* qp[4];

        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t cid = ordered_cand_ids[c + b];
            cands[b] = cid;
            src_dists[b] = ordered_cand_dists[c + b];
            c_meta[b] = q_meta[cid];
            c_u8[b] = q_data_u8 + static_cast<size_t>(cid) * q_stride;
            qp[b] = reinterpret_cast<const int32_t*>(c_u8[b]);
        }

        uint32_t alive = (1U << batch) - 1U;
        const uint32_t active_panels = (selected_count + 15U) / 16U;

        if (batch == 4U) {
            const __m512 vsd0 = _mm512_set1_ps(src_dists[0]);
            const __m512 vsd1 = _mm512_set1_ps(src_dists[1]);
            const __m512 vsd2 = _mm512_set1_ps(src_dists[2]);
            const __m512 vsd3 = _mm512_set1_ps(src_dists[3]);

            uint32_t p = 0;
            for (; p + 2U <= active_panels and alive != 0; p += 2U) {
                const uint32_t pb0 = p * 16U;
                const uint32_t pb1 = (p + 1U) * 16U;
                const uint32_t v0 = std::min<uint32_t>(16U, selected_count - pb0);
                const uint32_t v1 = std::min<uint32_t>(16U, selected_count - pb1);
                const uint16_t vm0 = (v0 == 16U) ? 0xFFFFU : static_cast<uint16_t>((1U << v0) - 1U);
                const uint16_t vm1 = (v1 == 16U) ? 0xFFFFU : static_cast<uint16_t>((1U << v1) - 1U);
                const int8_t* p0 = prune_scratch.panel_buf + static_cast<size_t>(p) * panel_bytes;
                const int8_t* p1 =
                    prune_scratch.panel_buf + static_cast<size_t>(p + 1U) * panel_bytes;

                __m512i a00 = _mm512_setzero_si512();
                __m512i a01 = _mm512_setzero_si512();
                __m512i a10 = _mm512_setzero_si512();
                __m512i a11 = _mm512_setzero_si512();
                __m512i a20 = _mm512_setzero_si512();
                __m512i a21 = _mm512_setzero_si512();
                __m512i a30 = _mm512_setzero_si512();
                __m512i a31 = _mm512_setzero_si512();

                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const size_t off = t * 64;
                    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const void*>(p0 + off));
                    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const void*>(p1 + off));

                    const __m512i q0 = _mm512_set1_epi32(qp[0][t]);
                    a00 = _mm512_dpbusd_epi32(a00, q0, b0);
                    a01 = _mm512_dpbusd_epi32(a01, q0, b1);

                    const __m512i q1 = _mm512_set1_epi32(qp[1][t]);
                    a10 = _mm512_dpbusd_epi32(a10, q1, b0);
                    a11 = _mm512_dpbusd_epi32(a11, q1, b1);

                    const __m512i q2 = _mm512_set1_epi32(qp[2][t]);
                    a20 = _mm512_dpbusd_epi32(a20, q2, b0);
                    a21 = _mm512_dpbusd_epi32(a21, q2, b1);

                    const __m512i q3 = _mm512_set1_epi32(qp[3][t]);
                    a30 = _mm512_dpbusd_epi32(a30, q3, b0);
                    a31 = _mm512_dpbusd_epi32(a31, q3, b1);
                }

                {
                    const __m512 vc = _mm512_load_ps(prune_scratch.c_vals + pb0);
                    const __m512 vx = _mm512_load_ps(prune_scratch.sum_x + pb0);
                    const __m512 vs = _mm512_load_ps(prune_scratch.steps + pb0);
                    const __m512 vn = _mm512_load_ps(prune_scratch.norms + pb0);
                    if ((alive & 1U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a00,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[0].step,
                                                                              c_meta[0].min_val,
                                                                              c_meta[0].su,
                                                                              c_meta[0].norm)),
                             vsd0,
                             _CMP_LT_OQ) &
                         vm0) != 0) {
                        alive &= ~1U;
                    }
                    if ((alive & 2U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a10,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[1].step,
                                                                              c_meta[1].min_val,
                                                                              c_meta[1].su,
                                                                              c_meta[1].norm)),
                             vsd1,
                             _CMP_LT_OQ) &
                         vm0) != 0) {
                        alive &= ~2U;
                    }
                    if ((alive & 4U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a20,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[2].step,
                                                                              c_meta[2].min_val,
                                                                              c_meta[2].su,
                                                                              c_meta[2].norm)),
                             vsd2,
                             _CMP_LT_OQ) &
                         vm0) != 0) {
                        alive &= ~4U;
                    }
                    if ((alive & 8U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a30,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[3].step,
                                                                              c_meta[3].min_val,
                                                                              c_meta[3].su,
                                                                              c_meta[3].norm)),
                             vsd3,
                             _CMP_LT_OQ) &
                         vm0) != 0) {
                        alive &= ~8U;
                    }
                }
                if (alive != 0) {
                    const __m512 vc = _mm512_load_ps(prune_scratch.c_vals + pb1);
                    const __m512 vx = _mm512_load_ps(prune_scratch.sum_x + pb1);
                    const __m512 vs = _mm512_load_ps(prune_scratch.steps + pb1);
                    const __m512 vn = _mm512_load_ps(prune_scratch.norms + pb1);
                    if ((alive & 1U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a01,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[0].step,
                                                                              c_meta[0].min_val,
                                                                              c_meta[0].su,
                                                                              c_meta[0].norm)),
                             vsd0,
                             _CMP_LT_OQ) &
                         vm1) != 0) {
                        alive &= ~1U;
                    }
                    if ((alive & 2U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a11,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[1].step,
                                                                              c_meta[1].min_val,
                                                                              c_meta[1].su,
                                                                              c_meta[1].norm)),
                             vsd1,
                             _CMP_LT_OQ) &
                         vm1) != 0) {
                        alive &= ~2U;
                    }
                    if ((alive & 4U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a21,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[2].step,
                                                                              c_meta[2].min_val,
                                                                              c_meta[2].su,
                                                                              c_meta[2].norm)),
                             vsd2,
                             _CMP_LT_OQ) &
                         vm1) != 0) {
                        alive &= ~4U;
                    }
                    if ((alive & 8U) != 0 and
                        (_mm512_cmp_ps_mask(
                             _mm512_mul_ps(
                                 v_alpha,
                                 compute_panel_distance_avx512_t<metric_kind>(a31,
                                                                              vc,
                                                                              vx,
                                                                              vs,
                                                                              vn,
                                                                              c_meta[3].step,
                                                                              c_meta[3].min_val,
                                                                              c_meta[3].su,
                                                                              c_meta[3].norm)),
                             vsd3,
                             _CMP_LT_OQ) &
                         vm1) != 0) {
                        alive &= ~8U;
                    }
                }
            }

            if (p < active_panels and alive != 0) {
                const uint32_t pb = p * 16U;
                const uint32_t valid = std::min<uint32_t>(16U, selected_count - pb);
                const uint16_t vm =
                    (valid == 16U) ? 0xFFFFU : static_cast<uint16_t>((1U << valid) - 1U);
                const int8_t* p0 = prune_scratch.panel_buf + static_cast<size_t>(p) * panel_bytes;
                __m512i a0 = _mm512_setzero_si512();
                __m512i a1 = _mm512_setzero_si512();
                __m512i a2 = _mm512_setzero_si512();
                __m512i a3 = _mm512_setzero_si512();
                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const __m512i b0 =
                        _mm512_load_si512(reinterpret_cast<const void*>(p0 + t * 64));
                    a0 = _mm512_dpbusd_epi32(a0, _mm512_set1_epi32(qp[0][t]), b0);
                    a1 = _mm512_dpbusd_epi32(a1, _mm512_set1_epi32(qp[1][t]), b0);
                    a2 = _mm512_dpbusd_epi32(a2, _mm512_set1_epi32(qp[2][t]), b0);
                    a3 = _mm512_dpbusd_epi32(a3, _mm512_set1_epi32(qp[3][t]), b0);
                }
                const __m512 vc = _mm512_load_ps(prune_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(prune_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(prune_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(prune_scratch.norms + pb);
                if ((alive & 1U) != 0 and
                    (_mm512_cmp_ps_mask(_mm512_mul_ps(v_alpha,
                                                      compute_panel_distance_avx512_t<metric_kind>(
                                                          a0,
                                                          vc,
                                                          vx,
                                                          vs,
                                                          vn,
                                                          c_meta[0].step,
                                                          c_meta[0].min_val,
                                                          c_meta[0].su,
                                                          c_meta[0].norm)),
                                        vsd0,
                                        _CMP_LT_OQ) &
                     vm) != 0) {
                    alive &= ~1U;
                }
                if ((alive & 2U) != 0 and
                    (_mm512_cmp_ps_mask(_mm512_mul_ps(v_alpha,
                                                      compute_panel_distance_avx512_t<metric_kind>(
                                                          a1,
                                                          vc,
                                                          vx,
                                                          vs,
                                                          vn,
                                                          c_meta[1].step,
                                                          c_meta[1].min_val,
                                                          c_meta[1].su,
                                                          c_meta[1].norm)),
                                        vsd1,
                                        _CMP_LT_OQ) &
                     vm) != 0) {
                    alive &= ~2U;
                }
                if ((alive & 4U) != 0 and
                    (_mm512_cmp_ps_mask(_mm512_mul_ps(v_alpha,
                                                      compute_panel_distance_avx512_t<metric_kind>(
                                                          a2,
                                                          vc,
                                                          vx,
                                                          vs,
                                                          vn,
                                                          c_meta[2].step,
                                                          c_meta[2].min_val,
                                                          c_meta[2].su,
                                                          c_meta[2].norm)),
                                        vsd2,
                                        _CMP_LT_OQ) &
                     vm) != 0) {
                    alive &= ~4U;
                }
                if ((alive & 8U) != 0 and
                    (_mm512_cmp_ps_mask(_mm512_mul_ps(v_alpha,
                                                      compute_panel_distance_avx512_t<metric_kind>(
                                                          a3,
                                                          vc,
                                                          vx,
                                                          vs,
                                                          vn,
                                                          c_meta[3].step,
                                                          c_meta[3].min_val,
                                                          c_meta[3].su,
                                                          c_meta[3].norm)),
                                        vsd3,
                                        _CMP_LT_OQ) &
                     vm) != 0) {
                    alive &= ~8U;
                }
            }
        } else {
            for (uint32_t p = 0; p < active_panels and alive != 0; ++p) {
                const uint32_t pb = p * 16U;
                const uint32_t valid = std::min<uint32_t>(16U, selected_count - pb);
                const uint16_t vm =
                    (valid == 16U) ? 0xFFFFU : static_cast<uint16_t>((1U << valid) - 1U);
                const int8_t* p0 = prune_scratch.panel_buf + static_cast<size_t>(p) * panel_bytes;
                const __m512 vc = _mm512_load_ps(prune_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(prune_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(prune_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(prune_scratch.norms + pb);
                for (uint32_t b = 0; b < batch; ++b) {
                    if ((alive & (1U << b)) == 0) {
                        continue;
                    }
                    __m512i acc = _mm512_setzero_si512();
                    for (uint64_t t = 0; t < total_tiles; ++t) {
                        const __m512i b0 =
                            _mm512_load_si512(reinterpret_cast<const void*>(p0 + t * 64));
                        acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(qp[b][t]), b0);
                    }
                    const __m512 d = compute_panel_distance_avx512_t<metric_kind>(acc,
                                                                                  vc,
                                                                                  vx,
                                                                                  vs,
                                                                                  vn,
                                                                                  c_meta[b].step,
                                                                                  c_meta[b].min_val,
                                                                                  c_meta[b].su,
                                                                                  c_meta[b].norm);
                    if ((_mm512_cmp_ps_mask(
                             _mm512_mul_ps(v_alpha, d), _mm512_set1_ps(src_dists[b]), _CMP_LT_OQ) &
                         vm) != 0) {
                        alive &= ~(1U << b);
                    }
                }
            }
        }

        uint32_t batch_new[4];
        uint32_t batch_new_cnt = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            if ((alive & (1U << b)) == 0) {
                continue;
            }
            bool pruned_intra = false;
            for (uint32_t s = 0; s < batch_new_cnt; ++s) {
                const uint32_t sb = batch_new[s];
                const float d = quantized_pair_distance_avx512_vnni(
                    c_u8[b], c_meta[b], c_u8[sb], c_meta[sb], dim, dim_pad4, metric_kind);
                if (alpha * d < src_dists[b]) {
                    pruned_intra = true;
                    break;
                }
            }
            if (not pruned_intra) {
                insert_survivor(cands[b], c_u8[b], selected_count);
                batch_new[batch_new_cnt++] = b;
                ++selected_count;
                if (selected_count == max_degree) {
                    return selected_count;
                }
            }
        }
        c += batch;
    }
    return selected_count;
}

// =============================================================================
// Fused Leaf Lower-Triangle 4 x 4 AVX-512 VNNI + 4-Way Register-Resident Top-K
// =============================================================================
template <MetricKind metric_kind, uint32_t k_const>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) void
fused_leaf_topk_simd_avx512_vnni_impl(const uint32_t* leaf_points,
                                      uint64_t point_count,
                                      const uint8_t* q_data_u8,
                                      const QMeta* q_meta,
                                      const uint32_t* global_ids_map,
                                      bool ids_is_identity,
                                      uint64_t dim,
                                      uint64_t dim_pad4,
                                      uint64_t q_stride,
                                      uint32_t neighbor_count,
                                      VnniScratch& scratch) {
    const auto dim_f = static_cast<float>(dim);
    const uint32_t k = (k_const > 0) ? k_const : neighbor_count;
    const uint64_t total_tiles = dim_pad4 / 4;
    const uint64_t num_panels = (point_count + 15) / 16;
    const uint64_t padded_pts = num_panels * 16;
    const size_t panel_bytes = total_tiles * 64;

    scratch.EnsureU8(point_count * dim_pad4);
    scratch.EnsurePanels(num_panels * panel_bytes);
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, k);
    scratch.EnsureDbTopK(num_panels, k);

    for (uint64_t i = 0; i < point_count; ++i) {
        if (i + 6 < point_count) {
            const uint32_t pf = leaf_points[i + 6];
            prefetch_vec_u8(q_data_u8 + static_cast<size_t>(pf) * q_stride, dim_pad4);
            _mm_prefetch(reinterpret_cast<const char*>(q_meta + pf), _MM_HINT_T0);
        }
        const uint32_t pid = leaf_points[i];
        copy_vec_u8_avx512(q_data_u8 + static_cast<size_t>(pid) * q_stride,
                           scratch.u8_buf + i * dim_pad4,
                           dim_pad4);
        const QMeta& m = q_meta[pid];
        scratch.steps[i] = m.step;
        scratch.min_vals[i] = m.min_val;
        scratch.su[i] = m.su;
        scratch.c_vals[i] = m.min_val + 128.0F * m.step;
        scratch.sum_x[i] = dim_f * m.min_val + m.su;
        scratch.norms[i] = m.norm;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.topk_count[i] = 0;
    }
    for (uint64_t i = point_count; i < padded_pts; ++i) {
        scratch.steps[i] = 0.0F;
        scratch.min_vals[i] = 0.0F;
        scratch.su[i] = 0.0F;
        scratch.c_vals[i] = 0.0F;
        scratch.sum_x[i] = 0.0F;
        scratch.norms[i] = SENTINEL;
        scratch.global_ids[i] = 0;
    }

    for (uint64_t p = 0; p < num_panels; ++p) {
        const uint64_t base = p * 16;
        const uint64_t valid = std::min<uint64_t>(16, point_count - base);
        auto* panel_u32 = reinterpret_cast<uint32_t*>(scratch.panel_buf + p * panel_bytes);
        const uint32_t* ptrs[16];
        for (uint64_t lane = 0; lane < valid; ++lane) {
            ptrs[lane] =
                reinterpret_cast<const uint32_t*>(scratch.u8_buf + (base + lane) * dim_pad4);
        }
        for (uint64_t t = 0; t < total_tiles; ++t) {
            uint32_t* dst_tile = panel_u32 + t * 16;
            for (uint64_t lane = 0; lane < valid; ++lane) {
                dst_tile[lane] = ptrs[lane][t] ^ FLIP_U32;
            }
            for (uint64_t lane = valid; lane < 16; ++lane) {
                dst_tile[lane] = 0;
            }
        }
    }

    const __m512 v_inf = _mm512_set1_ps(SENTINEL);
    const __m512i v_invalid = _mm512_set1_epi32(-1);
    const __m512i v_lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const size_t total_db_vecs = num_panels * k;
    for (size_t v = 0; v < total_db_vecs; ++v) {
        _mm512_store_ps(scratch.db_topk_d + v * 16, v_inf);
        _mm512_store_si512(reinterpret_cast<void*>(scratch.db_topk_i + v * 16), v_invalid);
    }

    const uint64_t full_interior_end = point_count / 16;
    alignas(64) float block_d[4][4][16];
    alignas(64) float q_d[4][8][16];
    alignas(64) uint32_t q_i[4][8][16];

    for (uint64_t q0 = 0; q0 < point_count; q0 += 4) {
        const uint64_t mq = std::min<uint64_t>(4, point_count - q0);
        const uint64_t first_panel = q0 / 16;

        for (uint64_t r = 0; r < mq; ++r) {
            for (uint32_t s = 0; s < k; ++s) {
                _mm512_store_ps(q_d[r][s], v_inf);
                _mm512_store_si512(reinterpret_cast<void*>(q_i[r][s]), v_invalid);
            }
        }

        if (mq == 4) {
            const auto* qp0 =
                reinterpret_cast<const int32_t*>(scratch.u8_buf + (q0 + 0) * dim_pad4);
            const auto* qp1 =
                reinterpret_cast<const int32_t*>(scratch.u8_buf + (q0 + 1) * dim_pad4);
            const auto* qp2 =
                reinterpret_cast<const int32_t*>(scratch.u8_buf + (q0 + 2) * dim_pad4);
            const auto* qp3 =
                reinterpret_cast<const int32_t*>(scratch.u8_buf + (q0 + 3) * dim_pad4);
            const float st0 = scratch.steps[q0 + 0];
            const float mn0 = scratch.min_vals[q0 + 0];
            const float su0 = scratch.su[q0 + 0];
            const float nq0 = scratch.norms[q0 + 0];
            const float st1 = scratch.steps[q0 + 1];
            const float mn1 = scratch.min_vals[q0 + 1];
            const float su1 = scratch.su[q0 + 1];
            const float nq1 = scratch.norms[q0 + 1];
            const float st2 = scratch.steps[q0 + 2];
            const float mn2 = scratch.min_vals[q0 + 2];
            const float su2 = scratch.su[q0 + 2];
            const float nq2 = scratch.norms[q0 + 2];
            const float st3 = scratch.steps[q0 + 3];
            const float mn3 = scratch.min_vals[q0 + 3];
            const float su3 = scratch.su[q0 + 3];
            const float nq3 = scratch.norms[q0 + 3];
            const __m512i v_q0 = _mm512_set1_epi32(static_cast<int32_t>(q0 + 0));
            const __m512i v_q1 = _mm512_set1_epi32(static_cast<int32_t>(q0 + 1));
            const __m512i v_q2 = _mm512_set1_epi32(static_cast<int32_t>(q0 + 2));
            const __m512i v_q3 = _mm512_set1_epi32(static_cast<int32_t>(q0 + 3));

            uint64_t p = first_panel;
            while (p < num_panels and p * 16 <= q0 + 3) {
                const uint64_t pb = p * 16;
                const int8_t* p_ptr = scratch.panel_buf + p * panel_bytes;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const __m512i b =
                        _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(qp0[t]), b);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(qp1[t]), b);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(qp2[t]), b);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(qp3[t]), b);
                }
                const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                const uint64_t hi = std::min<uint64_t>(16, point_count - pb);
                auto mask_for = [&](uint64_t qi) -> __mmask16 {
                    const uint64_t lo = (qi >= pb) ? (qi - pb + 1) : 0;
                    if (lo >= hi) {
                        return 0;
                    }
                    return static_cast<__mmask16>(((1U << hi) - 1U) & ~((1U << lo) - 1U));
                };
                const __m512 dv0 =
                    _mm512_mask_mov_ps(v_inf,
                                       mask_for(q0 + 0),
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc0, vc, vx, vs, vn, st0, mn0, su0, nq0));
                const __m512 dv1 =
                    _mm512_mask_mov_ps(v_inf,
                                       mask_for(q0 + 1),
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc1, vc, vx, vs, vn, st1, mn1, su1, nq1));
                const __m512 dv2 =
                    _mm512_mask_mov_ps(v_inf,
                                       mask_for(q0 + 2),
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc2, vc, vx, vs, vn, st2, mn2, su2, nq2));
                const __m512 dv3 =
                    _mm512_mask_mov_ps(v_inf,
                                       mask_for(q0 + 3),
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc3, vc, vx, vs, vn, st3, mn3, su3, nq3));

                const __m512i v_db_idx =
                    _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(pb)), v_lanes);
                simd_topk_insert<k_const>(k, q_d[0][0], q_i[0][0], dv0, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[1][0], q_i[1][0], dv1, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[2][0], q_i[2][0], dv2, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[3][0], q_i[3][0], dv3, v_db_idx);
                simd_topk_insert_batch4<k_const>(k,
                                                 scratch.db_topk_d + p * k * 16,
                                                 scratch.db_topk_i + p * k * 16,
                                                 dv0,
                                                 dv1,
                                                 dv2,
                                                 dv3,
                                                 v_q0,
                                                 v_q1,
                                                 v_q2,
                                                 v_q3);
                ++p;
            }

            for (; p + 4 <= full_interior_end; p += 4) {
                const int8_t* p0 = scratch.panel_buf + (p + 0) * panel_bytes;
                const int8_t* p1 = scratch.panel_buf + (p + 1) * panel_bytes;
                const int8_t* p2 = scratch.panel_buf + (p + 2) * panel_bytes;
                const int8_t* p3 = scratch.panel_buf + (p + 3) * panel_bytes;

                __m512i c00 = _mm512_setzero_si512();
                __m512i c01 = _mm512_setzero_si512();
                __m512i c02 = _mm512_setzero_si512();
                __m512i c03 = _mm512_setzero_si512();
                __m512i c10 = _mm512_setzero_si512();
                __m512i c11 = _mm512_setzero_si512();
                __m512i c12 = _mm512_setzero_si512();
                __m512i c13 = _mm512_setzero_si512();
                __m512i c20 = _mm512_setzero_si512();
                __m512i c21 = _mm512_setzero_si512();
                __m512i c22 = _mm512_setzero_si512();
                __m512i c23 = _mm512_setzero_si512();
                __m512i c30 = _mm512_setzero_si512();
                __m512i c31 = _mm512_setzero_si512();
                __m512i c32 = _mm512_setzero_si512();
                __m512i c33 = _mm512_setzero_si512();

                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const size_t off = t * 64;
                    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const void*>(p0 + off));
                    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const void*>(p1 + off));
                    const __m512i b2 = _mm512_load_si512(reinterpret_cast<const void*>(p2 + off));
                    const __m512i b3 = _mm512_load_si512(reinterpret_cast<const void*>(p3 + off));

                    const __m512i qv0 = _mm512_set1_epi32(qp0[t]);
                    c00 = _mm512_dpbusd_epi32(c00, qv0, b0);
                    c01 = _mm512_dpbusd_epi32(c01, qv0, b1);
                    c02 = _mm512_dpbusd_epi32(c02, qv0, b2);
                    c03 = _mm512_dpbusd_epi32(c03, qv0, b3);

                    const __m512i qv1 = _mm512_set1_epi32(qp1[t]);
                    c10 = _mm512_dpbusd_epi32(c10, qv1, b0);
                    c11 = _mm512_dpbusd_epi32(c11, qv1, b1);
                    c12 = _mm512_dpbusd_epi32(c12, qv1, b2);
                    c13 = _mm512_dpbusd_epi32(c13, qv1, b3);

                    const __m512i qv2 = _mm512_set1_epi32(qp2[t]);
                    c20 = _mm512_dpbusd_epi32(c20, qv2, b0);
                    c21 = _mm512_dpbusd_epi32(c21, qv2, b1);
                    c22 = _mm512_dpbusd_epi32(c22, qv2, b2);
                    c23 = _mm512_dpbusd_epi32(c23, qv2, b3);

                    const __m512i qv3 = _mm512_set1_epi32(qp3[t]);
                    c30 = _mm512_dpbusd_epi32(c30, qv3, b0);
                    c31 = _mm512_dpbusd_epi32(c31, qv3, b1);
                    c32 = _mm512_dpbusd_epi32(c32, qv3, b2);
                    c33 = _mm512_dpbusd_epi32(c33, qv3, b3);
                }

                {
                    const uint64_t pb = (p + 0) * 16;
                    const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                    _mm512_store_ps(block_d[0][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c00, vc, vx, vs, vn, st0, mn0, su0, nq0));
                    _mm512_store_ps(block_d[1][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c10, vc, vx, vs, vn, st1, mn1, su1, nq1));
                    _mm512_store_ps(block_d[2][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c20, vc, vx, vs, vn, st2, mn2, su2, nq2));
                    _mm512_store_ps(block_d[3][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c30, vc, vx, vs, vn, st3, mn3, su3, nq3));
                }
                {
                    const uint64_t pb = (p + 1) * 16;
                    const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                    _mm512_store_ps(block_d[0][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c01, vc, vx, vs, vn, st0, mn0, su0, nq0));
                    _mm512_store_ps(block_d[1][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c11, vc, vx, vs, vn, st1, mn1, su1, nq1));
                    _mm512_store_ps(block_d[2][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c21, vc, vx, vs, vn, st2, mn2, su2, nq2));
                    _mm512_store_ps(block_d[3][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c31, vc, vx, vs, vn, st3, mn3, su3, nq3));
                }
                {
                    const uint64_t pb = (p + 2) * 16;
                    const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                    _mm512_store_ps(block_d[0][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c02, vc, vx, vs, vn, st0, mn0, su0, nq0));
                    _mm512_store_ps(block_d[1][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c12, vc, vx, vs, vn, st1, mn1, su1, nq1));
                    _mm512_store_ps(block_d[2][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c22, vc, vx, vs, vn, st2, mn2, su2, nq2));
                    _mm512_store_ps(block_d[3][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c32, vc, vx, vs, vn, st3, mn3, su3, nq3));
                }
                {
                    const uint64_t pb = (p + 3) * 16;
                    const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                    _mm512_store_ps(block_d[0][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c03, vc, vx, vs, vn, st0, mn0, su0, nq0));
                    _mm512_store_ps(block_d[1][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c13, vc, vx, vs, vn, st1, mn1, su1, nq1));
                    _mm512_store_ps(block_d[2][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c23, vc, vx, vs, vn, st2, mn2, su2, nq2));
                    _mm512_store_ps(block_d[3][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c33, vc, vx, vs, vn, st3, mn3, su3, nq3));
                }

                const __m512i vi0 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 0) * 16)), v_lanes);
                const __m512i vi1 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 1) * 16)), v_lanes);
                const __m512i vi2 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 2) * 16)), v_lanes);
                const __m512i vi3 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 3) * 16)), v_lanes);
                for (uint64_t r = 0; r < 4; ++r) {
                    simd_topk_insert_batch4<k_const>(k,
                                                     q_d[r][0],
                                                     q_i[r][0],
                                                     _mm512_load_ps(block_d[r][0]),
                                                     _mm512_load_ps(block_d[r][1]),
                                                     _mm512_load_ps(block_d[r][2]),
                                                     _mm512_load_ps(block_d[r][3]),
                                                     vi0,
                                                     vi1,
                                                     vi2,
                                                     vi3);
                }
                for (uint64_t j = 0; j < 4; ++j) {
                    const uint64_t pj = p + j;
                    simd_topk_insert_batch4<k_const>(k,
                                                     scratch.db_topk_d + pj * k * 16,
                                                     scratch.db_topk_i + pj * k * 16,
                                                     _mm512_load_ps(block_d[0][j]),
                                                     _mm512_load_ps(block_d[1][j]),
                                                     _mm512_load_ps(block_d[2][j]),
                                                     _mm512_load_ps(block_d[3][j]),
                                                     v_q0,
                                                     v_q1,
                                                     v_q2,
                                                     v_q3);
                }
            }

            for (; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const int8_t* p_ptr = scratch.panel_buf + p * panel_bytes;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const __m512i b =
                        _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(qp0[t]), b);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(qp1[t]), b);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(qp2[t]), b);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(qp3[t]), b);
                }
                const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                const uint64_t hi = std::min<uint64_t>(16, point_count - pb);
                const __mmask16 vmask =
                    (hi == 16) ? 0xFFFFU : static_cast<__mmask16>((1U << hi) - 1U);
                const __m512 dv0 =
                    _mm512_mask_mov_ps(v_inf,
                                       vmask,
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc0, vc, vx, vs, vn, st0, mn0, su0, nq0));
                const __m512 dv1 =
                    _mm512_mask_mov_ps(v_inf,
                                       vmask,
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc1, vc, vx, vs, vn, st1, mn1, su1, nq1));
                const __m512 dv2 =
                    _mm512_mask_mov_ps(v_inf,
                                       vmask,
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc2, vc, vx, vs, vn, st2, mn2, su2, nq2));
                const __m512 dv3 =
                    _mm512_mask_mov_ps(v_inf,
                                       vmask,
                                       compute_panel_distance_avx512_t<metric_kind>(
                                           acc3, vc, vx, vs, vn, st3, mn3, su3, nq3));

                const __m512i v_db_idx =
                    _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(pb)), v_lanes);
                simd_topk_insert<k_const>(k, q_d[0][0], q_i[0][0], dv0, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[1][0], q_i[1][0], dv1, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[2][0], q_i[2][0], dv2, v_db_idx);
                simd_topk_insert<k_const>(k, q_d[3][0], q_i[3][0], dv3, v_db_idx);
                simd_topk_insert_batch4<k_const>(k,
                                                 scratch.db_topk_d + p * k * 16,
                                                 scratch.db_topk_i + p * k * 16,
                                                 dv0,
                                                 dv1,
                                                 dv2,
                                                 dv3,
                                                 v_q0,
                                                 v_q1,
                                                 v_q2,
                                                 v_q3);
            }
        } else {
            for (uint64_t p = first_panel; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const int8_t* p_ptr = scratch.panel_buf + p * panel_bytes;
                const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(scratch.norms + pb);
                const __m512i v_db_idx =
                    _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(pb)), v_lanes);
                float* db_d = scratch.db_topk_d + p * k * 16;
                uint32_t* db_i = scratch.db_topk_i + p * k * 16;
                const uint64_t hi = std::min<uint64_t>(16, point_count - pb);
                for (uint64_t r = 0; r < mq; ++r) {
                    const uint64_t qi = q0 + r;
                    const uint64_t lo = (qi >= pb) ? (qi - pb + 1) : 0;
                    if (lo >= hi) {
                        continue;
                    }
                    const auto* qp =
                        reinterpret_cast<const int32_t*>(scratch.u8_buf + qi * dim_pad4);
                    __m512i acc = _mm512_setzero_si512();
                    for (uint64_t t = 0; t < total_tiles; ++t) {
                        const __m512i b =
                            _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64));
                        acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(qp[t]), b);
                    }
                    const auto vmask =
                        static_cast<__mmask16>(((1U << hi) - 1U) & ~((1U << lo) - 1U));
                    const __m512 dv = _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(acc,
                                                                     vc,
                                                                     vx,
                                                                     vs,
                                                                     vn,
                                                                     scratch.steps[qi],
                                                                     scratch.min_vals[qi],
                                                                     scratch.su[qi],
                                                                     scratch.norms[qi]));
                    simd_topk_insert<k_const>(k, q_d[r][0], q_i[r][0], dv, v_db_idx);
                    simd_topk_insert<k_const>(
                        k, db_d, db_i, dv, _mm512_set1_epi32(static_cast<int32_t>(qi)));
                }
            }
        }

        for (uint64_t r = 0; r < mq; ++r) {
            const uint64_t qi = q0 + r;
            scratch.topk_count[qi] = horizontal_reduce_simd_topk(
                q_d[r][0], q_i[r][0], k, scratch.topk_dists + qi * k, scratch.topk_idx + qi * k);
        }
    }

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint64_t p = i / 16;
        const uint64_t lane = i % 16;
        const float* db_d_panel = scratch.db_topk_d + p * k * 16;
        const uint32_t* db_i_panel = scratch.db_topk_i + p * k * 16;

        float qd[8];
        uint32_t qi_arr[8];
        const uint16_t q_cnt = scratch.topk_count[i];
        float* out_d = scratch.topk_dists + i * k;
        uint32_t* out_i = scratch.topk_idx + i * k;
        for (uint16_t a = 0; a < q_cnt; ++a) {
            qd[a] = out_d[a];
            qi_arr[a] = out_i[a];
        }

        uint16_t a = 0;
        uint16_t b = 0;
        uint16_t out = 0;
        float db_d_cur = db_d_panel[lane];
        bool db_valid = (k > 0) and (db_d_cur < SENTINEL);
        while (out < k) {
            const bool a_ok = (a < q_cnt);
            if (not a_ok and not db_valid) {
                break;
            }
            if (a_ok and (not db_valid or qd[a] <= db_d_cur)) {
                out_d[out] = qd[a];
                out_i[out] = qi_arr[a];
                ++a;
            } else {
                out_d[out] = db_d_cur;
                out_i[out] = db_i_panel[static_cast<size_t>(b) * 16 + lane];
                ++b;
                if (b < k) {
                    db_d_cur = db_d_panel[static_cast<size_t>(b) * 16 + lane];
                    db_valid = (db_d_cur < SENTINEL);
                } else {
                    db_valid = false;
                }
            }
            ++out;
        }
        scratch.topk_count[i] = out;
    }
}

template <MetricKind metric_kind>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) void
fused_leaf_topk_scalar_fallback_impl(const uint32_t* leaf_points,
                                     uint64_t point_count,
                                     const uint8_t* q_data_u8,
                                     const QMeta* q_meta,
                                     const uint32_t* global_ids_map,
                                     bool ids_is_identity,
                                     uint64_t dim,
                                     uint64_t dim_pad4,
                                     uint64_t q_stride,
                                     uint32_t neighbor_count,
                                     VnniScratch& scratch) {
    const auto dim_f = static_cast<float>(dim);
    const uint64_t total_tiles = dim_pad4 / 4;
    const uint64_t num_panels = (point_count + 15) / 16;
    const uint64_t padded_pts = num_panels * 16;
    const size_t panel_bytes = total_tiles * 64;

    scratch.EnsureU8(point_count * dim_pad4);
    scratch.EnsurePanels(num_panels * panel_bytes);
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, neighbor_count);

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t pid = leaf_points[i];
        copy_vec_u8_avx512(q_data_u8 + static_cast<size_t>(pid) * q_stride,
                           scratch.u8_buf + i * dim_pad4,
                           dim_pad4);
        const QMeta& m = q_meta[pid];
        scratch.steps[i] = m.step;
        scratch.min_vals[i] = m.min_val;
        scratch.su[i] = m.su;
        scratch.c_vals[i] = m.min_val + 128.0F * m.step;
        scratch.sum_x[i] = dim_f * m.min_val + m.su;
        scratch.norms[i] = m.norm;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.farthest_dist[i] = SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }
    for (uint64_t i = point_count; i < padded_pts; ++i) {
        scratch.steps[i] = 0.0F;
        scratch.min_vals[i] = 0.0F;
        scratch.su[i] = 0.0F;
        scratch.c_vals[i] = 0.0F;
        scratch.sum_x[i] = 0.0F;
        scratch.norms[i] = SENTINEL;
        scratch.global_ids[i] = 0;
        scratch.farthest_dist[i] = -SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }

    for (uint64_t p = 0; p < num_panels; ++p) {
        const uint64_t base = p * 16;
        const uint64_t valid = std::min<uint64_t>(16, point_count - base);
        auto* panel_u32 = reinterpret_cast<uint32_t*>(scratch.panel_buf + p * panel_bytes);
        const uint32_t* ptrs[16];
        for (uint64_t lane = 0; lane < valid; ++lane) {
            ptrs[lane] =
                reinterpret_cast<const uint32_t*>(scratch.u8_buf + (base + lane) * dim_pad4);
        }
        for (uint64_t t = 0; t < total_tiles; ++t) {
            uint32_t* dst_tile = panel_u32 + t * 16;
            for (uint64_t lane = 0; lane < valid; ++lane) {
                dst_tile[lane] = ptrs[lane][t] ^ FLIP_U32;
            }
            for (uint64_t lane = valid; lane < 16; ++lane) {
                dst_tile[lane] = 0;
            }
        }
    }

    for (uint64_t q0 = 0; q0 < point_count; ++q0) {
        const auto* qp = reinterpret_cast<const int32_t*>(scratch.u8_buf + q0 * dim_pad4);
        const float st = scratch.steps[q0];
        const float mn = scratch.min_vals[q0];
        const float su = scratch.su[q0];
        const float nq = scratch.norms[q0];
        for (uint64_t p = q0 / 16; p < num_panels; ++p) {
            const uint64_t pb = p * 16;
            const int8_t* p_ptr = scratch.panel_buf + p * panel_bytes;
            __m512i acc = _mm512_setzero_si512();
            for (uint64_t t = 0; t < total_tiles; ++t) {
                acc = _mm512_dpbusd_epi32(
                    acc,
                    _mm512_set1_epi32(qp[t]),
                    _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64)));
            }
            const __m512 vc = _mm512_load_ps(scratch.c_vals + pb);
            const __m512 vx = _mm512_load_ps(scratch.sum_x + pb);
            const __m512 vs = _mm512_load_ps(scratch.steps + pb);
            const __m512 vn = _mm512_load_ps(scratch.norms + pb);
            check_and_scatter_leaf_masked_avx512(
                q0,
                pb,
                point_count,
                neighbor_count,
                compute_panel_distance_avx512_t<metric_kind>(acc, vc, vx, vs, vn, st, mn, su, nq),
                scratch);
        }
    }
    for (uint64_t i = 0; i < point_count; ++i) {
        sort_leaf_topk_row(static_cast<uint32_t>(i), neighbor_count, scratch.global_ids, scratch);
    }
}

template <MetricKind metric_kind>
void
dispatch_leaf_topk_by_k(const uint32_t* leaf_points,
                        uint64_t point_count,
                        const uint8_t* q_data_u8,
                        const QMeta* q_meta,
                        const uint32_t* global_ids_map,
                        bool ids_is_identity,
                        uint64_t dim,
                        uint64_t dim_pad4,
                        uint64_t q_stride,
                        uint32_t neighbor_count,
                        VnniScratch& scratch) {
    if (neighbor_count == 5) {
        fused_leaf_topk_simd_avx512_vnni_impl<metric_kind, 5>(leaf_points,
                                                              point_count,
                                                              q_data_u8,
                                                              q_meta,
                                                              global_ids_map,
                                                              ids_is_identity,
                                                              dim,
                                                              dim_pad4,
                                                              q_stride,
                                                              neighbor_count,
                                                              scratch);
    } else if (neighbor_count == 2) {
        fused_leaf_topk_simd_avx512_vnni_impl<metric_kind, 2>(leaf_points,
                                                              point_count,
                                                              q_data_u8,
                                                              q_meta,
                                                              global_ids_map,
                                                              ids_is_identity,
                                                              dim,
                                                              dim_pad4,
                                                              q_stride,
                                                              neighbor_count,
                                                              scratch);
    } else if (neighbor_count == 3) {
        fused_leaf_topk_simd_avx512_vnni_impl<metric_kind, 3>(leaf_points,
                                                              point_count,
                                                              q_data_u8,
                                                              q_meta,
                                                              global_ids_map,
                                                              ids_is_identity,
                                                              dim,
                                                              dim_pad4,
                                                              q_stride,
                                                              neighbor_count,
                                                              scratch);
    } else if (neighbor_count == 4) {
        fused_leaf_topk_simd_avx512_vnni_impl<metric_kind, 4>(leaf_points,
                                                              point_count,
                                                              q_data_u8,
                                                              q_meta,
                                                              global_ids_map,
                                                              ids_is_identity,
                                                              dim,
                                                              dim_pad4,
                                                              q_stride,
                                                              neighbor_count,
                                                              scratch);
    } else if (neighbor_count >= 1 and neighbor_count <= 8) {
        fused_leaf_topk_simd_avx512_vnni_impl<metric_kind, 0>(leaf_points,
                                                              point_count,
                                                              q_data_u8,
                                                              q_meta,
                                                              global_ids_map,
                                                              ids_is_identity,
                                                              dim,
                                                              dim_pad4,
                                                              q_stride,
                                                              neighbor_count,
                                                              scratch);
    } else {
        fused_leaf_topk_scalar_fallback_impl<metric_kind>(leaf_points,
                                                          point_count,
                                                          q_data_u8,
                                                          q_meta,
                                                          global_ids_map,
                                                          ids_is_identity,
                                                          dim,
                                                          dim_pad4,
                                                          q_stride,
                                                          neighbor_count,
                                                          scratch);
    }
}

void
fused_leaf_topk_avx512_vnni(const uint32_t* leaf_points,
                            uint64_t point_count,
                            const uint8_t* q_data_u8,
                            const QMeta* q_meta,
                            const uint32_t* global_ids_map,
                            bool ids_is_identity,
                            uint64_t dim,
                            uint64_t dim_pad4,
                            uint64_t q_stride,
                            MetricKind metric,
                            uint32_t neighbor_count,
                            VnniScratch& scratch) {
    if (metric == MetricKind::L2_SQR) {
        dispatch_leaf_topk_by_k<MetricKind::L2_SQR>(leaf_points,
                                                    point_count,
                                                    q_data_u8,
                                                    q_meta,
                                                    global_ids_map,
                                                    ids_is_identity,
                                                    dim,
                                                    dim_pad4,
                                                    q_stride,
                                                    neighbor_count,
                                                    scratch);
    } else if (metric == MetricKind::IP) {
        dispatch_leaf_topk_by_k<MetricKind::IP>(leaf_points,
                                                point_count,
                                                q_data_u8,
                                                q_meta,
                                                global_ids_map,
                                                ids_is_identity,
                                                dim,
                                                dim_pad4,
                                                q_stride,
                                                neighbor_count,
                                                scratch);
    } else {
        dispatch_leaf_topk_by_k<MetricKind::COSINE>(leaf_points,
                                                    point_count,
                                                    q_data_u8,
                                                    q_meta,
                                                    global_ids_map,
                                                    ids_is_identity,
                                                    dim,
                                                    dim_pad4,
                                                    q_stride,
                                                    neighbor_count,
                                                    scratch);
    }
}

// =============================================================================
// Zero-copy Fused Leader Assignment AVX-512 VNNI (asymmetric sq8_local)
// =============================================================================
template <MetricKind metric_kind, uint32_t k_const>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) void
assign_stripe_small_fanout_avx512_impl(const uint32_t* points,
                                       uint64_t stripe_size,
                                       const VnniScratch& leader_scratch,
                                       uint64_t leader_count,
                                       const uint8_t* q_data_u8,
                                       const QMeta* q_meta,
                                       uint64_t dim_pad4,
                                       uint64_t q_stride,
                                       uint32_t fanout,
                                       uint32_t* out_assignments) {
    const uint32_t k = (k_const > 0) ? k_const : fanout;
    const uint64_t total_tiles = dim_pad4 / 4;
    const uint64_t num_panels = (leader_count + 15) / 16;
    const uint64_t full_panels = leader_count / 16;
    const size_t panel_bytes = total_tiles * 64;

    const __m512 v_inf = _mm512_set1_ps(SENTINEL);
    const __m512i v_invalid = _mm512_set1_epi32(-1);
    const __m512i v_lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    alignas(64) float block_d[4][4][16];
    alignas(64) float q_d[4][4][16];
    alignas(64) uint32_t q_i[4][4][16];

    for (uint64_t q0 = 0; q0 < stripe_size; q0 += 4) {
        const uint64_t mq = std::min<uint64_t>(4, stripe_size - q0);
        for (uint64_t r = 0; r < 4 and q0 + 4 + r < stripe_size; ++r) {
            const uint32_t pf = points[q0 + 4 + r];
            prefetch_vec_u8(q_data_u8 + static_cast<size_t>(pf) * q_stride, dim_pad4);
            _mm_prefetch(reinterpret_cast<const char*>(q_meta + pf), _MM_HINT_T0);
        }

        for (uint64_t r = 0; r < mq; ++r) {
            for (uint32_t s = 0; s < k; ++s) {
                _mm512_store_ps(q_d[r][s], v_inf);
                _mm512_store_si512(reinterpret_cast<void*>(q_i[r][s]), v_invalid);
            }
        }

        if (mq == 4) {
            const uint32_t pid0 = points[q0 + 0];
            const uint32_t pid1 = points[q0 + 1];
            const uint32_t pid2 = points[q0 + 2];
            const uint32_t pid3 = points[q0 + 3];
            const auto* qp0 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid0) * q_stride);
            const auto* qp1 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid1) * q_stride);
            const auto* qp2 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid2) * q_stride);
            const auto* qp3 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid3) * q_stride);
            const QMeta& m0 = q_meta[pid0];
            const QMeta& m1 = q_meta[pid1];
            const QMeta& m2 = q_meta[pid2];
            const QMeta& m3 = q_meta[pid3];

            uint64_t p = 0;
            for (; p + 4 <= full_panels; p += 4) {
                const int8_t* p0 = leader_scratch.panel_buf + (p + 0) * panel_bytes;
                const int8_t* p1 = leader_scratch.panel_buf + (p + 1) * panel_bytes;
                const int8_t* p2 = leader_scratch.panel_buf + (p + 2) * panel_bytes;
                const int8_t* p3 = leader_scratch.panel_buf + (p + 3) * panel_bytes;

                __m512i c00 = _mm512_setzero_si512();
                __m512i c01 = _mm512_setzero_si512();
                __m512i c02 = _mm512_setzero_si512();
                __m512i c03 = _mm512_setzero_si512();
                __m512i c10 = _mm512_setzero_si512();
                __m512i c11 = _mm512_setzero_si512();
                __m512i c12 = _mm512_setzero_si512();
                __m512i c13 = _mm512_setzero_si512();
                __m512i c20 = _mm512_setzero_si512();
                __m512i c21 = _mm512_setzero_si512();
                __m512i c22 = _mm512_setzero_si512();
                __m512i c23 = _mm512_setzero_si512();
                __m512i c30 = _mm512_setzero_si512();
                __m512i c31 = _mm512_setzero_si512();
                __m512i c32 = _mm512_setzero_si512();
                __m512i c33 = _mm512_setzero_si512();

                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const size_t off = t * 64;
                    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const void*>(p0 + off));
                    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const void*>(p1 + off));
                    const __m512i b2 = _mm512_load_si512(reinterpret_cast<const void*>(p2 + off));
                    const __m512i b3 = _mm512_load_si512(reinterpret_cast<const void*>(p3 + off));

                    const __m512i qv0 = _mm512_set1_epi32(qp0[t]);
                    c00 = _mm512_dpbusd_epi32(c00, qv0, b0);
                    c01 = _mm512_dpbusd_epi32(c01, qv0, b1);
                    c02 = _mm512_dpbusd_epi32(c02, qv0, b2);
                    c03 = _mm512_dpbusd_epi32(c03, qv0, b3);

                    const __m512i qv1 = _mm512_set1_epi32(qp1[t]);
                    c10 = _mm512_dpbusd_epi32(c10, qv1, b0);
                    c11 = _mm512_dpbusd_epi32(c11, qv1, b1);
                    c12 = _mm512_dpbusd_epi32(c12, qv1, b2);
                    c13 = _mm512_dpbusd_epi32(c13, qv1, b3);

                    const __m512i qv2 = _mm512_set1_epi32(qp2[t]);
                    c20 = _mm512_dpbusd_epi32(c20, qv2, b0);
                    c21 = _mm512_dpbusd_epi32(c21, qv2, b1);
                    c22 = _mm512_dpbusd_epi32(c22, qv2, b2);
                    c23 = _mm512_dpbusd_epi32(c23, qv2, b3);

                    const __m512i qv3 = _mm512_set1_epi32(qp3[t]);
                    c30 = _mm512_dpbusd_epi32(c30, qv3, b0);
                    c31 = _mm512_dpbusd_epi32(c31, qv3, b1);
                    c32 = _mm512_dpbusd_epi32(c32, qv3, b2);
                    c33 = _mm512_dpbusd_epi32(c33, qv3, b3);
                }

                {
                    const uint64_t pb = (p + 0) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c00, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c10, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c20, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c30, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 1) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c01, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c11, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c21, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c31, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 2) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c02, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c12, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c22, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c32, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 3) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c03, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c13, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c23, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c33, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }

                const __m512i vi0 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 0) * 16)), v_lanes);
                const __m512i vi1 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 1) * 16)), v_lanes);
                const __m512i vi2 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 2) * 16)), v_lanes);
                const __m512i vi3 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 3) * 16)), v_lanes);
                for (uint64_t r = 0; r < 4; ++r) {
                    simd_topk_insert_batch4<k_const>(k,
                                                     q_d[r][0],
                                                     q_i[r][0],
                                                     _mm512_load_ps(block_d[r][0]),
                                                     _mm512_load_ps(block_d[r][1]),
                                                     _mm512_load_ps(block_d[r][2]),
                                                     _mm512_load_ps(block_d[r][3]),
                                                     vi0,
                                                     vi1,
                                                     vi2,
                                                     vi3);
                }
            }

            for (; p + 2 <= full_panels; p += 2) {
                const int8_t* p0 = leader_scratch.panel_buf + (p + 0) * panel_bytes;
                const int8_t* p1 = leader_scratch.panel_buf + (p + 1) * panel_bytes;
                __m512i c00 = _mm512_setzero_si512();
                __m512i c01 = _mm512_setzero_si512();
                __m512i c10 = _mm512_setzero_si512();
                __m512i c11 = _mm512_setzero_si512();
                __m512i c20 = _mm512_setzero_si512();
                __m512i c21 = _mm512_setzero_si512();
                __m512i c30 = _mm512_setzero_si512();
                __m512i c31 = _mm512_setzero_si512();

                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const size_t off = t * 64;
                    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const void*>(p0 + off));
                    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const void*>(p1 + off));

                    const __m512i qv0 = _mm512_set1_epi32(qp0[t]);
                    c00 = _mm512_dpbusd_epi32(c00, qv0, b0);
                    c01 = _mm512_dpbusd_epi32(c01, qv0, b1);

                    const __m512i qv1 = _mm512_set1_epi32(qp1[t]);
                    c10 = _mm512_dpbusd_epi32(c10, qv1, b0);
                    c11 = _mm512_dpbusd_epi32(c11, qv1, b1);

                    const __m512i qv2 = _mm512_set1_epi32(qp2[t]);
                    c20 = _mm512_dpbusd_epi32(c20, qv2, b0);
                    c21 = _mm512_dpbusd_epi32(c21, qv2, b1);

                    const __m512i qv3 = _mm512_set1_epi32(qp3[t]);
                    c30 = _mm512_dpbusd_epi32(c30, qv3, b0);
                    c31 = _mm512_dpbusd_epi32(c31, qv3, b1);
                }

                {
                    const uint64_t pb = (p + 0) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c00, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c10, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c20, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c30, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 1) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c01, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c11, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c21, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c31, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                const __m512i vi0 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 0) * 16)), v_lanes);
                const __m512i vi1 = _mm512_add_epi32(
                    _mm512_set1_epi32(static_cast<int32_t>((p + 1) * 16)), v_lanes);
                for (uint64_t r = 0; r < 4; ++r) {
                    simd_topk_insert_batch2<k_const>(k,
                                                     q_d[r][0],
                                                     q_i[r][0],
                                                     _mm512_load_ps(block_d[r][0]),
                                                     _mm512_load_ps(block_d[r][1]),
                                                     vi0,
                                                     vi1);
                }
            }

            for (; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const uint64_t valid = std::min<uint64_t>(16, leader_count - pb);
                const __mmask16 vmask =
                    (valid == 16) ? 0xFFFFU : static_cast<__mmask16>((1U << valid) - 1U);
                const int8_t* p_ptr = leader_scratch.panel_buf + p * panel_bytes;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const __m512i b =
                        _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(qp0[t]), b);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(qp1[t]), b);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(qp2[t]), b);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(qp3[t]), b);
                }
                const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                const __m512i v_idx =
                    _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(pb)), v_lanes);
                simd_topk_insert<k_const>(
                    k,
                    q_d[0][0],
                    q_i[0][0],
                    _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(
                            acc0, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm)),
                    v_idx);
                simd_topk_insert<k_const>(
                    k,
                    q_d[1][0],
                    q_i[1][0],
                    _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(
                            acc1, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm)),
                    v_idx);
                simd_topk_insert<k_const>(
                    k,
                    q_d[2][0],
                    q_i[2][0],
                    _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(
                            acc2, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm)),
                    v_idx);
                simd_topk_insert<k_const>(
                    k,
                    q_d[3][0],
                    q_i[3][0],
                    _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(
                            acc3, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm)),
                    v_idx);
            }
        } else {
            for (uint64_t p = 0; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const uint64_t valid = std::min<uint64_t>(16, leader_count - pb);
                const __mmask16 vmask =
                    (valid == 16) ? 0xFFFFU : static_cast<__mmask16>((1U << valid) - 1U);
                const int8_t* p_ptr = leader_scratch.panel_buf + p * panel_bytes;
                const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                const __m512i v_idx =
                    _mm512_add_epi32(_mm512_set1_epi32(static_cast<int32_t>(pb)), v_lanes);
                for (uint64_t r = 0; r < mq; ++r) {
                    const uint32_t pid = points[q0 + r];
                    const QMeta& mq_meta = q_meta[pid];
                    const auto* qp = reinterpret_cast<const int32_t*>(
                        q_data_u8 + static_cast<size_t>(pid) * q_stride);
                    __m512i acc = _mm512_setzero_si512();
                    for (uint64_t t = 0; t < total_tiles; ++t) {
                        acc = _mm512_dpbusd_epi32(
                            acc,
                            _mm512_set1_epi32(qp[t]),
                            _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64)));
                    }
                    const __m512 dv = _mm512_mask_mov_ps(
                        v_inf,
                        vmask,
                        compute_panel_distance_avx512_t<metric_kind>(acc,
                                                                     vc,
                                                                     vx,
                                                                     vs,
                                                                     vn,
                                                                     mq_meta.step,
                                                                     mq_meta.min_val,
                                                                     mq_meta.su,
                                                                     mq_meta.norm));
                    simd_topk_insert<k_const>(k, q_d[r][0], q_i[r][0], dv, v_idx);
                }
            }
        }

        float tmp_d[4];
        for (uint64_t r = 0; r < mq; ++r) {
            const uint64_t qi = q0 + r;
            horizontal_reduce_simd_topk(q_d[r][0], q_i[r][0], k, tmp_d, out_assignments + qi * k);
        }
    }
}

template <MetricKind metric_kind>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"))) void
assign_stripe_large_fanout_avx512_impl(const uint32_t* points,
                                       uint64_t stripe_size,
                                       const VnniScratch& leader_scratch,
                                       uint64_t leader_count,
                                       const uint8_t* q_data_u8,
                                       const QMeta* q_meta,
                                       uint64_t dim_pad4,
                                       uint64_t q_stride,
                                       uint32_t fanout,
                                       uint32_t* out_assignments,
                                       VnniScratch& worker_scratch) {
    const uint64_t total_tiles = dim_pad4 / 4;
    const uint64_t num_panels = (leader_count + 15) / 16;
    const uint64_t full_panels = leader_count / 16;
    const size_t panel_bytes = total_tiles * 64;

    worker_scratch.EnsureTopK(stripe_size, fanout);
    for (uint64_t i = 0; i < stripe_size; ++i) {
        worker_scratch.farthest_dist[i] = SENTINEL;
        worker_scratch.topk_count[i] = 0;
        worker_scratch.topk_farthest_pos[i] = 0;
    }

    alignas(64) float block_d[4][4][16];

    for (uint64_t q0 = 0; q0 < stripe_size; q0 += 4) {
        const uint64_t mq = std::min<uint64_t>(4, stripe_size - q0);
        for (uint64_t r = 0; r < 4 and q0 + 4 + r < stripe_size; ++r) {
            const uint32_t pf = points[q0 + 4 + r];
            prefetch_vec_u8(q_data_u8 + static_cast<size_t>(pf) * q_stride, dim_pad4);
            _mm_prefetch(reinterpret_cast<const char*>(q_meta + pf), _MM_HINT_T0);
        }
        if (mq == 4) {
            const uint32_t pid0 = points[q0 + 0];
            const uint32_t pid1 = points[q0 + 1];
            const uint32_t pid2 = points[q0 + 2];
            const uint32_t pid3 = points[q0 + 3];
            const auto* qp0 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid0) * q_stride);
            const auto* qp1 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid1) * q_stride);
            const auto* qp2 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid2) * q_stride);
            const auto* qp3 =
                reinterpret_cast<const int32_t*>(q_data_u8 + static_cast<size_t>(pid3) * q_stride);
            const QMeta& m0 = q_meta[pid0];
            const QMeta& m1 = q_meta[pid1];
            const QMeta& m2 = q_meta[pid2];
            const QMeta& m3 = q_meta[pid3];

            uint64_t p = 0;
            for (; p + 4 <= full_panels; p += 4) {
                const int8_t* p0 = leader_scratch.panel_buf + (p + 0) * panel_bytes;
                const int8_t* p1 = leader_scratch.panel_buf + (p + 1) * panel_bytes;
                const int8_t* p2 = leader_scratch.panel_buf + (p + 2) * panel_bytes;
                const int8_t* p3 = leader_scratch.panel_buf + (p + 3) * panel_bytes;

                __m512i c00 = _mm512_setzero_si512();
                __m512i c01 = _mm512_setzero_si512();
                __m512i c02 = _mm512_setzero_si512();
                __m512i c03 = _mm512_setzero_si512();
                __m512i c10 = _mm512_setzero_si512();
                __m512i c11 = _mm512_setzero_si512();
                __m512i c12 = _mm512_setzero_si512();
                __m512i c13 = _mm512_setzero_si512();
                __m512i c20 = _mm512_setzero_si512();
                __m512i c21 = _mm512_setzero_si512();
                __m512i c22 = _mm512_setzero_si512();
                __m512i c23 = _mm512_setzero_si512();
                __m512i c30 = _mm512_setzero_si512();
                __m512i c31 = _mm512_setzero_si512();
                __m512i c32 = _mm512_setzero_si512();
                __m512i c33 = _mm512_setzero_si512();

                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const size_t off = t * 64;
                    const __m512i b0 = _mm512_load_si512(reinterpret_cast<const void*>(p0 + off));
                    const __m512i b1 = _mm512_load_si512(reinterpret_cast<const void*>(p1 + off));
                    const __m512i b2 = _mm512_load_si512(reinterpret_cast<const void*>(p2 + off));
                    const __m512i b3 = _mm512_load_si512(reinterpret_cast<const void*>(p3 + off));

                    const __m512i qv0 = _mm512_set1_epi32(qp0[t]);
                    c00 = _mm512_dpbusd_epi32(c00, qv0, b0);
                    c01 = _mm512_dpbusd_epi32(c01, qv0, b1);
                    c02 = _mm512_dpbusd_epi32(c02, qv0, b2);
                    c03 = _mm512_dpbusd_epi32(c03, qv0, b3);

                    const __m512i qv1 = _mm512_set1_epi32(qp1[t]);
                    c10 = _mm512_dpbusd_epi32(c10, qv1, b0);
                    c11 = _mm512_dpbusd_epi32(c11, qv1, b1);
                    c12 = _mm512_dpbusd_epi32(c12, qv1, b2);
                    c13 = _mm512_dpbusd_epi32(c13, qv1, b3);

                    const __m512i qv2 = _mm512_set1_epi32(qp2[t]);
                    c20 = _mm512_dpbusd_epi32(c20, qv2, b0);
                    c21 = _mm512_dpbusd_epi32(c21, qv2, b1);
                    c22 = _mm512_dpbusd_epi32(c22, qv2, b2);
                    c23 = _mm512_dpbusd_epi32(c23, qv2, b3);

                    const __m512i qv3 = _mm512_set1_epi32(qp3[t]);
                    c30 = _mm512_dpbusd_epi32(c30, qv3, b0);
                    c31 = _mm512_dpbusd_epi32(c31, qv3, b1);
                    c32 = _mm512_dpbusd_epi32(c32, qv3, b2);
                    c33 = _mm512_dpbusd_epi32(c33, qv3, b3);
                }

                {
                    const uint64_t pb = (p + 0) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c00, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c10, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c20, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][0],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c30, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 1) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c01, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c11, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c21, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][1],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c31, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 2) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c02, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c12, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c22, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][2],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c32, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }
                {
                    const uint64_t pb = (p + 3) * 16;
                    const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                    const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                    const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                    const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                    _mm512_store_ps(block_d[0][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c03, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                    _mm512_store_ps(block_d[1][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c13, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                    _mm512_store_ps(block_d[2][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c23, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                    _mm512_store_ps(block_d[3][3],
                                    compute_panel_distance_avx512_t<metric_kind>(
                                        c33, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                }

                for (uint64_t r = 0; r < 4; ++r) {
                    const uint64_t qi = q0 + r;
                    __m512 v_far = _mm512_set1_ps(worker_scratch.farthest_dist[qi]);
                    for (uint64_t j = 0; j < 4; ++j) {
                        const __mmask16 mask =
                            _mm512_cmp_ps_mask(_mm512_load_ps(block_d[r][j]), v_far, _CMP_LE_OQ);
                        if (mask != 0) {
                            flush_leader_mask_hits(qi,
                                                   (p + j) * 16,
                                                   fanout,
                                                   mask,
                                                   block_d[r][j],
                                                   leader_scratch,
                                                   worker_scratch);
                            v_far = _mm512_set1_ps(worker_scratch.farthest_dist[qi]);
                        }
                    }
                }
            }

            for (; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const uint64_t valid = std::min<uint64_t>(16, leader_count - pb);
                const uint16_t valid_mask =
                    (valid == 16) ? 0xFFFFU : static_cast<uint16_t>((1U << valid) - 1U);
                const int8_t* p_ptr = leader_scratch.panel_buf + p * panel_bytes;
                __m512i acc0 = _mm512_setzero_si512();
                __m512i acc1 = _mm512_setzero_si512();
                __m512i acc2 = _mm512_setzero_si512();
                __m512i acc3 = _mm512_setzero_si512();
                for (uint64_t t = 0; t < total_tiles; ++t) {
                    const __m512i b =
                        _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64));
                    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_set1_epi32(qp0[t]), b);
                    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_set1_epi32(qp1[t]), b);
                    acc2 = _mm512_dpbusd_epi32(acc2, _mm512_set1_epi32(qp2[t]), b);
                    acc3 = _mm512_dpbusd_epi32(acc3, _mm512_set1_epi32(qp3[t]), b);
                }
                const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                _mm512_store_ps(block_d[0][0],
                                compute_panel_distance_avx512_t<metric_kind>(
                                    acc0, vc, vx, vs, vn, m0.step, m0.min_val, m0.su, m0.norm));
                _mm512_store_ps(block_d[1][0],
                                compute_panel_distance_avx512_t<metric_kind>(
                                    acc1, vc, vx, vs, vn, m1.step, m1.min_val, m1.su, m1.norm));
                _mm512_store_ps(block_d[2][0],
                                compute_panel_distance_avx512_t<metric_kind>(
                                    acc2, vc, vx, vs, vn, m2.step, m2.min_val, m2.su, m2.norm));
                _mm512_store_ps(block_d[3][0],
                                compute_panel_distance_avx512_t<metric_kind>(
                                    acc3, vc, vx, vs, vn, m3.step, m3.min_val, m3.su, m3.norm));
                for (uint64_t r = 0; r < 4; ++r) {
                    check_and_update_leader_avx512(q0 + r,
                                                   pb,
                                                   valid_mask,
                                                   fanout,
                                                   _mm512_load_ps(block_d[r][0]),
                                                   leader_scratch,
                                                   worker_scratch);
                }
            }
        } else {
            for (uint64_t p = 0; p < num_panels; ++p) {
                const uint64_t pb = p * 16;
                const uint64_t valid = std::min<uint64_t>(16, leader_count - pb);
                const uint16_t valid_mask =
                    (valid == 16) ? 0xFFFFU : static_cast<uint16_t>((1U << valid) - 1U);
                const int8_t* p_ptr = leader_scratch.panel_buf + p * panel_bytes;
                const __m512 vc = _mm512_load_ps(leader_scratch.c_vals + pb);
                const __m512 vx = _mm512_load_ps(leader_scratch.sum_x + pb);
                const __m512 vs = _mm512_load_ps(leader_scratch.steps + pb);
                const __m512 vn = _mm512_load_ps(leader_scratch.norms + pb);
                for (uint64_t r = 0; r < mq; ++r) {
                    const uint32_t pid = points[q0 + r];
                    const QMeta& mq_meta = q_meta[pid];
                    const auto* qp = reinterpret_cast<const int32_t*>(
                        q_data_u8 + static_cast<size_t>(pid) * q_stride);
                    __m512i acc = _mm512_setzero_si512();
                    for (uint64_t t = 0; t < total_tiles; ++t) {
                        acc = _mm512_dpbusd_epi32(
                            acc,
                            _mm512_set1_epi32(qp[t]),
                            _mm512_load_si512(reinterpret_cast<const void*>(p_ptr + t * 64)));
                    }
                    check_and_update_leader_avx512(
                        q0 + r,
                        pb,
                        valid_mask,
                        fanout,
                        compute_panel_distance_avx512_t<metric_kind>(acc,
                                                                     vc,
                                                                     vx,
                                                                     vs,
                                                                     vn,
                                                                     mq_meta.step,
                                                                     mq_meta.min_val,
                                                                     mq_meta.su,
                                                                     mq_meta.norm),
                        leader_scratch,
                        worker_scratch);
                }
            }
        }
    }
    write_sorted_assignments(stripe_size, fanout, leader_scratch, worker_scratch, out_assignments);
}

template <MetricKind metric_kind>
void
dispatch_assign_by_fanout(const uint32_t* points,
                          uint64_t stripe_size,
                          const VnniScratch& leader_scratch,
                          uint64_t leader_count,
                          const uint8_t* q_data_u8,
                          const QMeta* q_meta,
                          uint64_t dim_pad4,
                          uint64_t q_stride,
                          uint32_t fanout,
                          uint32_t* out_assignments,
                          VnniScratch& worker_scratch) {
    if (fanout == 2) {
        assign_stripe_small_fanout_avx512_impl<metric_kind, 2>(points,
                                                               stripe_size,
                                                               leader_scratch,
                                                               leader_count,
                                                               q_data_u8,
                                                               q_meta,
                                                               dim_pad4,
                                                               q_stride,
                                                               fanout,
                                                               out_assignments);
    } else if (fanout == 1) {
        assign_stripe_small_fanout_avx512_impl<metric_kind, 1>(points,
                                                               stripe_size,
                                                               leader_scratch,
                                                               leader_count,
                                                               q_data_u8,
                                                               q_meta,
                                                               dim_pad4,
                                                               q_stride,
                                                               fanout,
                                                               out_assignments);
    } else if (fanout == 3) {
        assign_stripe_small_fanout_avx512_impl<metric_kind, 3>(points,
                                                               stripe_size,
                                                               leader_scratch,
                                                               leader_count,
                                                               q_data_u8,
                                                               q_meta,
                                                               dim_pad4,
                                                               q_stride,
                                                               fanout,
                                                               out_assignments);
    } else if (fanout == 4) {
        assign_stripe_small_fanout_avx512_impl<metric_kind, 4>(points,
                                                               stripe_size,
                                                               leader_scratch,
                                                               leader_count,
                                                               q_data_u8,
                                                               q_meta,
                                                               dim_pad4,
                                                               q_stride,
                                                               fanout,
                                                               out_assignments);
    } else {
        assign_stripe_large_fanout_avx512_impl<metric_kind>(points,
                                                            stripe_size,
                                                            leader_scratch,
                                                            leader_count,
                                                            q_data_u8,
                                                            q_meta,
                                                            dim_pad4,
                                                            q_stride,
                                                            fanout,
                                                            out_assignments,
                                                            worker_scratch);
    }
}

// =============================================================================
// x86-64 AVX-512BW (non-VNNI) sq8_local Micro-Kernels (32 dims/iter, exact int16->int32)
// =============================================================================
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) int32_t
sq8_dot_1x1_avx512_bw(const uint8_t* a, const int8_t* b_i8, uint64_t dim_pad32) {
    __m512i acc = _mm512_setzero_si512();
    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m512i va =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k)));
        const __m512i vb =
            _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8 + k)));
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(va, vb));
    }
    return _mm512_reduce_add_epi32(acc);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) int32_t
sq8_dot_1x1_raw_b_avx512_bw(const uint8_t* a, const uint8_t* b_u8, uint64_t dim_pad32) {
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    __m512i acc = _mm512_setzero_si512();
    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m512i va =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k)));
        const __m256i br =
            _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_u8 + k)), flip);
        const __m512i vb = _mm512_cvtepi8_epi16(br);
        acc = _mm512_add_epi32(acc, _mm512_madd_epi16(va, vb));
    }
    return _mm512_reduce_add_epi32(acc);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq"))) void
sq8_dot_4x4_avx512_bw(const uint8_t* const a[4],
                      const int8_t* const b_i8[4],
                      uint64_t dim_pad32,
                      int32_t out[4][4]) {
    __m512i c00 = _mm512_setzero_si512();
    __m512i c01 = _mm512_setzero_si512();
    __m512i c02 = _mm512_setzero_si512();
    __m512i c03 = _mm512_setzero_si512();
    __m512i c10 = _mm512_setzero_si512();
    __m512i c11 = _mm512_setzero_si512();
    __m512i c12 = _mm512_setzero_si512();
    __m512i c13 = _mm512_setzero_si512();
    __m512i c20 = _mm512_setzero_si512();
    __m512i c21 = _mm512_setzero_si512();
    __m512i c22 = _mm512_setzero_si512();
    __m512i c23 = _mm512_setzero_si512();
    __m512i c30 = _mm512_setzero_si512();
    __m512i c31 = _mm512_setzero_si512();
    __m512i c32 = _mm512_setzero_si512();
    __m512i c33 = _mm512_setzero_si512();

    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m512i b0 =
            _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[0] + k)));
        const __m512i b1 =
            _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[1] + k)));
        const __m512i b2 =
            _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[2] + k)));
        const __m512i b3 =
            _mm512_cvtepi8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[3] + k)));

        const __m512i a0 =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[0] + k)));
        c00 = _mm512_add_epi32(c00, _mm512_madd_epi16(a0, b0));
        c01 = _mm512_add_epi32(c01, _mm512_madd_epi16(a0, b1));
        c02 = _mm512_add_epi32(c02, _mm512_madd_epi16(a0, b2));
        c03 = _mm512_add_epi32(c03, _mm512_madd_epi16(a0, b3));

        const __m512i a1 =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[1] + k)));
        c10 = _mm512_add_epi32(c10, _mm512_madd_epi16(a1, b0));
        c11 = _mm512_add_epi32(c11, _mm512_madd_epi16(a1, b1));
        c12 = _mm512_add_epi32(c12, _mm512_madd_epi16(a1, b2));
        c13 = _mm512_add_epi32(c13, _mm512_madd_epi16(a1, b3));

        const __m512i a2 =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[2] + k)));
        c20 = _mm512_add_epi32(c20, _mm512_madd_epi16(a2, b0));
        c21 = _mm512_add_epi32(c21, _mm512_madd_epi16(a2, b1));
        c22 = _mm512_add_epi32(c22, _mm512_madd_epi16(a2, b2));
        c23 = _mm512_add_epi32(c23, _mm512_madd_epi16(a2, b3));

        const __m512i a3 =
            _mm512_cvtepu8_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[3] + k)));
        c30 = _mm512_add_epi32(c30, _mm512_madd_epi16(a3, b0));
        c31 = _mm512_add_epi32(c31, _mm512_madd_epi16(a3, b1));
        c32 = _mm512_add_epi32(c32, _mm512_madd_epi16(a3, b2));
        c33 = _mm512_add_epi32(c33, _mm512_madd_epi16(a3, b3));
    }

    out[0][0] = _mm512_reduce_add_epi32(c00);
    out[0][1] = _mm512_reduce_add_epi32(c01);
    out[0][2] = _mm512_reduce_add_epi32(c02);
    out[0][3] = _mm512_reduce_add_epi32(c03);
    out[1][0] = _mm512_reduce_add_epi32(c10);
    out[1][1] = _mm512_reduce_add_epi32(c11);
    out[1][2] = _mm512_reduce_add_epi32(c12);
    out[1][3] = _mm512_reduce_add_epi32(c13);
    out[2][0] = _mm512_reduce_add_epi32(c20);
    out[2][1] = _mm512_reduce_add_epi32(c21);
    out[2][2] = _mm512_reduce_add_epi32(c22);
    out[2][3] = _mm512_reduce_add_epi32(c23);
    out[3][0] = _mm512_reduce_add_epi32(c30);
    out[3][1] = _mm512_reduce_add_epi32(c31);
    out[3][2] = _mm512_reduce_add_epi32(c32);
    out[3][3] = _mm512_reduce_add_epi32(c33);
}

// =============================================================================
// x86-64 AVX2 & AVX-VNNI Helpers + sq8_local Micro-Kernels
// =============================================================================
__attribute__((always_inline, target("avx2"))) inline int32_t
hsum256_i32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

__attribute__((always_inline, target("avx2"))) inline uint16_t
cmp_ge_u16_mask_avx2(__m256i a, __m256i b) {
    const __m256i gt = _mm256_cmpgt_epi16(b, a);
    const __m128i lo = _mm256_castsi256_si128(gt);
    const __m128i hi = _mm256_extracti128_si256(gt, 1);
    const auto lt_mask = static_cast<uint16_t>(_mm_movemask_epi8(_mm_packs_epi16(lo, hi)));
    return static_cast<uint16_t>(~lt_mask);
}

__attribute__((always_inline, target("avx2"))) inline uint16_t
cmp_eq_u16_mask_avx2(__m256i sv, __m256i su) {
    const __m256i eq = _mm256_cmpeq_epi16(sv, su);
    const __m128i lo = _mm256_castsi256_si128(eq);
    const __m128i hi = _mm256_extracti128_si256(eq, 1);
    return static_cast<uint16_t>(_mm_movemask_epi8(_mm_packs_epi16(lo, hi)));
}

__attribute__((target("avx2"))) uint16_t
relative_hash16_avx2(const float* sketch_u,
                     const float* sketch_v,
                     uint64_t planes,
                     uint16_t coincident_flag) {
    const __m256 su0 = _mm256_loadu_ps(sketch_u);
    const __m256 su1 = _mm256_loadu_ps(sketch_u + 8);
    const __m256 sv0 = _mm256_loadu_ps(sketch_v);
    const __m256 sv1 = _mm256_loadu_ps(sketch_v + 8);
    const auto plane_mask = static_cast<uint16_t>((1U << planes) - 1U);
    auto ge_mask =
        static_cast<uint16_t>((_mm256_movemask_ps(_mm256_cmp_ps(sv0, su0, _CMP_GE_OQ)) |
                               (_mm256_movemask_ps(_mm256_cmp_ps(sv1, su1, _CMP_GE_OQ)) << 8)) &
                              plane_mask);
    const auto eq_mask =
        static_cast<uint16_t>((_mm256_movemask_ps(_mm256_cmp_ps(sv0, su0, _CMP_EQ_OQ)) |
                               (_mm256_movemask_ps(_mm256_cmp_ps(sv1, su1, _CMP_EQ_OQ)) << 8)) &
                              plane_mask);
    if (eq_mask == plane_mask) [[unlikely]] {
        ge_mask |= coincident_flag;
    }
    return ge_mask;
}

__attribute__((target("avx2"))) uint16_t
relative_hash16_u16_avx2(const uint16_t* sketch_u,
                         const uint16_t* sketch_v,
                         uint64_t planes,
                         uint16_t coincident_flag) {
    const __m256i sign_flip = _mm256_set1_epi16(static_cast<int16_t>(0x8000));
    const __m256i su = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sketch_u));
    const __m256i sv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sketch_v));
    const auto plane_mask = static_cast<uint16_t>((1U << planes) - 1U);
    uint16_t ge_mask =
        cmp_ge_u16_mask_avx2(_mm256_xor_si256(sv, sign_flip), _mm256_xor_si256(su, sign_flip)) &
        plane_mask;
    const uint16_t eq_mask = cmp_eq_u16_mask_avx2(sv, su) & plane_mask;
    if (eq_mask == plane_mask) [[unlikely]] {
        ge_mask |= coincident_flag;
    }
    return ge_mask;
}

__attribute__((target("avx2"))) uint32_t
group_and_hash_leaf_edges_fp32_avx2(const uint32_t* leaf_points,
                                    uint64_t point_count,
                                    uint32_t neighbor_count,
                                    const uint32_t* topk_idx,
                                    const float* topk_dists,
                                    const uint16_t* topk_count,
                                    const float* global_sketches_f32,
                                    uint64_t hash_planes,
                                    uint16_t coincident_flag,
                                    VnniScratch& scratch,
                                    uint32_t* offsets,
                                    uint32_t* g_targets,
                                    uint16_t* g_dist_keys,
                                    uint16_t* g_hashes) {
    scratch.EnsureLeafSketchesF32(point_count * 16);
    float* leaf_sk = scratch.leaf_sketches_f32;
    for (uint64_t i = 0; i < point_count; ++i) {
        if (i + 8 < point_count) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             global_sketches_f32 + static_cast<uint64_t>(leaf_points[i + 8]) * 16),
                         _MM_HINT_T0);
        }
        const float* src = global_sketches_f32 + static_cast<uint64_t>(leaf_points[i]) * 16;
        _mm256_store_ps(leaf_sk + i * 16, _mm256_loadu_ps(src));
        _mm256_store_ps(leaf_sk + i * 16 + 8, _mm256_loadu_ps(src + 8));
    }

    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;

    const auto plane_mask = static_cast<uint16_t>((1U << hash_planes) - 1U);
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const __m256 su0 = _mm256_load_ps(leaf_sk + src * 16);
        const __m256 su1 = _mm256_load_ps(leaf_sk + src * 16 + 8);
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const __m256 sv0 = _mm256_load_ps(leaf_sk + static_cast<size_t>(dst_local) * 16);
            const __m256 sv1 = _mm256_load_ps(leaf_sk + static_cast<size_t>(dst_local) * 16 + 8);
            auto fwd_h = static_cast<uint16_t>(
                (_mm256_movemask_ps(_mm256_cmp_ps(sv0, su0, _CMP_GE_OQ)) |
                 (_mm256_movemask_ps(_mm256_cmp_ps(sv1, su1, _CMP_GE_OQ)) << 8)) &
                plane_mask);
            auto rev_h = static_cast<uint16_t>(
                (_mm256_movemask_ps(_mm256_cmp_ps(su0, sv0, _CMP_GE_OQ)) |
                 (_mm256_movemask_ps(_mm256_cmp_ps(su1, sv1, _CMP_GE_OQ)) << 8)) &
                plane_mask);
            const auto eq = static_cast<uint16_t>(
                (_mm256_movemask_ps(_mm256_cmp_ps(sv0, su0, _CMP_EQ_OQ)) |
                 (_mm256_movemask_ps(_mm256_cmp_ps(sv1, su1, _CMP_EQ_OQ)) << 8)) &
                plane_mask);
            if (eq == plane_mask) [[unlikely]] {
                fwd_h |= coincident_flag;
                rev_h |= coincident_flag;
            }
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

__attribute__((target("avx2"))) uint32_t
group_and_hash_leaf_edges_u16_avx2(const uint32_t* leaf_points,
                                   uint64_t point_count,
                                   uint32_t neighbor_count,
                                   const uint32_t* topk_idx,
                                   const float* topk_dists,
                                   const uint16_t* topk_count,
                                   const uint16_t* global_sketches_u16,
                                   uint64_t hash_planes,
                                   uint16_t coincident_flag,
                                   VnniScratch& scratch,
                                   uint32_t* offsets,
                                   uint32_t* g_targets,
                                   uint16_t* g_dist_keys,
                                   uint16_t* g_hashes) {
    scratch.EnsureLeafSketchesU16(point_count * 16);
    uint16_t* leaf_sk = scratch.leaf_sketches_u16;
    const __m256i sign_flip = _mm256_set1_epi16(static_cast<int16_t>(0x8000));
    for (uint64_t i = 0; i < point_count; ++i) {
        if (i + 8 < point_count) {
            _mm_prefetch(reinterpret_cast<const char*>(
                             global_sketches_u16 + static_cast<uint64_t>(leaf_points[i + 8]) * 16),
                         _MM_HINT_T0);
        }
        const __m256i sk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
            global_sketches_u16 + static_cast<uint64_t>(leaf_points[i]) * 16));
        _mm256_store_si256(reinterpret_cast<__m256i*>(leaf_sk + i * 16),
                           _mm256_xor_si256(sk, sign_flip));
    }

    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;

    const auto plane_mask = static_cast<uint16_t>((1U << hash_planes) - 1U);
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const __m256i su_s =
            _mm256_load_si256(reinterpret_cast<const __m256i*>(leaf_sk + src * 16));
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const __m256i sv_s = _mm256_load_si256(
                reinterpret_cast<const __m256i*>(leaf_sk + static_cast<size_t>(dst_local) * 16));
            uint16_t fwd_h = cmp_ge_u16_mask_avx2(sv_s, su_s) & plane_mask;
            uint16_t rev_h = cmp_ge_u16_mask_avx2(su_s, sv_s) & plane_mask;
            const uint16_t eq = cmp_eq_u16_mask_avx2(sv_s, su_s) & plane_mask;
            if (eq == plane_mask) [[unlikely]] {
                fwd_h |= coincident_flag;
                rev_h |= coincident_flag;
            }
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

template <MetricKind metric_kind>
__attribute__((target("avx2"))) void
select_leaf_topk_from_dots_fp32_avx2_impl(const uint32_t* leaf_points,
                                          uint64_t point_count,
                                          const float* dot_matrix,
                                          const float* global_norms,
                                          const uint32_t* global_ids_map,
                                          bool ids_is_identity,
                                          uint32_t neighbor_count,
                                          VnniScratch& scratch) {
    const uint64_t padded_pts = ((point_count + 15ULL) & ~15ULL);
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, neighbor_count);

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t pid = leaf_points[i];
        scratch.norms[i] = (global_norms != nullptr) ? global_norms[pid] : 0.0F;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.farthest_dist[i] = SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }

    const __m256 v_zero = _mm256_setzero_ps();
    const __m256 v_one = _mm256_set1_ps(1.0F);
    const __m256 v_neg_one = _mm256_set1_ps(-1.0F);

    for (uint64_t src = 0; src < point_count; ++src) {
        const float* row = dot_matrix + src * point_count;
        const float n_src = scratch.norms[src];
        const __m256 v_nsrc = _mm256_set1_ps(n_src);
        __m256 v_far = _mm256_set1_ps(SENTINEL);

        uint64_t j = 0;
        for (; j + 8 <= point_count; j += 8) {
            const __m256 v_dot = _mm256_loadu_ps(row + j);
            __m256 v_d;
            if constexpr (metric_kind == MetricKind::L2_SQR) {
                const __m256 v_sum = _mm256_add_ps(v_nsrc, _mm256_loadu_ps(scratch.norms + j));
                v_d = _mm256_max_ps(v_zero, _mm256_sub_ps(_mm256_sub_ps(v_sum, v_dot), v_dot));
            } else if constexpr (metric_kind == MetricKind::IP) {
                v_d = _mm256_sub_ps(v_one, v_dot);
            } else {
                const __m256 v_mul =
                    _mm256_mul_ps(_mm256_mul_ps(v_dot, v_nsrc), _mm256_loadu_ps(scratch.norms + j));
                const __m256 v_sim = _mm256_min_ps(v_one, _mm256_max_ps(v_neg_one, v_mul));
                v_d = _mm256_sub_ps(v_one, v_sim);
            }
            uint32_t mask = (scratch.topk_count[src] < neighbor_count or
                             not is_finite_f32(scratch.farthest_dist[src]))
                                ? 0xFFU
                                : static_cast<uint32_t>(
                                      _mm256_movemask_ps(_mm256_cmp_ps(v_d, v_far, _CMP_LE_OQ)));
            if (src >= j and src < j + 8) {
                mask &= ~(1U << (src - j));
            }
            if (mask != 0) {
                alignas(32) float tmp_d[8];
                _mm256_store_ps(tmp_d, v_d);
                while (mask != 0) {
                    const auto lane = static_cast<uint32_t>(__builtin_ctz(mask));
                    mask &= (mask - 1U);
                    update_leaf_topk(static_cast<uint32_t>(src),
                                     static_cast<uint32_t>(j + lane),
                                     sanitize_distance(tmp_d[lane]),
                                     neighbor_count,
                                     scratch.global_ids,
                                     scratch);
                }
                v_far = _mm256_set1_ps(scratch.farthest_dist[src]);
            }
        }
        for (; j < point_count; ++j) {
            if (j == src) {
                continue;
            }
            const float d =
                finalize_scalar_distance_fp32(row[j], n_src, scratch.norms[j], metric_kind);
            if (scratch.topk_count[src] < neighbor_count or d <= scratch.farthest_dist[src] or
                not is_finite_f32(scratch.farthest_dist[src])) {
                update_leaf_topk(static_cast<uint32_t>(src),
                                 static_cast<uint32_t>(j),
                                 d,
                                 neighbor_count,
                                 scratch.global_ids,
                                 scratch);
            }
        }
        sort_leaf_topk_row(static_cast<uint32_t>(src), neighbor_count, scratch.global_ids, scratch);
    }
}

__attribute__((target("avx2"))) uint16_t
find_hash_u16_avx2(const uint16_t* hashes, uint16_t deg, uint16_t target) {
    const __m256i t = _mm256_set1_epi16(static_cast<int16_t>(target));
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 16U <= deg; i = static_cast<uint16_t>(i + 16U)) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hashes + i));
        const auto m = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, t)));
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m) >> 1));
        }
    }
    for (; i < deg; ++i) {
        if (hashes[i] == target) {
            return i;
        }
    }
    return deg;
}

__attribute__((target("avx2"))) uint16_t
find_id_u32_avx2(const uint32_t* ids, uint16_t deg, uint32_t target) {
    const __m256i t = _mm256_set1_epi32(static_cast<int32_t>(target));
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 8U <= deg; i = static_cast<uint16_t>(i + 8U)) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids + i));
        const auto m = static_cast<uint32_t>(
            _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, t))));
        if (m != 0) {
            return static_cast<uint16_t>(i + static_cast<uint16_t>(__builtin_ctz(m)));
        }
    }
    for (; i < deg; ++i) {
        if (ids[i] == target) {
            return i;
        }
    }
    return deg;
}

__attribute__((target("avx2"))) uint16_t
find_farthest_slot_u16_avx2(const uint16_t* dists,
                            const uint16_t* hashes,
                            const uint32_t* neighbors,
                            const uint32_t* ids,
                            bool ids_is_identity,
                            uint16_t deg,
                            uint16_t& out_farthest_dist) {
    __m256i vmax = _mm256_setzero_si256();
    uint16_t i = 0;
    for (; static_cast<uint32_t>(i) + 16U <= deg; i = static_cast<uint16_t>(i + 16U)) {
        vmax =
            _mm256_max_epu16(vmax, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dists + i)));
    }
    const __m128i lo128 = _mm256_castsi256_si128(vmax);
    const __m128i hi128 = _mm256_extracti128_si256(vmax, 1);
    const __m128i m128 = _mm_max_epu16(lo128, hi128);
    const __m128i inv = _mm_xor_si128(m128, _mm_set1_epi16(-1));
    const __m128i minpos = _mm_minpos_epu16(inv);
    auto max_d = static_cast<uint16_t>(~static_cast<uint32_t>(_mm_cvtsi128_si32(minpos)) & 0xFFFFU);
    for (; i < deg; ++i) {
        if (dists[i] > max_d) {
            max_d = dists[i];
        }
    }
    out_farthest_dist = max_d;

    const __m256i vt = _mm256_set1_epi16(static_cast<int16_t>(max_d));
    uint16_t best_idx = 0;
    bool have_best = false;
    uint32_t far_id = 0;
    uint16_t far_hash = 0;
    uint16_t base = 0;
    for (; static_cast<uint32_t>(base) + 16U <= deg; base = static_cast<uint16_t>(base + 16U)) {
        const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dists + base));
        auto m = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi16(v, vt)));
        while (m != 0) {
            const auto lane = static_cast<uint16_t>(__builtin_ctz(m) >> 1);
            m &= ~(3U << (lane * 2U));
            const auto idx = static_cast<uint16_t>(base + lane);
            const uint32_t cid = ids_is_identity ? neighbors[idx] : ids[neighbors[idx]];
            const uint16_t chash = hashes[idx];
            if (not have_best or cid > far_id or (cid == far_id and chash > far_hash)) {
                best_idx = idx;
                far_id = cid;
                far_hash = chash;
                have_best = true;
            }
        }
    }
    for (; base < deg; ++base) {
        if (dists[base] == max_d) {
            const uint32_t cid = ids_is_identity ? neighbors[base] : ids[neighbors[base]];
            const uint16_t chash = hashes[base];
            if (not have_best or cid > far_id or (cid == far_id and chash > far_hash)) {
                best_idx = base;
                far_id = cid;
                far_hash = chash;
                have_best = true;
            }
        }
    }
    return best_idx;
}

__attribute__((target("avx2"))) void
quantize_row_sq8_local_avx2(const float* src,
                            uint64_t dim,
                            uint64_t stride,
                            float pre_scale,
                            MetricKind metric,
                            uint8_t* dst_u8,
                            QMeta& out_meta) {
    if (dim == 0 or not is_finite_f32(pre_scale)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const __m256 vpre = _mm256_set1_ps(pre_scale);
    __m256 vmin = _mm256_set1_ps(SENTINEL);
    __m256 vmax = _mm256_set1_ps(-SENTINEL);
    __m256 vnan = _mm256_setzero_ps();
    uint64_t k = 0;
    for (; k + 8 <= dim; k += 8) {
        const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(src + k), vpre);
        vnan = _mm256_or_ps(vnan, _mm256_cmp_ps(v, v, _CMP_UNORD_Q));
        vmin = _mm256_min_ps(vmin, v);
        vmax = _mm256_max_ps(vmax, v);
    }
    bool has_non_finite = (_mm256_movemask_ps(vnan) != 0);
    float min_val = src[0] * pre_scale;
    float max_val = min_val;
    if (k > 0 and not has_non_finite) {
        alignas(32) float mn_arr[8];
        alignas(32) float mx_arr[8];
        _mm256_store_ps(mn_arr, vmin);
        _mm256_store_ps(mx_arr, vmax);
        min_val = mn_arr[0];
        max_val = mx_arr[0];
        for (int i = 1; i < 8; ++i) {
            if (mn_arr[i] < min_val) {
                min_val = mn_arr[i];
            }
            if (mx_arr[i] > max_val) {
                max_val = mx_arr[i];
            }
        }
    }
    for (; k < dim and not has_non_finite; ++k) {
        const float v = src[k] * pre_scale;
        if (not is_finite_f32(v)) {
            has_non_finite = true;
            break;
        }
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }
    if (has_non_finite or not is_finite_f32(min_val) or not is_finite_f32(max_val)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    const auto step = static_cast<float>(range / 255.0);
    if (range <= 0.0 or step <= 0.0F or not is_finite_f32(step)) {
        std::memset(dst_u8, 0, stride);
        finalize_quantize_meta(dim, min_val, 0.0F, 0, 0, metric, out_meta);
        return;
    }

    const auto inv_step = static_cast<float>(255.0 / range);
    const __m256 vmin_v = _mm256_set1_ps(min_val);
    const __m256 vinv = _mm256_set1_ps(inv_step);
    const __m256 vzero = _mm256_setzero_ps();
    const __m256 v255 = _mm256_set1_ps(255.0F);
    __m256i vsum = _mm256_setzero_si256();
    __m256i vsum2 = _mm256_setzero_si256();

    k = 0;
    for (; k + 8 <= dim; k += 8) {
        __m256 v = _mm256_sub_ps(_mm256_mul_ps(_mm256_loadu_ps(src + k), vpre), vmin_v);
        v = _mm256_mul_ps(v, vinv);
        v = _mm256_min_ps(v255, _mm256_max_ps(vzero, v));
        const __m256i vi = _mm256_cvtps_epi32(v);
        vsum = _mm256_add_epi32(vsum, vi);
        vsum2 = _mm256_add_epi32(vsum2, _mm256_mullo_epi32(vi, vi));
        const __m128i lo = _mm256_castsi256_si128(vi);
        const __m128i hi = _mm256_extracti128_si256(vi, 1);
        const __m128i p16 = _mm_packus_epi32(lo, hi);
        const __m128i p8 = _mm_packus_epi16(p16, p16);
        const int64_t packed = _mm_cvtsi128_si64(p8);
        std::memcpy(dst_u8 + k, &packed, sizeof(packed));
    }
    alignas(32) uint32_t s1_lanes[8];
    alignas(32) uint32_t s2_lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(s1_lanes), vsum);
    _mm256_store_si256(reinterpret_cast<__m256i*>(s2_lanes), vsum2);
    uint64_t sum_u = 0;
    uint64_t sum_u2 = 0;
    for (int l = 0; l < 8; ++l) {
        sum_u += s1_lanes[l];
        sum_u2 += s2_lanes[l];
    }
    for (; k < dim; ++k) {
        const float scaled = std::clamp((src[k] * pre_scale - min_val) * inv_step, 0.0F, 255.0F);
        const auto q = static_cast<uint32_t>(std::nearbyint(scaled));
        dst_u8[k] = static_cast<uint8_t>(q);
        sum_u += q;
        sum_u2 += static_cast<uint64_t>(q) * static_cast<uint64_t>(q);
    }
    if (dim < stride) {
        std::memset(dst_u8 + dim, 0, stride - dim);
    }
    finalize_quantize_meta(dim, min_val, step, sum_u, sum_u2, metric, out_meta);
}

#if VSAG_PIPNN_COMPILE_AVX_VNNI
__attribute__((target("avx2,avxvnni"))) int32_t
sq8_dot_1x1_avx_vnni(const uint8_t* a, const int8_t* b_i8, uint64_t dim_pad32) {
    __m256i acc = _mm256_setzero_si256();
    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8 + k));
        acc = _mm256_dpbusd_epi32(acc, va, vb);
    }
    return hsum256_i32(acc);
}

__attribute__((target("avx2,avxvnni"))) int32_t
sq8_dot_1x1_raw_b_avx_vnni(const uint8_t* a, const uint8_t* b_u8, uint64_t dim_pad32) {
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    __m256i acc = _mm256_setzero_si256();
    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + k));
        const __m256i vb =
            _mm256_xor_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_u8 + k)), flip);
        acc = _mm256_dpbusd_epi32(acc, va, vb);
    }
    return hsum256_i32(acc);
}

__attribute__((target("avx2,avxvnni"))) void
sq8_dot_4x2_avx_vnni(const uint8_t* const a[4],
                     const int8_t* const b_i8[2],
                     uint64_t dim_pad32,
                     int32_t out[4][2]) {
    __m256i c00 = _mm256_setzero_si256();
    __m256i c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256();
    __m256i c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256();
    __m256i c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256();
    __m256i c31 = _mm256_setzero_si256();

    for (uint64_t k = 0; k < dim_pad32; k += 32) {
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[0] + k));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_i8[1] + k));

        const __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[0] + k));
        c00 = _mm256_dpbusd_epi32(c00, a0, b0);
        c01 = _mm256_dpbusd_epi32(c01, a0, b1);

        const __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[1] + k));
        c10 = _mm256_dpbusd_epi32(c10, a1, b0);
        c11 = _mm256_dpbusd_epi32(c11, a1, b1);

        const __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[2] + k));
        c20 = _mm256_dpbusd_epi32(c20, a2, b0);
        c21 = _mm256_dpbusd_epi32(c21, a2, b1);

        const __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[3] + k));
        c30 = _mm256_dpbusd_epi32(c30, a3, b0);
        c31 = _mm256_dpbusd_epi32(c31, a3, b1);
    }

    out[0][0] = hsum256_i32(c00);
    out[0][1] = hsum256_i32(c01);
    out[1][0] = hsum256_i32(c10);
    out[1][1] = hsum256_i32(c11);
    out[2][0] = hsum256_i32(c20);
    out[2][1] = hsum256_i32(c21);
    out[3][0] = hsum256_i32(c30);
    out[3][1] = hsum256_i32(c31);
}
#endif

__attribute__((target("avx2"))) int32_t
sq8_dot_1x1_avx2(const uint8_t* a, const int8_t* b_i8, uint64_t dim_pad16) {
    __m256i acc = _mm256_setzero_si256();
    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const __m256i va =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k)));
        const __m256i vb =
            _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b_i8 + k)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    return hsum256_i32(acc);
}

__attribute__((target("avx2"))) int32_t
sq8_dot_1x1_raw_b_avx2(const uint8_t* a, const uint8_t* b_u8, uint64_t dim_pad16) {
    const __m128i flip = _mm_set1_epi8(static_cast<char>(0x80));
    __m256i acc = _mm256_setzero_si256();
    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const __m256i va =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + k)));
        const __m128i br =
            _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b_u8 + k)), flip);
        const __m256i vb = _mm256_cvtepi8_epi16(br);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    return hsum256_i32(acc);
}

__attribute__((target("avx2"))) void
sq8_dot_4x2_avx2(const uint8_t* const a[4],
                 const int8_t* const b_i8[2],
                 uint64_t dim_pad16,
                 int32_t out[4][2]) {
    __m256i c00 = _mm256_setzero_si256();
    __m256i c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256();
    __m256i c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256();
    __m256i c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256();
    __m256i c31 = _mm256_setzero_si256();

    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const __m256i b0 =
            _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b_i8[0] + k)));
        const __m256i b1 =
            _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b_i8[1] + k)));

        const __m256i a0 =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a[0] + k)));
        c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(a0, b0));
        c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(a0, b1));

        const __m256i a1 =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a[1] + k)));
        c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(a1, b0));
        c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(a1, b1));

        const __m256i a2 =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a[2] + k)));
        c20 = _mm256_add_epi32(c20, _mm256_madd_epi16(a2, b0));
        c21 = _mm256_add_epi32(c21, _mm256_madd_epi16(a2, b1));

        const __m256i a3 =
            _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a[3] + k)));
        c30 = _mm256_add_epi32(c30, _mm256_madd_epi16(a3, b0));
        c31 = _mm256_add_epi32(c31, _mm256_madd_epi16(a3, b1));
    }

    out[0][0] = hsum256_i32(c00);
    out[0][1] = hsum256_i32(c01);
    out[1][0] = hsum256_i32(c10);
    out[1][1] = hsum256_i32(c11);
    out[2][0] = hsum256_i32(c20);
    out[2][1] = hsum256_i32(c21);
    out[3][0] = hsum256_i32(c30);
    out[3][1] = hsum256_i32(c31);
}

struct Avx512BwBackend {
    static constexpr bool SIGNED_B = true;
    static constexpr uint64_t COL_TILE = 4;
    static constexpr uint64_t PAD_ALIGN = 32;
    using AccType = int32_t;
    using BPtr = const int8_t*;

    static int32_t
    Dot1x1(const uint8_t* a, const int8_t* b, uint64_t /*dim*/, uint64_t pad) {
        return sq8_dot_1x1_avx512_bw(a, b, pad);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const int8_t* const b[4],
           uint64_t /*dim*/,
           uint64_t pad,
           int32_t out[4][4]) {
        sq8_dot_4x4_avx512_bw(a, b, pad, out);
    }
};

#if VSAG_PIPNN_COMPILE_AVX_VNNI
struct AvxVnniBackend {
    static constexpr bool SIGNED_B = true;
    static constexpr uint64_t COL_TILE = 2;
    static constexpr uint64_t PAD_ALIGN = 32;
    using AccType = int32_t;
    using BPtr = const int8_t*;

    static int32_t
    Dot1x1(const uint8_t* a, const int8_t* b, uint64_t /*dim*/, uint64_t pad) {
        return sq8_dot_1x1_avx_vnni(a, b, pad);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const int8_t* const b[2],
           uint64_t /*dim*/,
           uint64_t pad,
           int32_t out[4][2]) {
        sq8_dot_4x2_avx_vnni(a, b, pad, out);
    }
};
#endif

struct Avx2Backend {
    static constexpr bool SIGNED_B = true;
    static constexpr uint64_t COL_TILE = 2;
    static constexpr uint64_t PAD_ALIGN = 16;
    using AccType = int32_t;
    using BPtr = const int8_t*;

    static int32_t
    Dot1x1(const uint8_t* a, const int8_t* b, uint64_t /*dim*/, uint64_t pad) {
        return sq8_dot_1x1_avx2(a, b, pad);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const int8_t* const b[2],
           uint64_t /*dim*/,
           uint64_t pad,
           int32_t out[4][2]) {
        sq8_dot_4x2_avx2(a, b, pad, out);
    }
};

#endif  // defined(__x86_64__) || defined(_M_X64)

// =============================================================================
// ARM64 (AArch64) NEON DotProd & Baseline NEON Kernels
// =============================================================================
#ifdef __aarch64__

void
quantize_row_sq8_local_neon(const float* src,
                            uint64_t dim,
                            uint64_t stride,
                            float pre_scale,
                            MetricKind metric,
                            uint8_t* dst_u8,
                            QMeta& out_meta) {
    if (dim == 0 or not is_finite_f32(pre_scale)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const float32x4_t vpre = vdupq_n_f32(pre_scale);
    float32x4_t vmin = vdupq_n_f32(SENTINEL);
    float32x4_t vmax = vdupq_n_f32(-SENTINEL);
    uint32x4_t vnan = vdupq_n_u32(0);
    uint64_t k = 0;
    for (; k + 4 <= dim; k += 4) {
        const float32x4_t v = vmulq_f32(vld1q_f32(src + k), vpre);
        vnan = vorrq_u32(vnan, vmvnq_u32(vceqq_f32(v, v)));
        vmin = vminq_f32(vmin, v);
        vmax = vmaxq_f32(vmax, v);
    }
    bool has_non_finite = (vmaxvq_u32(vnan) != 0);
    float min_val = (k > 0) ? vminvq_f32(vmin) : (src[0] * pre_scale);
    float max_val = (k > 0) ? vmaxvq_f32(vmax) : (src[0] * pre_scale);
    for (; k < dim and not has_non_finite; ++k) {
        const float v = src[k] * pre_scale;
        if (not is_finite_f32(v)) {
            has_non_finite = true;
            break;
        }
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }
    if (has_non_finite or not is_finite_f32(min_val) or not is_finite_f32(max_val)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    const auto step = static_cast<float>(range / 255.0);
    if (range <= 0.0 or step <= 0.0F or not is_finite_f32(step)) {
        std::memset(dst_u8, 0, stride);
        finalize_quantize_meta(dim, min_val, 0.0F, 0, 0, metric, out_meta);
        return;
    }
    const auto inv_step = static_cast<float>(255.0 / range);
    const float32x4_t vmin_v = vdupq_n_f32(min_val);
    const float32x4_t vinv = vdupq_n_f32(inv_step);
    const float32x4_t vzero = vdupq_n_f32(0.0F);
    const float32x4_t v255 = vdupq_n_f32(255.0F);
    uint32x4_t vsum = vdupq_n_u32(0);
    uint32x4_t vsum2 = vdupq_n_u32(0);

    k = 0;
    for (; k + 4 <= dim; k += 4) {
        float32x4_t v = vsubq_f32(vmulq_f32(vld1q_f32(src + k), vpre), vmin_v);
        v = vmulq_f32(v, vinv);
        v = vminq_f32(v255, vmaxq_f32(vzero, v));
        const uint32x4_t vi = vcvtnq_u32_f32(v);
        vsum = vaddq_u32(vsum, vi);
        vsum2 = vmlaq_u32(vsum2, vi, vi);
        const uint16x4_t v16 = vmovn_u32(vi);
        const uint8x8_t v8 = vmovn_u16(vcombine_u16(v16, v16));
        const uint32_t packed = vget_lane_u32(vreinterpret_u32_u8(v8), 0);
        std::memcpy(dst_u8 + k, &packed, sizeof(packed));
    }
    uint32_t s1_lanes[4];
    uint32_t s2_lanes[4];
    vst1q_u32(s1_lanes, vsum);
    vst1q_u32(s2_lanes, vsum2);
    uint64_t sum_u = 0;
    uint64_t sum_u2 = 0;
    for (int l = 0; l < 4; ++l) {
        sum_u += s1_lanes[l];
        sum_u2 += s2_lanes[l];
    }
    for (; k < dim; ++k) {
        const float scaled = std::clamp((src[k] * pre_scale - min_val) * inv_step, 0.0F, 255.0F);
        const auto q = static_cast<uint32_t>(std::nearbyint(scaled));
        dst_u8[k] = static_cast<uint8_t>(q);
        sum_u += q;
        sum_u2 += static_cast<uint64_t>(q) * static_cast<uint64_t>(q);
    }
    if (dim < stride) {
        std::memset(dst_u8 + dim, 0, stride - dim);
    }
    finalize_quantize_meta(dim, min_val, step, sum_u, sum_u2, metric, out_meta);
}

#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
#ifndef __ARM_FEATURE_DOTPROD
__attribute__((target("+dotprod")))
#endif
uint32_t
sq8_dot_1x1_neon_dotprod(const uint8_t* a, const uint8_t* b_u8, uint64_t dim_pad16) {
    uint32x4_t acc = vdupq_n_u32(0);
    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        acc = vdotq_u32(acc, vld1q_u8(a + k), vld1q_u8(b_u8 + k));
    }
    return vaddvq_u32(acc);
}

#ifndef __ARM_FEATURE_DOTPROD
__attribute__((target("+dotprod")))
#endif
void
sq8_dot_4x4_neon_dotprod(const uint8_t* const a[4],
                         const uint8_t* const b_u8[4],
                         uint64_t dim_pad16,
                         uint32_t out[4][4]) {
    uint32x4_t c00 = vdupq_n_u32(0);
    uint32x4_t c01 = vdupq_n_u32(0);
    uint32x4_t c02 = vdupq_n_u32(0);
    uint32x4_t c03 = vdupq_n_u32(0);
    uint32x4_t c10 = vdupq_n_u32(0);
    uint32x4_t c11 = vdupq_n_u32(0);
    uint32x4_t c12 = vdupq_n_u32(0);
    uint32x4_t c13 = vdupq_n_u32(0);
    uint32x4_t c20 = vdupq_n_u32(0);
    uint32x4_t c21 = vdupq_n_u32(0);
    uint32x4_t c22 = vdupq_n_u32(0);
    uint32x4_t c23 = vdupq_n_u32(0);
    uint32x4_t c30 = vdupq_n_u32(0);
    uint32x4_t c31 = vdupq_n_u32(0);
    uint32x4_t c32 = vdupq_n_u32(0);
    uint32x4_t c33 = vdupq_n_u32(0);

    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const uint8x16_t b0 = vld1q_u8(b_u8[0] + k);
        const uint8x16_t b1 = vld1q_u8(b_u8[1] + k);
        const uint8x16_t b2 = vld1q_u8(b_u8[2] + k);
        const uint8x16_t b3 = vld1q_u8(b_u8[3] + k);

        const uint8x16_t a0 = vld1q_u8(a[0] + k);
        c00 = vdotq_u32(c00, a0, b0);
        c01 = vdotq_u32(c01, a0, b1);
        c02 = vdotq_u32(c02, a0, b2);
        c03 = vdotq_u32(c03, a0, b3);

        const uint8x16_t a1 = vld1q_u8(a[1] + k);
        c10 = vdotq_u32(c10, a1, b0);
        c11 = vdotq_u32(c11, a1, b1);
        c12 = vdotq_u32(c12, a1, b2);
        c13 = vdotq_u32(c13, a1, b3);

        const uint8x16_t a2 = vld1q_u8(a[2] + k);
        c20 = vdotq_u32(c20, a2, b0);
        c21 = vdotq_u32(c21, a2, b1);
        c22 = vdotq_u32(c22, a2, b2);
        c23 = vdotq_u32(c23, a2, b3);

        const uint8x16_t a3 = vld1q_u8(a[3] + k);
        c30 = vdotq_u32(c30, a3, b0);
        c31 = vdotq_u32(c31, a3, b1);
        c32 = vdotq_u32(c32, a3, b2);
        c33 = vdotq_u32(c33, a3, b3);
    }

    out[0][0] = vaddvq_u32(c00);
    out[0][1] = vaddvq_u32(c01);
    out[0][2] = vaddvq_u32(c02);
    out[0][3] = vaddvq_u32(c03);
    out[1][0] = vaddvq_u32(c10);
    out[1][1] = vaddvq_u32(c11);
    out[1][2] = vaddvq_u32(c12);
    out[1][3] = vaddvq_u32(c13);
    out[2][0] = vaddvq_u32(c20);
    out[2][1] = vaddvq_u32(c21);
    out[2][2] = vaddvq_u32(c22);
    out[2][3] = vaddvq_u32(c23);
    out[3][0] = vaddvq_u32(c30);
    out[3][1] = vaddvq_u32(c31);
    out[3][2] = vaddvq_u32(c32);
    out[3][3] = vaddvq_u32(c33);
}

struct NeonDotProdBackend {
    static constexpr bool SIGNED_B = false;
    static constexpr uint64_t COL_TILE = 4;
    static constexpr uint64_t PAD_ALIGN = 16;
    using AccType = uint32_t;
    using BPtr = const uint8_t*;

    static uint32_t
    Dot1x1(const uint8_t* a, const uint8_t* b, uint64_t /*dim*/, uint64_t pad) {
        return sq8_dot_1x1_neon_dotprod(a, b, pad);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const uint8_t* const b[4],
           uint64_t /*dim*/,
           uint64_t pad,
           uint32_t out[4][4]) {
        sq8_dot_4x4_neon_dotprod(a, b, pad, out);
    }
};
#endif  // VSAG_PIPNN_COMPILE_NEON_DOTPROD

uint32_t
sq8_dot_1x1_neon(const uint8_t* a, const uint8_t* b_u8, uint64_t dim_pad16) {
    uint32x4_t acc = vdupq_n_u32(0);
    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const uint8x16_t va = vld1q_u8(a + k);
        const uint8x16_t vb = vld1q_u8(b_u8 + k);
        acc = vpadalq_u16(acc, vmull_u8(vget_low_u8(va), vget_low_u8(vb)));
        acc = vpadalq_u16(acc, vmull_u8(vget_high_u8(va), vget_high_u8(vb)));
    }
    return vaddvq_u32(acc);
}

void
sq8_dot_4x2_neon(const uint8_t* const a[4],
                 const uint8_t* const b_u8[2],
                 uint64_t dim_pad16,
                 uint32_t out[4][2]) {
    uint32x4_t c00 = vdupq_n_u32(0);
    uint32x4_t c01 = vdupq_n_u32(0);
    uint32x4_t c10 = vdupq_n_u32(0);
    uint32x4_t c11 = vdupq_n_u32(0);
    uint32x4_t c20 = vdupq_n_u32(0);
    uint32x4_t c21 = vdupq_n_u32(0);
    uint32x4_t c30 = vdupq_n_u32(0);
    uint32x4_t c31 = vdupq_n_u32(0);

    for (uint64_t k = 0; k < dim_pad16; k += 16) {
        const uint8x16_t b0 = vld1q_u8(b_u8[0] + k);
        const uint8x16_t b1 = vld1q_u8(b_u8[1] + k);
        const uint8x8_t b0l = vget_low_u8(b0);
        const uint8x8_t b0h = vget_high_u8(b0);
        const uint8x8_t b1l = vget_low_u8(b1);
        const uint8x8_t b1h = vget_high_u8(b1);

        const uint8x16_t a0 = vld1q_u8(a[0] + k);
        c00 = vpadalq_u16(c00, vmull_u8(vget_low_u8(a0), b0l));
        c00 = vpadalq_u16(c00, vmull_u8(vget_high_u8(a0), b0h));
        c01 = vpadalq_u16(c01, vmull_u8(vget_low_u8(a0), b1l));
        c01 = vpadalq_u16(c01, vmull_u8(vget_high_u8(a0), b1h));

        const uint8x16_t a1 = vld1q_u8(a[1] + k);
        c10 = vpadalq_u16(c10, vmull_u8(vget_low_u8(a1), b0l));
        c10 = vpadalq_u16(c10, vmull_u8(vget_high_u8(a1), b0h));
        c11 = vpadalq_u16(c11, vmull_u8(vget_low_u8(a1), b1l));
        c11 = vpadalq_u16(c11, vmull_u8(vget_high_u8(a1), b1h));

        const uint8x16_t a2 = vld1q_u8(a[2] + k);
        c20 = vpadalq_u16(c20, vmull_u8(vget_low_u8(a2), b0l));
        c20 = vpadalq_u16(c20, vmull_u8(vget_high_u8(a2), b0h));
        c21 = vpadalq_u16(c21, vmull_u8(vget_low_u8(a2), b1l));
        c21 = vpadalq_u16(c21, vmull_u8(vget_high_u8(a2), b1h));

        const uint8x16_t a3 = vld1q_u8(a[3] + k);
        c30 = vpadalq_u16(c30, vmull_u8(vget_low_u8(a3), b0l));
        c30 = vpadalq_u16(c30, vmull_u8(vget_high_u8(a3), b0h));
        c31 = vpadalq_u16(c31, vmull_u8(vget_low_u8(a3), b1l));
        c31 = vpadalq_u16(c31, vmull_u8(vget_high_u8(a3), b1h));
    }

    out[0][0] = vaddvq_u32(c00);
    out[0][1] = vaddvq_u32(c01);
    out[1][0] = vaddvq_u32(c10);
    out[1][1] = vaddvq_u32(c11);
    out[2][0] = vaddvq_u32(c20);
    out[2][1] = vaddvq_u32(c21);
    out[3][0] = vaddvq_u32(c30);
    out[3][1] = vaddvq_u32(c31);
}

struct NeonBackend {
    static constexpr bool SIGNED_B = false;
    static constexpr uint64_t COL_TILE = 2;
    static constexpr uint64_t PAD_ALIGN = 16;
    using AccType = uint32_t;
    using BPtr = const uint8_t*;

    static uint32_t
    Dot1x1(const uint8_t* a, const uint8_t* b, uint64_t /*dim*/, uint64_t pad) {
        return sq8_dot_1x1_neon(a, b, pad);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const uint8_t* const b[2],
           uint64_t /*dim*/,
           uint64_t pad,
           uint32_t out[4][2]) {
        sq8_dot_4x2_neon(a, b, pad, out);
    }
};

#endif  // defined(__aarch64__)

// =============================================================================
// Portable Scalar sq8_local & FP32 Fallbacks
// =============================================================================
uint32_t
sq8_dot_1x1_scalar(const uint8_t* a, const uint8_t* b_u8, uint64_t dim) {
    uint32_t acc = 0;
    for (uint64_t k = 0; k < dim; ++k) {
        acc += static_cast<uint32_t>(a[k]) * static_cast<uint32_t>(b_u8[k]);
    }
    return acc;
}

void
sq8_dot_4x2_scalar(const uint8_t* const a[4],
                   const uint8_t* const b_u8[2],
                   uint64_t dim,
                   uint32_t out[4][2]) {
    uint32_t c00 = 0;
    uint32_t c01 = 0;
    uint32_t c10 = 0;
    uint32_t c11 = 0;
    uint32_t c20 = 0;
    uint32_t c21 = 0;
    uint32_t c30 = 0;
    uint32_t c31 = 0;
    for (uint64_t k = 0; k < dim; ++k) {
        const uint32_t b0 = b_u8[0][k];
        const uint32_t b1 = b_u8[1][k];
        const uint32_t a0 = a[0][k];
        const uint32_t a1 = a[1][k];
        const uint32_t a2 = a[2][k];
        const uint32_t a3 = a[3][k];
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
        c20 += a2 * b0;
        c21 += a2 * b1;
        c30 += a3 * b0;
        c31 += a3 * b1;
    }
    out[0][0] = c00;
    out[0][1] = c01;
    out[1][0] = c10;
    out[1][1] = c11;
    out[2][0] = c20;
    out[2][1] = c21;
    out[3][0] = c30;
    out[3][1] = c31;
}

struct ScalarBackend {
    static constexpr bool SIGNED_B = false;
    static constexpr uint64_t COL_TILE = 2;
    static constexpr uint64_t PAD_ALIGN = 1;
    using AccType = uint32_t;
    using BPtr = const uint8_t*;

    static uint32_t
    Dot1x1(const uint8_t* a, const uint8_t* b, uint64_t dim, uint64_t /*pad*/) {
        return sq8_dot_1x1_scalar(a, b, dim);
    }

    static void
    Dot4xN(const uint8_t* const a[4],
           const uint8_t* const b[2],
           uint64_t dim,
           uint64_t /*pad*/,
           uint32_t out[4][2]) {
        sq8_dot_4x2_scalar(a, b, dim, out);
    }
};

void
quantize_row_sq8_local_scalar(const float* src,
                              uint64_t dim,
                              uint64_t stride,
                              float pre_scale,
                              MetricKind metric,
                              uint8_t* dst_u8,
                              QMeta& out_meta) {
    if (dim == 0 or not is_finite_f32(pre_scale)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    float min_val = src[0] * pre_scale;
    float max_val = min_val;
    bool has_non_finite = not is_finite_f32(min_val);
    for (uint64_t k = 1; k < dim and not has_non_finite; ++k) {
        const float v = src[k] * pre_scale;
        if (not is_finite_f32(v)) {
            has_non_finite = true;
            break;
        }
        if (v < min_val) {
            min_val = v;
        }
        if (v > max_val) {
            max_val = v;
        }
    }
    if (has_non_finite or not is_finite_f32(min_val) or not is_finite_f32(max_val)) {
        std::memset(dst_u8, 0, stride);
        out_meta = QMeta{};
        return;
    }
    const double range = static_cast<double>(max_val) - static_cast<double>(min_val);
    const auto step = static_cast<float>(range / 255.0);
    if (range <= 0.0 or step <= 0.0F or not is_finite_f32(step)) {
        std::memset(dst_u8, 0, stride);
        finalize_quantize_meta(dim, min_val, 0.0F, 0, 0, metric, out_meta);
        return;
    }
    const auto inv_step = static_cast<float>(255.0 / range);
    uint64_t sum_u = 0;
    uint64_t sum_u2 = 0;
    for (uint64_t k = 0; k < dim; ++k) {
        const float scaled = std::clamp((src[k] * pre_scale - min_val) * inv_step, 0.0F, 255.0F);
        const auto q = static_cast<uint32_t>(std::nearbyint(scaled));
        dst_u8[k] = static_cast<uint8_t>(q);
        sum_u += q;
        sum_u2 += static_cast<uint64_t>(q) * static_cast<uint64_t>(q);
    }
    if (dim < stride) {
        std::memset(dst_u8 + dim, 0, stride - dim);
    }
    finalize_quantize_meta(dim, min_val, step, sum_u, sum_u2, metric, out_meta);
}

void
select_leaf_topk_from_dots_fp32_scalar(const uint32_t* leaf_points,
                                       uint64_t point_count,
                                       const float* dot_matrix,
                                       const float* global_norms,
                                       const uint32_t* global_ids_map,
                                       bool ids_is_identity,
                                       MetricKind metric,
                                       uint32_t neighbor_count,
                                       VnniScratch& scratch) {
    const uint64_t padded_pts = ((point_count + 15ULL) & ~15ULL);
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, neighbor_count);

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t pid = leaf_points[i];
        scratch.norms[i] = (global_norms != nullptr) ? global_norms[pid] : 0.0F;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.farthest_dist[i] = SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }
    for (uint64_t src = 0; src < point_count; ++src) {
        const float* row = dot_matrix + src * point_count;
        const float n_src = scratch.norms[src];
        for (uint64_t j = 0; j < point_count; ++j) {
            if (j == src) {
                continue;
            }
            const float d = finalize_scalar_distance_fp32(row[j], n_src, scratch.norms[j], metric);
            if (scratch.topk_count[src] < neighbor_count or d <= scratch.farthest_dist[src] or
                not is_finite_f32(scratch.farthest_dist[src])) {
                update_leaf_topk(static_cast<uint32_t>(src),
                                 static_cast<uint32_t>(j),
                                 d,
                                 neighbor_count,
                                 scratch.global_ids,
                                 scratch);
            }
        }
        sort_leaf_topk_row(static_cast<uint32_t>(src), neighbor_count, scratch.global_ids, scratch);
    }
}

// =============================================================================
// Shared Generic Blocked Drivers for All Non-AVX512-VNNI Backends
// (AVX-512BW, AVX-VNNI, AVX2, NEON DotProd, Baseline NEON, Portable Scalar)
// =============================================================================
inline uint64_t
align_pad(uint64_t dim, uint64_t align) {
    if (align <= 1) {
        return dim;
    }
    return (dim + align - 1ULL) & ~(align - 1ULL);
}

inline void
copy_and_flip_u8_to_i8(const uint8_t* src, int8_t* dst, uint64_t bytes) {
    uint64_t k = 0;
    constexpr uint64_t flip64 = 0x8080808080808080ULL;
    for (; k + 8 <= bytes; k += 8) {
        uint64_t v = 0;
        std::memcpy(&v, src + k, sizeof(v));
        v ^= flip64;
        std::memcpy(dst + k, &v, sizeof(v));
    }
    for (; k < bytes; ++k) {
        dst[k] = static_cast<int8_t>(src[k] ^ 0x80U);
    }
}

template <typename Backend>
inline float
eval_backend_dist(typename Backend::AccType acc,
                  float step_a,
                  float min_a,
                  float su_a,
                  float norm_a,
                  float step_b,
                  float c_or_min_b,
                  float sum_x_b,
                  float norm_b,
                  MetricKind metric) {
    float dot = 0.0F;
    if constexpr (Backend::SIGNED_B) {
        dot = sq8_dot_from_signed_acc(acc, step_a, min_a, su_a, step_b, c_or_min_b, sum_x_b);
    } else {
        dot = sq8_dot_from_unsigned_acc(acc, step_a, min_a, su_a, step_b, c_or_min_b, sum_x_b);
    }
    return finalize_scalar_distance(dot, norm_a, norm_b, metric);
}

template <typename Backend>
void
fused_leaf_topk_generic(const uint32_t* leaf_points,
                        uint64_t point_count,
                        const uint8_t* q_data_u8,
                        const QMeta* q_meta,
                        const uint32_t* global_ids_map,
                        bool ids_is_identity,
                        uint64_t dim,
                        uint64_t q_stride,
                        MetricKind metric,
                        uint32_t neighbor_count,
                        VnniScratch& scratch) {
    const auto dim_f = static_cast<float>(dim);
    const uint64_t pad = align_pad(dim, Backend::PAD_ALIGN);
    const uint64_t padded_pts = ((point_count + 15ULL) & ~15ULL);

    scratch.EnsureU8(point_count * pad);
    if constexpr (Backend::SIGNED_B) {
        scratch.EnsurePanels(point_count * pad);
    }
    scratch.EnsureMeta(padded_pts);
    scratch.EnsureTopK(padded_pts, neighbor_count);

    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t pid = leaf_points[i];
        const uint8_t* src = q_data_u8 + static_cast<size_t>(pid) * q_stride;
        std::memcpy(scratch.u8_buf + i * pad, src, pad);
        if constexpr (Backend::SIGNED_B) {
            copy_and_flip_u8_to_i8(src, scratch.panel_buf + i * pad, pad);
        }
        const QMeta& m = q_meta[pid];
        scratch.steps[i] = m.step;
        scratch.min_vals[i] = m.min_val;
        scratch.su[i] = m.su;
        scratch.c_vals[i] = Backend::SIGNED_B ? (m.min_val + 128.0F * m.step) : m.min_val;
        scratch.sum_x[i] = dim_f * m.min_val + m.su;
        scratch.norms[i] = m.norm;
        scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
        scratch.farthest_dist[i] = SENTINEL;
        scratch.topk_count[i] = 0;
        scratch.topk_farthest_pos[i] = 0;
    }

    auto get_b_ptr = [&](uint64_t idx) -> typename Backend::BPtr {
        if constexpr (Backend::SIGNED_B) {
            return scratch.panel_buf + idx * pad;
        } else {
            return scratch.u8_buf + idx * pad;
        }
    };

    auto update_pair = [&](uint64_t i, uint64_t j, typename Backend::AccType acc) {
        const float d = eval_backend_dist<Backend>(acc,
                                                   scratch.steps[i],
                                                   scratch.min_vals[i],
                                                   scratch.su[i],
                                                   scratch.norms[i],
                                                   scratch.steps[j],
                                                   scratch.c_vals[j],
                                                   scratch.sum_x[j],
                                                   scratch.norms[j],
                                                   metric);
        if (d <= scratch.farthest_dist[i]) {
            update_leaf_topk(static_cast<uint32_t>(i),
                             static_cast<uint32_t>(j),
                             d,
                             neighbor_count,
                             scratch.global_ids,
                             scratch);
        }
        if (d <= scratch.farthest_dist[j]) {
            update_leaf_topk(static_cast<uint32_t>(j),
                             static_cast<uint32_t>(i),
                             d,
                             neighbor_count,
                             scratch.global_ids,
                             scratch);
        }
    };

    constexpr uint64_t col_tile = Backend::COL_TILE;
    for (uint64_t i0 = 0; i0 < point_count; i0 += 4) {
        const uint64_t mi = std::min<uint64_t>(4, point_count - i0);
        // Diagonal mini-triangle within [i0, i0 + mi)
        for (uint64_t r = 0; r < mi; ++r) {
            const uint64_t i = i0 + r;
            const uint8_t* a_ptr = scratch.u8_buf + i * pad;
            for (uint64_t c = r + 1; c < mi; ++c) {
                const uint64_t j = i0 + c;
                update_pair(i, j, Backend::Dot1x1(a_ptr, get_b_ptr(j), dim, pad));
            }
        }
        if (mi == 4) {
            const uint8_t* a_ptrs[4] = {
                scratch.u8_buf + (i0 + 0) * pad,
                scratch.u8_buf + (i0 + 1) * pad,
                scratch.u8_buf + (i0 + 2) * pad,
                scratch.u8_buf + (i0 + 3) * pad,
            };
            uint64_t j0 = i0 + 4;
            for (; j0 + col_tile <= point_count; j0 += col_tile) {
                typename Backend::BPtr b_ptrs[col_tile];
                for (uint64_t c = 0; c < col_tile; ++c) {
                    b_ptrs[c] = get_b_ptr(j0 + c);
                }
                typename Backend::AccType acc_out[4][col_tile];
                Backend::Dot4xN(a_ptrs, b_ptrs, dim, pad, acc_out);
                for (uint64_t r = 0; r < 4; ++r) {
                    for (uint64_t c = 0; c < col_tile; ++c) {
                        update_pair(i0 + r, j0 + c, acc_out[r][c]);
                    }
                }
            }
            for (; j0 < point_count; ++j0) {
                const auto b_ptr = get_b_ptr(j0);
                for (uint64_t r = 0; r < 4; ++r) {
                    update_pair(i0 + r, j0, Backend::Dot1x1(a_ptrs[r], b_ptr, dim, pad));
                }
            }
        } else {
            for (uint64_t r = 0; r < mi; ++r) {
                const uint64_t i = i0 + r;
                const uint8_t* a_ptr = scratch.u8_buf + i * pad;
                for (uint64_t j = i0 + mi; j < point_count; ++j) {
                    update_pair(i, j, Backend::Dot1x1(a_ptr, get_b_ptr(j), dim, pad));
                }
            }
        }
    }

    for (uint64_t i = 0; i < point_count; ++i) {
        sort_leaf_topk_row(static_cast<uint32_t>(i), neighbor_count, scratch.global_ids, scratch);
    }
}

template <typename Backend>
void
assign_stripe_to_leaders_generic(const uint32_t* points,
                                 uint64_t stripe_size,
                                 const VnniScratch& leader_scratch,
                                 uint64_t leader_count,
                                 const uint8_t* q_data_u8,
                                 const QMeta* q_meta,
                                 uint64_t dim,
                                 uint64_t q_stride,
                                 MetricKind metric,
                                 uint32_t fanout,
                                 uint32_t* out_assignments,
                                 VnniScratch& worker_scratch) {
    const uint64_t pad = align_pad(dim, Backend::PAD_ALIGN);
    const uint64_t leader_row_stride = align_pad(dim, 32);

    worker_scratch.EnsureTopK(stripe_size, fanout);
    for (uint64_t i = 0; i < stripe_size; ++i) {
        worker_scratch.farthest_dist[i] = SENTINEL;
        worker_scratch.topk_count[i] = 0;
        worker_scratch.topk_farthest_pos[i] = 0;
    }

    auto get_b_ptr = [&](uint64_t leader_idx) -> typename Backend::BPtr {
        if constexpr (Backend::SIGNED_B) {
            return leader_scratch.panel_buf + leader_idx * leader_row_stride;
        } else {
            return leader_scratch.u8_buf + leader_idx * leader_row_stride;
        }
    };

    auto consider = [&](uint64_t qi, const QMeta& mq, uint64_t li, typename Backend::AccType acc) {
        const float c_or_min =
            Backend::SIGNED_B ? leader_scratch.c_vals[li] : leader_scratch.min_vals[li];
        const float d = eval_backend_dist<Backend>(acc,
                                                   mq.step,
                                                   mq.min_val,
                                                   mq.su,
                                                   mq.norm,
                                                   leader_scratch.steps[li],
                                                   c_or_min,
                                                   leader_scratch.sum_x[li],
                                                   leader_scratch.norms[li],
                                                   metric);
        if (d <= worker_scratch.farthest_dist[qi]) {
            update_leader_topk(static_cast<uint32_t>(qi),
                               static_cast<uint32_t>(li),
                               d,
                               fanout,
                               leader_scratch.global_ids,
                               worker_scratch);
        }
    };

    constexpr uint64_t col_tile = Backend::COL_TILE;
    for (uint64_t q0 = 0; q0 < stripe_size; q0 += 4) {
        const uint64_t mq_cnt = std::min<uint64_t>(4, stripe_size - q0);
        if (mq_cnt == 4) {
            const uint32_t pids[4] = {
                points[q0 + 0], points[q0 + 1], points[q0 + 2], points[q0 + 3]};
            const QMeta metas[4] = {
                q_meta[pids[0]], q_meta[pids[1]], q_meta[pids[2]], q_meta[pids[3]]};
            const uint8_t* a_ptrs[4] = {
                q_data_u8 + static_cast<size_t>(pids[0]) * q_stride,
                q_data_u8 + static_cast<size_t>(pids[1]) * q_stride,
                q_data_u8 + static_cast<size_t>(pids[2]) * q_stride,
                q_data_u8 + static_cast<size_t>(pids[3]) * q_stride,
            };
            uint64_t l0 = 0;
            for (; l0 + col_tile <= leader_count; l0 += col_tile) {
                typename Backend::BPtr b_ptrs[col_tile];
                for (uint64_t c = 0; c < col_tile; ++c) {
                    b_ptrs[c] = get_b_ptr(l0 + c);
                }
                typename Backend::AccType acc_out[4][col_tile];
                Backend::Dot4xN(a_ptrs, b_ptrs, dim, pad, acc_out);
                for (uint64_t r = 0; r < 4; ++r) {
                    for (uint64_t c = 0; c < col_tile; ++c) {
                        consider(q0 + r, metas[r], l0 + c, acc_out[r][c]);
                    }
                }
            }
            for (; l0 < leader_count; ++l0) {
                const auto b_ptr = get_b_ptr(l0);
                for (uint64_t r = 0; r < 4; ++r) {
                    consider(q0 + r, metas[r], l0, Backend::Dot1x1(a_ptrs[r], b_ptr, dim, pad));
                }
            }
        } else {
            for (uint64_t r = 0; r < mq_cnt; ++r) {
                const uint32_t pid = points[q0 + r];
                const QMeta& mq = q_meta[pid];
                const uint8_t* a_ptr = q_data_u8 + static_cast<size_t>(pid) * q_stride;
                for (uint64_t li = 0; li < leader_count; ++li) {
                    consider(q0 + r, mq, li, Backend::Dot1x1(a_ptr, get_b_ptr(li), dim, pad));
                }
            }
        }
    }
    write_sorted_assignments(stripe_size, fanout, leader_scratch, worker_scratch, out_assignments);
}

template <typename Backend>
uint32_t
quantized_robust_prune_generic(const uint8_t* q_data_u8,
                               const QMeta* q_meta,
                               uint64_t dim,
                               uint64_t q_stride,
                               MetricKind metric,
                               const uint32_t* ordered_cand_ids,
                               const float* ordered_cand_dists,
                               uint32_t num_cands,
                               uint32_t max_degree,
                               float alpha,
                               uint32_t* out_selected) {
    if (num_cands == 0 or max_degree == 0) {
        return 0;
    }

    const auto dim_f = static_cast<float>(dim);
    const uint64_t pad = align_pad(dim, Backend::PAD_ALIGN);

    thread_local VnniScratch prune_scratch;
    if constexpr (Backend::SIGNED_B) {
        prune_scratch.EnsurePanels(static_cast<size_t>(max_degree) * pad);
    } else {
        prune_scratch.EnsureU8(static_cast<size_t>(max_degree) * pad);
    }
    prune_scratch.EnsureMeta(max_degree);

    auto get_surv_ptr = [&](uint32_t slot) -> typename Backend::BPtr {
        if constexpr (Backend::SIGNED_B) {
            return prune_scratch.panel_buf + static_cast<size_t>(slot) * pad;
        } else {
            return prune_scratch.u8_buf + static_cast<size_t>(slot) * pad;
        }
    };

    auto insert_survivor = [&](uint32_t surv_id, const uint8_t* surv_u8, uint32_t slot) {
        out_selected[slot] = surv_id;
        const QMeta& m = q_meta[surv_id];
        prune_scratch.steps[slot] = m.step;
        prune_scratch.min_vals[slot] = m.min_val;
        prune_scratch.c_vals[slot] = Backend::SIGNED_B ? (m.min_val + 128.0F * m.step) : m.min_val;
        prune_scratch.sum_x[slot] = dim_f * m.min_val + m.su;
        prune_scratch.norms[slot] = m.norm;
        if constexpr (Backend::SIGNED_B) {
            copy_and_flip_u8_to_i8(
                surv_u8, prune_scratch.panel_buf + static_cast<size_t>(slot) * pad, pad);
        } else {
            std::memcpy(prune_scratch.u8_buf + static_cast<size_t>(slot) * pad, surv_u8, pad);
        }
    };

    uint32_t selected_count = 0;
    {
        const uint32_t c0 = ordered_cand_ids[0];
        insert_survivor(c0, q_data_u8 + static_cast<size_t>(c0) * q_stride, 0);
        selected_count = 1;
        if (selected_count == max_degree) {
            return selected_count;
        }
    }

    constexpr uint64_t col_tile = Backend::COL_TILE;
    uint32_t c = 1;
    while (c < num_cands) {
        const uint32_t batch = std::min<uint32_t>(4U, num_cands - c);
        uint32_t cands[4];
        float src_dists[4];
        QMeta c_meta[4];
        const uint8_t* c_u8[4];
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t cid = ordered_cand_ids[c + b];
            cands[b] = cid;
            src_dists[b] = ordered_cand_dists[c + b];
            c_meta[b] = q_meta[cid];
            c_u8[b] = q_data_u8 + static_cast<size_t>(cid) * q_stride;
        }

        uint32_t alive = (1U << batch) - 1U;
        if (batch == 4U) {
            uint32_t s = 0;
            for (; s + col_tile <= selected_count and alive != 0;
                 s += static_cast<uint32_t>(col_tile)) {
                typename Backend::BPtr b_ptrs[col_tile];
                for (uint64_t col = 0; col < col_tile; ++col) {
                    b_ptrs[col] = get_surv_ptr(s + static_cast<uint32_t>(col));
                }
                typename Backend::AccType acc_out[4][col_tile];
                Backend::Dot4xN(c_u8, b_ptrs, dim, pad, acc_out);
                for (uint32_t b = 0; b < 4; ++b) {
                    if ((alive & (1U << b)) == 0) {
                        continue;
                    }
                    for (uint64_t col = 0; col < col_tile; ++col) {
                        const uint32_t slot = s + static_cast<uint32_t>(col);
                        const float d = eval_backend_dist<Backend>(acc_out[b][col],
                                                                   c_meta[b].step,
                                                                   c_meta[b].min_val,
                                                                   c_meta[b].su,
                                                                   c_meta[b].norm,
                                                                   prune_scratch.steps[slot],
                                                                   prune_scratch.c_vals[slot],
                                                                   prune_scratch.sum_x[slot],
                                                                   prune_scratch.norms[slot],
                                                                   metric);
                        if (alpha * d < src_dists[b]) {
                            alive &= ~(1U << b);
                            break;
                        }
                    }
                }
            }
            for (; s < selected_count and alive != 0; ++s) {
                const auto s_ptr = get_surv_ptr(s);
                for (uint32_t b = 0; b < 4; ++b) {
                    if ((alive & (1U << b)) == 0) {
                        continue;
                    }
                    const auto acc = Backend::Dot1x1(c_u8[b], s_ptr, dim, pad);
                    const float d = eval_backend_dist<Backend>(acc,
                                                               c_meta[b].step,
                                                               c_meta[b].min_val,
                                                               c_meta[b].su,
                                                               c_meta[b].norm,
                                                               prune_scratch.steps[s],
                                                               prune_scratch.c_vals[s],
                                                               prune_scratch.sum_x[s],
                                                               prune_scratch.norms[s],
                                                               metric);
                    if (alpha * d < src_dists[b]) {
                        alive &= ~(1U << b);
                    }
                }
            }
        } else {
            for (uint32_t s = 0; s < selected_count and alive != 0; ++s) {
                const auto s_ptr = get_surv_ptr(s);
                for (uint32_t b = 0; b < batch; ++b) {
                    if ((alive & (1U << b)) == 0) {
                        continue;
                    }
                    const auto acc = Backend::Dot1x1(c_u8[b], s_ptr, dim, pad);
                    const float d = eval_backend_dist<Backend>(acc,
                                                               c_meta[b].step,
                                                               c_meta[b].min_val,
                                                               c_meta[b].su,
                                                               c_meta[b].norm,
                                                               prune_scratch.steps[s],
                                                               prune_scratch.c_vals[s],
                                                               prune_scratch.sum_x[s],
                                                               prune_scratch.norms[s],
                                                               metric);
                    if (alpha * d < src_dists[b]) {
                        alive &= ~(1U << b);
                    }
                }
            }
        }

        uint32_t batch_new_slots[4];
        uint32_t batch_new_cnt = 0;
        for (uint32_t b = 0; b < batch; ++b) {
            if ((alive & (1U << b)) == 0) {
                continue;
            }
            bool pruned_intra = false;
            for (uint32_t s = 0; s < batch_new_cnt; ++s) {
                const uint32_t slot = batch_new_slots[s];
                const auto acc = Backend::Dot1x1(c_u8[b], get_surv_ptr(slot), dim, pad);
                const float d = eval_backend_dist<Backend>(acc,
                                                           c_meta[b].step,
                                                           c_meta[b].min_val,
                                                           c_meta[b].su,
                                                           c_meta[b].norm,
                                                           prune_scratch.steps[slot],
                                                           prune_scratch.c_vals[slot],
                                                           prune_scratch.sum_x[slot],
                                                           prune_scratch.norms[slot],
                                                           metric);
                if (alpha * d < src_dists[b]) {
                    pruned_intra = true;
                    break;
                }
            }
            if (not pruned_intra) {
                insert_survivor(cands[b], c_u8[b], selected_count);
                batch_new_slots[batch_new_cnt++] = selected_count;
                ++selected_count;
                if (selected_count == max_degree) {
                    return selected_count;
                }
            }
        }
        c += batch;
    }
    return selected_count;
}

}  // namespace

// =============================================================================
// VnniScratch Lifecycle
// =============================================================================
VnniScratch::~VnniScratch() {
    std::free(u8_buf);
    std::free(panel_buf);
    std::free(steps);
    std::free(min_vals);
    std::free(su);
    std::free(c_vals);
    std::free(sum_x);
    std::free(norms);
    std::free(global_ids);
    std::free(farthest_dist);
    std::free(topk_dists);
    std::free(topk_idx);
    std::free(topk_count);
    std::free(topk_farthest_pos);
    std::free(db_topk_d);
    std::free(db_topk_i);
    std::free(leaf_sketches_u16);
    std::free(leaf_sketches_f32);
}

void
VnniScratch::EnsureU8(size_t bytes) {
    if (bytes > u8_cap) {
        auto* next = static_cast<uint8_t*>(alloc_aligned64(bytes));
        std::free(u8_buf);
        u8_buf = next;
        u8_cap = bytes;
    }
}

void
VnniScratch::EnsurePanels(size_t bytes) {
    if (bytes > panel_cap) {
        auto* next = static_cast<int8_t*>(alloc_aligned64(bytes));
        std::free(panel_buf);
        panel_buf = next;
        panel_cap = bytes;
    }
}

void
VnniScratch::EnsureMeta(size_t padded_pts) {
    if (padded_pts > meta_cap) {
        using Guard = std::unique_ptr<void, decltype(&std::free)>;
        auto alloc_guard = [](size_t bytes) { return Guard(alloc_aligned64(bytes), &std::free); };
        Guard new_steps = alloc_guard(padded_pts * sizeof(float));
        Guard new_min_vals = alloc_guard(padded_pts * sizeof(float));
        Guard new_su = alloc_guard(padded_pts * sizeof(float));
        Guard new_c_vals = alloc_guard(padded_pts * sizeof(float));
        Guard new_sum_x = alloc_guard(padded_pts * sizeof(float));
        Guard new_norms = alloc_guard(padded_pts * sizeof(float));
        Guard new_global_ids = alloc_guard(padded_pts * sizeof(uint32_t));
        std::free(steps);
        std::free(min_vals);
        std::free(su);
        std::free(c_vals);
        std::free(sum_x);
        std::free(norms);
        std::free(global_ids);
        steps = static_cast<float*>(new_steps.release());
        min_vals = static_cast<float*>(new_min_vals.release());
        su = static_cast<float*>(new_su.release());
        c_vals = static_cast<float*>(new_c_vals.release());
        sum_x = static_cast<float*>(new_sum_x.release());
        norms = static_cast<float*>(new_norms.release());
        global_ids = static_cast<uint32_t*>(new_global_ids.release());
        meta_cap = padded_pts;
    }
}

void
VnniScratch::EnsureTopK(size_t padded_pts, size_t k) {
    if (padded_pts > topk_pts_cap or k > topk_k_cap) {
        const size_t new_pts = std::max(padded_pts, topk_pts_cap);
        const size_t new_k = std::max(k, topk_k_cap);
        using Guard = std::unique_ptr<void, decltype(&std::free)>;
        auto alloc_guard = [](size_t bytes) { return Guard(alloc_aligned64(bytes), &std::free); };
        Guard new_farthest_dist = alloc_guard(new_pts * sizeof(float));
        Guard new_topk_dists = alloc_guard(new_pts * new_k * sizeof(float));
        Guard new_topk_idx = alloc_guard(new_pts * new_k * sizeof(uint32_t));
        Guard new_topk_count = alloc_guard(new_pts * sizeof(uint16_t));
        Guard new_topk_farthest_pos = alloc_guard(new_pts * sizeof(uint16_t));
        std::free(farthest_dist);
        std::free(topk_dists);
        std::free(topk_idx);
        std::free(topk_count);
        std::free(topk_farthest_pos);
        farthest_dist = static_cast<float*>(new_farthest_dist.release());
        topk_dists = static_cast<float*>(new_topk_dists.release());
        topk_idx = static_cast<uint32_t*>(new_topk_idx.release());
        topk_count = static_cast<uint16_t*>(new_topk_count.release());
        topk_farthest_pos = static_cast<uint16_t*>(new_topk_farthest_pos.release());
        topk_pts_cap = new_pts;
        topk_k_cap = new_k;
    }
}

void
VnniScratch::EnsureDbTopK(size_t num_panels, size_t k) {
    const size_t elems = num_panels * k * 16;
    if (elems > db_topk_elems_cap) {
        using Guard = std::unique_ptr<void, decltype(&std::free)>;
        auto alloc_guard = [](size_t bytes) { return Guard(alloc_aligned64(bytes), &std::free); };
        Guard new_d = alloc_guard(elems * sizeof(float));
        Guard new_i = alloc_guard(elems * sizeof(uint32_t));
        std::free(db_topk_d);
        std::free(db_topk_i);
        db_topk_d = static_cast<float*>(new_d.release());
        db_topk_i = static_cast<uint32_t*>(new_i.release());
        db_topk_elems_cap = elems;
    }
}

void
VnniScratch::EnsureLeafSketchesU16(size_t elems) {
    if (elems > leaf_sketches_u16_cap) {
        auto* next = static_cast<uint16_t*>(alloc_aligned64(elems * sizeof(uint16_t)));
        std::free(leaf_sketches_u16);
        leaf_sketches_u16 = next;
        leaf_sketches_u16_cap = elems;
    }
}

void
VnniScratch::EnsureLeafSketchesF32(size_t elems) {
    if (elems > leaf_sketches_f32_cap) {
        auto* next = static_cast<float*>(alloc_aligned64(elems * sizeof(float)));
        std::free(leaf_sketches_f32);
        leaf_sketches_f32 = next;
        leaf_sketches_f32_cap = elems;
    }
}

// =============================================================================
// Runtime CPU Feature Detection & ISA Selection
// =============================================================================
bool
has_avx512_vnni() {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512bw") and
               __builtin_cpu_supports("avx512vl") and __builtin_cpu_supports("avx512dq") and
               __builtin_cpu_supports("avx512vnni");
    }();
    return supported;
#else
    return false;
#endif
}

bool
has_avx512_bw() {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") and __builtin_cpu_supports("avx512bw") and
               __builtin_cpu_supports("avx512vl") and __builtin_cpu_supports("avx512dq");
    }();
    return supported;
#else
    return false;
#endif
}

bool
has_avx_vnni() {
#if VSAG_PIPNN_COMPILE_AVX_VNNI
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") and __builtin_cpu_supports("avxvnni");
    }();
    return supported;
#else
    return false;
#endif
}

bool
has_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return supported;
#else
    return false;
#endif
}

bool
has_neon_dotprod() {
#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
    static const bool supported = []() {
        if (not cpuinfo_initialize()) {
            return false;
        }
        return cpuinfo_has_arm_neon_dot();
    }();
    return supported;
#else
    return false;
#endif
}

bool
has_neon() {
#ifdef __aarch64__
    return true;
#else
    return false;
#endif
}

Sq8Isa
active_sq8_isa() {
    static const Sq8Isa active = []() {
        if (has_avx512_vnni()) {
            return Sq8Isa::AVX512_VNNI;
        }
        if (has_avx512_bw()) {
            return Sq8Isa::AVX512_BW;
        }
        if (has_avx_vnni()) {
            return Sq8Isa::AVX_VNNI;
        }
        if (has_avx2()) {
            return Sq8Isa::AVX2;
        }
        if (has_neon_dotprod()) {
            return Sq8Isa::NEON_DOTPROD;
        }
        if (has_neon()) {
            return Sq8Isa::NEON;
        }
        return Sq8Isa::SCALAR;
    }();
    return active;
}

bool
is_sq8_isa_supported(Sq8Isa isa) {
    switch (isa) {
        case Sq8Isa::SCALAR:
            return true;
        case Sq8Isa::AVX2:
            return has_avx2();
        case Sq8Isa::AVX_VNNI:
            return has_avx_vnni();
        case Sq8Isa::AVX512_BW:
            return has_avx512_bw();
        case Sq8Isa::AVX512_VNNI:
            return has_avx512_vnni();
        case Sq8Isa::NEON:
            return has_neon();
        case Sq8Isa::NEON_DOTPROD:
            return has_neon_dotprod();
    }
    return false;
}

const char*
sq8_isa_name(Sq8Isa isa) {
    switch (isa) {
        case Sq8Isa::SCALAR:
            return "scalar";
        case Sq8Isa::AVX2:
            return "avx2";
        case Sq8Isa::AVX_VNNI:
            return "avx_vnni";
        case Sq8Isa::AVX512_BW:
            return "avx512bw";
        case Sq8Isa::AVX512_VNNI:
            return "avx512_vnni";
        case Sq8Isa::NEON:
            return "neon";
        case Sq8Isa::NEON_DOTPROD:
            return "neon_dotprod";
    }
    return "unknown";
}

// =============================================================================
// Public API Entrypoints
// =============================================================================
void
quantize_row_sq8_local(const float* src,
                       uint64_t dim,
                       uint64_t stride,
                       float pre_scale,
                       MetricKind metric,
                       uint8_t* dst_u8,
                       QMeta& out_meta) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        quantize_row_sq8_local_avx512(src, dim, stride, pre_scale, metric, dst_u8, out_meta);
        return;
    }
    if (has_avx2()) {
        quantize_row_sq8_local_avx2(src, dim, stride, pre_scale, metric, dst_u8, out_meta);
        return;
    }
#elif defined(__aarch64__)
    quantize_row_sq8_local_neon(src, dim, stride, pre_scale, metric, dst_u8, out_meta);
    return;
#endif
    quantize_row_sq8_local_scalar(src, dim, stride, pre_scale, metric, dst_u8, out_meta);
}

float
quantized_pair_distance_with_isa(const uint8_t* qa_u8,
                                 const QMeta& ma,
                                 const uint8_t* qb_u8,
                                 const QMeta& mb,
                                 uint64_t dim,
                                 uint64_t q_stride,
                                 MetricKind metric,
                                 Sq8Isa isa) {
    (void)q_stride;
    const float sum_x_b = static_cast<float>(dim) * mb.min_val + mb.su;
    const float c_b = mb.min_val + 128.0F * mb.step;
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Sq8Isa::AVX512_VNNI and has_avx512_vnni()) {
        return quantized_pair_distance_avx512_vnni(
            qa_u8, ma, qb_u8, mb, dim, align_pad(dim, 4), metric);
    }
    if (isa == Sq8Isa::AVX512_BW and has_avx512_bw()) {
        const int32_t acc = sq8_dot_1x1_raw_b_avx512_bw(qa_u8, qb_u8, align_pad(dim, 32));
        const float dot =
            sq8_dot_from_signed_acc(acc, ma.step, ma.min_val, ma.su, mb.step, c_b, sum_x_b);
        return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
    }
#if VSAG_PIPNN_COMPILE_AVX_VNNI
    if (isa == Sq8Isa::AVX_VNNI and has_avx_vnni()) {
        const int32_t acc = sq8_dot_1x1_raw_b_avx_vnni(qa_u8, qb_u8, align_pad(dim, 32));
        const float dot =
            sq8_dot_from_signed_acc(acc, ma.step, ma.min_val, ma.su, mb.step, c_b, sum_x_b);
        return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
    }
#endif
    if (isa == Sq8Isa::AVX2 and has_avx2()) {
        const int32_t acc = sq8_dot_1x1_raw_b_avx2(qa_u8, qb_u8, align_pad(dim, 16));
        const float dot =
            sq8_dot_from_signed_acc(acc, ma.step, ma.min_val, ma.su, mb.step, c_b, sum_x_b);
        return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
    }
#elif defined(__aarch64__)
#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
    if (isa == Sq8Isa::NEON_DOTPROD and has_neon_dotprod()) {
        const uint32_t acc = sq8_dot_1x1_neon_dotprod(qa_u8, qb_u8, align_pad(dim, 16));
        const float dot = sq8_dot_from_unsigned_acc(
            acc, ma.step, ma.min_val, ma.su, mb.step, mb.min_val, sum_x_b);
        return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
    }
#endif
    if (isa == Sq8Isa::NEON) {
        const uint32_t acc = sq8_dot_1x1_neon(qa_u8, qb_u8, align_pad(dim, 16));
        const float dot = sq8_dot_from_unsigned_acc(
            acc, ma.step, ma.min_val, ma.su, mb.step, mb.min_val, sum_x_b);
        return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
    }
#endif
    (void)c_b;
    const uint32_t acc = sq8_dot_1x1_scalar(qa_u8, qb_u8, dim);
    const float dot =
        sq8_dot_from_unsigned_acc(acc, ma.step, ma.min_val, ma.su, mb.step, mb.min_val, sum_x_b);
    return finalize_scalar_distance(dot, ma.norm, mb.norm, metric);
}

float
quantized_pair_distance(const uint8_t* qa_u8,
                        const QMeta& ma,
                        const uint8_t* qb_u8,
                        const QMeta& mb,
                        uint64_t dim,
                        uint64_t q_stride,
                        MetricKind metric) {
    return quantized_pair_distance_with_isa(
        qa_u8, ma, qb_u8, mb, dim, q_stride, metric, active_sq8_isa());
}

void
convert_sketch_row_to_u16(const float* src16, uint16_t* dst16) {
    for (uint64_t p = 0; p < 16; ++p) {
        dst16[p] = float_to_order_key(src16[p]);
    }
}

uint16_t
relative_hash16(const float* sketch_u,
                const float* sketch_v,
                uint64_t planes,
                uint16_t coincident_flag) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return relative_hash16_avx512(sketch_u, sketch_v, planes, coincident_flag);
    }
    if (has_avx2()) {
        return relative_hash16_avx2(sketch_u, sketch_v, planes, coincident_flag);
    }
#endif
    uint16_t hash = 0;
    bool coincident = true;
    for (uint64_t plane = 0; plane < planes; ++plane) {
        if (sketch_v[plane] >= sketch_u[plane]) {
            hash |= static_cast<uint16_t>(1U << plane);
        }
        coincident = coincident and (sketch_v[plane] == sketch_u[plane]);
    }
    if (coincident) {
        hash |= coincident_flag;
    }
    return hash;
}

uint16_t
relative_hash16_u16(const uint16_t* sketch_u,
                    const uint16_t* sketch_v,
                    uint64_t planes,
                    uint16_t coincident_flag) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return relative_hash16_u16_avx512(sketch_u, sketch_v, planes, coincident_flag);
    }
    if (has_avx2()) {
        return relative_hash16_u16_avx2(sketch_u, sketch_v, planes, coincident_flag);
    }
#endif
    uint16_t hash = 0;
    bool coincident = true;
    for (uint64_t plane = 0; plane < planes; ++plane) {
        if (sketch_v[plane] >= sketch_u[plane]) {
            hash |= static_cast<uint16_t>(1U << plane);
        }
        coincident = coincident and (sketch_v[plane] == sketch_u[plane]);
    }
    if (coincident) {
        hash |= coincident_flag;
    }
    return hash;
}

void
select_leaf_topk_from_dots_fp32(const uint32_t* leaf_points,
                                uint64_t point_count,
                                const float* dot_matrix,
                                const float* global_norms,
                                const uint32_t* global_ids_map,
                                bool ids_is_identity,
                                MetricKind metric,
                                uint32_t neighbor_count,
                                VnniScratch& scratch) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        if (metric == MetricKind::L2_SQR) {
            select_leaf_topk_from_dots_fp32_avx512_impl<MetricKind::L2_SQR>(leaf_points,
                                                                            point_count,
                                                                            dot_matrix,
                                                                            global_norms,
                                                                            global_ids_map,
                                                                            ids_is_identity,
                                                                            neighbor_count,
                                                                            scratch);
        } else if (metric == MetricKind::IP) {
            select_leaf_topk_from_dots_fp32_avx512_impl<MetricKind::IP>(leaf_points,
                                                                        point_count,
                                                                        dot_matrix,
                                                                        global_norms,
                                                                        global_ids_map,
                                                                        ids_is_identity,
                                                                        neighbor_count,
                                                                        scratch);
        } else {
            select_leaf_topk_from_dots_fp32_avx512_impl<MetricKind::COSINE>(leaf_points,
                                                                            point_count,
                                                                            dot_matrix,
                                                                            global_norms,
                                                                            global_ids_map,
                                                                            ids_is_identity,
                                                                            neighbor_count,
                                                                            scratch);
        }
        return;
    }
    if (has_avx2()) {
        if (metric == MetricKind::L2_SQR) {
            select_leaf_topk_from_dots_fp32_avx2_impl<MetricKind::L2_SQR>(leaf_points,
                                                                          point_count,
                                                                          dot_matrix,
                                                                          global_norms,
                                                                          global_ids_map,
                                                                          ids_is_identity,
                                                                          neighbor_count,
                                                                          scratch);
        } else if (metric == MetricKind::IP) {
            select_leaf_topk_from_dots_fp32_avx2_impl<MetricKind::IP>(leaf_points,
                                                                      point_count,
                                                                      dot_matrix,
                                                                      global_norms,
                                                                      global_ids_map,
                                                                      ids_is_identity,
                                                                      neighbor_count,
                                                                      scratch);
        } else {
            select_leaf_topk_from_dots_fp32_avx2_impl<MetricKind::COSINE>(leaf_points,
                                                                          point_count,
                                                                          dot_matrix,
                                                                          global_norms,
                                                                          global_ids_map,
                                                                          ids_is_identity,
                                                                          neighbor_count,
                                                                          scratch);
        }
        return;
    }
#endif
    select_leaf_topk_from_dots_fp32_scalar(leaf_points,
                                           point_count,
                                           dot_matrix,
                                           global_norms,
                                           global_ids_map,
                                           ids_is_identity,
                                           metric,
                                           neighbor_count,
                                           scratch);
}

uint32_t
group_and_hash_leaf_edges_fp32(const uint32_t* leaf_points,
                               uint64_t point_count,
                               uint32_t neighbor_count,
                               const uint32_t* topk_idx,
                               const float* topk_dists,
                               const uint16_t* topk_count,
                               const float* global_sketches_f32,
                               uint64_t hash_planes,
                               uint16_t coincident_flag,
                               VnniScratch& scratch,
                               uint32_t* offsets,
                               uint32_t* g_targets,
                               uint16_t* g_dist_keys,
                               uint16_t* g_hashes) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return group_and_hash_leaf_edges_fp32_avx512(leaf_points,
                                                     point_count,
                                                     neighbor_count,
                                                     topk_idx,
                                                     topk_dists,
                                                     topk_count,
                                                     global_sketches_f32,
                                                     hash_planes,
                                                     coincident_flag,
                                                     scratch,
                                                     offsets,
                                                     g_targets,
                                                     g_dist_keys,
                                                     g_hashes);
    }
    if (has_avx2()) {
        return group_and_hash_leaf_edges_fp32_avx2(leaf_points,
                                                   point_count,
                                                   neighbor_count,
                                                   topk_idx,
                                                   topk_dists,
                                                   topk_count,
                                                   global_sketches_f32,
                                                   hash_planes,
                                                   coincident_flag,
                                                   scratch,
                                                   offsets,
                                                   g_targets,
                                                   g_dist_keys,
                                                   g_hashes);
    }
#endif
    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const float* sk_src = global_sketches_f32 + static_cast<uint64_t>(src_global) * 16;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const float* sk_dst = global_sketches_f32 + static_cast<uint64_t>(dst_global) * 16;
            const uint16_t fwd_h = relative_hash16(sk_src, sk_dst, hash_planes, coincident_flag);
            const uint16_t rev_h = relative_hash16(sk_dst, sk_src, hash_planes, coincident_flag);
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

uint32_t
group_and_hash_leaf_edges_u16(const uint32_t* leaf_points,
                              uint64_t point_count,
                              uint32_t neighbor_count,
                              const uint32_t* topk_idx,
                              const float* topk_dists,
                              const uint16_t* topk_count,
                              const uint16_t* global_sketches_u16,
                              uint64_t hash_planes,
                              uint16_t coincident_flag,
                              VnniScratch& scratch,
                              uint32_t* offsets,
                              uint32_t* g_targets,
                              uint16_t* g_dist_keys,
                              uint16_t* g_hashes) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return group_and_hash_leaf_edges_u16_avx512(leaf_points,
                                                    point_count,
                                                    neighbor_count,
                                                    topk_idx,
                                                    topk_dists,
                                                    topk_count,
                                                    global_sketches_u16,
                                                    hash_planes,
                                                    coincident_flag,
                                                    scratch,
                                                    offsets,
                                                    g_targets,
                                                    g_dist_keys,
                                                    g_hashes);
    }
    if (has_avx2()) {
        return group_and_hash_leaf_edges_u16_avx2(leaf_points,
                                                  point_count,
                                                  neighbor_count,
                                                  topk_idx,
                                                  topk_dists,
                                                  topk_count,
                                                  global_sketches_u16,
                                                  hash_planes,
                                                  coincident_flag,
                                                  scratch,
                                                  offsets,
                                                  g_targets,
                                                  g_dist_keys,
                                                  g_hashes);
    }
#endif
    std::memset(offsets, 0, (point_count + 1) * sizeof(uint32_t));
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        offsets[src] += cnt;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            ++offsets[row_idx[r]];
        }
    }
    uint32_t total_edges = 0;
    for (uint64_t i = 0; i < point_count; ++i) {
        const uint32_t deg = offsets[i];
        offsets[i] = total_edges;
        total_edges += deg;
    }
    offsets[point_count] = total_edges;
    for (uint64_t src = 0; src < point_count; ++src) {
        const uint32_t cnt = topk_count[src];
        const uint32_t src_global = leaf_points[src];
        const uint16_t* sk_src = global_sketches_u16 + static_cast<uint64_t>(src_global) * 16;
        const uint32_t* row_idx = topk_idx + src * neighbor_count;
        const float* row_d = topk_dists + src * neighbor_count;
        for (uint32_t r = 0; r < cnt; ++r) {
            const uint32_t dst_local = row_idx[r];
            const uint32_t dst_global = leaf_points[dst_local];
            const uint16_t dkey = float_to_order_key(row_d[r]);
            const uint16_t* sk_dst = global_sketches_u16 + static_cast<uint64_t>(dst_global) * 16;
            const uint16_t fwd_h =
                relative_hash16_u16(sk_src, sk_dst, hash_planes, coincident_flag);
            const uint16_t rev_h =
                relative_hash16_u16(sk_dst, sk_src, hash_planes, coincident_flag);
            const uint32_t fwd_pos = offsets[src]++;
            g_targets[fwd_pos] = dst_global;
            g_dist_keys[fwd_pos] = dkey;
            g_hashes[fwd_pos] = fwd_h;
            const uint32_t rev_pos = offsets[dst_local]++;
            g_targets[rev_pos] = src_global;
            g_dist_keys[rev_pos] = dkey;
            g_hashes[rev_pos] = rev_h;
        }
    }
    return total_edges;
}

uint16_t
find_hash_u16(const uint16_t* hashes, uint16_t deg, uint16_t target) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return find_hash_u16_avx512(hashes, deg, target);
    }
    if (has_avx2()) {
        return find_hash_u16_avx2(hashes, deg, target);
    }
#endif
    for (uint16_t i = 0; i < deg; ++i) {
        if (hashes[i] == target) {
            return i;
        }
    }
    return deg;
}

uint16_t
find_id_u32(const uint32_t* ids, uint16_t deg, uint32_t target) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return find_id_u32_avx512(ids, deg, target);
    }
    if (has_avx2()) {
        return find_id_u32_avx2(ids, deg, target);
    }
#endif
    for (uint16_t i = 0; i < deg; ++i) {
        if (ids[i] == target) {
            return i;
        }
    }
    return deg;
}

uint16_t
find_farthest_slot_u16(const uint16_t* dists,
                       const uint16_t* hashes,
                       const uint32_t* neighbors,
                       const uint32_t* ids,
                       bool ids_is_identity,
                       uint16_t deg,
                       uint16_t& out_farthest_dist) {
#if defined(__x86_64__) || defined(_M_X64)
    if (has_avx512_bw()) {
        return find_farthest_slot_u16_avx512(
            dists, hashes, neighbors, ids, ids_is_identity, deg, out_farthest_dist);
    }
    if (has_avx2()) {
        return find_farthest_slot_u16_avx2(
            dists, hashes, neighbors, ids, ids_is_identity, deg, out_farthest_dist);
    }
#endif
    uint16_t best_idx = 0;
    uint16_t best_dist = dists[0];
    for (uint16_t i = 1; i < deg; ++i) {
        const uint16_t cd = dists[i];
        if (cd > best_dist) {
            best_idx = i;
            best_dist = cd;
        } else if (cd == best_dist) {
            const uint32_t cid = ids_is_identity ? neighbors[i] : ids[neighbors[i]];
            const uint32_t fid = ids_is_identity ? neighbors[best_idx] : ids[neighbors[best_idx]];
            if (cid > fid or (cid == fid and hashes[i] > hashes[best_idx])) {
                best_idx = i;
            }
        }
    }
    out_farthest_dist = best_dist;
    return best_idx;
}

uint32_t
quantized_robust_prune_panels(const uint8_t* q_data_u8,
                              const QMeta* q_meta,
                              uint64_t dim,
                              uint64_t dim_pad4,
                              uint64_t q_stride,
                              MetricKind metric,
                              const uint32_t* ordered_cand_ids,
                              const float* ordered_cand_dists,
                              uint32_t num_cands,
                              uint32_t max_degree,
                              float alpha,
                              // NOLINTNEXTLINE(readability-non-const-parameter)
                              uint32_t* out_selected) {
    const Sq8Isa isa = active_sq8_isa();
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Sq8Isa::AVX512_VNNI) {
        if (metric == MetricKind::L2_SQR) {
            return quantized_robust_prune_panels_avx512_vnni_impl<MetricKind::L2_SQR>(
                q_data_u8,
                q_meta,
                dim,
                dim_pad4,
                q_stride,
                ordered_cand_ids,
                ordered_cand_dists,
                num_cands,
                max_degree,
                alpha,
                out_selected);
        }
        if (metric == MetricKind::IP) {
            return quantized_robust_prune_panels_avx512_vnni_impl<MetricKind::IP>(
                q_data_u8,
                q_meta,
                dim,
                dim_pad4,
                q_stride,
                ordered_cand_ids,
                ordered_cand_dists,
                num_cands,
                max_degree,
                alpha,
                out_selected);
        }
        return quantized_robust_prune_panels_avx512_vnni_impl<MetricKind::COSINE>(
            q_data_u8,
            q_meta,
            dim,
            dim_pad4,
            q_stride,
            ordered_cand_ids,
            ordered_cand_dists,
            num_cands,
            max_degree,
            alpha,
            out_selected);
    }
    if (isa == Sq8Isa::AVX512_BW) {
        return quantized_robust_prune_generic<Avx512BwBackend>(q_data_u8,
                                                               q_meta,
                                                               dim,
                                                               q_stride,
                                                               metric,
                                                               ordered_cand_ids,
                                                               ordered_cand_dists,
                                                               num_cands,
                                                               max_degree,
                                                               alpha,
                                                               out_selected);
    }
#if VSAG_PIPNN_COMPILE_AVX_VNNI
    if (isa == Sq8Isa::AVX_VNNI) {
        return quantized_robust_prune_generic<AvxVnniBackend>(q_data_u8,
                                                              q_meta,
                                                              dim,
                                                              q_stride,
                                                              metric,
                                                              ordered_cand_ids,
                                                              ordered_cand_dists,
                                                              num_cands,
                                                              max_degree,
                                                              alpha,
                                                              out_selected);
    }
#endif
    if (isa == Sq8Isa::AVX2) {
        return quantized_robust_prune_generic<Avx2Backend>(q_data_u8,
                                                           q_meta,
                                                           dim,
                                                           q_stride,
                                                           metric,
                                                           ordered_cand_ids,
                                                           ordered_cand_dists,
                                                           num_cands,
                                                           max_degree,
                                                           alpha,
                                                           out_selected);
    }
#elif defined(__aarch64__)
#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
    if (isa == Sq8Isa::NEON_DOTPROD) {
        return quantized_robust_prune_generic<NeonDotProdBackend>(q_data_u8,
                                                                  q_meta,
                                                                  dim,
                                                                  q_stride,
                                                                  metric,
                                                                  ordered_cand_ids,
                                                                  ordered_cand_dists,
                                                                  num_cands,
                                                                  max_degree,
                                                                  alpha,
                                                                  out_selected);
    }
#endif
    if (isa == Sq8Isa::NEON) {
        return quantized_robust_prune_generic<NeonBackend>(q_data_u8,
                                                           q_meta,
                                                           dim,
                                                           q_stride,
                                                           metric,
                                                           ordered_cand_ids,
                                                           ordered_cand_dists,
                                                           num_cands,
                                                           max_degree,
                                                           alpha,
                                                           out_selected);
    }
#endif
    (void)dim_pad4;
    (void)isa;
    return quantized_robust_prune_generic<ScalarBackend>(q_data_u8,
                                                         q_meta,
                                                         dim,
                                                         q_stride,
                                                         metric,
                                                         ordered_cand_ids,
                                                         ordered_cand_dists,
                                                         num_cands,
                                                         max_degree,
                                                         alpha,
                                                         out_selected);
}

void
fused_leaf_topk_sq8_local(const uint32_t* leaf_points,
                          uint64_t point_count,
                          const uint8_t* q_data_u8,
                          const QMeta* q_meta,
                          const uint32_t* global_ids_map,
                          bool ids_is_identity,
                          uint64_t dim,
                          uint64_t dim_pad4,
                          uint64_t q_stride,
                          MetricKind metric,
                          uint32_t neighbor_count,
                          VnniScratch& scratch) {
    const Sq8Isa isa = active_sq8_isa();
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Sq8Isa::AVX512_VNNI) {
        fused_leaf_topk_avx512_vnni(leaf_points,
                                    point_count,
                                    q_data_u8,
                                    q_meta,
                                    global_ids_map,
                                    ids_is_identity,
                                    dim,
                                    dim_pad4,
                                    q_stride,
                                    metric,
                                    neighbor_count,
                                    scratch);
        return;
    }
    if (isa == Sq8Isa::AVX512_BW) {
        fused_leaf_topk_generic<Avx512BwBackend>(leaf_points,
                                                 point_count,
                                                 q_data_u8,
                                                 q_meta,
                                                 global_ids_map,
                                                 ids_is_identity,
                                                 dim,
                                                 q_stride,
                                                 metric,
                                                 neighbor_count,
                                                 scratch);
        return;
    }
#if VSAG_PIPNN_COMPILE_AVX_VNNI
    if (isa == Sq8Isa::AVX_VNNI) {
        fused_leaf_topk_generic<AvxVnniBackend>(leaf_points,
                                                point_count,
                                                q_data_u8,
                                                q_meta,
                                                global_ids_map,
                                                ids_is_identity,
                                                dim,
                                                q_stride,
                                                metric,
                                                neighbor_count,
                                                scratch);
        return;
    }
#endif
    if (isa == Sq8Isa::AVX2) {
        fused_leaf_topk_generic<Avx2Backend>(leaf_points,
                                             point_count,
                                             q_data_u8,
                                             q_meta,
                                             global_ids_map,
                                             ids_is_identity,
                                             dim,
                                             q_stride,
                                             metric,
                                             neighbor_count,
                                             scratch);
        return;
    }
#elif defined(__aarch64__)
#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
    if (isa == Sq8Isa::NEON_DOTPROD) {
        fused_leaf_topk_generic<NeonDotProdBackend>(leaf_points,
                                                    point_count,
                                                    q_data_u8,
                                                    q_meta,
                                                    global_ids_map,
                                                    ids_is_identity,
                                                    dim,
                                                    q_stride,
                                                    metric,
                                                    neighbor_count,
                                                    scratch);
        return;
    }
#endif
    if (isa == Sq8Isa::NEON) {
        fused_leaf_topk_generic<NeonBackend>(leaf_points,
                                             point_count,
                                             q_data_u8,
                                             q_meta,
                                             global_ids_map,
                                             ids_is_identity,
                                             dim,
                                             q_stride,
                                             metric,
                                             neighbor_count,
                                             scratch);
        return;
    }
#endif
    (void)dim_pad4;
    (void)isa;
    fused_leaf_topk_generic<ScalarBackend>(leaf_points,
                                           point_count,
                                           q_data_u8,
                                           q_meta,
                                           global_ids_map,
                                           ids_is_identity,
                                           dim,
                                           q_stride,
                                           metric,
                                           neighbor_count,
                                           scratch);
}

void
pack_leader_panels_sq8_local(const uint32_t* leaders,
                             uint64_t leader_count,
                             const uint8_t* q_data_u8,
                             const QMeta* q_meta,
                             const uint32_t* global_ids_map,
                             bool ids_is_identity,
                             uint64_t dim,
                             uint64_t dim_pad4,
                             uint64_t q_stride,
                             VnniScratch& leader_scratch) {
    const auto dim_f = static_cast<float>(dim);
    const Sq8Isa isa = active_sq8_isa();
    const uint64_t num_panels = (leader_count + 15) / 16;
    const uint64_t padded_leaders = num_panels * 16;

    leader_scratch.EnsureMeta(padded_leaders);
    for (uint64_t i = 0; i < leader_count; ++i) {
        const uint32_t pid = leaders[i];
        const QMeta& m = q_meta[pid];
        leader_scratch.steps[i] = m.step;
        leader_scratch.min_vals[i] = m.min_val;
        leader_scratch.su[i] = m.su;
        leader_scratch.c_vals[i] = m.min_val + 128.0F * m.step;
        leader_scratch.sum_x[i] = dim_f * m.min_val + m.su;
        leader_scratch.norms[i] = m.norm;
        leader_scratch.global_ids[i] = ids_is_identity ? pid : global_ids_map[pid];
    }
    for (uint64_t i = leader_count; i < padded_leaders; ++i) {
        leader_scratch.steps[i] = 0.0F;
        leader_scratch.min_vals[i] = 0.0F;
        leader_scratch.su[i] = 0.0F;
        leader_scratch.c_vals[i] = 0.0F;
        leader_scratch.sum_x[i] = 0.0F;
        leader_scratch.norms[i] = SENTINEL;
        leader_scratch.global_ids[i] = 0;
    }

    if (isa == Sq8Isa::AVX512_VNNI) {
        const uint64_t total_tiles = dim_pad4 / 4;
        leader_scratch.EnsurePanels(num_panels * total_tiles * 64);
        for (uint64_t p = 0; p < num_panels; ++p) {
            const uint64_t base = p * 16;
            const uint64_t valid = std::min<uint64_t>(16, leader_count - base);
            auto* panel_u32 =
                reinterpret_cast<uint32_t*>(leader_scratch.panel_buf + p * total_tiles * 64);
            const uint32_t* ptrs[16];
            for (uint64_t lane = 0; lane < valid; ++lane) {
                ptrs[lane] = reinterpret_cast<const uint32_t*>(
                    q_data_u8 + static_cast<size_t>(leaders[base + lane]) * q_stride);
            }
            for (uint64_t t = 0; t < total_tiles; ++t) {
                uint32_t* dst_tile = panel_u32 + t * 16;
                for (uint64_t lane = 0; lane < valid; ++lane) {
                    dst_tile[lane] = ptrs[lane][t] ^ FLIP_U32;
                }
                for (uint64_t lane = valid; lane < 16; ++lane) {
                    dst_tile[lane] = 0;
                }
            }
        }
        return;
    }

    const uint64_t row_stride = align_pad(dim, 32);
    if (isa == Sq8Isa::AVX512_BW or isa == Sq8Isa::AVX_VNNI or isa == Sq8Isa::AVX2) {
        leader_scratch.EnsurePanels(leader_count * row_stride);
        for (uint64_t i = 0; i < leader_count; ++i) {
            const uint8_t* src = q_data_u8 + static_cast<size_t>(leaders[i]) * q_stride;
            copy_and_flip_u8_to_i8(src, leader_scratch.panel_buf + i * row_stride, row_stride);
        }
    } else {
        leader_scratch.EnsureU8(leader_count * row_stride);
        for (uint64_t i = 0; i < leader_count; ++i) {
            const uint8_t* src = q_data_u8 + static_cast<size_t>(leaders[i]) * q_stride;
            std::memcpy(leader_scratch.u8_buf + i * row_stride, src, row_stride);
        }
    }
}

void
assign_stripe_to_leaders_sq8_local(const uint32_t* points,
                                   uint64_t stripe_size,
                                   const VnniScratch& leader_scratch,
                                   uint64_t leader_count,
                                   const uint8_t* q_data_u8,
                                   const QMeta* q_meta,
                                   uint64_t dim,
                                   uint64_t dim_pad4,
                                   uint64_t q_stride,
                                   MetricKind metric,
                                   uint32_t fanout,
                                   // NOLINTNEXTLINE(readability-non-const-parameter)
                                   uint32_t* out_assignments,
                                   VnniScratch& worker_scratch) {
    const Sq8Isa isa = active_sq8_isa();
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Sq8Isa::AVX512_VNNI) {
        if (metric == MetricKind::L2_SQR) {
            dispatch_assign_by_fanout<MetricKind::L2_SQR>(points,
                                                          stripe_size,
                                                          leader_scratch,
                                                          leader_count,
                                                          q_data_u8,
                                                          q_meta,
                                                          dim_pad4,
                                                          q_stride,
                                                          fanout,
                                                          out_assignments,
                                                          worker_scratch);
        } else if (metric == MetricKind::IP) {
            dispatch_assign_by_fanout<MetricKind::IP>(points,
                                                      stripe_size,
                                                      leader_scratch,
                                                      leader_count,
                                                      q_data_u8,
                                                      q_meta,
                                                      dim_pad4,
                                                      q_stride,
                                                      fanout,
                                                      out_assignments,
                                                      worker_scratch);
        } else {
            dispatch_assign_by_fanout<MetricKind::COSINE>(points,
                                                          stripe_size,
                                                          leader_scratch,
                                                          leader_count,
                                                          q_data_u8,
                                                          q_meta,
                                                          dim_pad4,
                                                          q_stride,
                                                          fanout,
                                                          out_assignments,
                                                          worker_scratch);
        }
        return;
    }
    if (isa == Sq8Isa::AVX512_BW) {
        assign_stripe_to_leaders_generic<Avx512BwBackend>(points,
                                                          stripe_size,
                                                          leader_scratch,
                                                          leader_count,
                                                          q_data_u8,
                                                          q_meta,
                                                          dim,
                                                          q_stride,
                                                          metric,
                                                          fanout,
                                                          out_assignments,
                                                          worker_scratch);
        return;
    }
#if VSAG_PIPNN_COMPILE_AVX_VNNI
    if (isa == Sq8Isa::AVX_VNNI) {
        assign_stripe_to_leaders_generic<AvxVnniBackend>(points,
                                                         stripe_size,
                                                         leader_scratch,
                                                         leader_count,
                                                         q_data_u8,
                                                         q_meta,
                                                         dim,
                                                         q_stride,
                                                         metric,
                                                         fanout,
                                                         out_assignments,
                                                         worker_scratch);
        return;
    }
#endif
    if (isa == Sq8Isa::AVX2) {
        assign_stripe_to_leaders_generic<Avx2Backend>(points,
                                                      stripe_size,
                                                      leader_scratch,
                                                      leader_count,
                                                      q_data_u8,
                                                      q_meta,
                                                      dim,
                                                      q_stride,
                                                      metric,
                                                      fanout,
                                                      out_assignments,
                                                      worker_scratch);
        return;
    }
#elif defined(__aarch64__)
#if VSAG_PIPNN_COMPILE_NEON_DOTPROD
    if (isa == Sq8Isa::NEON_DOTPROD) {
        assign_stripe_to_leaders_generic<NeonDotProdBackend>(points,
                                                             stripe_size,
                                                             leader_scratch,
                                                             leader_count,
                                                             q_data_u8,
                                                             q_meta,
                                                             dim,
                                                             q_stride,
                                                             metric,
                                                             fanout,
                                                             out_assignments,
                                                             worker_scratch);
        return;
    }
#endif
    if (isa == Sq8Isa::NEON) {
        assign_stripe_to_leaders_generic<NeonBackend>(points,
                                                      stripe_size,
                                                      leader_scratch,
                                                      leader_count,
                                                      q_data_u8,
                                                      q_meta,
                                                      dim,
                                                      q_stride,
                                                      metric,
                                                      fanout,
                                                      out_assignments,
                                                      worker_scratch);
        return;
    }
#endif
    (void)dim_pad4;
    (void)isa;
    assign_stripe_to_leaders_generic<ScalarBackend>(points,
                                                    stripe_size,
                                                    leader_scratch,
                                                    leader_count,
                                                    q_data_u8,
                                                    q_meta,
                                                    dim,
                                                    q_stride,
                                                    metric,
                                                    fanout,
                                                    out_assignments,
                                                    worker_scratch);
}

}  // namespace vsag::pipnn_simd
