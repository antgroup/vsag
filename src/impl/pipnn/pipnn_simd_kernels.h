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

#include <cstddef>
#include <cstdint>

namespace vsag::pipnn_simd {

/// Metric identifier mirrored from MetricType so SIMD kernels stay decoupled.
enum class MetricKind : uint8_t {
    L2_SQR = 0,
    IP = 1,
    COSINE = 2,
};

/// Active hardware SIMD backend for `sq8_local` quantized distance kernels.
enum class Sq8Isa : uint8_t {
    SCALAR = 0,
    AVX2 = 1,
    AVX_VNNI = 2,
    AVX512_BW = 3,
    AVX512_VNNI = 4,
    NEON = 5,
    NEON_DOTPROD = 6,
};

/// Compact 16-byte per-vector metadata for asymmetric per-vector `sq8_local` quantization:
///   x_hat[d] = min_val + step * u[d],   u[d] in [0, 255],   u[d >= dim] = 0.
/// Inner product identity (exact across unsigned u*u and signed-B u*(u^0x80) ISAs):
///   <x_hat_a, x_hat_b> = (step_a * step_b) * <u_a, u_b> + min_b * su_a + min_a * sum_x_b
///                      = (step_a * step_b) * <u_a, u_b - 128> + c_b * su_a + min_a * sum_x_b
/// where su_a = step_a * sum(u_a), c_b = min_b + 128 * step_b, sum_x_b = dim * min_b + su_b.
struct alignas(16) QMeta {
    float step{0.0F};
    float min_val{0.0F};
    float su{0.0F};
    float norm{0.0F};  // L2Sqr: ||x_hat||^2; IP/Cosine: 0.0F
};
static_assert(sizeof(QMeta) == 16, "QMeta must be 16 bytes");

/// Runtime CPU feature queries.
bool
has_avx512_vnni();
bool
has_avx512_bw();
bool
has_avx_vnni();
bool
has_avx2();
bool
has_neon_dotprod();
bool
has_neon();

/// Returns the best `sq8_local` ISA backend supported on the host CPU.
Sq8Isa
active_sq8_isa();

/// Returns whether a specific `Sq8Isa` backend is compiled in and supported on the host CPU.
bool
is_sq8_isa_supported(Sq8Isa isa);

/// Returns a human-readable backend name for logging and diagnostics.
const char*
sq8_isa_name(Sq8Isa isa);

/// Quantizes one FP32 vector (length `dim`) into asymmetric per-vector `sq8_local` codes
/// `u[d] in [0, 255]` (`stride` bytes, zero-padded on `[dim, stride)`).
/// For Cosine metric, multiplies by `pre_scale = inv_norm` before quantization.
void
quantize_row_sq8_local(const float* src,
                       uint64_t dim,
                       uint64_t stride,
                       float pre_scale,
                       MetricKind metric,
                       uint8_t* dst_u8,
                       QMeta& out_meta);

/// Computes exact `sq8_local` pairwise distance between two quantized vectors using `active_sq8_isa()`.
float
quantized_pair_distance(const uint8_t* qa_u8,
                        const QMeta& ma,
                        const uint8_t* qb_u8,
                        const QMeta& mb,
                        uint64_t dim,
                        uint64_t q_stride,
                        MetricKind metric);

/// Computes `sq8_local` pairwise distance forcing a specific supported `Sq8Isa` (for testing).
float
quantized_pair_distance_with_isa(const uint8_t* qa_u8,
                                 const QMeta& ma,
                                 const uint8_t* qb_u8,
                                 const QMeta& mb,
                                 uint64_t dim,
                                 uint64_t q_stride,
                                 MetricKind metric,
                                 Sq8Isa isa);

/// Converts a 16-float SimHash sketch row into 16 order-preserving uint16_t BF16 keys
/// (32 bytes per vector, halving sketch memory footprint so 1M sketches fit in L3).
void
convert_sketch_row_to_u16(const float* src16, uint16_t* dst16);

/// Computes 12..15-bit relative SimHash between two 16-float-padded sketch rows
/// using SIMD when available. Sets bit 15 (`coincident_flag`) iff all `planes`
/// values are equal.
uint16_t
relative_hash16(const float* sketch_u,
                const float* sketch_v,
                uint64_t planes,
                uint16_t coincident_flag);

/// Computes 12..15-bit relative SimHash between two 16-uint16 order-key sketch rows.
uint16_t
relative_hash16_u16(const uint16_t* sketch_u,
                    const uint16_t* sketch_v,
                    uint64_t planes,
                    uint16_t coincident_flag);

/// Vectorized search for `target` in a 64B-aligned `uint16_t` hash array of length `deg`.
/// Returns index in `[0, deg)` if found, else `deg`.
uint16_t
find_hash_u16(const uint16_t* hashes, uint16_t deg, uint16_t target);

/// Vectorized search for `target` in a `uint32_t` neighbor ID array of length `deg`.
/// Returns index in `[0, deg)` if found, else `deg`.
uint16_t
find_id_u32(const uint32_t* ids, uint16_t deg, uint32_t target);

/// Vectorized search for the farthest slot in a 64B-aligned `uint16_t` distance-key array
/// of length `deg` (> 0), breaking ties deterministically by (neighbor global ID, hash).
uint16_t
find_farthest_slot_u16(const uint16_t* dists,
                       const uint16_t* hashes,
                       const uint32_t* neighbors,
                       const uint32_t* ids,
                       bool ids_is_identity,
                       uint16_t deg,
                       uint16_t& out_farthest_dist);

/// Workspace reused per worker thread across leaves and partition stripes (zero heap alloc).
struct VnniScratch {
    // Packed uint8 row-major vectors for leaf queries or unsigned-dot leaders
    uint8_t* u8_buf{nullptr};
    size_t u8_cap{0};
    // Transposed 16-point x 4-byte int8 panels (AVX-512 VNNI) or row-major XOR-flipped int8 rows
    int8_t* panel_buf{nullptr};
    size_t panel_cap{0};
    // Per-point padded scalar metadata for SIMD loads (aligned to 64B)
    float* steps{nullptr};
    float* min_vals{nullptr};
    float* su{nullptr};
    float* c_vals{nullptr};  // min_val + 128.0f * step
    float* sum_x{nullptr};   // dim * min_val + su
    float* norms{nullptr};
    uint32_t* global_ids{nullptr};
    size_t meta_cap{0};
    // Leaf Top-K SoA buffers: farthest distance per point + topk arrays
    float* farthest_dist{nullptr};
    float* topk_dists{nullptr};
    uint32_t* topk_idx{nullptr};
    uint16_t* topk_count{nullptr};
    uint16_t* topk_farthest_pos{nullptr};
    size_t topk_pts_cap{0};
    size_t topk_k_cap{0};
    // Panel-major SIMD DB-side Top-K buffers (num_panels * k * 16 elements, 64B aligned)
    float* db_topk_d{nullptr};
    uint32_t* db_topk_i{nullptr};
    size_t db_topk_elems_cap{0};
    // Leaf-local gathered uint16_t sketches (point_count * 16 uint16_t, 64B aligned)
    uint16_t* leaf_sketches_u16{nullptr};
    size_t leaf_sketches_u16_cap{0};
    // Leaf-local gathered FP32 sketches (point_count * 16 float, 64B aligned)
    float* leaf_sketches_f32{nullptr};
    size_t leaf_sketches_f32_cap{0};

    VnniScratch() = default;
    ~VnniScratch();
    VnniScratch(const VnniScratch&) = delete;
    VnniScratch&
    operator=(const VnniScratch&) = delete;

    void
    EnsureU8(size_t bytes);
    void
    EnsurePanels(size_t bytes);
    void
    EnsureMeta(size_t padded_pts);
    void
    EnsureTopK(size_t padded_pts, size_t k);
    void
    EnsureDbTopK(size_t num_panels, size_t k);
    void
    EnsureLeafSketchesU16(size_t elems);
    void
    EnsureLeafSketchesF32(size_t elems);
};

/// Selects exact Top-K leaf neighbors per vertex from a symmetric FP32 dot-product matrix
/// (produced by Ssyrk + lower->upper mirror), writing results into `scratch.topk_*`.
void
select_leaf_topk_from_dots_fp32(const uint32_t* leaf_points,
                                uint64_t point_count,
                                const float* dot_matrix,
                                const float* global_norms,
                                const uint32_t* global_ids_map,
                                bool ids_is_identity,
                                MetricKind metric,
                                uint32_t neighbor_count,
                                VnniScratch& scratch);

/// Gathers leaf-local 64-byte FP32 sketches into L1d, performs counting-sort edge grouping,
/// and computes exact FP32 relative SimHashes + BF16 distance order keys outside locks.
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
                               uint16_t* g_hashes);

/// Gathers leaf-local 32-byte BF16 sketches into L1d, performs counting-sort edge grouping,
/// and computes distance_order_key + both directed SimHashes inline with SIMD.
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
                              uint16_t* g_hashes);

/// Runs full `sq8_local` RobustPrune for one vertex using the active SIMD backend
/// (AVX-512 VNNI / AVX-512BW / AVX-VNNI / AVX2 / NEON DotProd / NEON / Scalar).
/// Writes up to `max_degree` selected local candidate indices into `out_selected`.
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
                              uint32_t* out_selected);

/// Fused lower-triangle symmetric leaf Top-K kernel for `sq8_local` across all SIMD backends.
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
                          VnniScratch& scratch);

/// Packs `leader_count` leaders into `leader_scratch` in the optimal layout for `active_sq8_isa()`.
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
                             VnniScratch& leader_scratch);

/// Fused leader assignment for a stripe of `stripe_size` points against pre-packed leaders
/// (`leader_scratch`), dispatching to the active SIMD backend.
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
                                   uint32_t* out_assignments,
                                   VnniScratch& worker_scratch);

}  // namespace vsag::pipnn_simd
