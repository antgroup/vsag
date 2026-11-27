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

// Fully device-resident PiPNN graph build.
//
// The dataset is uploaded once, centred and quantized to INT8, and every later
// stage works on that copy without returning to the host:
//
//   1. quantize  (cuda_pipnn_quantize.cu) - global centring and one INT8 scale, plus
//                  per-row squared norms; cosine rows are normalized first and inner
//                  product is scored as 2 * (max_norm - dot) on uncentred rows.
//   2. partition (cuda_pipnn_partition.cu) - two flattened levels of overlapping
//                  leader buckets; the second level handles every oversized bucket in
//                  one launch.
//   3. pick      (cuda_pipnn_score.cu) - a fused INT8 score-tile kernel selects each
//                  leaf member's nearest leaf neighbours in registers and emits the
//                  edge together with its folded sign-hash bucket; no leaf distance
//                  matrix is ever stored.
//   4. prune     (cuda_pipnn_prune.cu) - the edges are radix-sorted on (source,
//                  bucket, distance) and one warp per vertex keeps the closest edge of
//                  every bucket, the closest `reservoir_size` of those, and finally
//                  the ones an alpha-relaxed occlusion rule leaves, up to `max_degree`.
//
// This file only sequences the stages and owns the memory budget: it decides whether the
// candidate edges fit in one pass or are sorted and pruned one slab of source vertices at
// a time. Shared types live in cuda_pipnn_internal.h.
//
// Only the final adjacency is copied back. Anything unexpected returns false with the
// caller's outputs untouched, which sends the caller to its CPU build.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cuda_backend.h"
#include "cuda_pipnn_internal.h"

namespace vsag::gpu {

namespace pipnn {

namespace {

constexpr uint64_t kEdgeCountLimit = 1ULL << 30;
constexpr uint64_t kMinMargin = 64ULL << 20;

bool
build_impl(const float* vectors,
           uint64_t count,
           const BuildPlan& plan,
           uint32_t* out_neighbors,
           uint32_t* out_degrees,
           uint64_t budget_bytes) {
    Workspace ws(budget_bytes);
    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t degree = plan.max_degree;
    const uint32_t reservoir = plan.reservoir;
    const float alpha = plan.alpha;

    DeviceArray<int8_t> d_q;
    DeviceArray<int32_t> d_norms;
    DeviceArray<uint32_t> d_out_nb;
    DeviceArray<uint32_t> d_out_deg;
    if (not ws.alloc(d_q, count * plan.dim_pad) or not ws.alloc(d_norms, count) or
        not ws.alloc(d_out_nb, count * degree) or not ws.alloc(d_out_deg, count)) {
        return false;
    }
    VSAG_CUDA_TRY(cudaMemset(d_out_deg.Get(), 0, count * sizeof(uint32_t)));

    // Stage 1: quantization.
    if (not QuantizeRows(vectors, count, plan, ws, d_q, d_norms)) {
        return false;
    }

    ScoreArgs base{};
    base.q = d_q.Get();
    base.norms = d_norms.Get();
    base.dim_words = plan.dim_words;
    base.dim_pad = plan.dim_pad;
    base.count = n;
    base.hash_bits = plan.hash_bits;

    // Stage 2: flattened two-level partition.
    DeviceArray<uint32_t> d_members;  // point of every leaf row, leaf after leaf
    std::vector<uint32_t> leaf_offsets;
    if (not PartitionLeaves(ws, plan, base, count, d_members, leaf_offsets)) {
        return false;
    }

    // Stage 3 + 4: pick leaf neighbours and prune them into the final adjacency.
    const uint64_t rows = leaf_offsets.back();
    const uint32_t k = plan.neighbors;
    const uint64_t total_edges = rows * 2ULL * k;

    DeviceArray<uint32_t> d_leaf_off;
    DeviceArray<uint32_t> d_tile_leaf;
    DeviceArray<uint32_t> d_tile_row0;
    Tiles tiles = build_tiles(leaf_offsets, leaf_offsets, false);
    if (not upload(ws, d_leaf_off, leaf_offsets) or not upload(ws, d_tile_leaf, tiles.seg) or
        not upload(ws, d_tile_row0, tiles.row0)) {
        return false;
    }
    const uint32_t tile_count = static_cast<uint32_t>(tiles.seg.size());
    auto tile_position = [&](uint32_t t) { return leaf_offsets[tiles.seg[t]] + tiles.row0[t]; };
    auto tile_end_position = [&](uint32_t t) {
        const uint32_t leaf = tiles.seg[t];
        return leaf_offsets[leaf] +
               std::min(tiles.row0[t] + kTileRows, leaf_offsets[leaf + 1] - leaf_offsets[leaf]);
    };

    ScoreArgs pick = base;
    pick.q_ids = d_members.Get();
    pick.c_ids = d_members.Get();
    pick.member_off = d_leaf_off.Get();
    pick.cand_off = d_leaf_off.Get();
    pick.out_k = k;

    const uint32_t key_bits = 32 + bits_for(n);
    const uint64_t margin = kMinMargin + total_edges / 64;
    const uint64_t reserved = margin + count * 8;
    const uint64_t avail = ws.available() > reserved ? ws.available() - reserved : 0;
    const bool single_pass = total_edges <= kEdgeCountLimit and total_edges * 24 <= avail;

    uint64_t slab_cap = 0;
    uint64_t chunk_cap = 0;
    uint32_t slab_vertices = n;
    if (not single_pass) {
        slab_cap = std::min<uint64_t>(avail / 43, kEdgeCountLimit / 2);
        chunk_cap = slab_cap / 2;
        if (chunk_cap < 2ULL * k * kTileRows) {
            return false;
        }
        const double fraction =
            0.8 * static_cast<double>(slab_cap) / static_cast<double>(total_edges);
        slab_vertices = static_cast<uint32_t>(
            std::max<double>(1.0, std::min<double>(n, std::floor(fraction * n))));
    }

    const uint64_t buffer_edges = single_pass ? total_edges : slab_cap + chunk_cap;
    DeviceArray<uint64_t> d_keys_a;
    DeviceArray<uint64_t> d_keys_b;
    DeviceArray<uint32_t> d_vals_a;
    DeviceArray<uint32_t> d_vals_b;
    DeviceArray<uint32_t> d_begin;
    DeviceArray<uint32_t> d_end;
    if (not ws.alloc(d_keys_a, buffer_edges) or not ws.alloc(d_keys_b, buffer_edges) or
        not ws.alloc(d_vals_a, buffer_edges) or not ws.alloc(d_vals_b, buffer_edges) or
        not ws.alloc(d_begin, slab_vertices) or not ws.alloc(d_end, slab_vertices)) {
        return false;
    }
    cub::DoubleBuffer<uint64_t> edge_keys(d_keys_a.Get(), d_keys_b.Get());
    cub::DoubleBuffer<uint32_t> edge_vals(d_vals_a.Get(), d_vals_b.Get());

    DeviceArray<uint64_t> d_layout_keys;
    DeviceArray<uint32_t> d_layout_vals;
    DeviceArray<uint8_t> d_flags;
    DeviceArray<int> d_selected;
    if (not single_pass) {
        if (not ws.alloc(d_layout_keys, chunk_cap) or not ws.alloc(d_layout_vals, chunk_cap) or
            not ws.alloc(d_flags, chunk_cap) or not ws.alloc(d_selected, 2)) {
            return false;
        }
    }

    // Runs one slab of source vertices. 0 = done, 1 = slab overflowed, -1 = failure.
    auto run_slab = [&](uint32_t u0, uint32_t u1) -> int {
        uint64_t edges = 0;
        pick.slab_begin = u0;
        pick.slab_end = u1;
        if (single_pass) {
            pick.pos_base = 0;
            pick.tile_seg = d_tile_leaf.Get();
            pick.tile_row0 = d_tile_row0.Get();
            pick.out_keys = edge_keys.Current();
            pick.out_vals = edge_vals.Current();
            if (not launch_score<true>(pick, tile_count)) {
                return -1;
            }
            edges = total_edges;
        } else {
            uint32_t t0 = 0;
            while (t0 < tile_count) {
                uint32_t t1 = t0 + 1;
                const uint32_t base_pos = tile_position(t0);
                while (t1 < tile_count and
                       (static_cast<uint64_t>(tile_end_position(t1)) - base_pos) * 2ULL * k <=
                           chunk_cap) {
                    ++t1;
                }
                const uint64_t chunk_edges =
                    (static_cast<uint64_t>(tile_end_position(t1 - 1)) - base_pos) * 2ULL * k;
                pick.pos_base = base_pos;
                pick.tile_seg = d_tile_leaf.Get() + t0;
                pick.tile_row0 = d_tile_row0.Get() + t0;
                pick.out_keys = d_layout_keys.Get();
                pick.out_vals = d_layout_vals.Get();
                if (not launch_score<true>(pick, t1 - t0)) {
                    return -1;
                }
                if (not FlagEdges(d_layout_keys.Get(), chunk_edges, n, d_flags.Get())) {
                    return -1;
                }
                size_t bytes = 0;
                cub::DeviceSelect::Flagged(nullptr,
                                           bytes,
                                           d_layout_keys.Get(),
                                           d_flags.Get(),
                                           edge_keys.Current() + edges,
                                           d_selected.Get(),
                                           static_cast<int>(chunk_edges));
                if (not ws.ensure_temp(bytes)) {
                    return -1;
                }
                if (cub::DeviceSelect::Flagged(ws.temp(),
                                               bytes,
                                               d_layout_keys.Get(),
                                               d_flags.Get(),
                                               edge_keys.Current() + edges,
                                               d_selected.Get(),
                                               static_cast<int>(chunk_edges)) != cudaSuccess or
                    cub::DeviceSelect::Flagged(ws.temp(),
                                               bytes,
                                               d_layout_vals.Get(),
                                               d_flags.Get(),
                                               edge_vals.Current() + edges,
                                               d_selected.Get() + 1,
                                               static_cast<int>(chunk_edges)) != cudaSuccess) {
                    return -1;
                }
                int selected = 0;
                if (cudaMemcpy(&selected, d_selected.Get(), sizeof(int), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
                    return -1;
                }
                edges += static_cast<uint64_t>(selected);
                if (edges > slab_cap) {
                    return 1;
                }
                t0 = t1;
            }
        }
        if (not sort_pairs(ws, edge_keys, edge_vals, edges, key_bits)) {
            return -1;
        }
        PruneJob job{};
        job.keys = edge_keys.Current();
        job.vals = edge_vals.Current();
        job.edges = edges;
        job.q = base.q;
        job.norms = base.norms;
        job.dim_words = base.dim_words;
        job.vertex_count = n;
        job.slab_begin = u0;
        job.slab_end = u1;
        job.max_degree = degree;
        job.reservoir = reservoir;
        job.alpha = alpha;
        job.begin = d_begin.Get();
        job.end = d_end.Get();
        job.out_neighbors = d_out_nb.Get();
        job.out_degrees = d_out_deg.Get();
        return PruneSlab(job) ? 0 : -1;
    };

    for (uint32_t u0 = 0; u0 < n;) {
        const uint32_t u1 = std::min<uint64_t>(n, static_cast<uint64_t>(u0) + slab_vertices);
        const int status = run_slab(u0, u1);
        if (status < 0) {
            return false;
        }
        if (status == 1) {
            if (slab_vertices == 1) {
                return false;
            }
            slab_vertices = std::max<uint32_t>(1, slab_vertices / 2);
            continue;
        }
        u0 = u1;
    }

    VSAG_CUDA_TRY(cudaDeviceSynchronize());
    VSAG_CUDA_TRY(cudaMemcpy(
        out_neighbors, d_out_nb.Get(), count * degree * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    VSAG_CUDA_TRY(
        cudaMemcpy(out_degrees, d_out_deg.Get(), count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return true;
}

}  // namespace

}  // namespace pipnn

bool
CudaBuildPiPNNGraph(const float* vectors,
                    uint64_t count,
                    int32_t dim,
                    const CudaPiPNNConfig& config,
                    uint32_t* out_neighbors,
                    uint32_t* out_degrees,
                    uint64_t budget_bytes) {
    if (vectors == nullptr or out_neighbors == nullptr or out_degrees == nullptr or dim <= 0 or
        count < 2 or count >= (1ULL << 31) or budget_bytes == 0) {
        return false;
    }
    if (config.distance_type != DistanceType::L2Sqr and
        config.distance_type != DistanceType::InnerProduct and
        config.distance_type != DistanceType::Cosine) {
        return false;
    }
    if (config.leaf_neighbor_count == 0 or config.leaf_neighbor_count > pipnn::kMaxNeighbors or
        config.fanout_l1 == 0 or config.fanout_l1 > pipnn::kMaxFanout or config.fanout_l2 == 0 or
        config.fanout_l2 > pipnn::kMaxFanout or config.hash_plane_count == 0 or
        config.hash_plane_count > pipnn::kMaxHashBits or config.max_degree == 0 or
        config.max_degree > pipnn::kMaxDegree or config.reservoir_size > pipnn::kMaxDegree or
        not(config.alpha >= 1.0F) or config.min_leaf_size == 0 or
        config.max_leaf_size < config.min_leaf_size or config.max_leaf_size <= config.fanout_l2 or
        config.max_leaf_size > (1ULL << 20) or config.leader_sample_rate < 0.0F) {
        return false;
    }
    if (not CudaAvailable()) {
        return false;
    }

    pipnn::BuildPlan plan{};
    plan.dim = static_cast<uint32_t>(dim);
    plan.dim_pad = ((plan.dim + 15U) / 16U) * 16U;
    plan.dim_words = plan.dim_pad / 4U;
    const double want = std::floor(static_cast<double>(count) * config.leader_sample_rate);
    plan.level1_leaders = static_cast<uint32_t>(
        std::min<double>(static_cast<double>(count), std::min(512.0, std::max(64.0, want))));
    plan.fanout1 = std::min(config.fanout_l1, plan.level1_leaders);
    plan.fanout2 = config.fanout_l2;
    plan.neighbors = config.leaf_neighbor_count;
    plan.hash_bits = config.hash_plane_count;
    plan.max_degree = config.max_degree;
    plan.reservoir = std::max(config.reservoir_size, config.max_degree);
    plan.alpha = config.alpha;
    plan.min_leaf = static_cast<uint32_t>(config.min_leaf_size);
    plan.max_leaf = static_cast<uint32_t>(config.max_leaf_size);
    plan.target_leaf = std::min(std::max(pipnn::kTargetLeafSize, plan.min_leaf), plan.max_leaf);
    plan.seed = config.seed;
    plan.metric = config.distance_type;
    if (count * plan.fanout1 * plan.fanout2 >= (1ULL << 31)) {
        return false;
    }

    const bool ok =
        pipnn::build_impl(vectors, count, plan, out_neighbors, out_degrees, budget_bytes);
    ClearPendingError();
    return ok;
}

}  // namespace vsag::gpu
