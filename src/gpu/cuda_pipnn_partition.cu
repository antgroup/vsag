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

// Stage 2 of the device-resident PiPNN build: two flattened levels of overlapping
// leader buckets. The second level handles every oversized bucket in one launch, and
// the resulting groups are regularized into leaves of a bounded size.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include "cuda_pipnn_internal.h"

namespace vsag::gpu::pipnn {

namespace {

constexpr uint32_t kMaxSubLeaders = 128;

// ---------------------------------------------------------------------------
// Partition helpers
// ---------------------------------------------------------------------------

__global__ void
count_kernel(const uint32_t* keys, uint64_t n, uint32_t limit, uint32_t* counts) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        uint32_t key = keys[i];
        if (key < limit) {
            atomicAdd(counts + key, 1U);
        }
    }
}

__global__ void
level1_pairs_kernel(
    const uint32_t* assign, uint64_t n, uint32_t fanout, uint32_t* keys, uint32_t* vals) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        keys[i] = assign[i];
        vals[i] = static_cast<uint32_t>(i / fanout);
    }
}

__global__ void
gather_kernel(const uint32_t* src, const uint32_t* index, uint64_t n, uint32_t* dst) {
    uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[index[i]];
    }
}

// Every position of the level-1 order owns `fanout` slots. A member of an oversized
// bucket fills them with its sub-bucket ids; a member of a small bucket fills the
// first with the bucket itself and parks the rest past the last group, where the
// sort leaves them behind the real pairs.
__global__ void
level2_pairs_kernel(const uint32_t* bucket_of_pos,
                    const uint32_t* point_of_pos,
                    const uint32_t* assign2,
                    const uint32_t* cand_off,
                    uint64_t n,
                    uint32_t fanout,
                    uint32_t small_base,
                    uint32_t group_count,
                    uint32_t* keys,
                    uint32_t* vals) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        uint32_t bucket = bucket_of_pos[i];
        uint32_t point = point_of_pos[i];
        bool big = cand_off[bucket + 1] > cand_off[bucket];
        for (uint32_t s = 0; s < fanout; ++s) {
            uint64_t slot = i * fanout + s;
            if (big) {
                keys[slot] = assign2[slot];
            } else {
                keys[slot] = s == 0 ? small_base + bucket : group_count;
            }
            vals[slot] = point;
        }
    }
}

struct Leaves {
    std::vector<uint32_t> offsets;
};

// Turns the per-group sizes, in group order, into leaves: runs above the cap are cut
// into near-equal pieces of about the target size, and runs below the floor are
// merged with the runs that follow until they reach it.
Leaves
regularize_leaves(const std::vector<uint32_t>& group_sizes,
                  uint32_t min_leaf,
                  uint32_t max_leaf,
                  uint32_t target_leaf) {
    Leaves leaves;
    leaves.offsets.push_back(0);
    uint32_t position = 0;
    uint32_t pending_begin = 0;
    uint32_t pending_size = 0;

    auto emit = [&](uint32_t begin, uint32_t size) {
        if (size == 0) {
            return;
        }
        if (size <= max_leaf) {
            leaves.offsets.push_back(begin + size);
            return;
        }
        const uint32_t pieces = (size + target_leaf - 1) / target_leaf;
        const uint32_t piece = (size + pieces - 1) / pieces;
        for (uint32_t done = 0; done < size; done += piece) {
            leaves.offsets.push_back(begin + std::min(done + piece, size));
        }
    };
    auto flush = [&]() {
        emit(pending_begin, pending_size);
        pending_size = 0;
    };

    for (uint32_t size : group_sizes) {
        if (size == 0) {
            continue;
        }
        const uint32_t begin = position;
        position += size;
        if (size >= min_leaf) {
            flush();
            emit(begin, size);
            continue;
        }
        if (pending_size > 0 and pending_size + size > max_leaf) {
            flush();
        }
        if (pending_size == 0) {
            pending_begin = begin;
        }
        pending_size += size;
        if (pending_size >= min_leaf) {
            flush();
        }
    }
    flush();
    return leaves;
}

std::vector<uint32_t>
sample_leaders(uint64_t count, uint32_t picks, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::unordered_set<uint32_t> chosen;
    chosen.reserve(picks * 2);
    for (uint64_t j = count - picks; j < count; ++j) {
        uint32_t t = static_cast<uint32_t>(rng() % (j + 1));
        if (not chosen.insert(t).second) {
            chosen.insert(static_cast<uint32_t>(j));
        }
    }
    std::vector<uint32_t> out(chosen.begin(), chosen.end());
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

bool
PartitionLeaves(Workspace& ws,
                const BuildPlan& plan,
                const ScoreArgs& base,
                uint64_t count,
                DeviceArray<uint32_t>& d_members,
                std::vector<uint32_t>& leaf_offsets) {
    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t m1 = plan.level1_leaders;
    const uint32_t f1 = plan.fanout1;
    const uint32_t f2 = plan.fanout2;
    const uint64_t pos_count = count * f1;
    const uint64_t slot_count = pos_count * f2;

    const uint64_t partition_mark = ws.used();
    {
        DeviceArray<uint32_t> d_leader_ids;
        DeviceArray<uint32_t> d_assign1;
        DeviceArray<uint32_t> d_key_a;
        DeviceArray<uint32_t> d_key_b;
        DeviceArray<uint32_t> d_val_a;
        DeviceArray<uint32_t> d_val_b;
        DeviceArray<uint32_t> d_counts1;
        DeviceArray<uint32_t> d_member_off1;
        DeviceArray<uint32_t> d_cand_off1;
        DeviceArray<uint32_t> d_tile_seg;
        DeviceArray<uint32_t> d_tile_row0;
        const std::vector<uint32_t> leaders = sample_leaders(count, m1, plan.seed);
        if (not upload(ws, d_leader_ids, leaders) or not ws.alloc(d_assign1, pos_count) or
            not ws.alloc(d_key_a, pos_count) or not ws.alloc(d_key_b, pos_count) or
            not ws.alloc(d_val_a, pos_count) or not ws.alloc(d_val_b, pos_count) or
            not ws.alloc(d_counts1, m1)) {
            return false;
        }
        {
            const std::vector<uint32_t> member_off = {0, n};
            const std::vector<uint32_t> cand_off = {0, m1};
            Tiles tiles = build_tiles(member_off, cand_off, true);
            if (not upload(ws, d_member_off1, member_off) or
                not upload(ws, d_cand_off1, cand_off) or not upload(ws, d_tile_seg, tiles.seg) or
                not upload(ws, d_tile_row0, tiles.row0)) {
                return false;
            }
            ScoreArgs args = base;
            args.q_ids = nullptr;
            args.c_ids = d_leader_ids.Get();
            args.member_off = d_member_off1.Get();
            args.cand_off = d_cand_off1.Get();
            args.tile_seg = d_tile_seg.Get();
            args.tile_row0 = d_tile_row0.Get();
            // out_k never exceeds the candidate count of a segment (f1 <= m1), so the assignment
            // holds no kNoValue; the radix sort on the assignment keys relies on that.
            args.out_k = f1;
            args.out_assign = d_assign1.Get();
            if (not launch_score<false>(args, static_cast<uint32_t>(tiles.seg.size()))) {
                return false;
            }
        }
        VSAG_CUDA_TRY(cudaMemset(d_counts1.Get(), 0, m1 * sizeof(uint32_t)));
        count_kernel<<<grid_for(pos_count), kBlockThreads>>>(
            d_assign1.Get(), pos_count, m1, d_counts1.Get());
        level1_pairs_kernel<<<grid_for(pos_count), kBlockThreads>>>(
            d_assign1.Get(), pos_count, f1, d_key_a.Get(), d_val_a.Get());
        VSAG_CUDA_TRY(cudaGetLastError());
        cub::DoubleBuffer<uint32_t> keys1(d_key_a.Get(), d_key_b.Get());
        cub::DoubleBuffer<uint32_t> vals1(d_val_a.Get(), d_val_b.Get());
        if (not sort_pairs(ws, keys1, vals1, pos_count, bits_for(m1 - 1))) {
            return false;
        }
        std::vector<uint32_t> counts1(m1);
        VSAG_CUDA_TRY(cudaMemcpy(
            counts1.data(), d_counts1.Get(), m1 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Sub-leaders for every bucket that is too big to be a leaf.
        std::vector<uint32_t> bucket_off(m1 + 1, 0);
        std::vector<uint32_t> sub_off(m1 + 1, 0);
        std::vector<uint32_t> sub_positions;
        std::mt19937_64 rng(plan.seed ^ 0x5DEECE66DULL);
        for (uint32_t b = 0; b < m1; ++b) {
            bucket_off[b + 1] = bucket_off[b] + counts1[b];
            uint32_t subs = 0;
            const uint32_t size = counts1[b];
            if (size > plan.max_leaf) {
                const uint64_t want =
                    (static_cast<uint64_t>(size) * f2 + plan.target_leaf - 1) / plan.target_leaf;
                subs = static_cast<uint32_t>(
                    std::min<uint64_t>(std::max<uint64_t>(want, f2 + 1), kMaxSubLeaders));
                subs = std::min(subs, size);
                for (uint32_t j = 0; j < subs; ++j) {
                    const uint64_t lo = static_cast<uint64_t>(j) * size / subs;
                    const uint64_t hi = static_cast<uint64_t>(j + 1) * size / subs;
                    sub_positions.push_back(bucket_off[b] +
                                            static_cast<uint32_t>(lo + rng() % (hi - lo)));
                }
            }
            sub_off[b + 1] = sub_off[b] + subs;
        }
        const uint32_t total_sub = sub_off[m1];
        const uint32_t group_count = total_sub + m1;

        DeviceArray<uint32_t> d_assign2;
        DeviceArray<uint32_t> d_pair_key_a;
        DeviceArray<uint32_t> d_pair_key_b;
        DeviceArray<uint32_t> d_pair_val_a;
        DeviceArray<uint32_t> d_pair_val_b;
        DeviceArray<uint32_t> d_group_counts;
        if (total_sub > 0) {
            DeviceArray<uint32_t> d_sub_ids;
            DeviceArray<uint32_t> d_sub_positions;
            DeviceArray<uint32_t> d_member_off2;
            DeviceArray<uint32_t> d_cand_off2;
            DeviceArray<uint32_t> d_tile_seg2;
            DeviceArray<uint32_t> d_tile_row02;
            Tiles tiles = build_tiles(bucket_off, sub_off, true);
            if (not upload(ws, d_sub_positions, sub_positions) or
                not ws.alloc(d_sub_ids, total_sub) or not ws.alloc(d_assign2, slot_count) or
                not upload(ws, d_member_off2, bucket_off) or not upload(ws, d_cand_off2, sub_off) or
                not upload(ws, d_tile_seg2, tiles.seg) or
                not upload(ws, d_tile_row02, tiles.row0)) {
                return false;
            }
            gather_kernel<<<grid_for(total_sub), kBlockThreads>>>(
                vals1.Current(), d_sub_positions.Get(), total_sub, d_sub_ids.Get());
            VSAG_CUDA_TRY(cudaGetLastError());
            ScoreArgs args = base;
            args.q_ids = vals1.Current();
            args.c_ids = d_sub_ids.Get();
            args.member_off = d_member_off2.Get();
            args.cand_off = d_cand_off2.Get();
            args.tile_seg = d_tile_seg2.Get();
            args.tile_row0 = d_tile_row02.Get();
            // Likewise f2 < the sub-leader count of every big bucket (max_leaf_size > fanout_l2).
            args.out_k = f2;
            args.out_assign = d_assign2.Get();
            if (not launch_score<false>(args, static_cast<uint32_t>(tiles.seg.size()))) {
                return false;
            }
            VSAG_CUDA_TRY(cudaDeviceSynchronize());
        }

        DeviceArray<uint32_t> d_cand_off_full;
        if (not upload(ws, d_cand_off_full, sub_off) or not ws.alloc(d_pair_key_a, slot_count) or
            not ws.alloc(d_pair_key_b, slot_count) or not ws.alloc(d_pair_val_a, slot_count) or
            not ws.alloc(d_pair_val_b, slot_count) or not ws.alloc(d_group_counts, group_count)) {
            return false;
        }
        level2_pairs_kernel<<<grid_for(pos_count), kBlockThreads>>>(keys1.Current(),
                                                                    vals1.Current(),
                                                                    d_assign2.Get(),
                                                                    d_cand_off_full.Get(),
                                                                    pos_count,
                                                                    f2,
                                                                    total_sub,
                                                                    group_count,
                                                                    d_pair_key_a.Get(),
                                                                    d_pair_val_a.Get());
        VSAG_CUDA_TRY(cudaGetLastError());
        VSAG_CUDA_TRY(cudaMemset(d_group_counts.Get(), 0, group_count * sizeof(uint32_t)));
        count_kernel<<<grid_for(slot_count), kBlockThreads>>>(
            d_pair_key_a.Get(), slot_count, group_count, d_group_counts.Get());
        VSAG_CUDA_TRY(cudaGetLastError());
        cub::DoubleBuffer<uint32_t> keys2(d_pair_key_a.Get(), d_pair_key_b.Get());
        cub::DoubleBuffer<uint32_t> vals2(d_pair_val_a.Get(), d_pair_val_b.Get());
        if (not sort_pairs(ws, keys2, vals2, slot_count, bits_for(group_count))) {
            return false;
        }
        std::vector<uint32_t> group_sizes(group_count);
        VSAG_CUDA_TRY(cudaMemcpy(group_sizes.data(),
                                 d_group_counts.Get(),
                                 group_count * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost));
        leaf_offsets =
            regularize_leaves(group_sizes, plan.min_leaf, plan.max_leaf, plan.target_leaf).offsets;
        const uint64_t rows = leaf_offsets.back();
        if (not ws.alloc(d_members, rows)) {
            return false;
        }
        VSAG_CUDA_TRY(cudaMemcpy(
            d_members.Get(), vals2.Current(), rows * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        VSAG_CUDA_TRY(cudaDeviceSynchronize());
    }
    ws.release_to(partition_mark + leaf_offsets.back() * sizeof(uint32_t));
    return true;
}

}  // namespace vsag::gpu::pipnn
