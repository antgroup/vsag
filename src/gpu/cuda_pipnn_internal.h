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

// Internal to the PiPNN build: the stage translation units share these and nothing
// outside src/gpu includes them.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"

namespace vsag::gpu::pipnn {

constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kTileRows = 64;
constexpr uint32_t kMaxFanout = 16;
constexpr uint32_t kMaxNeighbors = 16;
constexpr uint32_t kMaxHashBits = 12;
constexpr uint32_t kMaxDegree = 128;
constexpr uint32_t kTargetLeafSize = 512;

#define VSAG_CUDA_TRY(expr)          \
    do {                             \
        if ((expr) != cudaSuccess) { \
            return false;            \
        }                            \
    } while (false)

inline uint32_t
bits_for(uint64_t max_value) {
    uint32_t bits = 1;
    while (bits < 64 and (max_value >> bits) != 0) {
        ++bits;
    }
    return bits;
}

inline uint32_t
grid_for(uint64_t items) {
    uint64_t blocks = (items + kBlockThreads - 1) / kBlockThreads;
    return static_cast<uint32_t>(std::min<uint64_t>(std::max<uint64_t>(blocks, 1), 65535ULL * 16));
}

/// Tracks the device bytes this build holds so the working set stays under the budget

class Workspace {
public:
    explicit Workspace(uint64_t limit) : limit_(limit) {
    }

    template <typename T>
    bool
    alloc(DeviceArray<T>& array, uint64_t count) {
        const uint64_t bytes = std::max<uint64_t>(count, 1) * sizeof(T);
        if (bytes > limit_ - std::min(limit_, used_)) {
            return false;
        }
        if (not array.Alloc(std::max<uint64_t>(count, 1))) {
            return false;
        }
        used_ += bytes;
        return true;
    }

    void
    credit(uint64_t bytes) {
        used_ -= std::min(used_, bytes);
    }

    bool
    ensure_temp(uint64_t bytes) {
        if (bytes <= temp_bytes_) {
            return true;
        }
        credit(temp_bytes_);
        temp_bytes_ = 0;
        temp_ = DeviceArray<uint8_t>();
        if (not alloc(temp_, bytes)) {
            return false;
        }
        temp_bytes_ = bytes;
        return true;
    }

    [[nodiscard]] void*
    temp() const {
        return temp_.Get();
    }

    [[nodiscard]] uint64_t
    available() const {
        return limit_ - std::min(limit_, used_);
    }

    [[nodiscard]] uint64_t
    used() const {
        return used_;
    }

    /// Drops the scratch buffer and sets the tally to what the caller still holds.
    void
    release_to(uint64_t used) {
        temp_ = DeviceArray<uint8_t>();
        temp_bytes_ = 0;
        used_ = used;
    }

private:
    uint64_t limit_;
    uint64_t used_{0};
    DeviceArray<uint8_t> temp_;
    uint64_t temp_bytes_{0};
};

template <typename T>
bool
upload(Workspace& ws, DeviceArray<T>& device, const std::vector<T>& host) {
    if (not ws.alloc(device, host.size())) {
        return false;
    }
    if (host.empty()) {
        return true;
    }
    return cudaMemcpy(device.Get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

// Streams the caller's rows through the device. When the whole matrix fits it is

struct BuildPlan {
    uint32_t dim;
    uint32_t dim_pad;
    uint32_t dim_words;
    uint32_t level1_leaders;
    uint32_t fanout1;
    uint32_t fanout2;
    uint32_t neighbors;
    uint32_t hash_bits;
    uint32_t max_degree;
    uint32_t reservoir;
    float alpha;
    uint32_t min_leaf;
    uint32_t max_leaf;
    uint32_t target_leaf;
    uint64_t seed;
    DistanceType metric;
};

/// Stage 1: uploads the rows, centres them and quantizes them to the INT8 scoring copy
/// `d_q` (count x dim_pad) with the squared norms `d_norms`. Cosine rows are normalized
/// first. Inner product is not centred and every norm is the largest one, so that the
/// shared score `norm_a + norm_b - 2 * dot` is `2 * (max_norm - dot)`: non-negative and
/// ordered like -dot. The device is idle when it
/// returns and only the two outputs stay accounted in `ws`.
bool
QuantizeRows(const float* vectors,
             uint64_t count,
             const BuildPlan& plan,
             Workspace& ws,
             DeviceArray<int8_t>& d_q,
             DeviceArray<int32_t>& d_norms);

template <typename K, typename V>
bool
sort_pairs(Workspace& ws,
           cub::DoubleBuffer<K>& keys,
           cub::DoubleBuffer<V>& vals,
           uint64_t n,
           uint32_t end_bit) {
    if (n == 0) {
        return true;
    }
    size_t bytes = 0;
    VSAG_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        nullptr, bytes, keys, vals, static_cast<int>(n), 0, static_cast<int>(end_bit)));
    if (not ws.ensure_temp(bytes)) {
        return false;
    }
    VSAG_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        ws.temp(), bytes, keys, vals, static_cast<int>(n), 0, static_cast<int>(end_bit)));
    return true;
}

struct Tiles {
    std::vector<uint32_t> seg;
    std::vector<uint32_t> row0;
};

inline Tiles
build_tiles(const std::vector<uint32_t>& member_off,
            const std::vector<uint32_t>& cand_off,
            bool need_candidates) {
    Tiles tiles;
    const uint64_t segs = member_off.size() - 1;
    for (uint64_t s = 0; s < segs; ++s) {
        uint32_t size = member_off[s + 1] - member_off[s];
        if (need_candidates and cand_off[s + 1] == cand_off[s]) {
            continue;
        }
        for (uint32_t r = 0; r < size; r += kTileRows) {
            tiles.seg.push_back(static_cast<uint32_t>(s));
            tiles.row0.push_back(r);
        }
    }
    return tiles;
}

struct ScoreArgs {
    const int8_t* q;
    const int32_t* norms;
    uint32_t dim_words;
    uint32_t dim_pad;
    const uint32_t* q_ids;
    const uint32_t* c_ids;
    const uint32_t* member_off;
    const uint32_t* cand_off;
    const uint32_t* tile_seg;
    const uint32_t* tile_row0;
    uint32_t out_k;
    uint32_t* out_assign;
    uint64_t* out_keys;
    uint32_t* out_vals;
    uint32_t pos_base;
    uint32_t slab_begin;
    uint32_t slab_end;
    uint32_t count;
    uint32_t hash_bits;
};

/// Runs the fused score-tile kernel over `tiles` thread blocks. kPick = false assigns
/// every member of a segment its nearest candidate leaders; kPick = true emits the
/// leaf-neighbour edges with their hash buckets.
template <bool kPick>
bool
launch_score(const ScoreArgs& args, uint32_t tiles);

/// Marks the edges of `keys` whose source vertex is a real one (< count).
bool
FlagEdges(const uint64_t* keys, uint64_t n, uint32_t count, uint8_t* flags);

/// Stage 2: partitions the rows into overlapping leaves. `d_members` receives the point
/// of every leaf row, leaf after leaf, and `leaf_offsets` the CSR offsets of the leaves.
bool
PartitionLeaves(Workspace& ws,
                const BuildPlan& plan,
                const ScoreArgs& base,
                uint64_t count,
                DeviceArray<uint32_t>& d_members,
                std::vector<uint32_t>& leaf_offsets);

/// One slab of the prune stage: `edges` sorted candidate edges whose sources lie in
/// [slab_begin, slab_end) are folded into each vertex's final neighbours.
struct PruneJob {
    const uint64_t* keys;
    const uint32_t* vals;
    uint64_t edges;
    const int8_t* q;
    const int32_t* norms;
    uint32_t dim_words;
    uint32_t vertex_count;
    uint32_t slab_begin;
    uint32_t slab_end;
    uint32_t max_degree;
    uint32_t reservoir;
    float alpha;
    uint32_t* begin;  // slab_end - slab_begin entries of scratch
    uint32_t* end;
    uint32_t* out_neighbors;
    uint32_t* out_degrees;
};

/// Stage 4: finds each vertex's edge range, then keeps the closest edge of every hash
/// bucket and thins them with the alpha occlusion rule.
bool
PruneSlab(const PruneJob& job);

}  // namespace vsag::gpu::pipnn
