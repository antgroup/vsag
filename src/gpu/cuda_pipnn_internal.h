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
};

/// Stage 1: uploads the rows, centres them and quantizes them to the INT8 scoring copy
/// `d_q` (count x dim_pad) with the squared norms `d_norms`. The device is idle when it
/// returns and only the two outputs stay accounted in `ws`.
bool
QuantizeRows(const float* vectors,
             uint64_t count,
             const BuildPlan& plan,
             Workspace& ws,
             DeviceArray<int8_t>& d_q,
             DeviceArray<int32_t>& d_norms);

}  // namespace vsag::gpu::pipnn
