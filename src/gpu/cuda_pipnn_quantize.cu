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

// Stage 1 of the device-resident PiPNN build: global centring and one INT8 scale, a
// per-coordinate random sign flip and per-row squared norms.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "cuda_pipnn_internal.h"

namespace vsag::gpu::pipnn {

namespace {

constexpr uint32_t kSumSlices = 512;

// ---------------------------------------------------------------------------
// Quantization
// ---------------------------------------------------------------------------

__device__ __forceinline__ int
coordinate_sign(uint32_t d, uint32_t seed) {
    uint32_t x = d + seed * 0x9E3779B9U;
    x ^= x >> 16;
    x *= 0x85EBCA6BU;
    x ^= x >> 13;
    x *= 0xC2B2AE35U;
    x ^= x >> 16;
    return (x & 1U) != 0 ? -1 : 1;
}

__global__ void
column_sum_kernel(const float* x, uint64_t rows, uint32_t dim, double* partial) {
    for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
        double sum = 0.0;
        for (uint64_t r = blockIdx.x; r < rows; r += gridDim.x) {
            sum += static_cast<double>(x[r * dim + d]);
        }
        partial[static_cast<uint64_t>(blockIdx.x) * dim + d] += sum;
    }
}

__global__ void
mean_kernel(const double* partial, uint32_t slices, uint32_t dim, double inv_count, float* mean) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) {
        return;
    }
    double sum = 0.0;
    for (uint32_t s = 0; s < slices; ++s) {
        sum += partial[static_cast<uint64_t>(s) * dim + d];
    }
    mean[d] = static_cast<float>(sum * inv_count);
}

__global__ void
abs_max_kernel(const float* x, uint64_t rows, uint32_t dim, const float* mean, uint32_t* max_bits) {
    float local = 0.0F;
    uint64_t total = rows * dim;
    for (uint64_t e = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < total;
         e += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        float v = fabsf(x[e] - mean[e % dim]);
        local = fmaxf(local, v);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = fmaxf(local, __shfl_down_sync(0xFFFFFFFFU, local, offset));
    }
    if ((threadIdx.x & 31U) == 0) {
        atomicMax(max_bits, __float_as_uint(local));
    }
}

// The per-coordinate sign flip is an isometry, so distances are unchanged; it only
// makes the fold in the pick kernel a sparse random projection instead of a
// projection onto raw, correlated coordinates.
__global__ void
quantize_kernel(const float* x,
                uint64_t rows,
                uint32_t dim,
                uint32_t dim_pad,
                const float* mean,
                float scale,
                uint32_t seed,
                int8_t* q) {
    uint64_t total = rows * dim_pad;
    for (uint64_t e = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; e < total;
         e += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        uint64_t r = e / dim_pad;
        uint32_t d = static_cast<uint32_t>(e - r * dim_pad);
        int value = 0;
        if (d < dim) {
            float v = rintf((x[r * dim + d] - mean[d]) * scale);
            value = static_cast<int>(fminf(127.0F, fmaxf(-127.0F, v))) * coordinate_sign(d, seed);
        }
        q[e] = static_cast<int8_t>(value);
    }
}

__global__ void
norm_kernel(const int8_t* q, uint64_t count, uint32_t dim_words, int32_t* norms) {
    uint64_t r = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= count) {
        return;
    }
    const int* row = reinterpret_cast<const int*>(q) + r * dim_words;
    int sum = 0;
    for (uint32_t w = 0; w < dim_words; ++w) {
        sum = __dp4a(row[w], row[w], sum);
    }
    norms[r] = sum;
}

class RawFeeder {
public:
    RawFeeder(const float* host, uint64_t count, uint32_t dim)
        : host_(host), count_(count), dim_(dim) {
    }

    bool
    init(Workspace& ws) {
        const uint64_t row_bytes = static_cast<uint64_t>(dim_) * sizeof(float);
        const uint64_t total = count_ * row_bytes;
        const uint64_t avail = ws.available();
        if (total <= avail / 2) {
            resident_ = true;
            chunk_rows_ = count_;
        } else {
            chunk_rows_ = std::min<uint64_t>(count_, (avail / 4) / row_bytes);
            if (chunk_rows_ == 0) {
                return false;
            }
        }
        return ws.alloc(buffer_, chunk_rows_ * dim_);
    }

    template <typename Fn>
    bool
    for_each(Fn fn) {
        for (uint64_t start = 0; start < count_; start += chunk_rows_) {
            const uint64_t rows = std::min(chunk_rows_, count_ - start);
            if (not(resident_ and loaded_)) {
                VSAG_CUDA_TRY(cudaMemcpy(buffer_.Get(),
                                         host_ + start * dim_,
                                         rows * dim_ * sizeof(float),
                                         cudaMemcpyHostToDevice));
                loaded_ = resident_;
            }
            if (not fn(buffer_.Get(), start, rows)) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] uint64_t
    bytes() const {
        return chunk_rows_ * dim_ * sizeof(float);
    }

private:
    const float* host_;
    uint64_t count_;
    uint32_t dim_;
    bool resident_{false};
    bool loaded_{false};
    uint64_t chunk_rows_{0};
    DeviceArray<float> buffer_;
};

}  // namespace

bool
QuantizeRows(const float* vectors,
             uint64_t count,
             const BuildPlan& plan,
             Workspace& ws,
             DeviceArray<int8_t>& d_q,
             DeviceArray<int32_t>& d_norms) {
    {
        RawFeeder feeder(vectors, count, plan.dim);
        if (not feeder.init(ws)) {
            return false;
        }
        DeviceArray<double> d_partial;
        DeviceArray<float> d_mean;
        DeviceArray<uint32_t> d_max;
        if (not ws.alloc(d_partial, static_cast<uint64_t>(kSumSlices) * plan.dim) or
            not ws.alloc(d_mean, plan.dim) or not ws.alloc(d_max, 1)) {
            return false;
        }
        VSAG_CUDA_TRY(cudaMemset(d_partial.Get(), 0, kSumSlices * plan.dim * sizeof(double)));
        VSAG_CUDA_TRY(cudaMemset(d_max.Get(), 0, sizeof(uint32_t)));
        const uint32_t sum_threads = std::min<uint32_t>(plan.dim, 256);
        if (not feeder.for_each([&](const float* x, uint64_t, uint64_t rows) {
                column_sum_kernel<<<kSumSlices, sum_threads>>>(x, rows, plan.dim, d_partial.Get());
                return cudaGetLastError() == cudaSuccess;
            })) {
            return false;
        }
        mean_kernel<<<(plan.dim + 127) / 128, 128>>>(
            d_partial.Get(), kSumSlices, plan.dim, 1.0 / static_cast<double>(count), d_mean.Get());
        VSAG_CUDA_TRY(cudaGetLastError());
        if (not feeder.for_each([&](const float* x, uint64_t, uint64_t rows) {
                abs_max_kernel<<<grid_for(rows * plan.dim), kBlockThreads>>>(
                    x, rows, plan.dim, d_mean.Get(), d_max.Get());
                return cudaGetLastError() == cudaSuccess;
            })) {
            return false;
        }
        uint32_t max_bits = 0;
        VSAG_CUDA_TRY(cudaMemcpy(&max_bits, d_max.Get(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        float max_abs;
        std::memcpy(&max_abs, &max_bits, sizeof(float));
        if (not std::isfinite(max_abs)) {
            return false;
        }
        const float scale = max_abs > 0.0F ? 127.0F / max_abs : 1.0F;
        const uint32_t sign_seed = static_cast<uint32_t>(plan.seed ^ (plan.seed >> 32));
        if (not feeder.for_each([&](const float* x, uint64_t start, uint64_t rows) {
                quantize_kernel<<<grid_for(rows * plan.dim_pad), kBlockThreads>>>(
                    x,
                    rows,
                    plan.dim,
                    plan.dim_pad,
                    d_mean.Get(),
                    scale,
                    sign_seed,
                    d_q.Get() + start * plan.dim_pad);
                return cudaGetLastError() == cudaSuccess;
            })) {
            return false;
        }
        norm_kernel<<<grid_for(count), kBlockThreads>>>(
            d_q.Get(), count, plan.dim_words, d_norms.Get());
        VSAG_CUDA_TRY(cudaGetLastError());
        VSAG_CUDA_TRY(cudaDeviceSynchronize());
        ws.credit(feeder.bytes());
        ws.credit(static_cast<uint64_t>(kSumSlices) * plan.dim * sizeof(double));
    }
    return true;
}

}  // namespace vsag::gpu::pipnn
