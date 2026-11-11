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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"
#include "gpu_plan.h"

// Nearest-centroid assignment.
//
// ||x - c||^2 = ||x||^2 - 2<x,c> + ||c||^2, and the argmin does not depend on
// ||x||^2, so the pass is one SGEMM for the inner products and a reduction that
// folds in ||c||^2. The k x n distance matrix is the largest thing here and never
// leaves the device: the reduction consumes it in place, only labels come back.

namespace vsag::gpu {

namespace {

constexpr int kThreads = 256;

__global__ void
init_best(float* best_val, int* best_idx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_val[i] = FLT_MAX;
        best_idx[i] = -1;
    }
}

__global__ void
row_sqr_norms(const float* __restrict__ x, int dim, int rows, float* __restrict__ out) {
    int r = blockIdx.x;
    const float* p = x + static_cast<uint64_t>(r) * dim;
    float acc = 0.F;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        acc += p[i] * p[i];
    }
    __shared__ float s[kThreads];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int t = blockDim.x / 2; t > 0; t >>= 1) {
        if (threadIdx.x < t) {
            s[threadIdx.x] += s[threadIdx.x + t];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[r] = s[0];
    }
}

// dist is column-major (kc rows x n cols). Fuses "+ ||c||^2" into the argmin so
// the k x n matrix is read exactly once and never leaves the device.
__global__ void
fused_add_csqr_argmin(const float* __restrict__ dist,
                      int ld,
                      const float* __restrict__ c_sqr,
                      int kc,
                      int centroid_offset,
                      int* __restrict__ best_idx,
                      float* __restrict__ best_val) {
    extern __shared__ char raw[];
    float* s_val = reinterpret_cast<float*>(raw);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    int q = blockIdx.x;
    const float* col = dist + static_cast<uint64_t>(q) * ld;
    float lv = FLT_MAX;
    int li = -1;
    for (int c = threadIdx.x; c < kc; c += blockDim.x) {
        float v = col[c] + c_sqr[c];
        if (v < lv) {
            lv = v;
            li = c + centroid_offset;
        }
    }
    s_val[threadIdx.x] = lv;
    s_idx[threadIdx.x] = li;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s and s_val[threadIdx.x + s] < s_val[threadIdx.x]) {
            s_val[threadIdx.x] = s_val[threadIdx.x + s];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 and s_val[0] < best_val[q]) {
        best_val[q] = s_val[0];
        best_idx[q] = s_idx[0];
    }
}

__global__ void
add_qsqr(const float* __restrict__ q_sqr, float* __restrict__ best_val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        best_val[i] += q_sqr[i];
    }
}

/// Everything one stream owns. They alternate chunks, so each needs its own copy
/// of every buffer; that is what lets a transfer overlap the other's compute.
struct AssignStream {
    Stream stream;
    BlasHandle blas;
    DeviceArray<float> rows;       ///< the chunk's query rows
    DeviceArray<float> distances;  ///< k_chunk x n_chunk, stays on the device
    DeviceArray<float> best_value;
    DeviceArray<float> row_sqr;
    DeviceArray<int> best_index;
    DeviceArray<float> centroids;  ///< only used when they are streamed
    DeviceArray<float> centroid_sqr;
    PinnedArray<float> host_rows;
    PinnedArray<int> host_labels;
    PinnedArray<float> host_values;
    uint64_t pending_offset{0};  ///< chunk in flight, waiting to be read back
    uint64_t pending_count{0};

    /// An error return unwinds this while the stream may still be copying into
    /// the buffers it owns, so drain it before any of them is released. That
    /// covers `blas` as well: Alloc creates the handle from this stream and binds
    /// it there, so there is no handle without a stream and no cuBLAS work
    /// anywhere but in it.
    ~AssignStream() {
        if (stream.Get() != nullptr) {
            cudaStreamSynchronize(stream.Get());
        }
    }

    bool
    Alloc(const AssignPlan& plan, int32_t dim, bool own_centroids) {
        return stream.Create() and blas.Create(stream.Get()) and rows.Alloc(plan.n_chunk * dim) and
               distances.Alloc(plan.k_chunk * plan.n_chunk) and best_value.Alloc(plan.n_chunk) and
               row_sqr.Alloc(plan.n_chunk) and best_index.Alloc(plan.n_chunk) and
               host_rows.Alloc(plan.n_chunk * dim) and host_labels.Alloc(plan.n_chunk) and
               host_values.Alloc(plan.n_chunk) and
               (not own_centroids or
                (centroids.Alloc(plan.k_chunk * dim) and centroid_sqr.Alloc(plan.k_chunk)));
    }
};

/// What every chunk needs, fixed once the plan is made.
struct AssignContext {
    const float* query{nullptr};
    const float* centroids{nullptr};  ///< host side, read when they are streamed
    uint64_t k{0};
    int32_t dim{0};
    const AssignPlan* plan{nullptr};
    const float* device_centroids{nullptr};  ///< device side, null when streamed
    const float* device_centroid_sqr{nullptr};
};

/// Uploads the whole centroid set once with its row norms, for when it fits and
/// both streams share it. Synchronous, because both read what it leaves behind.
bool
UploadResidentCentroids(const AssignContext& ctx,
                        cudaStream_t stream,
                        float* device_centroids,
                        float* device_centroid_sqr) {
    const uint64_t bytes = ctx.k * static_cast<uint64_t>(ctx.dim) * sizeof(float);
    if (cudaMemcpyAsync(device_centroids, ctx.centroids, bytes, cudaMemcpyHostToDevice, stream) !=
        cudaSuccess) {
        return false;
    }
    row_sqr_norms<<<static_cast<int>(ctx.k), kThreads, 0, stream>>>(
        device_centroids, ctx.dim, static_cast<int>(ctx.k), device_centroid_sqr);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

/// Folds one slice of centroids into the running best for a chunk of rows: the
/// SGEMM gives -2<x,c>, the reduction adds ||c||^2 and keeps the winner. ||x||^2
/// is the same for every centroid, so it goes in once at the end, where the
/// distance itself is wanted rather than the argmin.
bool
IssueCentroidSlice(const AssignContext& ctx, AssignStream& slot, uint64_t koff, uint64_t rows) {
    const uint64_t kc = std::min<uint64_t>(ctx.plan->k_chunk, ctx.k - koff);
    const cudaStream_t stream = slot.stream.Get();
    const float* slice = nullptr;
    const float* slice_sqr = nullptr;
    if (ctx.plan->centroids_resident) {
        slice = ctx.device_centroids + koff * ctx.dim;
        slice_sqr = ctx.device_centroid_sqr + koff;
    } else {
        // Each stream owns its slice, so the uploads cannot clobber each other.
        if (cudaMemcpyAsync(slot.centroids.Get(),
                            ctx.centroids + koff * ctx.dim,
                            kc * ctx.dim * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
            return false;
        }
        row_sqr_norms<<<static_cast<int>(kc), kThreads, 0, stream>>>(
            slot.centroids.Get(), ctx.dim, static_cast<int>(kc), slot.centroid_sqr.Get());
        slice = slot.centroids.Get();
        slice_sqr = slot.centroid_sqr.Get();
    }

    const float alpha = -2.0F;
    const float beta = 0.0F;
    if (cublasSgemm(slot.blas.Get(),
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    static_cast<int>(kc),
                    static_cast<int>(rows),
                    ctx.dim,
                    &alpha,
                    slice,
                    ctx.dim,
                    slot.rows.Get(),
                    ctx.dim,
                    &beta,
                    slot.distances.Get(),
                    static_cast<int>(kc)) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    fused_add_csqr_argmin<<<static_cast<int>(rows),
                            kThreads,
                            kThreads * (sizeof(float) + sizeof(int)),
                            stream>>>(slot.distances.Get(),
                                      static_cast<int>(kc),
                                      slice_sqr,
                                      static_cast<int>(kc),
                                      static_cast<int>(koff),
                                      slot.best_index.Get(),
                                      slot.best_value.Get());
    return true;
}

/// Issues one chunk on its stream: the upload, every centroid slice, then the two
/// read-backs into pinned memory. Nothing is synchronized, so the other stream
/// runs meanwhile; the caller waits on the slot before reading.
bool
IssueChunk(const AssignContext& ctx, AssignStream& slot, uint64_t offset, uint64_t rows) {
    const cudaStream_t stream = slot.stream.Get();
    std::copy(
        ctx.query + offset * ctx.dim, ctx.query + (offset + rows) * ctx.dim, slot.host_rows.Get());
    if (cudaMemcpyAsync(slot.rows.Get(),
                        slot.host_rows.Get(),
                        rows * ctx.dim * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
        return false;
    }
    init_best<<<static_cast<int>((rows + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        slot.best_value.Get(), slot.best_index.Get(), static_cast<int>(rows));
    row_sqr_norms<<<static_cast<int>(rows), kThreads, 0, stream>>>(
        slot.rows.Get(), ctx.dim, static_cast<int>(rows), slot.row_sqr.Get());

    for (uint64_t koff = 0; koff < ctx.k; koff += ctx.plan->k_chunk) {
        if (not IssueCentroidSlice(ctx, slot, koff, rows)) {
            return false;
        }
    }

    // The winner is settled, so the row norm turns the key into a distance.
    add_qsqr<<<static_cast<int>((rows + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        slot.row_sqr.Get(), slot.best_value.Get(), static_cast<int>(rows));
    if (cudaMemcpyAsync(slot.host_labels.Get(),
                        slot.best_index.Get(),
                        rows * sizeof(int),
                        cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess or
        cudaMemcpyAsync(slot.host_values.Get(),
                        slot.best_value.Get(),
                        rows * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
        return false;
    }
    slot.pending_offset = offset;
    slot.pending_count = rows;
    return true;
}

}  // namespace

bool
CudaAssignNearest(const float* query,
                  uint64_t query_count,
                  const float* centroids,
                  uint64_t k,
                  int32_t dim,
                  int32_t* labels,
                  double* error,
                  uint64_t budget_bytes,
                  uint64_t min_work) {
    if (query == nullptr or centroids == nullptr or labels == nullptr or query_count == 0 or
        k == 0 or dim <= 0) {
        return false;
    }
    // The plan first, because it needs no CUDA: a build that did not ask for a
    // device passes 0, which it refuses, so the runtime is never touched.
    const AssignPlan plan = PlanAssign(query_count, k, dim, budget_bytes, min_work);
    if (not plan.offload or not CudaAvailable()) {
        return false;
    }
    // Clear any latched error, so the launch check below reports only this pass.
    ClearPendingError();

    // Resident centroids are shared, so they live outside either slot, and are
    // declared first so they outlive the streams that read them.
    DeviceArray<float> shared_centroids;
    DeviceArray<float> shared_centroid_sqr;
    AssignStream slots[kAssignStreams];
    for (auto& slot : slots) {
        if (not slot.Alloc(plan, dim, not plan.centroids_resident)) {
            return false;
        }
    }
    AssignContext ctx;
    ctx.query = query;
    ctx.centroids = centroids;
    ctx.k = k;
    ctx.dim = dim;
    ctx.plan = &plan;
    if (plan.centroids_resident) {
        if (not shared_centroids.Alloc(k * dim) or not shared_centroid_sqr.Alloc(k)) {
            return false;
        }
        if (not UploadResidentCentroids(
                ctx, slots[0].stream.Get(), shared_centroids.Get(), shared_centroid_sqr.Get())) {
            return false;
        }
        ctx.device_centroids = shared_centroids.Get();
        ctx.device_centroid_sqr = shared_centroid_sqr.Get();
    }

    // Staged and handed over only once the whole pass succeeds, because false has
    // to leave the caller's buffer as it found it.
    std::vector<int32_t> staged_labels(query_count);
    double error_sum = 0.0;

    // Copies a finished chunk out of pinned memory; the slot must be synchronized.
    auto drain = [&](AssignStream& slot) {
        for (uint64_t i = 0; i < slot.pending_count; ++i) {
            staged_labels[slot.pending_offset + i] = slot.host_labels.Get()[i];
            error_sum += slot.host_values.Get()[i];
        }
        slot.pending_count = 0;
    };

    for (uint64_t off = 0, chunk = 0; off < query_count; off += plan.n_chunk, ++chunk) {
        const uint64_t cur = std::min<uint64_t>(plan.n_chunk, query_count - off);
        AssignStream& slot = slots[chunk % kAssignStreams];

        // Wait only for the chunk that used this slot two iterations ago.
        if (cudaStreamSynchronize(slot.stream.Get()) != cudaSuccess) {
            return false;
        }
        drain(slot);
        if (not IssueChunk(ctx, slot, off, cur)) {
            return false;
        }
        // No synchronize: the next chunk goes to the other stream and overlaps.
    }

    for (auto& slot : slots) {
        if (cudaStreamSynchronize(slot.stream.Get()) != cudaSuccess) {
            return false;
        }
        drain(slot);
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    std::copy(staged_labels.begin(), staged_labels.end(), labels);
    if (error != nullptr) {
        *error = error_sum / static_cast<double>(query_count);
    }
    return true;
}

}  // namespace vsag::gpu
