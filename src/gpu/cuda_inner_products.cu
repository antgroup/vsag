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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"
#include "gpu_plan.h"

namespace vsag::gpu {

namespace {

constexpr int32_t kThreads = 256;
constexpr uint64_t kLeafStagingBytes = 64ULL << 20;

bool
checked_multiply(uint64_t lhs, uint64_t rhs, uint64_t& result) {
    if (lhs != 0 and rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool
checked_add(uint64_t lhs, uint64_t rhs, uint64_t& result) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

__device__ float
sanitize_distance(float distance) {
    if (not isfinite(distance)) {
        return CUDART_INF_F;
    }
    return distance == 0.0F ? 0.0F : distance;
}

__device__ bool
better(float lhs_distance, uint64_t lhs_id, float rhs_distance, uint64_t rhs_id) {
    return lhs_distance < rhs_distance or (lhs_distance == rhs_distance and lhs_id < rhs_id);
}

__device__ uint64_t
id_gap(uint64_t lhs, uint64_t rhs) {
    return lhs >= rhs ? lhs - rhs : rhs - lhs;
}

__device__ bool
better_leaf(
    float lhs_distance, uint64_t source_id, uint64_t lhs_id, float rhs_distance, uint64_t rhs_id) {
    if (lhs_distance != rhs_distance) {
        return lhs_distance < rhs_distance;
    }
    const uint64_t lhs_gap = id_gap(source_id, lhs_id);
    const uint64_t rhs_gap = id_gap(source_id, rhs_id);
    return lhs_gap < rhs_gap or (lhs_gap == rhs_gap and lhs_id < rhs_id);
}

__global__ void
select_nearest(const float* __restrict__ dots,
               const float* __restrict__ lhs_norms,
               const float* __restrict__ rhs_norms,
               const uint64_t* __restrict__ rhs_ids,
               int32_t lhs_count,
               int32_t rhs_count,
               uint32_t fanout,
               DistanceType distance_type,
               uint32_t* __restrict__ labels) {
    const int32_t lhs = blockIdx.x * blockDim.x + threadIdx.x;
    if (lhs >= lhs_count) {
        return;
    }
    float best_distances[MAX_SELECT_NEIGHBORS];
    uint64_t best_ids[MAX_SELECT_NEIGHBORS];
    uint32_t best_labels[MAX_SELECT_NEIGHBORS];
    uint32_t retained = 0;
    const float* row = dots + static_cast<uint64_t>(lhs) * rhs_count;
    for (int32_t rhs = 0; rhs < rhs_count; ++rhs) {
        float distance = 0.0F;
        if (distance_type == DistanceType::L2Sqr) {
            distance = fmaxf(
                0.0F, sanitize_distance(lhs_norms[lhs] + rhs_norms[rhs] - row[rhs] - row[rhs]));
        } else if (distance_type == DistanceType::InnerProduct) {
            distance = sanitize_distance(1.0F - row[rhs]);
        } else {
            const float similarity =
                fminf(1.0F, fmaxf(-1.0F, row[rhs] * lhs_norms[lhs] * rhs_norms[rhs]));
            distance = sanitize_distance(1.0F - similarity);
        }
        const uint64_t id = rhs_ids[rhs];
        uint32_t position = retained;
        if (retained == fanout) {
            position = fanout - 1;
            if (not better(distance, id, best_distances[position], best_ids[position])) {
                continue;
            }
        } else {
            ++retained;
        }
        while (position > 0 and
               better(distance, id, best_distances[position - 1], best_ids[position - 1])) {
            best_distances[position] = best_distances[position - 1];
            best_ids[position] = best_ids[position - 1];
            best_labels[position] = best_labels[position - 1];
            --position;
        }
        best_distances[position] = distance;
        best_ids[position] = id;
        best_labels[position] = static_cast<uint32_t>(rhs);
    }
    // The API enforces fanout <= rhs_count, so every slot below fanout has
    // been initialized by the selection loop.
    for (uint32_t selected = 0; selected < fanout; ++selected) {
        labels[static_cast<uint64_t>(lhs) * fanout + selected] = best_labels[selected];
    }
}

__global__ void
gather_leaf_vectors(const float* __restrict__ source_vectors,
                    const uint32_t* __restrict__ row_sources,
                    int32_t row_count,
                    int32_t dim,
                    float* __restrict__ vectors) {
    const uint64_t value = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t value_count = static_cast<uint64_t>(row_count) * dim;
    if (value >= value_count) {
        return;
    }
    const uint32_t row = static_cast<uint32_t>(value / dim);
    const uint32_t column = static_cast<uint32_t>(value % dim);
    vectors[value] = source_vectors[static_cast<uint64_t>(row_sources[row]) * dim + column];
}

__global__ void
select_leaf_neighbors(const float* __restrict__ dots,
                      const float* __restrict__ norms,
                      const uint64_t* __restrict__ ids,
                      const uint64_t* __restrict__ leaf_offsets,
                      const uint64_t* __restrict__ dot_offsets,
                      const uint32_t* __restrict__ neighbor_counts,
                      const uint32_t* __restrict__ row_leaves,
                      int32_t row_count,
                      uint32_t max_neighbors,
                      DistanceType distance_type,
                      uint32_t* __restrict__ labels,
                      float* __restrict__ distances) {
    const int32_t source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= row_count) {
        return;
    }
    const uint32_t leaf = row_leaves[source];
    const uint64_t begin = leaf_offsets[leaf];
    const uint64_t end = leaf_offsets[leaf + 1];
    const uint32_t leaf_size = static_cast<uint32_t>(end - begin);
    const uint32_t source_local = static_cast<uint32_t>(source - begin);
    const uint32_t retained_count = neighbor_counts[leaf];
    const uint64_t source_id = ids[source];
    float best_distances[MAX_SELECT_NEIGHBORS];
    uint64_t best_ids[MAX_SELECT_NEIGHBORS];
    uint32_t best_labels[MAX_SELECT_NEIGHBORS];
    uint32_t retained = 0;
    const float* leaf_dots = dots + dot_offsets[leaf];
    for (uint32_t target_local = 0; target_local < leaf_size; ++target_local) {
        if (source_local == target_local) {
            continue;
        }
        const uint32_t row = max(source_local, target_local);
        const uint32_t column = min(source_local, target_local);
        const float dot = leaf_dots[static_cast<uint64_t>(row) * leaf_size + column];
        const uint32_t target = static_cast<uint32_t>(begin + target_local);
        float distance = 0.0F;
        if (distance_type == DistanceType::L2Sqr) {
            distance = fmaxf(0.0F, sanitize_distance(norms[source] + norms[target] - dot - dot));
        } else if (distance_type == DistanceType::InnerProduct) {
            distance = sanitize_distance(1.0F - dot);
        } else {
            const float similarity = fminf(1.0F, fmaxf(-1.0F, dot * norms[source] * norms[target]));
            distance = sanitize_distance(1.0F - similarity);
        }
        const uint64_t target_id = ids[target];
        uint32_t position = retained;
        if (retained == retained_count) {
            position = retained_count - 1;
            if (not better_leaf(
                    distance, source_id, target_id, best_distances[position], best_ids[position])) {
                continue;
            }
        } else {
            ++retained;
        }
        while (position > 0 and better_leaf(distance,
                                            source_id,
                                            target_id,
                                            best_distances[position - 1],
                                            best_ids[position - 1])) {
            best_distances[position] = best_distances[position - 1];
            best_ids[position] = best_ids[position - 1];
            best_labels[position] = best_labels[position - 1];
            --position;
        }
        best_distances[position] = distance;
        best_ids[position] = target_id;
        best_labels[position] = target;
    }
    // The API validates retained_count < leaf_size, so every output slot below
    // retained_count has been initialized by a non-self target above.
    const uint64_t output = static_cast<uint64_t>(source) * max_neighbors;
    for (uint32_t selected = 0; selected < retained_count; ++selected) {
        labels[output + selected] = best_labels[selected];
        distances[output + selected] = best_distances[selected];
    }
}

}  // namespace

bool
CudaSelectNearest(const float* lhs,
                  const float* lhs_norms,
                  uint64_t lhs_count,
                  const float* rhs,
                  const float* rhs_norms,
                  const uint64_t* rhs_ids,
                  uint64_t rhs_count,
                  int32_t dim,
                  uint32_t fanout,
                  DistanceType distance_type,
                  uint32_t* labels,
                  uint64_t budget_bytes,
                  uint64_t min_work) {
    const bool needs_norms = distance_type != DistanceType::InnerProduct;
    if (lhs == nullptr or rhs == nullptr or rhs_ids == nullptr or labels == nullptr or
        (needs_norms and (lhs_norms == nullptr or rhs_norms == nullptr)) or lhs_count == 0 or
        rhs_count == 0 or dim <= 0 or fanout == 0 or fanout > MAX_SELECT_NEIGHBORS or
        fanout > rhs_count or lhs_count > std::numeric_limits<int32_t>::max() or
        rhs_count > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    if (not WorthOffloading(lhs_count, rhs_count, dim, min_work) or not CudaAvailable()) {
        return false;
    }
    uint64_t lhs_values = 0;
    uint64_t rhs_values = 0;
    uint64_t dot_values = 0;
    uint64_t label_values = 0;
    if (not checked_multiply(lhs_count, static_cast<uint64_t>(dim), lhs_values) or
        not checked_multiply(rhs_count, static_cast<uint64_t>(dim), rhs_values) or
        not checked_multiply(lhs_count, rhs_count, dot_values) or
        not checked_multiply(lhs_count, fanout, label_values)) {
        return false;
    }
    uint64_t float_values = 0;
    uint64_t device_bytes = 0;
    if (not checked_add(lhs_values, rhs_values, float_values) or
        not checked_add(float_values, dot_values, float_values) or
        (needs_norms and (not checked_add(float_values, lhs_count, float_values) or
                          not checked_add(float_values, rhs_count, float_values))) or
        not checked_multiply(float_values, sizeof(float), device_bytes)) {
        return false;
    }
    uint64_t id_bytes = 0;
    uint64_t label_bytes = 0;
    if (not checked_multiply(rhs_count, sizeof(uint64_t), id_bytes) or
        not checked_multiply(label_values, sizeof(uint32_t), label_bytes) or
        not checked_add(device_bytes, id_bytes, device_bytes) or
        not checked_add(device_bytes, label_bytes, device_bytes)) {
        return false;
    }
    if (device_bytes > budget_bytes) {
        return false;
    }

    ClearPendingError();
    DeviceArray<float> device_lhs;
    DeviceArray<float> device_rhs;
    DeviceArray<float> device_dots;
    DeviceArray<float> device_lhs_norms;
    DeviceArray<float> device_rhs_norms;
    DeviceArray<uint64_t> device_rhs_ids;
    DeviceArray<uint32_t> device_labels;
    Stream stream;
    BlasHandle blas;
    if (not stream.Create() or not blas.Create(stream.Get()) or not device_lhs.Alloc(lhs_values) or
        not device_rhs.Alloc(rhs_values) or not device_dots.Alloc(dot_values) or
        not device_rhs_ids.Alloc(rhs_count) or not device_labels.Alloc(label_values) or
        (needs_norms and
         (not device_lhs_norms.Alloc(lhs_count) or not device_rhs_norms.Alloc(rhs_count)))) {
        return false;
    }
    if (cudaMemcpyAsync(device_lhs.Get(),
                        lhs,
                        lhs_values * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_rhs.Get(),
                        rhs,
                        rhs_values * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_rhs_ids.Get(),
                        rhs_ids,
                        rhs_count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        (needs_norms and (cudaMemcpyAsync(device_lhs_norms.Get(),
                                          lhs_norms,
                                          lhs_count * sizeof(float),
                                          cudaMemcpyHostToDevice,
                                          stream.Get()) != cudaSuccess or
                          cudaMemcpyAsync(device_rhs_norms.Get(),
                                          rhs_norms,
                                          rhs_count * sizeof(float),
                                          cudaMemcpyHostToDevice,
                                          stream.Get()) != cudaSuccess))) {
        return false;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    if (cublasSgemm(blas.Get(),
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    static_cast<int32_t>(rhs_count),
                    static_cast<int32_t>(lhs_count),
                    dim,
                    &alpha,
                    device_rhs.Get(),
                    dim,
                    device_lhs.Get(),
                    dim,
                    &beta,
                    device_dots.Get(),
                    static_cast<int32_t>(rhs_count)) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    select_nearest<<<(static_cast<int32_t>(lhs_count) + kThreads - 1) / kThreads,
                     kThreads,
                     0,
                     stream.Get()>>>(device_dots.Get(),
                                     needs_norms ? device_lhs_norms.Get() : nullptr,
                                     needs_norms ? device_rhs_norms.Get() : nullptr,
                                     device_rhs_ids.Get(),
                                     static_cast<int32_t>(lhs_count),
                                     static_cast<int32_t>(rhs_count),
                                     fanout,
                                     distance_type,
                                     device_labels.Get());
    std::vector<uint32_t> staged(label_values);
    if (cudaMemcpyAsync(staged.data(),
                        device_labels.Get(),
                        label_values * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        stream.Get()) != cudaSuccess or
        cudaStreamSynchronize(stream.Get()) != cudaSuccess or cudaGetLastError() != cudaSuccess) {
        return false;
    }
    std::copy(staged.begin(), staged.end(), labels);
    return true;
}

bool
CudaSelectLeafNeighbors(const float* const* source_rows,
                        uint64_t source_count,
                        const uint32_t* row_sources,
                        const float* norms,
                        const uint64_t* ids,
                        const uint64_t* leaf_offsets,
                        const uint32_t* neighbor_counts,
                        uint64_t leaf_count,
                        int32_t dim,
                        uint32_t max_neighbors,
                        DistanceType distance_type,
                        uint32_t* labels,
                        float* distances,
                        uint64_t budget_bytes,
                        uint64_t min_work) {
    const bool needs_norms = distance_type != DistanceType::InnerProduct;
    if (source_rows == nullptr or row_sources == nullptr or ids == nullptr or
        leaf_offsets == nullptr or neighbor_counts == nullptr or labels == nullptr or
        distances == nullptr or (needs_norms and norms == nullptr) or source_count == 0 or
        leaf_count == 0 or leaf_count > std::numeric_limits<uint32_t>::max() or dim <= 0 or
        max_neighbors == 0 or max_neighbors > MAX_SELECT_NEIGHBORS or leaf_offsets[0] != 0) {
        return false;
    }
    for (uint64_t source = 0; source < source_count; ++source) {
        if (source_rows[source] == nullptr) {
            return false;
        }
    }

    uint64_t dot_values = 0;
    uint64_t work = 0;
    std::vector<uint64_t> dot_offsets(leaf_count);
    for (uint64_t leaf = 0; leaf < leaf_count; ++leaf) {
        if (leaf_offsets[leaf + 1] <= leaf_offsets[leaf]) {
            return false;
        }
        const uint64_t leaf_size = leaf_offsets[leaf + 1] - leaf_offsets[leaf];
        if (leaf_size > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) or
            neighbor_counts[leaf] == 0 or neighbor_counts[leaf] > max_neighbors or
            neighbor_counts[leaf] >= leaf_size) {
            return false;
        }
        uint64_t leaf_dots = 0;
        uint64_t leaf_work = 0;
        if (not checked_multiply(leaf_size, leaf_size, leaf_dots) or
            not checked_multiply(leaf_dots, static_cast<uint64_t>(dim), leaf_work)) {
            return false;
        }
        dot_offsets[leaf] = dot_values;
        if (not checked_add(dot_values, leaf_dots, dot_values)) {
            return false;
        }
        if (not checked_add(work, leaf_work, work)) {
            work = std::numeric_limits<uint64_t>::max();
        }
    }
    const uint64_t row_count = leaf_offsets[leaf_count];
    const uint64_t work_floor = min_work > 0 ? min_work : kMinWorkForGpu;
    if (row_count == 0 or row_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) or
        row_count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) or
        work < work_floor or not CudaAvailable()) {
        return false;
    }
    for (uint64_t row = 0; row < row_count; ++row) {
        if (row_sources[row] >= source_count) {
            return false;
        }
    }

    uint64_t source_values = 0;
    uint64_t vector_values = 0;
    uint64_t output_values = 0;
    if (not checked_multiply(source_count, static_cast<uint64_t>(dim), source_values) or
        not checked_multiply(row_count, static_cast<uint64_t>(dim), vector_values) or
        not checked_multiply(row_count, max_neighbors, output_values)) {
        return false;
    }
    uint64_t float_values = 0;
    if (not checked_add(source_values, vector_values, float_values) or
        not checked_add(float_values, dot_values, float_values) or
        not checked_add(float_values, output_values, float_values) or
        (needs_norms and not checked_add(float_values, row_count, float_values))) {
        return false;
    }
    uint64_t device_bytes = 0;
    uint64_t extra_bytes = 0;
    if (not checked_multiply(float_values, sizeof(float), device_bytes) or
        not checked_multiply(row_count, sizeof(uint64_t) + 2 * sizeof(uint32_t), extra_bytes) or
        not checked_add(device_bytes, extra_bytes, device_bytes) or
        not checked_multiply(output_values, sizeof(uint32_t), extra_bytes) or
        not checked_add(device_bytes, extra_bytes, device_bytes) or
        not checked_multiply(leaf_count + 1, sizeof(uint64_t), extra_bytes) or
        not checked_add(device_bytes, extra_bytes, device_bytes) or
        not checked_multiply(leaf_count, sizeof(uint64_t) + sizeof(uint32_t), extra_bytes) or
        not checked_add(device_bytes, extra_bytes, device_bytes) or device_bytes > budget_bytes) {
        return false;
    }

    std::vector<uint32_t> row_leaves(row_count);
    for (uint64_t leaf = 0; leaf < leaf_count; ++leaf) {
        std::fill(row_leaves.begin() + static_cast<int64_t>(leaf_offsets[leaf]),
                  row_leaves.begin() + static_cast<int64_t>(leaf_offsets[leaf + 1]),
                  static_cast<uint32_t>(leaf));
    }

    ClearPendingError();
    DeviceArray<float> device_source_vectors;
    DeviceArray<float> device_vectors;
    DeviceArray<float> device_norms;
    DeviceArray<float> device_dots;
    DeviceArray<float> device_distances;
    DeviceArray<uint64_t> device_ids;
    DeviceArray<uint64_t> device_leaf_offsets;
    DeviceArray<uint64_t> device_dot_offsets;
    DeviceArray<uint32_t> device_neighbor_counts;
    DeviceArray<uint32_t> device_row_sources;
    DeviceArray<uint32_t> device_row_leaves;
    DeviceArray<uint32_t> device_labels;
    const uint64_t row_bytes = static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t staging_rows =
        std::max<uint64_t>(1, std::min<uint64_t>(source_count, kLeafStagingBytes / row_bytes));
    PinnedArray<float> staging_vectors;
    Stream stream;
    BlasHandle blas;
    if (not staging_vectors.Alloc(staging_rows * static_cast<uint64_t>(dim)) or
        not stream.Create() or not blas.Create(stream.Get()) or
        not device_source_vectors.Alloc(source_values) or not device_vectors.Alloc(vector_values) or
        not device_dots.Alloc(dot_values) or not device_distances.Alloc(output_values) or
        not device_ids.Alloc(row_count) or not device_leaf_offsets.Alloc(leaf_count + 1) or
        not device_dot_offsets.Alloc(leaf_count) or not device_neighbor_counts.Alloc(leaf_count) or
        not device_row_sources.Alloc(row_count) or not device_row_leaves.Alloc(row_count) or
        not device_labels.Alloc(output_values) or
        (needs_norms and not device_norms.Alloc(row_count))) {
        return false;
    }

    for (uint64_t begin = 0; begin < source_count; begin += staging_rows) {
        const uint64_t rows = std::min<uint64_t>(staging_rows, source_count - begin);
        for (uint64_t row = 0; row < rows; ++row) {
            std::copy(source_rows[begin + row],
                      source_rows[begin + row] + dim,
                      staging_vectors.Get() + row * static_cast<uint64_t>(dim));
        }
        if (cudaMemcpyAsync(device_source_vectors.Get() + begin * static_cast<uint64_t>(dim),
                            staging_vectors.Get(),
                            rows * row_bytes,
                            cudaMemcpyHostToDevice,
                            stream.Get()) != cudaSuccess or
            cudaStreamSynchronize(stream.Get()) != cudaSuccess) {
            return false;
        }
    }

    if (cudaMemcpyAsync(device_row_sources.Get(),
                        row_sources,
                        row_count * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_ids.Get(),
                        ids,
                        row_count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_leaf_offsets.Get(),
                        leaf_offsets,
                        (leaf_count + 1) * sizeof(uint64_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_dot_offsets.Get(),
                        dot_offsets.data(),
                        leaf_count * sizeof(uint64_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_neighbor_counts.Get(),
                        neighbor_counts,
                        leaf_count * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(device_row_leaves.Get(),
                        row_leaves.data(),
                        row_count * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess or
        (needs_norms and cudaMemcpyAsync(device_norms.Get(),
                                         norms,
                                         row_count * sizeof(float),
                                         cudaMemcpyHostToDevice,
                                         stream.Get()) != cudaSuccess)) {
        return false;
    }

    const uint64_t gather_values = row_count * static_cast<uint64_t>(dim);
    gather_leaf_vectors<<<(gather_values + kThreads - 1) / kThreads, kThreads, 0, stream.Get()>>>(
        device_source_vectors.Get(),
        device_row_sources.Get(),
        static_cast<int32_t>(row_count),
        dim,
        device_vectors.Get());

    const float alpha = 1.0F;
    const float beta = 0.0F;
    for (uint64_t leaf = 0; leaf < leaf_count; ++leaf) {
        const int32_t leaf_size = static_cast<int32_t>(leaf_offsets[leaf + 1] - leaf_offsets[leaf]);
        const float* leaf_vectors =
            device_vectors.Get() + leaf_offsets[leaf] * static_cast<uint64_t>(dim);
        if (cublasSsyrk(blas.Get(),
                        CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_T,
                        leaf_size,
                        dim,
                        &alpha,
                        leaf_vectors,
                        dim,
                        &beta,
                        device_dots.Get() + dot_offsets[leaf],
                        leaf_size) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
    }
    select_leaf_neighbors<<<(row_count + kThreads - 1) / kThreads, kThreads, 0, stream.Get()>>>(
        device_dots.Get(),
        needs_norms ? device_norms.Get() : nullptr,
        device_ids.Get(),
        device_leaf_offsets.Get(),
        device_dot_offsets.Get(),
        device_neighbor_counts.Get(),
        device_row_leaves.Get(),
        static_cast<int32_t>(row_count),
        max_neighbors,
        distance_type,
        device_labels.Get(),
        device_distances.Get());

    std::vector<uint32_t> staged_labels(output_values);
    std::vector<float> staged_distances(output_values);
    if (cudaMemcpyAsync(staged_labels.data(),
                        device_labels.Get(),
                        output_values * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(staged_distances.data(),
                        device_distances.Get(),
                        output_values * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream.Get()) != cudaSuccess or
        cudaStreamSynchronize(stream.Get()) != cudaSuccess or cudaGetLastError() != cudaSuccess) {
        return false;
    }
    std::copy(staged_labels.begin(), staged_labels.end(), labels);
    std::copy(staged_distances.begin(), staged_distances.end(), distances);
    return true;
}

}  // namespace vsag::gpu
