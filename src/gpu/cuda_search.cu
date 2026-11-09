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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "cuda_backend.h"
#include "cuda_raii.h"

namespace vsag::gpu {

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kFixedReserveBytes = 1ULL << 20;
constexpr uint32_t kRabitqMaxCandidates = 64;
constexpr uint64_t kInvalidRabitqRecord = 0xffffffffffffffffULL;

struct RabitqSegment {
    uint32_t query;
    uint32_t probe;
    uint64_t begin;
    uint32_t count;
};

__device__ bool
rabitq_less(float lhs_distance, uint64_t lhs_record, float rhs_distance, uint64_t rhs_record) {
    return lhs_distance < rhs_distance or
           (lhs_distance == rhs_distance and lhs_record < rhs_record);
}

__global__ void
build_rabitq_lut(const float* queries, uint64_t dim, uint64_t group_count, float* lut) {
    const uint64_t group = blockIdx.x;
    const uint64_t query = blockIdx.y;
    if (threadIdx.x >= 16) {
        return;
    }
    const uint32_t mask = threadIdx.x;
    float value = 0.0F;
    for (uint32_t bit = 0; bit < 4; ++bit) {
        const uint64_t d = group * 4 + bit;
        if (d < dim and (mask & (1U << bit)) != 0U) {
            value += queries[query * dim + d];
        }
    }
    lut[(query * group_count + group) * 16 + mask] = value;
}

__global__ void
scan_rabitq_segments(const RabitqSegment* segments,
                     uint64_t segment_count,
                     const uint8_t* filter_codes,
                     const uint64_t* inner_ids,
                     const float* filter_adds,
                     const float* filter_scales,
                     const float* filter_norm_codes,
                     const float* query_norms,
                     const float* query_sums,
                     const float* query_bucket_norm_sqrs,
                     const float* lut,
                     uint64_t group_count,
                     uint64_t filter_code_size,
                     uint32_t filter_bits,
                     uint32_t nprobe,
                     uint32_t candidate_count,
                     float* candidate_distances,
                     float* candidate_filter_ips,
                     uint64_t* candidate_records,
                     uint32_t* candidate_probes) {
    const uint64_t segment_id = blockIdx.x;
    if (segment_id >= segment_count) {
        return;
    }
    const auto segment = segments[segment_id];
    const uint32_t lane = threadIdx.x;
    float distance = FLT_MAX;
    float filter_ip = 0.0F;
    uint64_t record = kInvalidRabitqRecord;
    if (lane < segment.count) {
        record = segment.begin + lane;
        const float filter_norm = filter_norm_codes[record];
        if (inner_ids[record] != kInvalidRabitqRecord and isfinite(filter_norm) and
            filter_norm > 0.0F and isfinite(filter_adds[record]) and
            isfinite(filter_scales[record]) and fabsf(filter_scales[record]) > FLT_EPSILON) {
            const auto* code = filter_codes + record * filter_code_size;
            float lookup_sum = 0.0F;
            for (uint32_t plane = 0; plane < filter_bits; ++plane) {
                float plane_sum = 0.0F;
                const auto* plane_code = code + plane * ((group_count + 1) / 2);
                for (uint64_t group = 0; group < group_count; ++group) {
                    const uint8_t byte = plane_code[group >> 1];
                    const uint8_t nibble = (group & 1U) == 0U ? byte & 0x0FU : byte >> 4U;
                    plane_sum +=
                        lut[(static_cast<uint64_t>(segment.query) * group_count + group) * 16 +
                            nibble];
                }
                lookup_sum += static_cast<float>(1U << (filter_bits - plane - 1U)) * plane_sum;
            }
            const float center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
            filter_ip = (lookup_sum - center * query_sums[segment.query]) / filter_norm;
            distance = filter_adds[record] +
                       query_bucket_norm_sqrs[static_cast<uint64_t>(segment.query) * nprobe +
                                              segment.probe] +
                       filter_scales[record] * query_norms[segment.query] * filter_ip;
            if (not isfinite(distance)) {
                distance = FLT_MAX;
            }
        }
    }

    __shared__ float shared_distances[kThreads];
    __shared__ float shared_filter_ips[kThreads];
    __shared__ uint64_t shared_records[kThreads];
    shared_distances[lane] = distance;
    shared_filter_ips[lane] = filter_ip;
    shared_records[lane] = record;
    __syncthreads();
    for (uint32_t size = 2; size <= kThreads; size <<= 1U) {
        for (uint32_t stride = size >> 1U; stride > 0; stride >>= 1U) {
            const uint32_t other = lane ^ stride;
            if (other > lane) {
                const bool ascending = (lane & size) == 0U;
                const bool other_less = rabitq_less(shared_distances[other],
                                                    shared_records[other],
                                                    shared_distances[lane],
                                                    shared_records[lane]);
                if (other_less == ascending) {
                    const float swap_distance = shared_distances[lane];
                    const float swap_ip = shared_filter_ips[lane];
                    const uint64_t swap_record = shared_records[lane];
                    shared_distances[lane] = shared_distances[other];
                    shared_filter_ips[lane] = shared_filter_ips[other];
                    shared_records[lane] = shared_records[other];
                    shared_distances[other] = swap_distance;
                    shared_filter_ips[other] = swap_ip;
                    shared_records[other] = swap_record;
                }
            }
            __syncthreads();
        }
    }
    if (lane < candidate_count) {
        const uint64_t output = segment_id * candidate_count + lane;
        candidate_distances[output] = shared_distances[lane];
        candidate_filter_ips[output] = shared_filter_ips[lane];
        candidate_records[output] = shared_records[lane];
        candidate_probes[output] = segment.probe;
    }
}

__global__ void
merge_rabitq_candidates(float* candidate_distances,
                        const float* candidate_filter_ips,
                        uint64_t* candidate_records,
                        const uint32_t* candidate_probes,
                        const uint64_t* query_segment_offsets,
                        uint32_t candidate_count,
                        float* selected_filter_ips,
                        uint64_t* selected_records,
                        uint32_t* selected_probes) {
    const uint64_t query = blockIdx.x;
    const uint64_t begin = query_segment_offsets[query] * candidate_count;
    const uint64_t end = query_segment_offsets[query + 1] * candidate_count;
    __shared__ float best_distances[kThreads];
    __shared__ uint64_t best_offsets[kThreads];
    for (uint32_t rank = 0; rank < candidate_count; ++rank) {
        float best_distance = FLT_MAX;
        uint64_t best_offset = kInvalidRabitqRecord;
        for (uint64_t offset = begin + threadIdx.x; offset < end; offset += blockDim.x) {
            const auto record = candidate_records[offset];
            const float distance = candidate_distances[offset];
            if (record != kInvalidRabitqRecord and
                rabitq_less(distance,
                            record,
                            best_distance,
                            best_offset == kInvalidRabitqRecord ? best_offset
                                                                : candidate_records[best_offset])) {
                best_distance = distance;
                best_offset = offset;
            }
        }
        best_distances[threadIdx.x] = best_distance;
        best_offsets[threadIdx.x] = best_offset;
        __syncthreads();
        for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1U) {
            if (threadIdx.x < stride) {
                const uint64_t other_offset = best_offsets[threadIdx.x + stride];
                const uint64_t current_offset = best_offsets[threadIdx.x];
                const uint64_t other_record = other_offset == kInvalidRabitqRecord
                                                  ? kInvalidRabitqRecord
                                                  : candidate_records[other_offset];
                const uint64_t current_record = current_offset == kInvalidRabitqRecord
                                                    ? kInvalidRabitqRecord
                                                    : candidate_records[current_offset];
                if (rabitq_less(best_distances[threadIdx.x + stride],
                                other_record,
                                best_distances[threadIdx.x],
                                current_record)) {
                    best_distances[threadIdx.x] = best_distances[threadIdx.x + stride];
                    best_offsets[threadIdx.x] = other_offset;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const uint64_t selected = best_offsets[0];
            const uint64_t output = query * candidate_count + rank;
            if (selected == kInvalidRabitqRecord) {
                selected_records[output] = kInvalidRabitqRecord;
                selected_filter_ips[output] = 0.0F;
                selected_probes[output] = 0;
            } else {
                selected_records[output] = candidate_records[selected];
                selected_filter_ips[output] = candidate_filter_ips[selected];
                selected_probes[output] = candidate_probes[selected];
                candidate_distances[selected] = FLT_MAX;
                candidate_records[selected] = kInvalidRabitqRecord;
            }
        }
        __syncthreads();
    }
}

__global__ void
refine_rabitq_candidates(const uint8_t* supplement_codes,
                         const uint64_t* index_inner_ids,
                         const float* filter_norm_codes,
                         const float* full_adds,
                         const float* full_norm_codes,
                         const float* base_norms,
                         const float* base_errors,
                         const float* normalized_queries,
                         const float* query_norms,
                         const float* query_sums,
                         const float* query_bucket_norm_sqrs,
                         const float* selected_filter_ips,
                         const uint64_t* selected_records,
                         const uint32_t* selected_probes,
                         uint64_t dim,
                         uint64_t supplement_code_size,
                         uint32_t filter_bits,
                         uint32_t supplement_bits,
                         uint32_t nprobe,
                         uint32_t candidate_count,
                         uint32_t topk,
                         uint64_t* output_inner_ids,
                         float* output_distances) {
    const uint64_t query = blockIdx.x;
    const uint32_t rank = threadIdx.x;
    float distance = FLT_MAX;
    uint64_t inner_id = kInvalidRabitqRecord;
    if (rank < candidate_count) {
        const uint64_t selected = query * candidate_count + rank;
        const uint64_t record = selected_records[selected];
        if (record != kInvalidRabitqRecord) {
            inner_id = index_inner_ids[record];
            const auto* supplement = supplement_codes + record * supplement_code_size;
            const uint64_t plane_bytes = (dim + 7) / 8;
            float supplement_ip = 0.0F;
            for (uint64_t d = 0; d < dim; ++d) {
                uint32_t code = 0;
                const uint8_t mask = static_cast<uint8_t>(1U << (d & 7U));
                for (uint32_t bit = 0; bit < supplement_bits; ++bit) {
                    if ((supplement[static_cast<uint64_t>(bit) * plane_bytes + (d >> 3U)] & mask) !=
                        0U) {
                        code |= 1U << bit;
                    }
                }
                supplement_ip += normalized_queries[query * dim + d] * code;
            }
            const float filter_center = 0.5F * static_cast<float>((1U << filter_bits) - 1U);
            const float original_filter_ip =
                selected_filter_ips[selected] * filter_norm_codes[record] +
                filter_center * query_sums[query];
            const float raw_ip = ldexpf(original_filter_ip, supplement_bits) + supplement_ip;
            const float full_center =
                0.5F * static_cast<float>((1U << (filter_bits + supplement_bits)) - 1U);
            const float full_norm = full_norm_codes[record];
            float error = base_errors[record];
            if (fabsf(error) < 1e-5F) {
                error = error >= 0.0F ? 1.0F : -1.0F;
            }
            if (isfinite(full_norm) and full_norm > 0.0F and isfinite(error)) {
                const float full_ip = (raw_ip - full_center * query_sums[query]) / full_norm;
                const float scale = -2.0F * base_norms[record] / error;
                distance = full_adds[record] +
                           query_bucket_norm_sqrs[query * nprobe + selected_probes[selected]] +
                           scale * query_norms[query] * full_ip;
                if (not isfinite(distance)) {
                    distance = FLT_MAX;
                }
            }
        }
    }
    __shared__ float refined_distances[kRabitqMaxCandidates];
    __shared__ uint64_t refined_ids[kRabitqMaxCandidates];
    if (rank < candidate_count) {
        refined_distances[rank] = distance;
        refined_ids[rank] = inner_id;
    }
    __syncthreads();
    if (rank == 0) {
        for (uint32_t result_rank = 0; result_rank < topk; ++result_rank) {
            uint32_t best = result_rank;
            for (uint32_t candidate = result_rank + 1; candidate < candidate_count; ++candidate) {
                if (rabitq_less(refined_distances[candidate],
                                refined_ids[candidate],
                                refined_distances[best],
                                refined_ids[best])) {
                    best = candidate;
                }
            }
            const float swap_distance = refined_distances[result_rank];
            const uint64_t swap_id = refined_ids[result_rank];
            refined_distances[result_rank] = refined_distances[best];
            refined_ids[result_rank] = refined_ids[best];
            refined_distances[best] = swap_distance;
            refined_ids[best] = swap_id;
            output_distances[query * topk + result_rank] = refined_distances[result_rank];
            output_inner_ids[query * topk + result_rank] = refined_ids[result_rank];
        }
    }
}

__global__ void
row_sqr_norms_search(const float* __restrict__ rows,
                     int32_t dim,
                     int32_t count,
                     float* __restrict__ norms) {
    const int32_t row = blockIdx.x;
    const float* data = rows + static_cast<uint64_t>(row) * dim;
    float value = 0.0F;
    for (int32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        value += data[i] * data[i];
    }
    __shared__ float partial[kThreads];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        norms[row] = partial[0];
    }
}

__global__ void
add_centroid_norms(float* routing_keys,
                   const float* centroid_norms,
                   uint64_t value_count,
                   uint64_t centroid_count) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < value_count) {
        routing_keys[index] += centroid_norms[index % centroid_count];
    }
}

__global__ void
select_nearest(float* routing_keys,
               uint64_t centroid_count,
               uint32_t select_count,
               int32_t* labels) {
    __shared__ float best_values[kThreads];
    __shared__ int32_t best_indices[kThreads];
    const uint64_t query = blockIdx.x;
    float* row = routing_keys + query * centroid_count;
    for (uint32_t rank = 0; rank < select_count; ++rank) {
        float local_value = FLT_MAX;
        int32_t local_index = -1;
        for (uint64_t centroid = threadIdx.x; centroid < centroid_count; centroid += blockDim.x) {
            const float value = row[centroid];
            if (value < local_value or
                (value == local_value and static_cast<int32_t>(centroid) < local_index)) {
                local_value = value;
                local_index = static_cast<int32_t>(centroid);
            }
        }
        best_values[threadIdx.x] = local_value;
        best_indices[threadIdx.x] = local_index;
        __syncthreads();
        for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_value = best_values[threadIdx.x + stride];
                const int32_t other_index = best_indices[threadIdx.x + stride];
                if (other_value < best_values[threadIdx.x] or
                    (other_value == best_values[threadIdx.x] and
                     other_index < best_indices[threadIdx.x])) {
                    best_values[threadIdx.x] = other_value;
                    best_indices[threadIdx.x] = other_index;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int32_t selected = best_indices[0];
            labels[query * select_count + rank] = selected;
            if (selected >= 0) {
                row[selected] = FLT_MAX;
            }
        }
        __syncthreads();
    }
}

}  // namespace

class CudaRabitqIndexImpl final : public CudaRabitqIndex {
public:
    std::vector<uint64_t> bucket_offsets;
    uint64_t bucket_count{0};
    uint64_t record_count{0};
    uint64_t dim{0};
    uint64_t filter_code_size{0};
    uint64_t supplement_code_size{0};
    uint64_t resident_bytes{0};
    uint32_t filter_bits{0};
    uint32_t supplement_bits{0};
    DeviceArray<uint8_t> filter_codes;
    DeviceArray<uint8_t> supplement_codes;
    DeviceArray<uint64_t> inner_ids;
    DeviceArray<float> filter_adds;
    DeviceArray<float> filter_scales;
    DeviceArray<float> filter_norm_codes;
    DeviceArray<float> full_adds;
    DeviceArray<float> full_norm_codes;
    DeviceArray<float> base_norms;
    DeviceArray<float> base_errors;
};

std::shared_ptr<CudaRabitqIndex>
CudaCreateRabitqIndex(const CudaRabitqBuildData& data, uint64_t budget_bytes) {
    if (data.bucket_offsets == nullptr or data.filter_codes == nullptr or
        data.supplement_codes == nullptr or data.inner_ids == nullptr or
        data.filter_adds == nullptr or data.filter_scales == nullptr or
        data.filter_norm_codes == nullptr or data.full_adds == nullptr or
        data.full_norm_codes == nullptr or data.base_norms == nullptr or
        data.base_errors == nullptr or data.bucket_count == 0 or data.record_count == 0 or
        data.dim == 0 or data.filter_code_size == 0 or data.supplement_code_size == 0 or
        data.filter_bits < 2 or data.filter_bits > 3 or data.supplement_bits == 0 or
        data.filter_bits + data.supplement_bits >= 31 or budget_bytes == 0) {
        return nullptr;
    }
    const uint64_t plane_bytes = (data.dim + 7) / 8;
    if (data.filter_code_size != plane_bytes * data.filter_bits or
        data.supplement_code_size < plane_bytes * data.supplement_bits or
        data.bucket_offsets[0] != 0 or
        data.bucket_offsets[data.bucket_count] != data.record_count) {
        return nullptr;
    }
    for (uint64_t bucket = 0; bucket < data.bucket_count; ++bucket) {
        if (data.bucket_offsets[bucket] > data.bucket_offsets[bucket + 1]) {
            return nullptr;
        }
    }
    const uint64_t resident_bytes =
        (data.bucket_count + 1) * sizeof(uint64_t) +
        data.record_count * (data.filter_code_size + data.supplement_code_size + sizeof(uint64_t) +
                             7 * sizeof(float));
    if (resident_bytes + kFixedReserveBytes > budget_bytes) {
        return nullptr;
    }
    auto index = std::make_shared<CudaRabitqIndexImpl>();
    index->bucket_offsets.assign(data.bucket_offsets, data.bucket_offsets + data.bucket_count + 1);
    index->bucket_count = data.bucket_count;
    index->record_count = data.record_count;
    index->dim = data.dim;
    index->filter_code_size = data.filter_code_size;
    index->supplement_code_size = data.supplement_code_size;
    index->resident_bytes = resident_bytes;
    index->filter_bits = data.filter_bits;
    index->supplement_bits = data.supplement_bits;
    if (not index->filter_codes.Alloc(data.record_count * data.filter_code_size) or
        not index->supplement_codes.Alloc(data.record_count * data.supplement_code_size) or
        not index->inner_ids.Alloc(data.record_count) or
        not index->filter_adds.Alloc(data.record_count) or
        not index->filter_scales.Alloc(data.record_count) or
        not index->filter_norm_codes.Alloc(data.record_count) or
        not index->full_adds.Alloc(data.record_count) or
        not index->full_norm_codes.Alloc(data.record_count) or
        not index->base_norms.Alloc(data.record_count) or
        not index->base_errors.Alloc(data.record_count)) {
        return nullptr;
    }
    auto copy = [](void* destination, const void* source, uint64_t bytes) {
        return cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (not copy(index->filter_codes.Get(),
                 data.filter_codes,
                 data.record_count * data.filter_code_size) or
        not copy(index->supplement_codes.Get(),
                 data.supplement_codes,
                 data.record_count * data.supplement_code_size) or
        not copy(index->inner_ids.Get(), data.inner_ids, data.record_count * sizeof(uint64_t)) or
        not copy(index->filter_adds.Get(), data.filter_adds, data.record_count * sizeof(float)) or
        not copy(
            index->filter_scales.Get(), data.filter_scales, data.record_count * sizeof(float)) or
        not copy(index->filter_norm_codes.Get(),
                 data.filter_norm_codes,
                 data.record_count * sizeof(float)) or
        not copy(index->full_adds.Get(), data.full_adds, data.record_count * sizeof(float)) or
        not copy(index->full_norm_codes.Get(),
                 data.full_norm_codes,
                 data.record_count * sizeof(float)) or
        not copy(index->base_norms.Get(), data.base_norms, data.record_count * sizeof(float)) or
        not copy(index->base_errors.Get(), data.base_errors, data.record_count * sizeof(float))) {
        return nullptr;
    }
    return index;
}

bool
CudaSearchRabitq(const std::shared_ptr<CudaRabitqIndex>& opaque_index,
                 const float* normalized_queries,
                 const float* query_norms,
                 const float* query_sums,
                 const int32_t* bucket_ids,
                 const float* query_bucket_norm_sqrs,
                 uint64_t query_count,
                 uint32_t nprobe,
                 uint32_t candidate_count,
                 uint32_t topk,
                 uint64_t* inner_ids,
                 float* distances,
                 uint64_t budget_bytes) {
    auto index = std::dynamic_pointer_cast<CudaRabitqIndexImpl>(opaque_index);
    if (index == nullptr or normalized_queries == nullptr or query_norms == nullptr or
        query_sums == nullptr or bucket_ids == nullptr or query_bucket_norm_sqrs == nullptr or
        inner_ids == nullptr or distances == nullptr or query_count == 0 or nprobe == 0 or
        candidate_count == 0 or candidate_count > kRabitqMaxCandidates or topk == 0 or
        topk > candidate_count or query_count > 65535 or budget_bytes == 0) {
        return false;
    }

    std::vector<RabitqSegment> segments;
    std::vector<uint64_t> query_segment_offsets(query_count + 1, 0);
    for (uint64_t query = 0; query < query_count; ++query) {
        for (uint32_t probe = 0; probe < nprobe; ++probe) {
            const int32_t bucket = bucket_ids[query * nprobe + probe];
            if (bucket < 0 or static_cast<uint64_t>(bucket) >= index->bucket_count) {
                return false;
            }
            const uint64_t begin = index->bucket_offsets[static_cast<uint64_t>(bucket)];
            const uint64_t end = index->bucket_offsets[static_cast<uint64_t>(bucket) + 1];
            for (uint64_t offset = begin; offset < end; offset += kThreads) {
                segments.push_back(RabitqSegment{
                    static_cast<uint32_t>(query),
                    probe,
                    offset,
                    static_cast<uint32_t>(std::min<uint64_t>(kThreads, end - offset))});
            }
        }
        query_segment_offsets[query + 1] = segments.size();
    }
    if (segments.empty()) {
        return false;
    }
    if (segments.size() > std::numeric_limits<uint32_t>::max() or
        segments.size() > std::numeric_limits<uint64_t>::max() / candidate_count) {
        return false;
    }

    const uint64_t group_count = (index->dim + 3) / 4;
    const uint64_t candidate_slots = segments.size() * candidate_count;
    const uint64_t selected_slots = query_count * candidate_count;
    const uint64_t working_bytes =
        query_count * index->dim * sizeof(float) + query_count * 2 * sizeof(float) +
        query_count * nprobe * (sizeof(int32_t) + sizeof(float)) +
        query_count * group_count * 16 * sizeof(float) + segments.size() * sizeof(RabitqSegment) +
        candidate_slots * (2 * sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t)) +
        selected_slots * (sizeof(float) + sizeof(uint64_t) + sizeof(uint32_t)) +
        (query_count + 1) * sizeof(uint64_t);
    if (index->resident_bytes + working_bytes + kFixedReserveBytes > budget_bytes) {
        return false;
    }

    DeviceArray<float> device_queries;
    DeviceArray<float> device_query_norms;
    DeviceArray<float> device_query_sums;
    DeviceArray<int32_t> device_bucket_ids;
    DeviceArray<float> device_query_bucket_norms;
    DeviceArray<float> device_lut;
    DeviceArray<RabitqSegment> device_segments;
    DeviceArray<uint64_t> device_query_segment_offsets;
    DeviceArray<float> candidate_distances;
    DeviceArray<float> candidate_filter_ips;
    DeviceArray<uint64_t> candidate_records;
    DeviceArray<uint32_t> candidate_probes;
    DeviceArray<float> selected_filter_ips;
    DeviceArray<uint64_t> selected_records;
    DeviceArray<uint32_t> selected_probes;
    DeviceArray<uint64_t> output_inner_ids;
    DeviceArray<float> output_distances;
    Stream stream;
    if (not stream.Create() or not device_queries.Alloc(query_count * index->dim) or
        not device_query_norms.Alloc(query_count) or not device_query_sums.Alloc(query_count) or
        not device_bucket_ids.Alloc(query_count * nprobe) or
        not device_query_bucket_norms.Alloc(query_count * nprobe) or
        not device_lut.Alloc(query_count * group_count * 16) or
        not device_segments.Alloc(segments.size()) or
        not device_query_segment_offsets.Alloc(query_count + 1) or
        not candidate_distances.Alloc(candidate_slots) or
        not candidate_filter_ips.Alloc(candidate_slots) or
        not candidate_records.Alloc(candidate_slots) or
        not candidate_probes.Alloc(candidate_slots) or
        not selected_filter_ips.Alloc(selected_slots) or
        not selected_records.Alloc(selected_slots) or not selected_probes.Alloc(selected_slots) or
        not output_inner_ids.Alloc(query_count * topk) or
        not output_distances.Alloc(query_count * topk)) {
        return false;
    }
    auto copy_async = [&](void* destination, const void* source, uint64_t bytes) {
        return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream.Get()) ==
               cudaSuccess;
    };
    if (not copy_async(
            device_queries.Get(), normalized_queries, query_count * index->dim * sizeof(float)) or
        not copy_async(device_query_norms.Get(), query_norms, query_count * sizeof(float)) or
        not copy_async(device_query_sums.Get(), query_sums, query_count * sizeof(float)) or
        not copy_async(
            device_bucket_ids.Get(), bucket_ids, query_count * nprobe * sizeof(int32_t)) or
        not copy_async(device_query_bucket_norms.Get(),
                       query_bucket_norm_sqrs,
                       query_count * nprobe * sizeof(float)) or
        not copy_async(
            device_segments.Get(), segments.data(), segments.size() * sizeof(RabitqSegment)) or
        not copy_async(device_query_segment_offsets.Get(),
                       query_segment_offsets.data(),
                       (query_count + 1) * sizeof(uint64_t))) {
        return false;
    }

    build_rabitq_lut<<<dim3(static_cast<uint32_t>(group_count), static_cast<uint32_t>(query_count)),
                       16,
                       0,
                       stream.Get()>>>(
        device_queries.Get(), index->dim, group_count, device_lut.Get());
    scan_rabitq_segments<<<static_cast<uint32_t>(segments.size()), kThreads, 0, stream.Get()>>>(
        device_segments.Get(),
        segments.size(),
        index->filter_codes.Get(),
        index->inner_ids.Get(),
        index->filter_adds.Get(),
        index->filter_scales.Get(),
        index->filter_norm_codes.Get(),
        device_query_norms.Get(),
        device_query_sums.Get(),
        device_query_bucket_norms.Get(),
        device_lut.Get(),
        group_count,
        index->filter_code_size,
        index->filter_bits,
        nprobe,
        candidate_count,
        candidate_distances.Get(),
        candidate_filter_ips.Get(),
        candidate_records.Get(),
        candidate_probes.Get());
    merge_rabitq_candidates<<<static_cast<uint32_t>(query_count), kThreads, 0, stream.Get()>>>(
        candidate_distances.Get(),
        candidate_filter_ips.Get(),
        candidate_records.Get(),
        candidate_probes.Get(),
        device_query_segment_offsets.Get(),
        candidate_count,
        selected_filter_ips.Get(),
        selected_records.Get(),
        selected_probes.Get());
    refine_rabitq_candidates<<<static_cast<uint32_t>(query_count),
                               kRabitqMaxCandidates,
                               0,
                               stream.Get()>>>(index->supplement_codes.Get(),
                                               index->inner_ids.Get(),
                                               index->filter_norm_codes.Get(),
                                               index->full_adds.Get(),
                                               index->full_norm_codes.Get(),
                                               index->base_norms.Get(),
                                               index->base_errors.Get(),
                                               device_queries.Get(),
                                               device_query_norms.Get(),
                                               device_query_sums.Get(),
                                               device_query_bucket_norms.Get(),
                                               selected_filter_ips.Get(),
                                               selected_records.Get(),
                                               selected_probes.Get(),
                                               index->dim,
                                               index->supplement_code_size,
                                               index->filter_bits,
                                               index->supplement_bits,
                                               nprobe,
                                               candidate_count,
                                               topk,
                                               output_inner_ids.Get(),
                                               output_distances.Get());
    if (cudaMemcpyAsync(inner_ids,
                        output_inner_ids.Get(),
                        query_count * topk * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost,
                        stream.Get()) != cudaSuccess or
        cudaMemcpyAsync(distances,
                        output_distances.Get(),
                        query_count * topk * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream.Get()) != cudaSuccess or
        cudaStreamSynchronize(stream.Get()) != cudaSuccess) {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

bool
CudaSelectNearestCentroids(const float* query,
                           uint64_t query_count,
                           const float* centroids,
                           uint64_t centroid_count,
                           int32_t dim,
                           uint32_t select_count,
                           int32_t* labels,
                           uint64_t budget_bytes,
                           uint64_t min_query_count) {
    if (query == nullptr or centroids == nullptr or labels == nullptr or query_count == 0 or
        centroid_count == 0 or dim <= 0 or select_count == 0 or select_count > centroid_count or
        select_count > 256 or query_count < min_query_count or not CudaAvailable()) {
        return false;
    }

    const uint64_t centroid_values = centroid_count * static_cast<uint64_t>(dim);
    const uint64_t resident_bytes =
        centroid_values * sizeof(float) + centroid_count * sizeof(float) + kFixedReserveBytes;
    const uint64_t bytes_per_query = (static_cast<uint64_t>(dim) + centroid_count) * sizeof(float) +
                                     static_cast<uint64_t>(select_count) * sizeof(int32_t);
    if (budget_bytes <= resident_bytes + bytes_per_query) {
        return false;
    }
    const uint64_t chunk_size =
        std::min<uint64_t>(query_count, (budget_bytes - resident_bytes) / bytes_per_query);
    if (chunk_size == 0 or
        centroid_count > static_cast<uint64_t>(std::numeric_limits<int>::max()) or
        chunk_size > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    DeviceArray<float> device_centroids;
    DeviceArray<float> device_centroid_norms;
    DeviceArray<float> device_queries;
    DeviceArray<float> device_keys;
    DeviceArray<int32_t> device_labels;
    Stream stream;
    BlasHandle blas;
    if (not stream.Create() or not blas.Create(stream.Get()) or
        not device_centroids.Alloc(centroid_values) or
        not device_centroid_norms.Alloc(centroid_count) or
        not device_queries.Alloc(chunk_size * dim) or
        not device_keys.Alloc(chunk_size * centroid_count) or
        not device_labels.Alloc(chunk_size * select_count)) {
        return false;
    }

    if (cudaMemcpyAsync(device_centroids.Get(),
                        centroids,
                        centroid_values * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream.Get()) != cudaSuccess) {
        return false;
    }
    row_sqr_norms_search<<<static_cast<int>(centroid_count), kThreads, 0, stream.Get()>>>(
        device_centroids.Get(),
        dim,
        static_cast<int32_t>(centroid_count),
        device_centroid_norms.Get());

    for (uint64_t offset = 0; offset < query_count; offset += chunk_size) {
        const uint64_t count = std::min<uint64_t>(chunk_size, query_count - offset);
        if (cudaMemcpyAsync(device_queries.Get(),
                            query + offset * dim,
                            count * dim * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream.Get()) != cudaSuccess) {
            return false;
        }
        const float alpha = -2.0F;
        const float beta = 0.0F;
        if (cublasSgemm(blas.Get(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        static_cast<int>(centroid_count),
                        static_cast<int>(count),
                        dim,
                        &alpha,
                        device_centroids.Get(),
                        dim,
                        device_queries.Get(),
                        dim,
                        &beta,
                        device_keys.Get(),
                        static_cast<int>(centroid_count)) != CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        const uint64_t value_count = count * centroid_count;
        const uint64_t grid_count = (value_count + kThreads - 1) / kThreads;
        if (grid_count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return false;
        }
        add_centroid_norms<<<static_cast<int>(grid_count), kThreads, 0, stream.Get()>>>(
            device_keys.Get(), device_centroid_norms.Get(), value_count, centroid_count);
        select_nearest<<<static_cast<int>(count), kThreads, 0, stream.Get()>>>(
            device_keys.Get(), centroid_count, select_count, device_labels.Get());
        if (cudaMemcpyAsync(labels + offset * select_count,
                            device_labels.Get(),
                            count * select_count * sizeof(int32_t),
                            cudaMemcpyDeviceToHost,
                            stream.Get()) != cudaSuccess or
            cudaStreamSynchronize(stream.Get()) != cudaSuccess) {
            return false;
        }
    }
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    return true;
}

}  // namespace vsag::gpu
