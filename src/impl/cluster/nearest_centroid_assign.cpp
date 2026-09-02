
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

#include "nearest_centroid_assign.h"

#include <omp.h>

#include <algorithm>
#include <future>
#include <limits>
#include <mutex>

#include "impl/blas/blas_function.h"
#include "simd/amx_bf16_matmul.h"
#include "simd/fp32_simd.h"
#include "simd/simd_status.h"
#include "utils/byte_buffer.h"
#include "vsag_exception.h"

namespace vsag {

namespace {
constexpr uint64_t QUERY_BS = 65536ULL;
constexpr uint64_t ASSIGN_BS = 1024;
}  // namespace

double
NearestCentroidAssign(const float* centroids,
                      uint64_t centroid_count,
                      const float* queries,
                      uint64_t query_count,
                      uint64_t dim,
                      const SafeThreadPoolPtr& thread_pool,
                      Allocator* allocator,
                      int32_t* labels) {
    double error = 0.0;
    std::mutex error_mutex;
    if (centroids == nullptr || queries == nullptr || labels == nullptr || allocator == nullptr ||
        thread_pool == nullptr || centroid_count == 0 || query_count == 0 || dim == 0 ||
        centroid_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
        dim > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "invalid nearest-centroid assignment input");
    }
    const auto& thread_pool_ref = thread_pool;
    auto bs = ASSIGN_BS;
    std::vector<std::future<void>> futures;

    auto wait_futures_and_clear = [&]() {
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
    };

    const auto distance_batch_size = std::min(QUERY_BS, query_count);
    ByteBuffer y_sqr_buffer(centroid_count * sizeof(float), allocator);
    ByteBuffer distances_buffer(centroid_count * distance_batch_size * sizeof(float), allocator);
    auto* y_sqr = reinterpret_cast<float*>(y_sqr_buffer.data);
    auto* distances = reinterpret_cast<float*>(distances_buffer.data);

    auto compute_ip_func = [&](uint64_t start, uint64_t end) -> void {
        for (uint64_t i = start; i < end; ++i) {
            y_sqr[i] = FP32ComputeIP(centroids + i * dim, centroids + i * dim, dim);
        }
    };
    for (uint64_t i = 0; i < centroid_count; i += bs) {
        futures.emplace_back(
            thread_pool_ref->GeneralEnqueue(compute_ip_func, i, std::min(i + bs, centroid_count)));
    }
    wait_futures_and_clear();

    for (uint64_t i = 0; i < query_count; i += QUERY_BS) {
        auto end = std::min(i + QUERY_BS, query_count);
        auto cur_query_count = end - i;
        auto* cur_label = labels + i;

        // Try the AMX BF16 GEMM fast path first.  It returns false if
        // AMX-BF16 isn't available at runtime; in that case (or when the
        // shape is too small to amortize tile-config / packing overhead)
        // fall back to the BLAS SGEMM path.
        //
        // Math equivalence:
        //   SGEMM(ColMajor, Trans, NoTrans, M=centroid_count, N=cur_query_count, K=dim,
        //         alpha=-2, A=centroids (lda=dim), B=queries+i*dim (ldb=dim),
        //         beta=0, C=distances (ldc=centroid_count))
        //   produces  distances[m + n*centroid_count] = -2 * < centroid_m, query_{i+n} >
        // The AMX kernel takes the same inputs interpreted as row-major
        // (centroid_count x dim) and (cur_query_count x dim) and writes the same
        // column-major output.
        constexpr uint64_t amx_bf16_min_dim = 32;
        constexpr uint64_t amx_bf16_min_m = 16;
        constexpr uint64_t amx_bf16_min_n = 16;
        bool used_amx = false;
        if (dim >= amx_bf16_min_dim && centroid_count >= amx_bf16_min_m &&
            cur_query_count >= amx_bf16_min_n && SimdStatus::SupportAMXBF16()) {
            used_amx = amx::SgemmBF16IPColMajorOut(static_cast<int64_t>(centroid_count),
                                                   static_cast<int64_t>(cur_query_count),
                                                   static_cast<int64_t>(dim),
                                                   -2.0F,
                                                   centroids,
                                                   queries + i * dim,
                                                   distances,
                                                   static_cast<int64_t>(centroid_count));
        }
        if (!used_amx) {
            BlasFunction::Sgemm(BlasFunction::ColMajor,
                                BlasFunction::Trans,
                                BlasFunction::NoTrans,
                                static_cast<int32_t>(centroid_count),
                                static_cast<int32_t>(cur_query_count),
                                static_cast<int32_t>(dim),
                                -2.0F,
                                centroids,
                                static_cast<int32_t>(dim),
                                queries + i * dim,
                                static_cast<int32_t>(dim),
                                0.0F,
                                distances,
                                static_cast<int32_t>(centroid_count));
        }

        auto batch_offset = i;
        auto assign_labels_func = [&, batch_offset](uint64_t start, uint64_t end) -> void {
            omp_set_num_threads(1);
            double thread_local_error = 0.0;
            for (uint64_t j = start; j < end; ++j) {
                BlasFunction::Saxpy(static_cast<int32_t>(centroid_count),
                                    1.0,
                                    y_sqr,
                                    1,
                                    distances + j * centroid_count,
                                    1);
                auto* min_elem = std::min_element(distances + j * centroid_count,
                                                  distances + j * centroid_count + centroid_count);
                auto x_sqr = FP32ComputeIP(
                    queries + (batch_offset + j) * dim, queries + (batch_offset + j) * dim, dim);
                auto min_index = std::distance(distances + j * centroid_count, min_elem);
                thread_local_error += static_cast<double>(*min_elem + x_sqr);
                if (min_index != cur_label[j]) {
                    cur_label[j] = static_cast<int32_t>(min_index);
                }
            }
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                error += thread_local_error;
            }
        };
        for (uint64_t j = 0; j < cur_query_count; j += bs) {
            futures.emplace_back(thread_pool_ref->GeneralEnqueue(
                assign_labels_func, j, std::min(j + bs, cur_query_count)));
        }
        wait_futures_and_clear();
    }
    return error / static_cast<double>(query_count);
}

}  // namespace vsag
