
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

#include <cstdint>

#include "simd_marco.h"

namespace vsag {

#define DECLARE_PQFS_FUNCTIONS(ns)                                                \
    namespace ns {                                                                \
    void                                                                          \
    PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,                      \
                       const uint8_t* RESTRICT codes,                             \
                       uint64_t pq_dim,                                           \
                       int32_t* RESTRICT result);                                 \
    void                                                                          \
    PQFastScanLookUp32HighAcc(const uint8_t* RESTRICT low_lookup_table,           \
                              const uint8_t* RESTRICT high_lookup_table,          \
                              const uint8_t* RESTRICT codes,                      \
                              uint64_t pq_dim,                                    \
                              int32_t* RESTRICT result);                          \
    void                                                                          \
    PQFastScanLookUp32HighAccOverwrite(const uint8_t* RESTRICT low_lookup_table,  \
                                       const uint8_t* RESTRICT high_lookup_table, \
                                       const uint8_t* RESTRICT codes,             \
                                       uint64_t pq_dim,                           \
                                       int32_t* RESTRICT result);                 \
    uint32_t                                                                      \
    FP32LessThan32Mask(const float* values, float limit);                         \
    uint32_t                                                                      \
    RaBitQFastScan32ResidualPostprocess(const int32_t* accumulators,              \
                                        uint32_t filter_bits,                     \
                                        float delta,                              \
                                        float sum_vl,                             \
                                        float query_sum,                          \
                                        float query_norm,                         \
                                        float query_bucket_norm_sqr,              \
                                        float inv_sqrt_d,                         \
                                        const float* f_add,                       \
                                        const float* f_scale,                     \
                                        const float* filter_norm_codes,           \
                                        uint32_t valid_size,                      \
                                        float* dists,                             \
                                        float* filter_inner_products);            \
    }  // namespace ns

DECLARE_PQFS_FUNCTIONS(generic)
DECLARE_PQFS_FUNCTIONS(sse)
DECLARE_PQFS_FUNCTIONS(avx)
DECLARE_PQFS_FUNCTIONS(avx2)
DECLARE_PQFS_FUNCTIONS(avx512)
DECLARE_PQFS_FUNCTIONS(neon)
DECLARE_PQFS_FUNCTIONS(sve)

#undef DECLARE_PQFS_FUNCTIONS

using PQFastScanLookUp32Type = void (*)(const uint8_t* RESTRICT lookup_table,
                                        const uint8_t* RESTRICT codes,
                                        uint64_t pq_dim,
                                        int32_t* RESTRICT result);
extern PQFastScanLookUp32Type PQFastScanLookUp32;
using PQFastScanLookUp32HighAccType = void (*)(const uint8_t* RESTRICT low_lookup_table,
                                               const uint8_t* RESTRICT high_lookup_table,
                                               const uint8_t* RESTRICT codes,
                                               uint64_t pq_dim,
                                               int32_t* RESTRICT result);
extern PQFastScanLookUp32HighAccType PQFastScanLookUp32HighAcc;
using PQFastScanLookUp32HighAccOverwriteType = PQFastScanLookUp32HighAccType;
extern PQFastScanLookUp32HighAccOverwriteType PQFastScanLookUp32HighAccOverwrite;
using FP32LessThan32MaskType = uint32_t (*)(const float* values, float limit);
extern FP32LessThan32MaskType FP32LessThan32Mask;
using RaBitQFastScan32ResidualPostprocessType = uint32_t (*)(const int32_t* accumulators,
                                                             uint32_t filter_bits,
                                                             float delta,
                                                             float sum_vl,
                                                             float query_sum,
                                                             float query_norm,
                                                             float query_bucket_norm_sqr,
                                                             float inv_sqrt_d,
                                                             const float* f_add,
                                                             const float* f_scale,
                                                             const float* filter_norm_codes,
                                                             uint32_t valid_size,
                                                             float* dists,
                                                             float* filter_inner_products);
extern RaBitQFastScan32ResidualPostprocessType RaBitQFastScan32ResidualPostprocess;

}  // namespace vsag
