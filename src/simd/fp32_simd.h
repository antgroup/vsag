
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
#include <string>

namespace vsag {

namespace generic {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
}  // namespace generic

namespace sse {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
}  // namespace sse

namespace avx {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
}  // namespace avx

namespace avx2 {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
}  // namespace avx2

namespace avx512 {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* query,
                    uint64_t dim,
                    const float* codes1,
                    const float* codes2,
                    const float* codes3,
                    const float* codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* query,
                       uint64_t dim,
                       const float* codes1,
                       const float* codes2,
                       const float* codes3,
                       const float* codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
}  // namespace avx512

using FP32ComputeType = float (*)(const float* query, const float* codes, uint64_t dim);
extern FP32ComputeType FP32ComputeIP;
extern FP32ComputeType FP32ComputeL2Sqr;

using FP32ComputeBatch4Type = void (*)(const float* query,
                                       uint64_t dim,
                                       const float* codes1,
                                       const float* codes2,
                                       const float* codes3,
                                       const float* codes4,
                                       float& result1,
                                       float& result2,
                                       float& result3,
                                       float& result4);
extern FP32ComputeBatch4Type FP32ComputeIPBatch4;
extern FP32ComputeBatch4Type FP32ComputeL2SqrBatch4;

}  // namespace vsag
