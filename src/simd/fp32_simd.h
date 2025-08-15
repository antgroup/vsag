
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

#include "simd_marco.h"
namespace vsag {

namespace generic {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace generic

namespace sse {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace sse

namespace avx {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace avx

namespace avx2 {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace avx2

namespace avx512 {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace avx512

namespace neon {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace neon

namespace sve {
float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim);
void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4);
void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4);
void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Add(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim);
void
FP32Div(const float* x, const float* y, float* z, uint64_t dim);

float
FP32ReduceAdd(const float* x, uint64_t dim);
}  // namespace sve

using FP32ComputeType = float (*)(const float* RESTRICT query,
                                  const float* RESTRICT codes,
                                  uint64_t dim);
extern FP32ComputeType FP32ComputeIP;
extern FP32ComputeType FP32ComputeL2Sqr;

using FP32ComputeBatch4Type = void (*)(const float* RESTRICT query,
                                       uint64_t dim,
                                       const float* RESTRICT codes1,
                                       const float* RESTRICT codes2,
                                       const float* RESTRICT codes3,
                                       const float* RESTRICT codes4,
                                       float& result1,
                                       float& result2,
                                       float& result3,
                                       float& result4);
extern FP32ComputeBatch4Type FP32ComputeIPBatch4;
extern FP32ComputeBatch4Type FP32ComputeL2SqrBatch4;

using FP32ArithmeticType = void (*)(const float* x, const float* y, float* z, uint64_t dim);
extern FP32ArithmeticType FP32Sub;
extern FP32ArithmeticType FP32Add;
extern FP32ArithmeticType FP32Mul;
extern FP32ArithmeticType FP32Div;

using FP32ReduceType = float (*)(const float* x, uint64_t dim);
extern FP32ReduceType FP32ReduceAdd;
}  // namespace vsag
