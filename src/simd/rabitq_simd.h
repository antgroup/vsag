
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

namespace vsag {

namespace avx512vpopcntdq {

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

}  // namespace avx512vpopcntdq

namespace avx512 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, uint64_t len);

void
VecRescale(float* data, uint64_t dim, float val);

void
FHTRotate(float* data, uint64_t dim_);

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace avx512

namespace avx2 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

void
FHTRotate(float* data, uint64_t dim_);

void
RotateOp(float* data, int idx, int dim_, int step);

void
VecRescale(float* data, uint64_t dim, float val);

void
KacsWalk(float* data, uint64_t len);
}  // namespace avx2

namespace avx {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

void
VecRescale(float* data, uint64_t dim, float val);

void
FHTRotate(float* data, uint64_t dim_);

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);

void
KacsWalk(float* data, uint64_t len);
}  // namespace avx

namespace sse {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

void
FHTRotate(float* data, uint64_t dim_);

void
RotateOp(float* data, int idx, int dim_, int step);

void
VecRescale(float* data, uint64_t dim, float val);

void
KacsWalk(float* data, uint64_t len);
}  // namespace sse

namespace generic {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, uint64_t len);

void
VecRescale(float* data, uint64_t dim, float val);

void
FHTRotate(float* data, uint64_t dim_);

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace generic

namespace neon {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, uint64_t len);

void
VecRescale(float* data, uint64_t dim, float val);

void
FHTRotate(float* data, uint64_t dim_);

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace neon

namespace sve {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
RaBitQFloatBinaryIPBatch4(const float* vector,
                          const uint8_t* bits1,
                          const uint8_t* bits2,
                          const uint8_t* bits3,
                          const uint8_t* bits4,
                          uint64_t dim,
                          float inv_sqrt_d,
                          float* results);

float
RaBitQFloatSplitCodeIP(const float* vector,
                       const uint8_t* one_bit_code,
                       const uint8_t* supplement_code,
                       uint64_t dim,
                       uint32_t supplement_bits);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, uint64_t len);

void
VecRescale(float* data, uint64_t dim, float val);

void
FHTRotate(float* data, uint64_t dim_);

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace sve

using RaBitQFloatBinaryType = float (*)(const float* vector,
                                        const uint8_t* bits,
                                        uint64_t dim,
                                        float inv_sqrt_d);

using RaBitQFloatBinaryBatch4Type = void (*)(const float* vector,
                                             const uint8_t* bits1,
                                             const uint8_t* bits2,
                                             const uint8_t* bits3,
                                             const uint8_t* bits4,
                                             uint64_t dim,
                                             float inv_sqrt_d,
                                             float* results);

using RaBitQFloatSplitCodeType = float (*)(const float* vector,
                                           const uint8_t* one_bit_code,
                                           const uint8_t* supplement_code,
                                           uint64_t dim,
                                           uint32_t supplement_bits);

using RaBitQSQ4UBinaryType = uint32_t (*)(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

using RaBitQFloatSQType = float (*)(const float* vector, const uint8_t* codes, uint64_t dim);

using FHTRotateType = void (*)(float* data, uint64_t dim_);

using KacsWalkType = void (*)(float* data, uint64_t len);

using VecRescaleType = void (*)(float* data, uint64_t dim, float val);

using FlipSignType = void (*)(const uint8_t* flip, float* data, uint64_t dim);

using RotateOpType = void (*)(float* data, int idx, int dim_, int step);
extern RaBitQFloatBinaryType RaBitQFloatBinaryIP;
extern RaBitQFloatBinaryBatch4Type RaBitQFloatBinaryIPBatch4;
extern RaBitQFloatSplitCodeType RaBitQFloatSplitCodeIP;
extern RaBitQFloatSQType RaBitQFloatSQIP;
extern RaBitQSQ4UBinaryType RaBitQSQ4UBinaryIP;
extern FHTRotateType FHTRotate;
extern KacsWalkType KacsWalk;
extern VecRescaleType VecRescale;
extern FlipSignType FlipSign;
extern RotateOpType RotateOp;
}  // namespace vsag
