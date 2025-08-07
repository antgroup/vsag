
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

#include "simd_status.h"

namespace vsag {

namespace avx512vpopcntdq {

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

}  // namespace avx512vpopcntdq

namespace avx512 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, std::size_t len);

void
VecRescale(float* data, std::size_t dim, float val);

void
FHTRotate(float* data, std::size_t dim_);

void
FlipSign(const uint8_t* flip, float* data, std::size_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace avx512

namespace avx2 {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
FHTRotate(float* data, std::size_t dim_);

void
RotateOp(float* data, int idx, int dim_, int step);

void
VecRescale(float* data, std::size_t dim, float val);

void
KacsWalk(float* data, std::size_t len);
}  // namespace avx2

namespace avx {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
VecRescale(float* data, std::size_t dim, float val);

void
FHTRotate(float* data, std::size_t dim_);

void
FlipSign(const uint8_t* flip, float* data, std::size_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);

void
KacsWalk(float* data, std::size_t len);
}  // namespace avx

namespace sse {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

void
FHTRotate(float* data, std::size_t dim_);

void
RotateOp(float* data, int idx, int dim_, int step);

void
VecRescale(float* data, std::size_t dim, float val);

void
KacsWalk(float* data, std::size_t len);
}  // namespace sse

namespace generic {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, std::size_t len);

void
VecRescale(float* data, std::size_t dim, float val);

void
FHTRotate(float* data, std::size_t dim_);

void
FlipSign(const uint8_t* flip, float* data, std::size_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace generic

namespace neon {
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

void
KacsWalk(float* data, std::size_t len);

void
VecRescale(float* data, std::size_t dim, float val);

void
FHTRotate(float* data, std::size_t dim_);

void
FlipSign(const uint8_t* flip, float* data, std::size_t dim);

void
RotateOp(float* data, int idx, int dim_, int step);
}  // namespace neon

using RaBitQFloatBinaryType = float (*)(const float* vector,
                                        const uint8_t* bits,
                                        uint64_t dim,
                                        float inv_sqrt_d);

using RaBitQSQ4UBinaryType = uint32_t (*)(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

using FHTRotateType = void (*)(float* data, size_t dim_);

using KacsWalkType = void (*)(float* data, size_t len);

using VecRescaleType = void (*)(float* data, size_t dim, float val);

using FlipSignType = void (*)(const uint8_t* flip, float* data, size_t dim);

using RotateOpType = void (*)(float* data, int idx, int dim_, int step);
extern RaBitQFloatBinaryType RaBitQFloatBinaryIP;
extern RaBitQSQ4UBinaryType RaBitQSQ4UBinaryIP;
extern FHTRotateType FHTRotate;
extern KacsWalkType KacsWalk;
extern VecRescaleType VecRescale;
extern FlipSignType FlipSign;
extern RotateOpType RotateOp;
}  // namespace vsag
