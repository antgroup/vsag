
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
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ToFloat(const uint16_t bf16_value);
uint16_t
FloatToBF16(const float fp32_value);
}  // namespace generic

namespace sse {
float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
}  // namespace sse

namespace avx {
float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
}  // namespace avx

namespace avx2 {
float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
}  // namespace avx2

namespace avx512 {
float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
}  // namespace avx512

namespace neon {
float
BF16ComputeIP(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float
BF16ComputeL2Sqr(const uint8_t* query, const uint8_t* codes, uint64_t dim);
}  // namespace neon

using BF16ComputeType = float (*)(const uint8_t* query, const uint8_t* codes, uint64_t dim);
extern BF16ComputeType BF16ComputeIP;
extern BF16ComputeType BF16ComputeL2Sqr;

}  // namespace vsag
