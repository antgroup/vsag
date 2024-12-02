
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
namespace generic {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace generic

namespace sse {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace sse

namespace avx2 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx2

namespace avx512 {
void
DivScalar(const float* from, float* to, uint64_t dim, float scalar);

float
Normalize(const float* from, float* to, uint64_t dim);
}  // namespace avx512

using NormalizeType = float (*)(const float* from, float* to, uint64_t dim);
extern NormalizeType Normalize;
using DivScalarType = void (*)(const float* from, float* to, uint64_t dim, float scalar);
extern DivScalarType DivScalar;

}  // namespace vsag
