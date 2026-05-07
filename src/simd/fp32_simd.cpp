
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

#include "fp32_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeIP, FP32ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeL2Sqr, FP32ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeIPBatch4, FP32ComputeBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(FP32ComputeL2SqrBatch4, FP32ComputeBatch4Type);
VSAG_DEFINE_SIMD_DISPATCH(FP32Sub, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Add, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Mul, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32Div, FP32ArithmeticType);
VSAG_DEFINE_SIMD_DISPATCH(FP32ReduceAdd, FP32ReduceType);

}  // namespace vsag
