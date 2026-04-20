
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
#include "sq4_simd.h"

#include "simd_dispatch.h"

namespace vsag {

VSAG_DEFINE_SIMD_DISPATCH(SQ4ComputeIP, SQ4ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4ComputeL2Sqr, SQ4ComputeType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4ComputeCodesIP, SQ4ComputeCodesType);
VSAG_DEFINE_SIMD_DISPATCH(SQ4ComputeCodesL2Sqr, SQ4ComputeCodesType);
}  // namespace vsag
