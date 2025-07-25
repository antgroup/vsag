
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

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "basic_func.h"
#include "bf16_simd.h"
#include "bit_simd.h"
#include "fp16_simd.h"
#include "fp32_simd.h"
#include "normalize.h"
#include "pqfs_simd.h"
#include "rabitq_simd.h"
#include "simd_marco.h"
#include "simd_status.h"
#include "sq4_simd.h"
#include "sq4_uniform_simd.h"
#include "sq8_simd.h"
#include "sq8_uniform_simd.h"

namespace vsag {

SimdStatus
setup_simd();

}  // namespace vsag
