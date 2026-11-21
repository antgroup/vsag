
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

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace vsag {

// Hint the CPU during a spin wait. This is not a memory barrier or a scheduler yield.
inline void
CpuRelax() noexcept {
#if defined(__i386__) || defined(__x86_64__)
    _mm_pause();
#endif
}

}  // namespace vsag
