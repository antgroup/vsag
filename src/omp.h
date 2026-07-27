// Copyright 2026-present the vsag project
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

#if (defined(VSAG_ENABLE_OPENMP) || defined(_OPENMP)) && defined(__GNUC__)
#include_next <omp.h>
#else

#ifdef __cplusplus
extern "C" {
#endif

static inline int
omp_get_max_threads(void) {
    return 1;
}

static inline int
omp_get_num_threads(void) {
    return 1;
}

static inline int
omp_get_num_procs(void) {
    return 1;
}

static inline void
omp_set_num_threads(int num_threads) {
    (void)num_threads;
}

#ifdef __cplusplus
}
#endif

#endif
