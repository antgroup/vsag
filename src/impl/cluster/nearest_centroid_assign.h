
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

#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"

namespace vsag {

class Allocator;

// Assigns every query vector to its nearest centroid under the squared L2
// metric (`labels[i] = argmin_c ||queries[i] - centroids[c]||^2`) and returns
// the mean of the selected minimal squared distances.
//
// The batched GEMM formulation relies on the identity
//     ||q - c||^2 = ||q||^2 + ||c||^2 - 2 <c, q>
// and first tries the AMX BF16 GEMM fast path when the runtime CPU supports it
// and the problem shape is large enough to amortize tile setup and packing
// overhead; otherwise it falls back to BLAS SGEMM. Both paths select the same
// labels.
//
// `labels` must provide room for `query_count` entries; entries that are
// already equal to the selected label are left untouched.
double
NearestCentroidAssign(const float* centroids,
                      uint64_t centroid_count,
                      const float* queries,
                      uint64_t query_count,
                      uint64_t dim,
                      const SafeThreadPoolPtr& thread_pool,
                      Allocator* allocator,
                      int32_t* labels);

}  // namespace vsag
