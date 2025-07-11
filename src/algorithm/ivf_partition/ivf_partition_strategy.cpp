
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

#include "ivf_partition_strategy.h"

#include <cblas.h>

namespace vsag {

void
IVFPartitionStrategy::GetResidual(
    size_t n, const float* x, float* residuals, float* centroids, BucketIdType* assign) {
    // TODO: Directly implement c = a - b.
    memcpy(residuals, x, sizeof(float) * n * dim_);
    for (size_t i = 0; i < n; ++i) {
        BucketIdType bucket_id = assign[i];
        cblas_saxpy(
            static_cast<int>(dim_), -1.0, centroids + bucket_id * dim_, 1, residuals + i * dim_, 1);
    }
}

}  // namespace vsag
