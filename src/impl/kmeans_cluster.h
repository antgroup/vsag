
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

#include "safe_thread_pool.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class KMeansCluster {
public:
    explicit KMeansCluster(int32_t dim,
                           Allocator* allocator,
                           SafeThreadPoolPtr thread_pool = nullptr);

    ~KMeansCluster();

    Vector<int>
    Run(uint32_t k, const float* datas, uint64_t count, int iter = 25);

public:
    float* k_centroids_{nullptr};

private:
    Allocator* const allocator_{nullptr};

    SafeThreadPoolPtr thread_pool_{nullptr};

    const int32_t dim_{0};
};

}  // namespace vsag
