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

#include <algorithm>

#include "allocator/memory_record_allocator.h"
#include "impl/allocator/default_allocator.h"
#include "index/hnsw.h"
#include "test_index.h"
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag/search_param.h"

namespace fixtures {

inline int64_t
Intersection(const int64_t* x, int64_t x_count, const int64_t* y, int64_t y_count) {
    std::unordered_set<int64_t> set_x(x, x + x_count);
    int64_t result = 0;

    for (int i = 0; i < y_count; ++i) {
        if (set_x.count(y[i])) {
            ++result;
        }
    }
    return result;
}

inline vsag::DatasetPtr
get_one_query(const vsag::DatasetPtr& queries, int i) {
    vsag::DatasetPtr query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(queries->GetDim())->Owner(false);

    if (queries->GetSparseVectors() != nullptr) {
        query->SparseVectors(queries->GetSparseVectors() + i);
    }

    if (queries->GetMultiVectors() != nullptr) {
        query->MultiVectors(queries->GetMultiVectors() + i);
        query->MultiVectorDim(queries->GetMultiVectorDim());
    }

    if (queries->GetFloat32Vectors() != nullptr) {
        query->Float32Vectors(queries->GetFloat32Vectors() + i * queries->GetDim());
    }

    if (queries->GetPaths() != nullptr) {
        query->Paths(queries->GetPaths() + i);
    }
    return query;
}

}  // namespace fixtures
