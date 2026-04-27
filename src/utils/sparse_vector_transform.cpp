

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

#include "sparse_vector_transform.h"

namespace vsag {

bool
is_subset_of_sparse_vector(const SparseVector& sv1, const SparseVector& sv2) {
    return is_subset_of_sparse_vector_impl(SparseVectorAccessor{sv1}, SparseVectorAccessor{sv2});
}

bool
is_subset_of_sparse_vector(const SparseVector& sv1, const Vector<std::pair<uint32_t, float>>& sv2) {
    return is_subset_of_sparse_vector_impl(SparseVectorAccessor{sv1}, PairVectorAccessor{sv2});
}

}  // namespace vsag
