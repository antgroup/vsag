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

/**
 * @file sparse_vector_transform.h
 * @brief Utility functions for sparse vector operations.
 */

#pragma once

#include <algorithm>

#include "typing.h"
#include "vsag/dataset.h"

namespace vsag {

/**
 * @brief Sort a sparse vector's indices and values by index in ascending order.
 *
 * @param sparse_vector The sparse vector to sort.
 * @param[out] sorted_query Output vector of (index, value) pairs sorted by index.
 */
void
sort_sparse_vector(const SparseVector& sparse_vector,
                   Vector<std::pair<uint32_t, float>>& sorted_query);

/**
 * @brief Check whether one sparse vector is a subset of another.
 *
 * @param sv1 The sparse vector to check as a subset.
 * @param sv2 The sparse vector to check against as the superset.
 * @return true if sv1 is a subset of sv2, false otherwise.
 */
bool
is_subset_of_sparse_vector(const SparseVector& sv1, const SparseVector& sv2);
}  // namespace vsag