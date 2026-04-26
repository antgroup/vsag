

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
    if (sv1.len_ > sv2.len_) {
        // [case 1]: sv1 is larger than sv2
        return false;
    }

    uint32_t i = 0;
    uint32_t j = 0;

    // sv1 and sv2 term IDs need to be sorted in ascending order
    while (i < sv1.len_ && j < sv2.len_) {
        if (sv1.ids_[i] == sv2.ids_[j]) {
            if (std::abs(sv1.vals_[i] - sv2.vals_[j]) > 1e-3) {
                // [case 3]: The term VALUE in the sv1 is not equal to that in the sv2
                return false;
            }
            i++;
            j++;
        } else if (sv1.ids_[i] > sv2.ids_[j]) {
            j++;
        } else {
            // sv1.ids_[i] < sv2.ids_[j]
            // [case 2]: The term ID in the sv1 does not exist in the sv2
            return false;
        }
    }
    return i == sv1.len_;
}

}  // namespace vsag
