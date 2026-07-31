

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

void
sort_sparse_vector(const SparseVector& sparse_vector,
                   Vector<std::pair<uint32_t, float>>& sorted_query) {
    sorted_query.reserve(sparse_vector.len_);

    for (auto i = 0; i < sparse_vector.len_; i++) {
        sorted_query.emplace_back(sparse_vector.ids_[i], sparse_vector.vals_[i]);
    }

    std::sort(sorted_query.begin(),
              sorted_query.end(),
              [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                  return a.second > b.second;
              });
}

bool
is_subset_of_sparse_vector(const SparseVector& sv1, const SparseVector& sv2) {
    if (sv1.len_ > sv2.len_) {
        // [case 1]: sv1 is larger than sv2
        return false;
    }

    std::unordered_map<uint32_t, float> sv2_map;
    for (auto i = 0; i < sv2.len_; i++) {
        sv2_map[sv2.ids_[i]] = sv2.vals_[i];
    }

    for (auto i = 0; i < sv1.len_; i++) {
        auto search = sv2_map.find(sv1.ids_[i]);
        if (search == sv2_map.end()) {
            // [case 2]: The term ID in the sv1 does not exist in the sv2
            return false;
        }
        float new_term_value = search->second;
        if (std::abs(sv1.vals_[i] - new_term_value) > 1e-3) {
            // [case 3]: The term VALUE in the sv1 is not equal to that in the sv2
            return false;
        }
    }
    return true;
}

}  // namespace vsag
