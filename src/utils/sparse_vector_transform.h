

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

#include "typing.h"
#include "vsag/dataset.h"

namespace vsag {

enum class SortTarget { ID, VALUE };
enum class SortOrder { ASCENDING, DESCENDING };

template <SortTarget Target = SortTarget::VALUE, SortOrder Order = SortOrder::DESCENDING>
void
sort_sparse_vector(const SparseVector& sparse_vector,
                   Vector<std::pair<uint32_t, float>>& sorted_query) {
    sorted_query.clear();
    sorted_query.reserve(sparse_vector.len_);

    for (auto i = 0; i < sparse_vector.len_; i++) {
        sorted_query.emplace_back(sparse_vector.ids_[i], sparse_vector.vals_[i]);
    }

    std::sort(sorted_query.begin(),
              sorted_query.end(),
              [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                  auto get_key = [](const std::pair<uint32_t, float>& p) {
                      if constexpr (Target == SortTarget::ID) {
                          return p.first;
                      } else {
                          return p.second;
                      }
                  };

                  if constexpr (Order == SortOrder::ASCENDING) {
                      return get_key(a) < get_key(b);
                  } else {
                      return get_key(a) > get_key(b);
                  }
              });
}

/**
 * check whether sv1 is a subset of sv2
 * @param sv1
 * @param sv2
 * @return true iff sv1 is a subset of sv2
 */
bool
is_subset_of_sparse_vector(const SparseVector& sv1, const SparseVector& sv2);
}  // namespace vsag
