

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

struct SparseVectorAccessor {
    const SparseVector& ref;
    size_t
    size() const {
        return ref.len_;
    }
    uint32_t
    id(size_t i) const {
        return ref.ids_[i];
    }
    float
    val(size_t i) const {
        return ref.vals_[i];
    }
};

struct PairVectorAccessor {
    const Vector<std::pair<uint32_t, float>>& ref;
    size_t
    size() const {
        return ref.size();
    }
    uint32_t
    id(size_t i) const {
        return ref[i].first;
    }
    float
    val(size_t i) const {
        return ref[i].second;
    }
};

/**
 * check whether a is a subset of b
 * @param a
 * @param b
 * @return true if a is a subset of b
 */
template <typename T1, typename T2>
bool
is_subset_of_sparse_vector_impl(const T1& a, const T2& b) {
    if (a.size() > b.size())
        return false;

    uint32_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        uint32_t id1 = a.id(i), id2 = b.id(j);
        if (id1 == id2) {
            if (std::abs(a.val(i) - b.val(j)) > 1e-3)
                return false;
            i++;
            j++;
        } else if (id1 > id2) {
            j++;
        } else {
            return false;
        }
    }
    return i == a.size();
}

bool
is_subset_of_sparse_vector(const SparseVector& sv1, const SparseVector& sv2);

bool
is_subset_of_sparse_vector(const SparseVector& sv1, const Vector<std::pair<uint32_t, float>>& sv2);
}  // namespace vsag
