
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

#include "algorithm_interface.h"

namespace hnswlib {

template <typename dist_t>
std::vector<std::pair<dist_t, LabelType>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data,
                                                 size_t k,
                                                 size_t ef,
                                                 const vsag::FilterPtr& is_id_allowed) const {
    std::vector<std::pair<dist_t, LabelType>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, ef, is_id_allowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib
