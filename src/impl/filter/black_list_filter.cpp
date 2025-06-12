
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

#include "black_list_filter.h"

#include "common.h"

namespace vsag {

bool
BlackListFilter::CheckValid(int64_t id) const {
    if (is_bitset_filter_) {
        int64_t bit_index = id & ROW_ID_MASK;
        return not bitset_->Test(bit_index);
    }
    return not fallback_func_(id);
}
}  // namespace vsag
