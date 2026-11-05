
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

#include "white_list_filter.h"

#include "common.h"

namespace vsag {
bool
WhiteListFilter::CheckValid(int64_t id) const {
    if (is_bitset_filter_) {
        int64_t bit_index = id & ROW_ID_MASK;
        return bitset_->Test(bit_index);
    }
    return fallback_func_(id);
}

void
WhiteListFilter::Update(const IdFilterFuncType& fallback_func) {
    this->fallback_func_ = fallback_func;
    this->bitset_ = nullptr;
    this->is_bitset_filter_ = false;
}

void
WhiteListFilter::Update(const Bitset* bitset) {
    this->fallback_func_ = nullptr;
    this->bitset_ = bitset;
    this->is_bitset_filter_ = true;
}

void
WhiteListFilter::TryToUpdate(Filter*& ptr, const IdFilterFuncType& fallback_func) {
    if (ptr == nullptr) {
        ptr = new WhiteListFilter(fallback_func);
    } else {
        auto* white_ptr = static_cast<WhiteListFilter*>(ptr);
        white_ptr->Update(fallback_func);
    }
}

void
WhiteListFilter::TryToUpdate(Filter*& ptr, const Bitset* bitset) {
    if (ptr == nullptr) {
        ptr = new WhiteListFilter(bitset);
    } else {
        auto* white_ptr = static_cast<WhiteListFilter*>(ptr);
        white_ptr->Update(bitset);
    }
}

}  // namespace vsag
