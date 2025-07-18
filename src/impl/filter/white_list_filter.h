
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

#include <functional>

#include "typing.h"
#include "vsag/bitset.h"
#include "vsag/filter.h"

namespace vsag {

class WhiteListFilter : public Filter {
public:
    explicit WhiteListFilter(const IdFilterFuncType& fallback_func)
        : fallback_func_(fallback_func), is_bitset_filter_(false), bitset_(nullptr){};

    explicit WhiteListFilter(const BitsetPtr& bitset)
        : bitset_(bitset.get()), is_bitset_filter_(true){};

    explicit WhiteListFilter(const Bitset* bitset) : bitset_(bitset), is_bitset_filter_(true){};

    bool
    CheckValid(int64_t id) const override;

    void
    Update(const IdFilterFuncType& fallback_func);

    void
    Update(const Bitset* bitset);

    static void
    TryToUpdate(Filter*& ptr, const IdFilterFuncType& fallback_func);

    static void
    TryToUpdate(Filter*& ptr, const Bitset* bitset);

private:
    IdFilterFuncType fallback_func_{nullptr};
    const Bitset* bitset_;
    bool is_bitset_filter_;
};
}  // namespace vsag
