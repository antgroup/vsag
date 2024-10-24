
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

#include "bitset_impl.h"
#include "common.h"
#include "typing.h"

namespace vsag {
class BaseFilterFunctor {
public:
    virtual bool
    operator()(LabelType id) {
        return true;
    }
};

class BitsetOrCallbackFilter : public BaseFilterFunctor {
public:
    BitsetOrCallbackFilter(const std::function<bool(int64_t)>& func)
        : func_(func), is_bitset_filter_(false){};

    BitsetOrCallbackFilter(const BitsetPtr& bitset) : bitset_(bitset), is_bitset_filter_(true){};

    bool
    operator()(LabelType id) override {
        if (is_bitset_filter_) {
            int64_t bit_index = id & ROW_ID_MASK;
            return not bitset_->Test(bit_index);
        } else {
            return not func_(id);
        }
    }

private:
    std::function<bool(int64_t)> func_{nullptr};
    const BitsetPtr bitset_{nullptr};
    const bool is_bitset_filter_{false};
};

}  // namespace vsag
