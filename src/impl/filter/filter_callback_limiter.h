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

#include <cstdint>
#include <memory>
#include <utility>

#include "vsag/filter.h"

namespace vsag {

class FilterCallbackLimiter : public Filter {
public:
    FilterCallbackLimiter(FilterPtr filter, std::shared_ptr<uint64_t> remaining)
        : filter_(std::move(filter)), remaining_(std::move(remaining)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (*remaining_ == 0) {
            return false;
        }
        const bool valid = filter_->CheckValid(id);
        --(*remaining_);
        return valid;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_->FilterDistribution();
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        filter_->GetValidIds(valid_ids, count);
    }

private:
    FilterPtr filter_;
    std::shared_ptr<uint64_t> remaining_;
};

inline FilterPtr
create_filter_callback_limiter(const FilterPtr& filter,
                               const std::shared_ptr<uint64_t>& remaining) {
    if (filter == nullptr or remaining == nullptr) {
        return filter;
    }
    return std::make_shared<FilterCallbackLimiter>(filter, remaining);
}

}  // namespace vsag
