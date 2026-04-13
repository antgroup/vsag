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

/// @file combined_filter.h
/// @brief Combined filter that applies multiple filters in conjunction.

#pragma once

#include <vsag/filter.h>

#include <vector>
namespace vsag {

/// @brief Combined filter that applies multiple filters using AND logic.
///
/// This class combines multiple filters, where a result is valid only if
/// all filters in the combination pass.
class CombinedFilter : public Filter {
public:
    CombinedFilter() = default;

    ~CombinedFilter() override = default;

    /// @brief Appends a filter to the combination.
    /// @param[in] filter Filter to append (ignored if nullptr).
    void
    AppendFilter(const FilterPtr& filter) {
        if (filter == nullptr) {
            return;
        }
        this->filters_.emplace_back(filter);
    }

    /// @brief Checks if an ID is valid according to all filters.
    /// @param[in] inner_id Internal ID to check.
    /// @return True if ID passes all filters, false otherwise.
    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        for (const auto& filter : this->filters_) {
            if (not filter->CheckValid(inner_id)) {
                return false;
            }
        }
        return true;
    }

    /// @brief Gets the combined valid ratio.
    /// @return Product of all filter valid ratios.
    [[nodiscard]] float
    ValidRatio() const override {
        float valid_ratio = 1.0F;
        for (const auto& filter : this->filters_) {
            valid_ratio *= filter->ValidRatio();
        }
        return valid_ratio;
    }

    /// @brief Checks if the filter combination is empty.
    /// @return True if no filters are added, false otherwise.
    bool
    IsEmpty() const {
        return this->filters_.empty();
    }

private:
    std::vector<FilterPtr> filters_{};  /// Collection of filters
};

/// @brief Shared pointer type for CombinedFilter.
using CombinedFilterPtr = std::shared_ptr<CombinedFilter>;

}  // namespace vsag