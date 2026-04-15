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

/// @file inner_id_wrapper_filter.h
/// @brief Wrapper filter that converts internal IDs to labels for filtering.

#pragma once

#include "impl/label_table.h"
#include "vsag/filter.h"

namespace vsag {

/// @brief Wrapper filter that translates internal IDs to labels.
///
/// This filter wraps another filter and converts internal IDs to external
/// labels before passing them to the underlying filter implementation.
class InnerIdWrapperFilter : public Filter {
public:
    /// @brief Constructs a wrapper filter.
    /// @param[in] filter_impl Underlying filter implementation.
    /// @param[in] label_table Label table for ID-to-label conversion.
    InnerIdWrapperFilter(const FilterPtr filter_impl, const LabelTable& label_table)
        : filter_impl_(filter_impl), label_table_(label_table){};

    /// @brief Checks if an internal ID is valid.
    /// @param[in] inner_id Internal ID to check.
    /// @return True if the corresponding label passes the filter, false otherwise.
    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        return filter_impl_->CheckValid(label_table_.GetLabelById(inner_id));
    }

    /// @brief Gets the valid ratio from the underlying filter.
    /// @return Valid ratio value.
    [[nodiscard]] float
    ValidRatio() const override {
        return filter_impl_->ValidRatio();
    }

    /// @brief Gets the filter distribution from the underlying filter.
    /// @return Distribution type.
    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_impl_->FilterDistribution();
    }

    /// @brief Gets the valid IDs from the underlying filter.
    /// @param[out] valid_ids Pointer to array of valid IDs.
    /// @param[out] count Number of valid IDs.
    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        filter_impl_->GetValidIds(valid_ids, count);
    }

private:
    const FilterPtr filter_impl_;    /// Underlying filter implementation
    const LabelTable& label_table_;  /// Label table for conversion
};

}  // namespace vsag