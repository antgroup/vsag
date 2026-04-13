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

/// @file extrainfo_wrapper_filter.h
/// @brief Wrapper filter that provides extra information for filtering.

#pragma once

#include <functional>

#include "datacell/extra_info_interface.h"
#include "typing.h"
#include "vsag/bitset.h"
#include "vsag/filter.h"

namespace vsag {

/// @brief Wrapper filter that provides extra information access for filtering.
///
/// This filter wraps another filter and provides access to extra information
/// associated with internal IDs during filtering operations.
class ExtraInfoWrapperFilter : public Filter {
public:
    /// @brief Constructs a wrapper filter with extra information.
    /// @param[in] filter_impl Underlying filter implementation.
    /// @param[in] extra_infos Extra information interface pointer.
    ExtraInfoWrapperFilter(const FilterPtr filter_impl, const ExtraInfoInterfacePtr& extra_infos)
        : filter_impl_(filter_impl), extra_infos_(extra_infos){};

    /// @brief Checks if an internal ID is valid.
    /// @param[in] inner_id Internal ID to check.
    /// @return True if ID passes the filter, false otherwise.
    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override;

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

private:
    const FilterPtr filter_impl_;               /// Underlying filter implementation
    const ExtraInfoInterfacePtr& extra_infos_;  /// Extra information interface
};

}  // namespace vsag