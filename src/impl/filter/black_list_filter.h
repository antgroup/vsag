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

/// @file black_list_filter.h
/// @brief Black list filter for blocking specific IDs.

#pragma once

#include <functional>

#include "typing.h"
#include "vsag/bitset.h"
#include "vsag/filter.h"

namespace vsag {

/// @brief Black list filter that blocks specified IDs.
///
/// This filter implements a blacklist pattern where IDs in the
/// blocked set fail the filter check. It supports both callback function
/// and bitset-based implementations.
class BlackListFilter : public Filter {
public:
    /// @brief Constructs a filter with a callback function.
    /// @param[in] fallback_func Function to check if an ID is blocked.
    explicit BlackListFilter(const IdFilterFuncType& fallback_func)
        : fallback_func_(fallback_func), is_bitset_filter_(false), bitset_(nullptr){};

    /// @brief Constructs a filter with a bitset pointer.
    /// @param[in] bitset Shared pointer to bitset containing blocked IDs.
    explicit BlackListFilter(const BitsetPtr& bitset)
        : bitset_(bitset.get()), is_bitset_filter_(true){};

    /// @brief Constructs a filter with a raw bitset pointer.
    /// @param[in] bitset Raw pointer to bitset containing blocked IDs.
    explicit BlackListFilter(const Bitset* bitset) : bitset_(bitset), is_bitset_filter_(true){};

    /// @brief Checks if an ID is not in the blacklist.
    /// @param[in] id ID to check.
    /// @return True if ID is not blocked, false otherwise.
    bool
    CheckValid(int64_t id) const override;

private:
    IdFilterFuncType fallback_func_{nullptr};  /// Callback function for filtering
    const Bitset* bitset_{nullptr};            /// Bitset for filtering
    const bool is_bitset_filter_;              /// Whether using bitset mode
};

}  // namespace vsag