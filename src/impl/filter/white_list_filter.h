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

/// @file white_list_filter.h
/// @brief White list filter for allowing specific IDs.

#pragma once

#include <functional>

#include "typing.h"
#include "vsag/bitset.h"
#include "vsag/filter.h"

namespace vsag {

/// @brief White list filter that allows only specified IDs.
///
/// This filter implements a whitelist pattern where only IDs in the
/// allowed set pass the filter check. It supports both callback function
/// and bitset-based implementations.
class WhiteListFilter : public Filter {
public:
    /// @brief Constructs a filter with a callback function.
    /// @param[in] fallback_func Function to check if an ID is allowed.
    explicit WhiteListFilter(const IdFilterFuncType& fallback_func)
        : fallback_func_(fallback_func), is_bitset_filter_(false), bitset_(nullptr){};

    /// @brief Constructs a filter with a bitset pointer.
    /// @param[in] bitset Shared pointer to bitset containing allowed IDs.
    explicit WhiteListFilter(const BitsetPtr& bitset)
        : bitset_(bitset.get()), is_bitset_filter_(true){};

    /// @brief Constructs a filter with a raw bitset pointer.
    /// @param[in] bitset Raw pointer to bitset containing allowed IDs.
    explicit WhiteListFilter(const Bitset* bitset) : bitset_(bitset), is_bitset_filter_(true){};

    /// @brief Checks if an ID is in the whitelist.
    /// @param[in] id ID to check.
    /// @return True if ID is allowed, false otherwise.
    bool
    CheckValid(int64_t id) const override;

    /// @brief Updates the filter with a new callback function.
    /// @param[in] fallback_func New callback function.
    void
    Update(const IdFilterFuncType& fallback_func);

    /// @brief Updates the filter with a new bitset.
    /// @param[in] bitset New bitset pointer.
    void
    Update(const Bitset* bitset);

    /// @brief Attempts to update an existing filter pointer with a callback.
    /// @param[in,out] ptr Reference to filter pointer to potentially update.
    /// @param[in] fallback_func Callback function for the new filter.
    static void
    TryToUpdate(Filter*& ptr, const IdFilterFuncType& fallback_func);

    /// @brief Attempts to update an existing filter pointer with a bitset.
    /// @param[in,out] ptr Reference to filter pointer to potentially update.
    /// @param[in] bitset Bitset for the new filter.
    static void
    TryToUpdate(Filter*& ptr, const Bitset* bitset);

private:
    IdFilterFuncType fallback_func_{nullptr};  /// Callback function for filtering
    const Bitset* bitset_;                     /// Bitset for filtering
    bool is_bitset_filter_;                    /// Whether using bitset mode
};

}  // namespace vsag