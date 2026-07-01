
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

#include <fmt/format.h>

#include <functional>
#include <string>

#include "common.h"

namespace vsag::param {

// Validator signature: inspects the parsed scalar value and throws a
// VsagException(INVALID_ARGUMENT) on failure. Returning normally signals
// "accepted". Validators are stored inside schema field descriptors and
// invoked from FromJson() after the JSON value is converted to the field
// type.
template <typename T>
using ValidatorFn = std::function<void(const std::string& key, const T& value)>;

// Inclusive numeric range validator. The template parameter T must match the
// field type exactly to avoid silent overflow/underflow/truncation from
// cross-type comparisons. Use explicit type at construction site:
//
//     VSAG_PARAM(uint64_t, dim, DIM_KEY, 128, Range<uint64_t>(1, 1000000))
//
// The constructor accepts arbitrary arithmetic literals and static_casts them
// to T once during construction, after which all comparisons are same-type.
template <typename T>
struct Range {
    T min_value;
    T max_value;

    Range(T min_v, T max_v) : min_value(min_v), max_value(max_v) {
    }

    template <typename A,
              typename B,
              std::enable_if_t<std::is_arithmetic_v<A> && std::is_arithmetic_v<B>, int> = 0>
    Range(A min_v, B max_v) : min_value(static_cast<T>(min_v)), max_value(static_cast<T>(max_v)) {
    }

    void
    operator()(const std::string& key, const T& value) const {
        CHECK_ARGUMENT(
            value >= min_value && value <= max_value,
            fmt::format(
                "parameter '{}' value {} out of range [{}, {}]", key, value, min_value, max_value));
    }
};

}  // namespace vsag::param
