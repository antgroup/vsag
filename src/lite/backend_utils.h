// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>

#include "vsag/lite/index.h"

namespace vsag::lite::detail {

inline auto
failure(ErrorType type, const char* message) {
    return tl::unexpected(Error(type, message));
}

inline tl::expected<void, Error>
validate(const float* data, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        return failure(ErrorType::DIMENSION_NOT_EQUAL, "dimension mismatch");
    }
    if (data == nullptr) {
        return failure(ErrorType::INVALID_ARGUMENT, "null vector");
    }
    for (uint64_t i = 0; i < actual; ++i) {
        if (not std::isfinite(data[i])) {
            return failure(ErrorType::INVALID_ARGUMENT, "non-finite vector");
        }
    }
    return {};
}

}  // namespace vsag::lite::detail
