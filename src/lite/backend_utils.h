// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>

#include "vsag/lite/index.h"

namespace vsag::lite::detail {

class FilterException final : public std::exception {};

// Isolate caller exceptions so graph search cannot mistake invalid_argument
// from a predicate for a query encoding/range failure. No wrapper allocation.
inline bool
invoke_filter(const IdFilter& filter, int64_t id) {
    try {
        return filter(id);
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::length_error&) {
        throw;
    } catch (...) {
        throw FilterException{};
    }
}

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
