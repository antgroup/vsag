
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

#include "vsag_exception.h"

#define SAFE_CALL(stmt)                                                              \
    try {                                                                            \
        stmt;                                                                        \
        return {};                                                                   \
    } catch (const vsag::VsagException& e) {                                         \
        LOG_ERROR_AND_RETURNS(e.error_.type, e.error_.message);                      \
    } catch (const std::exception& e) {                                              \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what()); \
    } catch (...) {                                                                  \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknown error");            \
    }

#define CHECK_ARGUMENT(expr, message)                                        \
    do {                                                                     \
        if (not(expr)) {                                                     \
            throw vsag::VsagException(ErrorType::INVALID_ARGUMENT, message); \
        }                                                                    \
    } while (0);

#define ROW_ID_MASK 0xFFFFFFFFLL

constexpr static const int64_t INIT_CAPACITY = 10;
constexpr static const int64_t MAX_CAPACITY_EXTEND = 10000;
constexpr static const int64_t AMPLIFICATION_FACTOR = 10;

// sindi related
constexpr static const uint32_t ESTIMATE_DOC_TERM = 100;
constexpr static const uint32_t DEFAULT_TERM_ID_LIMIT = 1000000;
constexpr static const uint32_t DEFAULT_WINDOW_SIZE = 100000;
constexpr static const bool DEFAULT_USE_REORDER = false;
constexpr static const float DEFAULT_QUERY_PRUNE_RATIO = 0.0F;
constexpr static const float DEFAULT_DOC_PRUNE_RATIO = 0.0F;
constexpr static const float DEFAULT_TERM_PRUNE_RATIO = 0.0F;
constexpr static const uint32_t DEFAULT_N_CANDIDATE = 0;
