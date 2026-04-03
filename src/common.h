
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
#include <limits>

#include "vsag_exception.h"

#define SAFE_CALL(stmt)                                                                      \
    try {                                                                                    \
        stmt;                                                                                \
        return {};                                                                           \
    } catch (const vsag::VsagException& e) {                                                 \
        LOG_ERROR_AND_RETURNS(e.error_.type, e.error_.message);                              \
    } catch (const std::bad_alloc& e) {                                                      \
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY, "not enough memory: ", e.what()); \
    } catch (const std::exception& e) {                                                      \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what());         \
    } catch (...) {                                                                          \
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknown error");                    \
    }

#define CHECK_ARGUMENT(expr, message)                                        \
    do {                                                                     \
        if (not(expr)) {                                                     \
            throw vsag::VsagException(ErrorType::INVALID_ARGUMENT, message); \
        }                                                                    \
    } while (0);

#define ROW_ID_MASK 0xFFFFFFFFLL

static constexpr const int64_t INIT_CAPACITY = 10;
static constexpr const int64_t MAX_CAPACITY_EXTEND = 10000;
static constexpr const int64_t AMPLIFICATION_FACTOR = 100;
static constexpr const int64_t SPARSE_AMPLIFICATION_FACTOR = 500;
static constexpr const int64_t EXPANSION_NUM = 1000000;
static constexpr const int64_t DEFAULT_MAX_ELEMENT = 1;
static constexpr const int MINIMAL_M = 8;
static constexpr const int MAXIMAL_M = 128;
static constexpr const uint32_t GENERATE_SEARCH_K = 50;
static constexpr const uint32_t UPDATE_CHECK_SEARCH_K = 10;
static constexpr const uint32_t GENERATE_SEARCH_L = 400;
static constexpr const uint32_t UPDATE_CHECK_SEARCH_L = 100;
static constexpr const float GENERATE_OMEGA = 0.51;
static constexpr const uint32_t MAX_TRAIN_COUNT = 65536;

// sindi related
static constexpr const uint32_t ESTIMATE_DOC_TERM = 100;
static constexpr const uint32_t DEFAULT_TERM_ID_LIMIT = 1000000;
static constexpr const uint32_t DEFAULT_WINDOW_SIZE = 50000;
static constexpr const bool DEFAULT_USE_REORDER = false;
static constexpr const float DEFAULT_QUERY_PRUNE_RATIO = 0.0F;
static constexpr const float DEFAULT_DOC_PRUNE_RATIO = 0.0F;
static constexpr const float DEFAULT_TERM_PRUNE_RATIO = 0.0F;
static constexpr const uint32_t DEFAULT_N_CANDIDATE = 0;
static constexpr const uint32_t DEFAULT_AVG_DOC_TERM_LENGTH = 100;
static constexpr const uint32_t INVALID_ENTRY_POINT = std::numeric_limits<uint32_t>::max();
