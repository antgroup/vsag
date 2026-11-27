// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag::lite::detail::rabitq {

constexpr uint32_t K_TOTAL_BITS = 8;
constexpr uint32_t K_FILTER_BITS = 3;
constexpr uint32_t K_SUPPLEMENT_BITS = 5;
constexpr uint32_t K_ROUNDS = 4;
constexpr uint32_t K_ENCODE_ROUNDS = 6;
constexpr float K_ERROR_RATE = 1.9F;

static_assert(K_TOTAL_BITS == K_FILTER_BITS + K_SUPPLEMENT_BITS,
              "K_TOTAL_BITS must equal K_FILTER_BITS + K_SUPPLEMENT_BITS");

}  // namespace vsag::lite::detail::rabitq
