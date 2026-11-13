// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>

namespace vsag {
struct GraphTailSearchParameters {
    uint32_t start_after{20};
    uint32_t batch_size{1};
    std::optional<float> relative_gap;

    bool
    Enabled() const {
        return batch_size > 1 or relative_gap.has_value();
    }
};

}  // namespace vsag
