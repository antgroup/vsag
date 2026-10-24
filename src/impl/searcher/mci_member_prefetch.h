// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace vsag {

struct MCIMemberPrefetch {
    uint32_t mode{4};
    uint32_t distance{96};
    uint32_t block{32};
    uint32_t locality{1};
};

inline MCIMemberPrefetch
GetMCIMemberPrefetch() {
    MCIMemberPrefetch result;
    auto get = [](const char* name, uint32_t fallback, uint32_t limit) {
        const char* value = std::getenv(name);
        return value == nullptr
                   ? fallback
                   : std::min(limit, static_cast<uint32_t>(std::strtoul(value, nullptr, 10)));
    };
    result.mode = get("VSAG_MCI_MEMBER_MODE", result.mode, 4);
    result.distance = std::max(1U, get("VSAG_MCI_MEMBER_DISTANCE", result.distance, 256));
    result.block = std::max(1U, get("VSAG_MCI_MEMBER_BLOCK", result.block, 64));
    result.locality = get("VSAG_MCI_MEMBER_LOCALITY", result.locality, 3);
    return result;
}

inline void
MCIPrefetchBitmap(const uint8_t* address, uint32_t locality) {
    switch (locality) {
        case 0:
            __builtin_prefetch(address, 0, 0);
            break;
        case 1:
            __builtin_prefetch(address, 0, 1);
            break;
        case 2:
            __builtin_prefetch(address, 0, 2);
            break;
        default:
            __builtin_prefetch(address, 0, 3);
            break;
    }
}

// Preserve member order and duplicate visits. Only the caller mutates query state or queues.
// Mode 1: rolling bitmap lookahead; 2: block prefetch; 3: block validity mask;
// mode 4 combines the rolling lookahead with a block validity mask.
template <typename Id, typename Visit, typename VisitValid>
inline void
VisitMCIMembers(const Id* members,
                uint64_t count,
                const uint8_t* bitmap,
                uint64_t total,
                const MCIMemberPrefetch& config,
                Visit&& visit,
                VisitValid&& visit_valid) {
    if (config.mode == 0 or bitmap == nullptr) {
        for (uint64_t i = 0; i < count; ++i) {
            visit(members[i]);
        }
        return;
    }
    const auto prefetch = [&](uint64_t i) {
        const auto id = members[i];
        if (id < total) {
            MCIPrefetchBitmap(bitmap + id, config.locality);
        }
    };
    if (config.mode == 1) {
        const auto initial = std::min<uint64_t>(config.distance, count);
        for (uint64_t i = 0; i < initial; ++i) {
            prefetch(i);
        }
        for (uint64_t i = 0; i < count; ++i) {
            if (i + config.distance < count) {
                prefetch(i + config.distance);
            }
            visit(members[i]);
        }
        return;
    }
    if (config.mode == 4) {
        for (uint64_t i = 0; i < std::min<uint64_t>(config.distance, count); ++i) {
            prefetch(i);
        }
    }
    const auto block = std::clamp(config.block, 1U, 64U);
    for (uint64_t begin = 0; begin < count; begin += block) {
        const auto size = std::min<uint64_t>(block, count - begin);
        if (config.mode == 2) {
            for (uint64_t j = 0; j < size; ++j) {
                prefetch(begin + j);
            }
            for (uint64_t j = 0; j < size; ++j) {
                visit(members[begin + j]);
            }
            continue;
        }
        uint64_t valid = 0;
        for (uint64_t j = 0; j < size; ++j) {
            if (config.mode == 4 and begin + j + config.distance < count) {
                prefetch(begin + j + config.distance);
            }
            const auto id = members[begin + j];
            if (id < total) {
                valid |= static_cast<uint64_t>(bitmap[id] != 0) << j;
            }
        }
        while (valid != 0) {
            const auto lane = static_cast<uint32_t>(__builtin_ctzll(valid));
            visit_valid(members[begin + lane]);
            valid &= valid - 1;
        }
    }
}

}  // namespace vsag
