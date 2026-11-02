// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "filter.h"

namespace vsag {

/** A request-owned lease. Evaluators observe labels, just like Filter::CheckValid. */
struct ConcurrentFilterLease {
    virtual ~ConcurrentFilterLease() = default;
    std::vector<FilterPtr> evaluators;
};

/** Optional capability implemented alongside Filter, without changing its vtable.
 * TryLease never waits for another search to release resources. Null means unavailable;
 * unsupported filters and construction errors must throw. Each evaluator is exclusive
 * to one logical execution slot, independently of the operating-system thread.
 */
class ConcurrentFilterProvider {
public:
    virtual ~ConcurrentFilterProvider() = default;
    virtual std::unique_ptr<ConcurrentFilterLease>
    TryLease(uint32_t maximum) const = 0;
};

}  // namespace vsag
