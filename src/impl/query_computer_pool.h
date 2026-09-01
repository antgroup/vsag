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

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>

#include "datacell/flatten_interface.h"
#include "query_context.h"
#include "vsag_exception.h"

namespace vsag {

struct ComputerLease {
    FlattenInterfacePtr owner;
    ComputerInterfacePtr computer;
    const void* query{nullptr};

    void
    Validate(const FlattenInterfacePtr& expected_owner, const void* expected_query) const {
        if (owner == nullptr or computer == nullptr or owner.get() != expected_owner.get() or
            query != expected_query) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "invalid query computer lease");
        }
    }
};

inline void
RecordQueryComputerCreation(SearchStatistics* stats) {
    if (stats != nullptr) {
        stats->query_computer_count.fetch_add(1, std::memory_order_relaxed);
    }
}

class QueryComputerPool {
public:
    explicit QueryComputerPool(const void* query, SearchStatistics* stats = nullptr)
        : query_(query), stats_(stats) {
    }

    [[nodiscard]] ComputerLease
    Acquire(const FlattenInterfacePtr& cell, const void* query) {
        if (cell == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "cannot acquire a query computer for a null data cell");
        }
        if (query != query_) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "query computer pool used with a different query");
        }
        for (uint64_t i = 0; i < size_; ++i) {
            const auto& entry = entries_[i];
            if (entry.owner.get() == cell.get()) {
                entry.Validate(cell, query);
                return entry;
            }
        }
        if (size_ == kCapacity) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                "query computer pool capacity exceeded: HGraph supports at most base, precise, "
                "and raw cells");
        }

        auto computer = cell->FactoryComputer(query);
        RecordQueryComputerCreation(stats_);
        ComputerLease lease{cell, std::move(computer), query};
        lease.Validate(cell, query);
        entries_[size_] = lease;
        ++size_;
        return lease;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

private:
    static constexpr uint64_t kCapacity = 3;

    const void* query_{nullptr};
    SearchStatistics* stats_{nullptr};
    std::array<ComputerLease, kCapacity> entries_;
    uint64_t size_{0};
};

[[nodiscard]] inline ComputerLease
AcquireQueryComputer(const FlattenInterfacePtr& cell, const void* query, QueryContext* ctx) {
    if (ctx != nullptr and ctx->computer_pool != nullptr) {
        return ctx->computer_pool->Acquire(cell, query);
    }
    if (cell == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            "cannot acquire a query computer for a null data cell");
    }

    auto computer = cell->FactoryComputer(query);
    RecordQueryComputerCreation(ctx == nullptr ? nullptr : ctx->stats);
    ComputerLease lease{cell, std::move(computer), query};
    lease.Validate(cell, query);
    return lease;
}

}  // namespace vsag
