
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

#include <atomic>
#include <cstdint>

#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {

class SearchStatistics;

struct QueryContext {
    Allocator* alloc = nullptr;
    SearchStatistics* stats = nullptr;
};

class SearchStatistics {
public:
    [[nodiscard]] std::string
    Dump() const {
        JsonType j;
        j["is_timeout"].SetBool(is_timeout.load(std::memory_order_relaxed));
        j["dist_cmp"].SetInt(dist_cmp.load(std::memory_order_relaxed));
        j["hops"].SetInt(hops.load(std::memory_order_relaxed));
        j["io_cnt"].SetInt(io_cnt.load(std::memory_order_relaxed));
        j["io_time_ms"].SetInt(io_time_ms.load(std::memory_order_relaxed));
        return j.Dump();
    }

public:
    std::atomic<bool> is_timeout{false};
    std::atomic<uint32_t> dist_cmp{0};
    std::atomic<uint32_t> hops{0};
    std::atomic<uint32_t> io_cnt{0};
    std::atomic<uint32_t> io_time_ms{0};
};

inline Allocator*
select_query_allocator(QueryContext* ctx, Allocator* index_allocator) {
    if (ctx != nullptr and ctx->alloc != nullptr) {
        // use the query specified memory allocator
        return ctx->alloc;
    }

    // use the index allocator
    return index_allocator;
}

inline Allocator*
select_query_allocator(Allocator* query_allocator, Allocator* index_allocator) {
    if (query_allocator != nullptr) {
        // use the query specified memory allocator
        return query_allocator;
    }

    // use the index allocator
    return index_allocator;
}

}  // namespace vsag
