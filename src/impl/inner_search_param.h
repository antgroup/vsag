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

/// @file inner_search_param.h
/// @brief Internal search parameters for vector index search operations.

#pragma once

#include <limits>
#include <mutex>

#include "typing.h"
#include "utils/pointer_define.h"
#include "utils/timer.h"

namespace vsag {

DEFINE_POINTER(Filter);
DEFINE_POINTER(Executor);

/// @brief Search mode enumeration for internal search operations.
enum InnerSearchMode { KNN_SEARCH = 1, RANGE_SEARCH = 2 };

/// @brief Search type enumeration for internal search operations.
enum InnerSearchType { PURE = 1, WITH_FILTER = 2 };

/// @brief Parameters for internal search operations in vector indexes.
///
/// This class encapsulates all parameters needed for search operations,
/// including KNN search, range search, filtering, and various optimization hints.
class InnerSearchParam {
public:
    InnerSearchParam() = default;

public:
    /// Number of nearest neighbors to return (for KNN search).
    int64_t topk{0};
    /// Radius threshold for range search.
    float radius{0.0F};
    /// Entry point ID for graph-based search.
    InnerIdType ep{0};
    /// Ef parameter for HNSW search (size of dynamic candidate list).
    uint64_t ef{10};
    /// Maximum number of hops during search.
    uint32_t hops_limit{std::numeric_limits<uint32_t>::max()};
    /// Filter function to determine if an inner ID is allowed.
    FilterPtr is_inner_id_allowed{nullptr};
    /// Ratio for skipping unnecessary computation.
    float skip_ratio{0.8F};
    /// Search mode (KNN or range search).
    InnerSearchMode search_mode{KNN_SEARCH};
    /// Maximum number of results for range search (-1 for unlimited).
    int range_search_limit_size{-1};
    /// Number of threads for parallel search.
    int64_t parallel_search_thread_count{1};

    /// Number of buckets to scan (for IVF indexes).
    int scan_bucket_size{1};
    /// Factor for search parameter adjustment.
    float factor{2.0F};
    /// Ratio for first-order scan optimization.
    float first_order_scan_ratio{1.0F};
    /// Executors for parallel search operations.
    std::vector<ExecutorPtr> executors;

    /// ID found during duplicate detection (mutable for internal use).
    mutable int64_t duplicate_id{-1};
    /// Whether to find duplicate IDs during search.
    bool find_duplicate{false};

    /// Whether to consider duplicates during search.
    bool consider_duplicate{false};

    /// Timer for measuring search time cost.
    std::shared_ptr<Timer> time_cost{nullptr};

    InnerSearchParam&
    operator=(const InnerSearchParam& other) = default;
};

}  // namespace vsag