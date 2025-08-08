
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

#include "attr/executor/executor.h"
#include "typing.h"
#include "utils/timer.h"
#include "vsag/filter.h"

namespace vsag {

enum InnerSearchMode { KNN_SEARCH = 1, RANGE_SEARCH = 2 };

enum InnerSearchType { PURE = 1, WITH_FILTER = 2 };

class InnerSearchParam {
public:
    int64_t topk{0};
    float radius{0.0f};
    InnerIdType ep{0};
    uint64_t ef{10};
    FilterPtr is_inner_id_allowed{nullptr};
    float skip_ratio{0.8F};
    InnerSearchMode search_mode{KNN_SEARCH};
    int range_search_limit_size{-1};
    int64_t parallel_search_thread_count{1};

    // for ivf
    int scan_bucket_size{1};
    float factor{2.0F};
    float first_order_scan_ratio{1.0F};
    Allocator* search_alloc{nullptr};
    std::vector<ExecutorPtr> executors;
    mutable int64_t duplicate_id{-1};
    bool consider_duplicate{false};

    // time record
    std::shared_ptr<Timer> time_cost{nullptr};

    InnerSearchParam&
    operator=(const InnerSearchParam& other) {
        if (this != &other) {
            topk = other.topk;
            radius = other.radius;
            ep = other.ep;
            ef = other.ef;
            skip_ratio = other.skip_ratio;
            search_mode = other.search_mode;
            range_search_limit_size = other.range_search_limit_size;
            is_inner_id_allowed = other.is_inner_id_allowed;
            scan_bucket_size = other.scan_bucket_size;
            factor = other.factor;
            first_order_scan_ratio = other.first_order_scan_ratio;
        }
        return *this;
    }
};
}  // namespace vsag
