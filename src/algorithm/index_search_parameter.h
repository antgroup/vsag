
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

#include "inner_string_params.h"
#include "parameter.h"

namespace vsag {
class IndexSearchParameter {
public:
    IndexSearchParameter() = default;
    ~IndexSearchParameter() = default;

    inline void
    FromJson(const JsonType& json) {
        if (json.Contains(SEARCH_PARALLELISM)) {
            parallel_search_thread_count = json[SEARCH_PARALLELISM].GetInt();
            if (parallel_search_thread_count <= 0) {
                parallel_search_thread_count = 1;
            }
        }

        if (json.Contains(SEARCH_MAX_TIME_COST_MS)) {
            timeout_ms = json[SEARCH_MAX_TIME_COST_MS].GetInt();
            enable_time_record = true;
        }
    }

public:
    int64_t parallel_search_thread_count{1};  // for parallel search

    // for timeout
    double timeout_ms{std::numeric_limits<double>::max()};
    bool enable_time_record{false};
};
}  // namespace vsag
