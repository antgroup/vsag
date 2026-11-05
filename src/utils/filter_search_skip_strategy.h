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

#include <memory>
#include <string>

namespace vsag {

enum class FilterSearchSkipStrategyType {
    RANDOM,
    DETERMINISTIC_ACCUMULATIVE,
};

class FilterSearchSkipStrategy {
public:
    virtual ~FilterSearchSkipStrategy() = default;

    // Probabilistic gate for visiting neighbors during graph traversal.
    // visit_ratio = valid_ratio + (1 - valid_ratio) * skip_ratio
    // This ensures we visit more neighbors when most pass the filter.
    virtual bool
    ShouldVisit() = 0;
};

using FilterSearchSkipStrategyPtr = std::unique_ptr<FilterSearchSkipStrategy>;

FilterSearchSkipStrategyPtr
create_filter_search_skip_strategy(FilterSearchSkipStrategyType type,
                                   float valid_ratio,
                                   float skip_ratio);

FilterSearchSkipStrategyType
parse_filter_search_skip_strategy_type(const std::string& strategy_name);

const char*
filter_search_skip_strategy_type_to_string(FilterSearchSkipStrategyType type);

}  // namespace vsag
