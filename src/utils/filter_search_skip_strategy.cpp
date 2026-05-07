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

#include "filter_search_skip_strategy.h"

#include "linear_congruential_generator.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

constexpr const char* RANDOM_SKIP_STRATEGY = "random";
constexpr const char* DETERMINISTIC_ACCUMULATIVE_SKIP_STRATEGY = "deterministic_accumulative";

double
get_retain_ratio(float valid_ratio, float skip_ratio) {
    if (valid_ratio == 1.0F) {
        return 1.0;
    }
    return static_cast<double>(1.0F - valid_ratio) * static_cast<double>(skip_ratio);
}

class RandomFilterSearchSkipStrategy : public FilterSearchSkipStrategy {
public:
    explicit RandomFilterSearchSkipStrategy(double retain_ratio)
        : skip_threshold_(1.0 - retain_ratio) {
    }

    bool
    ShouldSkipFilterCheck() override {
        return generator_.NextFloat() > skip_threshold_;
    }

private:
    LinearCongruentialGenerator generator_;
    double skip_threshold_{0.0};
};

class AccumulativeFilterSearchSkipStrategy : public FilterSearchSkipStrategy {
public:
    explicit AccumulativeFilterSearchSkipStrategy(double retain_ratio)
        : retain_ratio_(retain_ratio) {
    }

    bool
    ShouldSkipFilterCheck() override {
        accumulative_alpha_ += retain_ratio_;
        if (accumulative_alpha_ >= 1.0) {
            accumulative_alpha_ -= 1.0;
            return true;
        }
        return false;
    }

private:
    double retain_ratio_{1.0};
    double accumulative_alpha_{0.0};
};

}  // namespace

FilterSearchSkipStrategyPtr
create_filter_search_skip_strategy(FilterSearchSkipStrategyType type,
                                   float valid_ratio,
                                   float skip_ratio) {
    auto retain_ratio = get_retain_ratio(valid_ratio, skip_ratio);
    switch (type) {
        case FilterSearchSkipStrategyType::RANDOM:
            return std::make_unique<RandomFilterSearchSkipStrategy>(retain_ratio);
        case FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE:
            return std::make_unique<AccumulativeFilterSearchSkipStrategy>(retain_ratio);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT, "Unknown FilterSearchSkipStrategyType");
}

FilterSearchSkipStrategyType
parse_filter_search_skip_strategy_type(const std::string& strategy_name) {
    if (strategy_name == RANDOM_SKIP_STRATEGY) {
        return FilterSearchSkipStrategyType::RANDOM;
    }
    if (strategy_name == DETERMINISTIC_ACCUMULATIVE_SKIP_STRATEGY) {
        return FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE;
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        "invalid filter search skip strategy: " + strategy_name);
}

const char*
filter_search_skip_strategy_type_to_string(FilterSearchSkipStrategyType type) {
    switch (type) {
        case FilterSearchSkipStrategyType::RANDOM:
            return RANDOM_SKIP_STRATEGY;
        case FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE:
            return DETERMINISTIC_ACCUMULATIVE_SKIP_STRATEGY;
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT, "Unknown FilterSearchSkipStrategyType");
}

}  // namespace vsag
