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

/// @file basic_optimizer.h
/// @brief Basic optimizer for tuning index parameters.

#pragma once

#include <random>

#include "common.h"
#include "index_common_param.h"
#include "runtime_parameter.h"
#include "typing.h"

namespace vsag {

/// @brief Optimizer for tuning index parameters using parameter search.
///
/// This template class provides a framework for optimizing index parameters
/// by iterating through registered runtime parameters and evaluating performance.
///
/// @tparam OptimizableOBJ Type of the object to optimize.
template <typename OptimizableOBJ>
class Optimizer {
public:
    /// @brief Constructs an optimizer with common parameters.
    /// @param common_param Common parameters including allocator.
    Optimizer(const IndexCommonParam& common_param)
        : parameters_(common_param.allocator_.get()),
          best_params_(common_param.allocator_.get()),
          allocator_(common_param.allocator_.get()) {
        std::random_device rd;
        gen_.seed(rd());
    }

    /// @brief Optimizes the given object by searching for the best parameters.
    /// @param obj Object to optimize.
    /// @return Optimization score (higher is better).
    double
    Optimize(std::shared_ptr<OptimizableOBJ> obj);

    /// @brief Registers a runtime parameter for optimization.
    /// @param runtime_parameter Parameter to register.
    void
    RegisterParameter(const RuntimeParameter& runtime_parameter) {
        parameters_.emplace_back(runtime_parameter);
    }

private:
    /// Allocator for memory management.
    Allocator* const allocator_{nullptr};

    /// Random number generator for parameter sampling.
    std::mt19937 gen_;

    /// Vector of registered runtime parameters.
    Vector<RuntimeParameter> parameters_;

    /// Map storing the best parameter values found.
    UnorderedMap<std::string, float> best_params_;
};

}  // namespace vsag