
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

#include "basic_optimizer.h"

#include "basic_searcher.h"

namespace vsag {

template <typename OptimizableOBJ>
double
Optimizer<OptimizableOBJ>::Optimize(std::shared_ptr<OptimizableOBJ> obj) {
    vsag::logger::info(fmt::format("============Start Optimize==========="));
    bool have_improvement = false;
    double original_loss = obj->MockRun();

    // generate a group of runtime params
    UnorderedMap<std::string, float> current_params(allocator_);

    if (parameters_.size() == 0) {
        return 0;
    }

    bool has_next_combination = true;
    double baseline_loss = obj->MockRun();
    double previous_best_loss = baseline_loss;
    double best_loss = baseline_loss;
    double best_improve = 0;

    while (has_next_combination) {
        // 1. set param
        std::string set_info = "";
        for (auto& param : parameters_) {
            current_params[param.name_] = param.Cur();
            set_info += fmt::format("setting {} -> {}, ", param.name_, param.Cur());
        }
        auto set_status = obj->SetRuntimeParameters(current_params);
        if (not set_status) {
            continue;
        }

        // 2. evaluate
        double loss = obj->MockRun();
        double improvement = (baseline_loss - loss) / baseline_loss * 100;
        set_info +=
            fmt::format("get new loss = {:.3f} from baseline = {:.3f}, improving = {:.3f} %",
                        loss,
                        baseline_loss,
                        improvement);
        vsag::logger::info(set_info);

        // 3. update
        // overall principle: choose smaller param
        if (loss < best_loss and                  // condition1. has improvement
            improvement - best_improve > 0.2 and  // condition2. improvement is noteworthy
            improvement > 2.0) {                  // condition3. improvement is valid
            have_improvement = true;
            best_loss = loss;
            best_improve = improvement;

            for (auto& param : parameters_) {
                best_params_[param.name_] = param.Cur();
            }
        }

        // 4. iter parameter
        size_t level = parameters_.size() - 1;
        while (level < parameters_.size()) {
            parameters_[level].Next();

            if (!parameters_[level].IsEnd()) {
                break;
            }

            parameters_[level].Reset();
            if (level == 0) {
                has_next_combination = false;
            }
            --level;
        }
    }

    vsag::logger::info(fmt::format("============Optimize Report==========="));
    double end2end_improvement = 0;
    if (have_improvement) {
        obj->SetRuntimeParameters(best_params_);
        double optimized_loss = obj->MockRun();
        end2end_improvement = (original_loss - optimized_loss) / original_loss * 100;
        for (const auto& param : best_params_) {
            vsag::logger::info(fmt::format("setting {} -> {:.1f}", param.first, param.second));
        }
        vsag::logger::info(
            fmt::format("get new loss = {:.3f}, from original = {:.3f}, improving: {:.3f}%",
                        optimized_loss,
                        original_loss,
                        end2end_improvement));
    } else {
        vsag::logger::info(fmt::format("no improvement"));
    }

    vsag::logger::info(fmt::format("============Finish Optimize==========="));
    return end2end_improvement;
}

template class Optimizer<BasicSearcher>;
}  // namespace vsag
