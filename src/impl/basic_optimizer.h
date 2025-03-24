
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
#include "basic_searcher.h"
#include "common.h"
#include "data_cell/flatten_datacell.h"
#include "index/index_common_param.h"
#include "runtime_parameter.h"
#include "typing.h"

namespace vsag {

template <typename OptimizableOBJ>
class Optimizer {
public:
    Optimizer(const IndexCommonParam& common_param, uint32_t trials = OPTIMIZE_TRIALS)
        : parameters_(common_param.allocator_.get()),
          best_params_(common_param.allocator_.get()),
          n_trials_(trials),
          best_loss_(std::numeric_limits<double>::max()) {
        allocator_ = common_param.allocator_.get();
        std::random_device rd;
        gen_.seed(rd());
    }

    void
    RegisterParameter(const std::shared_ptr<RuntimeParameter>& runtime_parameter) {
        parameters_.push_back(runtime_parameter);
    }

    bool
    Optimize(std::shared_ptr<OptimizableOBJ> obj) {
        double original_loss = obj->MockRun();

        bool successful_optimized = false;
        for (uint32_t i = 0; i < n_trials_; ++i) {
            // generate a group of runtime params
            UnorderedMap<std::string, ParamValue> current_params(allocator_);
            for (auto& param : parameters_) {
                while (not param->IsEnd()) {
                    current_params[param->name_] = param->Cur();
                    auto set_status = obj->SetRuntimeParameters(current_params);
                    if (not set_status) {
                        continue;
                    }

                    // evaluate
                    double loss = obj->MockRun();
                    double improvement = (original_loss - loss) / original_loss * 100;
                    vsag::logger::debug(fmt::format(
                        "Trial {}: setting {} = {}, get new loss = {}, improving = {} %",
                        i + 1,
                        param->name_,
                        std::get<int>(current_params[param->name_]),
                        loss,
                        improvement));

                    // update
                    if (loss < best_loss_ and improvement > 1) {
                        successful_optimized = true;
                        best_loss_ = loss;
                        best_params_ = current_params;
                    }

                    param->Next();
                }
            }
        }

        if (successful_optimized) {
            obj->SetRuntimeParameters(best_params_);
        }
        return successful_optimized;
    }

    UnorderedMap<std::string, ParamValue>
    GetBestParameters() const {
        return best_params_;
    }

    double
    GetBestLoss() const {
        return best_loss_;
    }

private:
    Allocator* allocator_{nullptr};

    Vector<std::shared_ptr<RuntimeParameter>> parameters_;
    uint32_t n_trials_{0};
    std::mt19937 gen_;

    UnorderedMap<std::string, ParamValue> best_params_;
    double best_loss_{0};
};

}  // namespace vsag