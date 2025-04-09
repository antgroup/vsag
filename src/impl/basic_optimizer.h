
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
          n_trials_(trials) {
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
        vsag::logger::info(fmt::format("============Start Optimize==========="));
        bool ret = false;
        double original_loss = obj->MockRun();

        // generate a group of runtime params
        UnorderedMap<std::string, ParamValue> current_params(allocator_);
        for (auto& param : parameters_) {
            bool successful_optimized = false;
            double before_loss = obj->MockRun();
            double best_loss = before_loss;
            double best_improve = 0;

            while (not param->IsEnd()) {
                current_params[param->name_] = param->Cur();
                auto set_status = obj->SetRuntimeParameters(current_params);
                if (not set_status) {
                    continue;
                }

                // evaluate
                double loss = obj->MockRun();
                double improvement = (before_loss - loss) / before_loss * 100;
                vsag::logger::info(
                    fmt::format("setting {} -> {}, get new loss = {:.3f} from before = {:.3f}, "
                                "improving = {:.3f} %",
                                param->name_,
                                std::get<int>(current_params[param->name_]),
                                loss,
                                before_loss,
                                improvement));

                // update
                if (loss < best_loss and improvement > 2.0) {
                    successful_optimized = true;
                    ret = true;
                    best_loss = loss;
                    best_improve = improvement;
                    best_params_[param->name_] = current_params[param->name_];
                }

                param->Next();
            }

            if (successful_optimized) {
                current_params[param->name_] = best_params_[param->name_];
                vsag::logger::info(fmt::format("setting to best param: {} -> {}, improving {:.3f}%",
                                               param->name_,
                                               std::get<int>(best_params_[param->name_]),
                                               best_improve));
            } else {
                param->Reset();
                current_params[param->name_] = param->Cur();
                obj->SetRuntimeParameters(current_params);

                vsag::logger::info(fmt::format("reset to original param: {} -> {}",
                                               param->name_,
                                               std::get<int>(current_params[param->name_])));
            }

            param->Reset();
        }

        vsag::logger::info(fmt::format("============Optimize Report==========="));
        if (ret) {
            double optimized_loss = obj->MockRun();
            double improvement = (original_loss - optimized_loss) / original_loss * 100;
            for (auto& param : best_params_) {
                vsag::logger::info(
                    fmt::format("setting {} -> {}", param.first, std::get<int>(param.second)));
            }
            vsag::logger::info(fmt::format("improving: {:.3f}%", improvement));
        } else {
            vsag::logger::info(fmt::format("no improvement"));
        }

        vsag::logger::info(fmt::format("============Finish Optimize==========="));
        return ret;
    }

    UnorderedMap<std::string, ParamValue>
    GetBestParameters() const {
        return best_params_;
    }

private:
    Allocator* allocator_{nullptr};

    Vector<std::shared_ptr<RuntimeParameter>> parameters_;
    uint32_t n_trials_{0};
    std::mt19937 gen_;

    UnorderedMap<std::string, ParamValue> best_params_;
};

}  // namespace vsag
