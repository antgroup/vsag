
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

#include <random>

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
        : parameters_(common_param.allocator_.get()), best_params_(common_param.allocator_.get()) {
        allocator_ = common_param.allocator_.get();
        std::random_device rd;
        gen_.seed(rd());
    }

    bool
    Optimize(std::shared_ptr<OptimizableOBJ> obj);

    void
    RegisterParameter(const std::shared_ptr<RuntimeParameter>& runtime_parameter) {
        parameters_.push_back(runtime_parameter);
    }

    UnorderedMap<std::string, ParamValue>
    GetBestParameters() const {
        return best_params_;
    }

private:
    Allocator* allocator_{nullptr};

    std::mt19937 gen_;

    Vector<std::shared_ptr<RuntimeParameter>> parameters_;

    UnorderedMap<std::string, ParamValue> best_params_;
};

}  // namespace vsag
