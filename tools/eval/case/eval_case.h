
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

#include "../eval_config.h"
#include "../eval_dataset.h"
#include "nlohmann/json.hpp"
#include "vsag/index.h"
#include "vsag/logger.h"

namespace vsag::eval {
using JsonType = nlohmann::json;

class EvalCase;
using EvalCasePtr = std::shared_ptr<EvalCase>;

class EvalCase {
public:
    static EvalCasePtr
    MakeInstance(const EvalConfig& config, std::string type = "none");

    static void
    MergeJsonType(const JsonType& input, JsonType& output) {
        for (auto& [key, value] : input.items()) {
            output[key] = value;
        }
    }

    static void
    PrintResult(const JsonType& result) {
        std::cout << result.dump(4) << std::endl;
    }

public:
    explicit EvalCase(std::string dataset_path, std::string index_path, vsag::IndexPtr index);

    virtual ~EvalCase() = default;

    virtual JsonType
    Run() = 0;

    using Logger = vsag::Logger*;

protected:
    const std::string dataset_path_{};
    const std::string index_path_{};

    EvalDatasetPtr dataset_ptr_{nullptr};

    vsag::IndexPtr index_{nullptr};

    Logger logger_{nullptr};

    JsonType basic_info_{};
};

}  // namespace vsag::eval
