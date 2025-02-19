
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

#include "./build_eval_case.h"
#include "./eval_case.h"

namespace vsag::eval {

class BuildSearchEvalCase : public EvalCase {
public:
    BuildSearchEvalCase(const std::string& dataset_path,
                        const std::string& index_path,
                        vsag::IndexPtr index,
                        EvalConfig config)
        : EvalCase(dataset_path, index_path, index) {
        build_ = EvalCase::MakeInstance(config, "build");
        search_ = EvalCase::MakeInstance(config, "search");
        config_ = std::move(config);
    }

    ~BuildSearchEvalCase() override = default;

    JsonType
    Run() override {
        auto build_result = build_->Run();
        auto search_result = search_->Run();
        return merge_results(build_result, search_result);
    }

private:
    static JsonType
    merge_results(JsonType result1, JsonType result2) {
        result1.merge_patch(result2);
        result1["action"] = "build,search";
        return result1;
    }

private:
    EvalCasePtr build_;
    EvalCasePtr search_;
    EvalConfig config_;
};

}  // namespace vsag::eval
