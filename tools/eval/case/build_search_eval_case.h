
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
                        EvalConfig config,
                        EvalDatasetPtr dataset = nullptr)
        : EvalCase(dataset_path, index_path, nullptr, std::move(dataset)),
          config_(std::move(config)) {
        build_ = std::make_shared<BuildEvalCase>(
            dataset_path_, index_path_, std::move(index), config_, dataset_ptr_);
    }

    ~BuildSearchEvalCase() override = default;

    JsonType
    Run() override {
        auto build = std::move(build_);
        if (build == nullptr) {
            build = EvalCase::MakeInstance(config_, "build", dataset_ptr_);
        }
        auto build_result = build->Run();
        build.reset();

        auto search = EvalCase::MakeInstance(config_, "search", dataset_ptr_);
        auto search_result = search->Run();
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
    EvalConfig config_;
    EvalCasePtr build_;
};

}  // namespace vsag::eval
