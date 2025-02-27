

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

#include "eval_case.h"

#include <utility>

#include "./build_eval_case.h"
#include "./build_search_eval_case.h"
#include "./search_eval_case.h"
#include "vsag/factory.h"
#include "vsag/options.h"

namespace vsag::eval {

EvalCase::EvalCase(std::string dataset_path, std::string index_path, vsag::IndexPtr index)
    : dataset_path_(std::move(dataset_path)), index_path_(std::move(index_path)), index_(index) {
    this->dataset_ptr_ = EvalDataset::Load(dataset_path_);
    this->logger_ = vsag::Options::Instance().logger();
    this->basic_info_ = this->dataset_ptr_->GetInfo();
}

EvalCasePtr
EvalCase::MakeInstance(const EvalConfig& config, std::string type) {
    auto dataset_path = config.dataset_path;
    auto index_path = config.index_path;
    auto index_name = config.index_name;
    auto create_params = config.build_param;

    auto index = vsag::Factory::CreateIndex(index_name, create_params);

    // to support BuildSearch
    if (type == "none") {
        type = config.action_type;
    }

    if (type == "build") {
        return std::make_shared<BuildEvalCase>(dataset_path, index_path, index.value(), config);
    }
    if (type == "search") {
        return std::make_shared<SearchEvalCase>(dataset_path, index_path, index.value(), config);
    }
    if (type == "build,search") {
        return std::make_shared<BuildSearchEvalCase>(
            dataset_path, index_path, index.value(), config);
    }
    return nullptr;
}
}  // namespace vsag::eval
