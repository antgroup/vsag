

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

#include "build_eval_case.h"
#include "search_eval_case.h"
#include "vsag/factory.h"
#include "vsag/options.h"

namespace eval {

EvalCase::EvalCase(std::string dataset_path, std::string index_path, vsag::IndexPtr index)
    : dataset_path_(std::move(dataset_path)), index_path_(std::move(index_path)), index_(index) {
    this->dataset_ptr_ = EvalDataset::Load(dataset_path_);
    this->logger_ = vsag::Options::Instance().logger();
    this->basic_info_ = this->dataset_ptr_->GetInfo();
}
EvalCasePtr
EvalCase::MakeInstance(argparse::ArgumentParser& parser) {
    auto dataset_path = parser.get<std::string>("--datapath");
    auto index_path = parser.get<std::string>("--indexpath");
    auto index_name = parser.get<std::string>("--index_name");
    auto create_params = parser.get<std::string>("--create_params");

    auto index = vsag::Factory::CreateIndex(index_name, create_params);

    auto type = parser.get<std::string>("--type");
    if (type == "build") {
        return std::make_shared<BuildEvalCase>(dataset_path, index_path, index.value(), parser);
    } else if (type == "search") {
        return std::make_shared<SearchEvalCase>(dataset_path, index_path, index.value(), parser);
    } else {
        return nullptr;
    }
}
}  // namespace eval
