
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

#include "eval_case.h"
#include "monitor/monitor.h"
namespace vsag::eval {

class BuildEvalCase : public EvalCase {
public:
    BuildEvalCase(const std::string& dataset_path,
                  const std::string& index_path,
                  vsag::IndexPtr index,
                  EvalConfig config);

    ~BuildEvalCase() override = default;

    void
    Run() override;

private:
    void
    init_monitors();

    void
    do_build();

    void
    serialize();

    JsonType
    process_result();

private:
    std::vector<MonitorPtr> monitors_{};

    EvalConfig config_;
};
}  // namespace vsag::eval
