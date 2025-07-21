
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

#include "executor.h"

namespace vsag {
class LogicalExecutor : public Executor {
public:
    explicit LogicalExecutor(Allocator* allocator,
                             const ExprPtr& expr,
                             const AttrInvertedInterfacePtr& attr_index);

    void
    Clear() override;

    void
    Init() override;

    Filter*
    Run(BucketIdType bucket_id) override;

private:
    Filter*
    logical_run();

private:
    ExecutorPtr left_{nullptr};
    ExecutorPtr right_{nullptr};

    LogicalOperator op_;
};

}  // namespace vsag
