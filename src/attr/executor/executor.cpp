
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

#include "executor.h"

#include "comparison_executor.h"
#include "integer_list_executor.h"
#include "logical_executor.h"
#include "string_list_executor.h"
#include "region_filter_executor.h"
namespace vsag {

ExecutorPtr
Executor::MakeInstance(Allocator* allocator,
                       const ExprPtr& expression,
                       const AttrInvertedInterfacePtr& attr_index) {
    if (std::dynamic_pointer_cast<ComparisonExpression>(expression)) {
        return std::make_shared<ComparisonExecutor>(allocator, expression, attr_index);
    }
    if (std::dynamic_pointer_cast<IntListExpression>(expression)) {
        return std::make_shared<IntegerListExecutor>(allocator, expression, attr_index);
    }
    if (std::dynamic_pointer_cast<StrListExpression>(expression)) {
        return std::make_shared<StringListExecutor>(allocator, expression, attr_index);
    }
    if (std::dynamic_pointer_cast<LogicalExpression>(expression)) {
        return std::make_shared<LogicalExecutor>(allocator, expression, attr_index);
    }
    if (std::dynamic_pointer_cast<RegionFilterExpression>(expression)) {
        return std::make_shared<RegionFilterExecutor>(allocator, expression, attr_index);
    }
    throw VsagException(ErrorType::INTERNAL_ERROR, "Unsupported expression type");
}
}  // namespace vsag
