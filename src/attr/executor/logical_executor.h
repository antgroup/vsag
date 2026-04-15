
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

/**
 * @file logical_executor.h
 * @brief Executor for logical attribute expressions.
 */

#pragma once

#include "executor.h"

namespace vsag {
/**
 * @brief Executor for handling logical attribute expressions.
 *
 * Processes expressions involving logical operators (AND, OR, NOT)
 * combining other executors.
 */
class LogicalExecutor : public Executor {
public:
    /**
     * @brief Constructs a logical executor.
     *
     * @param allocator Memory allocator for resource management.
     * @param expr The logical expression to execute.
     * @param attr_index Interface to the attribute inverted index.
     */
    explicit LogicalExecutor(Allocator* allocator,
                             const ExprPtr& expr,
                             const AttrInvertedInterfacePtr& attr_index);

    /**
     * @brief Clears the left and right child executors.
     */
    void
    Clear() override;

    /**
     * @brief Initializes the left and right child executors.
     */
    void
    Init() override;

    /**
     * @brief Executes the logical expression for the specified bucket.
     *
     * @param bucket_id The bucket identifier to process.
     * @return Filter* Pointer to the resulting filter.
     */
    Filter*
    Run(BucketIdType bucket_id) override;

private:
    /**
     * @brief Performs the logical operation on child executor results.
     *
     * @return Filter* Pointer to the resulting filter.
     */
    Filter*
    logical_run();

private:
    /// Left operand executor for binary logical operations.
    ExecutorPtr left_{nullptr};

    /// Right operand executor for binary logical operations.
    ExecutorPtr right_{nullptr};

    /// The logical operator to apply.
    LogicalOperator op_;
};

}  // namespace vsag
