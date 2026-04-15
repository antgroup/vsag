
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
 * @file comparison_executor.h
 * @brief Executor for comparison attribute expressions.
 */

#pragma once

#include "attr/multi_bitset_manager.h"
#include "executor.h"
#include "vsag/attribute.h"

namespace vsag {
/**
 * @brief Executor for handling comparison attribute expressions.
 *
 * Processes expressions involving comparison operators (e.g., <, <=, >, >=, ==, !=)
 * against attribute values.
 */
class ComparisonExecutor : public Executor {
public:
    /**
     * @brief Constructs a comparison executor.
     *
     * @param allocator Memory allocator for resource management.
     * @param expr The comparison expression to execute.
     * @param attr_index Interface to the attribute inverted index.
     */
    explicit ComparisonExecutor(Allocator* allocator,
                                const ExprPtr& expr,
                                const AttrInvertedInterfacePtr& attr_index);

    /**
     * @brief Clears the internal state and managers.
     */
    void
    Clear() override;

    /**
     * @brief Initializes the executor with field and operator information.
     */
    void
    Init() override;

    /**
     * @brief Executes the comparison expression for the specified bucket.
     *
     * @param bucket_id The bucket identifier to process.
     * @return Filter* Pointer to the resulting filter.
     */
    Filter*
    Run(BucketIdType bucket_id) override;

private:
    /// Name of the field being compared.
    std::string field_name_{};

    /// Attribute value used for comparison filtering.
    AttributePtr filter_attribute_{nullptr};

    /// The comparison operator to apply.
    ComparisonOperator op_{};

    /// Managers for multi-bitset operations.
    std::vector<const MultiBitsetManager*> managers_;
};

}  // namespace vsag
