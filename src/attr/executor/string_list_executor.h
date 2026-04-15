
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
 * @file string_list_executor.h
 * @brief Executor for string list attribute expressions (IN/NOT IN).
 */

#pragma once

#include "executor.h"
#include "vsag/attribute.h"

namespace vsag {
/**
 * @brief Executor for handling string list attribute expressions.
 *
 * Processes IN and NOT IN expressions for string attribute values.
 */
class StringListExecutor : public Executor {
public:
    /**
     * @brief Constructs a string list executor.
     *
     * @param allocator Memory allocator for resource management.
     * @param expr The string list expression to execute.
     * @param attr_index Interface to the attribute inverted index.
     */
    explicit StringListExecutor(Allocator* allocator,
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
     * @brief Executes the string list expression for the specified bucket.
     *
     * @param bucket_id The bucket identifier to process.
     * @return Filter* Pointer to the resulting filter.
     */
    Filter*
    Run(BucketIdType bucket_id) override;

private:
    /// Name of the field being checked.
    std::string field_name_{};

    /// Attribute value list for filtering.
    AttributePtr filter_attribute_{nullptr};

    /// True if NOT IN operation, false for IN operation.
    bool is_not_in_{false};

    /// Managers for multi-bitset operations.
    std::vector<const MultiBitsetManager*> managers_;
};

}  // namespace vsag
