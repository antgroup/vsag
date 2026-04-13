
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
 * @file executor.h
 * @brief Base class for attribute expression executors.
 */

#pragma once
#include "attr/expression.h"
#include "datacell/attribute_inverted_interface.h"
#include "impl/bitset/computable_bitset.h"
#include "impl/filter/filter_headers.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(Executor);

/**
 * @brief Base class for executing attribute filter expressions.
 *
 * Provides the interface for running attribute expressions and managing
 * the resulting filter and bitset structures.
 */
class Executor {
public:
    /**
     * @brief Creates an executor instance based on the expression type.
     *
     * @param allocator Memory allocator for resource management.
     * @param expression The attribute expression to execute.
     * @param attr_index Interface to the attribute inverted index.
     * @return ExecutorPtr Pointer to the created executor instance.
     */
    static ExecutorPtr
    MakeInstance(Allocator* allocator,
                 const ExprPtr& expression,
                 const AttrInvertedInterfacePtr& attr_index);

    /**
     * @brief Constructs an executor with the given expression and index.
     *
     * @param allocator Memory allocator for resource management.
     * @param expression The attribute expression to execute.
     * @param attr_index Interface to the attribute inverted index.
     */
    Executor(Allocator* allocator,
             const ExprPtr& expression,
             const AttrInvertedInterfacePtr& attr_index)
        : expr_(expression),
          attr_index_(attr_index),
          allocator_(allocator),
          bitset_type_(attr_index->GetBitsetType()){};

    virtual ~Executor() {
        if (this->own_bitset_) {
            delete bitset_;
            bitset_ = nullptr;
        }
        delete filter_;
    }

    /**
     * @brief Clears the internal bitset state.
     */
    virtual void
    Clear() {
        if (this->bitset_ != nullptr) {
            this->bitset_->Clear();
        }
    };

    /**
     * @brief Initializes the executor before running.
     */
    virtual void
    Init(){};

    /**
     * @brief Executes the expression for a specific bucket.
     *
     * @param bucket_id The bucket identifier to process.
     * @return Filter* Pointer to the resulting filter.
     */
    virtual Filter*
    Run(BucketIdType bucket_id) = 0;

    /**
     * @brief Executes the expression for bucket 0.
     *
     * @return Filter* Pointer to the resulting filter.
     */
    Filter*
    Run() {
        return this->Run(0);
    }

public:
    /// Whether the executor only produces bitset output (no filter).
    bool only_bitset_{true};

    /// Pointer to the result filter structure.
    Filter* filter_{nullptr};

    /// Pointer to the computable bitset result.
    ComputableBitset* bitset_{nullptr};

    /// The expression to be executed.
    ExprPtr expr_{nullptr};

    /// Interface to the attribute inverted index.
    AttrInvertedInterfacePtr attr_index_{nullptr};

    /// Memory allocator for resource management.
    Allocator* const allocator_{nullptr};

    /// Whether this executor owns the bitset memory.
    bool own_bitset_{false};

    /// Type of the computable bitset used.
    ComputableBitsetType bitset_type_{ComputableBitsetType::FastBitset};
};
}  // namespace vsag
