
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

#include <string>

#include "impl/bitset/computable_bitset.h"
#include "vsag/filter.h"

namespace vsag {
enum class OpType {
    kNone,
    kUnary,
    kBinary,
};

enum class ExpressionType {
    kNumericConstant,
    kFieldExpression,
    kStringConstant,
    kStrListConstant,
    kIntListConstant,
    kArithmeticExpression,
    kComparisonExpression,
    kIntListExpression,
    kStrListExpression,
    kNotExpression,
    kLogicalExpression,
};

/**
 * @class Expression
 * @brief Abstract base class for all expression types.
 *
 * This class provides the interface for expression evaluation and string representation.
 * All concrete expression types should inherit from this class and implement the pure virtual methods.
 */
class Expression {
public:
    Expression(ExpressionType expr_type, OpType op_type)
        : expr_type_(expr_type), op_type_(op_type) {
    }

    virtual ~Expression() = default;

    virtual std::string
    ToString() const = 0;

    virtual void
    Execute() = 0;

    ExpressionType
    GetExprType() const {
        return expr_type_;
    }

    OpType
    GetOpType() const {
        return op_type_;
    }

public:
    ExpressionType expr_type_;
    OpType op_type_;

    FilterPtr filter_func_{nullptr};
    ComputableBitsetPtr bitset_{nullptr};

    bool only_bitset_{true};
};

using ExprPtr = std::shared_ptr<Expression>;
}  // namespace vsag
