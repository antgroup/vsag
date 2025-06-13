
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

#include <memory>
#include <string>
#include <variant>
#include <vector>

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

    ExpressionType
    GetExprType() const {
        return expr_type_;
    }

    OpType
    GetOpType() const {
        return op_type_;
    }

protected:
    ExpressionType expr_type_;
    OpType op_type_;
};

using ExprPtr = std::shared_ptr<Expression>;

// Comparison operators
enum class ComparisonOperator {
    EQ,  // =
    NE,  // !=
    GT,  // >
    LT,  // <
    GE,  // >=
    LE   // <=
};

// Logical operators
enum class LogicalOperator { AND, OR, NOT };

// Arithmetic operators
enum class ArithmeticOperator {
    ADD,  // +
    SUB,  // -
    MUL,  // *
    DIV   // /
};

static std::string
ToString(const ArithmeticOperator& op) {
    switch (op) {
        case ArithmeticOperator::ADD:
            return "+";
        case ArithmeticOperator::SUB:
            return "-";
        case ArithmeticOperator::MUL:
            return "*";
        case ArithmeticOperator::DIV:
            return "/";
    }
    throw std::runtime_error("unsupported type");
}

static std::string
ToString(const LogicalOperator& op) {
    switch (op) {
        case LogicalOperator::AND:
            return "AND";
        case LogicalOperator::OR:
            return "OR";
        case LogicalOperator::NOT:
            return "!";
    }
    throw std::runtime_error("unsupported type");
}

static std::string
ToString(const ComparisonOperator& op) {
    switch (op) {
        case ComparisonOperator::EQ:
            return "=";
        case ComparisonOperator::NE:
            return "!=";
        case ComparisonOperator::GT:
            return ">";
        case ComparisonOperator::LT:
            return "<";
        case ComparisonOperator::GE:
            return ">=";
        case ComparisonOperator::LE:
            return "<=";
    }
    throw std::runtime_error("unsupported type");
}

using NumericValue = std::variant<long, double>;
using StrList = std::vector<std::string>;

inline bool
CheckSameVType(const NumericValue&& lhs, const NumericValue&& rhs) {
    return (std::holds_alternative<double>(lhs) && std::holds_alternative<double>(rhs)) ||
           (std::holds_alternative<long>(lhs) && std::holds_alternative<long>(rhs));
}

inline NumericValue
operator+(const NumericValue& lhs, const NumericValue& rhs) {
    return std::visit(
        [](auto&& l, auto&& r) -> NumericValue {
            if (CheckSameVType(l, r)) {
                return l + r;
            }
            throw std::runtime_error("unsupported type");
        },
        lhs,
        rhs);
}

inline NumericValue
operator-(const NumericValue& lhs, const NumericValue& rhs) {
    return std::visit(
        [](auto&& l, auto&& r) -> NumericValue {
            if (CheckSameVType(l, r)) {
                return l - r;
            }
            throw std::runtime_error("unsupported type");
        },
        lhs,
        rhs);
}

inline NumericValue
operator*(const NumericValue& lhs, const NumericValue& rhs) {
    return std::visit(
        [](auto&& l, auto&& r) -> NumericValue {
            if (CheckSameVType(l, r)) {
                return l * r;
            }
            throw std::runtime_error("unsupported type");
        },
        lhs,
        rhs);
}

inline NumericValue
operator/(const NumericValue& lhs, const NumericValue& rhs) {
    return std::visit(
        [](auto&& l, auto&& r) -> NumericValue {
            if (CheckSameVType(l, r) && r != 0) {
                return l / r;
            }
            throw std::runtime_error("unsupported type");
        },
        lhs,
        rhs);
}

// Field reference
class FieldExpression : public Expression {
public:
    using ptr = std::shared_ptr<FieldExpression>;

    explicit FieldExpression(const std::string& name)
        : Expression(ExpressionType::kFieldExpression, OpType::kNone), fieldName(name) {
    }

    std::string
    ToString() const override {
        return fieldName;
    }

    std::string fieldName;
};

// Numeric constant
class NumericConstant : public Expression {
public:
    using ptr = std::shared_ptr<NumericConstant>;

    explicit NumericConstant(NumericValue value)
        : Expression(ExpressionType::kNumericConstant, OpType::kNone), value(value) {
    }

    std::string
    ToString() const override {
        return std::visit(
            [](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;

                if constexpr (std::is_same_v<T, long>) {
                    return std::to_string(arg);
                } else if constexpr (std::is_same_v<T, double>) {
                    std::string str = std::to_string(arg);
                    str.erase(str.find_last_not_of('0') + 1, std::string::npos);
                    if (str.back() == '.') {
                        str = str + '0';
                    }
                    return str;
                } else {
                    return "not supported type";
                }
            },
            value);
    }

    NumericValue value;
};

// String constant
class StringConstant : public Expression {
public:
    using ptr = std::shared_ptr<StringConstant>;

    explicit StringConstant(const std::string& value)
        : Expression(ExpressionType::kStringConstant, OpType::kNone), value(value) {
    }

    std::string
    ToString() const override {
        return '"' + value + '"';
    }

    std::string value;
};

class StrListConstant : public Expression {
public:
    using ptr = std::shared_ptr<StrListConstant>;

    explicit StrListConstant(StrList values)
        : Expression(ExpressionType::kStrListConstant, OpType::kNone), values(std::move(values)) {
    }

    std::string
    ToString() const override {
        std::string result = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0)
                result += ", ";
            result += '"' + values[i] + '"';
        }
        return result + "]";
    }

    StrList values;
};

class IntListConstant : public Expression {
public:
    using ptr = std::shared_ptr<IntListConstant>;

    explicit IntListConstant(std::vector<long> values)
        : Expression(ExpressionType::kIntListConstant, OpType::kNone), values(std::move(values)) {
    }

    std::string
    ToString() const override {
        std::string result = "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0)
                result += ", ";
            result += std::to_string(values[i]);
        }
        return result + "]";
    }

    std::vector<long> values;
};

// Arithmetic expression
class ArithmeticExpression : public Expression {
public:
    ArithmeticExpression(ExprPtr left, ArithmeticOperator op, ExprPtr right)
        : Expression(ExpressionType::kArithmeticExpression, OpType::kBinary),
          left(std::move(left)),
          op(op),
          right(std::move(right)) {
    }

    std::string
    ToString() const override {
        return "(" + left->ToString() + " " + vsag::ToString(op) + " " + right->ToString() + ")";
    }

    ExprPtr left;
    ArithmeticOperator op;
    ExprPtr right;
};

// Comparison expression
class ComparisonExpression : public Expression {
public:
    ComparisonExpression(ExprPtr left, ComparisonOperator op, ExprPtr right)
        : Expression(ExpressionType::kComparisonExpression, OpType::kBinary),
          left(std::move(left)),
          op(op),
          right(std::move(right)) {
    }

    std::string
    ToString() const override {
        return "(" + left->ToString() + " " + vsag::ToString(op) + " " + right->ToString() + ")";
    }

    ExprPtr left;
    ComparisonOperator op;
    ExprPtr right;
};

// List membership expression
class IntListExpression : public Expression {
public:
    IntListExpression(ExprPtr field, const bool is_not_in, IntListConstant::ptr values)
        : Expression(ExpressionType::kIntListExpression, OpType::kBinary),
          field(std::move(field)),
          is_not_in(is_not_in),
          values(std::move(values)) {
    }

    std::string
    ToString() const override {
        const std::string op = is_not_in ? "NOT_IN" : "IN";
        return "(" + field->ToString() + " " + op + " " + values->ToString() + ")";
    }

    ExprPtr field;
    bool is_not_in;  // true for NOT IN, false for IN
    IntListConstant::ptr values;
};

// List expression
class StrListExpression : public Expression {
public:
    StrListExpression(ExprPtr field, const bool is_not_in, StrListConstant::ptr values)
        : Expression(ExpressionType::kStrListExpression, OpType::kBinary),
          field(std::move(field)),
          is_not_in(is_not_in),
          values(std::move(values)) {
    }

    std::string
    ToString() const override {
        const std::string op = is_not_in ? "NOT_IN" : "IN";
        return "(" + field->ToString() + " " + op + " " + values->ToString() + ")";
    }

    ExprPtr field;
    bool is_not_in;  // true for NOT IN, false for IN
    StrListConstant::ptr values;
};

// Logical expression
class LogicalExpression : public Expression {
public:
    LogicalExpression(ExprPtr left, LogicalOperator op, ExprPtr right)
        : Expression(ExpressionType::kLogicalExpression, OpType::kBinary),
          left(std::move(left)),
          op(op),
          right(std::move(right)) {
    }

    std::string
    ToString() const override {
        return "(" + left->ToString() + " " + vsag::ToString(op) + " " + right->ToString() + ")";
    }

    ExprPtr left;
    LogicalOperator op;
    ExprPtr right;
};

// Not expression
class NotExpression : public Expression {
public:
    explicit NotExpression(ExprPtr expr)
        : Expression(ExpressionType::kNotExpression, OpType::kUnary), expr(std::move(expr)) {
    }

    std::string
    ToString() const override {
        return "! (" + expr->ToString() + ")";
    }

    ExprPtr expr;
};
}  // namespace vsag
