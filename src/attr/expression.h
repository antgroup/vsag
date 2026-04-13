
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
 * @file expression.h
 * @brief Expression AST (Abstract Syntax Tree) definitions for filter condition parsing.
 *
 * This file contains all expression types used in the VSAG filter condition system,
 * including numeric constants, string constants, field references, arithmetic expressions,
 * comparison expressions, and logical expressions.
 */

#pragma once

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace vsag {

/**
 * @enum OpType
 * @brief Enumeration of operator types for expressions.
 */
enum class OpType {
    kNone,    ///< No operator (leaf expression)
    kUnary,   ///< Unary operator (e.g., NOT)
    kBinary,  ///< Binary operator (e.g., AND, OR, +, -)
};

/**
 * @enum ExpressionType
 * @brief Enumeration of all supported expression types in the AST.
 */
enum class ExpressionType {
    kNumericConstant,       ///< Numeric constant value
    kFieldExpression,       ///< Field reference
    kStringConstant,        ///< String constant value
    kStrListConstant,       ///< String list constant
    kIntListConstant,       ///< Integer list constant
    kArithmeticExpression,  ///< Arithmetic operation (e.g., +, -, *, /)
    kComparisonExpression,  ///< Comparison operation (e.g., =, !=, >, <)
    kIntListExpression,     ///< Integer list membership (IN/NOT IN)
    kStrListExpression,     ///< String list membership (IN/NOT IN)
    kNotExpression,         ///< Logical NOT expression
    kLogicalExpression,     ///< Logical operation (AND, OR)
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
    /**
     * @brief Constructs an Expression with the specified type and operator type.
     * @param expr_type The type of the expression.
     * @param op_type The operator type of the expression.
     */
    Expression(ExpressionType expr_type, OpType op_type)
        : expr_type_(expr_type), op_type_(op_type) {
    }

    virtual ~Expression() = default;

    /**
     * @brief Converts the expression to a string representation.
     * @return String representation of the expression.
     */
    virtual std::string
    ToString() const = 0;

    /**
     * @brief Gets the expression type.
     * @return The expression type.
     */
    ExpressionType
    GetExprType() const {
        return expr_type_;
    }

    /**
     * @brief Gets the operator type.
     * @return The operator type.
     */
    OpType
    GetOpType() const {
        return op_type_;
    }

protected:
    ExpressionType expr_type_;  ///< The type of this expression
    OpType op_type_;            ///< The operator type of this expression
};

/// Smart pointer type for Expression objects
using ExprPtr = std::shared_ptr<Expression>;

/**
 * @enum ComparisonOperator
 * @brief Enumeration of comparison operators.
 */
enum class ComparisonOperator {
    EQ,  ///< Equal (=)
    NE,  ///< Not equal (!=)
    GT,  ///< Greater than (>)
    LT,  ///< Less than (<)
    GE,  ///< Greater than or equal (>=)
    LE   ///< Less than or equal (<=)
};

/**
 * @enum LogicalOperator
 * @brief Enumeration of logical operators.
 */
enum class LogicalOperator {
    AND,  ///< Logical AND
    OR,   ///< Logical OR
    NOT   ///< Logical NOT
};

/**
 * @enum ArithmeticOperator
 * @brief Enumeration of arithmetic operators.
 */
enum class ArithmeticOperator {
    ADD,  ///< Addition (+)
    SUB,  ///< Subtraction (-)
    MUL,  ///< Multiplication (*)
    DIV   ///< Division (/)
};

/**
 * @brief Converts an arithmetic operator to its string representation.
 * @param op The arithmetic operator to convert.
 * @return String representation of the operator.
 * @throws std::runtime_error if the operator is not supported.
 */
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

/**
 * @brief Converts a logical operator to its string representation.
 * @param op The logical operator to convert.
 * @return String representation of the operator.
 * @throws std::runtime_error if the operator is not supported.
 */
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

/**
 * @brief Converts a comparison operator to its string representation.
 * @param op The comparison operator to convert.
 * @return String representation of the operator.
 * @throws std::runtime_error if the operator is not supported.
 */
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

/// Variant type for numeric values supporting int64_t, uint64_t, and double
using NumericValue = std::variant<int64_t, uint64_t, double>;

/// Vector of strings for string list constants
using StrList = std::vector<std::string>;

/**
 * @brief Checks if two NumericValue variants hold the same type.
 * @param lhs The left-hand side numeric value.
 * @param rhs The right-hand side numeric value.
 * @return True if both values have the same type, false otherwise.
 */
inline bool
CheckSameVType(const NumericValue&& lhs, const NumericValue&& rhs) {
    return (std::holds_alternative<double>(lhs) && std::holds_alternative<double>(rhs)) ||
           (std::holds_alternative<int64_t>(lhs) && std::holds_alternative<int64_t>(rhs)) ||
           (std::holds_alternative<uint64_t>(lhs) && std::holds_alternative<uint64_t>(rhs));
}

/**
 * @brief Addition operator for NumericValue.
 * @param lhs Left-hand side operand.
 * @param rhs Right-hand side operand.
 * @return Result of addition.
 * @throws std::runtime_error if types don't match.
 */
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

/**
 * @brief Subtraction operator for NumericValue.
 * @param lhs Left-hand side operand.
 * @param rhs Right-hand side operand.
 * @return Result of subtraction.
 * @throws std::runtime_error if types don't match.
 */
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

/**
 * @brief Multiplication operator for NumericValue.
 * @param lhs Left-hand side operand.
 * @param rhs Right-hand side operand.
 * @return Result of multiplication.
 * @throws std::runtime_error if types don't match.
 */
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

/**
 * @brief Division operator for NumericValue.
 * @param lhs Left-hand side operand.
 * @param rhs Right-hand side operand.
 * @return Result of division.
 * @throws std::runtime_error if types don't match or division by zero.
 */
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

/**
 * @brief Extracts a numeric value of type T from a NumericValue variant.
 * @tparam T The target numeric type.
 * @param value The NumericValue variant to extract from.
 * @return The extracted value of type T.
 * @throws std::runtime_error if the conversion is invalid (e.g., type mismatch, out of range).
 */
template <typename T>
T
GetNumericValue(const NumericValue& value) {
    return std::visit(
        [](auto&& arg) -> T {
            using ArgType = std::decay_t<decltype(arg)>;

            if constexpr (std::is_floating_point_v<T>) {
                if constexpr (std::is_integral_v<ArgType>) {
                    throw std::runtime_error("Cannot convert integer to floating-point");
                }
            } else {
                if constexpr (std::is_floating_point_v<ArgType>) {
                    throw std::runtime_error("Cannot convert floating-point to integer");
                } else {
                    if constexpr (std::is_unsigned_v<T>) {
                        if (arg < 0) {
                            throw std::runtime_error("Cannot convert negative value to unsigned");
                        }
                    }
                    if (arg < std::numeric_limits<T>::min() ||
                        arg > std::numeric_limits<T>::max()) {
                        throw std::runtime_error("Numeric value out of range");
                    }
                }
            }

            return static_cast<T>(arg);
        },
        value);
}

/**
 * @class FieldExpression
 * @brief Expression representing a reference to a field in the attribute schema.
 */
class FieldExpression : public Expression {
public:
    /// Smart pointer type for FieldExpression
    using ptr = std::shared_ptr<FieldExpression>;

    /**
     * @brief Constructs a FieldExpression with the given field name.
     * @param name The name of the field to reference.
     */
    explicit FieldExpression(const std::string& name)
        : Expression(ExpressionType::kFieldExpression, OpType::kNone), fieldName(name) {
    }

    std::string
    ToString() const override {
        return fieldName;
    }

    std::string fieldName;  ///< The name of the referenced field
};

/**
 * @class NumericConstant
 * @brief Expression representing a numeric constant value.
 */
class NumericConstant : public Expression {
public:
    /// Smart pointer type for NumericConstant
    using ptr = std::shared_ptr<NumericConstant>;

    /**
     * @brief Constructs a NumericConstant with the given value.
     * @param value The numeric value to store.
     */
    explicit NumericConstant(NumericValue value)
        : Expression(ExpressionType::kNumericConstant, OpType::kNone), value(value) {
    }

    std::string
    ToString() const override {
        return std::visit(
            [](auto&& arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;

                if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
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

    NumericValue value;  ///< The stored numeric value
};

/**
 * @class StringConstant
 * @brief Expression representing a string constant value.
 */
class StringConstant : public Expression {
public:
    /// Smart pointer type for StringConstant
    using ptr = std::shared_ptr<StringConstant>;

    /**
     * @brief Constructs a StringConstant with the given value.
     * @param value The string value to store.
     */
    explicit StringConstant(const std::string& value)
        : Expression(ExpressionType::kStringConstant, OpType::kNone), value(value) {
    }

    std::string
    ToString() const override {
        return '"' + value + '"';
    }

    std::string value;  ///< The stored string value
};

/**
 * @class StrListConstant
 * @brief Expression representing a list of string constants.
 */
class StrListConstant : public Expression {
public:
    /// Smart pointer type for StrListConstant
    using ptr = std::shared_ptr<StrListConstant>;

    /**
     * @brief Constructs a StrListConstant with the given string values.
     * @param values The list of string values to store.
     */
    explicit StrListConstant(StrList values)
        : Expression(ExpressionType::kStrListConstant, OpType::kNone), values(std::move(values)) {
    }

    std::string
    ToString() const override {
        std::string result = "[";
        for (uint64_t i = 0; i < values.size(); ++i) {
            if (i != 0)
                result += ", ";
            result += '"' + values[i] + '"';
        }
        return result + "]";
    }

    StrList values;  ///< The stored list of string values
};

/**
 * @class IntListConstant
 * @brief Template expression representing a list of integer constants.
 * @tparam V The integer type for the list elements.
 */
template <typename V>
class IntListConstant : public Expression {
public:
    /// Smart pointer type for IntListConstant
    using ptr = std::shared_ptr<IntListConstant>;

    /**
     * @brief Constructs an IntListConstant with the given integer values.
     * @param values The list of integer values to store.
     */
    explicit IntListConstant(std::vector<V> values)
        : Expression(ExpressionType::kIntListConstant, OpType::kNone), values(std::move(values)) {
    }

    std::string
    ToString() const override {
        std::string result = "[";
        for (uint64_t i = 0; i < values.size(); ++i) {
            if (i != 0)
                result += ", ";
            result += std::to_string(values[i]);
        }
        return result + "]";
    }

    std::vector<V> values;  ///< The stored list of integer values
};

/**
 * @class ArithmeticExpression
 * @brief Expression representing an arithmetic operation between two operands.
 */
class ArithmeticExpression : public Expression {
public:
    /**
     * @brief Constructs an ArithmeticExpression with the given operands and operator.
     * @param left Left operand expression.
     * @param op Arithmetic operator.
     * @param right Right operand expression.
     */
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

    ExprPtr left;           ///< Left operand
    ArithmeticOperator op;  ///< Arithmetic operator
    ExprPtr right;          ///< Right operand
};

/**
 * @class ComparisonExpression
 * @brief Expression representing a comparison between two operands.
 */
class ComparisonExpression : public Expression {
public:
    /**
     * @brief Constructs a ComparisonExpression with the given operands and operator.
     * @param left Left operand expression.
     * @param op Comparison operator.
     * @param right Right operand expression.
     */
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

    ExprPtr left;           ///< Left operand
    ComparisonOperator op;  ///< Comparison operator
    ExprPtr right;          ///< Right operand
};

/**
 * @class IntListExpression
 * @brief Expression representing integer list membership (IN/NOT IN) operation.
 */
class IntListExpression : public Expression {
public:
    /**
     * @brief Constructs an IntListExpression for IN or NOT IN operation.
     * @param field The field expression to check.
     * @param is_not_in True for NOT IN, false for IN.
     * @param values The list of integer values.
     */
    IntListExpression(ExprPtr field, const bool is_not_in, ExprPtr values)
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

    ExprPtr field;   ///< The field expression to check
    bool is_not_in;  ///< True for NOT IN, false for IN
    ExprPtr values;  ///< The list of integer values
};

/**
 * @class StrListExpression
 * @brief Expression representing string list membership (IN/NOT IN) operation.
 */
class StrListExpression : public Expression {
public:
    /**
     * @brief Constructs a StrListExpression for IN or NOT IN operation.
     * @param field The field expression to check.
     * @param is_not_in True for NOT IN, false for IN.
     * @param values The list of string values.
     */
    StrListExpression(ExprPtr field, const bool is_not_in, ExprPtr values)
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

    ExprPtr field;   ///< The field expression to check
    bool is_not_in;  ///< True for NOT IN, false for IN
    ExprPtr values;  ///< The list of string values
};

/**
 * @class LogicalExpression
 * @brief Expression representing a logical operation (AND/OR) between two operands.
 */
class LogicalExpression : public Expression {
public:
    /**
     * @brief Constructs a LogicalExpression with the given operands and operator.
     * @param left Left operand expression.
     * @param op Logical operator (AND/OR).
     * @param right Right operand expression.
     */
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

    ExprPtr left;        ///< Left operand
    LogicalOperator op;  ///< Logical operator
    ExprPtr right;       ///< Right operand
};

/**
 * @class NotExpression
 * @brief Expression representing a logical NOT operation on a single operand.
 */
class NotExpression : public Expression {
public:
    /**
     * @brief Constructs a NotExpression with the given operand.
     * @param expr The expression to negate.
     */
    explicit NotExpression(ExprPtr expr)
        : Expression(ExpressionType::kNotExpression, OpType::kUnary), expr(std::move(expr)) {
    }

    std::string
    ToString() const override {
        return "! (" + expr->ToString() + ")";
    }

    ExprPtr expr;  ///< The expression to negate
};
}  // namespace vsag
