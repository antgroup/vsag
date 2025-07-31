
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

#include <strings.h>

#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
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
    kFunctionExpression,
    kRegionFilterExpression,
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

using NumericValue = std::variant<int64_t, uint64_t, double>;
using StrList = std::vector<std::string>;

inline bool
CheckSameVType(const NumericValue&& lhs, const NumericValue&& rhs) {
    return (std::holds_alternative<double>(lhs) && std::holds_alternative<double>(rhs)) ||
           (std::holds_alternative<int64_t>(lhs) && std::holds_alternative<int64_t>(rhs)) ||
           (std::holds_alternative<uint64_t>(lhs) && std::holds_alternative<uint64_t>(rhs));
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

template <typename V>
class IntListConstant : public Expression {
public:
    using ptr = std::shared_ptr<IntListConstant>;

    explicit IntListConstant(std::vector<V> values)
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

    std::vector<V> values;
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

    ExprPtr field;
    bool is_not_in;  // true for NOT IN, false for IN
    ExprPtr values;
};

// List expression
class StrListExpression : public Expression {
public:
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

    ExprPtr field;
    bool is_not_in;  // true for NOT IN, false for IN
    ExprPtr values;
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

class FunctionExpression : public Expression {
public:
    enum class ArgTypes {
        kInnerId,
        kAttribute,
        kInt16,
        kUint16,
        kInt32,
        kUInt32,
        kInt64,
        kUInt64,
        kFloat,
        kFloat64,
        kString,
    };

    static std::string
    ArgTypesToString(ArgTypes arg_type) {
        switch (arg_type) {
            case ArgTypes::kInnerId:
                return "inner_id";
            case ArgTypes::kAttribute:
                return "attribute";
            case ArgTypes::kInt16:
                return "int16";
            case ArgTypes::kUint16:
                return "uint16";
            case ArgTypes::kInt32:
                return "int32";
            case ArgTypes::kUInt32:
                return "uint32";
            case ArgTypes::kInt64:
                return "int64";
            case ArgTypes::kUInt64:
                return "uint64";
            case ArgTypes::kFloat:
                return "float";
            case ArgTypes::kFloat64:
                return "float64";
            case ArgTypes::kString:
                return "string";
            default:
                throw std::runtime_error("unknown arg type");
        }
    }
    using ArgValueType = std::variant<std::string,
                                      std::int16_t,
                                      std::uint16_t,
                                      std::int32_t,
                                      std::uint32_t,
                                      std::int64_t,
                                      std::uint64_t,
                                      float,
                                      double>;

    explicit FunctionExpression(const std::string& name,
                                const std::vector<std::string>& args,
                                const std::vector<std::string>& types)
        : Expression(ExpressionType::kFunctionExpression, OpType::kNone), function_name(name) {
        if (args.size() != types.size()) {
            throw std::runtime_error(name +
                                     " function expression args size not equal to types size");
        }

        for (size_t i = 0; i < args.size(); ++i) {
            auto& arg_type = types[i];
            if (arg_type.empty()) {
                continue;
            }
            if (arg_type == "inner_id") {
                arg_types.emplace_back(ArgTypes::kInnerId);
                if (args[i] != "placeholder") {
                    throw std::runtime_error("inner_id arg must be placeholder");
                }
                arg_values.emplace_back("placeholder");
            } else if (arg_type == "attribute") {
                arg_types.emplace_back(ArgTypes::kAttribute);
                arg_values.emplace_back(args[i]);
            } else if (!strcasecmp(arg_type.c_str(), "int16")) {
                arg_types.emplace_back(ArgTypes::kInt16);
                auto value = std::stoll(args[i]);
                if (value < std::numeric_limits<std::int16_t>::min() ||
                    value > std::numeric_limits<std::int16_t>::max()) {
                    throw std::runtime_error("Numeric value out of range");
                }
                arg_values.emplace_back(static_cast<int16_t>(value));
            } else if (!strcasecmp(arg_type.c_str(), "uint16")) {
                arg_types.emplace_back(ArgTypes::kUint16);
                arg_values.emplace_back(GetNumericValue<std::uint16_t>(args[i]));
            } else if (!strcasecmp(arg_type.c_str(), "int32")) {
                arg_types.emplace_back(ArgTypes::kInt32);
                arg_values.emplace_back(std::stoi(args[i]));
            } else if (!strcasecmp(arg_type.c_str(), "uint32")) {
                arg_types.emplace_back(ArgTypes::kUInt32);
                arg_values.emplace_back(GetNumericValue<std::uint32_t>(args[i]));
            } else if (!strcasecmp(arg_type.c_str(), "int64")) {
                arg_types.emplace_back(ArgTypes::kInt64);
                arg_values.emplace_back(static_cast<std::int64_t>(std::stoll(args[i])));
            } else if (!strcasecmp(arg_type.c_str(), "uint64")) {
                arg_types.emplace_back(ArgTypes::kUInt64);
                arg_values.emplace_back(static_cast<std::uint64_t>(std::stoull(args[i])));
            } else if (!strcasecmp(arg_type.c_str(), "float")) {
                arg_types.emplace_back(ArgTypes::kFloat);
                arg_values.emplace_back(std::stof(args[i]));
            } else if (!strcasecmp(arg_type.c_str(), "float64")) {
                arg_types.emplace_back(ArgTypes::kFloat64);
                arg_values.emplace_back(std::stod(args[i]));
            } else if (!strcasecmp(arg_type.c_str(), "string")) {
                arg_types.emplace_back(ArgTypes::kString);
                arg_values.emplace_back(args[i]);
            } else {
                throw std::runtime_error("Unknown argument type: " + arg_type);
            }
        }
    }

    template <typename T>
    [[nodiscard]] std::optional<T>
    GetArg(size_t index) const {
        if (index >= arg_values.size()) {
            return std::nullopt;
        }
        auto& arg_value = arg_values[index];
        return std::visit(
            [](auto&& arg) -> std::optional<T> {
                using V = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<V, T>) {
                    return static_cast<T>(arg);
                }
                return std::nullopt;
            },
            arg_value);
    }

    [[nodiscard]] std::string
    ToString() const override {
        std::string result = "FUNCTION (" + function_name + ",\"";
        for (size_t i = 0; i < arg_values.size(); ++i) {
            if (i != 0)
                result += "|";
            auto& arg_value = arg_values[i];
            std::visit(
                [&result](auto&& arg) {
                    using V = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<V, std::string>) {
                        result += arg;
                    } else {
                        result += std::to_string(arg);
                    }
                },
                arg_value);
        }
        result += "\",\"";
        for (size_t i = 0; i < arg_types.size(); ++i) {
            if (i != 0)
                result += "|";
            result += ArgTypesToString(arg_types[i]);
        }

        return result + "\")";
    }

    std::string function_name;
    std::vector<ArgValueType> arg_values;
    std::vector<ArgTypes> arg_types;

private:
    template <typename T>
    [[nodiscard]] static T
    GetNumericValue(const std::string& str) {
        auto value = std::stoull(str);
        if (value > std::numeric_limits<T>::max()) {
            throw std::runtime_error("Numeric value out of range");
        }
        return static_cast<T>(value);
    }
};

class RegionFilterExpression : public Expression {
public:
    explicit RegionFilterExpression(ExprPtr region_type,
                                    ExprPtr region_list,
                                    ExprPtr residence_tag_list,
                                    ExprPtr arg0,
                                    ExprPtr arg1,
                                    ExprPtr arg2)
        : Expression(ExpressionType::kRegionFilterExpression, OpType::kNone),
          region_type(std::move(region_type)),
          region_list(std::move(region_list)),
          residence_tag_list(std::move(residence_tag_list)),
          arg0(std::move(arg0)),
          arg1(std::move(arg1)),
          arg2(std::move(arg2)) {
    }

    std::string
    ToString() const override {
        return "region_filter (" + region_type->ToString() + ", " + region_list->ToString() + ", " +
               residence_tag_list->ToString() + ", " + arg0->ToString() + ", " + arg1->ToString() +
               ", " + arg2->ToString() + ")";
    }

    ExprPtr region_type;
    ExprPtr region_list;
    ExprPtr residence_tag_list;
    ExprPtr arg0;
    ExprPtr arg1;
    ExprPtr arg2;
};
}  // namespace vsag
