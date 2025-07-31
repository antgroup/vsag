
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

#include "expression_visitor.h"

namespace vsag {

static ComparisonOperator
trans_string_to_comparison_operator(const std::string& op) {
    if (op == ">=") {
        return ComparisonOperator::GE;
    }
    if (op == "<=") {
        return ComparisonOperator::LE;
    }
    if (op == ">") {
        return ComparisonOperator::GT;
    }
    if (op == "<") {
        return ComparisonOperator::LT;
    }
    if (op == "=") {
        return ComparisonOperator::EQ;
    }
    if (op == "!=") {
        return ComparisonOperator::NE;
    }
    throw std::runtime_error("Unknown comparison operator: " + op);
}

// Helper function to convert string to logical operator
static LogicalOperator
trans_string_to_logical_operator(const std::string& op) {
    if (op == "AND" or op == "and" or op == "&&") {
        return LogicalOperator::AND;
    }
    if (op == "OR" or op == "or" or op == "||") {
        return LogicalOperator::OR;
    }
    throw std::runtime_error("Unknown logical operator: " + op);
}

// Helper function to convert string to arithmetic operator
static ArithmeticOperator
trans_string_to_arithmetic_operator(const std::string& op) {
    if (op == "+") {
        return ArithmeticOperator::ADD;
    }
    if (op == "-") {
        return ArithmeticOperator::SUB;
    }
    if (op == "*") {
        return ArithmeticOperator::MUL;
    }
    if (op == "/") {
        return ArithmeticOperator::DIV;
    }
    throw std::runtime_error("Unknown arithmetic operator: " + op);
}

static NumericValue
visit_int_token(const std::string& int_token) {
    if (not int_token.empty() and int_token.c_str()[0] == '-') {
        int64_t val = std::stoll(int_token);
        return val;
    }
    uint64_t uval = std::stoull(int_token);
    if (uval > static_cast<uint64_t>(INT64_MAX)) {
        return uval;
    }
    return static_cast<int64_t>(uval);
}

static std::any
build_int_list_ptr(const int signed_cnt, const std::vector<NumericValue>& values) {
    if (signed_cnt == values.size()) {
        std::vector<int64_t> new_values;
        new_values.reserve(values.size());
        for (const auto& v : values) {
            new_values.emplace_back(std::get<int64_t>(v));
        }
        auto int_list_ptr = std::make_shared<IntListConstant<int64_t>>(std::move(new_values));
        return std::make_any<ExprPtr>(int_list_ptr);
    }
    std::vector<uint64_t> new_values;
    new_values.reserve(values.size());
    for (const auto& v : values) {
        new_values.emplace_back(GetNumericValue<uint64_t>(v));
    }
    auto int_list_ptr = std::make_shared<IntListConstant<uint64_t>>(std::move(new_values));
    return std::make_any<ExprPtr>(int_list_ptr);
}

static std::vector<std::string_view>
string_view_split(std::string_view str, char delim) {
    std::vector<std::string_view> result;
    size_t start = 0;
    size_t end = str.find(delim);

    while (end != std::string_view::npos) {
        result.emplace_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delim, start);
    }
    result.emplace_back(str.substr(start));
    return result;
}
FCExpressionVisitor::FCExpressionVisitor(AttrTypeSchema* schema) : schema_(schema){};

std::any
FCExpressionVisitor::visitFilter_condition(FCParser::Filter_conditionContext* ctx) {
    return visit(ctx->expr());
}

std::any
FCExpressionVisitor::visitParenExpr(FCParser::ParenExprContext* ctx) {
    return visit(ctx->expr());
}

std::any
FCExpressionVisitor::visitNotExpr(FCParser::NotExprContext* ctx) {
    auto expr = std::any_cast<ExprPtr>(visit(ctx->expr()));
    return std::make_any<ExprPtr>(std::make_shared<NotExpression>(expr));
}

std::any
FCExpressionVisitor::visitLogicalExpr(FCParser::LogicalExprContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->left));
    auto right = std::any_cast<ExprPtr>(visit(ctx->right));
    return std::make_any<ExprPtr>(std::make_shared<LogicalExpression>(
        std::move(left), trans_string_to_logical_operator(ctx->op->getText()), std::move(right)));
}

std::any
FCExpressionVisitor::visitCompExpr(FCParser::CompExprContext* ctx) {
    return visit(ctx->comparison());
}

std::any
FCExpressionVisitor::visitIntPipeListExpr(FCParser::IntPipeListExprContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    ExprPtr right = nullptr;
    if (schema_ != nullptr and is_string_type(left)) {
        right = std::any_cast<ExprPtr>(visitInt_pipe_list(ctx->int_pipe_list(), true));
        return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }
    right = std::any_cast<ExprPtr>(visit(ctx->int_pipe_list()));
    return std::make_any<ExprPtr>(std::make_shared<IntListExpression>(
        std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
}

std::any
FCExpressionVisitor::visitStrPipeListExpr(FCParser::StrPipeListExprContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    if (schema_ != nullptr and not is_string_type(left)) {
        throw std::runtime_error("attribute value type is not string type");
    }
    auto right = std::any_cast<ExprPtr>(visit(ctx->str_pipe_list()));
    return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
        std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
}

std::any
FCExpressionVisitor::visitIntListExpr(FCParser::IntListExprContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    if (schema_ != nullptr and is_string_type(left)) {
        auto right = std::any_cast<ExprPtr>(visitInt_value_list(ctx->int_value_list(), true));
        return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }
    auto right = std::any_cast<ExprPtr>(visit(ctx->int_value_list()));
    return std::make_any<ExprPtr>(std::make_shared<IntListExpression>(
        std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
}

std::any
FCExpressionVisitor::visitStrListExpr(FCParser::StrListExprContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    if (schema_ != nullptr and not is_string_type(left)) {
        throw std::runtime_error("attribute value type is not string type");
    }
    auto right = std::any_cast<ExprPtr>(visit(ctx->str_value_list()));
    return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
        std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
}

std::any
FCExpressionVisitor::visitNumericComparison(FCParser::NumericComparisonContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_expr()));
    auto right = std::any_cast<ExprPtr>(visit(ctx->numeric()));
    return std::make_any<ExprPtr>(std::make_shared<ComparisonExpression>(
        std::move(left),
        trans_string_to_comparison_operator(ctx->op->getText()),
        std::move(right)));
}

std::any
FCExpressionVisitor::visitStringComparison(FCParser::StringComparisonContext* ctx) {
    auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    if (schema_ != nullptr and not is_string_type(left)) {
        throw std::runtime_error("attribute value type is not string type");
    }
    auto str = ctx->STRING() != nullptr ? ctx->STRING()->getText() : ctx->INT_STRING()->getText();
    auto right = std::make_shared<StringConstant>(str.substr(1, str.size() - 2));
    return std::make_any<ExprPtr>(std::make_shared<ComparisonExpression>(
        std::move(left),
        trans_string_to_comparison_operator(ctx->op->getText()),
        std::move(right)));
}

std::any
FCExpressionVisitor::visitParenFieldExpr(FCParser::ParenFieldExprContext* ctx) {
    return visit(ctx->field_expr());
}

std::any
FCExpressionVisitor::visitFieldRef(FCParser::FieldRefContext* ctx) {
    auto field_ref = std::any_cast<ExprPtr>(visit(ctx->field_name()));
    if (schema_ != nullptr and is_string_type(field_ref)) {
        throw std::runtime_error("attribute value type is not numeric type");
    }
    return field_ref;
}

std::any
FCExpressionVisitor::visitArithmeticExpr(FCParser::ArithmeticExprContext* ctx) {
    // Handle parenthesized expressions
    if (ctx->children.size() == 3 and ctx->children[0]->getText() == "(") {
        return visit(ctx->children[1]);
    }

    // Handle arithmetic operations
    if (ctx->op != nullptr) {
        auto left = std::any_cast<ExprPtr>(visit(ctx->children[0]));
        auto right = std::any_cast<ExprPtr>(visit(ctx->children[2]));
        auto op = trans_string_to_arithmetic_operator(ctx->op->getText());
        return std::make_any<ExprPtr>(std::make_shared<ArithmeticExpression>(left, op, right));
    }
    // Handle simple field names
    if (auto* field_name_ctx = dynamic_cast<FCParser::Field_nameContext*>(ctx->children[0])) {
        return visit(field_name_ctx);
    }

    // Handle numeric literals in arithmetic expressions
    if (auto* numeric_ctx = dynamic_cast<FCParser::NumericContext*>(ctx->children[0])) {
        return visit(numeric_ctx);
    }

    throw std::runtime_error("Unsupported field expression: " + ctx->getText());
}

std::any
FCExpressionVisitor::visitNumericConst(FCParser::NumericConstContext* ctx) {
    return visit(ctx->numeric());
}
std::any
FCExpressionVisitor::visitFunctionExpr(FCParser::FunctionExprContext* ctx) {
    auto func_name = ctx->function_name()->ID()->getText();
    auto args = std::any_cast<ExprPtr>(visitArg_pipe_list(ctx->arg_pipe_list()));
    auto types = std::any_cast<ExprPtr>(visitStr_pipe_list(ctx->str_pipe_list()));
    auto args_ptr = std::dynamic_pointer_cast<StrListConstant>(args);
    auto types_ptr = std::dynamic_pointer_cast<StrListConstant>(types);
    auto function_expr =
        std::make_shared<FunctionExpression>(func_name, args_ptr->values, types_ptr->values);
    return std::make_any<ExprPtr>(function_expr);
}
std::any
FCExpressionVisitor::visitRegionFilterExpr(FCParser::RegionFilterExprContext* ctx) {
    auto region_filter_expr = std::make_shared<RegionFilterExpression>(
        std::any_cast<ExprPtr>(visitField_name(ctx->field_name(0))),
        std::any_cast<ExprPtr>(visitField_name(ctx->field_name(1))),
        std::any_cast<ExprPtr>(visitField_name(ctx->field_name(2))),
        std::any_cast<ExprPtr>(visitInt_pipe_list(ctx->int_pipe_list(0))),
        std::any_cast<ExprPtr>(visitInt_pipe_list(ctx->int_pipe_list(1))),
        std::any_cast<ExprPtr>(visitInt_pipe_list(ctx->int_pipe_list(2))));
    return std::make_any<ExprPtr>(region_filter_expr);
}

std::any
FCExpressionVisitor::visitArg_pipe_list(FCParser::Arg_pipe_listContext* ctx) {
    if (ctx->str_pipe_list()) {
        return visitStr_pipe_list(ctx->str_pipe_list());
    }
    if (ctx->int_pipe_list()) {
        return visitInt_pipe_list(ctx->int_pipe_list(), true);
    }
    StrList values;
    auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
    return std::make_any<ExprPtr>(str_list_ptr);
}

std::any
FCExpressionVisitor::visitStr_value_list(FCParser::Str_value_listContext* ctx) {
    StrList values;
    for (auto* str_token : ctx->STRING()) {
        auto str = str_token->getText();
        // Remove quotes
        str = str.substr(1, str.size() - 2);
        values.emplace_back(str);
    }

    for (auto* str_token : ctx->INT_STRING()) {
        auto str = str_token->getText();
        // Remove quotes
        str = str.substr(1, str.size() - 2);
        values.emplace_back(str);
    }

    auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
    return std::make_any<ExprPtr>(str_list_ptr);
}

std::any
FCExpressionVisitor::visitInt_value_list(FCParser::Int_value_listContext* ctx) {
    std::vector<NumericValue> values;
    int signed_cnt = 0;
    for (auto* int_token : ctx->INTEGER()) {
        auto numeric_value = visit_int_token(int_token->getText());
        if (std::holds_alternative<int64_t>(numeric_value)) {
            signed_cnt++;
        }
        values.emplace_back(numeric_value);
    }
    return build_int_list_ptr(signed_cnt, values);
}

std::any
FCExpressionVisitor::visitInt_value_list(FCParser::Int_value_listContext* ctx,
                                         const bool is_string_type) {
    if (is_string_type) {
        StrList values;
        for (auto* int_token : ctx->INTEGER()) {
            values.emplace_back(int_token->getText());
        }
        auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
        return std::make_any<ExprPtr>(str_list_ptr);
    }
    return visitInt_value_list(ctx);
}

std::any
FCExpressionVisitor::visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx) {
    std::vector<NumericValue> values;
    int signed_cnt = 0;
    if (ctx->INT_STRING() != nullptr and ctx->INT_STRING()->getText().size() >= 2) {
        auto str = ctx->INT_STRING()->getText();
        str = str.substr(1, str.size() - 2);
        auto numeric_value = visit_int_token(str);
        if (std::holds_alternative<int64_t>(numeric_value)) {
            signed_cnt++;
        }
        values.emplace_back(numeric_value);
    } else if (ctx->PIPE_INT_STR() != nullptr and ctx->PIPE_INT_STR()->getText().size() >= 2) {
        auto str = ctx->PIPE_INT_STR()->getText();
        str = str.substr(1, str.size() - 2);
        const auto& result_view = string_view_split(str, '|');
        for (const auto& s : result_view) {
            auto numeric_value = visit_int_token(s.data());
            if (std::holds_alternative<int64_t>(numeric_value)) {
                signed_cnt++;
            }
            values.emplace_back(numeric_value);
        }
    }
    return build_int_list_ptr(signed_cnt, values);
}

std::any
FCExpressionVisitor::visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx,
                                        const bool is_string_type) {
    if (is_string_type) {
        StrList values;
        if (ctx->INT_STRING() != nullptr and ctx->INT_STRING()->getText().size() >= 2) {
            auto str = ctx->INT_STRING()->getText();
            str = str.substr(1, str.size() - 2);
            values.emplace_back(str);
        } else if (ctx->PIPE_INT_STR() != nullptr and ctx->PIPE_INT_STR()->getText().size() >= 2) {
            auto str = ctx->PIPE_INT_STR()->getText();
            str = str.substr(1, str.size() - 2);
            const auto& result_view = string_view_split(str, '|');
            for (const auto& s : result_view) {
                values.emplace_back(s);
            }
        }
        auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
        return std::make_any<ExprPtr>(str_list_ptr);
    }
    return visitInt_pipe_list(ctx);
}

std::any
FCExpressionVisitor::visitStr_pipe_list(FCParser::Str_pipe_listContext* ctx) {
    StrList values;
    if (ctx->STRING() != nullptr and ctx->STRING()->getText().size() >= 2) {
        auto str = ctx->STRING()->getText();
        str = str.substr(1, str.size() - 2);
        values.emplace_back(str);
    } else if (ctx->PIPE_STR_STR() != nullptr and ctx->PIPE_STR_STR()->getText().size() >= 2) {
        auto str = ctx->PIPE_STR_STR()->getText();
        str = str.substr(1, str.size() - 2);
        const auto& result_view = string_view_split(str, '|');
        for (const auto& s : result_view) {
            values.emplace_back(s);
        }
    }
    auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
    return std::make_any<ExprPtr>(str_list_ptr);
}

std::any
FCExpressionVisitor::visitField_name(FCParser::Field_nameContext* ctx) {
    return std::make_any<ExprPtr>(std::make_shared<FieldExpression>(ctx->getText()));
}

std::any
FCExpressionVisitor::visitNumeric(FCParser::NumericContext* ctx) {
    if (ctx->INTEGER() != nullptr) {
        return std::make_any<ExprPtr>(
            std::make_shared<NumericConstant>(visit_int_token(ctx->INTEGER()->getText())));
    }
    if (ctx->FLOAT() != nullptr) {
        return std::make_any<ExprPtr>(
            std::make_shared<NumericConstant>(std::stod(ctx->FLOAT()->getText())));
    }
    throw std::runtime_error("Invalid numeric value: " + ctx->getText());
}

bool
FCExpressionVisitor::is_string_type(const ExprPtr& expr) {
    if (auto field_expr = std::dynamic_pointer_cast<FieldExpression>(expr); field_expr != nullptr) {
        return schema_->GetTypeOfField(field_expr->fieldName) == STRING;
    }
    throw std::runtime_error("Invalid field expression: " + expr->ToString());
}
}  // namespace vsag
