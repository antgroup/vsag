
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

#include <antlr4-autogen/FCBaseVisitor.h>
#include <antlr4-runtime/antlr4-runtime.h>
#include <fmt/chrono.h>
#include <fmt/format.h>

#include <any>
#include <cstdio>

#define EOF (-1)
#include <nlohmann/json.hpp>
#undef EOF

#include "attr_type_schema.h"
#include "expression.h"

namespace vsag {
class FCErrorListener final : public antlr4::BaseErrorListener {
public:
    FCErrorListener(const std::string& input) : input_(input) {
    }

    void
    syntaxError(antlr4::Recognizer* recognizer,
                antlr4::Token* offendingSymbol,
                size_t line,
                size_t charPositionInLine,
                const std::string& msg,
                std::exception_ptr e) override {
        std::string offendingText;
        if (offendingSymbol) {
            offendingText = offendingSymbol->getText();
        }
        throw std::runtime_error(
            fmt::format("Syntax error in filter condition, line({}), charPositionInLine({}), "
                        "msg({}), offendingText({}), input({})",
                        line,
                        charPositionInLine,
                        msg,
                        offendingText,
                        input_));
    }

private:
    std::string input_;
};

static ComparisonOperator
ToComparisonOperator(const std::string& op) {
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
ToLogicalOp(const std::string& op) {
    if (op == "AND" || op == "and" || op == "&&") {
        return LogicalOperator::AND;
    }
    if (op == "OR" || op == "or" || op == "||") {
        return LogicalOperator::OR;
    }
    throw std::runtime_error("Unknown logical operator: " + op);
}

// Helper function to convert string to arithmetic operator
static ArithmeticOperator
ToArithmeticOp(const std::string& op) {
    if (op == "+")
        return ArithmeticOperator::ADD;
    if (op == "-")
        return ArithmeticOperator::SUB;
    if (op == "*")
        return ArithmeticOperator::MUL;
    if (op == "/")
        return ArithmeticOperator::DIV;
    throw std::runtime_error("Unknown arithmetic operator: " + op);
}

static NumericValue
VisitIntToken(const std::string& int_token) {
    if (!int_token.empty() && int_token.c_str()[0] == '-') {
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
BuildIntListPtr(const int signed_cnt, const std::vector<NumericValue>& values) {
    if (signed_cnt == values.size()) {
        std::vector<int64_t> new_values;
        for (auto& v : values) {
            new_values.emplace_back(std::get<int64_t>(v));
        }
        auto int_list_ptr = std::make_shared<IntListConstant<int64_t>>(std::move(new_values));
        return std::make_any<ExprPtr>(int_list_ptr);
    }
    std::vector<uint64_t> new_values;
    for (auto& v : values) {
        new_values.emplace_back(GetNumericValue<uint64_t>(v));
    }
    auto int_list_ptr = std::make_shared<IntListConstant<uint64_t>>(std::move(new_values));
    return std::make_any<ExprPtr>(int_list_ptr);
}

class FCExpressionVisitor final : public FCBaseVisitor {
public:
    explicit FCExpressionVisitor(AttrTypeSchema* schema) : schema_(schema) {
    }

    std::any
    visitFilter_condition(FCParser::Filter_conditionContext* ctx) override {
        return visit(ctx->expr());
    }

    std::any
    visitParenExpr(FCParser::ParenExprContext* ctx) override {
        return visit(ctx->expr());
    }

    std::any
    visitNotExpr(FCParser::NotExprContext* ctx) override {
        auto expr = std::any_cast<ExprPtr>(visit(ctx->expr()));
        return std::make_any<ExprPtr>(std::make_shared<NotExpression>(expr));
    }

    std::any
    visitLogicalExpr(FCParser::LogicalExprContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->left));
        auto right = std::any_cast<ExprPtr>(visit(ctx->right));
        return std::make_any<ExprPtr>(std::make_shared<LogicalExpression>(
            std::move(left), ToLogicalOp(ctx->op->getText()), std::move(right)));
    }

    std::any
    visitCompExpr(FCParser::CompExprContext* ctx) override {
        return visit(ctx->comparison());
    }

    std::any
    visitIntPipeListExpr(FCParser::IntPipeListExprContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        ExprPtr right = nullptr;
        if (schema_ != nullptr && IsStringType(left)) {
            right = std::any_cast<ExprPtr>(visitInt_pipe_list(ctx->int_pipe_list(), true));
            return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
                std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
        }
        right = std::any_cast<ExprPtr>(visit(ctx->int_pipe_list()));
        return std::make_any<ExprPtr>(std::make_shared<IntListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }

    std::any
    visitStrPipeListExpr(FCParser::StrPipeListExprContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        if (schema_ != nullptr && not IsStringType(left)) {
            throw std::runtime_error("attribute value type is not string type");
        }
        auto right = std::any_cast<ExprPtr>(visit(ctx->str_pipe_list()));
        return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }

    std::any
    visitIntListExpr(FCParser::IntListExprContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        if (schema_ != nullptr && IsStringType(left)) {
            auto right = std::any_cast<ExprPtr>(visitInt_value_list(ctx->int_value_list(), true));
            return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
                std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
        }
        auto right = std::any_cast<ExprPtr>(visit(ctx->int_value_list()));
        return std::make_any<ExprPtr>(std::make_shared<IntListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }

    std::any
    visitStrListExpr(FCParser::StrListExprContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        if (schema_ != nullptr && not IsStringType(left)) {
            throw std::runtime_error("attribute value type is not string type");
        }
        auto right = std::any_cast<ExprPtr>(visit(ctx->str_value_list()));
        return std::make_any<ExprPtr>(std::make_shared<StrListExpression>(
            std::move(left), ctx->NOT_IN() != nullptr, std::move(right)));
    }

    std::any
    visitNumericComparison(FCParser::NumericComparisonContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_expr()));
        auto right = std::any_cast<ExprPtr>(visit(ctx->numeric()));
        return std::make_any<ExprPtr>(std::make_shared<ComparisonExpression>(
            std::move(left), ToComparisonOperator(ctx->op->getText()), std::move(right)));
    }

    std::any
    visitStringComparison(FCParser::StringComparisonContext* ctx) override {
        auto left = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        if (schema_ != nullptr && not IsStringType(left)) {
            throw std::runtime_error("attribute value type is not string type");
        }
        auto str =
            ctx->STRING() != nullptr ? ctx->STRING()->getText() : ctx->INT_STRING()->getText();
        auto right = std::make_shared<StringConstant>(str.substr(1, str.size() - 2));
        return std::make_any<ExprPtr>(std::make_shared<ComparisonExpression>(
            std::move(left), ToComparisonOperator(ctx->op->getText()), std::move(right)));
    }

    std::any
    visitParenFieldExpr(FCParser::ParenFieldExprContext* ctx) override {
        return visit(ctx->field_expr());
    }

    std::any
    visitFieldRef(FCParser::FieldRefContext* ctx) override {
        auto field_ref = std::any_cast<ExprPtr>(visit(ctx->field_name()));
        if (schema_ != nullptr && IsStringType(field_ref)) {
            throw std::runtime_error("attribute value type is not numeric type");
        }
        return field_ref;
    }

    std::any
    visitArithmeticExpr(FCParser::ArithmeticExprContext* ctx) override {
        // Handle parenthesized expressions
        if (ctx->children.size() == 3 && ctx->children[0]->getText() == "(") {
            return visit(ctx->children[1]);
        }

        // Handle arithmetic operations
        if (ctx->op) {
            auto left = std::any_cast<ExprPtr>(visit(ctx->children[0]));
            auto right = std::any_cast<ExprPtr>(visit(ctx->children[2]));
            auto op = ToArithmeticOp(ctx->op->getText());
            return std::make_any<ExprPtr>(std::make_shared<ArithmeticExpression>(left, op, right));
        }
        // Handle simple field names
        if (auto fieldNameCtx = dynamic_cast<FCParser::Field_nameContext*>(ctx->children[0])) {
            return visit(fieldNameCtx);
        }

        // Handle numeric literals in arithmetic expressions
        if (auto numericCtx = dynamic_cast<FCParser::NumericContext*>(ctx->children[0])) {
            return visit(numericCtx);
        }

        throw std::runtime_error("Unsupported field expression: " + ctx->getText());
    }

    std::any
    visitNumericConst(FCParser::NumericConstContext* ctx) override {
        return visit(ctx->numeric());
    }

    std::any
    visitStr_value_list(FCParser::Str_value_listContext* ctx) override {
        StrList values;
        for (auto strToken : ctx->STRING()) {
            auto str = strToken->getText();
            // Remove quotes
            str = str.substr(1, str.size() - 2);
            values.emplace_back(str);
        }

        for (auto strToken : ctx->INT_STRING()) {
            auto str = strToken->getText();
            // Remove quotes
            str = str.substr(1, str.size() - 2);
            values.emplace_back(str);
        }

        auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
        return std::make_any<ExprPtr>(str_list_ptr);
    }

    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx) override {
        std::vector<NumericValue> values;
        int signed_cnt = 0;
        for (auto int_token : ctx->INTEGER()) {
            auto numeric_value = VisitIntToken(int_token->getText());
            if (std::holds_alternative<int64_t>(numeric_value)) {
                signed_cnt++;
            }
            values.emplace_back(numeric_value);
        }
        return BuildIntListPtr(signed_cnt, values);
    }

    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx, const bool is_string_type) {
        if (is_string_type) {
            StrList values;
            for (auto int_token : ctx->INTEGER()) {
                values.emplace_back(int_token->getText());
            }
            auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
            return std::make_any<ExprPtr>(str_list_ptr);
        }
        return visitInt_value_list(ctx);
    }

    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx) override {
        std::vector<NumericValue> values;
        int signed_cnt = 0;
        if (ctx->INT_STRING() && ctx->INT_STRING()->getText().size() >= 2) {
            auto str = ctx->INT_STRING()->getText();
            str = str.substr(1, str.size() - 2);
            auto numeric_value = VisitIntToken(str);
            if (std::holds_alternative<int64_t>(numeric_value)) {
                signed_cnt++;
            }
            values.emplace_back(numeric_value);
        } else if (ctx->PIPE_INT_STR() && ctx->PIPE_INT_STR()->getText().size() >= 2) {
            auto str = ctx->PIPE_INT_STR()->getText();
            str = str.substr(1, str.size() - 2);
            const auto& result_view = StrViewSplit(str, '|');
            for (auto& s : result_view) {
                auto numeric_value = VisitIntToken(s.data());
                if (std::holds_alternative<int64_t>(numeric_value)) {
                    signed_cnt++;
                }
                values.emplace_back(numeric_value);
            }
        }
        return BuildIntListPtr(signed_cnt, values);
    }

    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx, const bool is_string_type) {
        if (is_string_type) {
            StrList values;
            if (ctx->INT_STRING() && ctx->INT_STRING()->getText().size() >= 2) {
                auto str = ctx->INT_STRING()->getText();
                str = str.substr(1, str.size() - 2);
                values.emplace_back(str);
            } else if (ctx->PIPE_INT_STR() && ctx->PIPE_INT_STR()->getText().size() >= 2) {
                auto str = ctx->PIPE_INT_STR()->getText();
                str = str.substr(1, str.size() - 2);
                const auto& result_view = StrViewSplit(str, '|');
                for (auto& s : result_view) {
                    values.emplace_back(s);
                }
            }
            auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
            return std::make_any<ExprPtr>(str_list_ptr);
        }
        return visitInt_pipe_list(ctx);
    }

    std::any
    visitStr_pipe_list(FCParser::Str_pipe_listContext* ctx) override {
        StrList values;
        if (ctx->STRING() && ctx->STRING()->getText().size() >= 2) {
            auto str = ctx->STRING()->getText();
            str = str.substr(1, str.size() - 2);
            values.emplace_back(str);
        } else if (ctx->PIPE_STR_STR() && ctx->PIPE_STR_STR()->getText().size() >= 2) {
            auto str = ctx->PIPE_STR_STR()->getText();
            str = str.substr(1, str.size() - 2);
            const auto& result_view = StrViewSplit(str, '|');
            for (auto& s : result_view) {
                values.emplace_back(s);
            }
        }
        auto str_list_ptr = std::make_shared<StrListConstant>(std::move(values));
        return std::make_any<ExprPtr>(str_list_ptr);
    }

    std::any
    visitField_name(FCParser::Field_nameContext* ctx) override {
        return std::make_any<ExprPtr>(std::make_shared<FieldExpression>(ctx->getText()));
    }

    std::any
    visitNumeric(FCParser::NumericContext* ctx) override {
        if (ctx->INTEGER()) {
            return std::make_any<ExprPtr>(
                std::make_shared<NumericConstant>(VisitIntToken(ctx->INTEGER()->getText())));
        }
        if (ctx->FLOAT()) {
            return std::make_any<ExprPtr>(
                std::make_shared<NumericConstant>(std::stod(ctx->FLOAT()->getText())));
        }
        throw std::runtime_error("Invalid numeric value: " + ctx->getText());
    }

private:
    std::vector<std::string_view>
    StrViewSplit(std::string_view str, char delim) {
        std::vector<std::string_view> result;
        size_t start = 0;
        size_t end = str.find(delim);

        while (end != std::string_view::npos) {
            result.emplace_back(str.substr(start, end - start));
            start = end + 1;
            end = str.find(delim, start);
        }
        result.emplace_back(str.substr(start));
        return std::move(result);
    }

    bool
    IsStringType(const ExprPtr& expr) {
        if (auto field_expr = std::dynamic_pointer_cast<FieldExpression>(expr);
            field_expr != nullptr) {
            return schema_->GetTypeOfField(field_expr->fieldName) == STRING;
        }
        throw std::runtime_error("Invalid field expression: " + expr->ToString());
    }

    AttrTypeSchema* schema_;
};

vsag::ExprPtr
AstParse(const std::string& filter_condition_str, AttrTypeSchema* schema = nullptr);
}  // namespace vsag
