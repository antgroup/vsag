
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

class FCExpressionVisitor final : public FCBaseVisitor {
public:
    explicit FCExpressionVisitor(AttrTypeSchema* schema);
    std::any
    visitFilter_condition(FCParser::Filter_conditionContext* ctx) override;

    std::any
    visitParenExpr(FCParser::ParenExprContext* ctx) override;

    std::any
    visitNotExpr(FCParser::NotExprContext* ctx) override;

    std::any
    visitLogicalExpr(FCParser::LogicalExprContext* ctx) override;

    std::any
    visitCompExpr(FCParser::CompExprContext* ctx) override;

    std::any
    visitIntPipeListExpr(FCParser::IntPipeListExprContext* ctx) override;

    std::any
    visitStrPipeListExpr(FCParser::StrPipeListExprContext* ctx) override;

    std::any
    visitIntListExpr(FCParser::IntListExprContext* ctx) override;

    std::any
    visitStrListExpr(FCParser::StrListExprContext* ctx) override;

    std::any
    visitNumericComparison(FCParser::NumericComparisonContext* ctx) override;

    std::any
    visitStringComparison(FCParser::StringComparisonContext* ctx) override;

    std::any
    visitParenFieldExpr(FCParser::ParenFieldExprContext* ctx) override;

    std::any
    visitFieldRef(FCParser::FieldRefContext* ctx) override;

    std::any
    visitArithmeticExpr(FCParser::ArithmeticExprContext* ctx) override;

    std::any
    visitNumericConst(FCParser::NumericConstContext* ctx) override;

    std::any
    visitFunctionExpr(FCParser::FunctionExprContext* ctx) override;

    std::any
    visitRegionFilterExpr(FCParser::RegionFilterExprContext* ctx) override;

    std::any
    visitStr_value_list(FCParser::Str_value_listContext* ctx) override;

    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx) override;

    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx, const bool is_string_type);

    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx) override;

    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx, const bool is_string_type);

    std::any
    visitStr_pipe_list(FCParser::Str_pipe_listContext* ctx) override;

    std::any
    visitField_name(FCParser::Field_nameContext* ctx) override;

    std::any
    visitNumeric(FCParser::NumericContext* ctx) override;

    std::any
    visitArg_pipe_list(FCParser::Arg_pipe_listContext* ctx) override;

private:
    bool
    is_string_type(const ExprPtr& expr);

private:
    AttrTypeSchema* schema_;
};

}  // namespace vsag
