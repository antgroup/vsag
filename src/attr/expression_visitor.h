
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
 * @file expression_visitor.h
 * @brief Visitor pattern implementation for parsing filter conditions into expression AST.
 *
 * This file contains the FCExpressionVisitor class which traverses the ANTLR4
 * parse tree and builds expression AST nodes for filter condition evaluation.
 */

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
#include "vsag_exception.h"

namespace vsag {

/**
 * @class FCErrorListener
 * @brief Custom error listener for ANTLR4 parser to capture syntax errors.
 *
 * This class extends antlr4::BaseErrorListener to provide detailed error messages
 * when parsing filter condition strings fails.
 */
class FCErrorListener final : public antlr4::BaseErrorListener {
public:
    /**
     * @brief Constructs an FCErrorListener with the input string.
     * @param input The original input string being parsed.
     */
    FCErrorListener(const std::string& input) : input_(input) {
    }

    /**
     * @brief Handles syntax errors encountered during parsing.
     * @param recognizer The recognizer that detected the error.
     * @param offendingSymbol The token that caused the error.
     * @param line The line number where the error occurred.
     * @param charPositionInLine The character position within the line.
     * @param msg The error message.
     * @param e The exception pointer, if any.
     * @throws std::runtime_error with detailed error information.
     */
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
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("Syntax error in filter condition, line({}), charPositionInLine({}), "
                        "msg({}), offendingText({}), input({})",
                        line,
                        charPositionInLine,
                        msg,
                        offendingText,
                        input_));
    }

private:
    std::string input_;  ///< The original input string being parsed
};

/**
 * @class FCExpressionVisitor
 * @brief ANTLR4 visitor that converts filter condition parse tree to expression AST.
 *
 * This class visits nodes in the ANTLR4 parse tree and constructs corresponding
 * Expression objects for filter condition evaluation.
 */
class FCExpressionVisitor final : public FCBaseVisitor {
public:
    /**
     * @brief Constructs an FCExpressionVisitor with an optional attribute type schema.
     * @param schema Pointer to the attribute type schema for type checking.
     */
    explicit FCExpressionVisitor(AttrTypeSchema* schema);

    /**
     * @brief Visits the filter condition context and returns the root expression.
     * @param ctx The filter condition context.
     * @return std::any containing an ExprPtr to the root expression.
     */
    std::any
    visitFilter_condition(FCParser::Filter_conditionContext* ctx) override;

    /**
     * @brief Visits a parenthesized expression.
     * @param ctx The parenthesized expression context.
     * @return std::any containing an ExprPtr.
     */
    std::any
    visitParenExpr(FCParser::ParenExprContext* ctx) override;

    /**
     * @brief Visits a NOT expression.
     * @param ctx The NOT expression context.
     * @return std::any containing an ExprPtr to a NotExpression.
     */
    std::any
    visitNotExpr(FCParser::NotExprContext* ctx) override;

    /**
     * @brief Visits a logical expression (AND/OR).
     * @param ctx The logical expression context.
     * @return std::any containing an ExprPtr to a LogicalExpression.
     */
    std::any
    visitLogicalExpr(FCParser::LogicalExprContext* ctx) override;

    /**
     * @brief Visits a comparison expression.
     * @param ctx The comparison expression context.
     * @return std::any containing an ExprPtr to a ComparisonExpression.
     */
    std::any
    visitCompExpr(FCParser::CompExprContext* ctx) override;

    /**
     * @brief Visits an integer pipe list expression.
     * @param ctx The integer pipe list expression context.
     * @return std::any containing an ExprPtr to an IntListExpression.
     */
    std::any
    visitIntPipeListExpr(FCParser::IntPipeListExprContext* ctx) override;

    /**
     * @brief Visits a string pipe list expression.
     * @param ctx The string pipe list expression context.
     * @return std::any containing an ExprPtr to a StrListExpression.
     */
    std::any
    visitStrPipeListExpr(FCParser::StrPipeListExprContext* ctx) override;

    /**
     * @brief Visits an integer list expression.
     * @param ctx The integer list expression context.
     * @return std::any containing an ExprPtr to an IntListExpression.
     */
    std::any
    visitIntListExpr(FCParser::IntListExprContext* ctx) override;

    /**
     * @brief Visits a string list expression.
     * @param ctx The string list expression context.
     * @return std::any containing an ExprPtr to a StrListExpression.
     */
    std::any
    visitStrListExpr(FCParser::StrListExprContext* ctx) override;

    /**
     * @brief Visits a numeric comparison expression.
     * @param ctx The numeric comparison context.
     * @return std::any containing an ExprPtr to a ComparisonExpression.
     */
    std::any
    visitNumericComparison(FCParser::NumericComparisonContext* ctx) override;

    /**
     * @brief Visits a string comparison expression.
     * @param ctx The string comparison context.
     * @return std::any containing an ExprPtr to a ComparisonExpression.
     */
    std::any
    visitStringComparison(FCParser::StringComparisonContext* ctx) override;

    /**
     * @brief Visits a parenthesized field expression.
     * @param ctx The parenthesized field expression context.
     * @return std::any containing an ExprPtr.
     */
    std::any
    visitParenFieldExpr(FCParser::ParenFieldExprContext* ctx) override;

    /**
     * @brief Visits a field reference expression.
     * @param ctx The field reference context.
     * @return std::any containing an ExprPtr to a FieldExpression.
     */
    std::any
    visitFieldRef(FCParser::FieldRefContext* ctx) override;

    /**
     * @brief Visits an arithmetic expression.
     * @param ctx The arithmetic expression context.
     * @return std::any containing an ExprPtr to an ArithmeticExpression.
     */
    std::any
    visitArithmeticExpr(FCParser::ArithmeticExprContext* ctx) override;

    /**
     * @brief Visits a numeric constant.
     * @param ctx The numeric constant context.
     * @return std::any containing an ExprPtr to a NumericConstant.
     */
    std::any
    visitNumericConst(FCParser::NumericConstContext* ctx) override;

    /**
     * @brief Visits a string value list.
     * @param ctx The string value list context.
     * @return std::any containing a StrList.
     */
    std::any
    visitStr_value_list(FCParser::Str_value_listContext* ctx) override;

    /**
     * @brief Visits an integer value list.
     * @param ctx The integer value list context.
     * @return std::any containing a vector of integer values.
     */
    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx) override;

    /**
     * @brief Visits an integer value list with type specification.
     * @param ctx The integer value list context.
     * @param is_string_type Flag indicating if the values should be treated as strings.
     * @return std::any containing a vector of values.
     */
    std::any
    visitInt_value_list(FCParser::Int_value_listContext* ctx, const bool is_string_type);

    /**
     * @brief Visits an integer pipe list.
     * @param ctx The integer pipe list context.
     * @return std::any containing a vector of integer values.
     */
    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx) override;

    /**
     * @brief Visits an integer pipe list with type specification.
     * @param ctx The integer pipe list context.
     * @param is_string_type Flag indicating if the values should be treated as strings.
     * @return std::any containing a vector of values.
     */
    std::any
    visitInt_pipe_list(FCParser::Int_pipe_listContext* ctx, const bool is_string_type);

    /**
     * @brief Visits a string pipe list.
     * @param ctx The string pipe list context.
     * @return std::any containing a StrList.
     */
    std::any
    visitStr_pipe_list(FCParser::Str_pipe_listContext* ctx) override;

    /**
     * @brief Visits a field name.
     * @param ctx The field name context.
     * @return std::any containing the field name string.
     */
    std::any
    visitField_name(FCParser::Field_nameContext* ctx) override;

    /**
     * @brief Visits a numeric value.
     * @param ctx The numeric context.
     * @return std::any containing a NumericValue.
     */
    std::any
    visitNumeric(FCParser::NumericContext* ctx) override;

private:
    /**
     * @brief Checks if an expression evaluates to a string type.
     * @param expr The expression to check.
     * @return True if the expression is string-typed, false otherwise.
     */
    bool
    is_string_type(const ExprPtr& expr);

private:
    AttrTypeSchema* schema_;  ///< Pointer to attribute type schema for type checking
};

}  // namespace vsag
