
// Generated from /home/lingyuan/workspace/vsag/src/attr/grammar/FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FCParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by FCParser.
 */
class  FCVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by FCParser.
   */
    virtual std::any visitFilter_condition(FCParser::Filter_conditionContext *context) = 0;

    virtual std::any visitNotExpr(FCParser::NotExprContext *context) = 0;

    virtual std::any visitCompExpr(FCParser::CompExprContext *context) = 0;

    virtual std::any visitLogicalExpr(FCParser::LogicalExprContext *context) = 0;

    virtual std::any visitParenExpr(FCParser::ParenExprContext *context) = 0;

    virtual std::any visitIntPipeListExpr(FCParser::IntPipeListExprContext *context) = 0;

    virtual std::any visitStrPipeListExpr(FCParser::StrPipeListExprContext *context) = 0;

    virtual std::any visitIntListExpr(FCParser::IntListExprContext *context) = 0;

    virtual std::any visitStrListExpr(FCParser::StrListExprContext *context) = 0;

    virtual std::any visitNumericComparison(FCParser::NumericComparisonContext *context) = 0;

    virtual std::any visitStringComparison(FCParser::StringComparisonContext *context) = 0;

    virtual std::any visitParenFieldExpr(FCParser::ParenFieldExprContext *context) = 0;

    virtual std::any visitFieldRef(FCParser::FieldRefContext *context) = 0;

    virtual std::any visitArithmeticExpr(FCParser::ArithmeticExprContext *context) = 0;

    virtual std::any visitNumericConst(FCParser::NumericConstContext *context) = 0;

    virtual std::any visitComparison_sop(FCParser::Comparison_sopContext *context) = 0;

    virtual std::any visitComparison_op(FCParser::Comparison_opContext *context) = 0;

    virtual std::any visitInt_value_list(FCParser::Int_value_listContext *context) = 0;

    virtual std::any visitInt_pipe_list(FCParser::Int_pipe_listContext *context) = 0;

    virtual std::any visitStr_value_list(FCParser::Str_value_listContext *context) = 0;

    virtual std::any visitStr_pipe_list(FCParser::Str_pipe_listContext *context) = 0;

    virtual std::any visitField_name(FCParser::Field_nameContext *context) = 0;

    virtual std::any visitNumeric(FCParser::NumericContext *context) = 0;


};

