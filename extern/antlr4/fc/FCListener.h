
// Generated from FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FCParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by FCParser.
 */
class  FCListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterFilter_condition(FCParser::Filter_conditionContext *ctx) = 0;
  virtual void exitFilter_condition(FCParser::Filter_conditionContext *ctx) = 0;

  virtual void enterNotExpr(FCParser::NotExprContext *ctx) = 0;
  virtual void exitNotExpr(FCParser::NotExprContext *ctx) = 0;

  virtual void enterCompExpr(FCParser::CompExprContext *ctx) = 0;
  virtual void exitCompExpr(FCParser::CompExprContext *ctx) = 0;

  virtual void enterLogicalExpr(FCParser::LogicalExprContext *ctx) = 0;
  virtual void exitLogicalExpr(FCParser::LogicalExprContext *ctx) = 0;

  virtual void enterParenExpr(FCParser::ParenExprContext *ctx) = 0;
  virtual void exitParenExpr(FCParser::ParenExprContext *ctx) = 0;

  virtual void enterIntPipeListExpr(FCParser::IntPipeListExprContext *ctx) = 0;
  virtual void exitIntPipeListExpr(FCParser::IntPipeListExprContext *ctx) = 0;

  virtual void enterStrPipeListExpr(FCParser::StrPipeListExprContext *ctx) = 0;
  virtual void exitStrPipeListExpr(FCParser::StrPipeListExprContext *ctx) = 0;

  virtual void enterIntListExpr(FCParser::IntListExprContext *ctx) = 0;
  virtual void exitIntListExpr(FCParser::IntListExprContext *ctx) = 0;

  virtual void enterStrListExpr(FCParser::StrListExprContext *ctx) = 0;
  virtual void exitStrListExpr(FCParser::StrListExprContext *ctx) = 0;

  virtual void enterNumericComparison(FCParser::NumericComparisonContext *ctx) = 0;
  virtual void exitNumericComparison(FCParser::NumericComparisonContext *ctx) = 0;

  virtual void enterStringComparison(FCParser::StringComparisonContext *ctx) = 0;
  virtual void exitStringComparison(FCParser::StringComparisonContext *ctx) = 0;

  virtual void enterParenFieldExpr(FCParser::ParenFieldExprContext *ctx) = 0;
  virtual void exitParenFieldExpr(FCParser::ParenFieldExprContext *ctx) = 0;

  virtual void enterFieldRef(FCParser::FieldRefContext *ctx) = 0;
  virtual void exitFieldRef(FCParser::FieldRefContext *ctx) = 0;

  virtual void enterArithmeticExpr(FCParser::ArithmeticExprContext *ctx) = 0;
  virtual void exitArithmeticExpr(FCParser::ArithmeticExprContext *ctx) = 0;

  virtual void enterNumericConst(FCParser::NumericConstContext *ctx) = 0;
  virtual void exitNumericConst(FCParser::NumericConstContext *ctx) = 0;

  virtual void enterComparison_sop(FCParser::Comparison_sopContext *ctx) = 0;
  virtual void exitComparison_sop(FCParser::Comparison_sopContext *ctx) = 0;

  virtual void enterComparison_op(FCParser::Comparison_opContext *ctx) = 0;
  virtual void exitComparison_op(FCParser::Comparison_opContext *ctx) = 0;

  virtual void enterInt_value_list(FCParser::Int_value_listContext *ctx) = 0;
  virtual void exitInt_value_list(FCParser::Int_value_listContext *ctx) = 0;

  virtual void enterInt_pipe_list(FCParser::Int_pipe_listContext *ctx) = 0;
  virtual void exitInt_pipe_list(FCParser::Int_pipe_listContext *ctx) = 0;

  virtual void enterStr_value_list(FCParser::Str_value_listContext *ctx) = 0;
  virtual void exitStr_value_list(FCParser::Str_value_listContext *ctx) = 0;

  virtual void enterStr_pipe_list(FCParser::Str_pipe_listContext *ctx) = 0;
  virtual void exitStr_pipe_list(FCParser::Str_pipe_listContext *ctx) = 0;

  virtual void enterField_name(FCParser::Field_nameContext *ctx) = 0;
  virtual void exitField_name(FCParser::Field_nameContext *ctx) = 0;

  virtual void enterNumeric(FCParser::NumericContext *ctx) = 0;
  virtual void exitNumeric(FCParser::NumericContext *ctx) = 0;


};

