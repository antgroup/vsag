
// Generated from FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FCListener.h"


/**
 * This class provides an empty implementation of FCListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  FCBaseListener : public FCListener {
public:

  virtual void enterFilter_condition(FCParser::Filter_conditionContext * /*ctx*/) override { }
  virtual void exitFilter_condition(FCParser::Filter_conditionContext * /*ctx*/) override { }

  virtual void enterNotExpr(FCParser::NotExprContext * /*ctx*/) override { }
  virtual void exitNotExpr(FCParser::NotExprContext * /*ctx*/) override { }

  virtual void enterCompExpr(FCParser::CompExprContext * /*ctx*/) override { }
  virtual void exitCompExpr(FCParser::CompExprContext * /*ctx*/) override { }

  virtual void enterLogicalExpr(FCParser::LogicalExprContext * /*ctx*/) override { }
  virtual void exitLogicalExpr(FCParser::LogicalExprContext * /*ctx*/) override { }

  virtual void enterParenExpr(FCParser::ParenExprContext * /*ctx*/) override { }
  virtual void exitParenExpr(FCParser::ParenExprContext * /*ctx*/) override { }

  virtual void enterIntPipeListExpr(FCParser::IntPipeListExprContext * /*ctx*/) override { }
  virtual void exitIntPipeListExpr(FCParser::IntPipeListExprContext * /*ctx*/) override { }

  virtual void enterStrPipeListExpr(FCParser::StrPipeListExprContext * /*ctx*/) override { }
  virtual void exitStrPipeListExpr(FCParser::StrPipeListExprContext * /*ctx*/) override { }

  virtual void enterIntListExpr(FCParser::IntListExprContext * /*ctx*/) override { }
  virtual void exitIntListExpr(FCParser::IntListExprContext * /*ctx*/) override { }

  virtual void enterStrListExpr(FCParser::StrListExprContext * /*ctx*/) override { }
  virtual void exitStrListExpr(FCParser::StrListExprContext * /*ctx*/) override { }

  virtual void enterNumericComparison(FCParser::NumericComparisonContext * /*ctx*/) override { }
  virtual void exitNumericComparison(FCParser::NumericComparisonContext * /*ctx*/) override { }

  virtual void enterStringComparison(FCParser::StringComparisonContext * /*ctx*/) override { }
  virtual void exitStringComparison(FCParser::StringComparisonContext * /*ctx*/) override { }

  virtual void enterParenFieldExpr(FCParser::ParenFieldExprContext * /*ctx*/) override { }
  virtual void exitParenFieldExpr(FCParser::ParenFieldExprContext * /*ctx*/) override { }

  virtual void enterFieldRef(FCParser::FieldRefContext * /*ctx*/) override { }
  virtual void exitFieldRef(FCParser::FieldRefContext * /*ctx*/) override { }

  virtual void enterArithmeticExpr(FCParser::ArithmeticExprContext * /*ctx*/) override { }
  virtual void exitArithmeticExpr(FCParser::ArithmeticExprContext * /*ctx*/) override { }

  virtual void enterNumericConst(FCParser::NumericConstContext * /*ctx*/) override { }
  virtual void exitNumericConst(FCParser::NumericConstContext * /*ctx*/) override { }

  virtual void enterComparison_sop(FCParser::Comparison_sopContext * /*ctx*/) override { }
  virtual void exitComparison_sop(FCParser::Comparison_sopContext * /*ctx*/) override { }

  virtual void enterComparison_op(FCParser::Comparison_opContext * /*ctx*/) override { }
  virtual void exitComparison_op(FCParser::Comparison_opContext * /*ctx*/) override { }

  virtual void enterInt_value_list(FCParser::Int_value_listContext * /*ctx*/) override { }
  virtual void exitInt_value_list(FCParser::Int_value_listContext * /*ctx*/) override { }

  virtual void enterInt_pipe_list(FCParser::Int_pipe_listContext * /*ctx*/) override { }
  virtual void exitInt_pipe_list(FCParser::Int_pipe_listContext * /*ctx*/) override { }

  virtual void enterStr_value_list(FCParser::Str_value_listContext * /*ctx*/) override { }
  virtual void exitStr_value_list(FCParser::Str_value_listContext * /*ctx*/) override { }

  virtual void enterStr_pipe_list(FCParser::Str_pipe_listContext * /*ctx*/) override { }
  virtual void exitStr_pipe_list(FCParser::Str_pipe_listContext * /*ctx*/) override { }

  virtual void enterField_name(FCParser::Field_nameContext * /*ctx*/) override { }
  virtual void exitField_name(FCParser::Field_nameContext * /*ctx*/) override { }

  virtual void enterNumeric(FCParser::NumericContext * /*ctx*/) override { }
  virtual void exitNumeric(FCParser::NumericContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

