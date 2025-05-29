
// Generated from /home/lingyuan/workspace/vsag/src/attr/grammar/FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "FCVisitor.h"


/**
 * This class provides an empty implementation of FCVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  FCBaseVisitor : public FCVisitor {
public:

  virtual std::any visitFilter_condition(FCParser::Filter_conditionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNotExpr(FCParser::NotExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompExpr(FCParser::CompExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLogicalExpr(FCParser::LogicalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenExpr(FCParser::ParenExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntPipeListExpr(FCParser::IntPipeListExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStrPipeListExpr(FCParser::StrPipeListExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntListExpr(FCParser::IntListExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStrListExpr(FCParser::StrListExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericComparison(FCParser::NumericComparisonContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringComparison(FCParser::StringComparisonContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenFieldExpr(FCParser::ParenFieldExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFieldRef(FCParser::FieldRefContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExpr(FCParser::ArithmeticExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericConst(FCParser::NumericConstContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparison_sop(FCParser::Comparison_sopContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparison_op(FCParser::Comparison_opContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInt_value_list(FCParser::Int_value_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInt_pipe_list(FCParser::Int_pipe_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStr_value_list(FCParser::Str_value_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStr_pipe_list(FCParser::Str_pipe_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitField_name(FCParser::Field_nameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumeric(FCParser::NumericContext *ctx) override {
    return visitChildren(ctx);
  }


};

