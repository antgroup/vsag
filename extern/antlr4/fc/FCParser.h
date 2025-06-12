
// Generated from /home/lingyuan/workspace/vsag/src/attr/grammar/FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  FCParser : public antlr4::Parser {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, AND = 6, OR = 7, NOT = 8, 
    IN = 9, NOT_IN = 10, EQ = 11, NQ = 12, GT = 13, LT = 14, GE = 15, LE = 16, 
    MUL = 17, DIV = 18, ADD = 19, SUB = 20, ID = 21, INTEGER = 22, SEP = 23, 
    SEP_STR = 24, INT_STRING = 25, STRING = 26, PIPE_INT_STR = 27, PIPE_ID_STR = 28, 
    FLOAT = 29, WS = 30, LINE_COMMENT = 31
  };

  enum {
    RuleFilter_condition = 0, RuleExpr = 1, RuleComparison = 2, RuleField_expr = 3, 
    RuleComparison_sop = 4, RuleComparison_op = 5, RuleInt_value_list = 6, 
    RuleInt_pipe_list = 7, RuleStr_value_list = 8, RuleStr_pipe_list = 9, 
    RuleField_name = 10, RuleNumeric = 11
  };

  explicit FCParser(antlr4::TokenStream *input);

  FCParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~FCParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class Filter_conditionContext;
  class ExprContext;
  class ComparisonContext;
  class Field_exprContext;
  class Comparison_sopContext;
  class Comparison_opContext;
  class Int_value_listContext;
  class Int_pipe_listContext;
  class Str_value_listContext;
  class Str_pipe_listContext;
  class Field_nameContext;
  class NumericContext; 

  class  Filter_conditionContext : public antlr4::ParserRuleContext {
  public:
    Filter_conditionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    antlr4::tree::TerminalNode *EOF();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Filter_conditionContext* filter_condition();

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ExprContext() = default;
    void copyFrom(ExprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  NotExprContext : public ExprContext {
  public:
    NotExprContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *NOT();
    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CompExprContext : public ExprContext {
  public:
    CompExprContext(ExprContext *ctx);

    ComparisonContext *comparison();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LogicalExprContext : public ExprContext {
  public:
    LogicalExprContext(ExprContext *ctx);

    FCParser::ExprContext *left = nullptr;
    antlr4::Token *op = nullptr;
    FCParser::ExprContext *right = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ParenExprContext : public ExprContext {
  public:
    ParenExprContext(ExprContext *ctx);

    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);
  class  ComparisonContext : public antlr4::ParserRuleContext {
  public:
    ComparisonContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ComparisonContext() = default;
    void copyFrom(ComparisonContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  StringComparisonContext : public ComparisonContext {
  public:
    StringComparisonContext(ComparisonContext *ctx);

    FCParser::Comparison_sopContext *op = nullptr;
    Field_nameContext *field_name();
    Comparison_sopContext *comparison_sop();
    antlr4::tree::TerminalNode *STRING();
    antlr4::tree::TerminalNode *INT_STRING();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  StrListExprContext : public ComparisonContext {
  public:
    StrListExprContext(ComparisonContext *ctx);

    antlr4::Token *op = nullptr;
    Field_nameContext *field_name();
    Str_value_listContext *str_value_list();
    antlr4::tree::TerminalNode *NOT_IN();
    antlr4::tree::TerminalNode *IN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IntListExprContext : public ComparisonContext {
  public:
    IntListExprContext(ComparisonContext *ctx);

    antlr4::Token *op = nullptr;
    Field_nameContext *field_name();
    Int_value_listContext *int_value_list();
    antlr4::tree::TerminalNode *NOT_IN();
    antlr4::tree::TerminalNode *IN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IntPipeListExprContext : public ComparisonContext {
  public:
    IntPipeListExprContext(ComparisonContext *ctx);

    antlr4::Token *op = nullptr;
    Field_nameContext *field_name();
    Int_pipe_listContext *int_pipe_list();
    antlr4::tree::TerminalNode *NOT_IN();
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *SEP_STR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  StrPipeListExprContext : public ComparisonContext {
  public:
    StrPipeListExprContext(ComparisonContext *ctx);

    antlr4::Token *op = nullptr;
    Field_nameContext *field_name();
    Str_pipe_listContext *str_pipe_list();
    antlr4::tree::TerminalNode *NOT_IN();
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *SEP_STR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  NumericComparisonContext : public ComparisonContext {
  public:
    NumericComparisonContext(ComparisonContext *ctx);

    FCParser::Comparison_opContext *op = nullptr;
    Field_exprContext *field_expr();
    NumericContext *numeric();
    Comparison_opContext *comparison_op();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ComparisonContext* comparison();

  class  Field_exprContext : public antlr4::ParserRuleContext {
  public:
    Field_exprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Field_exprContext() = default;
    void copyFrom(Field_exprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ParenFieldExprContext : public Field_exprContext {
  public:
    ParenFieldExprContext(Field_exprContext *ctx);

    Field_exprContext *field_expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FieldRefContext : public Field_exprContext {
  public:
    FieldRefContext(Field_exprContext *ctx);

    Field_nameContext *field_name();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArithmeticExprContext : public Field_exprContext {
  public:
    ArithmeticExprContext(Field_exprContext *ctx);

    antlr4::Token *op = nullptr;
    std::vector<Field_exprContext *> field_expr();
    Field_exprContext* field_expr(size_t i);
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    antlr4::tree::TerminalNode *ADD();
    antlr4::tree::TerminalNode *SUB();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  NumericConstContext : public Field_exprContext {
  public:
    NumericConstContext(Field_exprContext *ctx);

    NumericContext *numeric();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Field_exprContext* field_expr();
  Field_exprContext* field_expr(int precedence);
  class  Comparison_sopContext : public antlr4::ParserRuleContext {
  public:
    Comparison_sopContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NQ();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Comparison_sopContext* comparison_sop();

  class  Comparison_opContext : public antlr4::ParserRuleContext {
  public:
    Comparison_opContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NQ();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GE();
    antlr4::tree::TerminalNode *LE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Comparison_opContext* comparison_op();

  class  Int_value_listContext : public antlr4::ParserRuleContext {
  public:
    Int_value_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> INTEGER();
    antlr4::tree::TerminalNode* INTEGER(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Int_value_listContext* int_value_list();

  class  Int_pipe_listContext : public antlr4::ParserRuleContext {
  public:
    Int_pipe_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PIPE_INT_STR();
    antlr4::tree::TerminalNode *INT_STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Int_pipe_listContext* int_pipe_list();

  class  Str_value_listContext : public antlr4::ParserRuleContext {
  public:
    Str_value_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> STRING();
    antlr4::tree::TerminalNode* STRING(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Str_value_listContext* str_value_list();

  class  Str_pipe_listContext : public antlr4::ParserRuleContext {
  public:
    Str_pipe_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PIPE_ID_STR();
    antlr4::tree::TerminalNode *STRING();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Str_pipe_listContext* str_pipe_list();

  class  Field_nameContext : public antlr4::ParserRuleContext {
  public:
    Field_nameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Field_nameContext* field_name();

  class  NumericContext : public antlr4::ParserRuleContext {
  public:
    NumericContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INTEGER();
    antlr4::tree::TerminalNode *FLOAT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  NumericContext* numeric();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);
  bool field_exprSempred(Field_exprContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

