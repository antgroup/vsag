
// Generated from /home/lingyuan/workspace/vsag/src/attr/grammar/FC.g4 by ANTLR 4.13.2


#include "FCListener.h"
#include "FCVisitor.h"

#include "FCParser.h"


using namespace antlrcpp;

using namespace antlr4;

namespace {

struct FCParserStaticData final {
  FCParserStaticData(std::vector<std::string> ruleNames,
                        std::vector<std::string> literalNames,
                        std::vector<std::string> symbolicNames)
      : ruleNames(std::move(ruleNames)), literalNames(std::move(literalNames)),
        symbolicNames(std::move(symbolicNames)),
        vocabulary(this->literalNames, this->symbolicNames) {}

  FCParserStaticData(const FCParserStaticData&) = delete;
  FCParserStaticData(FCParserStaticData&&) = delete;
  FCParserStaticData& operator=(const FCParserStaticData&) = delete;
  FCParserStaticData& operator=(FCParserStaticData&&) = delete;

  std::vector<antlr4::dfa::DFA> decisionToDFA;
  antlr4::atn::PredictionContextCache sharedContextCache;
  const std::vector<std::string> ruleNames;
  const std::vector<std::string> literalNames;
  const std::vector<std::string> symbolicNames;
  const antlr4::dfa::Vocabulary vocabulary;
  antlr4::atn::SerializedATNView serializedATN;
  std::unique_ptr<antlr4::atn::ATN> atn;
};

::antlr4::internal::OnceFlag fcParserOnceFlag;
#if ANTLR4_USE_THREAD_LOCAL_CACHE
static thread_local
#endif
std::unique_ptr<FCParserStaticData> fcParserStaticData = nullptr;

void fcParserInitialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  if (fcParserStaticData != nullptr) {
    return;
  }
#else
  assert(fcParserStaticData == nullptr);
#endif
  auto staticData = std::make_unique<FCParserStaticData>(
    std::vector<std::string>{
      "filter_condition", "expr", "comparison", "field_expr", "comparison_sop", 
      "comparison_op", "int_value_list", "int_pipe_list", "str_value_list", 
      "str_pipe_list", "field_name", "numeric"
    },
    std::vector<std::string>{
      "", "'('", "')'", "','", "'['", "']'", "", "", "'!'", "", "", "'='", 
      "'!='", "'>'", "'<'", "'>='", "'<='", "'*'", "'/'", "'+'", "'-'", 
      "", "", "'|'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "AND", "OR", "NOT", "IN", "NOT_IN", "EQ", 
      "NQ", "GT", "LT", "GE", "LE", "MUL", "DIV", "ADD", "SUB", "ID", "INTEGER", 
      "SEP", "SEP_STR", "INT_STRING", "STRING", "PIPE_INT_STR", "PIPE_ID_STR", 
      "FLOAT", "WS", "LINE_COMMENT"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,31,153,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,1,0,1,0,1,0,1,1,1,1,1,1,1,1,
  	1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,39,8,1,1,1,1,1,1,1,5,1,44,8,1,10,1,12,
  	1,47,9,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
  	1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,
  	2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,97,
  	8,2,1,3,1,3,1,3,1,3,1,3,1,3,1,3,3,3,106,8,3,1,3,1,3,1,3,1,3,1,3,1,3,5,
  	3,114,8,3,10,3,12,3,117,9,3,1,4,1,4,1,5,1,5,1,6,1,6,1,6,1,6,5,6,127,8,
  	6,10,6,12,6,130,9,6,1,6,1,6,1,7,1,7,1,8,1,8,1,8,1,8,5,8,140,8,8,10,8,
  	12,8,143,9,8,1,8,1,8,1,9,1,9,1,10,1,10,1,11,1,11,1,11,0,2,2,6,12,0,2,
  	4,6,8,10,12,14,16,18,20,22,0,10,1,0,6,7,1,0,9,10,1,0,25,26,1,0,17,18,
  	1,0,19,20,1,0,11,12,1,0,11,16,2,0,25,25,27,27,2,0,26,26,28,28,2,0,22,
  	22,29,29,156,0,24,1,0,0,0,2,38,1,0,0,0,4,96,1,0,0,0,6,105,1,0,0,0,8,118,
  	1,0,0,0,10,120,1,0,0,0,12,122,1,0,0,0,14,133,1,0,0,0,16,135,1,0,0,0,18,
  	146,1,0,0,0,20,148,1,0,0,0,22,150,1,0,0,0,24,25,3,2,1,0,25,26,5,0,0,1,
  	26,1,1,0,0,0,27,28,6,1,-1,0,28,29,5,1,0,0,29,30,3,2,1,0,30,31,5,2,0,0,
  	31,39,1,0,0,0,32,33,5,8,0,0,33,34,5,1,0,0,34,35,3,2,1,0,35,36,5,2,0,0,
  	36,39,1,0,0,0,37,39,3,4,2,0,38,27,1,0,0,0,38,32,1,0,0,0,38,37,1,0,0,0,
  	39,45,1,0,0,0,40,41,10,2,0,0,41,42,7,0,0,0,42,44,3,2,1,3,43,40,1,0,0,
  	0,44,47,1,0,0,0,45,43,1,0,0,0,45,46,1,0,0,0,46,3,1,0,0,0,47,45,1,0,0,
  	0,48,49,7,1,0,0,49,50,5,1,0,0,50,51,3,20,10,0,51,52,5,3,0,0,52,53,3,14,
  	7,0,53,54,5,2,0,0,54,97,1,0,0,0,55,56,7,1,0,0,56,57,5,1,0,0,57,58,3,20,
  	10,0,58,59,5,3,0,0,59,60,3,14,7,0,60,61,5,3,0,0,61,62,5,24,0,0,62,63,
  	5,2,0,0,63,97,1,0,0,0,64,65,7,1,0,0,65,66,5,1,0,0,66,67,3,20,10,0,67,
  	68,5,3,0,0,68,69,3,18,9,0,69,70,5,2,0,0,70,97,1,0,0,0,71,72,7,1,0,0,72,
  	73,5,1,0,0,73,74,3,20,10,0,74,75,5,3,0,0,75,76,3,18,9,0,76,77,5,3,0,0,
  	77,78,5,24,0,0,78,79,5,2,0,0,79,97,1,0,0,0,80,81,3,20,10,0,81,82,7,1,
  	0,0,82,83,3,12,6,0,83,97,1,0,0,0,84,85,3,20,10,0,85,86,7,1,0,0,86,87,
  	3,16,8,0,87,97,1,0,0,0,88,89,3,6,3,0,89,90,3,10,5,0,90,91,3,22,11,0,91,
  	97,1,0,0,0,92,93,3,20,10,0,93,94,3,8,4,0,94,95,7,2,0,0,95,97,1,0,0,0,
  	96,48,1,0,0,0,96,55,1,0,0,0,96,64,1,0,0,0,96,71,1,0,0,0,96,80,1,0,0,0,
  	96,84,1,0,0,0,96,88,1,0,0,0,96,92,1,0,0,0,97,5,1,0,0,0,98,99,6,3,-1,0,
  	99,106,3,20,10,0,100,106,3,22,11,0,101,102,5,1,0,0,102,103,3,6,3,0,103,
  	104,5,2,0,0,104,106,1,0,0,0,105,98,1,0,0,0,105,100,1,0,0,0,105,101,1,
  	0,0,0,106,115,1,0,0,0,107,108,10,5,0,0,108,109,7,3,0,0,109,114,3,6,3,
  	6,110,111,10,4,0,0,111,112,7,4,0,0,112,114,3,6,3,5,113,107,1,0,0,0,113,
  	110,1,0,0,0,114,117,1,0,0,0,115,113,1,0,0,0,115,116,1,0,0,0,116,7,1,0,
  	0,0,117,115,1,0,0,0,118,119,7,5,0,0,119,9,1,0,0,0,120,121,7,6,0,0,121,
  	11,1,0,0,0,122,123,5,4,0,0,123,128,5,22,0,0,124,125,5,3,0,0,125,127,5,
  	22,0,0,126,124,1,0,0,0,127,130,1,0,0,0,128,126,1,0,0,0,128,129,1,0,0,
  	0,129,131,1,0,0,0,130,128,1,0,0,0,131,132,5,5,0,0,132,13,1,0,0,0,133,
  	134,7,7,0,0,134,15,1,0,0,0,135,136,5,4,0,0,136,141,5,26,0,0,137,138,5,
  	3,0,0,138,140,5,26,0,0,139,137,1,0,0,0,140,143,1,0,0,0,141,139,1,0,0,
  	0,141,142,1,0,0,0,142,144,1,0,0,0,143,141,1,0,0,0,144,145,5,5,0,0,145,
  	17,1,0,0,0,146,147,7,8,0,0,147,19,1,0,0,0,148,149,5,21,0,0,149,21,1,0,
  	0,0,150,151,7,9,0,0,151,23,1,0,0,0,8,38,45,96,105,113,115,128,141
  };
  staticData->serializedATN = antlr4::atn::SerializedATNView(serializedATNSegment, sizeof(serializedATNSegment) / sizeof(serializedATNSegment[0]));

  antlr4::atn::ATNDeserializer deserializer;
  staticData->atn = deserializer.deserialize(staticData->serializedATN);

  const size_t count = staticData->atn->getNumberOfDecisions();
  staticData->decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    staticData->decisionToDFA.emplace_back(staticData->atn->getDecisionState(i), i);
  }
  fcParserStaticData = std::move(staticData);
}

}

FCParser::FCParser(TokenStream *input) : FCParser(input, antlr4::atn::ParserATNSimulatorOptions()) {}

FCParser::FCParser(TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options) : Parser(input) {
  FCParser::initialize();
  _interpreter = new atn::ParserATNSimulator(this, *fcParserStaticData->atn, fcParserStaticData->decisionToDFA, fcParserStaticData->sharedContextCache, options);
}

FCParser::~FCParser() {
  delete _interpreter;
}

const atn::ATN& FCParser::getATN() const {
  return *fcParserStaticData->atn;
}

std::string FCParser::getGrammarFileName() const {
  return "FC.g4";
}

const std::vector<std::string>& FCParser::getRuleNames() const {
  return fcParserStaticData->ruleNames;
}

const dfa::Vocabulary& FCParser::getVocabulary() const {
  return fcParserStaticData->vocabulary;
}

antlr4::atn::SerializedATNView FCParser::getSerializedATN() const {
  return fcParserStaticData->serializedATN;
}


//----------------- Filter_conditionContext ------------------------------------------------------------------

FCParser::Filter_conditionContext::Filter_conditionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FCParser::ExprContext* FCParser::Filter_conditionContext::expr() {
  return getRuleContext<FCParser::ExprContext>(0);
}

tree::TerminalNode* FCParser::Filter_conditionContext::EOF() {
  return getToken(FCParser::EOF, 0);
}


size_t FCParser::Filter_conditionContext::getRuleIndex() const {
  return FCParser::RuleFilter_condition;
}

void FCParser::Filter_conditionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFilter_condition(this);
}

void FCParser::Filter_conditionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFilter_condition(this);
}


std::any FCParser::Filter_conditionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitFilter_condition(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Filter_conditionContext* FCParser::filter_condition() {
  Filter_conditionContext *_localctx = _tracker.createInstance<Filter_conditionContext>(_ctx, getState());
  enterRule(_localctx, 0, FCParser::RuleFilter_condition);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(24);
    expr(0);
    setState(25);
    match(FCParser::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

FCParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FCParser::ExprContext::getRuleIndex() const {
  return FCParser::RuleExpr;
}

void FCParser::ExprContext::copyFrom(ExprContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- NotExprContext ------------------------------------------------------------------

tree::TerminalNode* FCParser::NotExprContext::NOT() {
  return getToken(FCParser::NOT, 0);
}

FCParser::ExprContext* FCParser::NotExprContext::expr() {
  return getRuleContext<FCParser::ExprContext>(0);
}

FCParser::NotExprContext::NotExprContext(ExprContext *ctx) { copyFrom(ctx); }

void FCParser::NotExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNotExpr(this);
}
void FCParser::NotExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNotExpr(this);
}

std::any FCParser::NotExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitNotExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- CompExprContext ------------------------------------------------------------------

FCParser::ComparisonContext* FCParser::CompExprContext::comparison() {
  return getRuleContext<FCParser::ComparisonContext>(0);
}

FCParser::CompExprContext::CompExprContext(ExprContext *ctx) { copyFrom(ctx); }

void FCParser::CompExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCompExpr(this);
}
void FCParser::CompExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCompExpr(this);
}

std::any FCParser::CompExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitCompExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- LogicalExprContext ------------------------------------------------------------------

std::vector<FCParser::ExprContext *> FCParser::LogicalExprContext::expr() {
  return getRuleContexts<FCParser::ExprContext>();
}

FCParser::ExprContext* FCParser::LogicalExprContext::expr(size_t i) {
  return getRuleContext<FCParser::ExprContext>(i);
}

tree::TerminalNode* FCParser::LogicalExprContext::AND() {
  return getToken(FCParser::AND, 0);
}

tree::TerminalNode* FCParser::LogicalExprContext::OR() {
  return getToken(FCParser::OR, 0);
}

FCParser::LogicalExprContext::LogicalExprContext(ExprContext *ctx) { copyFrom(ctx); }

void FCParser::LogicalExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogicalExpr(this);
}
void FCParser::LogicalExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogicalExpr(this);
}

std::any FCParser::LogicalExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitLogicalExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ParenExprContext ------------------------------------------------------------------

FCParser::ExprContext* FCParser::ParenExprContext::expr() {
  return getRuleContext<FCParser::ExprContext>(0);
}

FCParser::ParenExprContext::ParenExprContext(ExprContext *ctx) { copyFrom(ctx); }

void FCParser::ParenExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParenExpr(this);
}
void FCParser::ParenExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParenExpr(this);
}

std::any FCParser::ParenExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitParenExpr(this);
  else
    return visitor->visitChildren(this);
}

FCParser::ExprContext* FCParser::expr() {
   return expr(0);
}

FCParser::ExprContext* FCParser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  FCParser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  FCParser::ExprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 2;
  enterRecursionRule(_localctx, 2, FCParser::RuleExpr, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(38);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<ParenExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(28);
      match(FCParser::T__0);
      setState(29);
      expr(0);
      setState(30);
      match(FCParser::T__1);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<NotExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(32);
      match(FCParser::NOT);
      setState(33);
      match(FCParser::T__0);
      setState(34);
      expr(0);
      setState(35);
      match(FCParser::T__1);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<CompExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(37);
      comparison();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(45);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        auto newContext = _tracker.createInstance<LogicalExprContext>(_tracker.createInstance<ExprContext>(parentContext, parentState));
        _localctx = newContext;
        newContext->left = previousContext;
        pushNewRecursionContext(newContext, startState, RuleExpr);
        setState(40);

        if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
        setState(41);
        antlrcpp::downCast<LogicalExprContext *>(_localctx)->op = _input->LT(1);
        _la = _input->LA(1);
        if (!(_la == FCParser::AND

        || _la == FCParser::OR)) {
          antlrcpp::downCast<LogicalExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
        }
        else {
          _errHandler->reportMatch(this);
          consume();
        }
        setState(42);
        antlrcpp::downCast<LogicalExprContext *>(_localctx)->right = expr(3); 
      }
      setState(47);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ComparisonContext ------------------------------------------------------------------

FCParser::ComparisonContext::ComparisonContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FCParser::ComparisonContext::getRuleIndex() const {
  return FCParser::RuleComparison;
}

void FCParser::ComparisonContext::copyFrom(ComparisonContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- StringComparisonContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::StringComparisonContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::Comparison_sopContext* FCParser::StringComparisonContext::comparison_sop() {
  return getRuleContext<FCParser::Comparison_sopContext>(0);
}

tree::TerminalNode* FCParser::StringComparisonContext::STRING() {
  return getToken(FCParser::STRING, 0);
}

tree::TerminalNode* FCParser::StringComparisonContext::INT_STRING() {
  return getToken(FCParser::INT_STRING, 0);
}

FCParser::StringComparisonContext::StringComparisonContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::StringComparisonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStringComparison(this);
}
void FCParser::StringComparisonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStringComparison(this);
}

std::any FCParser::StringComparisonContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitStringComparison(this);
  else
    return visitor->visitChildren(this);
}
//----------------- StrListExprContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::StrListExprContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::Str_value_listContext* FCParser::StrListExprContext::str_value_list() {
  return getRuleContext<FCParser::Str_value_listContext>(0);
}

tree::TerminalNode* FCParser::StrListExprContext::NOT_IN() {
  return getToken(FCParser::NOT_IN, 0);
}

tree::TerminalNode* FCParser::StrListExprContext::IN() {
  return getToken(FCParser::IN, 0);
}

FCParser::StrListExprContext::StrListExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::StrListExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStrListExpr(this);
}
void FCParser::StrListExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStrListExpr(this);
}

std::any FCParser::StrListExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitStrListExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IntListExprContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::IntListExprContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::Int_value_listContext* FCParser::IntListExprContext::int_value_list() {
  return getRuleContext<FCParser::Int_value_listContext>(0);
}

tree::TerminalNode* FCParser::IntListExprContext::NOT_IN() {
  return getToken(FCParser::NOT_IN, 0);
}

tree::TerminalNode* FCParser::IntListExprContext::IN() {
  return getToken(FCParser::IN, 0);
}

FCParser::IntListExprContext::IntListExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::IntListExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIntListExpr(this);
}
void FCParser::IntListExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIntListExpr(this);
}

std::any FCParser::IntListExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitIntListExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- IntPipeListExprContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::IntPipeListExprContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::Int_pipe_listContext* FCParser::IntPipeListExprContext::int_pipe_list() {
  return getRuleContext<FCParser::Int_pipe_listContext>(0);
}

tree::TerminalNode* FCParser::IntPipeListExprContext::NOT_IN() {
  return getToken(FCParser::NOT_IN, 0);
}

tree::TerminalNode* FCParser::IntPipeListExprContext::IN() {
  return getToken(FCParser::IN, 0);
}

tree::TerminalNode* FCParser::IntPipeListExprContext::SEP_STR() {
  return getToken(FCParser::SEP_STR, 0);
}

FCParser::IntPipeListExprContext::IntPipeListExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::IntPipeListExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIntPipeListExpr(this);
}
void FCParser::IntPipeListExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIntPipeListExpr(this);
}

std::any FCParser::IntPipeListExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitIntPipeListExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- StrPipeListExprContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::StrPipeListExprContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::Str_pipe_listContext* FCParser::StrPipeListExprContext::str_pipe_list() {
  return getRuleContext<FCParser::Str_pipe_listContext>(0);
}

tree::TerminalNode* FCParser::StrPipeListExprContext::NOT_IN() {
  return getToken(FCParser::NOT_IN, 0);
}

tree::TerminalNode* FCParser::StrPipeListExprContext::IN() {
  return getToken(FCParser::IN, 0);
}

tree::TerminalNode* FCParser::StrPipeListExprContext::SEP_STR() {
  return getToken(FCParser::SEP_STR, 0);
}

FCParser::StrPipeListExprContext::StrPipeListExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::StrPipeListExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStrPipeListExpr(this);
}
void FCParser::StrPipeListExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStrPipeListExpr(this);
}

std::any FCParser::StrPipeListExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitStrPipeListExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- NumericComparisonContext ------------------------------------------------------------------

FCParser::Field_exprContext* FCParser::NumericComparisonContext::field_expr() {
  return getRuleContext<FCParser::Field_exprContext>(0);
}

FCParser::NumericContext* FCParser::NumericComparisonContext::numeric() {
  return getRuleContext<FCParser::NumericContext>(0);
}

FCParser::Comparison_opContext* FCParser::NumericComparisonContext::comparison_op() {
  return getRuleContext<FCParser::Comparison_opContext>(0);
}

FCParser::NumericComparisonContext::NumericComparisonContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::NumericComparisonContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumericComparison(this);
}
void FCParser::NumericComparisonContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumericComparison(this);
}

std::any FCParser::NumericComparisonContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitNumericComparison(this);
  else
    return visitor->visitChildren(this);
}
FCParser::ComparisonContext* FCParser::comparison() {
  ComparisonContext *_localctx = _tracker.createInstance<ComparisonContext>(_ctx, getState());
  enterRule(_localctx, 4, FCParser::RuleComparison);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(96);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<FCParser::IntPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 1);
      setState(48);
      antlrcpp::downCast<IntPipeListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<IntPipeListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(49);
      match(FCParser::T__0);
      setState(50);
      field_name();
      setState(51);
      match(FCParser::T__2);
      setState(52);
      int_pipe_list();
      setState(53);
      match(FCParser::T__1);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<FCParser::IntPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 2);
      setState(55);
      antlrcpp::downCast<IntPipeListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<IntPipeListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(56);
      match(FCParser::T__0);
      setState(57);
      field_name();
      setState(58);
      match(FCParser::T__2);
      setState(59);
      int_pipe_list();
      setState(60);
      match(FCParser::T__2);
      setState(61);
      match(FCParser::SEP_STR);
      setState(62);
      match(FCParser::T__1);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<FCParser::StrPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 3);
      setState(64);
      antlrcpp::downCast<StrPipeListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<StrPipeListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(65);
      match(FCParser::T__0);
      setState(66);
      field_name();
      setState(67);
      match(FCParser::T__2);
      setState(68);
      str_pipe_list();
      setState(69);
      match(FCParser::T__1);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<FCParser::StrPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 4);
      setState(71);
      antlrcpp::downCast<StrPipeListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<StrPipeListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(72);
      match(FCParser::T__0);
      setState(73);
      field_name();
      setState(74);
      match(FCParser::T__2);
      setState(75);
      str_pipe_list();
      setState(76);
      match(FCParser::T__2);
      setState(77);
      match(FCParser::SEP_STR);
      setState(78);
      match(FCParser::T__1);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<FCParser::IntListExprContext>(_localctx);
      enterOuterAlt(_localctx, 5);
      setState(80);
      field_name();
      setState(81);
      antlrcpp::downCast<IntListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<IntListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(82);
      int_value_list();
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<FCParser::StrListExprContext>(_localctx);
      enterOuterAlt(_localctx, 6);
      setState(84);
      field_name();
      setState(85);
      antlrcpp::downCast<StrListExprContext *>(_localctx)->op = _input->LT(1);
      _la = _input->LA(1);
      if (!(_la == FCParser::IN

      || _la == FCParser::NOT_IN)) {
        antlrcpp::downCast<StrListExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      setState(86);
      str_value_list();
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<FCParser::NumericComparisonContext>(_localctx);
      enterOuterAlt(_localctx, 7);
      setState(88);
      field_expr(0);
      setState(89);
      antlrcpp::downCast<NumericComparisonContext *>(_localctx)->op = comparison_op();
      setState(90);
      numeric();
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<FCParser::StringComparisonContext>(_localctx);
      enterOuterAlt(_localctx, 8);
      setState(92);
      field_name();
      setState(93);
      antlrcpp::downCast<StringComparisonContext *>(_localctx)->op = comparison_sop();
      setState(94);
      _la = _input->LA(1);
      if (!(_la == FCParser::INT_STRING

      || _la == FCParser::STRING)) {
      _errHandler->recoverInline(this);
      }
      else {
        _errHandler->reportMatch(this);
        consume();
      }
      break;
    }

    default:
      break;
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Field_exprContext ------------------------------------------------------------------

FCParser::Field_exprContext::Field_exprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t FCParser::Field_exprContext::getRuleIndex() const {
  return FCParser::RuleField_expr;
}

void FCParser::Field_exprContext::copyFrom(Field_exprContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- ParenFieldExprContext ------------------------------------------------------------------

FCParser::Field_exprContext* FCParser::ParenFieldExprContext::field_expr() {
  return getRuleContext<FCParser::Field_exprContext>(0);
}

FCParser::ParenFieldExprContext::ParenFieldExprContext(Field_exprContext *ctx) { copyFrom(ctx); }

void FCParser::ParenFieldExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParenFieldExpr(this);
}
void FCParser::ParenFieldExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParenFieldExpr(this);
}

std::any FCParser::ParenFieldExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitParenFieldExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- FieldRefContext ------------------------------------------------------------------

FCParser::Field_nameContext* FCParser::FieldRefContext::field_name() {
  return getRuleContext<FCParser::Field_nameContext>(0);
}

FCParser::FieldRefContext::FieldRefContext(Field_exprContext *ctx) { copyFrom(ctx); }

void FCParser::FieldRefContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFieldRef(this);
}
void FCParser::FieldRefContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFieldRef(this);
}

std::any FCParser::FieldRefContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitFieldRef(this);
  else
    return visitor->visitChildren(this);
}
//----------------- ArithmeticExprContext ------------------------------------------------------------------

std::vector<FCParser::Field_exprContext *> FCParser::ArithmeticExprContext::field_expr() {
  return getRuleContexts<FCParser::Field_exprContext>();
}

FCParser::Field_exprContext* FCParser::ArithmeticExprContext::field_expr(size_t i) {
  return getRuleContext<FCParser::Field_exprContext>(i);
}

tree::TerminalNode* FCParser::ArithmeticExprContext::MUL() {
  return getToken(FCParser::MUL, 0);
}

tree::TerminalNode* FCParser::ArithmeticExprContext::DIV() {
  return getToken(FCParser::DIV, 0);
}

tree::TerminalNode* FCParser::ArithmeticExprContext::ADD() {
  return getToken(FCParser::ADD, 0);
}

tree::TerminalNode* FCParser::ArithmeticExprContext::SUB() {
  return getToken(FCParser::SUB, 0);
}

FCParser::ArithmeticExprContext::ArithmeticExprContext(Field_exprContext *ctx) { copyFrom(ctx); }

void FCParser::ArithmeticExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArithmeticExpr(this);
}
void FCParser::ArithmeticExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArithmeticExpr(this);
}

std::any FCParser::ArithmeticExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitArithmeticExpr(this);
  else
    return visitor->visitChildren(this);
}
//----------------- NumericConstContext ------------------------------------------------------------------

FCParser::NumericContext* FCParser::NumericConstContext::numeric() {
  return getRuleContext<FCParser::NumericContext>(0);
}

FCParser::NumericConstContext::NumericConstContext(Field_exprContext *ctx) { copyFrom(ctx); }

void FCParser::NumericConstContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumericConst(this);
}
void FCParser::NumericConstContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumericConst(this);
}

std::any FCParser::NumericConstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitNumericConst(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Field_exprContext* FCParser::field_expr() {
   return field_expr(0);
}

FCParser::Field_exprContext* FCParser::field_expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  FCParser::Field_exprContext *_localctx = _tracker.createInstance<Field_exprContext>(_ctx, parentState);
  FCParser::Field_exprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 6;
  enterRecursionRule(_localctx, 6, FCParser::RuleField_expr, precedence);

    size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(105);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FCParser::ID: {
        _localctx = _tracker.createInstance<FieldRefContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;

        setState(99);
        field_name();
        break;
      }

      case FCParser::INTEGER:
      case FCParser::FLOAT: {
        _localctx = _tracker.createInstance<NumericConstContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(100);
        numeric();
        break;
      }

      case FCParser::T__0: {
        _localctx = _tracker.createInstance<ParenFieldExprContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(101);
        match(FCParser::T__0);
        setState(102);
        field_expr(0);
        setState(103);
        match(FCParser::T__1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(115);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(113);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<ArithmeticExprContext>(_tracker.createInstance<Field_exprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleField_expr);
          setState(107);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(108);
          antlrcpp::downCast<ArithmeticExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == FCParser::MUL

          || _la == FCParser::DIV)) {
            antlrcpp::downCast<ArithmeticExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(109);
          field_expr(6);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<ArithmeticExprContext>(_tracker.createInstance<Field_exprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleField_expr);
          setState(110);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(111);
          antlrcpp::downCast<ArithmeticExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == FCParser::ADD

          || _la == FCParser::SUB)) {
            antlrcpp::downCast<ArithmeticExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(112);
          field_expr(5);
          break;
        }

        default:
          break;
        } 
      }
      setState(117);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Comparison_sopContext ------------------------------------------------------------------

FCParser::Comparison_sopContext::Comparison_sopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Comparison_sopContext::EQ() {
  return getToken(FCParser::EQ, 0);
}

tree::TerminalNode* FCParser::Comparison_sopContext::NQ() {
  return getToken(FCParser::NQ, 0);
}


size_t FCParser::Comparison_sopContext::getRuleIndex() const {
  return FCParser::RuleComparison_sop;
}

void FCParser::Comparison_sopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparison_sop(this);
}

void FCParser::Comparison_sopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparison_sop(this);
}


std::any FCParser::Comparison_sopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitComparison_sop(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Comparison_sopContext* FCParser::comparison_sop() {
  Comparison_sopContext *_localctx = _tracker.createInstance<Comparison_sopContext>(_ctx, getState());
  enterRule(_localctx, 8, FCParser::RuleComparison_sop);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(118);
    _la = _input->LA(1);
    if (!(_la == FCParser::EQ

    || _la == FCParser::NQ)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Comparison_opContext ------------------------------------------------------------------

FCParser::Comparison_opContext::Comparison_opContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Comparison_opContext::EQ() {
  return getToken(FCParser::EQ, 0);
}

tree::TerminalNode* FCParser::Comparison_opContext::NQ() {
  return getToken(FCParser::NQ, 0);
}

tree::TerminalNode* FCParser::Comparison_opContext::GT() {
  return getToken(FCParser::GT, 0);
}

tree::TerminalNode* FCParser::Comparison_opContext::LT() {
  return getToken(FCParser::LT, 0);
}

tree::TerminalNode* FCParser::Comparison_opContext::GE() {
  return getToken(FCParser::GE, 0);
}

tree::TerminalNode* FCParser::Comparison_opContext::LE() {
  return getToken(FCParser::LE, 0);
}


size_t FCParser::Comparison_opContext::getRuleIndex() const {
  return FCParser::RuleComparison_op;
}

void FCParser::Comparison_opContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterComparison_op(this);
}

void FCParser::Comparison_opContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitComparison_op(this);
}


std::any FCParser::Comparison_opContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitComparison_op(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Comparison_opContext* FCParser::comparison_op() {
  Comparison_opContext *_localctx = _tracker.createInstance<Comparison_opContext>(_ctx, getState());
  enterRule(_localctx, 10, FCParser::RuleComparison_op);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(120);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 129024) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Int_value_listContext ------------------------------------------------------------------

FCParser::Int_value_listContext::Int_value_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> FCParser::Int_value_listContext::INTEGER() {
  return getTokens(FCParser::INTEGER);
}

tree::TerminalNode* FCParser::Int_value_listContext::INTEGER(size_t i) {
  return getToken(FCParser::INTEGER, i);
}


size_t FCParser::Int_value_listContext::getRuleIndex() const {
  return FCParser::RuleInt_value_list;
}

void FCParser::Int_value_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInt_value_list(this);
}

void FCParser::Int_value_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInt_value_list(this);
}


std::any FCParser::Int_value_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitInt_value_list(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Int_value_listContext* FCParser::int_value_list() {
  Int_value_listContext *_localctx = _tracker.createInstance<Int_value_listContext>(_ctx, getState());
  enterRule(_localctx, 12, FCParser::RuleInt_value_list);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(122);
    match(FCParser::T__3);
    setState(123);
    match(FCParser::INTEGER);
    setState(128);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FCParser::T__2) {
      setState(124);
      match(FCParser::T__2);
      setState(125);
      match(FCParser::INTEGER);
      setState(130);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(131);
    match(FCParser::T__4);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Int_pipe_listContext ------------------------------------------------------------------

FCParser::Int_pipe_listContext::Int_pipe_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Int_pipe_listContext::PIPE_INT_STR() {
  return getToken(FCParser::PIPE_INT_STR, 0);
}

tree::TerminalNode* FCParser::Int_pipe_listContext::INT_STRING() {
  return getToken(FCParser::INT_STRING, 0);
}


size_t FCParser::Int_pipe_listContext::getRuleIndex() const {
  return FCParser::RuleInt_pipe_list;
}

void FCParser::Int_pipe_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInt_pipe_list(this);
}

void FCParser::Int_pipe_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInt_pipe_list(this);
}


std::any FCParser::Int_pipe_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitInt_pipe_list(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Int_pipe_listContext* FCParser::int_pipe_list() {
  Int_pipe_listContext *_localctx = _tracker.createInstance<Int_pipe_listContext>(_ctx, getState());
  enterRule(_localctx, 14, FCParser::RuleInt_pipe_list);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(133);
    _la = _input->LA(1);
    if (!(_la == FCParser::INT_STRING

    || _la == FCParser::PIPE_INT_STR)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Str_value_listContext ------------------------------------------------------------------

FCParser::Str_value_listContext::Str_value_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> FCParser::Str_value_listContext::STRING() {
  return getTokens(FCParser::STRING);
}

tree::TerminalNode* FCParser::Str_value_listContext::STRING(size_t i) {
  return getToken(FCParser::STRING, i);
}


size_t FCParser::Str_value_listContext::getRuleIndex() const {
  return FCParser::RuleStr_value_list;
}

void FCParser::Str_value_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStr_value_list(this);
}

void FCParser::Str_value_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStr_value_list(this);
}


std::any FCParser::Str_value_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitStr_value_list(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Str_value_listContext* FCParser::str_value_list() {
  Str_value_listContext *_localctx = _tracker.createInstance<Str_value_listContext>(_ctx, getState());
  enterRule(_localctx, 16, FCParser::RuleStr_value_list);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(135);
    match(FCParser::T__3);
    setState(136);
    match(FCParser::STRING);
    setState(141);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FCParser::T__2) {
      setState(137);
      match(FCParser::T__2);
      setState(138);
      match(FCParser::STRING);
      setState(143);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(144);
    match(FCParser::T__4);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Str_pipe_listContext ------------------------------------------------------------------

FCParser::Str_pipe_listContext::Str_pipe_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Str_pipe_listContext::PIPE_ID_STR() {
  return getToken(FCParser::PIPE_ID_STR, 0);
}

tree::TerminalNode* FCParser::Str_pipe_listContext::STRING() {
  return getToken(FCParser::STRING, 0);
}


size_t FCParser::Str_pipe_listContext::getRuleIndex() const {
  return FCParser::RuleStr_pipe_list;
}

void FCParser::Str_pipe_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStr_pipe_list(this);
}

void FCParser::Str_pipe_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStr_pipe_list(this);
}


std::any FCParser::Str_pipe_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitStr_pipe_list(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Str_pipe_listContext* FCParser::str_pipe_list() {
  Str_pipe_listContext *_localctx = _tracker.createInstance<Str_pipe_listContext>(_ctx, getState());
  enterRule(_localctx, 18, FCParser::RuleStr_pipe_list);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(146);
    _la = _input->LA(1);
    if (!(_la == FCParser::STRING

    || _la == FCParser::PIPE_ID_STR)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Field_nameContext ------------------------------------------------------------------

FCParser::Field_nameContext::Field_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Field_nameContext::ID() {
  return getToken(FCParser::ID, 0);
}


size_t FCParser::Field_nameContext::getRuleIndex() const {
  return FCParser::RuleField_name;
}

void FCParser::Field_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterField_name(this);
}

void FCParser::Field_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitField_name(this);
}


std::any FCParser::Field_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitField_name(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Field_nameContext* FCParser::field_name() {
  Field_nameContext *_localctx = _tracker.createInstance<Field_nameContext>(_ctx, getState());
  enterRule(_localctx, 20, FCParser::RuleField_name);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(148);
    match(FCParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- NumericContext ------------------------------------------------------------------

FCParser::NumericContext::NumericContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::NumericContext::INTEGER() {
  return getToken(FCParser::INTEGER, 0);
}

tree::TerminalNode* FCParser::NumericContext::FLOAT() {
  return getToken(FCParser::FLOAT, 0);
}


size_t FCParser::NumericContext::getRuleIndex() const {
  return FCParser::RuleNumeric;
}

void FCParser::NumericContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterNumeric(this);
}

void FCParser::NumericContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitNumeric(this);
}


std::any FCParser::NumericContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitNumeric(this);
  else
    return visitor->visitChildren(this);
}

FCParser::NumericContext* FCParser::numeric() {
  NumericContext *_localctx = _tracker.createInstance<NumericContext>(_ctx, getState());
  enterRule(_localctx, 22, FCParser::RuleNumeric);
  size_t _la = 0;

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(150);
    _la = _input->LA(1);
    if (!(_la == FCParser::INTEGER

    || _la == FCParser::FLOAT)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool FCParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 1: return exprSempred(antlrcpp::downCast<ExprContext *>(context), predicateIndex);
    case 3: return field_exprSempred(antlrcpp::downCast<Field_exprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool FCParser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 2);

  default:
    break;
  }
  return true;
}

bool FCParser::field_exprSempred(Field_exprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 1: return precpred(_ctx, 5);
    case 2: return precpred(_ctx, 4);

  default:
    break;
  }
  return true;
}

void FCParser::initialize() {
#if ANTLR4_USE_THREAD_LOCAL_CACHE
  fcParserInitialize();
#else
  ::antlr4::internal::call_once(fcParserOnceFlag, fcParserInitialize);
#endif
}
