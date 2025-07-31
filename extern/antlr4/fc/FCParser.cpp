
// Generated from FC.g4 by ANTLR 4.13.2


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
      "str_pipe_list", "arg_pipe_list", "field_name", "function_name", "numeric"
    },
    std::vector<std::string>{
      "", "'('", "')'", "','", "'['", "']'", "", "'region_filter'", "", 
      "", "'!'", "", "", "'='", "'!='", "'>'", "'<'", "'>='", "'<='", "'*'", 
      "'/'", "'+'", "'-'", "", "", "'|'"
    },
    std::vector<std::string>{
      "", "", "", "", "", "", "FUNCTION", "REGION_FILTER", "AND", "OR", 
      "NOT", "IN", "NOT_IN", "EQ", "NQ", "GT", "LT", "GE", "LE", "MUL", 
      "DIV", "ADD", "SUB", "ID", "INTEGER", "SEP", "SEP_STR", "INT_STRING", 
      "STRING", "PIPE_INT_STR", "PIPE_STR_STR", "FLOAT", "WS", "LINE_COMMENT"
    }
  );
  static const int32_t serializedATNSegment[] = {
  	4,1,33,198,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,6,7,6,2,
  	7,7,7,2,8,7,8,2,9,7,9,2,10,7,10,2,11,7,11,2,12,7,12,2,13,7,13,1,0,1,0,
  	1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,43,8,1,1,1,1,1,1,
  	1,5,1,48,8,1,10,1,12,1,51,9,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
  	1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,
  	2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,
  	1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,
  	2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,125,8,2,1,3,1,3,1,3,1,3,1,3,
  	1,3,1,3,3,3,134,8,3,1,3,1,3,1,3,1,3,1,3,1,3,5,3,142,8,3,10,3,12,3,145,
  	9,3,1,4,1,4,1,5,1,5,1,6,1,6,1,6,1,6,5,6,155,8,6,10,6,12,6,158,9,6,1,6,
  	1,6,1,7,1,7,1,8,1,8,1,8,1,8,5,8,168,8,8,10,8,12,8,171,9,8,1,8,1,8,1,8,
  	1,8,1,8,5,8,178,8,8,10,8,12,8,181,9,8,1,8,3,8,184,8,8,1,9,1,9,1,10,1,
  	10,3,10,190,8,10,1,11,1,11,1,12,1,12,1,13,1,13,1,13,0,2,2,6,14,0,2,4,
  	6,8,10,12,14,16,18,20,22,24,26,0,10,1,0,8,9,1,0,11,12,1,0,27,28,1,0,19,
  	20,1,0,21,22,1,0,13,14,1,0,13,18,2,0,27,27,29,29,2,0,28,28,30,30,2,0,
  	24,24,31,31,204,0,28,1,0,0,0,2,42,1,0,0,0,4,124,1,0,0,0,6,133,1,0,0,0,
  	8,146,1,0,0,0,10,148,1,0,0,0,12,150,1,0,0,0,14,161,1,0,0,0,16,183,1,0,
  	0,0,18,185,1,0,0,0,20,189,1,0,0,0,22,191,1,0,0,0,24,193,1,0,0,0,26,195,
  	1,0,0,0,28,29,3,2,1,0,29,30,5,0,0,1,30,1,1,0,0,0,31,32,6,1,-1,0,32,33,
  	5,1,0,0,33,34,3,2,1,0,34,35,5,2,0,0,35,43,1,0,0,0,36,37,5,10,0,0,37,38,
  	5,1,0,0,38,39,3,2,1,0,39,40,5,2,0,0,40,43,1,0,0,0,41,43,3,4,2,0,42,31,
  	1,0,0,0,42,36,1,0,0,0,42,41,1,0,0,0,43,49,1,0,0,0,44,45,10,2,0,0,45,46,
  	7,0,0,0,46,48,3,2,1,3,47,44,1,0,0,0,48,51,1,0,0,0,49,47,1,0,0,0,49,50,
  	1,0,0,0,50,3,1,0,0,0,51,49,1,0,0,0,52,53,7,1,0,0,53,54,5,1,0,0,54,55,
  	3,22,11,0,55,56,5,3,0,0,56,57,3,14,7,0,57,58,5,2,0,0,58,125,1,0,0,0,59,
  	60,7,1,0,0,60,61,5,1,0,0,61,62,3,22,11,0,62,63,5,3,0,0,63,64,3,14,7,0,
  	64,65,5,3,0,0,65,66,5,26,0,0,66,67,5,2,0,0,67,125,1,0,0,0,68,69,7,1,0,
  	0,69,70,5,1,0,0,70,71,3,22,11,0,71,72,5,3,0,0,72,73,3,18,9,0,73,74,5,
  	2,0,0,74,125,1,0,0,0,75,76,7,1,0,0,76,77,5,1,0,0,77,78,3,22,11,0,78,79,
  	5,3,0,0,79,80,3,18,9,0,80,81,5,3,0,0,81,82,5,26,0,0,82,83,5,2,0,0,83,
  	125,1,0,0,0,84,85,5,6,0,0,85,86,5,1,0,0,86,87,3,24,12,0,87,88,5,3,0,0,
  	88,89,3,20,10,0,89,90,5,3,0,0,90,91,3,18,9,0,91,92,5,2,0,0,92,125,1,0,
  	0,0,93,94,5,7,0,0,94,95,5,1,0,0,95,96,3,22,11,0,96,97,5,3,0,0,97,98,3,
  	22,11,0,98,99,5,3,0,0,99,100,3,22,11,0,100,101,5,3,0,0,101,102,3,14,7,
  	0,102,103,5,3,0,0,103,104,3,14,7,0,104,105,5,3,0,0,105,106,3,14,7,0,106,
  	107,5,2,0,0,107,125,1,0,0,0,108,109,3,22,11,0,109,110,7,1,0,0,110,111,
  	3,12,6,0,111,125,1,0,0,0,112,113,3,22,11,0,113,114,7,1,0,0,114,115,3,
  	16,8,0,115,125,1,0,0,0,116,117,3,6,3,0,117,118,3,10,5,0,118,119,3,26,
  	13,0,119,125,1,0,0,0,120,121,3,22,11,0,121,122,3,8,4,0,122,123,7,2,0,
  	0,123,125,1,0,0,0,124,52,1,0,0,0,124,59,1,0,0,0,124,68,1,0,0,0,124,75,
  	1,0,0,0,124,84,1,0,0,0,124,93,1,0,0,0,124,108,1,0,0,0,124,112,1,0,0,0,
  	124,116,1,0,0,0,124,120,1,0,0,0,125,5,1,0,0,0,126,127,6,3,-1,0,127,134,
  	3,22,11,0,128,134,3,26,13,0,129,130,5,1,0,0,130,131,3,6,3,0,131,132,5,
  	2,0,0,132,134,1,0,0,0,133,126,1,0,0,0,133,128,1,0,0,0,133,129,1,0,0,0,
  	134,143,1,0,0,0,135,136,10,5,0,0,136,137,7,3,0,0,137,142,3,6,3,6,138,
  	139,10,4,0,0,139,140,7,4,0,0,140,142,3,6,3,5,141,135,1,0,0,0,141,138,
  	1,0,0,0,142,145,1,0,0,0,143,141,1,0,0,0,143,144,1,0,0,0,144,7,1,0,0,0,
  	145,143,1,0,0,0,146,147,7,5,0,0,147,9,1,0,0,0,148,149,7,6,0,0,149,11,
  	1,0,0,0,150,151,5,4,0,0,151,156,5,24,0,0,152,153,5,3,0,0,153,155,5,24,
  	0,0,154,152,1,0,0,0,155,158,1,0,0,0,156,154,1,0,0,0,156,157,1,0,0,0,157,
  	159,1,0,0,0,158,156,1,0,0,0,159,160,5,5,0,0,160,13,1,0,0,0,161,162,7,
  	7,0,0,162,15,1,0,0,0,163,164,5,4,0,0,164,169,5,28,0,0,165,166,5,3,0,0,
  	166,168,5,28,0,0,167,165,1,0,0,0,168,171,1,0,0,0,169,167,1,0,0,0,169,
  	170,1,0,0,0,170,172,1,0,0,0,171,169,1,0,0,0,172,184,5,5,0,0,173,174,5,
  	4,0,0,174,179,5,27,0,0,175,176,5,3,0,0,176,178,5,27,0,0,177,175,1,0,0,
  	0,178,181,1,0,0,0,179,177,1,0,0,0,179,180,1,0,0,0,180,182,1,0,0,0,181,
  	179,1,0,0,0,182,184,5,5,0,0,183,163,1,0,0,0,183,173,1,0,0,0,184,17,1,
  	0,0,0,185,186,7,8,0,0,186,19,1,0,0,0,187,190,3,18,9,0,188,190,3,14,7,
  	0,189,187,1,0,0,0,189,188,1,0,0,0,190,21,1,0,0,0,191,192,5,23,0,0,192,
  	23,1,0,0,0,193,194,5,23,0,0,194,25,1,0,0,0,195,196,7,9,0,0,196,27,1,0,
  	0,0,11,42,49,124,133,141,143,156,169,179,183,189
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
    setState(28);
    expr(0);
    setState(29);
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
    setState(42);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<ParenExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;

      setState(32);
      match(FCParser::T__0);
      setState(33);
      expr(0);
      setState(34);
      match(FCParser::T__1);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<NotExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(36);
      match(FCParser::NOT);
      setState(37);
      match(FCParser::T__0);
      setState(38);
      expr(0);
      setState(39);
      match(FCParser::T__1);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<CompExprContext>(_localctx);
      _ctx = _localctx;
      previousContext = _localctx;
      setState(41);
      comparison();
      break;
    }

    default:
      break;
    }
    _ctx->stop = _input->LT(-1);
    setState(49);
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
        setState(44);

        if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
        setState(45);
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
        setState(46);
        antlrcpp::downCast<LogicalExprContext *>(_localctx)->right = expr(3); 
      }
      setState(51);
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
//----------------- RegionFilterExprContext ------------------------------------------------------------------

std::vector<FCParser::Field_nameContext *> FCParser::RegionFilterExprContext::field_name() {
  return getRuleContexts<FCParser::Field_nameContext>();
}

FCParser::Field_nameContext* FCParser::RegionFilterExprContext::field_name(size_t i) {
  return getRuleContext<FCParser::Field_nameContext>(i);
}

std::vector<FCParser::Int_pipe_listContext *> FCParser::RegionFilterExprContext::int_pipe_list() {
  return getRuleContexts<FCParser::Int_pipe_listContext>();
}

FCParser::Int_pipe_listContext* FCParser::RegionFilterExprContext::int_pipe_list(size_t i) {
  return getRuleContext<FCParser::Int_pipe_listContext>(i);
}

tree::TerminalNode* FCParser::RegionFilterExprContext::REGION_FILTER() {
  return getToken(FCParser::REGION_FILTER, 0);
}

FCParser::RegionFilterExprContext::RegionFilterExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::RegionFilterExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRegionFilterExpr(this);
}
void FCParser::RegionFilterExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRegionFilterExpr(this);
}

std::any FCParser::RegionFilterExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitRegionFilterExpr(this);
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
//----------------- FunctionExprContext ------------------------------------------------------------------

FCParser::Function_nameContext* FCParser::FunctionExprContext::function_name() {
  return getRuleContext<FCParser::Function_nameContext>(0);
}

FCParser::Arg_pipe_listContext* FCParser::FunctionExprContext::arg_pipe_list() {
  return getRuleContext<FCParser::Arg_pipe_listContext>(0);
}

FCParser::Str_pipe_listContext* FCParser::FunctionExprContext::str_pipe_list() {
  return getRuleContext<FCParser::Str_pipe_listContext>(0);
}

tree::TerminalNode* FCParser::FunctionExprContext::FUNCTION() {
  return getToken(FCParser::FUNCTION, 0);
}

FCParser::FunctionExprContext::FunctionExprContext(ComparisonContext *ctx) { copyFrom(ctx); }

void FCParser::FunctionExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunctionExpr(this);
}
void FCParser::FunctionExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunctionExpr(this);
}

std::any FCParser::FunctionExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitFunctionExpr(this);
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
    setState(124);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      _localctx = _tracker.createInstance<FCParser::IntPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 1);
      setState(52);
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
      setState(53);
      match(FCParser::T__0);
      setState(54);
      field_name();
      setState(55);
      match(FCParser::T__2);
      setState(56);
      int_pipe_list();
      setState(57);
      match(FCParser::T__1);
      break;
    }

    case 2: {
      _localctx = _tracker.createInstance<FCParser::IntPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 2);
      setState(59);
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
      setState(60);
      match(FCParser::T__0);
      setState(61);
      field_name();
      setState(62);
      match(FCParser::T__2);
      setState(63);
      int_pipe_list();
      setState(64);
      match(FCParser::T__2);
      setState(65);
      match(FCParser::SEP_STR);
      setState(66);
      match(FCParser::T__1);
      break;
    }

    case 3: {
      _localctx = _tracker.createInstance<FCParser::StrPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 3);
      setState(68);
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
      setState(69);
      match(FCParser::T__0);
      setState(70);
      field_name();
      setState(71);
      match(FCParser::T__2);
      setState(72);
      str_pipe_list();
      setState(73);
      match(FCParser::T__1);
      break;
    }

    case 4: {
      _localctx = _tracker.createInstance<FCParser::StrPipeListExprContext>(_localctx);
      enterOuterAlt(_localctx, 4);
      setState(75);
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
      setState(76);
      match(FCParser::T__0);
      setState(77);
      field_name();
      setState(78);
      match(FCParser::T__2);
      setState(79);
      str_pipe_list();
      setState(80);
      match(FCParser::T__2);
      setState(81);
      match(FCParser::SEP_STR);
      setState(82);
      match(FCParser::T__1);
      break;
    }

    case 5: {
      _localctx = _tracker.createInstance<FCParser::FunctionExprContext>(_localctx);
      enterOuterAlt(_localctx, 5);
      setState(84);
      antlrcpp::downCast<FunctionExprContext *>(_localctx)->op = match(FCParser::FUNCTION);
      setState(85);
      match(FCParser::T__0);
      setState(86);
      function_name();
      setState(87);
      match(FCParser::T__2);
      setState(88);
      arg_pipe_list();
      setState(89);
      match(FCParser::T__2);
      setState(90);
      str_pipe_list();
      setState(91);
      match(FCParser::T__1);
      break;
    }

    case 6: {
      _localctx = _tracker.createInstance<FCParser::RegionFilterExprContext>(_localctx);
      enterOuterAlt(_localctx, 6);
      setState(93);
      antlrcpp::downCast<RegionFilterExprContext *>(_localctx)->op = match(FCParser::REGION_FILTER);
      setState(94);
      match(FCParser::T__0);
      setState(95);
      field_name();
      setState(96);
      match(FCParser::T__2);
      setState(97);
      field_name();
      setState(98);
      match(FCParser::T__2);
      setState(99);
      field_name();
      setState(100);
      match(FCParser::T__2);
      setState(101);
      int_pipe_list();
      setState(102);
      match(FCParser::T__2);
      setState(103);
      int_pipe_list();
      setState(104);
      match(FCParser::T__2);
      setState(105);
      int_pipe_list();
      setState(106);
      match(FCParser::T__1);
      break;
    }

    case 7: {
      _localctx = _tracker.createInstance<FCParser::IntListExprContext>(_localctx);
      enterOuterAlt(_localctx, 7);
      setState(108);
      field_name();
      setState(109);
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
      setState(110);
      int_value_list();
      break;
    }

    case 8: {
      _localctx = _tracker.createInstance<FCParser::StrListExprContext>(_localctx);
      enterOuterAlt(_localctx, 8);
      setState(112);
      field_name();
      setState(113);
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
      setState(114);
      str_value_list();
      break;
    }

    case 9: {
      _localctx = _tracker.createInstance<FCParser::NumericComparisonContext>(_localctx);
      enterOuterAlt(_localctx, 9);
      setState(116);
      field_expr(0);
      setState(117);
      antlrcpp::downCast<NumericComparisonContext *>(_localctx)->op = comparison_op();
      setState(118);
      numeric();
      break;
    }

    case 10: {
      _localctx = _tracker.createInstance<FCParser::StringComparisonContext>(_localctx);
      enterOuterAlt(_localctx, 10);
      setState(120);
      field_name();
      setState(121);
      antlrcpp::downCast<StringComparisonContext *>(_localctx)->op = comparison_sop();
      setState(122);
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
    setState(133);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FCParser::ID: {
        _localctx = _tracker.createInstance<FieldRefContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;

        setState(127);
        field_name();
        break;
      }

      case FCParser::INTEGER:
      case FCParser::FLOAT: {
        _localctx = _tracker.createInstance<NumericConstContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(128);
        numeric();
        break;
      }

      case FCParser::T__0: {
        _localctx = _tracker.createInstance<ParenFieldExprContext>(_localctx);
        _ctx = _localctx;
        previousContext = _localctx;
        setState(129);
        match(FCParser::T__0);
        setState(130);
        field_expr(0);
        setState(131);
        match(FCParser::T__1);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(143);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(141);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
        case 1: {
          auto newContext = _tracker.createInstance<ArithmeticExprContext>(_tracker.createInstance<Field_exprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleField_expr);
          setState(135);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(136);
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
          setState(137);
          field_expr(6);
          break;
        }

        case 2: {
          auto newContext = _tracker.createInstance<ArithmeticExprContext>(_tracker.createInstance<Field_exprContext>(parentContext, parentState));
          _localctx = newContext;
          pushNewRecursionContext(newContext, startState, RuleField_expr);
          setState(138);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(139);
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
          setState(140);
          field_expr(5);
          break;
        }

        default:
          break;
        } 
      }
      setState(145);
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
    setState(146);
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
    setState(148);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & 516096) != 0))) {
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
    setState(150);
    match(FCParser::T__3);
    setState(151);
    match(FCParser::INTEGER);
    setState(156);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == FCParser::T__2) {
      setState(152);
      match(FCParser::T__2);
      setState(153);
      match(FCParser::INTEGER);
      setState(158);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(159);
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
    setState(161);
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

std::vector<tree::TerminalNode *> FCParser::Str_value_listContext::INT_STRING() {
  return getTokens(FCParser::INT_STRING);
}

tree::TerminalNode* FCParser::Str_value_listContext::INT_STRING(size_t i) {
  return getToken(FCParser::INT_STRING, i);
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
    setState(183);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(163);
      match(FCParser::T__3);
      setState(164);
      match(FCParser::STRING);
      setState(169);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == FCParser::T__2) {
        setState(165);
        match(FCParser::T__2);
        setState(166);
        match(FCParser::STRING);
        setState(171);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(172);
      match(FCParser::T__4);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(173);
      match(FCParser::T__3);
      setState(174);
      match(FCParser::INT_STRING);
      setState(179);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == FCParser::T__2) {
        setState(175);
        match(FCParser::T__2);
        setState(176);
        match(FCParser::INT_STRING);
        setState(181);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(182);
      match(FCParser::T__4);
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

//----------------- Str_pipe_listContext ------------------------------------------------------------------

FCParser::Str_pipe_listContext::Str_pipe_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Str_pipe_listContext::PIPE_STR_STR() {
  return getToken(FCParser::PIPE_STR_STR, 0);
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
    setState(185);
    _la = _input->LA(1);
    if (!(_la == FCParser::STRING

    || _la == FCParser::PIPE_STR_STR)) {
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

//----------------- Arg_pipe_listContext ------------------------------------------------------------------

FCParser::Arg_pipe_listContext::Arg_pipe_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

FCParser::Str_pipe_listContext* FCParser::Arg_pipe_listContext::str_pipe_list() {
  return getRuleContext<FCParser::Str_pipe_listContext>(0);
}

FCParser::Int_pipe_listContext* FCParser::Arg_pipe_listContext::int_pipe_list() {
  return getRuleContext<FCParser::Int_pipe_listContext>(0);
}


size_t FCParser::Arg_pipe_listContext::getRuleIndex() const {
  return FCParser::RuleArg_pipe_list;
}

void FCParser::Arg_pipe_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArg_pipe_list(this);
}

void FCParser::Arg_pipe_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArg_pipe_list(this);
}


std::any FCParser::Arg_pipe_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitArg_pipe_list(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Arg_pipe_listContext* FCParser::arg_pipe_list() {
  Arg_pipe_listContext *_localctx = _tracker.createInstance<Arg_pipe_listContext>(_ctx, getState());
  enterRule(_localctx, 20, FCParser::RuleArg_pipe_list);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    setState(189);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case FCParser::STRING:
      case FCParser::PIPE_STR_STR: {
        enterOuterAlt(_localctx, 1);
        setState(187);
        str_pipe_list();
        break;
      }

      case FCParser::INT_STRING:
      case FCParser::PIPE_INT_STR: {
        enterOuterAlt(_localctx, 2);
        setState(188);
        int_pipe_list();
        break;
      }

    default:
      throw NoViableAltException(this);
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
  enterRule(_localctx, 22, FCParser::RuleField_name);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(191);
    match(FCParser::ID);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Function_nameContext ------------------------------------------------------------------

FCParser::Function_nameContext::Function_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* FCParser::Function_nameContext::ID() {
  return getToken(FCParser::ID, 0);
}


size_t FCParser::Function_nameContext::getRuleIndex() const {
  return FCParser::RuleFunction_name;
}

void FCParser::Function_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFunction_name(this);
}

void FCParser::Function_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<FCListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFunction_name(this);
}


std::any FCParser::Function_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<FCVisitor*>(visitor))
    return parserVisitor->visitFunction_name(this);
  else
    return visitor->visitChildren(this);
}

FCParser::Function_nameContext* FCParser::function_name() {
  Function_nameContext *_localctx = _tracker.createInstance<Function_nameContext>(_ctx, getState());
  enterRule(_localctx, 24, FCParser::RuleFunction_name);

#if __cplusplus > 201703L
  auto onExit = finally([=, this] {
#else
  auto onExit = finally([=] {
#endif
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(193);
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
  enterRule(_localctx, 26, FCParser::RuleNumeric);
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
    setState(195);
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
