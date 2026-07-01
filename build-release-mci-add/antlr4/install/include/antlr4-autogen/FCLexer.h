
// Generated from FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  FCLexer : public antlr4::Lexer {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, AND = 6, OR = 7, NOT = 8, 
    IN = 9, NOT_IN = 10, EQ = 11, NQ = 12, GT = 13, LT = 14, GE = 15, LE = 16, 
    MUL = 17, DIV = 18, ADD = 19, SUB = 20, ID = 21, INTEGER = 22, SEP = 23, 
    SEP_STR = 24, INT_STRING = 25, STRING = 26, PIPE_INT_STR = 27, PIPE_STR_STR = 28, 
    FLOAT = 29, WS = 30, LINE_COMMENT = 31
  };

  explicit FCLexer(antlr4::CharStream *input);

  ~FCLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

