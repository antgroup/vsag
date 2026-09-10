
// Generated from FC.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  FCLexer : public antlr4::Lexer {
public:
  enum {
    T__0 = 1, T__1 = 2, T__2 = 3, T__3 = 4, T__4 = 5, FUNCTION = 6, REGION_FILTER = 7, 
    AND = 8, OR = 9, NOT = 10, IN = 11, NOT_IN = 12, EQ = 13, NQ = 14, GT = 15, 
    LT = 16, GE = 17, LE = 18, MUL = 19, DIV = 20, ADD = 21, SUB = 22, ID = 23, 
    INTEGER = 24, SEP = 25, SEP_STR = 26, INT_STRING = 27, STRING = 28, 
    PIPE_INT_STR = 29, PIPE_STR_STR = 30, FLOAT = 31, WS = 32, LINE_COMMENT = 33
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

