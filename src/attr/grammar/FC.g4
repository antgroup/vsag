grammar FC;

filter_condition: expr EOF;

expr: '(' expr ')'                                     # parenExpr
    | NOT '(' expr ')'                                 # notExpr
    | left=expr op=(AND|OR) right=expr                 # logicalExpr
    | comparison                                       # compExpr
    ;

comparison: op=(NOT_IN|IN) '(' field_name ',' int_pipe_list ')'                        # intPipeListExpr
          | op=(NOT_IN|IN) '(' field_name ',' int_pipe_list ',' SEP_STR ')'            # intPipeListExpr
          | op=(NOT_IN|IN) '(' field_name ',' str_pipe_list ')'                        # strPipeListExpr
          | op=(NOT_IN|IN) '(' field_name ',' str_pipe_list ',' SEP_STR ')'            # strPipeListExpr
          | field_name op=(NOT_IN|IN) int_value_list                                   # intListExpr
          | field_name op=(NOT_IN|IN) str_value_list                                   # strListExpr
          | field_expr op=comparison_op numeric                                        # numericComparison
          | field_name op=comparison_sop (STRING | INT_STRING)                         # stringComparison
          ;

field_expr: field_expr op=(MUL|DIV) field_expr   # arithmeticExpr
          | field_expr op=(ADD|SUB) field_expr   # arithmeticExpr
          | field_name                           # fieldRef
          | numeric                              # numericConst
          | '(' field_expr ')'                   # parenFieldExpr
          ;

comparison_sop: EQ | NQ;

comparison_op: EQ | NQ | GT | LT | GE | LE;

int_value_list: '[' INTEGER (',' INTEGER)* ']';
int_pipe_list: PIPE_INT_STR | INT_STRING;

str_value_list: '[' STRING (',' STRING)* ']';
str_pipe_list: PIPE_ID_STR | STRING;

field_name: ID;

numeric: INTEGER | FLOAT;

// 词法规则
AND: 'AND' | 'and' | '&&';
OR: 'OR' | 'or' | '||';
NOT: '!';
IN: 'IN' | 'in' | 'MULTI_IN' | 'multi_in';
NOT_IN: 'NOT_IN' | 'not_in' | 'NOTIN' | 'notin' | 'MULTI_NOTIN' | 'multi_notin';
EQ: '=';
NQ: '!=';
GT: '>';
LT: '<';
GE: '>=';
LE: '<=';

MUL: '*';
DIV: '/';
ADD: '+';
SUB: '-';


ID: [a-zA-Z_] [a-zA-Z0-9_]* ('.' [a-zA-Z_] [a-zA-Z0-9_]*)*;
INTEGER : ('+' | '-')?[1-9][0-9]* | '0';
SEP: '|';
SEP_STR: '"' SEP '"';
INT_STRING: '"' INTEGER '"';
STRING: '"' (~["|])* '"';
PIPE_INT_STR:'"' INTEGER (SEP INTEGER)* '"';
PIPE_ID_STR: '"' ID (SEP ID)* '"';
FLOAT : ('+' | '-')? (DIGIT+ '.' DIGIT* | '.' DIGIT+) ([eE] ('+' | '-')? DIGIT+)?;
fragment DIGIT : [0-9];

WS: [ \t\r\n]+ -> skip;

LINE_COMMENT: '#' ~[\r\n]* -> skip;
