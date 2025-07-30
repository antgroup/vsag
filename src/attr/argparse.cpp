
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "argparse.h"

#include <antlr4-autogen/FCLexer.h>

#include "expression_visitor.h"

namespace vsag {
vsag::ExprPtr
AstParse(const std::string& filter_condition_str, AttrTypeSchema* schema) {
    antlr4::ANTLRInputStream input(filter_condition_str);
    FCLexer lexer(&input);
    antlr4::CommonTokenStream tokens(&lexer);
    FCParser parser(&tokens);

    FCErrorListener error_listener(filter_condition_str);
    lexer.removeErrorListeners();
    lexer.addErrorListener(&error_listener);
    parser.removeErrorListeners();
    parser.addErrorListener(&error_listener);
    FCExpressionVisitor visitor(schema);
    auto expr_ptr = std::any_cast<ExprPtr>(visitor.visit(parser.filter_condition()));
    return std::move(expr_ptr);
}
}  // namespace vsag
