
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

/**
 * @file argparse.h
 * @brief AST parsing utilities for filter condition strings.
 *
 * This file provides the AstParse function for parsing filter condition
 * strings into expression AST using ANTLR4.
 */

#pragma once

#include "attr_type_schema.h"
#include "expression.h"

namespace vsag {

/**
 * @brief Parses a filter condition string into an expression AST.
 *
 * This function uses ANTLR4 to parse the filter condition string and
 * constructs an expression AST that can be evaluated for filtering.
 *
 * @param filter_condition_str The filter condition string to parse.
 * @param schema Optional pointer to the attribute type schema for type checking.
 *              If nullptr, type checking is not performed.
 * @return ExprPtr Smart pointer to the root of the expression AST.
 * @throws std::runtime_error If the filter condition string has syntax errors.
 */
ExprPtr
AstParse(const std::string& filter_condition_str, AttrTypeSchema* schema = nullptr);

}  // namespace vsag
