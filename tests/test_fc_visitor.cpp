
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

#include <antlr4-autogen/FCLexer.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#define EOF (-1)
#include <nlohmann/json.hpp>
#undef EOF

#include "attr/argparse.h"
#include "attr/expression_visitor.h"
#include "impl/allocator/safe_allocator.h"
#include "vsag_exception.h"

using namespace vsag;

TEST_CASE("Test BaseVisitor", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = "age > 18";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = "(age + 5) > 18";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = "age > 18.0";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = R"(name = "Alice")";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = R"(!(name = "Alice"))";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = R"(name in ["Alice", "Bob"])";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = R"(multi_notin(name, "Alice|Bob", "|"))";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
    {
        auto filter_condition_str = R"(multi_notin(age, "12|13", "|"))";
        antlr4::ANTLRInputStream input(filter_condition_str);
        FCLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        FCParser parser(&tokens);

        FCBaseVisitor visitor;
        REQUIRE_NOTHROW(visitor.visit(parser.filter_condition()));
    }
}

TEST_CASE("Test NumericComparison", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = "age > 18";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GT('>'), numeric -> INTEGER('18')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">");
        REQUIRE(comparison_expr->right->ToString() == "18");
        REQUIRE(comparison_expr->ToString() == "(age > 18)");
    }
    {
        auto filter_condition_str = "age >= 18.5";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GT('>'), numeric -> FLOAT('18.5')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">=");
        REQUIRE(comparison_expr->right->ToString() == "18.5");
        REQUIRE(comparison_expr->ToString() == "(age >= 18.5)");
    }
    {
        auto filter_condition_str = "age < 18";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LT('<'), numeric -> INTEGER('18')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<");
        REQUIRE(comparison_expr->right->ToString() == "18");
        REQUIRE(comparison_expr->ToString() == "(age < 18)");
    }
    {
        auto filter_condition_str = "age <= 18.5";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> FLOAT('18.5')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "18.5");
        REQUIRE(comparison_expr->ToString() == "(age <= 18.5)");
    }
    {
        auto filter_condition_str = "age = 18";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> EQ('='), numeric -> INTEGER('18')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "18");
        REQUIRE(comparison_expr->ToString() == "(age = 18)");
    }
    {
        auto filter_condition_str = "age != 18.5";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> NQ('!='), numeric -> FLOAT('18.5')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "!=");
        REQUIRE(comparison_expr->right->ToString() == "18.5");
        REQUIRE(comparison_expr->ToString() == "(age != 18.5)");
    }
    {
        auto filter_condition_str = "age > 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GT('>'), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age > 0)");
    }
    {
        auto filter_condition_str = "age < 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LT('<'), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age < 0)");
    }
    {
        auto filter_condition_str = "age = 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> EQ('='), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age = 0)");
    }
    {
        auto filter_condition_str = "age != 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> NQ('!='), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "!=");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age != 0)");
    }
    {
        auto filter_condition_str = "age >= 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GE('>='), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">=");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age >= 0)");
    }
    {
        auto filter_condition_str = "age <= 0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> INTEGER('0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "0");
        REQUIRE(comparison_expr->ToString() == "(age <= 0)");
    }
    {
        auto filter_condition_str = "age > 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GT('>'), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age > 18.0)");
    }
    {
        auto filter_condition_str = "age < 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LT('<'), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age < 18.0)");
    }
    {
        auto filter_condition_str = "age = 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> EQ('='), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age = 18.0)");
    }
    {
        auto filter_condition_str = "age != 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> NQ('!='), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "!=");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age != 18.0)");
    }
    {
        auto filter_condition_str = "age >= 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GE('>='), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">=");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age >= 18.0)");
    }
    {
        auto filter_condition_str = "age <= 18.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> FLOAT('18.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "18.0");
        REQUIRE(comparison_expr->ToString() == "(age <= 18.0)");
    }
    {
        auto filter_condition_str = "age <= .15";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> FLOAT('0.15')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "0.15");
        REQUIRE(comparison_expr->ToString() == "(age <= 0.15)");
    }
    {
        auto filter_condition_str = "age <= 15.";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> FLOAT('15.')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "15.0");
        REQUIRE(comparison_expr->ToString() == "(age <= 15.0)");
    }
    {
        auto filter_condition_str = "age <= 15.23e-4";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LE('<='), numeric -> FLOAT('15.23e-4')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<=");
        REQUIRE(comparison_expr->right->ToString() == "0.001523");
        REQUIRE(comparison_expr->ToString() == "(age <= 0.001523)");
    }
    // 边界检查
    {
        auto filter_condition_str = "age > 2147483647";  // INT_MAX
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> GT('>'), numeric -> INTEGER('2147483647')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == ">");
        REQUIRE(comparison_expr->right->ToString() == "2147483647");
        REQUIRE(comparison_expr->ToString() == "(age > 2147483647)");
    }
    {
        auto filter_condition_str = "age < -2147483648";  // INT_MIN
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> LT('<'), numeric -> INTEGER('-2147483648')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "<");
        REQUIRE(comparison_expr->right->ToString() == "-2147483648");
        REQUIRE(comparison_expr->ToString() == "(age < -2147483648)");
    }
    {
        auto filter_condition_str = "age = 0.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> EQ('='), numeric -> FLOAT('0.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "0.0");
        REQUIRE(comparison_expr->ToString() == "(age = 0.0)");
    }
    {
        auto filter_condition_str = "age != 0.0";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_expr -> field_name -> ID('age'), comparison_op -> NQ('!='), numeric -> FLOAT('0.0')
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "age");
        REQUIRE(ToString(comparison_expr->op) == "!=");
        REQUIRE(comparison_expr->right->ToString() == "0.0");
        REQUIRE(comparison_expr->ToString() == "(age != 0.0)");
    }
    // 错误语法判断
    {
        auto filter_condition_str = "age >";  // 缺少右侧数值
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age 18";  // 缺少比较操作符
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age > 18.5.5";  // 错误的浮点数格式
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age > 18 19";  // 多余的数值
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age > 18 AND";  // 缺少右侧表达式
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age > 18 AND age";  // 缺少右侧比较操作符和数值
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }
    {
        auto filter_condition_str = "age > 18 AND age >";  // 缺少右侧数值
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }

    {
        auto filter_condition_str = "age >= 18.0";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }

    {
        auto filter_condition_str = "age >= 18";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", UINT64);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
    }
}

TEST_CASE("Test ArithmeticExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"((age + 10) = 60)";
        auto expr_ptr = AstParse(filter_condition_str);
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "(age + 10)");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "60");
        REQUIRE(comparison_expr->ToString() == "((age + 10) = 60)");
    }
    {
        auto filter_condition_str = R"((age - 10) = 60)";
        auto expr_ptr = AstParse(filter_condition_str);
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "(age - 10)");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "60");
        REQUIRE(comparison_expr->ToString() == "((age - 10) = 60)");
    }
    {
        auto filter_condition_str = R"((age * 10) = 60)";
        auto expr_ptr = AstParse(filter_condition_str);
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "(age * 10)");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "60");
        REQUIRE(comparison_expr->ToString() == "((age * 10) = 60)");
    }
    {
        auto filter_condition_str = R"((age / 10) = 6)";
        auto expr_ptr = AstParse(filter_condition_str);
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "(age / 10)");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "6");
        REQUIRE(comparison_expr->ToString() == "((age / 10) = 6)");
    }

    {
        auto filter_condition_str = R"((age / 10) = 6)";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }

    {
        auto filter_condition_str = R"((age / 10) = 6)";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", INT32);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
    }
}

TEST_CASE("Test NotExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"(!(name = "Alice"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto expr = std::dynamic_pointer_cast<NotExpression>(expr_ptr);
        REQUIRE(expr != nullptr);
        REQUIRE(expr->ToString() == "! ((name = \"Alice\"))");
        REQUIRE(expr->GetExprType() == ExpressionType::kNotExpression);
    }

    {
        auto filter_condition_str = R"(!(name = "Alice"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }
}

TEST_CASE("Test StringComparison", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"(name = "Alice")";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), comparison_sop -> EQ('='), STRING -> "Alice"
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "name");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "\"Alice\"");
        REQUIRE(comparison_expr->ToString() == "(name = \"Alice\")");
    }

    {
        auto filter_condition_str = R"(name != "Alice")";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), comparison_sop -> NE('!='), STRING -> "Alice"
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "name");
        REQUIRE(ToString(comparison_expr->op) == "!=");
        REQUIRE(comparison_expr->right->ToString() == "\"Alice\"");
        REQUIRE(comparison_expr->ToString() == "(name != \"Alice\")");
    }

    {
        auto filter_condition_str = R"(name = "")";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), comparison_sop -> EQ('='), STRING -> ""
        auto comparison_expr = std::dynamic_pointer_cast<ComparisonExpression>(expr_ptr);
        REQUIRE(comparison_expr != nullptr);
        REQUIRE(comparison_expr->left->ToString() == "name");
        REQUIRE(ToString(comparison_expr->op) == "=");
        REQUIRE(comparison_expr->right->ToString() == "\"\"");
        REQUIRE(comparison_expr->ToString() == "(name = \"\")");
    }
    {
        auto filter_condition_str = R"(name = ")";  // 缺少右侧"
        REQUIRE_THROWS_AS(AstParse(filter_condition_str), std::runtime_error);
    }

    {
        auto filter_condition_str = R"(name != "Alice")";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }

    {
        auto filter_condition_str = R"(name != "Alice")";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
    }

    {
        auto filter_condition_str = R"(name != "Alice")";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), vsag::VsagException);
    }
}

TEST_CASE("Test InStrListExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"(name IN ["Alice", "Bob"])";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), IN('IN'), str_value_list -> [STRING("Alice"), STRING("Bob")]
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "name");
        REQUIRE_FALSE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"Alice\", \"Bob\"]");
        REQUIRE(str_list_expr->ToString() == "(name IN [\"Alice\", \"Bob\"])");
    }

    {
        auto filter_condition_str = R"(age IN ["16", "18"])";
        auto expr_ptr = AstParse(filter_condition_str);
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "age");
        REQUIRE_FALSE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"16\", \"18\"]");
        REQUIRE(str_list_expr->ToString() == "(age IN [\"16\", \"18\"])");
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "Alice|Bob", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), IN('IN'), str_value_list -> [STRING("Alice"), STRING("Bob")]
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "name");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"Alice\", \"Bob\"]");
        REQUIRE(str_list_expr->ToString() == "(name NOT_IN [\"Alice\", \"Bob\"])");
    }

    {
        auto filter_condition_str = R"(multi_notin(age, "18", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto str_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "age");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[18]");
        REQUIRE(str_list_expr->ToString() == "(age NOT_IN [18])");
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "\"Alice\"|\"Bob\"", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('name'), IN('IN'), str_value_list -> [STRING("Alice"), STRING("Bob")]
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "name");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == R"(["\"Alice\"", "\"Bob\""])");
        REQUIRE(str_list_expr->ToString() == R"((name NOT_IN ["\"Alice\"", "\"Bob\""]))");
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "Alice", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "name");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"Alice\"]");
        REQUIRE(str_list_expr->ToString() == "(name NOT_IN [\"Alice\"])");
    }
    {
        auto filter_condition_str = R"(name IN ["Alice", "Bob"])";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "Alice", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "Alice", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("name", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
    }

    {
        auto filter_condition_str = R"(multi_notin(name, "Alice", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), vsag::VsagException);
    }

    {
        auto filter_condition_str = R"(multi_notin(age, "123|456|789", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "age");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"123\", \"456\", \"789\"]");
        REQUIRE(str_list_expr->ToString() == "(age NOT_IN [\"123\", \"456\", \"789\"])");
    }

    {
        auto filter_condition_str = R"(multi_notin(age, "123_123|456_123|789_123", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "age");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"123_123\", \"456_123\", \"789_123\"]");
        REQUIRE(str_list_expr->ToString() ==
                "(age NOT_IN [\"123_123\", \"456_123\", \"789_123\"])");
    }
    {
        auto filter_condition_str = R"(multi_notin(age, "'123'|'456'|'789'", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "age");
        REQUIRE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"'123'\", \"'456'\", \"'789'\"]");
        REQUIRE(str_list_expr->ToString() == "(age NOT_IN [\"'123'\", \"'456'\", \"'789'\"])");
    }

    {
        auto filter_condition_str = R"(multi_notin(age, "'123'|'456'|'789'", "|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("age", INT32);
        REQUIRE_THROWS_AS(AstParse(filter_condition_str, schema.get()), std::runtime_error);
    }
}

TEST_CASE("Test FilterExpression", "[ft][expression_visitor]") {
    auto filter_condition_str =
        R"(region_filter(region_type, region_list, residence_tag_list, "-1|5598718269230453874|6538065452722828466", "-1|7860305485634463544|3217858936663843930", "-1|424535459702720964|7847373033918912390"))";
    auto expr_ptr = AstParse(filter_condition_str);
    auto filter_expr = std::dynamic_pointer_cast<RegionFilterExpression>(expr_ptr);
    REQUIRE(expr_ptr != nullptr);
    REQUIRE(filter_expr->region_type->ToString() == "region_type");
    REQUIRE(filter_expr->region_list->ToString() == "region_list");
    REQUIRE(filter_expr->residence_tag_list->ToString() == "residence_tag_list");
    REQUIRE(filter_expr->arg0->ToString() == "[-1, 5598718269230453874, 6538065452722828466]");
    REQUIRE(filter_expr->arg1->ToString() == "[-1, 7860305485634463544, 3217858936663843930]");
    REQUIRE(filter_expr->arg2->ToString() == "[-1, 424535459702720964, 7847373033918912390]");
}

TEST_CASE("Test FunctionExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"(function(time_filter, "", ""))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto function_expr = std::dynamic_pointer_cast<FunctionExpression>(expr_ptr);
        REQUIRE(function_expr != nullptr);
        REQUIRE(function_expr->ToString() == "FUNCTION (time_filter,\"\",\"\")");
        REQUIRE(function_expr->function_name == "time_filter");
        REQUIRE(function_expr->arg_values.empty());
        REQUIRE(function_expr->arg_types.empty());
    }
    {
        auto filter_condition_str = R"(function(time_filter, "1", "int32"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto function_expr = std::dynamic_pointer_cast<FunctionExpression>(expr_ptr);
        REQUIRE(function_expr != nullptr);
        REQUIRE(function_expr->ToString() == "FUNCTION (time_filter,\"1\",\"int32\")");
        REQUIRE(function_expr->function_name == "time_filter");
        REQUIRE(function_expr->arg_values.size() == 1);
        REQUIRE(function_expr->arg_types.size() == 1);
    }

    {
        auto filter_condition_str = R"(function(time_filter, "1|abc", "int32|string"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto function_expr = std::dynamic_pointer_cast<FunctionExpression>(expr_ptr);
        REQUIRE(function_expr != nullptr);
        REQUIRE(function_expr->ToString() == "FUNCTION (time_filter,\"1|abc\",\"int32|string\")");
        REQUIRE(function_expr->function_name == "time_filter");
        REQUIRE(function_expr->arg_values.size() == 2);
        REQUIRE(function_expr->arg_types.size() == 2);
    }

    {
        auto filter_condition_str =
            R"(function(time_filter, "8|1|121|-220000000000|abc|220000000000|attributeA|placeholder", "uint16|int16|int32|int64|string|uint64|attribute|inner_id"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto function_expr = std::dynamic_pointer_cast<FunctionExpression>(expr_ptr);
        REQUIRE(function_expr != nullptr);
        REQUIRE(function_expr->ToString() ==
                "FUNCTION "
                "(time_filter,\"8|1|121|-220000000000|abc|220000000000|attributeA|placeholder\","
                "\"uint16|int16|int32|int64|string|uint64|attribute|inner_id\")");
        REQUIRE(function_expr->function_name == "time_filter");
        REQUIRE(function_expr->arg_values.size() == 8);
        REQUIRE(function_expr->arg_types.size() == 8);
        REQUIRE(function_expr->GetArg<std::uint16_t>(0).value() == 8);
        REQUIRE(function_expr->GetArg<std::int16_t>(1).value() == 1);
        REQUIRE(function_expr->GetArg<std::int16_t>(2).has_value() == false);
    }

    {
        auto filter_condition_str = R"(function(time_filter, "-1|abc", "uint32|string"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = R"(function(time_filter, "220000000000|abc", "int32|string"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = R"(function(time_filter, "220000000000", "inner_id"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = R"(function(time_filter, "placeholder", "inner_id|int32"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = R"(function(time_filter, "1|abc", "int8|string"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }
}

TEST_CASE("Test LogicalExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = R"((age >= 18) and (age <= 60))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto logic_expr = std::dynamic_pointer_cast<LogicalExpression>(expr_ptr);
        REQUIRE(logic_expr != nullptr);
        REQUIRE(logic_expr->left->ToString() == "(age >= 18)");
        REQUIRE(logic_expr->right->ToString() == "(age <= 60)");
        REQUIRE(logic_expr->ToString() == "((age >= 18) AND (age <= 60))");
    }

    {
        auto filter_condition_str = R"((age >= 18) and (name = "Alice"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto logic_expr = std::dynamic_pointer_cast<LogicalExpression>(expr_ptr);
        REQUIRE(logic_expr != nullptr);
        REQUIRE(logic_expr->left->ToString() == "(age >= 18)");
        REQUIRE(logic_expr->right->ToString() == "(name = \"Alice\")");
        REQUIRE(logic_expr->ToString() == "((age >= 18) AND (name = \"Alice\"))");
    }

    {
        auto filter_condition_str =
            R"((id >= 9223372036854775808) and (id <= 18446744073709551615))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto logic_expr = std::dynamic_pointer_cast<LogicalExpression>(expr_ptr);
        REQUIRE(logic_expr != nullptr);
        REQUIRE(logic_expr->left->ToString() == "(id >= 9223372036854775808)");
        REQUIRE(logic_expr->right->ToString() == "(id <= 18446744073709551615)");
        REQUIRE(logic_expr->ToString() ==
                "((id >= 9223372036854775808) AND (id <= 18446744073709551615))");
    }
}

TEST_CASE("Test InIntListExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str = "id IN [1, 2, 3]";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('id'), IN('IN'), int_value_list -> [INTEGER('1'), INTEGER('2'), INTEGER('3')]
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[1, 2, 3]");
        REQUIRE(int_list_expr->ToString() == "(id IN [1, 2, 3])");
    }

    {
        auto filter_condition_str = "id IN [1, 2, 3]";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("id", INT32);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        // AST 结构: comparison -> field_name -> ID('id'), IN('IN'), int_value_list -> [INTEGER('1'), INTEGER('2'), INTEGER('3')]
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[1, 2, 3]");
        REQUIRE(int_list_expr->ToString() == "(id IN [1, 2, 3])");
    }

    {
        auto filter_condition_str = "id IN [1, 2, 3]";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("id", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() == "[\"1\", \"2\", \"3\"]");
        REQUIRE(str_list_expr->ToString() == "(id IN [\"1\", \"2\", \"3\"])");
    }

    {
        auto filter_condition_str =
            "id IN [9223372036854775808, 18446744073709551615, 13835058055282163712]";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() ==
                "[9223372036854775808, 18446744073709551615, 13835058055282163712]");
        REQUIRE(int_list_expr->ToString() ==
                "(id IN [9223372036854775808, 18446744073709551615, 13835058055282163712])");
    }

    {
        auto filter_condition_str = "id IN [9223372036854775808, -1, 3]";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = "id IN [9223372036854775808]";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[9223372036854775808]");
        REQUIRE(int_list_expr->ToString() == "(id IN [9223372036854775808])");
    }

    {
        auto filter_condition_str = R"(multi_in(id, "1|2|3", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('id'), IN('IN'), int_value_list -> [INTEGER('1'), INTEGER('2'), INTEGER('3')]
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[1, 2, 3]");
        REQUIRE(int_list_expr->ToString() == "(id IN [1, 2, 3])");
    }

    {
        auto filter_condition_str = R"(multi_in(id, "9223372036854775808|-2|3", "|"))";
        REQUIRE_THROWS(AstParse(filter_condition_str));
    }

    {
        auto filter_condition_str = R"(multi_in(id, "9223372036854775808", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[9223372036854775808]");
        REQUIRE(int_list_expr->ToString() == "(id IN [9223372036854775808])");
    }

    {
        auto filter_condition_str = R"(multi_in(id, "1", "|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('id'), IN('IN'), int_value_list -> [INTEGER('1'), INTEGER('2'), INTEGER('3')]
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[1]");
        REQUIRE(int_list_expr->ToString() == "(id IN [1])");
    }

    {
        auto filter_condition_str = "id IN [1, 2, 3, 4]";
        auto expr_ptr = AstParse(filter_condition_str);
        // AST 结构: comparison -> field_name -> ID('id'), IN('IN'), int_value_list -> [INTEGER('1'), INTEGER('2'), INTEGER('3'), INTEGER('4')]
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() == "[1, 2, 3, 4]");
        REQUIRE(int_list_expr->ToString() == "(id IN [1, 2, 3, 4])");
    }
}

TEST_CASE("Test LongMultiInExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str =
            R"(multi_notin(rta_uniq_id,"1961100458546670265|8669342430913282238|4295511754643557149|3820165547219126739|1615416169623352709|6314578306465646238|8519232450117756233|2756461647811536678|7246372942013239063|6016028506275409733|3202937297212420206|7510237410486928689|2693897507874692550|7942461091316593098|8468109175693762333|2246205382807091008|4762936000914802423|6797061107232090018|8530742052452506732|3327382205450771314|6522909056849751446|2097254881891107942|9081283293213749162|7659202351373227815|2426484522154290531|9028175612421629988|3979294040747128108|7616191334060074064|2216336493100479083|6983229653395630143|7061041066005578122|3196243134178854499|1239316581789112809|4146619685878866397|2993398114247915385|5239079003920071893|7306779578940296486|8664775826449197998|7292236495175168278|3792060204756314668|943879882149209231|67786373045767393|5460974927050884095|3985833378458068290|7260385150729580290|1471584193443402819|1093004338684365583|3888227820969699574|2263550131772636927|4925893694984496472|4053461338322378105|2309747396698563821|6079754561320324176|160623819949859655|2024781313054920143|8988650033226888193|6672705482565606954|3463594972398901639|4011710098241765573|2628278965639395857|7430455255531202418|9127158149800262638|6615696035589714492|6463699161937835373|4524875330387178960|8052583455002209669|7438806969338678071|765272482175117030|3216643116160913287|8937191440708183065|3599927177086139360|4603694906434758431|4839962215257436741|7951576000074314225|5404201438195616115|1933117509111472906|4420843354467922436|6949401081268099700|8286688749939713010|7471024169212191085|5756165102291555444|7855894148486084345|5425421488718376732|4212694001066848701|9207657085406962396|3201372394618115663|9089499071344019177|4485876234743098538|4261325010202841032|8937647154994060117|5363833089632775568|8563220203728265761|8757703765922223628|7720373886526897000|6940990643005730592|1422105493879373245|2371341722325808920|7660148443976918122|6952169319874687609|7406880929185273462|2114067842011407513|189373988687663231|8153479507569108764|3355979313800702697|9171725698747800278|8777056532130596413|3339864690631806628|3727437985819132169|4413544153042575492|7498777981318184046|7355443397257156893|5999546154177100186|3412685935113877458|2571149021240417685|8457673131098490986|5749328167325148894|5341841126750918352|257263441440370798","|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "rta_uniq_id");
        REQUIRE(int_list_expr->is_not_in);
        REQUIRE(
            int_list_expr->values->ToString() ==
            "[1961100458546670265, 8669342430913282238, 4295511754643557149, 3820165547219126739, "
            "1615416169623352709, 6314578306465646238, 8519232450117756233, 2756461647811536678, "
            "7246372942013239063, 6016028506275409733, 3202937297212420206, 7510237410486928689, "
            "2693897507874692550, 7942461091316593098, 8468109175693762333, 2246205382807091008, "
            "4762936000914802423, 6797061107232090018, 8530742052452506732, 3327382205450771314, "
            "6522909056849751446, 2097254881891107942, 9081283293213749162, 7659202351373227815, "
            "2426484522154290531, 9028175612421629988, 3979294040747128108, 7616191334060074064, "
            "2216336493100479083, 6983229653395630143, 7061041066005578122, 3196243134178854499, "
            "1239316581789112809, 4146619685878866397, 2993398114247915385, 5239079003920071893, "
            "7306779578940296486, 8664775826449197998, 7292236495175168278, 3792060204756314668, "
            "943879882149209231, 67786373045767393, 5460974927050884095, 3985833378458068290, "
            "7260385150729580290, 1471584193443402819, 1093004338684365583, 3888227820969699574, "
            "2263550131772636927, 4925893694984496472, 4053461338322378105, 2309747396698563821, "
            "6079754561320324176, 160623819949859655, 2024781313054920143, 8988650033226888193, "
            "6672705482565606954, 3463594972398901639, 4011710098241765573, 2628278965639395857, "
            "7430455255531202418, 9127158149800262638, 6615696035589714492, 6463699161937835373, "
            "4524875330387178960, 8052583455002209669, 7438806969338678071, 765272482175117030, "
            "3216643116160913287, 8937191440708183065, 3599927177086139360, 4603694906434758431, "
            "4839962215257436741, 7951576000074314225, 5404201438195616115, 1933117509111472906, "
            "4420843354467922436, 6949401081268099700, 8286688749939713010, 7471024169212191085, "
            "5756165102291555444, 7855894148486084345, 5425421488718376732, 4212694001066848701, "
            "9207657085406962396, 3201372394618115663, 9089499071344019177, 4485876234743098538, "
            "4261325010202841032, 8937647154994060117, 5363833089632775568, 8563220203728265761, "
            "8757703765922223628, 7720373886526897000, 6940990643005730592, 1422105493879373245, "
            "2371341722325808920, 7660148443976918122, 6952169319874687609, 7406880929185273462, "
            "2114067842011407513, 189373988687663231, 8153479507569108764, 3355979313800702697, "
            "9171725698747800278, 8777056532130596413, 3339864690631806628, 3727437985819132169, "
            "4413544153042575492, 7498777981318184046, 7355443397257156893, 5999546154177100186, "
            "3412685935113877458, 2571149021240417685, 8457673131098490986, 5749328167325148894, "
            "5341841126750918352, 257263441440370798]");
    }

    {
        auto filter_condition_str =
            R"(multi_notin(rta_uniq_id,"1961100458546670265|8669342430913282238|4295511754643557149|3820165547219126739|1615416169623352709|6314578306465646238|8519232450117756233|2756461647811536678|7246372942013239063|6016028506275409733|3202937297212420206|7510237410486928689|2693897507874692550|7942461091316593098|8468109175693762333|2246205382807091008|4762936000914802423|6797061107232090018|8530742052452506732|3327382205450771314|6522909056849751446|2097254881891107942|9081283293213749162|7659202351373227815|2426484522154290531|9028175612421629988|3979294040747128108|7616191334060074064|2216336493100479083|6983229653395630143|7061041066005578122|3196243134178854499|1239316581789112809|4146619685878866397|2993398114247915385|5239079003920071893|7306779578940296486|8664775826449197998|7292236495175168278|3792060204756314668|943879882149209231|67786373045767393|5460974927050884095|3985833378458068290|7260385150729580290|1471584193443402819|1093004338684365583|3888227820969699574|2263550131772636927|4925893694984496472|4053461338322378105|2309747396698563821|6079754561320324176|160623819949859655|2024781313054920143|8988650033226888193|6672705482565606954|3463594972398901639|4011710098241765573|2628278965639395857|7430455255531202418|9127158149800262638|6615696035589714492|6463699161937835373|4524875330387178960|8052583455002209669|7438806969338678071|765272482175117030|3216643116160913287|8937191440708183065|3599927177086139360|4603694906434758431|4839962215257436741|7951576000074314225|5404201438195616115|1933117509111472906|4420843354467922436|6949401081268099700|8286688749939713010|7471024169212191085|5756165102291555444|7855894148486084345|5425421488718376732|4212694001066848701|9207657085406962396|3201372394618115663|9089499071344019177|4485876234743098538|4261325010202841032|8937647154994060117|5363833089632775568|8563220203728265761|8757703765922223628|7720373886526897000|6940990643005730592|1422105493879373245|2371341722325808920|7660148443976918122|6952169319874687609|7406880929185273462|2114067842011407513|189373988687663231|8153479507569108764|3355979313800702697|9171725698747800278|8777056532130596413|3339864690631806628|3727437985819132169|4413544153042575492|7498777981318184046|7355443397257156893|5999546154177100186|3412685935113877458|2571149021240417685|8457673131098490986|5749328167325148894|5341841126750918352|257263441440370798","|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("rta_uniq_id", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(list_expr != nullptr);
        REQUIRE(list_expr->field->ToString() == "rta_uniq_id");
        REQUIRE(list_expr->is_not_in);
        REQUIRE(list_expr->values->ToString() ==
                "[\"1961100458546670265\", \"8669342430913282238\", \"4295511754643557149\", "
                "\"3820165547219126739\", "
                "\"1615416169623352709\", \"6314578306465646238\", \"8519232450117756233\", "
                "\"2756461647811536678\", "
                "\"7246372942013239063\", \"6016028506275409733\", \"3202937297212420206\", "
                "\"7510237410486928689\", "
                "\"2693897507874692550\", \"7942461091316593098\", \"8468109175693762333\", "
                "\"2246205382807091008\", "
                "\"4762936000914802423\", \"6797061107232090018\", \"8530742052452506732\", "
                "\"3327382205450771314\", "
                "\"6522909056849751446\", \"2097254881891107942\", \"9081283293213749162\", "
                "\"7659202351373227815\", "
                "\"2426484522154290531\", \"9028175612421629988\", \"3979294040747128108\", "
                "\"7616191334060074064\", "
                "\"2216336493100479083\", \"6983229653395630143\", \"7061041066005578122\", "
                "\"3196243134178854499\", "
                "\"1239316581789112809\", \"4146619685878866397\", \"2993398114247915385\", "
                "\"5239079003920071893\", "
                "\"7306779578940296486\", \"8664775826449197998\", \"7292236495175168278\", "
                "\"3792060204756314668\", "
                "\"943879882149209231\", \"67786373045767393\", \"5460974927050884095\", "
                "\"3985833378458068290\", "
                "\"7260385150729580290\", \"1471584193443402819\", \"1093004338684365583\", "
                "\"3888227820969699574\", "
                "\"2263550131772636927\", \"4925893694984496472\", \"4053461338322378105\", "
                "\"2309747396698563821\", "
                "\"6079754561320324176\", \"160623819949859655\", \"2024781313054920143\", "
                "\"8988650033226888193\", "
                "\"6672705482565606954\", \"3463594972398901639\", \"4011710098241765573\", "
                "\"2628278965639395857\", "
                "\"7430455255531202418\", \"9127158149800262638\", \"6615696035589714492\", "
                "\"6463699161937835373\", "
                "\"4524875330387178960\", \"8052583455002209669\", \"7438806969338678071\", "
                "\"765272482175117030\", "
                "\"3216643116160913287\", \"8937191440708183065\", \"3599927177086139360\", "
                "\"4603694906434758431\", "
                "\"4839962215257436741\", \"7951576000074314225\", \"5404201438195616115\", "
                "\"1933117509111472906\", "
                "\"4420843354467922436\", \"6949401081268099700\", \"8286688749939713010\", "
                "\"7471024169212191085\", "
                "\"5756165102291555444\", \"7855894148486084345\", \"5425421488718376732\", "
                "\"4212694001066848701\", "
                "\"9207657085406962396\", \"3201372394618115663\", \"9089499071344019177\", "
                "\"4485876234743098538\", "
                "\"4261325010202841032\", \"8937647154994060117\", \"5363833089632775568\", "
                "\"8563220203728265761\", "
                "\"8757703765922223628\", \"7720373886526897000\", \"6940990643005730592\", "
                "\"1422105493879373245\", "
                "\"2371341722325808920\", \"7660148443976918122\", \"6952169319874687609\", "
                "\"7406880929185273462\", "
                "\"2114067842011407513\", \"189373988687663231\", \"8153479507569108764\", "
                "\"3355979313800702697\", "
                "\"9171725698747800278\", \"8777056532130596413\", \"3339864690631806628\", "
                "\"3727437985819132169\", "
                "\"4413544153042575492\", \"7498777981318184046\", \"7355443397257156893\", "
                "\"5999546154177100186\", "
                "\"3412685935113877458\", \"2571149021240417685\", \"8457673131098490986\", "
                "\"5749328167325148894\", "
                "\"5341841126750918352\", \"257263441440370798\"]");
    }

    {
        auto filter_condition_str =
            R"((multi_notin(trade_id_v2,"75|75001|78002|78005|78010|78011|77002001|77004001|77005001|77005002|77005003|77005004|77005005|77005006|77005007|77005008|77005009|77005010|77005011|77005012|77005013|77005014|77005015|77005016|77005017|77010001|77010002|77010003|77010004|77010005|77010006|77010007|77010008|77010009|77010010|77011001|77011002|77011003","|")))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "trade_id_v2");
        REQUIRE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() ==
                "[75, 75001, 78002, 78005, 78010, 78011, 77002001, 77004001, 77005001, 77005002, "
                "77005003, 77005004, 77005005, 77005006, 77005007, 77005008, 77005009, 77005010, "
                "77005011, 77005012, 77005013, 77005014, 77005015, 77005016, 77005017, 77010001, "
                "77010002, 77010003, 77010004, 77010005, 77010006, 77010007, 77010008, 77010009, "
                "77010010, 77011001, 77011002, 77011003]");
    }

    {
        auto filter_condition_str =
            R"(notin(id,"6475634789599348344|195756431|189279530|216140542|213432363|4510564146827331015|3843306077049549813|217536181|198561015|216460872|198254807|3524173534128122501|215845445|209827459|217559498|169981625|214545359|214544614|192847772|193580054"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() ==
                "[6475634789599348344, 195756431, 189279530, 216140542, 213432363, "
                "4510564146827331015, 3843306077049549813, 217536181, 198561015, 216460872, "
                "198254807, 3524173534128122501, 215845445, 209827459, 217559498, 169981625, "
                "214545359, 214544614, 192847772, 193580054]");
    }

    {
        auto filter_condition_str =
            R"(multi_in(action_type,"128|160|177|129|116|117|118|120|138|156","|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "action_type");
        REQUIRE_FALSE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() ==
                "[128, 160, 177, 129, 116, 117, 118, 120, 138, 156]");
    }

    {
        auto filter_condition_str = R"(multi_in(action_type,"128|160|-1|-10","|"))";
        auto allocator = SafeAllocator::FactoryDefaultAllocator();
        auto schema = std::make_unique<AttrTypeSchema>(allocator.get());
        schema->SetTypeOfField("action_type", STRING);
        REQUIRE_NOTHROW(AstParse(filter_condition_str, schema.get()));
        auto expr_ptr = AstParse(filter_condition_str, schema.get());
        auto list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(list_expr != nullptr);
        REQUIRE(list_expr->field->ToString() == "action_type");
        REQUIRE_FALSE(list_expr->is_not_in);
        REQUIRE(list_expr->values->ToString() == "[\"128\", \"160\", \"-1\", \"-10\"]");
    }

    {
        auto filter_condition_str =
            R"(multi_in(tp_id_list,"TPID_1|TPID_12|TPID_17|TPID_20|TPID_21|TPID_22|TPID_24|TPID_45|TPID_46|TPID_48|TPID_50|TPID_51|TPID_53|TPID_59|TPID_61|TPID_62|TPID_63|TPID_65|TPID_67|TPID_75|TPID_76|TPID_8|TPID_89|TPID_91|TPID_92|TPID_95|TPID_96","|"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto str_list_expr = std::dynamic_pointer_cast<StrListExpression>(expr_ptr);
        REQUIRE(str_list_expr != nullptr);
        REQUIRE(str_list_expr->field->ToString() == "tp_id_list");
        REQUIRE_FALSE(str_list_expr->is_not_in);
        REQUIRE(str_list_expr->values->ToString() ==
                "[\"TPID_1\", \"TPID_12\", \"TPID_17\", \"TPID_20\", \"TPID_21\", \"TPID_22\", "
                "\"TPID_24\", \"TPID_45\", \"TPID_46\", \"TPID_48\", \"TPID_50\", \"TPID_51\", "
                "\"TPID_53\", \"TPID_59\", \"TPID_61\", \"TPID_62\", \"TPID_63\", \"TPID_65\", "
                "\"TPID_67\", \"TPID_75\", \"TPID_76\", \"TPID_8\", \"TPID_89\", \"TPID_91\", "
                "\"TPID_92\", \"TPID_95\", \"TPID_96\"]");
    }
}

TEST_CASE("Test ComplexExpression", "[ft][expression_visitor]") {
    {
        auto filter_condition_str =
            R"(notin(id,"6475634789599348344|195756431|189279530|216140542|213432363|4510564146827331015|3843306077049549813|217536181|198561015|216460872|198254807|3524173534128122501|215845445|209827459|217559498|169981625|214545359|214544614|192847772|193580054"))";
        auto expr_ptr = AstParse(filter_condition_str);
        auto int_list_expr = std::dynamic_pointer_cast<IntListExpression>(expr_ptr);
        REQUIRE(int_list_expr != nullptr);
        REQUIRE(int_list_expr->field->ToString() == "id");
        REQUIRE(int_list_expr->is_not_in);
        REQUIRE(int_list_expr->values->ToString() ==
                "[6475634789599348344, 195756431, 189279530, 216140542, 213432363, "
                "4510564146827331015, 3843306077049549813, 217536181, 198561015, 216460872, "
                "198254807, 3524173534128122501, 215845445, 209827459, 217559498, 169981625, "
                "214545359, 214544614, 192847772, 193580054]");
    }

    {
        auto filter_condition_str =
            R"((charge_type=5) AND (sell_mode=1) AND (balance=0) AND (plan_type!=3) AND (start_date<=1741691065051 AND end_date>=1741691065051) AND ((principal_status=0 OR sell_mode=0)) AND multi_notin(rta_uniq_id,"1961100458546670265|8669342430913282238|4295511754643557149|3820165547219126739|1615416169623352709|6314578306465646238|8519232450117756233|2756461647811536678|7246372942013239063|6016028506275409733|3202937297212420206|7510237410486928689|2693897507874692550|7942461091316593098|8468109175693762333|2246205382807091008|4762936000914802423|6797061107232090018|8530742052452506732|3327382205450771314|6522909056849751446|2097254881891107942|9081283293213749162|7659202351373227815|2426484522154290531|9028175612421629988|3979294040747128108|7616191334060074064|2216336493100479083|6983229653395630143|7061041066005578122|3196243134178854499|1239316581789112809|4146619685878866397|2993398114247915385|5239079003920071893|7306779578940296486|8664775826449197998|7292236495175168278|3792060204756314668|943879882149209231|67786373045767393|5460974927050884095|3985833378458068290|7260385150729580290|1471584193443402819|1093004338684365583|3888227820969699574|2263550131772636927|4925893694984496472|4053461338322378105|2309747396698563821|6079754561320324176|160623819949859655|2024781313054920143|8988650033226888193|6672705482565606954|3463594972398901639|4011710098241765573|2628278965639395857|7430455255531202418|9127158149800262638|6615696035589714492|6463699161937835373|4524875330387178960|8052583455002209669|7438806969338678071|765272482175117030|3216643116160913287|8937191440708183065|3599927177086139360|4603694906434758431|4839962215257436741|7951576000074314225|5404201438195616115|1933117509111472906|4420843354467922436|6949401081268099700|8286688749939713010|7471024169212191085|5756165102291555444|7855894148486084345|5425421488718376732|4212694001066848701|9207657085406962396|3201372394618115663|9089499071344019177|4485876234743098538|4261325010202841032|8937647154994060117|5363833089632775568|8563220203728265761|8757703765922223628|7720373886526897000|6940990643005730592|1422105493879373245|2371341722325808920|7660148443976918122|6952169319874687609|7406880929185273462|2114067842011407513|189373988687663231|8153479507569108764|3355979313800702697|9171725698747800278|8777056532130596413|3339864690631806628|3727437985819132169|4413544153042575492|7498777981318184046|7355443397257156893|5999546154177100186|3412685935113877458|2571149021240417685|8457673131098490986|5749328167325148894|5341841126750918352|257263441440370798","|") AND (((multi_notin(trade_id_v2,"75|75001|78002|78005|78010|78011|77002001|77004001|77005001|77005002|77005003|77005004|77005005|77005006|77005007|77005008|77005009|77005010|77005011|77005012|77005013|77005014|77005015|77005016|77005017|77010001|77010002|77010003|77010004|77010005|77010006|77010007|77010008|77010009|77010010|77011001|77011002|77011003","|")))) AND (plan_category=1) AND (ad_exp_tag=0  OR ad_exp_tag=5  OR ad_exp_tag=6) AND (industry_exact_crowd_tags=62259300007756367  OR industry_exact_crowd_tags=827821124185282642  OR industry_exact_crowd_tags=1736885184507453537  OR industry_exact_crowd_tags=1786381960166275773  OR industry_exact_crowd_tags=2447888834889859308  OR industry_exact_crowd_tags=4089293343970146178  OR industry_exact_crowd_tags=4997685626248627877  OR industry_exact_crowd_tags=5082906975180188694  OR industry_exact_crowd_tags=6233776986425357665  OR industry_exact_crowd_tags=6280213215139896165  OR industry_exact_crowd_tags=6611248751053673021  OR industry_exact_crowd_tags=6990041002335227794  OR industry_exact_crowd_tags=7824369360999110813  OR industry_exact_crowd_tags=-1) AND (ad_pos_list="202106072200013001"  OR ad_pos_list="-1") AND (age_tag_list=-1  OR age_tag_list=2776770492532234572) AND (adpos_template_id_list=582602252  OR adpos_template_id_list=582702253  OR adpos_template_id_list=586402254) AND (device_tag_list=-1) AND (gender_tag_list=-1  OR gender_tag_list=5506498716792272639) AND (media_scene_list=-1  OR media_scene_list=2) AND multi_in(tp_id_list,"TPID_1|TPID_12|TPID_17|TPID_20|TPID_21|TPID_22|TPID_24|TPID_45|TPID_46|TPID_48|TPID_50|TPID_51|TPID_53|TPID_59|TPID_61|TPID_62|TPID_63|TPID_65|TPID_67|TPID_75|TPID_76|TPID_8|TPID_89|TPID_91|TPID_92|TPID_95|TPID_96","|") AND (media_trade_list=-1  OR media_trade_list=30  OR media_trade_list=30001) AND (multi_geohash_surround="wx4gp"  OR multi_geohash_surround="wx4gpqj"  OR multi_geohash_surround="wx4gpq"  OR multi_geohash_surround="-1") AND (os_list=-1  OR os_list=8801913418331836625) AND (residence_level_list=-1  OR residence_level_list=4446806907895041723) AND (check_status=0) AND (industry_exact_crowd_tags=62259300007756367  OR industry_exact_crowd_tags=827821124185282642  OR industry_exact_crowd_tags=1736885184507453537  OR industry_exact_crowd_tags=1786381960166275773  OR industry_exact_crowd_tags=2447888834889859308  OR industry_exact_crowd_tags=4089293343970146178  OR industry_exact_crowd_tags=4997685626248627877  OR industry_exact_crowd_tags=5082906975180188694  OR industry_exact_crowd_tags=6233776986425357665  OR industry_exact_crowd_tags=6280213215139896165  OR industry_exact_crowd_tags=6611248751053673021  OR industry_exact_crowd_tags=6990041002335227794  OR industry_exact_crowd_tags=7824369360999110813  OR industry_exact_crowd_tags=-1) AND (((multi_in(task_type,"0|10000|50000|50001|60001|90000|90001|90002|110000|240000|260000|280002|280004|280005|350000|350001|350020|370000|370001|390000|460000|490001|600000|600001|600002","|") OR multi_in(action_type,"128|160|177|129|116|117|118|120|138|156","|")) AND multi_notin(action_type,"163|196|164|197|165|200|184|170|187|92|126","|"))) AND ((notin(id,"6475634789599348344|195756431|189279530|216140542|213432363|4510564146827331015|3843306077049549813|217536181|198561015|216460872|198254807|3524173534128122501|215845445|209827459|217559498|169981625|214545359|214544614|192847772|193580054") AND ((principal_id!=1227511248) OR (main_task_type!=370000)) AND ((principal_id!=1181011317) OR (main_task_type!=370000)) AND ((principal_id!=1172411226) OR (main_task_type!=370000)) AND ((principal_id!=1170311337) OR (main_task_type!=370000)))) AND notin(principal_id,"240479|2088241814009809") AND multi_notin(action_type,"92|126|163|164|165|170|184|187|196|197|200","|") AND (trade_white_list=-1  OR trade_white_list=6914483954861104026) AND ((time_schema=-1)) AND multi_notin(revert_dmp_crowd_list,"5637734191461532|10421326407713368|23367851710885002|27083629581040114|29400420043793316|30753831922038608|33294688080914456|36416540307015965|44052603451752495|48430184856964628|66721311728307473|87030316311563987|87529376695403269|90158626388090843|104033400389437772|107434936007719441|111592208306732731|133229218982973349|136508819363725343|143038867980011985|148790517211461772|149105332844477262|150584675695955010|155315078489404244|166767768408572878|183909698800259021|185264795492662105|189833035796356317|191580634316447036|194171822973045541|197179953056326843|199380149305513760|202320698763400962|215452771631579800|233599548987075335|239108176737916544|246909017834765044|249572067671046390|252234369200068838|258451594701723729|259850563434239687|272699876578457004|275738228941486304|283485009124391469|295343450409151711|295920237011172496|298548467087802939|302357212672712777|305894612110593709|308450464572146902|314328559526116851|315917626418726442|317393824538282122|345672431355907987|351177770250818815|358288581176222318|365867429390253057|366919957265993845|367875449641359254|375844281416563938|380930059517083120|385381034539043102|386244092518874298|387803733122001257|389952379943961663|395659960955585426|401459796096628169|409989846348156819|417366902844080468|418964542690841608|430843337843696834|438859066326173508|441484535687323116|445386387418839949|445685895250363945|457262340847272985|465256568020770346|465994058794084682|466925282859533922|474142912008717485|476010934626313553|480628198394047030|490516580181789988|492751743537576394|493019678082097277|497687872582857511|507932897101632871|507933788927197088|511087353217449425|512641356992506533|524664108052658646|529096450363668690|533238604908455091|537064158900225875|538008677777638650|541340875610020211|547268757921027174|548644040722331979|548757088529631241|548803121315534632|550413806044254934|550480418276948194|551790303398313210|575933196699733166|577247390262962169|586156889249376958|586210193436886643|587345323078085273|589134709704530438|591262554729901033|592553409130228688|593489193004271898|594321705793104786|596872933910002684|598403081154232540|602629373463340129|607305267358025385|608420815304738523|609325320743091585|613716478803091246|628912115463029403|632969883648488452|635779557714070338|645463977505087970|647780768480444423|648153624972239255|664312155779648781|673005055670732504|681302342700686339|681434360288199670|688174321637497873|688450386062323849|689441434638397342|691090288588152906|702911525079471119|705402620138606806|724879565061123912|729062876731615096|732748783493014733|740658952307933351|744093222480300852|744401799618715916|748025866173257429|752741678739753720|754275460112368015|756845201043546944|760254612639769451|762245412636628376|763456863780351582|764212277800875039|772324932999610202|783996548534194085|786070876613616395|791087070862410921|794508616080226452|796554158432554744|797352561971869983|800038456554015986|824555781203225211|827086726819218777|831976685841040580|835661315246773004|836470534800279277|842399904770120443|846377720977856759|857274708647691701|859275380421474931|861475274332649530|877618590944097304|889351768596069165|889829498328886911|893836511033851251|902139188415177936|904323583940839923|905974603775779132|906485830031512422|909542406996424979|928863632228479293|932416416456986439|934413286393169496|935316141516518691|937129864520265590|944387741063861580|946466870949064750|953725403889524129|955262793273736746|957625936587417475|964270511763724967|971839921652238979|972568829847659779|975649589957882907|977409077713306899|989301380339093395|990403175585484985|993663189416669436|1003948920988905987|1010564209742871119|1015554456128092479|1015935850694643445|1016569545476859224|1018135237949366105|1026443018251470178|1026709111870321301|1031859689658104547|1032417190634853238|1032971003171242104|1034775432336119992|1037619393132899306|1045381679538460090|1048477362092208216|1048624328365060735|1063153601770292188|1075244409467143793|1077908919238578677|1100876544853156264|1107488919115868516|1108392550188547140|1116159091908487697|1117997698639127974|1118744291562853614|1126142484100031424|1133597934396877411|1135062923302644064|1146719839142622551|1151416906385284503|1153655580274992015|1156079824044497964|1156503537449409593|1158777608372808668|1160732500952248957|1170015517401660312|1178987400530877956|1181644536786879753|1182399965847493114|1189397819509565710|1189824977556052449|1189847073211224697|1192140642758335702|1193309041866303243|1204736144812153469|1209699020252869380|1213429432915706734|1215650825818137910|1218089153521133036|1218955957467849816|1225249035406961350|1233638356551675934|1235368335981868023|1244292644102115376|1249412593275587332|1250752038958897510|1262400311970516798|1267437703480463813|1276336235859457668|1278675305981640472|1280950523484047624|1291189228635461975|1293456595765979885|1294125415507797616|1303624088204507125|1304119456319382630|1307525486276264273|1319486256408903397|1334933285350416964|1335368358870947698|1346259553084795024|1349590587630565684|1350796616793536463|1354896658082155747|1356092128310103173|1358711027045303066|1362981270616758271|1365886112415004795|1367551870585577945|1370307029700577883|1373719134292687745|1375426105464047396|1377062452431019815|1383017127105969368|1406757122057726348|1409102039743605876|1409329775286802288|1409825895920546724|1411570617729139715|1413845622640046948|1414797806369924775|1415427615580272912|1417555895995676841|1419632538424211619|1430639213511473614|1432678121379010272|1436651329486701242|1449446758868203865|1454296237776820844|1456109573124531169|1458645805793295012|1465163337139396260|1473414903860014888|1476339845163904882|1479067438865131218|1489708222385060052|1493053158595468932|1499468513289311896|1500108427745850097|1501367593904930810|1503550316221473325|1505151123242802667|1512864343749540042|1517380636369919313|1519098993159165769|1522021317591378249|1532640261621808102|1534319389660017365|1542003019410810562|1543853910519262436|1547615024824291362|1548458547380401800|1551348587731891737|1553586689717417458|1568302643848010238|1574933076144541530|1578466505563277620|1592240385985023547|1595263276984806504|1600525815616405238|1601129448843702498|1601617015645560582|1604372393927896015|1605188773904756187|1611974447653925695|1614474138215598975|1616177719481260220|1616811887045996992|1618801273322482570|1623021785443978034|1629976574434638473|1630546589722673776|1633376763762401725|1638103526754540575|1639201580476279449|1640922458831392091|1642220440357009595|1642953124233207109|1648506842391480955|1650144106744686118|1657304709848517023|1659273874347028424|1659970228717880113|1662768809631314423|1671753489931602335|1672554858630623252|1673709211225680797|1678858634611892628|1683587631463623326|1689754686230895463|1689871606324865786|1692700079436796602|1701668303614071895|1702769540331534965|1708329782358196461|1721567218767098675|1722857646052579237|1725094303710144975|1727224825537927057|1735874461438697548|1737602994923776275|1749449664194404810|1757576204406287116|1759239165347490459|1763536817728700404|1763630150707961740|1773182670683107826|1774531857138892390|1775132791176423962|1781640474815752592|1782356198213630758|1800838255042525153|1804635495971306083|1808306753693574433|1809930503852954547|1820003065400360294|1829449011676556479|1831594209646212365|1833051400484929547|1833750345119274767|1844287756924805970|1844366663500418249|1848149834112470684|1850923446989005330|1854395823384527969|1868320845862990917|1875765710267099669|1882597966350255090|1884637555844513804|1894789522889017819|1895446498805265772|1899506837648032995|1903481081205642053|1903921918188017240|1904692066799270589|1914262490880118643|1916619092437516875|1917547317089926816|1923140374637382480|1924201746631280893|1926511600657491954|1926660718349007944|1933123927426885458|1948005193480447449|1949303459705399844|1953741211255595589|1961692750956337750|1962747871556473880|1966063698621580524|1973853923975960340|1983307883372466256|1983370118414261279|1985072831820682491|1986670667262089001|1990821803888150116|2002195537064250521|2008041509861515668|2010452049158891615|2029633141236870078|2033331381796529704|2035016915112908158|2037431870591706352|2046642838223153980|2047277509970344038|2049870678294918005|2054545837768134725|2055151501373980576|2056263284271039374|2060008793822863644|2060047392149723862|2081159466628841234|2081684536650050391|2086333055204341748|2086875928567410544|2095982928840188189|2096213158033650149|2096973346350643507|2099474792984098529|2102780030900820078|2106338876120715248|2108737458356280528|2113413947016443975|2125038011301789445|2126123175046365189|2127372231925647680|2142577123405314319|2144772846429462140|2145678375115807502|2149274362459474197|2149597896280534229|2149995147593360696|2151623893605278500|2155912517027471041|2158752565093645763|2161571259421669865|2163170900500695861|2165849412289455546|2175080775368416467|2190524648079249138|2193860294221593119|2203664248504199182|2211938810840426492|2214978261395951201|2218228578254884355|2233585391425727110|2244086298372398655|2245382694629502995|2253247499129962997|2264585557133117643|2269719658902288266|2295198824691237846|2295381081716191993|2295831166636150513|2304737509844097729|2305807766688776164|2306659588197817981|2320318025450256629|2322437612842323994|2327152927842824155|2346095035996830835|2353815318792868398|2354242769496966372|2360126818722444653|2361086361773035298|2361514089374541938|2362740701407750754|2363950859866051166|2364213485790398828|2366732114153817601|2373566356276266565|2388783824845271417|2389450290526541995|2389923429908324520|2392047706711885419|2393022351239700931|2397546039274475894|2403873389707008078|2404810394320599195|2414351439981423303|2427755983648038880|2428597642003653026|2438243558048208801|2439116950152769546|2443079295525700128|2448578679635677155|2466371691562526581|2468124810427521803|2473144647860356997|2474884608455623757|2478384021179869466|2480371557222621022|2480827231141504295|2482599342614589839|2490567743385329027|2494793910430836002|2494956457224850764|2497011821635743479|2506305769643279158|2516923794279219644|2522309841891901585|2525142672783259774|2530054148388300581|2531745787100392971|2533195846813940197|2555066788308648011|2557960972641157318|2558262466997387862|2562389290214454827|2563099689900590670|2565916594383645876|2575703207969851524|2577739725951873680|2582712514445097718|2591425981504543223|2595856475971663311|2596356595376202745|2605355039786320844|2608155885408642784|2618785049916117307|2620498571006232324|2632339139186459452|2636387680822708255|2638612097829965638|2643258142040416747|2643453591577392192|2649862984896973367|2655142749365621444|2658498325382053214|2662354975572707351|2667381081733756423|2667925297010607241|2675415317638857261|2688434705692281974|2689629328857165576|2695016151923221430|2697946190272561207|2700951059180732264|2705477146678741175|2709718148206981991|2722383512088198070|2728000341714394313|2730279654927732516|2733231804483951316|2734926835689427567|2740767047632074192|2748230301981828567|2752387886358705095|2755897317060497207|2757138834794963004|2759574999022989662|2775978995170276962|2778684253974577811|2779742586423104108|2784124906672390235|2784927879597233636|2791018003356123328|2793038982927652493|2793877229334049678|2807717232258884885|2810200243641028728|2819893986312186836|2824409072163578397|2825312427970975785|2832700080546928668|2833912358365812910|2841891127303911767|2843161539269783469|2843834369572590636|2847477086209206628|2848530961384323193|2867224052186052900|2867808980992721096|2871854087559127248|2872637424205574689|2874051782127727403|2884452196347304052|2886345653147846505|2902286709660036982|2905059829840025218|2906039981595777811|2912975316042196607|2917439603692104996|2918807261694046828|2931965767916242284|2937378892054377559|2940895344234151681|2952012468008661743|2957926261669116707|2960873815340065345|2963023593952155161|2963270807423842803|2984037558639585913|2986345870523949269|2992693773065424191|3001276958500690676|3001547490758231358|3001863097760374072|3009855578441818281|3013194006060469746|3019923028763831515|3021272232688997816|3024213564643376315|3024428779631500464|3028893545148263813|3033523744219563474|3034494287124948475|3042125848423903510|3049064433781498532|3052658623920413922|3058232176497537360|3059710805123557602|3067232573817139861|3069654674733290329|3071173378262395709|3077728678967542653|3080431267274159104|3081820194176992101|3092593420626605645|3095194714435486840|3098351176009038545|3106697669704234399|3125116358420013601|3134381747317042307|3143036056335644832|3157119038081495428|3159848395679153998|3160549143984246455|3160878831964999952|3171910573349384978|3173298173222767967|3174465880044347428|3182435804201605674|3184334551038132041|3184454028992958931|3185497168565527849|3191514805239747615|3198228125033825985|3200306716849908400|3202035044903988889|3203008517208686526|3204720673652719003|3225417771663481396|3244183826734163396|3245796040374200973|3248978483292897922|3251633988282302693|3252639117004504346|3254024184558035222|3255894029066892669|3256133761038378821|3261085223935648113|3269277466917444520|3272515818739130528|3274989458195836492|3283158198824812672|3283662492806153079|3292181098789090416|3297439947374100573|3300827056858622979|3302282913511139161|3302789032980081588|3307957162320415612|3320581907248294330|3323071310563242585|3329002682378173432|3337292194888390264|3340739897282347353|3344518487318824635|3353444940169534931|3369461411826636722|3373888085997347830|3375971346285383601|3379403918392394285|3380368076272920749|3380778108054620000|3390094014912462612|3390508366812653658|3396503793801001529|3399148539924094718|3403006771712068983|3410244149131760311|3411826526540793206|3416538638666303124|3420028639497048145|3420747679175495193|3425544862116177740|3429776279885215777|3432492375719057461|3434286076318749587|3434324356552235307|3439660933292033909|3440800257849020965|3455894576291513754|3456180316951164059|3459162982531848181|3461494427763747944|3462367420440338375|3463032224496578132|3463654131302517675|3467717351674182517|3470369055234033527|3475365993814876012|3480968250485176084|3489507003812105967|3489818394865033320|3489821261930273863|3495972581458011928|3496803711424507615|3503781285445422673|3508892313392656933|3509229703728825229|3509802576052755489|3520230087826727833|3520388474346944427|3537517950924483930|3539372961032684495|3549286707899036020|3553098651045674329|3554551528325719984|3554686206164791954|3557958152608583487|3558016241757299354|3559391918815892287|3559528351764936669|3560358712445080929|3563349637278500138|3566077523056861850|3574052818647834954|3574766816666937211|3574860469440465439|3599583291125658152|3608520459048466702|3609644611709505753|3617906903502768064|3618777616842792949|3622348153824678489|3631576752168539405|3638143347573647486|3641942806650080969|3659328527642984204|3662206145065451694|3669939151827300495|3673568895658075182|3673767760835614655|3679225498097474744|3685123345042813699|3690279859640902157|3704354323851158694|3711218611905125323|3711630966892169280|3728998941528275592|3739191931152387760|3741937608032437889|3743055908108053839|3759430787189532358|3764344354001540261|3774520996410939525|3778472624835154921|3784945802421074977|3794331713349502693|3803602485262173182|3805891168852734845|3808116261771927314|3811038714126508648|3820614363250863078|3823927528848640197|3838657959916677451|3840206973384358139|3853398235500512833|3881096489949568598|3888532547080587187|3905936074744326026|3906200685460806543|3907998884767546514|3912660530970288906|3923434948547435780|3924299070097390011|3931358126031530325|3934643523093980365|3937181496129679688|3941345195311117139|3941466221140099767|3951892297840141842|3952127983788246207|3952185047972105017|3968995933836813324|3970483972005289595|3975871881212998986|3977147258083519776|3983284102422568023|3984522186143704798|3995339217169580604|3995392786349003848|4011776700610257531|4012044823370673957|4012442566651366301|4014493038838395919|4015064899623804331|4018732030904241843|4021108510932186985|4026367376110127833|4034300088310982790|4037134090695329733|4038409269828300304|4044645379378325192|4052001859156141668|4054694926783967922|4056174395398851900|4058180776970329135|4059592026356457503|4072604202124224422|4072990468456801361|4075410046483903530|4082290463359045077|4098640391338452268|4104806133536331873|4108924151311068693|4112672064329359926|4120871090823434322|4127496891904505060|4140994943129171709|4142535900170119214|4148955737548569429|4154382842214985180|4157913567381258791|4159432956581893383|4163269982005121081|4175853273657313146|4183868309206869924|4184628844986843800|4193597314914178874|4195573600200879293|4197285934356744338|4197819968567449026|4199107481887601148|4206987851521590993|4216660245435073472|4219629952864477663|4220590039464252518|4222707366536263745|4227600976964687723|4228106698646783936|4229568688327401851|4231071348316821643|4235413797016217376|4238159454873377921|4245253408159788597|4246543004624135098|4250591475555058955|4254272913733692412|4281670941117070315|4282264724930202723|4283017939155489514|4293183201963326300|4293852779792943223|4296184522045043116|4299788163683383478|4299909562970624794|4306446891395881447|4319366866451031405|4321319288945685043|4321512082895880471|4324598025897762853|4325434433407238059|4337181358226587212|4348357299215220511|4348979992265103665|4350170266710480191|4350809138898316057|4351775788568301237|4352377346915540336|4353030789783018716|4353529905915543178|4355986029291505646|4363760724426190871|4366016193675579339|4369385116759077642|4371493452855772198|4376537688844335527|4381191371594661012|4386844055596997482|4387455291899127642|4389427355246589146|4389856914184513550|4397548642899779859|4402187465346737740|4408651760735251793|4408867649212194569|4414611345678559174|4418643242434693134|4421984075002019422|4427652982569088882|4431489420982637976|4431915457182645948|4434502512410396232|4435065485730234823|4439933109056307129|4443032338661062161|4445326088033573281|4449876933395514699|4454841860739034947|4463664838098509526|4463862290348638840|4467191689692506294|4477055398147542710|4485732206843624691|4486071203834471944|4487842406653658873|4493378343145542247|4504216067943143508|4504295051450477279|4507467461770903401|4514173723503703424|4514930364965081801|4525741821565439279|4527915792442078651|4539957086139780409|4543013887976062074|4550542783877405105|4551703281759342594|4556104590302298656|4557427325124576968|4573251237730794197|4575060091961266667|4583174606283295648|4598224170733277643|4600920464274326339|4617438993514596617|4620129452780159194|4630998213412618118|4639749796637531569|4644321717101240928|4644921317275114487|4659734952783217676|4660389754470839922|4662673078761801111|4675547750955503146|4679837129536631517|4689226146540771366|4689229982094282303|4690403797490510537|4701366813499476862|4701671512746148302|4714635458572553812|4717028368788807419|4717483843982933173|4718849630511489598|4729123338413298847|4731537769182829422|4743832780126708924|4745679180379879355|4750049040531924506|4755992205200896737|4760178867268493251|4761595333979488844|4763012544107224994|4767967870776205252|4769678538415333229|4774922277500520201|4777073756267565414|4781602201768116415|4781901904033369860|4793916920076692947|4805543592267775880|4808234091369840277|4809001889294722846|4811082316306423939|4812903093841330797|4815021311670818990|4821506018803905782|4826553460072754103|4831343207115208931|4832700386324178586|4834128943123379009|4837961672293471253|4845822693583575754|4849237936747232697|4851629683686426249|4857230454301078616|4860977946260176666|4865554948577740027|4873255477797280864|4879743119899185914|4883555406448814901|4886552603672021359|4894854908959984226|4898575666399697035|4903477421708655678|4907559096669603550|4909947321921093431|4910776138005720986|4916837285238885256|4916931664078674676|4924449280824364930|4926295695812304109|4926642289678042433|4926855826369036914|4933643422047421533|4937340448616119334|4940926476029608713|4945290992396400937|4947257184529521860|4957773611697947604|4962331305699119953|4963285381472563400|4963680843186082473|4977126143116859084|4980128858928091959|4980840965774107027|4980961092679232167|4982003679575281062|4990238056167648438|4992665548419449201|4993090594392915194|5000577550748592645|5002399014698867142|5003793608495305467|5016288593038357765|5029638462337405616|5041393976407225547|5045538840433334273|5055576772511901349|5055897121292330910|5064801846530979849|5065013814301743079|5067647126955470787|5071361154015502993|5083722612149545454|5097913628069251073|5109434403130083175|5121568875271962212|5125310625670347014|5127157685940368720|5128020609025062586|5129659343333955816|5131275031348032505|5134556720318396237|5138186861373234674|5154411228035561956|5166421947530334008|5167418221880061636|5173498228448475688|5182176243384195293|5185724865275792228|5187147731091012855|5190280597173853371|5191192331134327088|5203408567537069391|5221297800323146117|5222783211479861429|5224020730497220518|5224960903157989852|5224994562460397326|5237612018180226891|5238477706456005958|5247556675994235508|5252831311376952371|5268611836877740855|5272905164199155145|5287903834790777090|5292800395722960316|5293367090417407765|5293696651857090147|5298707695819969250|5299907243906760603|5302080348157466972|5313596598538704004|5335469023431368822|5337460965685050833|5344656569146860827|5344948936568487347|5349649777614223114|5354696518022586143|5357947518877142563|5360805688886221139|5364388258426403989|5372387243276528819|5377519336311556795|5378344427357342558|5379813899527081597|5381426514494379971|5387842800310263669|5389955066582442929|5392886504389177941|5394945495608225914|5398543164896531998|5406528205722901353|5407399447428740555|5408159106011845634|5414789089019024881|5415620063608024214|5415750012910007275|5416297530539574986|5417418546111900537|5418087132472107600|5418374627551213460|5433786223103980486|5437247771236890699|5441497546498685816|5444016725099269548|5445980309172680649|5457359910717558746|5459824049596271508|5460296402711072519|5468319277184464568|5468427251794703980|5469837982534609037|5471534552081942566|5472287489170119380|5477258255908249572|5482128942362369272|5483812610715176170|5486765377074037634|5487122008234530316|5490134300792226800|5493451408616107946|5497473826186382029|5507343667701712646|5519898457427123959|5523404471425325698|5531029499762910203|5541552843745538006|5542862215049074327|5546969018530233349|5548544853901385594|5551287528219887729|5556505310502069390|5557600070811242456|5562490586236697755|5563281878151071093|5566973836637419858|5573448393943000276|5577569381405834367|5582564389875567537|5584761654632610884|5597719257377207152|5599024917032619780|5602444783721166347|5604192544205968618|5607398913059342204|5609435929072717361|5610055440914452168|5611392766277153203|5614025894205244891|5616743409128072812|5619798685497394989|5621015019499352636|5633202419088777433|5641614257358017466|5649004160370041306|5650417102604451087|5659519509529511131|5663648355822535169|5664565739542882103|5667980134220363693|5668928285040381377|5676639591350098546|5676840034302774532|5678563926593473355|5682244411619832535|5686190834956262854|5687624730205330843|5693537378019438970|5694353354235473911|5705629516372011450|5710538190477397665|5725029508601244525|5726300010002125179|5732177138198761896|5733067649791863337|5733573908721220456|5744385197957353859|5749119337571802624|5768598349530546119|5769599975061036713|5776800011958472495|5780509106636762018|5780862310345576333|5784345262791129468|5793292355019243990|5794587902620129772|5796774382155752430|5803969294789096855|5812279112792795337|5814907704875158504|5816034890450347516|5816630432364871749|5817887821142766123|5819039012037429443|5824680124642483391|5832218837170417192|5834585044056326269|5838040330153146281|5840363657063200246|5841194307479399784|5848124772517107968|5852379117506523080|5853635968699138889|5854409948674702272|5855488272497502452|5856337820855260731|5871070112577749245|5872276961625206870|5882695494863461719|5895647355243195068|5897614645473867621|5906391302397393790|5910717864255562434|5911049957502362892|5924106647802217771|5928197198259711891|5937610216541814475|5939660735061296759|5941171962748111525|5946775144414406790|5955716064372667812|5958337347488140261|5961484035703601184|5963154696805769405|5968792638987035505|5973371104146919796|5974103698373908689|5995073813216706272|6004200822853667897|6004441666452620906|6005535519817300773|6009991548865070705|6027481981553151987|6032404554012384255|6045086227030006437|6046509399973044300|6050739914349140077|6055600165863036587|6063320169344610641|6069475826700022832|6071510515019746729|6073191334637059767|6074657735022747118|6084267339273646238|6085458715529997944|6091807365899075536|6094778462803537553|6096008377933337351|6100304231699768306|6102941770155122146|6106076610316145790|6118559462115710058|6119284347150408736|6122996509223071780|6133440989325448467|6144751361972823795|6149307115505747379|6154596974833111052|6157658647001327798|6159488445096809249|6166825190363916095|6167069809757838479|6187652214884755104|6187859293966067717|6190011466112486857|6191403180303593823|6193221293509007403|6196883537846097590|6201035319775952763|6201599424237112764|6202905813962370759|6218855226393248227|6225131449350592701|6234276271055374006|6253397706812400066|6254170892432198428|6259323069698078500|6260078245338883215|6262784989736468183|6262993221114269256|6266035838226707399|6267856996063051337|6276165368858507131|6277658929013630369|6284099128881499364|6287570425926873467|6287689211827716777|6289931615324396286|6291915797941335266|6292839569038852137|6297195051599914393|6298633108777248701|6319061768670402707|6322026349658145437|6322285009656790112|6322687745934503686|6325403300141499255|6326029331066177377|6326959099516356598|6333878370574674958|6334626756158433334|6341140769406451706|6348665024099338756|6348776935949000220|6352304477256183270|6363439627623339586|6370920587410740564|6374509405177515056|6377703626441936496|6381378924215974072|6385865001439324596|6406641211335001706|6420814026058195560|6429530322507929882|6433621571751635643|6438337168762368034|6439118317857311255|6441216378547266542|6443728463956335390|6450220667825173356|6450652188547508882|6450912733351192426|6458445119853642744|6459526665829768154|6462949218275979687|6467639346265235186|6469496521501287676|6472287712654288232|6476542227608815917|6476747126336754702|6477703022831829029|6486421933055471118|6493586241476144729|6495866860732251149|6498078197156122111|6501611084072693984|6502478457314992655|6509964394070585010|6513348775047919375|6519242670461195775|6538587013535073714|6544810418382777365|6544944309935075127|6550800432030197468|6552132832323504669|6556219649081943322|6556232446528517845|6561794137716323014|6562850229369937039|6570928032608929666|6575603173999281759|6591484142096753783|6591718654328770432|6599526974063259913|6614546878327048872|6615203942937041267|6615433807033846342|6636432068232945527|6652499558907579509|6658275417495600301|6664731875915360052|6667121552625648916|6674842210888888370|6677392646082118593|6677788463790307997|6680040646309276955|6702701530574716005|6704927075817412309|6712293551382348259|6716052973080756867|6725743229378097525|6737933295778578519|6742722999341729010|6744866666169417518|6746604291808669922|6754084823451559302|6757537766482998640|6766472624543083567|6770145479267971874|6771038792897870411|6780339571053798264|6782309279225528120|6782462608575919079|6792921691235342138|6794875295577062192|6799650188554984410|6800438783364354240|6808741226756280140|6811420480921523497|6835474884878417654|6854364421649647704|6858432352256830570|6867122808833897416|6868247377639421027|6873387700294144236|6875595825546032713|6877345639457250282|6891481589721396878|6891831935515824424|6895678531284775574|6898934754607525708|6902769020027380982|6908177964903091132|6908939174870881058|6916086964794712102|6916828752204613125|6916919532180024948|6919064417556495152|6933065266645854823|6943249092018012001|6945465651164141663|6952773833795135640|6954377091165413365|6957133592295715460|6966117765143109833|6982243630366566813|6994774477791290724|7002187597783353423|7002312700664095496|7005053982430206896|7023563284586878323|7025355288730787783|7027979360657416083|7034275815611243548|7037548794142504665|7043227301514079724|7048421508781366392|7060670781880858356|7061321950704500637|7072477291020706174|7078361330605211457|7080850748737367133|7091867197285247959|7092128252178185096|7093489988734510568|7103585234918393150|7107863942583934078|7113910168229518088|7115670870961876877|7117052183052782647|7118428564014251239|7123034230101384433|7124085500064522244|7127734764678202995|7137228212069573292|7150237725971351834|7151404066139938643|7152867641240471648|7152897216453299742|7153513200481039295|7153946746447650059|7165521721378006131|7170372936009157316|7171139095417697199|7172503490980019608|7175244697360865113|7184486752382143674|7195180296689410538|7196039936002118364|7200956438087543167|7211422289939603368|7219499034088572478|7235824984627172031|7237567977282845859|7242126097655035122|7259895000157589525|7261499851522738016|7265461245248533151|7277020936485935243|7279194388097939860|7279716268695068888|7279808762255898217|7290587630921208946|7303698522878422834|7317665002862157587|7317844845955167941|7322823005816170907|7323657979477305002|7324960414535540935|7328162769465479183|7335361381469194553|7335981909764031719|7345399908932400129|7347214846049563646|7348275286438209675|7348530041422310088|7349238728522835661|7349287651739124365|7354777161350645910|7359874747333822663|7368339210377754716|7380846170499217851|7384951673460249766|7385559000830485030|7394843360063919380|7396519184734503116|7411221685841426659|7411603020380228027|7412247484251617660|7413721996324021866|7421593704804532521|7424265435790900654|7431542255952468482|7432379515900986334|7439311855191134124|7444518248726319854|7446756192051414128|7447727788035761039|7462015636436944522|7465771486077373319|7476236816124629726|7477274755974300585|7490219304434157105|7497005647853607784|7498591410538949497|7501162314519098707|7521561118773344549|7529303144963094356|7529609734435352123|7534359669237152653|7535115060019323709|7535850192267750777|7542841542965090533|7553709866366910753|7559656853084233420|7565031560048891840|7567369980324022117|7575240533541970957|7586283197450408357|7589992968183738878|7591885700795337287|7605376411243662767|7607073862444181384|7608209571732905899|7609409925000932142|7615708460422674063|7618507529497498933|7624777277171035904|7626683374343771241|7628468696011798809|7630295204531363682|7634413366179214328|7636497665892886065|7642103025154419805|7643659965562845892|7645544811651461513|7649880720836455546|7654339354391167325|7659229186680307241|7664696330994137507|7667708091037510729|7672069523792918281|7672932510390771754|7680208332745093715|7680597035796687496|7684673255914731812|7699383134439196363|7700642400737665227|7707798211226508580|7709720155401257896|7711850014690796327|7718754655273785620|7719674066438221647|7720043534553267382|7722660230540058570|7723621294253876279|7724960477808229584|7728732485545580830|7748826304607783819|7752590779816181045|7757714047675829949|7760348304066077409|7763410573786031565|7769759454101429450|7770554067873708229|7778702934315099131|7781578768488225902|7785273586446553703|7795875978641549161|7801268296698626617|7804021376467398957|7804766465441449900|7814008775470702553|7815961424343723104|7829825908671394106|7838011005099002983|7846769707704475808|7849852159758870497|7850835181058478203|7852307566517457384|7867346174089653611|7868822447952006407|7871562411633849406|7876615731937185754|7885130387109996377|7888316910623448906|7891323386651056096|7895632981436398041|7902022978864167408|7903553224320521142|7920803579322531179|7923264361902703790|7928386798370682309|7929513854428456691|7952662308059061811|7955254853200272555|7962515973930250879|7969087300055177075|7970483144793289296|7971483603727877950|7973745694580272507|7978754808204702113|7993584713044273593|8002298157727823182|8003141579741147683|8004935228890814817|8021215434523500797|8025303030061902768|8044808468476868848|8044818825107612195|8045545803248903524|8057826828383357437|8060404016850561910|8060898952746067554|8069712888855734022|8076121844657685340|8077489611373209231|8082544056844620238|8092240343796501260|8097418386160254702|8099488269015043724|8103850638713626512|8111289600043980040|8114075957216474308|8114878071256126220|8119358241409857676|8120929032782623278|8128996410901013529|8130526370586159288|8141716320532087908|8146284556047414536|8150950388236697146|8153227335946369636|8153545060257345005|8156184929557456281|8159013890065972859|8161704723665321020|8163152649220878988|8163987992050506463|8175515090326522904|8178876775434288455|8184993431829792499|8186369500638673393|8186838536639775349|8188490068373513243|8189445396297846435|8189883069116662168|8195593802427051243|8196816867545150954|8202363239937373696|8210285584902779463|8210418981266381804|8214974454505151920|8216608787954270056|8217270931362830975|8222318756648510876|8225712845889083239|8230683062511199381|8235343106696653639|8237273884567459338|8253374187706076502|8256277730398784662|8260291877063340832|8260896932811825534|8262409027666380668|8268210612408565385|8272431557759582976|8277057343048183207|8284506989492644472|8300722456710946043|8304938370342730030|8321258504841720937|8326064078270443526|8334228575723025955|8335807566320912439|8345558155204290783|8347954738097856284|8348469857284778577|8354997978662586578|8363538411356663393|8365520388681082691|8369064774809400725|8380222316129916115|8384318088960583435|8385402892429108552|8385507790830146739|8386488509135048360|8394123542861158915|8396220399679445178|8397162452616350719|8403564585524121350|8408512935580609151|8411157287263745098|8412987989657097085|8429564861692598596|8433931582757195233|8440211576474784438|8447590453195176115|8452131087037106425|8453010334782965550|8455850947222878845|8457976292115135982|8468524825705586881|8472467607935452032|8473292158398907470|8477255197965113214|8479047032913821539|8480507850606972546|8488717809256455734|8488772903933211399|8496553785736326481|8497832332151367369|8499633263511942880|8504812625883830545|8504946057027607411|8509957273726072874|8514505196725920505|8514853874514420856|8516858371450393131|8524860730010081732|8526547143001179891|8527725625486891920|8528655360312998618|8530100344045283147|8533221102537378603|8541164510730101569|8542339560579194887|8549616884876311189|8550463045365894609|8561499691937019152|8568638726697373574|8583802378335046150|8588722046507807559|8594156442546510155|8606831117442494281|8612569998378418925|8616626209625142531|8622432727368725087|8630775225010796362|8630970795421172217|8633978009760216977|8637957107368856783|8642523915236570669|8658571253084677234|8660687257083649065|8660752507942616726|8661490229867197088|8662698489571402669|8663071671053839905|8663966749022265898|8664961820560324187|8667827064879450289|8668389284388063285|8674245010268024161|8686136224344334714|8686233797279325071|8686507252415401279|8713440624918839266|8726573942870010839|8744158365734277096|8744728391682451576|8748531661249165751|8750688239256750671|8760009378168093804|8764282472449329447|8766379345326388075|8767398057650548033|8767653358788418024|8768481996794194010|8770065044936398260|8786605052210114368|8797551946656038784|8798740475536285182|8798882107021862478|8805068739489111014|8806245837772628398|8807764678048766143|8810861118092824146|8811958633774966148|8814396450831830931|8814661884821722120|8816129863008574709|8821732807022329448|8822636349109690461|8827036613175626403|8830284233770517921|8830808285324087034|8843451765839637371|8844129207698321161|8850907505472075137|8871539142344549682|8877553464155146066|8882186551434595007|8884247443319058410|8886518124080272897|8888802062739313066|8889602018639350470|8890285347710020174|8896126610287577584|8904455390323721979|8907216116451562340|8910568134323161895|8926247752269997446|8948342433508255283|8949058682849794544|8950827748584865834|8958293199814926788|8960329672528111489|8960913608160140911|8961965114355462817|8963403499699833965|8967296789866984346|8968677851216355785|8971656155345003197|8972152448693838982|8975381434902762797|8977182983567005327|8983570955441589649|8983837057304749534|8984033893347002243|8985379367116779272|8996942460885042773|8998117260837476154|9014346026039407275|9023325712016083455|9028957015147715778|9032251554561707119|9034761555997639796|9035332854328721347|9038032116989240613|9038797477039636376|9040862755500309347|9041890198225730833|9046935005737138274|9047567008726692759|9050793684933306863|9052576283060199846|9055206912876188807|9062474302971370580|9064074811885937637|9067135362633513909|9067945613386044028|9078335320958724143|9080271188985467577|9083418030816360723|9091072448137045362|9094087630145268426|9097139827007743916|9103343802825231458|9121548660176755032|9124724503646197348|9130306307614691278|9133362071061073666|9144258873473485642|9149316658309580577|9149391194014442172|9151448762464370130|9155693472854508682|9159352071276387901|9171401987906631436|9174899865811320194|9175095785076288313|9175267492619782275|9177429488594503400|9178639546010259037|9182539153750557051|9200327151224204803|9201264649328128991|9202433410635730204|9202715445372575119|9203633591692395478|9203965631422182353|9205817006560623980|9216824084338541970|9217450901867291051","|") AND (biz_flags!=2024091316) AND download_page_link_status=0 AND (risk_ad_pos_list!="202106072200013001") AND (risk_ad_pos_list!="-2") AND (((charge_type=2 AND (raw_sort_use_bid>=50)) OR (charge_type=5 AND charge_type=7 OR bid_type=2)) AND ((ad_exp_tag=6 OR crowd_tag_type=1 OR ((dmp_crowd_list=-1 AND crowd_list=-1 AND key_word_list=-1) OR multi_in(dmp_crowd_list,"3333522908856159|4718150878600077|26160726867707606|53911088436328605|65243075618122061|76652579469349746|100257447762335763|109355582336878889|116356864788899892|185787914509954634|198427439800095440|203655466248955951|219041062774877422|268903430501002758|277018178337710883|302898381895845665|313251666302118950|325855173509782181|339492949787999556|358140517351744992|391574804930835114|406090075306627263|443325460291168109|446916560089886196|453209415183085750|458337807184502663|466789795429563112|513784364527978371|527669212511735045|536552178597757642|578013561677050338|584820572453270387|591895978652857006|595491698327940949|620998302865744969|641040204120281916|650645373821933071|653102034071838842|662745882754689033|663682593741672102|664129175121231410|678384323849472690|688321686878029324|724608667021814820|750796028317478418|802191715434364348|836006249661184022|837675203631625869|851325211019984418|871481668228020071|879150631809792992|897977074915307755|917256772328712027|925336358934019739|932136818696898004|935807398469411932|943824641814774164|1007290938830126489|1009885115658494021|1011855452497513870|1025115252073401339|1037018974136859199|1047273767921822745|1057921441959380696|1064993704917260168|1074924661783043992|1075085904834389387|1081718483775517487|1088340455449935595|1105686232136214093|1113946222361478255|1132410656868306684|1147768013972481861|1151416906385284503|1152057279132306830|1159357539519688973|1184946800191570299|1190336982043423781|1233315277404942794|1249271543703994077|1264505640176793565|1301356185935334296|1303283293394716448|1308008939634332846|1312974368069885496|1339227623070282507|1342774350185748485|1359138166546724504|1380447908626712280|1393582373431966499|1403094469572049084|1415720602191572549|1425915859615927077|1450465874091086362|1455616966466443375|1486969131238184046|1502823998215572710|1533531390711352887|1541373589214095277|1550889407928119645|1560282108347320262|1587860822930680604|1636631323541272634|1648695742433934113|1655298145148078589|1658702962928202310|1661630160103756580|1671816878886616133|1684810572737980257|1688009537028505630|1692034695051687225|1702306780321733127|1709965251615865686|1741819868449092085|1747321983423538350|1750502987504919542|1751306800789486205|1777952685723720982|1782685697910495369|1786647712123167098|1788518021552912232|1789227705023098428|1796766699037451978|1812889071260362823|1834362621560960813|1845894248405183030|1900498456205828233|1930737125682048598|1987498595293065591|2013857220756973996|2023119839380815304|2026269707669297397|2029616604959386639|2070434679516963206|2087979431701527133|2093268276935998912|2105098366554422255|2115752105531007242|2125856657000330439|2178928711195527270|2182024294800461456|2241924765086232864|2263156102528915680|2266016749572510638|2277013485024543303|2293978742559578356|2303488890249353804|2304737509844097729|2310028416960600938|2316717016002360896|2320863848188780192|2330760027611991841|2361136356874578488|2364165494948289291|2369387204049430646|2384618038437585743|2400482282480130565|2407263158666995244|2422784458061260731|2430555166581924689|2430562721750545427|2446373332509695100|2448175981756411234|2465658022894490348|2492158142334039160|2504506025413156674|2509628056085648264|2513628651075576915|2516415571944455532|2554078929456858344|2569667103696481906|2584873166178273837|2606055715972582720|2628142447071961499|2631026808013842486|2671375934780846094|2673258232577936873|2676980266209283480|2685358301281452965|2689357253786697847|2709323300061812525|2710917715265626676|2765077939097288378|2765823501146090921|2793077483536871509|2793761424051117972|2796338120781823517|2798336282118262130|2814247378347461945|2819335669315922466|2820512094125287370|2838379508129052792|2852316696187513855|2868273468618503434|2878919835435175778|2915393482961955936|2919176271476917318|2922727480286306476|2925491738373242282|2928382365742681136|2948323707515269745|2950734842180041349|3004040922188064910|3079365767644047739|3086520840679518037|3139718706766475806|3153499766609748783|3162559237114733253|3174751296707753978|3176322485588416324|3178928058560855365|3179268243218709813|3185020287307809162|3187586522685884615|3193097198290742908|3199376914261686211|3206274190023104201|3244196122661414041|3261619001254973158|3283015302835204656|3298835633281545683|3313328164167581023|3324132033119735226|3347944386577242868|3418442360951685766|3437612907703212275|3443734014332092669|3450019037903541276|3452905203887350182|3460649117726763963|3467478624883198727|3480467936828271110|3480734697265197413|3489573311941649023|3496948755774977148|3504123994288262945|3518146973654770404|3520997673891517239|3527455998495023409|3560959286651030350|3571079935112461886|3601729555189078387|3609258476146101126|3613995476507468058|3617292760009235827|3638262971482959404|3646862348026359688|3648853029385698738|3659587841067122725|3660471921122767835|3662564970088306770|3732013559705746945|3732633633425075559|3732797089764857992|3735777237708549985|3753773621227429039|3759118564243242503|3761389161842229841|3779129054809922544|3789897698125141159|3790777500945068237|3832715533426131205|3838033561831625585|3876613650345697691|3913842769284172464|3929219809129427769|3958821739900379129|3980046451058588176|4011179817991396847|4013083127002466713|4026981073928993684|4035917763722958029|4058765459633694831|4062966911570957472|4064000074903655691|4073183410556889740|4090867196743364158|4099767024750591845|4116872420086975831|4125228236672804468|4133203630555790149|4139738525835858223|4158658762768044801|4179233440193662716|4188733833633056243|4196026516983053585|4197065194817228260|4206859164609527788|4220407510213682789|4301068447367333129|4309771300552209212|4315871574732528666|4340241647670870546|4370959315495422519|4372713249706031478|4380349414147605732|4396870323119699329|4418911194631916647|4419820308409441831|4424618844675315440|4450169933710802724|4486082019753162310|4487117859040182754|4505650729817145155|4507515428319700397|4508441381264708658|4519303767377571626|4520438203823877961|4540537859958625931|4553170918296415297|4563201564315348966|4582806204961490343|4632888099965744014|4653169601229889066|4667434169491719161|4735177503287166318|4744667030622675469|4755222600419821202|4762658556139371742|4789299615164310316|4794863145425929312|4811232845055650502|4856349058467211910|4866401983863604404|4877447804048676976|4883841839953605038|4919301547109539386|4922570555034664330|4922674582690718346|4924701709711753655|4930890522472646045|4992300330096236323|4996437991676876344|5036181355566868765|5067502371531107817|5078097456805788049|5091142497461410673|5114355173678165490|5116156989068562522|5142592512703335528|5156996349843224383|5168605359835898014|5192034790015307979|5196741133950898685|5198385305120561499|5209269431603941348|5223269728864895752|5235783770817161342|5246634370024749000|5268255641650656767|5273980080364416325|5314876301384989985|5321513241018361222|5359569616055110675|5360373181187690808|5381523985307223661|5411966570857863566|5428000033876997002|5446170158392104682|5464547906691259615|5470992068072760164|5502748917414281088|5540121187272718686|5555296100837357595|5561539890064731373|5563008688298112682|5626340019562959925|5656623684723523650|5666732302512946432|5678936438752675323|5698000887962308027|5701841063880116257|5761990704841072844|5788802399754500807|5816034890450347516|5819217405283071981|5894866778801922280|5910979301357874145|5912433890708234207|5914865582223918253|5917732898970743336|5934940025835148333|5949184949292275678|5950612434191422179|5967746373570117727|5987741875925489424|5993562808129371089|6005372228985895025|6006650751490534955|6018150009418673445|6030605245295433041|6055701408410101938|6055762413218544750|6127538826954913619|6132795046378426269|6171440850448102516|6174008953969152499|6180259766804939991|6181672486632962231|6216509582377068728|6225655457394121630|6227552063070101338|6227881869398388953|6255000541656267453|6265152529707270069|6269190146222625624|6277227284907756454|6285731390885743659|6290359512645204541|6292839569038852137|6325267305079748338|6333857343513797807|6356897939282257036|6373944076690380546|6388401109472871003|6396835851197605695|6397390237513023916|6416765935169123879|6417377546069511506|6430786200553732943|6444838863219193060|6459927585572504617|6493652967603206314|6504030908137471418|6505394304216396330|6536255651110691477|6537110204440322763|6551067197893158404|6578176273367860212|6602509945923740336|6604069107512359443|6605008938360481478|6607742295170942001|6619664978456740030|6634503929134791125|6646897446807323500|6691759492813528299|6703354532260384037|6703384876838010873|6737352386338522921|6787357223192120594|6795270209797737374|6798495371313719595|6800569347005684142|6804979072343331998|6833833812384208060|6835541168894131080|6840049551021694347|6841997939692158785|6842222538173303550|6877365403758500816|6878522337167673632|6891997545626666836|6904884736631446335|6911856159394953698|6941422714372371066|6964207797463982365|7039524079399891585|7074320802377616844|7124640353542841782|7149223201624721221|7216705686020523931|7266905897612298267|7326454106052765602|7364423568524019001|7403758856280174235|7510599540561426487|7657742845359682088|7692321969367827909|7704057018961184501|7856882379387174966|7870972097565009656|7884450537652143749|7892445567047991929|7909074847595563265|7911333272109193134|7914023695364369566|7935360542419341608|8034103428368928133|8168658414760534686|8208302624540242806|8260291877063340832|8268233053310768933|8355955036201461959|8384639211524764562|8478022142378609327|8484057335692109658|8677774231560641320|8705474295244212422|8715881242396764183|8790493674280757493|8850928433694682560|8978373355407839573|9096876850144342277|9098964975663866916|9121023066083530190|9125075555834744453|9146563955039203827|9182944273403196832|9203993174082142511","|") OR multi_in(crowd_list,"80208440959169149|126727916179217579|441857599918902050|622991980312462648|756546242383248048|806204695380585356|956286486692043235|986094948463067184|1015528958980853798|1153495089651018899|1189002445813326459|1359886823497760625|1390705226385319070|1506645033487869865|1631095193756118967|1695023931373027609|2449456684759890666|2830072783844700095|3017360041020387862|3262922817893297850|3358687588183023805|3536671334867586408|3556798903986397538|3621425958951569761|3691308072611513518|3813730419403189994|3844702060958406176|4091963686481662692|4354537438348046649|4456754188706088715|4608318686859986614|4745615494174571954|5114400170638063444|5190921344255446184|5337347457187223751|5378852286003043276|5791007184231547341|5833002873238925542|6043007466971721749|6209413849946061286|6297389366114400552|6345189558923803681|6421721170272640232|6716756335254352211|6745750077515288777|6764168334848623189|6887033286259509012|6889828565947080321|6986221294546637672|7242223708465403105|7653137160595604444|7762241365714297971|7775358271801602369|7852531180115200668|7860310059282692815|8039708017878496336|8180991401898235191|8277770181279897108|8292957569252983474|8318351211062628642|8435337892971780354|8596630131717633561|8707281399705271958|8888219301044562890","|") OR multi_in(key_word_list,"70921140622523212|136384988538140592|184143082490798526|251375347279794572|281374428738512042|326061457882163993|337895484754222953|388044005388973349|422077298938200330|469092643772690877|490614110476736251|504340491042526348|548940426150626302|566373997463084356|580034307737093828|589506784829957937|616640925216753511|653915655929729934|690152913252205636|692706723841523214|722425436411586928|723206330743900998|746845948137736117|752222126078285269|769773115444016286|804439358399174783|822392927278077357|828991179941849128|858377164915266383|889796985460797886|907474383829135682|918212275597629601|944547949312425701|955944705890511925|1036583895583310702|1038907428832581983|1042789972168179739|1046351912763486754|1081069444851925801|1106697011519269967|1159963923934715308|1183468731683282679|1200679132320815754|1207793623217989685|1275453963910098330|1350458927511047587|1357275696184322844|1363308459102220516|1396806276909906851|1417489306944128857|1462330420764409505|1491244293750023261|1491788469036897032|1507865948020056917|1585424888000263567|1615012914981921397|1681522196324756017|1754549248237904866|1776955029708284723|1839715113560659435|1904174833182270034|1907709143146888170|1998925039065299991|2022307828670329215|2027196033505164158|2059317280453769933|2094979672798198790|2128491521053130086|2131059015395737730|2167099334795167773|2197056741795432015|2292379120484214301|2293013839619080381|2308110701959747013|2320793020197454706|2328302494858748564|2423840766859593325|2471507637648756482|2508956589464556712|2517944315678000954|2556275686516788047|2601512061763499197|2608357614863026776|2620498091533377966|2681216726307835284|2696178747848410550|2742844547448952035|2745070386415805082|2763989062253543555|2768472716719145252|2780517473942943049|2807633691460106298|2824702113791817412|2897563933500957293|2977960903994804086|3045669713018968439|3052962952519712630|3061223384848405277|3065631628843163272|3132844383435949631|3189253102506145277|3212896658701811699|3258400947163188113|3263938269786938163|3274528603550352149|3293046621325228878|3313713264898944967|3317651021470668533|3331333900549803074|3334572970070883508|3391314203203109225|3395192740753227181|3423995984132505150|3521330882077817033|3547441034692334861|3626697873284087308|3721574776402515438|3757376334275374856|3810676249655232317|3844950250101665397|3861791390700816479|3897488253969845227|3968881520605279123|3981796199360371253|4043480455514645193|4070730178275197586|4102533763604629314|4105597400598932732|4219488780712267708|4229555992181340958|4243869198704743011|4311074224567414705|4311239284674938420|4339792073401886821|4366770915872664561|4367098238390618245|4480849302217266153|4481483416171330042|4499230683118253280|4613370642408189894|4629605997419753449|4646223564703706327|4678713620308968200|4679037784689660816|4681073049542170662|4713870933970569191|4726814676205360839|4737060303269736009|4777091105912693962|4803207236512759948|4928010330676060525|4932226047615685303|4933746328324347021|5024074225573041729|5039980347146076117|5042548150689562617|5117125185281598414|5178564452853646280|5181375057325523760|5201161625647882507|5214049681796504143|5249301984820315792|5288439718886225170|5300741760231754791|5302887173817376445|5345565602500335573|5378680439996900138|5383048550177303047|5420268007002875441|5442563000752825510|5444644410899873754|5464847665629596964|5470614848004855012|5499852928686379460|5500420668641734411|5527229148603045182|5547792960341656680|5561846519749055920|5564280915297305988|5586635600158756936|5677640573817119605|5699711072318378101|5755160282060448687|5778143111154275434|5809915131992100538|5812862138454206001|5849881572151691861|5864888729401877983|5918835238499220717|5969789630214154173|5971893694728522519|5984506969969361137|5996981933003393054|6017504745676398500|6030076463537117157|6073235832493842164|6082292083910296654|6121492566999879743|6197900137805551850|6262474345212810521|6327882296427516915|6404183025067025304|6416301386321636541|6457638592172679989|6472036611211048924|6480767811656200339|6546845904998393277|6571408675424605873|6587663624709154672|6686446441090484638|6691123367600903858|6738272387861420355|6785534606252369879|6800874659017179661|6855074331643834264|6864556157650470910|6890571671673361348|6920401950254941185|6947551711822494936|6952159892975752309|6966915393958399029|7070716925661700845|7071934731710646551|7105173923679094008|7129059602791883302|7219227354834563529|7232438694272419127|7307841830029007914|7316236751875305287|7342426491553658644|7350798533017733221|7360886442298245140|7367735060545943498|7380284547253407711|7388161714077758830|7407606000767214121|7422688296185265188|7497784435145663433|7553915847503191623|7559506639002631617|7671832981906098849|7733901415388141213|7745680787427752895|7753235693231808835|7758787682582242337|7776016390118652147|7778296386534924014|7805969065251784102|7812845159640975206|7816511824571712771|7820899484441711284|7825445052227195925|7832343616265394191|7852579384154337704|7858055210911592968|7875474084373016071|7953215667697979874|7970785345450898729|7981328112248859714|7988560505742599509|8009249437967493859|8043582833493457872|8107833138991521966|8151124216390902163|8217628752261404594|8320483608787417954|8324705151844631124|8375942310792287260|8384900118323450115|8385820984843889151|8418647031493182407|8442544468831313307|8499572069225694360|8547629238098822553|8594304443769791644|8594556265610563084|8636411164522413426|8659884145858377766|8661907488737101385|8691628972267343648|8732497180270615495|8783956603924944913|8822776851984079787|8855188398968475454|8882470940071159683|8898169977791237118|8932473720192130254|8946083614718239930|8958453408760589130|9002357508031696580|9003793281593179598|9019021402698807676|9052932717483094676|9083215064564194916|9107569824073182188|9111673566299158513|9175598358367427055|9178022243021643451|9184965579833663300|9206043479924474618","|"))))))";
        auto expr_ptr = AstParse(filter_condition_str);
        REQUIRE(expr_ptr != nullptr);
    }

    {
        auto filter_condition_str =
            R"((charge_type=5) AND (sell_mode=1) AND (balance=0) AND (plan_type!=3) AND (start_date<=1741691065051 AND end_date>=1741691065051) AND ((principal_status=0 OR sell_mode=0)) AND (function(time_filter, "8|1|121|-220000000000|abc|220000000000|attributeA|placeholder", "uint16|int16|int32|int64|string|uint64|attribute|inner_id")) AND multi_notin(rta_uniq_id,"1961100458546670265|8669342430913282238|4295511754643557149|3820165547219126739|1615416169623352709|6314578306465646238|8519232450117756233|2756461647811536678|7246372942013239063|6016028506275409733|3202937297212420206|7510237410486928689|2693897507874692550|7942461091316593098|8468109175693762333|2246205382807091008|4762936000914802423|6797061107232090018|8530742052452506732|3327382205450771314|6522909056849751446|2097254881891107942|9081283293213749162|7659202351373227815|2426484522154290531|9028175612421629988|3979294040747128108|7616191334060074064|2216336493100479083|6983229653395630143|7061041066005578122|3196243134178854499|1239316581789112809|4146619685878866397|2993398114247915385|5239079003920071893|7306779578940296486|8664775826449197998|7292236495175168278|3792060204756314668|943879882149209231|67786373045767393|5460974927050884095|3985833378458068290|7260385150729580290|1471584193443402819|1093004338684365583|3888227820969699574|2263550131772636927|4925893694984496472|4053461338322378105|2309747396698563821|6079754561320324176|160623819949859655|2024781313054920143|8988650033226888193|6672705482565606954|3463594972398901639|4011710098241765573|2628278965639395857|7430455255531202418|9127158149800262638|6615696035589714492|6463699161937835373|4524875330387178960|8052583455002209669|7438806969338678071|765272482175117030|3216643116160913287|8937191440708183065|3599927177086139360|4603694906434758431|4839962215257436741|7951576000074314225|5404201438195616115|1933117509111472906|4420843354467922436|6949401081268099700|8286688749939713010|7471024169212191085|5756165102291555444|7855894148486084345|5425421488718376732|4212694001066848701|9207657085406962396|3201372394618115663|9089499071344019177|4485876234743098538|4261325010202841032|8937647154994060117|5363833089632775568|8563220203728265761|8757703765922223628|7720373886526897000|6940990643005730592|1422105493879373245|2371341722325808920|7660148443976918122|6952169319874687609|7406880929185273462|2114067842011407513|189373988687663231|8153479507569108764|3355979313800702697|9171725698747800278|8777056532130596413|3339864690631806628|3727437985819132169|4413544153042575492|7498777981318184046|7355443397257156893|5999546154177100186|3412685935113877458|2571149021240417685|8457673131098490986|5749328167325148894|5341841126750918352|257263441440370798","|") AND (((multi_notin(trade_id_v2,"75|75001|78002|78005|78010|78011|77002001|77004001|77005001|77005002|77005003|77005004|77005005|77005006|77005007|77005008|77005009|77005010|77005011|77005012|77005013|77005014|77005015|77005016|77005017|77010001|77010002|77010003|77010004|77010005|77010006|77010007|77010008|77010009|77010010|77011001|77011002|77011003","|")))) AND (plan_category=1) AND (ad_exp_tag=0  OR ad_exp_tag=5  OR ad_exp_tag=6) AND (industry_exact_crowd_tags=62259300007756367  OR industry_exact_crowd_tags=827821124185282642  OR industry_exact_crowd_tags=1736885184507453537  OR industry_exact_crowd_tags=1786381960166275773  OR industry_exact_crowd_tags=2447888834889859308  OR industry_exact_crowd_tags=4089293343970146178  OR industry_exact_crowd_tags=4997685626248627877  OR industry_exact_crowd_tags=5082906975180188694  OR industry_exact_crowd_tags=6233776986425357665  OR industry_exact_crowd_tags=6280213215139896165  OR industry_exact_crowd_tags=6611248751053673021  OR industry_exact_crowd_tags=6990041002335227794  OR industry_exact_crowd_tags=7824369360999110813  OR industry_exact_crowd_tags=-1) AND (ad_pos_list="202106072200013001"  OR ad_pos_list="-1") AND (age_tag_list=-1  OR age_tag_list=2776770492532234572) AND (adpos_template_id_list=582602252  OR adpos_template_id_list=582702253  OR adpos_template_id_list=586402254) AND (device_tag_list=-1) AND (gender_tag_list=-1  OR gender_tag_list=5506498716792272639) AND (media_scene_list=-1  OR media_scene_list=2) AND multi_in(tp_id_list,"TPID_1|TPID_12|TPID_17|TPID_20|TPID_21|TPID_22|TPID_24|TPID_45|TPID_46|TPID_48|TPID_50|TPID_51|TPID_53|TPID_59|TPID_61|TPID_62|TPID_63|TPID_65|TPID_67|TPID_75|TPID_76|TPID_8|TPID_89|TPID_91|TPID_92|TPID_95|TPID_96","|") AND (media_trade_list=-1  OR media_trade_list=30  OR media_trade_list=30001) AND (multi_geohash_surround="wx4gp"  OR multi_geohash_surround="wx4gpqj"  OR multi_geohash_surround="wx4gpq"  OR multi_geohash_surround="-1") AND (os_list=-1  OR os_list=8801913418331836625) AND (residence_level_list=-1  OR residence_level_list=4446806907895041723) AND (check_status=0) AND (industry_exact_crowd_tags=62259300007756367  OR industry_exact_crowd_tags=827821124185282642  OR industry_exact_crowd_tags=1736885184507453537  OR industry_exact_crowd_tags=1786381960166275773  OR industry_exact_crowd_tags=2447888834889859308  OR industry_exact_crowd_tags=4089293343970146178  OR industry_exact_crowd_tags=4997685626248627877  OR industry_exact_crowd_tags=5082906975180188694  OR industry_exact_crowd_tags=6233776986425357665  OR industry_exact_crowd_tags=6280213215139896165  OR industry_exact_crowd_tags=6611248751053673021  OR industry_exact_crowd_tags=6990041002335227794  OR industry_exact_crowd_tags=7824369360999110813  OR industry_exact_crowd_tags=-1) AND (((multi_in(task_type,"0|10000|50000|50001|60001|90000|90001|90002|110000|240000|260000|280002|280004|280005|350000|350001|350020|370000|370001|390000|460000|490001|600000|600001|600002","|") OR multi_in(action_type,"128|160|177|129|116|117|118|120|138|156","|")) AND multi_notin(action_type,"163|196|164|197|165|200|184|170|187|92|126","|"))) AND ((notin(id,"6475634789599348344|195756431|189279530|216140542|213432363|4510564146827331015|3843306077049549813|217536181|198561015|216460872|198254807|3524173534128122501|215845445|209827459|217559498|169981625|214545359|214544614|192847772|193580054") AND ((principal_id!=1227511248) OR (main_task_type!=370000)) AND ((principal_id!=1181011317) OR (main_task_type!=370000)) AND ((principal_id!=1172411226) OR (main_task_type!=370000)) AND ((principal_id!=1170311337) OR (main_task_type!=370000)))) AND notin(principal_id,"240479|2088241814009809") AND multi_notin(action_type,"92|126|163|164|165|170|184|187|196|197|200","|") AND (trade_white_list=-1  OR trade_white_list=6914483954861104026) AND ((time_schema=-1)) AND multi_notin(revert_dmp_crowd_list,"5637734191461532|10421326407713368|23367851710885002|27083629581040114|29400420043793316|30753831922038608|33294688080914456|36416540307015965|44052603451752495|48430184856964628|66721311728307473|87030316311563987|87529376695403269|90158626388090843|104033400389437772|107434936007719441|111592208306732731|133229218982973349|136508819363725343|143038867980011985|148790517211461772|149105332844477262|150584675695955010|155315078489404244|166767768408572878|183909698800259021|185264795492662105|189833035796356317|191580634316447036|194171822973045541|197179953056326843|199380149305513760|202320698763400962|215452771631579800|233599548987075335|239108176737916544|246909017834765044|249572067671046390|252234369200068838|258451594701723729|259850563434239687|272699876578457004|275738228941486304|283485009124391469|295343450409151711|295920237011172496|298548467087802939|302357212672712777|305894612110593709|308450464572146902|314328559526116851|315917626418726442|317393824538282122|345672431355907987|351177770250818815|358288581176222318|365867429390253057|366919957265993845|367875449641359254|375844281416563938|380930059517083120|385381034539043102|386244092518874298|387803733122001257|389952379943961663|395659960955585426|401459796096628169|409989846348156819|417366902844080468|418964542690841608|430843337843696834|438859066326173508|441484535687323116|445386387418839949|445685895250363945|457262340847272985|465256568020770346|465994058794084682|466925282859533922|474142912008717485|476010934626313553|480628198394047030|490516580181789988|492751743537576394|493019678082097277|497687872582857511|507932897101632871|507933788927197088|511087353217449425|512641356992506533|524664108052658646|529096450363668690|533238604908455091|537064158900225875|538008677777638650|541340875610020211|547268757921027174|548644040722331979|548757088529631241|548803121315534632|550413806044254934|550480418276948194|551790303398313210|575933196699733166|577247390262962169|586156889249376958|586210193436886643|587345323078085273|589134709704530438|591262554729901033|592553409130228688|593489193004271898|594321705793104786|596872933910002684|598403081154232540|602629373463340129|607305267358025385|608420815304738523|609325320743091585|613716478803091246|628912115463029403|632969883648488452|635779557714070338|645463977505087970|647780768480444423|648153624972239255|664312155779648781|673005055670732504|681302342700686339|681434360288199670|688174321637497873|688450386062323849|689441434638397342|691090288588152906|702911525079471119|705402620138606806|724879565061123912|729062876731615096|732748783493014733|740658952307933351|744093222480300852|744401799618715916|748025866173257429|752741678739753720|754275460112368015|756845201043546944|760254612639769451|762245412636628376|763456863780351582|764212277800875039|772324932999610202|783996548534194085|786070876613616395|791087070862410921|794508616080226452|796554158432554744|797352561971869983|800038456554015986|824555781203225211|827086726819218777|831976685841040580|835661315246773004|836470534800279277|842399904770120443|846377720977856759|857274708647691701|859275380421474931|861475274332649530|877618590944097304|889351768596069165|889829498328886911|893836511033851251|902139188415177936|904323583940839923|905974603775779132|906485830031512422|909542406996424979|928863632228479293|932416416456986439|934413286393169496|935316141516518691|937129864520265590|944387741063861580|946466870949064750|953725403889524129|955262793273736746|957625936587417475|964270511763724967|971839921652238979|972568829847659779|975649589957882907|977409077713306899|989301380339093395|990403175585484985|993663189416669436|1003948920988905987|1010564209742871119|1015554456128092479|1015935850694643445|1016569545476859224|1018135237949366105|1026443018251470178|1026709111870321301|1031859689658104547|1032417190634853238|1032971003171242104|1034775432336119992|1037619393132899306|1045381679538460090|1048477362092208216|1048624328365060735|1063153601770292188|1075244409467143793|1077908919238578677|1100876544853156264|1107488919115868516|1108392550188547140|1116159091908487697|1117997698639127974|1118744291562853614|1126142484100031424|1133597934396877411|1135062923302644064|1146719839142622551|1151416906385284503|1153655580274992015|1156079824044497964|1156503537449409593|1158777608372808668|1160732500952248957|1170015517401660312|1178987400530877956|1181644536786879753|1182399965847493114|1189397819509565710|1189824977556052449|1189847073211224697|1192140642758335702|1193309041866303243|1204736144812153469|1209699020252869380|1213429432915706734|1215650825818137910|1218089153521133036|1218955957467849816|1225249035406961350|1233638356551675934|1235368335981868023|1244292644102115376|1249412593275587332|1250752038958897510|1262400311970516798|1267437703480463813|1276336235859457668|1278675305981640472|1280950523484047624|1291189228635461975|1293456595765979885|1294125415507797616|1303624088204507125|1304119456319382630|1307525486276264273|1319486256408903397|1334933285350416964|1335368358870947698|1346259553084795024|1349590587630565684|1350796616793536463|1354896658082155747|1356092128310103173|1358711027045303066|1362981270616758271|1365886112415004795|1367551870585577945|1370307029700577883|1373719134292687745|1375426105464047396|1377062452431019815|1383017127105969368|1406757122057726348|1409102039743605876|1409329775286802288|1409825895920546724|1411570617729139715|1413845622640046948|1414797806369924775|1415427615580272912|1417555895995676841|1419632538424211619|1430639213511473614|1432678121379010272|1436651329486701242|1449446758868203865|1454296237776820844|1456109573124531169|1458645805793295012|1465163337139396260|1473414903860014888|1476339845163904882|1479067438865131218|1489708222385060052|1493053158595468932|1499468513289311896|1500108427745850097|1501367593904930810|1503550316221473325|1505151123242802667|1512864343749540042|1517380636369919313|1519098993159165769|1522021317591378249|1532640261621808102|1534319389660017365|1542003019410810562|1543853910519262436|1547615024824291362|1548458547380401800|1551348587731891737|1553586689717417458|1568302643848010238|1574933076144541530|1578466505563277620|1592240385985023547|1595263276984806504|1600525815616405238|1601129448843702498|1601617015645560582|1604372393927896015|1605188773904756187|1611974447653925695|1614474138215598975|1616177719481260220|1616811887045996992|1618801273322482570|1623021785443978034|1629976574434638473|1630546589722673776|1633376763762401725|1638103526754540575|1639201580476279449|1640922458831392091|1642220440357009595|1642953124233207109|1648506842391480955|1650144106744686118|1657304709848517023|1659273874347028424|1659970228717880113|1662768809631314423|1671753489931602335|1672554858630623252|1673709211225680797|1678858634611892628|1683587631463623326|1689754686230895463|1689871606324865786|1692700079436796602|1701668303614071895|1702769540331534965|1708329782358196461|1721567218767098675|1722857646052579237|1725094303710144975|1727224825537927057|1735874461438697548|1737602994923776275|1749449664194404810|1757576204406287116|1759239165347490459|1763536817728700404|1763630150707961740|1773182670683107826|1774531857138892390|1775132791176423962|1781640474815752592|1782356198213630758|1800838255042525153|1804635495971306083|1808306753693574433|1809930503852954547|1820003065400360294|1829449011676556479|1831594209646212365|1833051400484929547|1833750345119274767|1844287756924805970|1844366663500418249|1848149834112470684|1850923446989005330|1854395823384527969|1868320845862990917|1875765710267099669|1882597966350255090|1884637555844513804|1894789522889017819|1895446498805265772|1899506837648032995|1903481081205642053|1903921918188017240|1904692066799270589|1914262490880118643|1916619092437516875|1917547317089926816|1923140374637382480|1924201746631280893|1926511600657491954|1926660718349007944|1933123927426885458|1948005193480447449|1949303459705399844|1953741211255595589|1961692750956337750|1962747871556473880|1966063698621580524|1973853923975960340|1983307883372466256|1983370118414261279|1985072831820682491|1986670667262089001|1990821803888150116|2002195537064250521|2008041509861515668|2010452049158891615|2029633141236870078|2033331381796529704|2035016915112908158|2037431870591706352|2046642838223153980|2047277509970344038|2049870678294918005|2054545837768134725|2055151501373980576|2056263284271039374|2060008793822863644|2060047392149723862|2081159466628841234|2081684536650050391|2086333055204341748|2086875928567410544|2095982928840188189|2096213158033650149|2096973346350643507|2099474792984098529|2102780030900820078|2106338876120715248|2108737458356280528|2113413947016443975|2125038011301789445|2126123175046365189|2127372231925647680|2142577123405314319|2144772846429462140|2145678375115807502|2149274362459474197|2149597896280534229|2149995147593360696|2151623893605278500|2155912517027471041|2158752565093645763|2161571259421669865|2163170900500695861|2165849412289455546|2175080775368416467|2190524648079249138|2193860294221593119|2203664248504199182|2211938810840426492|2214978261395951201|2218228578254884355|2233585391425727110|2244086298372398655|2245382694629502995|2253247499129962997|2264585557133117643|2269719658902288266|2295198824691237846|2295381081716191993|2295831166636150513|2304737509844097729|2305807766688776164|2306659588197817981|2320318025450256629|2322437612842323994|2327152927842824155|2346095035996830835|2353815318792868398|2354242769496966372|2360126818722444653|2361086361773035298|2361514089374541938|2362740701407750754|2363950859866051166|2364213485790398828|2366732114153817601|2373566356276266565|2388783824845271417|2389450290526541995|2389923429908324520|2392047706711885419|2393022351239700931|2397546039274475894|2403873389707008078|2404810394320599195|2414351439981423303|2427755983648038880|2428597642003653026|2438243558048208801|2439116950152769546|2443079295525700128|2448578679635677155|2466371691562526581|2468124810427521803|2473144647860356997|2474884608455623757|2478384021179869466|2480371557222621022|2480827231141504295|2482599342614589839|2490567743385329027|2494793910430836002|2494956457224850764|2497011821635743479|2506305769643279158|2516923794279219644|2522309841891901585|2525142672783259774|2530054148388300581|2531745787100392971|2533195846813940197|2555066788308648011|2557960972641157318|2558262466997387862|2562389290214454827|2563099689900590670|2565916594383645876|2575703207969851524|2577739725951873680|2582712514445097718|2591425981504543223|2595856475971663311|2596356595376202745|2605355039786320844|2608155885408642784|2618785049916117307|2620498571006232324|2632339139186459452|2636387680822708255|2638612097829965638|2643258142040416747|2643453591577392192|2649862984896973367|2655142749365621444|2658498325382053214|2662354975572707351|2667381081733756423|2667925297010607241|2675415317638857261|2688434705692281974|2689629328857165576|2695016151923221430|2697946190272561207|2700951059180732264|2705477146678741175|2709718148206981991|2722383512088198070|2728000341714394313|2730279654927732516|2733231804483951316|2734926835689427567|2740767047632074192|2748230301981828567|2752387886358705095|2755897317060497207|2757138834794963004|2759574999022989662|2775978995170276962|2778684253974577811|2779742586423104108|2784124906672390235|2784927879597233636|2791018003356123328|2793038982927652493|2793877229334049678|2807717232258884885|2810200243641028728|2819893986312186836|2824409072163578397|2825312427970975785|2832700080546928668|2833912358365812910|2841891127303911767|2843161539269783469|2843834369572590636|2847477086209206628|2848530961384323193|2867224052186052900|2867808980992721096|2871854087559127248|2872637424205574689|2874051782127727403|2884452196347304052|2886345653147846505|2902286709660036982|2905059829840025218|2906039981595777811|2912975316042196607|2917439603692104996|2918807261694046828|2931965767916242284|2937378892054377559|2940895344234151681|2952012468008661743|2957926261669116707|2960873815340065345|2963023593952155161|2963270807423842803|2984037558639585913|2986345870523949269|2992693773065424191|3001276958500690676|3001547490758231358|3001863097760374072|3009855578441818281|3013194006060469746|3019923028763831515|3021272232688997816|3024213564643376315|3024428779631500464|3028893545148263813|3033523744219563474|3034494287124948475|3042125848423903510|3049064433781498532|3052658623920413922|3058232176497537360|3059710805123557602|3067232573817139861|3069654674733290329|3071173378262395709|3077728678967542653|3080431267274159104|3081820194176992101|3092593420626605645|3095194714435486840|3098351176009038545|3106697669704234399|3125116358420013601|3134381747317042307|3143036056335644832|3157119038081495428|3159848395679153998|3160549143984246455|3160878831964999952|3171910573349384978|3173298173222767967|3174465880044347428|3182435804201605674|3184334551038132041|3184454028992958931|3185497168565527849|3191514805239747615|3198228125033825985|3200306716849908400|3202035044903988889|3203008517208686526|3204720673652719003|3225417771663481396|3244183826734163396|3245796040374200973|3248978483292897922|3251633988282302693|3252639117004504346|3254024184558035222|3255894029066892669|3256133761038378821|3261085223935648113|3269277466917444520|3272515818739130528|3274989458195836492|3283158198824812672|3283662492806153079|3292181098789090416|3297439947374100573|3300827056858622979|3302282913511139161|3302789032980081588|3307957162320415612|3320581907248294330|3323071310563242585|3329002682378173432|3337292194888390264|3340739897282347353|3344518487318824635|3353444940169534931|3369461411826636722|3373888085997347830|3375971346285383601|3379403918392394285|3380368076272920749|3380778108054620000|3390094014912462612|3390508366812653658|3396503793801001529|3399148539924094718|3403006771712068983|3410244149131760311|3411826526540793206|3416538638666303124|3420028639497048145|3420747679175495193|3425544862116177740|3429776279885215777|3432492375719057461|3434286076318749587|3434324356552235307|3439660933292033909|3440800257849020965|3455894576291513754|3456180316951164059|3459162982531848181|3461494427763747944|3462367420440338375|3463032224496578132|3463654131302517675|3467717351674182517|3470369055234033527|3475365993814876012|3480968250485176084|3489507003812105967|3489818394865033320|3489821261930273863|3495972581458011928|3496803711424507615|3503781285445422673|3508892313392656933|3509229703728825229|3509802576052755489|3520230087826727833|3520388474346944427|3537517950924483930|3539372961032684495|3549286707899036020|3553098651045674329|3554551528325719984|3554686206164791954|3557958152608583487|3558016241757299354|3559391918815892287|3559528351764936669|3560358712445080929|3563349637278500138|3566077523056861850|3574052818647834954|3574766816666937211|3574860469440465439|3599583291125658152|3608520459048466702|3609644611709505753|3617906903502768064|3618777616842792949|3622348153824678489|3631576752168539405|3638143347573647486|3641942806650080969|3659328527642984204|3662206145065451694|3669939151827300495|3673568895658075182|3673767760835614655|3679225498097474744|3685123345042813699|3690279859640902157|3704354323851158694|3711218611905125323|3711630966892169280|3728998941528275592|3739191931152387760|3741937608032437889|3743055908108053839|3759430787189532358|3764344354001540261|3774520996410939525|3778472624835154921|3784945802421074977|3794331713349502693|3803602485262173182|3805891168852734845|3808116261771927314|3811038714126508648|3820614363250863078|3823927528848640197|3838657959916677451|3840206973384358139|3853398235500512833|3881096489949568598|3888532547080587187|3905936074744326026|3906200685460806543|3907998884767546514|3912660530970288906|3923434948547435780|3924299070097390011|3931358126031530325|3934643523093980365|3937181496129679688|3941345195311117139|3941466221140099767|3951892297840141842|3952127983788246207|3952185047972105017|3968995933836813324|3970483972005289595|3975871881212998986|3977147258083519776|3983284102422568023|3984522186143704798|3995339217169580604|3995392786349003848|4011776700610257531|4012044823370673957|4012442566651366301|4014493038838395919|4015064899623804331|4018732030904241843|4021108510932186985|4026367376110127833|4034300088310982790|4037134090695329733|4038409269828300304|4044645379378325192|4052001859156141668|4054694926783967922|4056174395398851900|4058180776970329135|4059592026356457503|4072604202124224422|4072990468456801361|4075410046483903530|4082290463359045077|4098640391338452268|4104806133536331873|4108924151311068693|4112672064329359926|4120871090823434322|4127496891904505060|4140994943129171709|4142535900170119214|4148955737548569429|4154382842214985180|4157913567381258791|4159432956581893383|4163269982005121081|4175853273657313146|4183868309206869924|4184628844986843800|4193597314914178874|4195573600200879293|4197285934356744338|4197819968567449026|4199107481887601148|4206987851521590993|4216660245435073472|4219629952864477663|4220590039464252518|4222707366536263745|4227600976964687723|4228106698646783936|4229568688327401851|4231071348316821643|4235413797016217376|4238159454873377921|4245253408159788597|4246543004624135098|4250591475555058955|4254272913733692412|4281670941117070315|4282264724930202723|4283017939155489514|4293183201963326300|4293852779792943223|4296184522045043116|4299788163683383478|4299909562970624794|4306446891395881447|4319366866451031405|4321319288945685043|4321512082895880471|4324598025897762853|4325434433407238059|4337181358226587212|4348357299215220511|4348979992265103665|4350170266710480191|4350809138898316057|4351775788568301237|4352377346915540336|4353030789783018716|4353529905915543178|4355986029291505646|4363760724426190871|4366016193675579339|4369385116759077642|4371493452855772198|4376537688844335527|4381191371594661012|4386844055596997482|4387455291899127642|4389427355246589146|4389856914184513550|4397548642899779859|4402187465346737740|4408651760735251793|4408867649212194569|4414611345678559174|4418643242434693134|4421984075002019422|4427652982569088882|4431489420982637976|4431915457182645948|4434502512410396232|4435065485730234823|4439933109056307129|4443032338661062161|4445326088033573281|4449876933395514699|4454841860739034947|4463664838098509526|4463862290348638840|4467191689692506294|4477055398147542710|4485732206843624691|4486071203834471944|4487842406653658873|4493378343145542247|4504216067943143508|4504295051450477279|4507467461770903401|4514173723503703424|4514930364965081801|4525741821565439279|4527915792442078651|4539957086139780409|4543013887976062074|4550542783877405105|4551703281759342594|4556104590302298656|4557427325124576968|4573251237730794197|4575060091961266667|4583174606283295648|4598224170733277643|4600920464274326339|4617438993514596617|4620129452780159194|4630998213412618118|4639749796637531569|4644321717101240928|4644921317275114487|4659734952783217676|4660389754470839922|4662673078761801111|4675547750955503146|4679837129536631517|4689226146540771366|4689229982094282303|4690403797490510537|4701366813499476862|4701671512746148302|4714635458572553812|4717028368788807419|4717483843982933173|4718849630511489598|4729123338413298847|4731537769182829422|4743832780126708924|4745679180379879355|4750049040531924506|4755992205200896737|4760178867268493251|4761595333979488844|4763012544107224994|4767967870776205252|4769678538415333229|4774922277500520201|4777073756267565414|4781602201768116415|4781901904033369860|4793916920076692947|4805543592267775880|4808234091369840277|4809001889294722846|4811082316306423939|4812903093841330797|4815021311670818990|4821506018803905782|4826553460072754103|4831343207115208931|4832700386324178586|4834128943123379009|4837961672293471253|4845822693583575754|4849237936747232697|4851629683686426249|4857230454301078616|4860977946260176666|4865554948577740027|4873255477797280864|4879743119899185914|4883555406448814901|4886552603672021359|4894854908959984226|4898575666399697035|4903477421708655678|4907559096669603550|4909947321921093431|4910776138005720986|4916837285238885256|4916931664078674676|4924449280824364930|4926295695812304109|4926642289678042433|4926855826369036914|4933643422047421533|4937340448616119334|4940926476029608713|4945290992396400937|4947257184529521860|4957773611697947604|4962331305699119953|4963285381472563400|4963680843186082473|4977126143116859084|4980128858928091959|4980840965774107027|4980961092679232167|4982003679575281062|4990238056167648438|4992665548419449201|4993090594392915194|5000577550748592645|5002399014698867142|5003793608495305467|5016288593038357765|5029638462337405616|5041393976407225547|5045538840433334273|5055576772511901349|5055897121292330910|5064801846530979849|5065013814301743079|5067647126955470787|5071361154015502993|5083722612149545454|5097913628069251073|5109434403130083175|5121568875271962212|5125310625670347014|5127157685940368720|5128020609025062586|5129659343333955816|5131275031348032505|5134556720318396237|5138186861373234674|5154411228035561956|5166421947530334008|5167418221880061636|5173498228448475688|5182176243384195293|5185724865275792228|5187147731091012855|5190280597173853371|5191192331134327088|5203408567537069391|5221297800323146117|5222783211479861429|5224020730497220518|5224960903157989852|5224994562460397326|5237612018180226891|5238477706456005958|5247556675994235508|5252831311376952371|5268611836877740855|5272905164199155145|5287903834790777090|5292800395722960316|5293367090417407765|5293696651857090147|5298707695819969250|5299907243906760603|5302080348157466972|5313596598538704004|5335469023431368822|5337460965685050833|5344656569146860827|5344948936568487347|5349649777614223114|5354696518022586143|5357947518877142563|5360805688886221139|5364388258426403989|5372387243276528819|5377519336311556795|5378344427357342558|5379813899527081597|5381426514494379971|5387842800310263669|5389955066582442929|5392886504389177941|5394945495608225914|5398543164896531998|5406528205722901353|5407399447428740555|5408159106011845634|5414789089019024881|5415620063608024214|5415750012910007275|5416297530539574986|5417418546111900537|5418087132472107600|5418374627551213460|5433786223103980486|5437247771236890699|5441497546498685816|5444016725099269548|5445980309172680649|5457359910717558746|5459824049596271508|5460296402711072519|5468319277184464568|5468427251794703980|5469837982534609037|5471534552081942566|5472287489170119380|5477258255908249572|5482128942362369272|5483812610715176170|5486765377074037634|5487122008234530316|5490134300792226800|5493451408616107946|5497473826186382029|5507343667701712646|5519898457427123959|5523404471425325698|5531029499762910203|5541552843745538006|5542862215049074327|5546969018530233349|5548544853901385594|5551287528219887729|5556505310502069390|5557600070811242456|5562490586236697755|5563281878151071093|5566973836637419858|5573448393943000276|5577569381405834367|5582564389875567537|5584761654632610884|5597719257377207152|5599024917032619780|5602444783721166347|5604192544205968618|5607398913059342204|5609435929072717361|5610055440914452168|5611392766277153203|5614025894205244891|5616743409128072812|5619798685497394989|5621015019499352636|5633202419088777433|5641614257358017466|5649004160370041306|5650417102604451087|5659519509529511131|5663648355822535169|5664565739542882103|5667980134220363693|5668928285040381377|5676639591350098546|5676840034302774532|5678563926593473355|5682244411619832535|5686190834956262854|5687624730205330843|5693537378019438970|5694353354235473911|5705629516372011450|5710538190477397665|5725029508601244525|5726300010002125179|5732177138198761896|5733067649791863337|5733573908721220456|5744385197957353859|5749119337571802624|5768598349530546119|5769599975061036713|5776800011958472495|5780509106636762018|5780862310345576333|5784345262791129468|5793292355019243990|5794587902620129772|5796774382155752430|5803969294789096855|5812279112792795337|5814907704875158504|5816034890450347516|5816630432364871749|5817887821142766123|5819039012037429443|5824680124642483391|5832218837170417192|5834585044056326269|5838040330153146281|5840363657063200246|5841194307479399784|5848124772517107968|5852379117506523080|5853635968699138889|5854409948674702272|5855488272497502452|5856337820855260731|5871070112577749245|5872276961625206870|5882695494863461719|5895647355243195068|5897614645473867621|5906391302397393790|5910717864255562434|5911049957502362892|5924106647802217771|5928197198259711891|5937610216541814475|5939660735061296759|5941171962748111525|5946775144414406790|5955716064372667812|5958337347488140261|5961484035703601184|5963154696805769405|5968792638987035505|5973371104146919796|5974103698373908689|5995073813216706272|6004200822853667897|6004441666452620906|6005535519817300773|6009991548865070705|6027481981553151987|6032404554012384255|6045086227030006437|6046509399973044300|6050739914349140077|6055600165863036587|6063320169344610641|6069475826700022832|6071510515019746729|6073191334637059767|6074657735022747118|6084267339273646238|6085458715529997944|6091807365899075536|6094778462803537553|6096008377933337351|6100304231699768306|6102941770155122146|6106076610316145790|6118559462115710058|6119284347150408736|6122996509223071780|6133440989325448467|6144751361972823795|6149307115505747379|6154596974833111052|6157658647001327798|6159488445096809249|6166825190363916095|6167069809757838479|6187652214884755104|6187859293966067717|6190011466112486857|6191403180303593823|6193221293509007403|6196883537846097590|6201035319775952763|6201599424237112764|6202905813962370759|6218855226393248227|6225131449350592701|6234276271055374006|6253397706812400066|6254170892432198428|6259323069698078500|6260078245338883215|6262784989736468183|6262993221114269256|6266035838226707399|6267856996063051337|6276165368858507131|6277658929013630369|6284099128881499364|6287570425926873467|6287689211827716777|6289931615324396286|6291915797941335266|6292839569038852137|6297195051599914393|6298633108777248701|6319061768670402707|6322026349658145437|6322285009656790112|6322687745934503686|6325403300141499255|6326029331066177377|6326959099516356598|6333878370574674958|6334626756158433334|6341140769406451706|6348665024099338756|6348776935949000220|6352304477256183270|6363439627623339586|6370920587410740564|6374509405177515056|6377703626441936496|6381378924215974072|6385865001439324596|6406641211335001706|6420814026058195560|6429530322507929882|6433621571751635643|6438337168762368034|6439118317857311255|6441216378547266542|6443728463956335390|6450220667825173356|6450652188547508882|6450912733351192426|6458445119853642744|6459526665829768154|6462949218275979687|6467639346265235186|6469496521501287676|6472287712654288232|6476542227608815917|6476747126336754702|6477703022831829029|6486421933055471118|6493586241476144729|6495866860732251149|6498078197156122111|6501611084072693984|6502478457314992655|6509964394070585010|6513348775047919375|6519242670461195775|6538587013535073714|6544810418382777365|6544944309935075127|6550800432030197468|6552132832323504669|6556219649081943322|6556232446528517845|6561794137716323014|6562850229369937039|6570928032608929666|6575603173999281759|6591484142096753783|6591718654328770432|6599526974063259913|6614546878327048872|6615203942937041267|6615433807033846342|6636432068232945527|6652499558907579509|6658275417495600301|6664731875915360052|6667121552625648916|6674842210888888370|6677392646082118593|6677788463790307997|6680040646309276955|6702701530574716005|6704927075817412309|6712293551382348259|6716052973080756867|6725743229378097525|6737933295778578519|6742722999341729010|6744866666169417518|6746604291808669922|6754084823451559302|6757537766482998640|6766472624543083567|6770145479267971874|6771038792897870411|6780339571053798264|6782309279225528120|6782462608575919079|6792921691235342138|6794875295577062192|6799650188554984410|6800438783364354240|6808741226756280140|6811420480921523497|6835474884878417654|6854364421649647704|6858432352256830570|6867122808833897416|6868247377639421027|6873387700294144236|6875595825546032713|6877345639457250282|6891481589721396878|6891831935515824424|6895678531284775574|6898934754607525708|6902769020027380982|6908177964903091132|6908939174870881058|6916086964794712102|6916828752204613125|6916919532180024948|6919064417556495152|6933065266645854823|6943249092018012001|6945465651164141663|6952773833795135640|6954377091165413365|6957133592295715460|6966117765143109833|6982243630366566813|6994774477791290724|7002187597783353423|7002312700664095496|7005053982430206896|7023563284586878323|7025355288730787783|7027979360657416083|7034275815611243548|7037548794142504665|7043227301514079724|7048421508781366392|7060670781880858356|7061321950704500637|7072477291020706174|7078361330605211457|7080850748737367133|7091867197285247959|7092128252178185096|7093489988734510568|7103585234918393150|7107863942583934078|7113910168229518088|7115670870961876877|7117052183052782647|7118428564014251239|7123034230101384433|7124085500064522244|7127734764678202995|7137228212069573292|7150237725971351834|7151404066139938643|7152867641240471648|7152897216453299742|7153513200481039295|7153946746447650059|7165521721378006131|7170372936009157316|7171139095417697199|7172503490980019608|7175244697360865113|7184486752382143674|7195180296689410538|7196039936002118364|7200956438087543167|7211422289939603368|7219499034088572478|7235824984627172031|7237567977282845859|7242126097655035122|7259895000157589525|7261499851522738016|7265461245248533151|7277020936485935243|7279194388097939860|7279716268695068888|7279808762255898217|7290587630921208946|7303698522878422834|7317665002862157587|7317844845955167941|7322823005816170907|7323657979477305002|7324960414535540935|7328162769465479183|7335361381469194553|7335981909764031719|7345399908932400129|7347214846049563646|7348275286438209675|7348530041422310088|7349238728522835661|7349287651739124365|7354777161350645910|7359874747333822663|7368339210377754716|7380846170499217851|7384951673460249766|7385559000830485030|7394843360063919380|7396519184734503116|7411221685841426659|7411603020380228027|7412247484251617660|7413721996324021866|7421593704804532521|7424265435790900654|7431542255952468482|7432379515900986334|7439311855191134124|7444518248726319854|7446756192051414128|7447727788035761039|7462015636436944522|7465771486077373319|7476236816124629726|7477274755974300585|7490219304434157105|7497005647853607784|7498591410538949497|7501162314519098707|7521561118773344549|7529303144963094356|7529609734435352123|7534359669237152653|7535115060019323709|7535850192267750777|7542841542965090533|7553709866366910753|7559656853084233420|7565031560048891840|7567369980324022117|7575240533541970957|7586283197450408357|7589992968183738878|7591885700795337287|7605376411243662767|7607073862444181384|7608209571732905899|7609409925000932142|7615708460422674063|7618507529497498933|7624777277171035904|7626683374343771241|7628468696011798809|7630295204531363682|7634413366179214328|7636497665892886065|7642103025154419805|7643659965562845892|7645544811651461513|7649880720836455546|7654339354391167325|7659229186680307241|7664696330994137507|7667708091037510729|7672069523792918281|7672932510390771754|7680208332745093715|7680597035796687496|7684673255914731812|7699383134439196363|7700642400737665227|7707798211226508580|7709720155401257896|7711850014690796327|7718754655273785620|7719674066438221647|7720043534553267382|7722660230540058570|7723621294253876279|7724960477808229584|7728732485545580830|7748826304607783819|7752590779816181045|7757714047675829949|7760348304066077409|7763410573786031565|7769759454101429450|7770554067873708229|7778702934315099131|7781578768488225902|7785273586446553703|7795875978641549161|7801268296698626617|7804021376467398957|7804766465441449900|7814008775470702553|7815961424343723104|7829825908671394106|7838011005099002983|7846769707704475808|7849852159758870497|7850835181058478203|7852307566517457384|7867346174089653611|7868822447952006407|7871562411633849406|7876615731937185754|7885130387109996377|7888316910623448906|7891323386651056096|7895632981436398041|7902022978864167408|7903553224320521142|7920803579322531179|7923264361902703790|7928386798370682309|7929513854428456691|7952662308059061811|7955254853200272555|7962515973930250879|7969087300055177075|7970483144793289296|7971483603727877950|7973745694580272507|7978754808204702113|7993584713044273593|8002298157727823182|8003141579741147683|8004935228890814817|8021215434523500797|8025303030061902768|8044808468476868848|8044818825107612195|8045545803248903524|8057826828383357437|8060404016850561910|8060898952746067554|8069712888855734022|8076121844657685340|8077489611373209231|8082544056844620238|8092240343796501260|8097418386160254702|8099488269015043724|8103850638713626512|8111289600043980040|8114075957216474308|8114878071256126220|8119358241409857676|8120929032782623278|8128996410901013529|8130526370586159288|8141716320532087908|8146284556047414536|8150950388236697146|8153227335946369636|8153545060257345005|8156184929557456281|8159013890065972859|8161704723665321020|8163152649220878988|8163987992050506463|8175515090326522904|8178876775434288455|8184993431829792499|8186369500638673393|8186838536639775349|8188490068373513243|8189445396297846435|8189883069116662168|8195593802427051243|8196816867545150954|8202363239937373696|8210285584902779463|8210418981266381804|8214974454505151920|8216608787954270056|8217270931362830975|8222318756648510876|8225712845889083239|8230683062511199381|8235343106696653639|8237273884567459338|8253374187706076502|8256277730398784662|8260291877063340832|8260896932811825534|8262409027666380668|8268210612408565385|8272431557759582976|8277057343048183207|8284506989492644472|8300722456710946043|8304938370342730030|8321258504841720937|8326064078270443526|8334228575723025955|8335807566320912439|8345558155204290783|8347954738097856284|8348469857284778577|8354997978662586578|8363538411356663393|8365520388681082691|8369064774809400725|8380222316129916115|8384318088960583435|8385402892429108552|8385507790830146739|8386488509135048360|8394123542861158915|8396220399679445178|8397162452616350719|8403564585524121350|8408512935580609151|8411157287263745098|8412987989657097085|8429564861692598596|8433931582757195233|8440211576474784438|8447590453195176115|8452131087037106425|8453010334782965550|8455850947222878845|8457976292115135982|8468524825705586881|8472467607935452032|8473292158398907470|8477255197965113214|8479047032913821539|8480507850606972546|8488717809256455734|8488772903933211399|8496553785736326481|8497832332151367369|8499633263511942880|8504812625883830545|8504946057027607411|8509957273726072874|8514505196725920505|8514853874514420856|8516858371450393131|8524860730010081732|8526547143001179891|8527725625486891920|8528655360312998618|8530100344045283147|8533221102537378603|8541164510730101569|8542339560579194887|8549616884876311189|8550463045365894609|8561499691937019152|8568638726697373574|8583802378335046150|8588722046507807559|8594156442546510155|8606831117442494281|8612569998378418925|8616626209625142531|8622432727368725087|8630775225010796362|8630970795421172217|8633978009760216977|8637957107368856783|8642523915236570669|8658571253084677234|8660687257083649065|8660752507942616726|8661490229867197088|8662698489571402669|8663071671053839905|8663966749022265898|8664961820560324187|8667827064879450289|8668389284388063285|8674245010268024161|8686136224344334714|8686233797279325071|8686507252415401279|8713440624918839266|8726573942870010839|8744158365734277096|8744728391682451576|8748531661249165751|8750688239256750671|8760009378168093804|8764282472449329447|8766379345326388075|8767398057650548033|8767653358788418024|8768481996794194010|8770065044936398260|8786605052210114368|8797551946656038784|8798740475536285182|8798882107021862478|8805068739489111014|8806245837772628398|8807764678048766143|8810861118092824146|8811958633774966148|8814396450831830931|8814661884821722120|8816129863008574709|8821732807022329448|8822636349109690461|8827036613175626403|8830284233770517921|8830808285324087034|8843451765839637371|8844129207698321161|8850907505472075137|8871539142344549682|8877553464155146066|8882186551434595007|8884247443319058410|8886518124080272897|8888802062739313066|8889602018639350470|8890285347710020174|8896126610287577584|8904455390323721979|8907216116451562340|8910568134323161895|8926247752269997446|8948342433508255283|8949058682849794544|8950827748584865834|8958293199814926788|8960329672528111489|8960913608160140911|8961965114355462817|8963403499699833965|8967296789866984346|8968677851216355785|8971656155345003197|8972152448693838982|8975381434902762797|8977182983567005327|8983570955441589649|8983837057304749534|8984033893347002243|8985379367116779272|8996942460885042773|8998117260837476154|9014346026039407275|9023325712016083455|9028957015147715778|9032251554561707119|9034761555997639796|9035332854328721347|9038032116989240613|9038797477039636376|9040862755500309347|9041890198225730833|9046935005737138274|9047567008726692759|9050793684933306863|9052576283060199846|9055206912876188807|9062474302971370580|9064074811885937637|9067135362633513909|9067945613386044028|9078335320958724143|9080271188985467577|9083418030816360723|9091072448137045362|9094087630145268426|9097139827007743916|9103343802825231458|9121548660176755032|9124724503646197348|9130306307614691278|9133362071061073666|9144258873473485642|9149316658309580577|9149391194014442172|9151448762464370130|9155693472854508682|9159352071276387901|9171401987906631436|9174899865811320194|9175095785076288313|9175267492619782275|9177429488594503400|9178639546010259037|9182539153750557051|9200327151224204803|9201264649328128991|9202433410635730204|9202715445372575119|9203633591692395478|9203965631422182353|9205817006560623980|9216824084338541970|9217450901867291051","|") AND (biz_flags!=2024091316) AND download_page_link_status=0 AND (risk_ad_pos_list!="202106072200013001") AND (risk_ad_pos_list!="-2") AND (((charge_type=2 AND (raw_sort_use_bid>=50)) OR (charge_type=5 AND charge_type=7 OR bid_type=2)) AND ((ad_exp_tag=6 OR crowd_tag_type=1 OR ((dmp_crowd_list=-1 AND crowd_list=-1 AND key_word_list=-1) OR multi_in(dmp_crowd_list,"3333522908856159|4718150878600077|26160726867707606|53911088436328605|65243075618122061|76652579469349746|100257447762335763|109355582336878889|116356864788899892|185787914509954634|198427439800095440|203655466248955951|219041062774877422|268903430501002758|277018178337710883|302898381895845665|313251666302118950|325855173509782181|339492949787999556|358140517351744992|391574804930835114|406090075306627263|443325460291168109|446916560089886196|453209415183085750|458337807184502663|466789795429563112|513784364527978371|527669212511735045|536552178597757642|578013561677050338|584820572453270387|591895978652857006|595491698327940949|620998302865744969|641040204120281916|650645373821933071|653102034071838842|662745882754689033|663682593741672102|664129175121231410|678384323849472690|688321686878029324|724608667021814820|750796028317478418|802191715434364348|836006249661184022|837675203631625869|851325211019984418|871481668228020071|879150631809792992|897977074915307755|917256772328712027|925336358934019739|932136818696898004|935807398469411932|943824641814774164|1007290938830126489|1009885115658494021|1011855452497513870|1025115252073401339|1037018974136859199|1047273767921822745|1057921441959380696|1064993704917260168|1074924661783043992|1075085904834389387|1081718483775517487|1088340455449935595|1105686232136214093|1113946222361478255|1132410656868306684|1147768013972481861|1151416906385284503|1152057279132306830|1159357539519688973|1184946800191570299|1190336982043423781|1233315277404942794|1249271543703994077|1264505640176793565|1301356185935334296|1303283293394716448|1308008939634332846|1312974368069885496|1339227623070282507|1342774350185748485|1359138166546724504|1380447908626712280|1393582373431966499|1403094469572049084|1415720602191572549|1425915859615927077|1450465874091086362|1455616966466443375|1486969131238184046|1502823998215572710|1533531390711352887|1541373589214095277|1550889407928119645|1560282108347320262|1587860822930680604|1636631323541272634|1648695742433934113|1655298145148078589|1658702962928202310|1661630160103756580|1671816878886616133|1684810572737980257|1688009537028505630|1692034695051687225|1702306780321733127|1709965251615865686|1741819868449092085|1747321983423538350|1750502987504919542|1751306800789486205|1777952685723720982|1782685697910495369|1786647712123167098|1788518021552912232|1789227705023098428|1796766699037451978|1812889071260362823|1834362621560960813|1845894248405183030|1900498456205828233|1930737125682048598|1987498595293065591|2013857220756973996|2023119839380815304|2026269707669297397|2029616604959386639|2070434679516963206|2087979431701527133|2093268276935998912|2105098366554422255|2115752105531007242|2125856657000330439|2178928711195527270|2182024294800461456|2241924765086232864|2263156102528915680|2266016749572510638|2277013485024543303|2293978742559578356|2303488890249353804|2304737509844097729|2310028416960600938|2316717016002360896|2320863848188780192|2330760027611991841|2361136356874578488|2364165494948289291|2369387204049430646|2384618038437585743|2400482282480130565|2407263158666995244|2422784458061260731|2430555166581924689|2430562721750545427|2446373332509695100|2448175981756411234|2465658022894490348|2492158142334039160|2504506025413156674|2509628056085648264|2513628651075576915|2516415571944455532|2554078929456858344|2569667103696481906|2584873166178273837|2606055715972582720|2628142447071961499|2631026808013842486|2671375934780846094|2673258232577936873|2676980266209283480|2685358301281452965|2689357253786697847|2709323300061812525|2710917715265626676|2765077939097288378|2765823501146090921|2793077483536871509|2793761424051117972|2796338120781823517|2798336282118262130|2814247378347461945|2819335669315922466|2820512094125287370|2838379508129052792|2852316696187513855|2868273468618503434|2878919835435175778|2915393482961955936|2919176271476917318|2922727480286306476|2925491738373242282|2928382365742681136|2948323707515269745|2950734842180041349|3004040922188064910|3079365767644047739|3086520840679518037|3139718706766475806|3153499766609748783|3162559237114733253|3174751296707753978|3176322485588416324|3178928058560855365|3179268243218709813|3185020287307809162|3187586522685884615|3193097198290742908|3199376914261686211|3206274190023104201|3244196122661414041|3261619001254973158|3283015302835204656|3298835633281545683|3313328164167581023|3324132033119735226|3347944386577242868|3418442360951685766|3437612907703212275|3443734014332092669|3450019037903541276|3452905203887350182|3460649117726763963|3467478624883198727|3480467936828271110|3480734697265197413|3489573311941649023|3496948755774977148|3504123994288262945|3518146973654770404|3520997673891517239|3527455998495023409|3560959286651030350|3571079935112461886|3601729555189078387|3609258476146101126|3613995476507468058|3617292760009235827|3638262971482959404|3646862348026359688|3648853029385698738|3659587841067122725|3660471921122767835|3662564970088306770|3732013559705746945|3732633633425075559|3732797089764857992|3735777237708549985|3753773621227429039|3759118564243242503|3761389161842229841|3779129054809922544|3789897698125141159|3790777500945068237|3832715533426131205|3838033561831625585|3876613650345697691|3913842769284172464|3929219809129427769|3958821739900379129|3980046451058588176|4011179817991396847|4013083127002466713|4026981073928993684|4035917763722958029|4058765459633694831|4062966911570957472|4064000074903655691|4073183410556889740|4090867196743364158|4099767024750591845|4116872420086975831|4125228236672804468|4133203630555790149|4139738525835858223|4158658762768044801|4179233440193662716|4188733833633056243|4196026516983053585|4197065194817228260|4206859164609527788|4220407510213682789|4301068447367333129|4309771300552209212|4315871574732528666|4340241647670870546|4370959315495422519|4372713249706031478|4380349414147605732|4396870323119699329|4418911194631916647|4419820308409441831|4424618844675315440|4450169933710802724|4486082019753162310|4487117859040182754|4505650729817145155|4507515428319700397|4508441381264708658|4519303767377571626|4520438203823877961|4540537859958625931|4553170918296415297|4563201564315348966|4582806204961490343|4632888099965744014|4653169601229889066|4667434169491719161|4735177503287166318|4744667030622675469|4755222600419821202|4762658556139371742|4789299615164310316|4794863145425929312|4811232845055650502|4856349058467211910|4866401983863604404|4877447804048676976|4883841839953605038|4919301547109539386|4922570555034664330|4922674582690718346|4924701709711753655|4930890522472646045|4992300330096236323|4996437991676876344|5036181355566868765|5067502371531107817|5078097456805788049|5091142497461410673|5114355173678165490|5116156989068562522|5142592512703335528|5156996349843224383|5168605359835898014|5192034790015307979|5196741133950898685|5198385305120561499|5209269431603941348|5223269728864895752|5235783770817161342|5246634370024749000|5268255641650656767|5273980080364416325|5314876301384989985|5321513241018361222|5359569616055110675|5360373181187690808|5381523985307223661|5411966570857863566|5428000033876997002|5446170158392104682|5464547906691259615|5470992068072760164|5502748917414281088|5540121187272718686|5555296100837357595|5561539890064731373|5563008688298112682|5626340019562959925|5656623684723523650|5666732302512946432|5678936438752675323|5698000887962308027|5701841063880116257|5761990704841072844|5788802399754500807|5816034890450347516|5819217405283071981|5894866778801922280|5910979301357874145|5912433890708234207|5914865582223918253|5917732898970743336|5934940025835148333|5949184949292275678|5950612434191422179|5967746373570117727|5987741875925489424|5993562808129371089|6005372228985895025|6006650751490534955|6018150009418673445|6030605245295433041|6055701408410101938|6055762413218544750|6127538826954913619|6132795046378426269|6171440850448102516|6174008953969152499|6180259766804939991|6181672486632962231|6216509582377068728|6225655457394121630|6227552063070101338|6227881869398388953|6255000541656267453|6265152529707270069|6269190146222625624|6277227284907756454|6285731390885743659|6290359512645204541|6292839569038852137|6325267305079748338|6333857343513797807|6356897939282257036|6373944076690380546|6388401109472871003|6396835851197605695|6397390237513023916|6416765935169123879|6417377546069511506|6430786200553732943|6444838863219193060|6459927585572504617|6493652967603206314|6504030908137471418|6505394304216396330|6536255651110691477|6537110204440322763|6551067197893158404|6578176273367860212|6602509945923740336|6604069107512359443|6605008938360481478|6607742295170942001|6619664978456740030|6634503929134791125|6646897446807323500|6691759492813528299|6703354532260384037|6703384876838010873|6737352386338522921|6787357223192120594|6795270209797737374|6798495371313719595|6800569347005684142|6804979072343331998|6833833812384208060|6835541168894131080|6840049551021694347|6841997939692158785|6842222538173303550|6877365403758500816|6878522337167673632|6891997545626666836|6904884736631446335|6911856159394953698|6941422714372371066|6964207797463982365|7039524079399891585|7074320802377616844|7124640353542841782|7149223201624721221|7216705686020523931|7266905897612298267|7326454106052765602|7364423568524019001|7403758856280174235|7510599540561426487|7657742845359682088|7692321969367827909|7704057018961184501|7856882379387174966|7870972097565009656|7884450537652143749|7892445567047991929|7909074847595563265|7911333272109193134|7914023695364369566|7935360542419341608|8034103428368928133|8168658414760534686|8208302624540242806|8260291877063340832|8268233053310768933|8355955036201461959|8384639211524764562|8478022142378609327|8484057335692109658|8677774231560641320|8705474295244212422|8715881242396764183|8790493674280757493|8850928433694682560|8978373355407839573|9096876850144342277|9098964975663866916|9121023066083530190|9125075555834744453|9146563955039203827|9182944273403196832|9203993174082142511","|") OR multi_in(crowd_list,"80208440959169149|126727916179217579|441857599918902050|622991980312462648|756546242383248048|806204695380585356|956286486692043235|986094948463067184|1015528958980853798|1153495089651018899|1189002445813326459|1359886823497760625|1390705226385319070|1506645033487869865|1631095193756118967|1695023931373027609|2449456684759890666|2830072783844700095|3017360041020387862|3262922817893297850|3358687588183023805|3536671334867586408|3556798903986397538|3621425958951569761|3691308072611513518|3813730419403189994|3844702060958406176|4091963686481662692|4354537438348046649|4456754188706088715|4608318686859986614|4745615494174571954|5114400170638063444|5190921344255446184|5337347457187223751|5378852286003043276|5791007184231547341|5833002873238925542|6043007466971721749|6209413849946061286|6297389366114400552|6345189558923803681|6421721170272640232|6716756335254352211|6745750077515288777|6764168334848623189|6887033286259509012|6889828565947080321|6986221294546637672|7242223708465403105|7653137160595604444|7762241365714297971|7775358271801602369|7852531180115200668|7860310059282692815|8039708017878496336|8180991401898235191|8277770181279897108|8292957569252983474|8318351211062628642|8435337892971780354|8596630131717633561|8707281399705271958|8888219301044562890","|") OR multi_in(key_word_list,"70921140622523212|136384988538140592|184143082490798526|251375347279794572|281374428738512042|326061457882163993|337895484754222953|388044005388973349|422077298938200330|469092643772690877|490614110476736251|504340491042526348|548940426150626302|566373997463084356|580034307737093828|589506784829957937|616640925216753511|653915655929729934|690152913252205636|692706723841523214|722425436411586928|723206330743900998|746845948137736117|752222126078285269|769773115444016286|804439358399174783|822392927278077357|828991179941849128|858377164915266383|889796985460797886|907474383829135682|918212275597629601|944547949312425701|955944705890511925|1036583895583310702|1038907428832581983|1042789972168179739|1046351912763486754|1081069444851925801|1106697011519269967|1159963923934715308|1183468731683282679|1200679132320815754|1207793623217989685|1275453963910098330|1350458927511047587|1357275696184322844|1363308459102220516|1396806276909906851|1417489306944128857|1462330420764409505|1491244293750023261|1491788469036897032|1507865948020056917|1585424888000263567|1615012914981921397|1681522196324756017|1754549248237904866|1776955029708284723|1839715113560659435|1904174833182270034|1907709143146888170|1998925039065299991|2022307828670329215|2027196033505164158|2059317280453769933|2094979672798198790|2128491521053130086|2131059015395737730|2167099334795167773|2197056741795432015|2292379120484214301|2293013839619080381|2308110701959747013|2320793020197454706|2328302494858748564|2423840766859593325|2471507637648756482|2508956589464556712|2517944315678000954|2556275686516788047|2601512061763499197|2608357614863026776|2620498091533377966|2681216726307835284|2696178747848410550|2742844547448952035|2745070386415805082|2763989062253543555|2768472716719145252|2780517473942943049|2807633691460106298|2824702113791817412|2897563933500957293|2977960903994804086|3045669713018968439|3052962952519712630|3061223384848405277|3065631628843163272|3132844383435949631|3189253102506145277|3212896658701811699|3258400947163188113|3263938269786938163|3274528603550352149|3293046621325228878|3313713264898944967|3317651021470668533|3331333900549803074|3334572970070883508|3391314203203109225|3395192740753227181|3423995984132505150|3521330882077817033|3547441034692334861|3626697873284087308|3721574776402515438|3757376334275374856|3810676249655232317|3844950250101665397|3861791390700816479|3897488253969845227|3968881520605279123|3981796199360371253|4043480455514645193|4070730178275197586|4102533763604629314|4105597400598932732|4219488780712267708|4229555992181340958|4243869198704743011|4311074224567414705|4311239284674938420|4339792073401886821|4366770915872664561|4367098238390618245|4480849302217266153|4481483416171330042|4499230683118253280|4613370642408189894|4629605997419753449|4646223564703706327|4678713620308968200|4679037784689660816|4681073049542170662|4713870933970569191|4726814676205360839|4737060303269736009|4777091105912693962|4803207236512759948|4928010330676060525|4932226047615685303|4933746328324347021|5024074225573041729|5039980347146076117|5042548150689562617|5117125185281598414|5178564452853646280|5181375057325523760|5201161625647882507|5214049681796504143|5249301984820315792|5288439718886225170|5300741760231754791|5302887173817376445|5345565602500335573|5378680439996900138|5383048550177303047|5420268007002875441|5442563000752825510|5444644410899873754|5464847665629596964|5470614848004855012|5499852928686379460|5500420668641734411|5527229148603045182|5547792960341656680|5561846519749055920|5564280915297305988|5586635600158756936|5677640573817119605|5699711072318378101|5755160282060448687|5778143111154275434|5809915131992100538|5812862138454206001|5849881572151691861|5864888729401877983|5918835238499220717|5969789630214154173|5971893694728522519|5984506969969361137|5996981933003393054|6017504745676398500|6030076463537117157|6073235832493842164|6082292083910296654|6121492566999879743|6197900137805551850|6262474345212810521|6327882296427516915|6404183025067025304|6416301386321636541|6457638592172679989|6472036611211048924|6480767811656200339|6546845904998393277|6571408675424605873|6587663624709154672|6686446441090484638|6691123367600903858|6738272387861420355|6785534606252369879|6800874659017179661|6855074331643834264|6864556157650470910|6890571671673361348|6920401950254941185|6947551711822494936|6952159892975752309|6966915393958399029|7070716925661700845|7071934731710646551|7105173923679094008|7129059602791883302|7219227354834563529|7232438694272419127|7307841830029007914|7316236751875305287|7342426491553658644|7350798533017733221|7360886442298245140|7367735060545943498|7380284547253407711|7388161714077758830|7407606000767214121|7422688296185265188|7497784435145663433|7553915847503191623|7559506639002631617|7671832981906098849|7733901415388141213|7745680787427752895|7753235693231808835|7758787682582242337|7776016390118652147|7778296386534924014|7805969065251784102|7812845159640975206|7816511824571712771|7820899484441711284|7825445052227195925|7832343616265394191|7852579384154337704|7858055210911592968|7875474084373016071|7953215667697979874|7970785345450898729|7981328112248859714|7988560505742599509|8009249437967493859|8043582833493457872|8107833138991521966|8151124216390902163|8217628752261404594|8320483608787417954|8324705151844631124|8375942310792287260|8384900118323450115|8385820984843889151|8418647031493182407|8442544468831313307|8499572069225694360|8547629238098822553|8594304443769791644|8594556265610563084|8636411164522413426|8659884145858377766|8661907488737101385|8691628972267343648|8732497180270615495|8783956603924944913|8822776851984079787|8855188398968475454|8882470940071159683|8898169977791237118|8932473720192130254|8946083614718239930|8958453408760589130|9002357508031696580|9003793281593179598|9019021402698807676|9052932717483094676|9083215064564194916|9107569824073182188|9111673566299158513|9175598358367427055|9178022243021643451|9184965579833663300|9206043479924474618","|"))))))";
        auto expr_ptr = AstParse(filter_condition_str);
        REQUIRE(expr_ptr != nullptr);
    }
}
