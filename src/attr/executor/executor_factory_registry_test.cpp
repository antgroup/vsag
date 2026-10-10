// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "executor_factory_registry.h"

#include <future>

#include "attr/argparse.h"
#include "impl/allocator/safe_allocator.h"
#include "region_filter_executor.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("Immutable region factory errors", "[ut][RegionFactory]") {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    auto index = AttributeInvertedInterface::MakeInstance(alloc.get(), true);
    // Existing parser accepts NOT, but executor dispatch does not implement it.
    auto negated = AstParse(R"(!(region_filter(kind,geo,home,"1","1","2")))");
    REQUIRE(negated->GetExprType() == ExpressionType::kNotExpression);
    try {
        Executor::MakeInstance(alloc.get(), negated, index);
        FAIL("top-level NOT remains unsupported");
    } catch (const VsagException& error) {
        REQUIRE(error.error_.type == ErrorType::INTERNAL_ERROR);
        REQUIRE(std::string(error.what()) == "Unsupported expression type");
    }
    REQUIRE_THROWS_AS(Executor::MakeInstance(alloc.get(), nullptr, index), VsagException);
    const auto& registry = ExecutorFactoryRegistry::Builtins();
    REQUIRE_THROWS_AS(registry.Create("unknown", alloc.get(), nullptr, index), VsagException);
    REQUIRE_THROWS_AS(registry.Create("region_filter", alloc.get(), nullptr, index), VsagException);
    REQUIRE_THROWS_AS(registry.Create("region_filter", alloc.get(), AstParse("a = 1"), index),
                      VsagException);
    auto factory = +[](Allocator* allocator,
                       const ExprPtr& expr,
                       const AttrInvertedInterfacePtr& attrs) -> ExecutorPtr {
        return std::make_shared<RegionFilterExecutor>(allocator, expr, attrs);
    };
    REQUIRE_THROWS_AS((ExecutorFactoryRegistry{{"x", factory}, {"x", factory}}), VsagException);
    REQUIRE_THROWS_AS((ExecutorFactoryRegistry{{"", factory}}), VsagException);
    REQUIRE_THROWS_AS((ExecutorFactoryRegistry{{"x", nullptr}}), VsagException);
    const ExecutorFactoryRegistry empty{};
    REQUIRE_THROWS_AS(empty.Create("region_filter", alloc.get(), nullptr, index), VsagException);
}

TEST_CASE("Region factory combinations and independent lifecycle", "[ut][RegionFactory]") {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    auto index = AttributeInvertedInterface::MakeInstance(alloc.get(), true);
    for (int64_t id = 0; id < 64; ++id) {
        AttributeValue<int16_t> kind;
        AttributeValue<int64_t> geo;
        AttributeValue<int64_t> home;
        kind.name_ = "kind";
        kind.GetValue() = {2};
        geo.name_ = "geo";
        geo.GetValue() = {id % 2};
        home.name_ = "home";
        home.GetValue() = {id % 3};
        index->Insert(AttributeSet{{&kind, &geo, &home}}, id, 0);
    }
    using Oracle = bool (*)(int64_t);
    const std::pair<const char*, Oracle> cases[] = {
        {R"(region_filter(kind,geo,home,"1","1","2"))", [](int64_t id) { return id % 2 == 1; }},
        {R"(region_filter(kind,geo,home,"1","1","2") AND home = 1)",
         [](int64_t id) { return id % 2 == 1 && id % 3 == 1; }},
        {R"(region_filter(kind,geo,home,"1","1","2") OR home = 1)",
         [](int64_t id) { return id % 2 == 1 || id % 3 == 1; }},
        {R"(home = 1 AND region_filter(kind,geo,home,"1","1","2"))",
         [](int64_t id) { return id % 2 == 1 && id % 3 == 1; }},
        {R"(home = 1 OR region_filter(kind,geo,home,"1","1","2"))",
         [](int64_t id) { return id % 2 == 1 || id % 3 == 1; }},
        {R"(region_filter(kind,geo,home,"1","1","2") AND region_filter(kind,geo,home,"0","1","2"))",
         [](int64_t) { return false; }}};
    for (const auto& item : cases) {
        auto expr = AstParse(item.first, &index->field_type_map_);
        auto registered = Executor::MakeInstance(alloc.get(), expr, index);
        registered->Init();
        registered->Init();
        for (int repeat = 0; repeat < 4; ++repeat) {
            registered->Clear();
            const auto* result = registered->Run(0);
            for (int64_t id = 0; id < 64; ++id) {
                REQUIRE(result->CheckValid(id) == item.second(id));
            }
            registered->Clear();
            REQUIRE_FALSE(registered->Run(9)->CheckValid(int64_t{0}));
            registered->Clear();
            registered->Clear();
        }
    }
    auto expr = AstParse(R"(region_filter(kind,geo,home,"1","1","2"))");
    // Instrument a factory only in this correctness test, never in timed production code.
    static int factory_calls = 0;
    factory_calls = 0;
    const ExecutorFactoryRegistry counted{
        {"region_filter",
         +[](Allocator* allocator, const ExprPtr& expression, const AttrInvertedInterfacePtr& attrs)
             -> ExecutorPtr {
             ++factory_calls;
             return std::make_shared<RegionFilterExecutor>(allocator, expression, attrs);
         }}};
    auto counted_executor = counted.Create("region_filter", alloc.get(), expr, index);
    counted_executor->Init();
    for (int i = 0; i < 100; ++i) {
        counted_executor->Run(i % 2);
    }
    REQUIRE(factory_calls == 1);
    counted_executor.reset();
    auto survivor = Executor::MakeInstance(alloc.get(), expr, index);
    survivor->Init();
    auto other = Executor::MakeInstance(alloc.get(), expr, index);
    other->Init();
    survivor->Run(0);
    other->Run(9);
    REQUIRE(survivor->filter_->CheckValid(int64_t{1}));
    REQUIRE(survivor->bitset_ != other->bitset_);
    other.reset();
    expr.reset();
    index.reset();
    REQUIRE(survivor->Run(0)->CheckValid(int64_t{1}));
    auto work = [survivor] {
        auto executor =
            Executor::MakeInstance(survivor->allocator_, survivor->expr_, survivor->attr_index_);
        executor->Init();
        for (int i = 0; i < 100; ++i) {
            if (!executor->Run(0)->CheckValid(int64_t{1})) {
                return false;
            }
        }
        return true;
    };
    auto a = std::async(std::launch::async, work);
    auto b = std::async(std::launch::async, work);
    REQUIRE(a.get());
    REQUIRE(b.get());
}
