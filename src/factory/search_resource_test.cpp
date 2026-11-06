// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "vsag/search_resource.h"

#include "algorithm/sindi/sindi.h"
#include "impl/searcher/search_execution_context.h"
#include "index/index_impl.h"
#include "unittest.h"
#include "vsag/engine.h"
using namespace vsag;

TEST_CASE("SearchResource composes common resources and shares query admission",
          "[ut][SearchResource]") {
    auto threads = Engine::CreateThreadPool(2);
    REQUIRE(threads.has_value());
    Resource common(Engine::CreateDefaultAllocator(), threads.value());
    SearchResourceOptions options;
    options.max_background_workers = 0;
    SearchResource root(common, options);
    SearchResource overlay(common, root);
    Engine first(&root), second(&overlay), plain(&common);
    const auto json =
        R"({"dtype":"sparse","metric_type":"ip","dim":4,"index_param":{"immutable":true,"term_id_limit":4,"window_size":10000}})";
    auto a = first.CreateIndex("sindi", json);
    auto b = second.CreateIndex("sindi", json);
    auto c = plain.CreateIndex("sindi", json);
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    REQUIRE(c.has_value());
    const auto aa = std::dynamic_pointer_cast<IndexImpl<SINDI>>(a.value());
    const auto bb = std::dynamic_pointer_cast<IndexImpl<SINDI>>(b.value());
    const auto cc = std::dynamic_pointer_cast<IndexImpl<SINDI>>(c.value());
    REQUIRE(aa);
    REQUIRE(bb);
    REQUIRE(cc);
    const auto context = aa->GetCommonParam().search_resources_;
    REQUIRE(context);
    REQUIRE(context == bb->GetCommonParam().search_resources_);
    REQUIRE_FALSE(cc->GetCommonParam().search_resources_);
    REQUIRE(aa->GetCommonParam().thread_pool_);
    REQUIRE(context->limit == 0);
    REQUIRE_FALSE(context->pool);
    SearchResourceOptions invalid;
    invalid.max_background_workers = 513;
    REQUIRE_THROWS(SearchResource(common, invalid));
}

TEST_CASE("Index retains query context after the resource wrapper is released",
          "[ut][SearchResource]") {
    std::shared_ptr<Index> retained;
    std::weak_ptr<SearchExecutionContext> context;
    {
        Resource common(Engine::CreateDefaultAllocator(), nullptr);
        SearchResourceOptions options;
        options.max_background_workers = 0;
        SearchResource resource(common, options);
        Engine engine(&resource);
        auto created = engine.CreateIndex(
            "sindi",
            R"({"dtype":"sparse","metric_type":"ip","dim":4,"index_param":{"immutable":true,"term_id_limit":4,"window_size":10000}})");
        REQUIRE(created.has_value());
        retained = created.value();
        auto typed = std::dynamic_pointer_cast<IndexImpl<SINDI>>(retained);
        REQUIRE(typed);
        context = typed->GetCommonParam().search_resources_;
        engine.Shutdown();
        REQUIRE_FALSE(context.expired());
    }
    REQUIRE_FALSE(context.expired());
    retained.reset();
    REQUIRE(context.expired());
}
