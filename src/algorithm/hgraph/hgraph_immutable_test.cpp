// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "unittest.h"
#include "vsag/factory.h"

TEST_CASE("Immutable HGraph keeps searches and rejects mutations", "[ut][hgraph][immutable]") {
    auto created = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32","metric_type":"l2","dim":2,
        "index_param":{"base_quantization_type":"fp32","max_degree":8,"ef_construction":16}
    })");
    REQUIRE(created.has_value());
    auto index = created.value();
    int64_t ids[] = {0, 1};
    float vectors[] = {1, 0, 0, 1};
    auto base =
        vsag::Dataset::Make()->NumElements(2)->Dim(2)->Ids(ids)->Float32Vectors(vectors)->Owner(
            false);
    REQUIRE(index->Build(base).has_value());
    REQUIRE(index->UpdateId(0, 2).has_value());
    REQUIRE(index->SetImmutable().has_value());
    REQUIRE(index->SetImmutable().has_value());
    CHECK_FALSE(index->Add(base).has_value());
    CHECK_FALSE(index->UpdateId(2, 3).has_value());
    CHECK_FALSE(index->UpdateExtraInfo(base).has_value());
    auto query =
        vsag::Dataset::Make()->NumElements(1)->Dim(2)->Float32Vectors(vectors)->Owner(false);
    CHECK_FALSE(index->UpdateVector(2, query).has_value());
    auto result = index->KnnSearch(query, 1, R"({"hgraph":{"ef_search":16}})");
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 1);
    CHECK(result.value()->GetIds()[0] == 2);
}
