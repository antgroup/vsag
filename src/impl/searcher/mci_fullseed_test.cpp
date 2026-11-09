// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <numeric>
#include <ranges>

#include "datacell/clique_datacell.h"
#include "datacell/flatten_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/label_table/label_table.h"
#include "impl/searcher/mci_searcher.h"
#include "index_common_param.h"
#include "io/memory_io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "query_context.h"

using namespace vsag;

namespace {
struct FullSeedFixture {
    static constexpr uint64_t COUNT = 130;
    std::shared_ptr<Allocator> allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    std::vector<float> vectors = std::vector<float>(COUNT);
    Vector<InnerIdType> seeds{allocator.get()};
    std::vector<uint8_t> bitmap = std::vector<uint8_t>(COUNT, 7);
    FlattenInterfacePtr flatten;
    CliqueDataCellPtr cliques;

    FullSeedFixture() {
        common.dim_ = 1;
        common.allocator_ = allocator;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        std::iota(vectors.begin(), vectors.end(), 0.F);
        for (InnerIdType id = 0; id < COUNT; ++id) {
            seeds.push_back(id);
        }
        auto quantizer =
            QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
        auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
        flatten = std::make_shared<
            FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
            quantizer, io, common);
        flatten->Train(vectors.data(), COUNT);
        flatten->BatchInsertVector(vectors.data(), COUNT, seeds.data());
        cliques = std::make_shared<CliqueDataCell>(allocator.get());
        Vector<InnerIdType> offsets({0, COUNT}, allocator.get());
        Vector<InnerIdType> members(seeds, allocator.get());
        Vector<InnerIdType> node_offsets(allocator.get());
        Vector<InnerIdType> node_cliques(COUNT, 0, allocator.get());
        for (uint32_t i = 0; i <= COUNT; ++i) {
            node_offsets.push_back(i);
        }
        cliques->Assign(std::move(offsets),
                        std::move(members),
                        std::move(node_offsets),
                        std::move(node_cliques),
                        COUNT);
    }
};

struct CountingPredicate : Filter {
    mutable uint64_t calls{0};
    int64_t rejected{-1};

    bool
    CheckValid(int64_t id) const override {
        ++calls;
        return id != rejected;
    }
};
}  // namespace

TEST_CASE("MCI complete seeds share scoring and validate locked predicates once",
          "[ut][mci][MCIFullSeed]") {
    FullSeedFixture fixture;
    constexpr auto count = FullSeedFixture::COUNT;
    const bool raw = GENERATE(false, true);
    const bool batch = GENERATE(false, true);
    const bool bitmap = GENERATE(false, true);
    const bool revalidate = GENERATE(false, true);
    MCISearcherParam params;
    params.seed_inner_ids = &fixture.seeds;
    params.seed_count = count;
    params.enumerated_valid_count = count;
    params.batch_distance = batch;
    params.revalidate_predicate = revalidate;
    if (bitmap) {
        params.valid_bitmap = fixture.bitmap.data();
        params.valid_bitmap_size = count;
    }
    if (raw) {
        params.precise_vectors = fixture.vectors.data();
        params.precise_vector_stride = 1;
        params.dim = 1;
        params.metric = MetricType::METRIC_TYPE_L2SQR;
    }
    auto predicate = std::make_shared<CountingPredicate>();
    InnerSearchParam search;
    search.ef = search.topk = 64;
    search.rerank_topk = 10;
    search.is_inner_id_allowed = predicate;
    const float query = 111.25F;
    SearchStatistics stats;
    QueryContext ctx{.alloc = fixture.allocator.get(), .stats = &stats};
    auto result = MCISearcher(fixture.common)
                      .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == 10);
    REQUIRE(stats.dist_cmp.load() == count);
    REQUIRE(stats.hops.load() == 0);
    REQUIRE(stats.distance_evaluations.load() == count);
    REQUIRE(predicate->calls == (revalidate or not bitmap ? count : 0));
    std::vector<std::pair<float, InnerIdType>> truth;
    for (InnerIdType i = 0; i < count; ++i) {
        const auto delta = fixture.vectors[i] - query;
        truth.emplace_back(delta * delta, i);
    }
    std::sort(truth.begin(), truth.end());
    truth.resize(10);
    for (const auto& expected : std::views::reverse(truth)) {
        REQUIRE(result->Top() == expected);
        result->Pop();
    }

    // Unspecified hints and duplicate-aware callers retain the larger candidate budget.
    search.rerank_topk = 0;
    result = MCISearcher(fixture.common)
                 .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == 64);
    search.rerank_topk = 10;
    search.consider_duplicate = true;
    result = MCISearcher(fixture.common)
                 .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == 64);
}

TEST_CASE("MCI stale complete seeds preserve fallback budget and timeout semantics",
          "[ut][mci][MCIFullSeed]") {
    FullSeedFixture fixture;
    constexpr auto count = FullSeedFixture::COUNT;
    const bool raw = GENERATE(false, true);
    const int invalid = GENERATE(0, 1, 2, 3);
    MCISearcherParam params;
    params.seed_inner_ids = &fixture.seeds;
    params.seed_count = count;
    params.enumerated_valid_count = count;
    params.valid_bitmap = fixture.bitmap.data();
    params.valid_bitmap_size = count;
    params.revalidate_predicate = true;
    if (raw) {
        params.precise_vectors = fixture.vectors.data();
        params.precise_vector_stride = 1;
        params.dim = 1;
        params.metric = MetricType::METRIC_TYPE_L2SQR;
    }
    auto predicate = std::make_shared<CountingPredicate>();
    InnerSearchParam search;
    search.ef = search.topk = 64;
    search.rerank_topk = 10;
    search.is_inner_id_allowed = predicate;
    LabelTable labels(fixture.allocator.get());
    for (InnerIdType i = 0; i < count; ++i) {
        labels.Insert(i, i);
    }
    if (invalid == 0) {
        predicate->rejected = 1;
    } else if (invalid == 1) {
        REQUIRE(labels.MarkRemove({1}) == 1);
        params.deleted_ids_filter = labels.GetDeletedIdsReadView();
    } else if (invalid == 2) {
        fixture.seeds.back() = fixture.seeds[count - 2];
    } else {
        fixture.seeds.back() = count + 1;
    }
    SearchStatistics stats;
    QueryContext ctx{.alloc = fixture.allocator.get(), .stats = &stats};
    const float query = 111.25F;
    auto result = MCISearcher(fixture.common)
                      .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == 64);
    std::vector<std::pair<float, InnerIdType>> actual;
    while (not result->Empty()) {
        actual.push_back(result->Top());
        REQUIRE((invalid > 1 or result->Top().second != 1));
        result->Pop();
    }
    params.enumerated_valid_count = 0;
    SearchStatistics reference_stats;
    ctx.stats = &reference_stats;
    result = MCISearcher(fixture.common)
                 .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(result->Size() == actual.size());
    REQUIRE(stats.dist_cmp.load() == reference_stats.dist_cmp.load());
    REQUIRE(stats.hops.load() == reference_stats.hops.load());
    for (const auto& expected : actual) {
        REQUIRE(result->Top() == expected);
        result->Pop();
    }

    params.enumerated_valid_count = count;
    search.time_cost = std::make_shared<Timer>();
    search.time_cost->SetThreshold(0);
    SearchStatistics timeout_stats;
    ctx.stats = &timeout_stats;
    result = MCISearcher(fixture.common)
                 .Search(fixture.cliques, fixture.flatten, &query, search, params, &ctx);
    REQUIRE(timeout_stats.is_timeout.load());
}
