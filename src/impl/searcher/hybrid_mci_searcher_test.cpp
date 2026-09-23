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

#include "hybrid_mci_searcher.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "searcher_test.h"

using namespace vsag;

namespace {

constexpr uint64_t kDim = 2;

/**
 * Predicate fixture. The traversal only ever calls CheckValid(inner_id), so a
 * small id list is enough to model a selective pre-filter.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
class SelectiveFilter : public Filter {
public:
    SelectiveFilter(std::vector<int64_t> valid_ids, float valid_ratio)
        : valid_ids_(std::move(valid_ids)), valid_ratio_(valid_ratio) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return std::find(valid_ids_.begin(), valid_ids_.end(), id) != valid_ids_.end();
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    std::vector<int64_t> valid_ids_;
    float valid_ratio_;
};

// NOLINTNEXTLINE(readability-identifier-naming)
struct HybridFixture {
    IndexCommonParam common;
    FlattenInterfacePtr flatten;
    GraphInterfacePtr graph;
    CliqueDataCellPtr cliques;
    std::vector<InnerIdType> ids{0, 1, 2, 3, 4, 5};
};

/**
 * Six 2-D vectors laid out on a line at x = 0.1, 1, 2, 3, 4, 5, so the L2
 * distances to the origin query are 0.01, 1, 4, 9, 16, 25.
 *
 * The sparse graph only connects 0 - 1 - 2, i.e. the satisfactory vectors 3, 4
 * and 5 are reachable exclusively through the clique companion. That split lets
 * a test tell the Hgraph part and the MCI part apart.
 *
 *   cliques: c0 = {0, 3, 4}, c1 = {1, 5}
 */
HybridFixture
MakeFixture() {
    HybridFixture fixture;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixture.common.allocator_ = allocator;
    fixture.common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    fixture.common.dim_ = kDim;

    std::vector<float> vectors = {
        0.1F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 4.0F, 0.0F, 5.0F, 0.0F};

    constexpr const char* param_template = R"({{"type": "{}"}})";
    auto quantizer_param = QuantizerParameter::GetQuantizerParameterByJson(
        JsonType::Parse(fmt::format(param_template, "fp32")));
    auto io_param = IOParameter::GetIOParameterByJson(
        JsonType::Parse(fmt::format(param_template, "memory_io")));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryIO>>>(
        quantizer_param, io_param, fixture.common);
    flatten->SetQuantizer(
        std::make_shared<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>>(kDim, allocator.get()));
    flatten->SetIO(std::make_unique<MemoryIO>(allocator.get()));
    flatten->Train(vectors.data(), fixture.ids.size());
    flatten->BatchInsertVector(vectors.data(), fixture.ids.size(), fixture.ids.data());
    fixture.flatten = flatten;

    fixture.graph = std::make_shared<MockGraphDataCell>(
        std::vector<std::vector<InnerIdType>>{{1}, {0, 2}, {1}, {}, {}, {}});

    auto cliques = std::make_shared<CliqueDataCell>(allocator.get());
    Vector<InnerIdType> p_maxc(allocator.get());
    Vector<InnerIdType> maxcs(allocator.get());
    Vector<InnerIdType> p_node_to_cid(allocator.get());
    Vector<InnerIdType> node_to_cids(allocator.get());
    p_maxc.insert(p_maxc.end(), {0, 3, 5});
    maxcs.insert(maxcs.end(), {0, 3, 4, 1, 5});
    p_node_to_cid.insert(p_node_to_cid.end(), {0, 1, 2, 2, 3, 4, 5});
    node_to_cids.insert(node_to_cids.end(), {0, 1, 0, 0, 1});
    cliques->Assign(std::move(p_maxc),
                    std::move(maxcs),
                    std::move(p_node_to_cid),
                    std::move(node_to_cids),
                    fixture.ids.size());
    fixture.cliques = cliques;
    return fixture;
}

std::vector<InnerIdType>
ResultIds(const DistHeapPtr& heap) {
    std::vector<InnerIdType> ids;
    ids.reserve(heap->Size());
    for (uint64_t i = 0; i < heap->Size(); ++i) {
        ids.push_back(heap->GetData()[i].second);
    }
    std::sort(ids.begin(), ids.end());
    return ids;
}

HybridTraversalParam
MakeParam(int64_t topk, uint64_t ef, double vob) {
    HybridTraversalParam param;
    param.topk = topk;
    param.ef = ef;
    param.entry_id = 0;
    param.vob = vob;
    param.filter_cost_ratio = 1.0;
    return param;
}

}  // namespace

TEST_CASE("HybridMCISearcher matches brute force when the budget is disabled",
          "[ut][HybridMCISearcher][correctness]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};
    auto filter = std::make_shared<SelectiveFilter>(std::vector<int64_t>{1, 3, 4, 5}, 0.66F);

    HybridSearchStats stats;
    // vob <= 0 disables traverse_should_stop(): every clique of v_0 is expanded.
    auto result = HybridMCISearcher(fixture.common)
                      .Search(fixture.graph,
                              fixture.cliques,
                              fixture.flatten,
                              query.data(),
                              filter,
                              MakeParam(/*topk=*/3, /*ef=*/10, /*vob=*/0.0),
                              nullptr,
                              &stats);

    // The three nearest satisfactory vectors are 1 (d=1), 3 (d=9) and 4 (d=16).
    // Vector 4 is only reachable through clique c0.
    REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 4});
    REQUIRE(stats.mci_stopped_early == false);
    REQUIRE(stats.mci_cliques_expanded == 2);
    // c0 contributes {3, 4} and c1 contributes {5}; the already-visited members
    // 0 and 1 are rejected by the visited table and never accounted.
    REQUIRE(stats.mci_members_considered == 3);
    REQUIRE(stats.mci_members_satisfied == 3);
}

TEST_CASE(
    "HybridMCISearcher stops the MCI traversal once the virtual overhead reaches the boundary",
    "[ut][HybridMCISearcher][budget]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};
    auto filter = std::make_shared<SelectiveFilter>(std::vector<int64_t>{1, 3, 4, 5}, 0.66F);

    // ratio = 1 and local selectivity 1.0 mean every considered
    // member costs 2.0, so vob = 2.0 stops right after the first one.
    HybridSearchStats stats;
    auto result = HybridMCISearcher(fixture.common)
                      .Search(fixture.graph,
                              fixture.cliques,
                              fixture.flatten,
                              query.data(),
                              filter,
                              MakeParam(/*topk=*/3, /*ef=*/10, /*vob=*/2.0),
                              nullptr,
                              &stats);

    REQUIRE(stats.mci_stopped_early == true);
    // Member 4 of clique c0 is never reached, so the result degrades to
    // {1, 3, 5} instead of the exact {1, 3, 4}.
    REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 5});
    REQUIRE(stats.mci_members_considered == 2);
    REQUIRE(stats.mci_members_considered < 3);
}

TEST_CASE("HybridMCISearcher degrades to the sparse graph without an MCI companion",
          "[ut][HybridMCISearcher][sparse-only]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};
    auto filter = std::make_shared<SelectiveFilter>(std::vector<int64_t>{1, 3, 4, 5}, 0.66F);

    HybridSearchStats stats;
    auto result = HybridMCISearcher(fixture.common)
                      .Search(fixture.graph,
                              /*cliques=*/nullptr,
                              fixture.flatten,
                              query.data(),
                              filter,
                              MakeParam(/*topk=*/1, /*ef=*/10, /*vob=*/2.0),
                              nullptr,
                              &stats);

    // Only 0, 1 and 2 are reachable through the sparse graph, and 1 is the only
    // satisfactory one among them.
    REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1});
    REQUIRE(stats.mci_cliques_expanded == 0);
    REQUIRE(stats.mci_stopped_early == false);
}

TEST_CASE("HybridMCISearcher treats a null filter as accept-all",
          "[ut][HybridMCISearcher][no-filter]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};

    auto result = HybridMCISearcher(fixture.common)
                      .Search(fixture.graph,
                              fixture.cliques,
                              fixture.flatten,
                              query.data(),
                              /*filter=*/nullptr,
                              MakeParam(/*topk=*/2, /*ef=*/10, /*vob=*/0.0));

    // The two closest vectors overall are the entry 0 (d=0.01) and 1 (d=1).
    REQUIRE(ResultIds(result) == std::vector<InnerIdType>{0, 1});
    // Top() is the farthest member of the result heap.
    REQUIRE(result->Top().second == 1);
    REQUIRE(std::fabs(result->Top().first - 1.0F) < 1e-5F);
}

TEST_CASE("HybridMCISearcher honours the hops limit", "[ut][HybridMCISearcher][hops]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};
    auto filter = std::make_shared<SelectiveFilter>(std::vector<int64_t>{1, 3, 4, 5}, 0.66F);
    auto param = MakeParam(/*topk=*/3, /*ef=*/10, /*vob=*/0.0);

    // hops_limit = 0 seeds the entry vector but expands nothing. The entry 0 is
    // not satisfactory, so no result is produced at all.
    param.hops_limit = 0;
    {
        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        REQUIRE(stats.expanded_nodes == 0);
        REQUIRE(result->Empty());
    }

    // hops_limit = 1 expands the entry only, and a single expansion already
    // pulls in its whole clique c0 = {0, 3, 4} plus the sparse neighbour 1.
    param.hops_limit = 1;
    {
        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        REQUIRE(stats.expanded_nodes == 1);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 4});
    }
}

TEST_CASE("HybridMCISearcher seeds the heaps from an injected valid set",
          "[ut][HybridMCISearcher][seeds]") {
    auto fixture = MakeFixture();
    const auto query = std::vector<float>{0.0F, 0.0F};
    auto filter = std::make_shared<SelectiveFilter>(std::vector<int64_t>{1, 3, 4, 5}, 0.66F);
    // hops_limit = 0 removes every expansion, so whatever shows up in the result was
    // injected as a seed and nothing else.
    auto param = MakeParam(/*topk=*/3, /*ef=*/10, /*vob=*/0.0);
    param.hops_limit = 0;

    SECTION("no seed list falls back to the entry point") {
        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        // The entry 0 is not satisfactory and nothing is expanded.
        REQUIRE(result->Empty());
        REQUIRE(stats.seeded_entries == 0);
    }

    SECTION("a seed list seeds every listed vector") {
        Vector<InnerIdType> seeds(fixture.common.allocator_.get());
        seeds.insert(seeds.end(), {1, 3, 4});
        param.seed_inner_ids = &seeds;

        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        REQUIRE(stats.seeded_entries == 3);
        REQUIRE(stats.expanded_nodes == 0);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 4});
    }

    SECTION("seed_count samples the list at even strides") {
        Vector<InnerIdType> seeds(fixture.common.allocator_.get());
        seeds.insert(seeds.end(), {0, 1, 3, 4, 5});
        param.seed_inner_ids = &seeds;
        param.seed_count = 3;

        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        // Offsets 0 * 5 / 3 = 0, 1 * 5 / 3 = 1 and 2 * 5 / 3 = 3 select {0, 1, 4};
        // only the satisfactory ones reach the result heap.
        REQUIRE(stats.seeded_entries == 3);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 4});
    }

    SECTION("duplicate seeds only seed once") {
        Vector<InnerIdType> seeds(fixture.common.allocator_.get());
        seeds.insert(seeds.end(), {1, 1, 1});
        param.seed_inner_ids = &seeds;

        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        REQUIRE(stats.seeded_entries == 1);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1});
    }

    SECTION("exhaustive seeds answer without expanding") {
        Vector<InnerIdType> seeds(fixture.common.allocator_.get());
        seeds.insert(seeds.end(), {1, 3, 4});
        param.seed_inner_ids = &seeds;
        param.seeds_are_exhaustive = true;
        // Lift the hop cap: the skip must come from the flag, not from the cap.
        param.hops_limit = std::numeric_limits<uint32_t>::max();

        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        REQUIRE(stats.expansion_skipped);
        REQUIRE(stats.expanded_nodes == 0);
        REQUIRE(stats.mci_cliques_expanded == 0);
        REQUIRE(stats.seeded_entries == 3);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 4});
    }

    SECTION("the exhaustive flag without a seed list still expands") {
        param.seeds_are_exhaustive = true;
        param.hops_limit = std::numeric_limits<uint32_t>::max();

        HybridSearchStats stats;
        auto result = HybridMCISearcher(fixture.common)
                          .Search(fixture.graph,
                                  fixture.cliques,
                                  fixture.flatten,
                                  query.data(),
                                  filter,
                                  param,
                                  nullptr,
                                  &stats);
        // Nothing was seeded, so the flag on its own must not short-circuit the traversal.
        REQUIRE_FALSE(stats.expansion_skipped);
        REQUIRE(stats.expanded_nodes > 0);
        REQUIRE(ResultIds(result) == std::vector<InnerIdType>{1, 3, 4});
    }
}
