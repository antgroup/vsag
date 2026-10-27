// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <numeric>

#include "datacell/clique_datacell.h"
#include "datacell/flatten_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/mci_depth_stats.h"
#include "impl/searcher/mci_searcher.h"
#include "io/memory_block_io/memory_block_io.h"
#include "quantization/fp32_quantizer.h"
#include "utils/search_lock.h"

namespace vsag {
namespace {

class HubTestGraph : public GraphInterface {
public:
    std::vector<std::vector<InnerIdType>> rows;
    mutable uint64_t reads{0};

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& ids) override {
        rows.at(id).assign(ids.begin(), ids.end());
    }
    uint32_t
    GetNeighborSize(InnerIdType id) const override {
        return rows.at(id).size();
    }
    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& ids) const override {
        ++reads;
        ids.assign(rows.at(id).begin(), rows.at(id).end());
    }
    bool
    CheckIdExists(InnerIdType id) const override {
        return id < rows.size();
    }
    void
    Resize(InnerIdType n) override {
        rows.resize(n);
    }
    void
    Prefetch(InnerIdType, uint32_t) override {
    }
};

class HubTestFilter : public Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id >= 0 and id < 8 and id != 1;
    }
};

}  // namespace

TEST_CASE("Hub HGraph replaces outgoing activation without deleting result points",
          "[ut][mci_hub_hgraph]") {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 1;
    common.allocator_ = alloc;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    const std::vector<float> vectors{3, 2, 1, .5F, 4, 5, 6, 7};
    const std::vector<InnerIdType> valid{0, 2, 3, 4, 5, 6, 7};
    const std::vector<uint8_t> bitmap{1, 0, 1, 1, 1, 1, 1, 1};
    auto quantizer =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"block_memory_io"})"));
    auto flatten = std::make_shared<
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, FixedLayout<MemoryBlockIO>>>(
        quantizer, io, common);
    Vector<InnerIdType> ids(8, alloc.get());
    std::iota(ids.begin(), ids.end(), 0);
    flatten->Train(vectors.data(), 8);
    flatten->BatchInsertVector(vectors.data(), 8, ids.data());
    auto cliques = std::make_shared<CliqueDataCell>(alloc.get());
    Vector<InnerIdType> offsets({0, 2, 4, 6, 8, 10}, alloc.get());
    Vector<InnerIdType> members({0, 4, 0, 5, 0, 6, 0, 7, 2, 3}, alloc.get());
    Vector<InnerIdType> inverse_offsets({0, 4, 4, 5, 6, 7, 8, 9, 10}, alloc.get());
    Vector<InnerIdType> inverse({0, 1, 2, 3, 4, 4, 0, 1, 2, 3}, alloc.get());
    cliques->Assign(
        std::move(offsets), std::move(members), std::move(inverse_offsets), std::move(inverse), 8);
    auto graph = std::make_shared<HubTestGraph>();
    graph->Resize(8);
    graph->rows[0] = {1, 4, 1};
    graph->rows[1] = {2, 0, 2};
    graph->rows[2] = {3, 1};
    Vector<InnerIdType> seeds(alloc.get());
    seeds.push_back(0);
    const float query = 0;
    SearchLockScope no_lock(true);
    for (auto passes : {0U, 1U, 2U}) {
        for (bool raw : {false, true}) {
            // bitmap, callback, and the complete-ID-list fallback must agree.
            for (uint64_t filter_kind = 0; filter_kind < 3; ++filter_kind) {
                for (uint64_t threshold : {0, 1, 4, 5}) {
                    for (auto hops : {1U, 2U}) {
                        MCISearcherParam p;
                        p.dynamic_clique_scan = true;
                        p.clique_scan_passes = passes;
                        p.seed_count = 1;
                        p.seed_inner_ids = &seeds;
                        p.valid_inner_ids = valid.data();
                        p.valid_inner_count = valid.size();
                        p.valid_bitmap = filter_kind == 0 ? bitmap.data() : nullptr;
                        p.valid_bitmap_size = bitmap.size();
                        p.precise_vectors = raw ? vectors.data() : nullptr;
                        p.dim = p.precise_vector_stride = 1;
                        p.hub_clique_threshold = threshold;
                        p.hub_hgraph_hops = hops;
                        p.hub_graph = graph;
                        InnerSearchParam search;
                        search.ef = search.topk = 1;
                        if (filter_kind == 1) {
                            search.is_inner_id_allowed = std::make_shared<HubTestFilter>();
                        }
                        graph->reads = 0;
                        auto result = MCISearcher(common).Search(
                            cliques, flatten, &query, search, p, nullptr);
                        REQUIRE(result->Size() == 1);
                        const bool enabled = threshold == 1 or threshold == 4;
                        const bool chain = threshold == 1 and hops == 2;
                        REQUIRE(result->Top().second == (enabled and hops == 2 ? 3 : 0));
                        const auto s = GetMCIDepthSearchStatistics();
                        if (enabled) {
                            REQUIRE(s.hub_candidates == (chain ? 3 : 1));
                            REQUIRE(s.hub_skipped_memberships == (chain ? 6 : 4));
                            REQUIRE(s.hub_expanded_nodes == (chain ? 3 : 1));
                            REQUIRE(s.hub_bridge_expansions == (hops == 2 ? 1 : 0));
                            REQUIRE(graph->reads == (chain ? 4 : (hops == 2 ? 2 : 1)));
                            REQUIRE(s.hub_distance_evaluations ==
                                    (chain ? 3 : (hops == 2 ? 2 : 1)));
                            REQUIRE(s.dynamic_scored_points == (hops == 2 ? 4 : 2));
                            REQUIRE(s.dynamic_active_cliques == (hops == 2 and not chain ? 1 : 0));
                        } else {
                            REQUIRE(graph->reads == 0);
                            REQUIRE(s.hub_expanded_nodes == 0);
                            REQUIRE(s.dynamic_scored_points == 5);
                            REQUIRE(s.dynamic_active_cliques == 4);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace vsag
