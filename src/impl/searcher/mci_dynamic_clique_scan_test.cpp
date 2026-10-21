// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "datacell/clique_datacell.h"
#include "datacell/flatten_datacell.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/searcher/mci_clique_counts.h"
#include "impl/searcher/mci_depth_stats.h"
#include "impl/searcher/mci_ordered_clique_scan.h"
#include "impl/searcher/mci_searcher.h"
#include "io/memory_block_io/memory_block_io.h"
#include "quantization/fp32_quantizer.h"
#include "utils/search_lock.h"

namespace vsag {

TEST_CASE("MCI dynamic clique counts reuse and removal", "[ut][mci_dynamic_clique_scan]") {
    MCICliqueCounts<uint8_t> counts;
    counts.Reset(8);
    REQUIRE(counts.Add(3));
    REQUIRE_FALSE(counts.Add(3));
    REQUIRE(counts.Get(3) == 2);
    REQUIRE(counts.Remove(3));
    REQUIRE(counts.Get(3) == 1);
    REQUIRE(counts.Remove(3));
    REQUIRE(counts.Get(3) == 0);
    REQUIRE_FALSE(counts.Remove(3));
    REQUIRE_FALSE(counts.Add(8));
    REQUIRE_FALSE(counts.Remove(8));
    REQUIRE(counts.Get(8) == 0);
    for (uint64_t q = 0; q < 600; ++q) {
        counts.Reset(8);
        REQUIRE(counts.Get(3) == 0);
        REQUIRE(counts.Add(3));
    }
    counts.Reset(16);
    REQUIRE(counts.Get(3) == 0);
    REQUIRE(counts.Add(15));
    counts.Reset(4);
    REQUIRE(counts.Get(15) == 0);
    REQUIRE_FALSE(counts.Remove(15));
    std::mt19937 rng(4719);
    std::vector<uint32_t> expected(97);
    counts.Reset(expected.size());
    for (uint64_t i = 0; i < 20000; ++i) {
        const auto id = rng() % expected.size();
        if (rng() & 1U) {
            REQUIRE(counts.Add(id) == (expected[id]++ == 0));
        } else {
            const bool present = expected[id] != 0;
            REQUIRE(counts.Remove(id) == present);
            if (present) {
                --expected[id];
            }
        }
        REQUIRE(counts.Get(id) == expected[id]);
    }
}

class DynamicTestFilter : public Filter {
public:
    std::set<InnerIdType> valid;
    bool
    CheckValid(int64_t id) const override {
        return valid.count(id) != 0;
    }
};

// Deliberately slow oracle: rebuild X from sets after each heap change and fully sort all scored
// points. It uses neither reference counts nor the production heap/inverse-membership scan.
template <MetricType metric>
void
CheckDynamicSearch(const std::vector<float>& vectors,
                   uint64_t dim,
                   const std::vector<std::vector<InnerIdType>>& rows,
                   const std::vector<InnerIdType>& seeds,
                   const std::vector<InnerIdType>& valid,
                   uint64_t ef,
                   uint64_t k,
                   bool raw,
                   bool saturated = false,
                   uint32_t pass_limit = 0,
                   bool gallop = false) {
    auto alloc = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = dim;
    common.allocator_ = alloc;
    common.metric_ = metric;
    auto quantizer =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"block_memory_io"})"));
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<metric>, FixedLayout<MemoryBlockIO>>>(
            quantizer, io, common);
    const auto n = vectors.size() / dim;
    std::vector<InnerIdType> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    flatten->Train(vectors.data(), n);
    flatten->BatchInsertVector(vectors.data(), n, ids.data());
    Vector<InnerIdType> offsets(alloc.get()), members(alloc.get());
    Vector<InnerIdType> inverse_offsets(alloc.get()), inverse(alloc.get());
    offsets.push_back(0);
    inverse_offsets.push_back(0);
    std::vector<std::vector<InnerIdType>> inverse_rows(n);
    for (uint64_t cid = 0; cid < rows.size(); ++cid) {
        for (auto id : rows[cid]) {
            members.push_back(id);
            inverse_rows[id].push_back(cid);
        }
        offsets.push_back(members.size());
    }
    for (const auto& row : inverse_rows) {
        inverse.insert(inverse.end(), row.begin(), row.end());
        inverse_offsets.push_back(inverse.size());
    }
    auto cliques = std::make_shared<CliqueDataCell>(alloc.get());
    cliques->Assign(
        std::move(offsets), std::move(members), std::move(inverse_offsets), std::move(inverse), n);
    // Use exact 1-D L2 integer distances in structural tests and normalized vectors for cosine.
    std::vector<float> query(dim, 0.0F);
    if constexpr (metric != MetricType::METRIC_TYPE_L2SQR) {
        query[0] = 1.0F;
    }
    auto distance = [&](InnerIdType id) {
        float d = metric == MetricType::METRIC_TYPE_L2SQR ? 0.0F : 1.0F;
        for (uint64_t j = 0; j < dim; ++j) {
            if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                const auto delta = query[j] - vectors[id * dim + j];
                d += delta * delta;
            } else {
                d -= query[j] * vectors[id * dim + j];
            }
        }
        return d;
    };
    std::set<InnerIdType> scored;
    std::vector<std::pair<float, InnerIdType>> top;
    auto score = [&](InnerIdType id) {
        if (not scored.insert(id).second) {
            return;
        }
        top.emplace_back(distance(id), id);
        std::sort(top.begin(), top.end());
        if (top.size() > ef) {
            top.pop_back();
        }
    };
    std::set<InnerIdType> legal(valid.begin(), valid.end());
    for (auto id : seeds) {
        if (legal.count(id)) {
            score(id);
        }
    }
    auto covered = [&] {
        std::set<InnerIdType> result;
        for (const auto& row : rows) {
            bool active = false;
            for (const auto& record : top) {
                active |= std::find(row.begin(), row.end(), record.second) != row.end();
            }
            if (active) {
                result.insert(row.begin(), row.end());
            }
        }
        return result;
    };
    auto scan_ids = valid;
    if (pass_limit != 0) {
        CliqueDataCellBaseView view;
        REQUIRE(cliques->TryGetBaseView(n, view));
        std::sort(scan_ids.begin(), scan_ids.end(), [&](auto a, auto b) {
            const auto* ab = view.node_to_cids + view.p_node_to_cid[a];
            const auto* ae = view.node_to_cids + view.p_node_to_cid[a + 1];
            const auto* bb = view.node_to_cids + view.p_node_to_cid[b];
            const auto* be = view.node_to_cids + view.p_node_to_cid[b + 1];
            if (std::lexicographical_compare(ab, ae, bb, be))
                return true;
            if (std::lexicographical_compare(bb, be, ab, ae))
                return false;
            return a < b;
        });
    }
    uint32_t oracle_passes = 0;
    for (;;) {
        if (pass_limit != 0 and oracle_passes++ == pass_limit)
            break;
        const auto before = scored.size();
        for (auto id : scan_ids) {
            if (not scored.count(id) and covered().count(id)) {
                score(id);
            }
        }
        if (scored.size() == before) {
            break;
        }
    }
    std::vector<std::pair<float, InnerIdType>> expected;
    for (auto id : scored) {
        expected.emplace_back(distance(id), id);
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(std::min<uint64_t>(k, expected.size()));
    Vector<InnerIdType> seed_ids(seeds.begin(), seeds.end(), alloc.get());
    std::vector<uint8_t> bitmap(n);
    for (auto id : valid) {
        bitmap[id] = 1;
    }
    auto filter = std::make_shared<DynamicTestFilter>();
    filter->valid = legal;
    for (bool use_bitmap : {false, true}) {
        MCISearcherParam params;
        params.dynamic_clique_scan = true;
        params.clique_scan_passes = pass_limit;
        params.clique_scan_gallop = gallop;
        params.seed_count = seeds.size();
        params.seed_inner_ids = &seed_ids;
        params.valid_inner_ids = valid.data();
        params.valid_inner_count = valid.size();
        params.precise_vectors = raw ? vectors.data() : nullptr;
        params.precise_vector_stride = dim;
        params.dim = dim;
        params.metric = metric;
        params.valid_bitmap = use_bitmap ? bitmap.data() : nullptr;
        params.valid_bitmap_size = bitmap.size();
        params.enumerated_valid_count = saturated ? valid.size() : 0;
        bool skipped = false;
        params.skipped_expansion = &skipped;
        InnerSearchParam search;
        search.ef = ef;
        search.topk = k;
        search.is_inner_id_allowed = filter;
        SearchLockScope no_lock(true);
        auto result =
            MCISearcher(common).Search(cliques, flatten, query.data(), search, params, nullptr);
        std::vector<std::pair<float, InnerIdType>> actual;
        while (not result->Empty()) {
            actual.push_back(result->Top());
            result->Pop();
        }
        std::sort(actual.begin(), actual.end());
        REQUIRE(actual.size() == expected.size());
        for (uint64_t i = 0; i < actual.size(); ++i) {
            REQUIRE(actual[i].second == expected[i].second);
            REQUIRE(std::abs(actual[i].first - expected[i].first) < 1e-5F);
        }
        const auto& stats = GetMCIDepthSearchStatistics();
        REQUIRE(stats.dynamic_clique_scan_used);
        REQUIRE(stats.dynamic_scored_points == scored.size());
        REQUIRE(stats.hops_limit == 0);
        REQUIRE(stats.member_checks == 0);
        if (pass_limit != 0) {
            REQUIRE(stats.dynamic_scan_passes <= pass_limit);
            REQUIRE(stats.bounded_scan_pass_limit == pass_limit);
        }
        REQUIRE(skipped == saturated);
        if (saturated) {
            REQUIRE(stats.dynamic_scan_passes == 0);
            REQUIRE(stats.dynamic_count_updates == 0);
        } else {
            CliqueDataCellBaseView view;
            REQUIRE(cliques->TryGetBaseView(n, view));
            uint64_t active_cliques = 0;
            for (uint64_t c = 0; c < view.total_clique_count; ++c) {
                bool active = false;
                for (auto off = view.p_maxc[c]; off < view.p_maxc[c + 1]; ++off) {
                    for (const auto& candidate : top) {
                        active |= view.maxcs[off] == candidate.second;
                    }
                }
                active_cliques += active;
            }
            REQUIRE(stats.dynamic_active_cliques == active_cliques);
        }
    }
}

TEST_CASE("MCI ordered active bitmap hierarchy and epochs", "[ut][mci_dynamic_clique_scan]") {
    MCIOrderedCliqueCounts<uint8_t> counts;
    constexpr auto none = MCIOrderedCliqueCounts<uint8_t>::kNone;
    counts.Reset(300100);
    REQUIRE(counts.Minimum() == none);
    REQUIRE(counts.Next(0) == none);
    REQUIRE(counts.Previous(1000000) == none);
    std::set<uint32_t> active{0, 63, 64, 4095, 4096, 262143, 262144, 300000};
    for (auto id : active) REQUIRE(counts.Add(id));
    REQUIRE(counts.Minimum() == 0);
    REQUIRE(counts.Maximum() == 300000);
    REQUIRE_FALSE(counts.Add(64));
    REQUIRE(counts.Remove(64));
    REQUIRE(counts.Contains(64));
    for (uint32_t id : {0U, 1U, 63U, 64U, 65U, 4000U, 4096U, 262144U, 300099U, 400000U}) {
        auto next = active.lower_bound(id);
        REQUIRE(counts.Next(id) == (next == active.end() ? none : *next));
        auto prev = active.upper_bound(id);
        REQUIRE(counts.Previous(id) == (prev == active.begin() ? none : *--prev));
    }
    while (not active.empty()) {
        auto id = *active.begin();
        active.erase(active.begin());
        REQUIRE(counts.Remove(id));
        REQUIRE_FALSE(counts.Contains(id));
        REQUIRE(counts.Minimum() == (active.empty() ? none : *active.begin()));
        REQUIRE(counts.Maximum() == (active.empty() ? 0 : *active.rbegin()));
    }
    REQUIRE_FALSE(counts.Remove(300100));
    REQUIRE_FALSE(counts.Add(300100));
    for (uint32_t query = 0; query < 600; ++query) {
        counts.Reset(query % 2 == 0 ? 300100 : 65);
        REQUIRE_FALSE(counts.Contains(64));
        REQUIRE(counts.Minimum() == none);
        REQUIRE(counts.Add(64));
        REQUIRE(counts.Next(0) == 64);
        REQUIRE(counts.Previous(300000) == 64);
    }
    counts.Reset(8193);
    std::mt19937 rng(7129);
    active.clear();
    std::vector<uint32_t> expected(8193);
    for (uint64_t i = 0; i < 20000; ++i) {
        const auto id = rng() % expected.size();
        if (rng() & 1U) {
            REQUIRE(counts.Add(id) == (expected[id]++ == 0));
            active.insert(id);
        } else {
            const auto present = expected[id] != 0;
            REQUIRE(counts.Remove(id) == present);
            if (present and --expected[id] == 0)
                active.erase(id);
        }
        REQUIRE(counts.Contains(id) == (expected[id] != 0));
        REQUIRE(counts.ContainsKnownValid(id) == (expected[id] != 0));
        REQUIRE(counts.Get(id) == expected[id]);
        REQUIRE(counts.Minimum() == (active.empty() ? none : *active.begin()));
        REQUIRE(counts.Maximum() == (active.empty() ? 0 : *active.rbegin()));
        auto probe = rng() % 9000;
        auto next = active.lower_bound(probe);
        REQUIRE(counts.Next(probe) == (next == active.end() ? none : *next));
        auto prev = active.upper_bound(probe);
        REQUIRE(counts.Previous(probe) == (prev == active.begin() ? none : *--prev));
    }
    counts.Reset(0);
    REQUIRE(counts.Next(0) == none);
    REQUIRE(counts.Previous(0) == none);
    REQUIRE_FALSE(counts.Contains(0));
}

TEST_CASE("MCI minimum lexicographic and maximum sequences", "[ut][mci_dynamic_clique_scan]") {
    const std::vector<uint32_t> offsets{0, 2, 3, 5, 5, 7, 8};
    const std::vector<uint32_t> cliques{0, 7, 5, 0, 9, 0, 7, 0};
    const std::vector<uint32_t> valid{0, 1, 2, 3, 4, 5};
    MCIOrderedValidRows order;
    order.Build(offsets.data(), cliques.data(), 6, valid.data(), valid.size());
    std::vector<uint32_t> ids;
    for (auto& row : order.rows) ids.push_back(row.id);
    REQUIRE(ids == std::vector<uint32_t>{5, 0, 4, 2, 1});
    for (uint32_t lower = 0; lower < 12; ++lower) {
        uint64_t expected = order.rows.size();
        for (uint64_t i = 0; i < order.rows.size(); ++i) {
            if (order.rows[i].maximum >= lower) {
                expected = i;
                break;
            }
        }
        REQUIRE(order.Start(lower) == expected);
    }
    order.Build(offsets.data(), cliques.data(), 6, nullptr, 0);
    REQUIRE(order.rows.empty());
    REQUIRE(order.Start(0) == 0);
}

TEST_CASE("MCI radix sequences preserve full uint32 lexicographic order",
          "[ut][mci_dynamic_clique_scan]") {
    std::mt19937 rng(80219);
    MCIOrderedValidRows order;
    for (uint32_t count : {0U, 1U, 1500U, 20U, 1200U}) {
        std::vector<uint32_t> offsets{0}, cliques, valid;
        std::vector<std::pair<std::vector<uint32_t>, uint32_t>> expected;
        for (uint32_t id = 0; id < count; ++id) {
            std::set<uint32_t> row;
            const auto size = id % 7;
            while (row.size() < size) {
                row.insert(id % 2 == 0 ? rng() : rng() % 11);
            }
            std::vector<uint32_t> members(row.begin(), row.end());
            if (not row.empty())
                expected.emplace_back(members, id);
            cliques.insert(cliques.end(), members.begin(), members.end());
            offsets.push_back(cliques.size());
            valid.push_back(id);
        }
        std::sort(expected.begin(), expected.end());
        order.Build(offsets.data(), cliques.data(), count, valid.data(), count);
        REQUIRE(order.rows.size() == expected.size());
        for (uint64_t i = 0; i < expected.size(); ++i) {
            REQUIRE(order.rows[i].id == expected[i].second);
        }
        for (uint32_t probe = 0; probe < 50; ++probe) {
            const auto lower = rng();
            uint64_t first = order.rows.size();
            for (uint64_t i = 0; i < order.rows.size(); ++i) {
                if (order.rows[i].maximum >= lower) {
                    first = i;
                    break;
                }
            }
            REQUIRE(order.Start(lower) == first);
        }
    }
}

TEST_CASE("MCI dynamic scan revisits earlier IDs and keeps shared coverage",
          "[ut][mci_dynamic_clique_scan]") {
    for (bool raw : {false, true}) {
        // New point 2 covers point 0, which was already skipped earlier in the first pass.
        CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>(
            {0, 10, 5}, 1, {{1, 2}, {2, 0}}, {1}, {0, 1, 2}, 1, 1, raw);
        // Shared cliques must retain positive counts when just one of their candidates leaves.
        CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>({0, 10, 8, 7, 1, 2, 3},
                                                          1,
                                                          {{1, 2, 3}, {2, 4}, {3, 5}, {4, 6}},
                                                          {1, 2},
                                                          {0, 1, 2, 3, 4, 5, 6},
                                                          2,
                                                          2,
                                                          raw);
        // A closer disconnected legal point and an illegal neighbor must both remain unscored.
        CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>(
            {0, 10, 5, 1}, 1, {{1, 2, 3}}, {1}, {0, 1, 2}, 1, 1, raw);
        CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>(
            {2, 4, 1}, 1, {{0, 1, 2}}, {0, 1, 2}, {0, 1, 2}, 2, 1, raw, true);
        CheckDynamicSearch<MetricType::METRIC_TYPE_IP>(
            {0.1F, 0.2F, 0.7F, 0.9F}, 1, {{0, 1}, {1, 2}, {2, 3}}, {0}, {0, 1, 2, 3}, 1, 1, raw);
        CheckDynamicSearch<MetricType::METRIC_TYPE_COSINE>({0, 1, .6F, .8F, .8F, .6F, 1, 0},
                                                           2,
                                                           {{0, 1}, {1, 2}, {2, 3}},
                                                           {0},
                                                           {0, 1, 2, 3},
                                                           1,
                                                           1,
                                                           raw);
    }
}

TEST_CASE("MCI dynamic scan agrees with explicit graph oracle", "[ut][mci_dynamic_clique_scan]") {
    std::mt19937 rng(90019);
    for (uint64_t trial = 0; trial < 36; ++trial) {
        std::vector<float> vectors(24);
        std::iota(vectors.begin(), vectors.end(), 1);
        std::shuffle(vectors.begin(), vectors.end(), rng);
        std::vector<std::vector<InnerIdType>> rows;
        for (uint64_t c = 0; c < 30; ++c) {
            std::set<InnerIdType> ids;
            const auto target = 2 + rng() % 4;
            while (ids.size() < target) {
                ids.insert(rng() % vectors.size());
            }
            rows.emplace_back(ids.begin(), ids.end());
        }
        std::vector<InnerIdType> valid;
        for (uint64_t i = 0; i < vectors.size(); ++i) {
            if (rng() % 3 != 0) {
                valid.push_back(i);
            }
        }
        REQUIRE(valid.size() >= 3);
        std::vector<InnerIdType> seeds{valid[1], valid.back(), valid[1]};
        const auto ef = 1 + trial % 5;
        CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>(
            vectors, 1, rows, seeds, valid, ef, std::min<uint64_t>(ef, 3), trial % 2 != 0);
        for (uint32_t limit : {1U, 2U})
            for (bool gallop : {false, true}) {
                CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>(vectors,
                                                                  1,
                                                                  rows,
                                                                  seeds,
                                                                  valid,
                                                                  ef,
                                                                  std::min<uint64_t>(ef, 3),
                                                                  trial % 2 != 0,
                                                                  false,
                                                                  limit,
                                                                  gallop);
            }
    }
}

TEST_CASE("MCI bounded second pass finds newly covered skipped prefix",
          "[ut][mci_dynamic_clique_scan]") {
    for (bool raw : {false, true})
        for (uint32_t limit : {1U, 2U})
            for (bool gallop : {false, true}) {
                CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>({0, 10, 5},
                                                                  1,
                                                                  {{0, 2}, {1, 2}},
                                                                  {1},
                                                                  {0, 1, 2},
                                                                  1,
                                                                  1,
                                                                  raw,
                                                                  false,
                                                                  limit,
                                                                  gallop);
                CheckDynamicSearch<MetricType::METRIC_TYPE_L2SQR>({0, 10, 5},
                                                                  1,
                                                                  {{0, 2}, {1, 2}},
                                                                  {0, 1, 2},
                                                                  {0, 1, 2},
                                                                  1,
                                                                  1,
                                                                  raw,
                                                                  true,
                                                                  limit,
                                                                  gallop);
                CheckDynamicSearch<MetricType::METRIC_TYPE_COSINE>({0, 1, .6F, .8F, .8F, .6F, 1, 0},
                                                                   2,
                                                                   {{0, 1}, {1, 2}, {2, 3}},
                                                                   {0},
                                                                   {0, 1, 2, 3},
                                                                   1,
                                                                   1,
                                                                   raw,
                                                                   false,
                                                                   limit,
                                                                   gallop);
            }
}

}  // namespace vsag
