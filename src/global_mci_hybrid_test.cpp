// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/global_mci_hybrid.h"

#include <catch2/catch_test_macros.hpp>
#include <future>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>

#include "impl/searcher/mci_neighborhood_probe.h"
#include "utils/search_lock.h"
#include "vsag/vsag.h"

namespace vsag {
namespace {
using Json = nlohmann::json;

class ProbeFilter : public Filter {
public:
    std::vector<int64_t> ids;
    float ratio{0};
    bool enumerate{true};
    bool
    CheckValid(int64_t id) const override {
        return std::find(ids.begin(), ids.end(), id) != ids.end();
    }
    float
    ValidRatio() const override {
        return ratio;
    }
    void
    GetValidIds(const int64_t** out, int64_t& count) const override {
        if (enumerate) {
            *out = ids.data();
            count = ids.size();
        }
    }
};

void
same(const DatasetPtr& a, const DatasetPtr& b) {
    REQUIRE(a->GetDim() == b->GetDim());
    for (int64_t i = 0; i < a->GetDim(); ++i) {
        CHECK(a->GetIds()[i] == b->GetIds()[i]);
        CHECK(a->GetDistances()[i] == b->GetDistances()[i]);
    }
}
}  // namespace

TEST_CASE("MCI neighborhood sampling deduplicates points and respects bounds",
          "[ut][global_mci_probe]") {
    InnerIdType offsets[]{0, 3, 6, 9};
    InnerIdType members[]{0, 1, 2, 1, 2, 3, 3, 4, 5};
    InnerIdType node_offsets[]{0, 1, 3, 5, 7, 8, 9};
    InnerIdType cids[]{0, 0, 1, 0, 1, 1, 2, 2, 2};
    CliqueDataCellBaseView view;
    view.p_maxc = offsets;
    view.maxcs = members;
    view.p_node_to_cid = node_offsets;
    view.node_to_cids = cids;
    view.total_clique_count = 3;
    MCINeighborhoodProbeScratch scratch;
    for (int repeat = 0; repeat < 4; ++repeat) {
        auto r = probe_mci_neighborhood(
            view, 6, {0, 1}, 3, 100, [](auto id) { return id % 2 == 0; }, scratch);
        CHECK(r.cliques == 2);
        CHECK(r.member_slots == 6);
        CHECK(r.considered == 4);
        CHECK(r.satisfied == 2);
    }
    for (uint64_t budget : {1, 2, 3, 4, 8}) {
        auto r = probe_mci_neighborhood(
            view, 6, {0, 1, 3, 99}, 2, budget, [](auto) { return true; }, scratch);
        CHECK(r.cliques <= 2);
        CHECK(r.member_slots <= budget);
        CHECK(r.considered <= r.member_slots);
        CHECK(r.satisfied == r.considered);
    }
    CHECK(probe_mci_neighborhood(
              view, 6, {}, 3, 100, [](auto) { return true; }, scratch)
              .considered == 0);
    CHECK(probe_mci_neighborhood(
              view, 6, {99}, 3, 100, [](auto) { return true; }, scratch)
              .considered == 0);
    for (auto entry : {std::pair<uint64_t, const char*>{0, "negative"},
                       {1, "negative"},
                       {10, "neutral"},
                       {30, "positive"}}) {
        MCINeighborhoodProbeResult r;
        r.considered = 100;
        r.satisfied = entry.first;
        classify_mci_neighborhood(r, .1, .5, 2, 1);
        CHECK(std::string(r.prediction) == entry.second);
    }
    MCINeighborhoodProbeResult empty;
    classify_mci_neighborhood(empty, .1, .5, 2, 1);
    CHECK(std::string(empty.prediction) == "unknown");
    empty.considered = 10;
    classify_mci_neighborhood(empty, 0, .5, 2, 1);
    CHECK(std::string(empty.prediction) == "unknown");
    classify_mci_neighborhood(empty, .001, .5, 2, 1);
    CHECK(std::string(empty.prediction) == "neutral");
}

TEST_CASE("Global MCI correlation router reuses one index and searches without locks",
          "[ut][global_mci_probe]") {
    constexpr int64_t n = 128, dim = 8;
    std::vector<float> vectors(n * dim);
    std::vector<int64_t> ids(n);
    for (int64_t i = 0; i < n; ++i) {
        ids[i] = i * 3 - 17;
        for (int64_t d = 0; d < dim; ++d) vectors[i * dim + d] = float(i) * 10 + d * .01F;
    }
    const Json config{{"dim", dim},
                      {"dtype", "float32"},
                      {"metric_type", "l2"},
                      {"index_param",
                       {{"base_quantization_type", "fp32"},
                        {"max_degree", 12},
                        {"ef_construction", 64},
                        {"graph_type", "nsw"},
                        {"store_raw_vector", true},
                        {"use_mci", true},
                        {"mci_mcs", 12},
                        {"mci_clique_max", 4},
                        {"mci_alpha", 1.2},
                        {"mci_knng_source", "hgraph"},
                        {"build_thread_count", 1}}}};
    auto made = Factory::CreateIndex("hgraph", config.dump());
    REQUIRE(made.has_value());
    auto index = made.value();
    CHECK_FALSE(GlobalMCIHybrid::Create(index).has_value());
    auto built = index->Build(Dataset::Make()
                                  ->NumElements(n)
                                  ->Dim(dim)
                                  ->Ids(ids.data())
                                  ->Float32Vectors(vectors.data())
                                  ->Owner(false));
    REQUIRE(built.has_value());
    auto created = GlobalMCIHybrid::Create(index);
    INFO((created ? "created" : created.error().message));
    REQUIRE(created.has_value());
    auto router = created.value();
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    auto near = std::make_shared<ProbeFilter>();
    near->ids.assign(ids.begin(), ids.begin() + 16);
    near->ratio = .125F;
    auto far = std::make_shared<ProbeFilter>();
    far->ids.assign(ids.end() - 8, ids.end());
    far->ratio = .0625F;
    Json p{{"correlation_method", "clique_probe"},
           {"hgraph_valid_ratio_threshold", .5},
           {"ef_search", 32},
           {"mci_seed_ratio", .1},
           {"mci_seed_coverage", 0},
           {"mci_seed_max_count", 10000},
           {"search_no_lock", true},
           {"correlation_member_budget", 8192},
           {"correlation_min_expected", .01}};
    auto got = router->KnnSearch(query, 3, p.dump(), near);
    REQUIRE(got.has_value());
    auto stats = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["navigation_top10"] == 10);
    CHECK(stats["navigation_distances"] > 0);
    CHECK(stats["navigation_seeds"] > 0);
    CHECK(Json::parse(got.value()->GetStatistics())["mci_seed_count"].get<uint64_t>() >=
          stats["navigation_seeds"].get<uint64_t>());
    CHECK(stats["partition_dependency"] == false);
    CHECK_FALSE(SearchLockScope::IsNoLock());
    for (bool enumerate : {true, false}) {
        far->enumerate = enumerate;
        got = router->KnnSearch(query, 3, p.dump(), far);
        REQUIRE(got.has_value());
        stats = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
        REQUIRE(stats["prediction"] == "negative");
        REQUIRE(stats["route"] == "scan");
        CHECK(stats["search_distances"] == 8);
        for (int64_t i = 0; i < 3; ++i) CHECK(got.value()->GetIds()[i] == ids[n - 8 + i]);
    }
    far->enumerate = true;
    far->ids.push_back(far->ids.front());
    far->ids.push_back(1000000);
    got = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(got.has_value());
    CHECK(Json::parse(got.value()->GetStatistics())["global_mci_hybrid"]["search_distances"] == 8);
    p["correlation_probe_only"] = true;
    got = router->KnnSearch(query, 3, p.dump(), near);
    REQUIRE(got.has_value());
    CHECK(Json::parse(got.value()->GetStatistics())["global_mci_hybrid"]["route"] == "mci");
    p["hgraph_valid_ratio_threshold"] = .125;
    got = router->KnnSearch(query, 3, p.dump(), near);
    REQUIRE(got.has_value());
    stats = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["route"] == "hgraph");
    CHECK(stats["navigation_distances"] == 0);
    auto direct = index->KnnSearch(
        query,
        3,
        R"({"hgraph":{"use_mci":false,"ef_search":32,"search_no_lock":true,"parallelism":1}})",
        near);
    REQUIRE(direct.has_value());
    same(got.value(), direct.value());
    auto empty = std::make_shared<ProbeFilter>();
    got = router->KnnSearch(query, 3, Json{{"hgraph_valid_ratio_threshold", .5}}.dump(), empty);
    REQUIRE(got.has_value());
    CHECK(got.value()->GetDim() == 0);
    CHECK_FALSE(GlobalMCIHybrid::Create(nullptr).has_value());
    CHECK_FALSE(router->KnnSearch(nullptr, 3, "{}", near).has_value());
    CHECK_FALSE(router->KnnSearch(query, 0, "{}", near).has_value());
    for (const auto& text : {"[]",
                             "{",
                             "{\"unknown\":1}",
                             "{\"search_no_lock\":false}",
                             "{\"parallelism\":2}",
                             "{\"correlation_member_budget\":0}",
                             "{\"correlation_max_cliques\":-1}",
                             "{\"correlation_navigation_ef\":9}",
                             "{\"correlation_navigation_ef\":10.5}",
                             "{\"correlation_negative_ratio\":1}",
                             "{\"correlation_positive_ratio\":1}",
                             "{\"correlation_min_expected\":0}",
                             "{\"hgraph_valid_ratio_threshold\":-1}",
                             "{\"global_mci_hybrid\":{},\"other\":0}"}) {
        CHECK_FALSE(router->KnnSearch(query, 3, text, near).has_value());
    }
    std::vector<std::future<DatasetPtr>> tasks;
    for (int i = 0; i < 16; ++i)
        tasks.push_back(std::async(std::launch::async, [&]() {
            auto r = router->KnnSearch(query, 3, p.dump(), near);
            if (not r)
                throw std::runtime_error(r.error().message);
            return r.value();
        }));
    for (auto& task : tasks) same(task.get(), direct.value());
    p["hgraph_valid_ratio_threshold"] = .5;
    p["correlation_probe_only"] = false;
    auto reference = router->KnnSearch(query, 3, Json{{"global_mci_hybrid", p}}.dump(), near);
    REQUIRE(reference.has_value());
    tasks.clear();
    for (int i = 0; i < 16; ++i) {
        tasks.push_back(std::async(std::launch::async, [&]() {
            auto r = router->KnnSearch(query, 3, p.dump(), near);
            if (not r)
                throw std::runtime_error(r.error().message);
            return r.value();
        }));
    }
    for (auto& task : tasks) same(task.get(), reference.value());
    p["mci_seed_max_count"] = 3;
    got = router->KnnSearch(query, 3, p.dump(), near);
    REQUIRE(got.has_value());
    CHECK(Json::parse(got.value()->GetStatistics())["mci_seed_count"] == 3);
}
}  // namespace vsag

#include "impl/searcher/mci_distance_sampling.h"

namespace vsag {
TEST_CASE("Distance sampling is bounded, distinct and deterministic",
          "[ut][mci_distance_sampling]") {
    for (uint64_t population : {0, 1, 10, 1000}) {
        for (uint64_t requested : {0, 1, 8, 64}) {
            auto a = mci_sample_positions(population, requested, 42);
            auto b = mci_sample_positions(population, requested, 42);
            REQUIRE(a == b);
            REQUIRE(a.size() == std::min(population, requested));
            REQUIRE(std::adjacent_find(a.begin(), a.end()) == a.end());
            for (auto id : a) REQUIRE(id < population);
        }
    }
    std::vector<float> a(512), positive(256), neutral(256), negative(256);
    for (uint64_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i);
    for (uint64_t i = 0; i < positive.size(); ++i) {
        positive[i] = static_cast<float>(i % 32);
        neutral[i] = static_cast<float>(i * 2);
        negative[i] = static_cast<float>(1000 + i);
    }
    for (const auto& item : {std::make_pair(positive, "positive"),
                             std::make_pair(neutral, "neutral"),
                             std::make_pair(negative, "negative")}) {
        const auto r = mci_classify_distance_samples(a, item.first, .1, .5, 2, 1.96);
        CHECK(std::string(r.prediction) == item.second);
        CHECK(r.lower <= r.enrichment);
        CHECK(r.upper >= r.enrichment);
    }
    CHECK(std::string(mci_classify_distance_samples({}, neutral, .1, .5, 2, 1.96).prediction) ==
          "unknown");
    CHECK(std::string(mci_classify_distance_samples(a, {}, .1, .5, 2, 1.96).prediction) ==
          "unknown");
    CHECK(std::string(
              mci_classify_distance_samples({1, 1, 1}, {1, 1, 1}, .1, .5, 2, 1.96).prediction) ==
          "neutral");
}

TEST_CASE("MCI distance samples reuse exact distances for seeds and scans",
          "[ut][mci_distance_sampling]") {
    constexpr int64_t n = 128, dim = 8;
    std::vector<float> vectors(n * dim);
    std::vector<int64_t> labels(n);
    for (int64_t i = 0; i < n; ++i) {
        labels[i] = i * 3 - 17;
        for (int64_t d = 0; d < dim; ++d) vectors[i * dim + d] = static_cast<float>(i * 10 + d);
    }
    const Json config{{"dim", dim},
                      {"dtype", "float32"},
                      {"metric_type", "l2"},
                      {"index_param",
                       {{"base_quantization_type", "fp32"},
                        {"max_degree", 12},
                        {"ef_construction", 64},
                        {"graph_type", "nsw"},
                        {"use_mci", true},
                        {"mci_mcs", 12},
                        {"mci_clique_max", 4},
                        {"mci_alpha", 1.2},
                        {"mci_knng_source", "hgraph"},
                        {"build_thread_count", 1}}}};
    auto made = Factory::CreateIndex("hgraph", config.dump());
    REQUIRE(made);
    auto index = made.value();
    REQUIRE(index->Build(Dataset::Make()
                             ->NumElements(n)
                             ->Dim(dim)
                             ->Ids(labels.data())
                             ->Float32Vectors(vectors.data())
                             ->Owner(false)));
    auto created = GlobalMCIHybrid::Create(index);
    REQUIRE(created);
    auto router = created.value();
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    auto far = std::make_shared<ProbeFilter>();
    far->ids.assign(labels.end() - 8, labels.end());
    far->ratio = 8.0F / n;
    Json p{{"hgraph_valid_ratio_threshold", 1.0},
           {"ef_search", 16},
           {"mci_seed_coverage", 0},
           {"mci_seed_max_count", 10000},
           {"mci_hops_limit", 1},
           {"correlation_method", "sample_seeds"},
           {"correlation_valid_samples", 8},
           {"correlation_global_samples", 64}};
    auto got = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(got);
    auto s = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(s["sample_distances"] == 8);
    CHECK(s["search_distances"] == 0);
    CHECK(s["reused_sample_distances"] == 8);
    CHECK(s["navigation_distances"] == 0);
    for (int64_t i = 0; i < 3; ++i) CHECK(got.value()->GetIds()[i] == labels[n - 8 + i]);
    auto direct = index->KnnSearch(
        query,
        3,
        R"({"hgraph":{"use_mci":true,"use_hybrid_traversal":false,"ef_search":16,"mci_seed_ratio":10,"mci_seed_coverage":0,"mci_hops_limit":1,"hgraph_valid_ratio_threshold":1,"brute_force_threshold":0,"search_no_lock":true,"parallelism":1}})",
        far);
    REQUIRE(direct);
    same(got.value(), direct.value());
    p["correlation_method"] = "distance_samples";
    p["correlation_confidence_z"] = .1;
    got = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(got);
    s = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(s["prediction"] == "negative");
    CHECK(s["route"] == "scan");
    CHECK(s["reused_sample_distances"] == 8);
    CHECK(s["search_distances"] == 0);
    CHECK(s["sample_distances"].get<int>() >= 64);
    CHECK(s["sample_distances"].get<int>() <= 72);
    for (int64_t i = 0; i < 3; ++i) CHECK(got.value()->GetIds()[i] == labels[n - 8 + i]);
    p["correlation_probe_only"] = true;
    got = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(got);
    s = Json::parse(got.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(s["route"] == "mci");
    CHECK(s["search_distances"] == 0);
    far->enumerate = false;
    CHECK_FALSE(router->KnnSearch(query, 3, p.dump(), far));
    far->enumerate = true;
    for (const auto& bad : {Json{{"correlation_method", "bad"}},
                            Json{{"correlation_global_samples", 0}},
                            Json{{"correlation_valid_samples", 0}},
                            Json{{"correlation_sample_seed", -1}},
                            Json{{"correlation_radius_quantile", 0}},
                            Json{{"correlation_radius_quantile", 2}},
                            Json{{"correlation_confidence_z", 0}}}) {
        CHECK_FALSE(router->KnnSearch(query, 3, bad.dump(), far));
    }
    p["mci_seed_max_count"] = 1;
    CHECK_FALSE(router->KnnSearch(query, 3, p.dump(), far));
}

TEST_CASE("Distance sampling routes conserve distances and remain reproducible",
          "[ut][mci_distance_sampling][regression]") {
    constexpr int64_t n = 128, dim = 8;
    std::vector<float> vectors(n * dim);
    std::vector<int64_t> labels(n);
    for (int64_t i = 0; i < n; ++i) {
        labels[i] = i * 3 - 17;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[i * dim + d] = static_cast<float>(i * 10 + d);
        }
    }
    const Json config{{"dim", dim},
                      {"dtype", "float32"},
                      {"metric_type", "l2"},
                      {"index_param",
                       {{"base_quantization_type", "fp32"},
                        {"max_degree", 12},
                        {"ef_construction", 64},
                        {"graph_type", "nsw"},
                        {"use_mci", true},
                        {"mci_mcs", 12},
                        {"mci_clique_max", 4},
                        {"mci_alpha", 1.2},
                        {"mci_knng_source", "hgraph"},
                        {"build_thread_count", 1}}}};
    auto made = Factory::CreateIndex("hgraph", config.dump());
    REQUIRE(made);
    auto index = made.value();
    REQUIRE(index->Build(Dataset::Make()
                             ->NumElements(n)
                             ->Dim(dim)
                             ->Ids(labels.data())
                             ->Float32Vectors(vectors.data())
                             ->Owner(false)));
    auto created = GlobalMCIHybrid::Create(index);
    REQUIRE(created);
    auto router = created.value();
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    auto far = std::make_shared<ProbeFilter>();
    far->ids.assign(labels.end() - 8, labels.end());
    far->ratio = 8.0F / n;
    Json p{{"hgraph_valid_ratio_threshold", 1.0},
           {"ef_search", 16},
           {"mci_seed_coverage", 0},
           {"mci_seed_max_count", 0},
           {"mci_hops_limit", 1},
           {"mci_expansion_idle_window", 0},
           {"correlation_method", "distance_samples"},
           {"correlation_confidence_z", .1},
           {"correlation_valid_samples", 3},
           {"correlation_global_samples", n}};
    auto scanned = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(scanned);
    auto stats = Json::parse(scanned.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["route"] == "scan");
    CHECK(stats["prediction"] == "negative");
    // Every B point is already in A. Only the five unsampled valid points need new distances.
    CHECK(stats["sample_distances"] == n);
    CHECK(stats["reused_sample_distances"] == 3);
    CHECK(stats["search_distances"] == 5);
    CHECK(stats["total_distances"] == n + 5);
    CHECK(stats["navigation_distances"] == 0);
    for (int64_t i = 0; i < 3; ++i) {
        CHECK(scanned.value()->GetIds()[i] == labels[n - 8 + i]);
    }
    // Negative prediction with probe-only must use exactly the same cached seed set as control.
    p["correlation_probe_only"] = true;
    auto predicted = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(predicted);
    stats = Json::parse(predicted.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["route"] == "mci");
    CHECK(stats["search_distances"] == 0);
    CHECK(stats["reused_sample_distances"] == 3);
    CHECK(Json::parse(predicted.value()->GetStatistics())["mci_seed_count"] == 3);
    p["correlation_method"] = "sample_seeds";
    auto control = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(control);
    same(predicted.value(), control.value());
    stats = Json::parse(control.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["sample_distances"] == 3);
    CHECK(stats["total_distances"] == 3);
    std::vector<std::future<DatasetPtr>> tasks;
    for (int i = 0; i < 16; ++i) {
        tasks.push_back(std::async(std::launch::async, [&]() {
            auto result = router->KnnSearch(query, 3, p.dump(), far);
            if (not result) {
                throw std::runtime_error(result.error().message);
            }
            return result.value();
        }));
    }
    for (auto& task : tasks) {
        same(control.value(), task.get());
    }
    p["correlation_routing"] = false;
    auto disabled = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(disabled);
    stats = Json::parse(disabled.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["sample_distances"] == 0);
    CHECK(stats["reused_sample_distances"] == 0);
    p["correlation_routing"] = true;
    p["hgraph_valid_ratio_threshold"] = far->ratio;
    auto high = router->KnnSearch(query, 3, p.dump(), far);
    REQUIRE(high);
    stats = Json::parse(high.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["route"] == "hgraph");
    CHECK(stats["sample_distances"] == 0);
    p["hgraph_valid_ratio_threshold"] = 1.0;
    for (const auto& bad : {Json{{"correlation_global_samples", 65537}},
                            Json{{"correlation_valid_samples", 1025}},
                            Json{{"correlation_sample_seed", 1000000001}},
                            Json{{"correlation_valid_samples", 1.5}},
                            Json{{"correlation_radius_quantile", nullptr}},
                            Json{{"mci_seed_max_count", 2}},
                            Json{{"correlation_method", "invalid"}}}) {
        auto invalid = p;
        invalid.update(bad);
        auto result = router->KnnSearch(query, 3, invalid.dump(), far);
        REQUIRE_FALSE(result);
        CHECK(result.error().type == ErrorType::INVALID_ARGUMENT);
    }
    p["correlation_valid_samples"] = 8;
    far->ids.back() = far->ids.front();
    CHECK_FALSE(router->KnnSearch(query, 3, p.dump(), far));
    far->ids.back() = 1000000;
    CHECK_FALSE(router->KnnSearch(query, 3, p.dump(), far));
    auto empty = std::make_shared<ProbeFilter>();
    auto result = router->KnnSearch(query, 3, p.dump(), empty);
    REQUIRE(result);
    CHECK(result.value()->GetDim() == 0);
    stats = Json::parse(result.value()->GetStatistics())["global_mci_hybrid"];
    CHECK(stats["sample_distances"] == 0);
    CHECK(stats["total_distances"] == 0);
}

TEST_CASE("Distance classification includes radius ties and threshold equality",
          "[ut][mci_distance_sampling][regression]") {
    std::vector<float> global(512);
    std::iota(global.begin(), global.end(), 0.0F);
    std::vector<float> far(256, 1000.0F);
    auto negative = mci_classify_distance_samples(global, far, .1, .5, 2, 1.96);
    CHECK(negative.radius == 51.0F);
    CHECK(negative.global_inside == 52);
    CHECK(negative.valid_inside == 0);
    CHECK(std::string(
              mci_classify_distance_samples(global, far, .1, negative.upper, 2, 1.96).prediction) ==
          "negative");
    std::vector<float> near(256, 0.0F);
    auto positive = mci_classify_distance_samples(global, near, .1, .5, 2, 1.96);
    CHECK(std::string(mci_classify_distance_samples(global, near, .1, .5, positive.lower, 1.96)
                          .prediction) == "positive");
    auto ties = mci_classify_distance_samples({1, 1, 1, 2}, {1, 2}, .1, .5, 2, 1.96);
    CHECK(ties.global_inside == 3);
    CHECK(ties.valid_inside == 1);
    auto all = mci_classify_distance_samples(global, global, 1, .5, 2, 1.96);
    CHECK(all.radius == 511.0F);
    CHECK(all.global_inside == 512);
    CHECK(all.valid_inside == 512);
    CHECK(std::string(all.prediction) == "neutral");
}

TEST_CASE("Global MCI defaults match explicit distance sample settings",
          "[ut][mci_distance_sampling][regression]") {
    constexpr int64_t n = 4096, dim = 8;
    std::vector<float> vectors(n * dim);
    std::vector<int64_t> labels(n);
    for (int64_t i = 0; i < n; ++i) {
        labels[i] = i * 3 - 17;
        for (int64_t d = 0; d < dim; ++d) vectors[i * dim + d] = static_cast<float>(i * 10 + d);
    }
    const Json config{{"dim", dim},
                      {"dtype", "float32"},
                      {"metric_type", "l2"},
                      {"index_param",
                       {{"base_quantization_type", "fp32"},
                        {"max_degree", 12},
                        {"ef_construction", 64},
                        {"graph_type", "nsw"},
                        {"use_mci", true},
                        {"mci_mcs", 12},
                        {"mci_clique_max", 4},
                        {"mci_alpha", 1.2},
                        {"mci_knng_source", "hgraph"},
                        {"build_thread_count", 1}}}};
    auto made = Factory::CreateIndex("hgraph", config.dump());
    REQUIRE(made);
    auto index = made.value();
    REQUIRE(index->Build(Dataset::Make()
                             ->NumElements(n)
                             ->Dim(dim)
                             ->Ids(labels.data())
                             ->Float32Vectors(vectors.data())
                             ->Owner(false)));
    auto created = GlobalMCIHybrid::Create(index);
    REQUIRE(created);
    auto router = created.value();
    auto query =
        Dataset::Make()->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);
    const Json explicit_defaults{{"correlation_method", "distance_samples"},
                                 {"correlation_routing", true},
                                 {"correlation_probe_only", false},
                                 {"correlation_global_samples", 512},
                                 {"correlation_valid_samples", 256},
                                 {"correlation_sample_seed", 42},
                                 {"correlation_radius_quantile", 0.1},
                                 {"correlation_confidence_z", 1.96},
                                 {"correlation_negative_ratio", 0.5},
                                 {"correlation_positive_ratio", 2.0}};
    for (const bool negative : {false, true}) {
        auto filter = std::make_shared<ProbeFilter>();
        if (negative) {
            filter->ids.assign(labels.end() - 128, labels.end());
        } else {
            filter->ids.assign(labels.begin(), labels.begin() + 128);
        }
        filter->ratio = 128.0F / n;
        auto explicit_result = router->KnnSearch(query, 3, explicit_defaults.dump(), filter);
        REQUIRE(explicit_result);
        const auto expected =
            Json::parse(explicit_result.value()->GetStatistics())["global_mci_hybrid"];
        for (const std::string parameters : {"", "{}", R"({"global_mci_hybrid":{}})"}) {
            auto implicit_result = router->KnnSearch(query, 3, parameters, filter);
            REQUIRE(implicit_result);
            same(implicit_result.value(), explicit_result.value());
            auto stats = Json::parse(implicit_result.value()->GetStatistics())["global_mci_hybrid"];
            CHECK(stats["correlation_method"] == "distance_samples");
            CHECK(stats["prediction"] == (negative ? "negative" : "positive"));
            CHECK(stats["route"] == (negative ? "scan" : "mci"));
            CHECK(stats["global_sample_count"] == 512);
            CHECK(stats["valid_sample_count"] == 128);
            CHECK(stats["reused_sample_distances"] == 128);
            CHECK(stats["navigation_distances"] == 0);
            CHECK(stats["search_distances"] == 0);
            for (const auto* key : {"radius",
                                    "enrichment",
                                    "enrichment_lower",
                                    "enrichment_upper",
                                    "sample_distances",
                                    "total_distances"}) {
                CHECK(stats[key] == expected[key]);
            }
        }
    }
}
}  // namespace vsag
