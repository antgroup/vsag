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

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "algorithm/hgraph/hgraph_continue_session.h"
#include "impl/searcher/searcher_utils.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"
#include "vsag/index.h"
#include "vsag/index_features.h"
#include "vsag/iterator_context.h"
#include "vsag/search_session.h"

namespace {
constexpr int64_t kCount = 128;
constexpr int64_t kBatch = 8;
constexpr const char* kSearch =
    R"({"hgraph":{"ef_search":128,"parallelism":1,"brute_force_threshold":0.0}})";

vsag::IndexPtr
MakeSessionIndex(bool populate = true) {
    auto created = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32", "metric_type":"l2", "dim":1,
        "index_param":{"base_quantization_type":"fp32","max_degree":16,
        "ef_construction":128,"use_reorder":false}
    })");
    REQUIRE(created.has_value());
    auto index = created.value();
    if (populate) {
        std::vector<float> vectors(kCount);
        std::vector<int64_t> ids(kCount);
        for (int64_t i = 0; i < kCount; ++i) {
            vectors[i] = static_cast<float>(i);
            // Exercise external labels rather than internal graph offsets.
            ids[i] = 1000 + 3 * i;
        }
        auto base = vsag::Dataset::Make();
        base->NumElements(kCount)
            ->Dim(1)
            ->Ids(ids.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        auto built = index->Build(base);
        REQUIRE(built.has_value());
        REQUIRE(built.value().empty());
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_CONTINUE_SEARCH_SESSION));
        REQUIRE(index->CheckFeature(vsag::IndexFeature::SUPPORT_KNN_ITERATOR_FILTER_SEARCH));
    }
    return index;
}

vsag::DatasetPtr
MakeSessionQuery(float& value) {
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(1)->Float32Vectors(&value)->Owner(false);
    return query;
}

void
RequireSamePage(const vsag::DatasetPtr& actual, const vsag::DatasetPtr& expected) {
    REQUIRE(actual != nullptr);
    REQUIRE(expected != nullptr);
    REQUIRE(actual->GetDim() == expected->GetDim());
    for (int64_t i = 0; i < actual->GetDim(); ++i) {
        REQUIRE(actual->GetIds()[i] == expected->GetIds()[i]);
        REQUIRE(actual->GetDistances()[i] == expected->GetDistances()[i]);
    }
}

class SessionFilter : public vsag::Filter {
public:
    explicit SessionFilter(bool accept) : accept_(accept) {
    }
    bool
    CheckValid(int64_t) const override {
        return accept_;
    }
    bool
    CheckValid(const char*) const override {
        return accept_;
    }
    float
    ValidRatio() const override {
        return accept_ ? 1.0F : 0.0F;
    }

    void
    SetAccept(bool accept) {
        accept_ = accept;
    }

private:
    bool accept_;
};

// Own even an updated legacy context if a REQUIRE unwinds the test.
struct LegacyContext {
    vsag::IteratorContext* value{nullptr};
    ~LegacyContext() {
        delete value;
    }
};
}  // namespace

TEST_CASE("HGraph search session three batches compare legacy recall",
          "[ft][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto query = MakeSessionQuery(value);
    auto opened = index->OpenSearchSession(query, kBatch, kSearch);
    REQUIRE(opened.has_value());
    auto session = std::move(opened.value());
    REQUIRE(session != nullptr);
    LegacyContext legacy;
    std::set<int64_t> session_labels;
    std::set<int64_t> legacy_labels;
    for (int batch = 0; batch < 3; ++batch) {
        INFO("batch=" << batch);
        REQUIRE(session->HasMore());
        auto actual = session->Next(kBatch);
        auto expected =
            index->KnnSearch(query, kBatch, kSearch, vsag::FilterPtr(nullptr), legacy.value, false);
        REQUIRE(actual.has_value());
        REQUIRE(expected.has_value());
        REQUIRE(actual.value()->GetDim() == kBatch);
        // Compare measured recall below, not an algorithm-specific sequence.
        for (int64_t i = 0; i < kBatch; ++i) {
            REQUIRE(session_labels.insert(actual.value()->GetIds()[i]).second);
            REQUIRE(legacy_labels.insert(expected.value()->GetIds()[i]).second);
        }
    }
    // For this one-dimensional dataset the exact nearest 24 labels are known.
    int session_hits = 0;
    int legacy_hits = 0;
    for (int64_t i = 0; i < 3 * kBatch; ++i) {
        session_hits += session_labels.count(1000 + 3 * i);
        legacy_hits += legacy_labels.count(1000 + 3 * i);
    }
    const double session_recall = session_hits / static_cast<double>(3 * kBatch);
    const double legacy_recall = legacy_hits / static_cast<double>(3 * kBatch);
    REQUIRE(session_recall >= legacy_recall);
}

TEST_CASE("HGraph search session splits pending batch without skipping or duplicate labels",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value), kBatch, kSearch);
    REQUIRE(opened.has_value());
    auto session = std::move(opened.value());
    std::vector<int64_t> expected_ids;
    std::vector<float> expected_distances;
    {
        LegacyContext legacy;
        for (int batch = 0; batch < 2; ++batch) {
            auto expected = index->KnnSearch(MakeSessionQuery(value),
                                             kBatch,
                                             kSearch,
                                             vsag::FilterPtr(nullptr),
                                             legacy.value,
                                             false);
            REQUIRE(expected.has_value());
            REQUIRE(expected.value()->GetDim() == kBatch);
            for (int64_t i = 0; i < kBatch; ++i) {
                expected_ids.push_back(expected.value()->GetIds()[i]);
                expected_distances.push_back(expected.value()->GetDistances()[i]);
            }
        }
    }
    std::set<int64_t> labels;
    int64_t offset = 0;
    // The first two calls split the initial eight candidates; the third resumes search.
    for (int64_t requested : {3, 5, 8}) {
        INFO("requested=" << requested << ", offset=" << offset);
        REQUIRE(session->HasMore());
        auto page = session->Next(requested);
        REQUIRE(page.has_value());
        REQUIRE(page.value() != nullptr);
        REQUIRE(page.value()->GetDim() == requested);
        for (int64_t i = 0; i < requested; ++i) {
            REQUIRE(page.value()->GetIds()[i] == expected_ids[offset + i]);
            REQUIRE(page.value()->GetDistances()[i] == expected_distances[offset + i]);
            REQUIRE(labels.insert(page.value()->GetIds()[i]).second);
        }
        offset += requested;
    }
    REQUIRE(offset == 16);
    REQUIRE(labels.size() == 16);
}

TEST_CASE("HGraph search session Close is idempotent and Next is empty",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value), kBatch, kSearch);
    REQUIRE(opened.has_value());
    auto session = std::move(opened.value());
    auto first = session->Next(kBatch);
    REQUIRE(first.has_value());
    REQUIRE(first.value()->GetDim() == kBatch);
    const auto saved_id = first.value()->GetIds()[0];
    const auto saved_distance = first.value()->GetDistances()[0];
    for (int repeat = 0; repeat < 2; ++repeat) {
        session->Close();
        REQUIRE_FALSE(session->HasMore());
        auto empty = session->Next(kBatch);
        REQUIRE(empty.has_value());
        REQUIRE(empty.value() != nullptr);
        REQUIRE(empty.value()->GetDim() == 0);
        REQUIRE_FALSE(session->HasMore());
    }
    session.reset();
    REQUIRE(first.value()->GetIds()[0] == saved_id);
    REQUIRE(first.value()->GetDistances()[0] == saved_distance);
}

TEST_CASE("HGraph search session empty index is terminal", "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex(false);
    float value = -0.25F;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value), kBatch, kSearch);
    REQUIRE(opened.has_value());
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto empty = opened.value()->Next(kBatch);
        REQUIRE(empty.has_value());
        REQUIRE(empty.value() != nullptr);
        REQUIRE(empty.value()->GetDim() == 0);
        REQUIRE_FALSE(opened.value()->HasMore());
    }
}

TEST_CASE("HGraph search session reject-all filter retains undelivered candidates",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto opened = index->OpenSearchSession(
        MakeSessionQuery(value), kBatch, kSearch, std::make_shared<SessionFilter>(false));
    REQUIRE(opened.has_value());
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto empty = opened.value()->Next(kBatch);
        REQUIRE(empty.has_value());
        REQUIRE(empty.value() != nullptr);
        REQUIRE(empty.value()->GetDim() == 0);
        REQUIRE(opened.value()->HasMore());
    }
}

TEST_CASE("HGraph search session deeply copies query before first Next",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float original = -0.25F;
    auto control = index->OpenSearchSession(MakeSessionQuery(original), kBatch, kSearch);
    REQUIRE(control.has_value());
    std::unique_ptr<vsag::SearchSession> session;
    {
        float mutable_value = original;
        auto query = MakeSessionQuery(mutable_value);
        auto opened = index->OpenSearchSession(query, kBatch, kSearch);
        REQUIRE(opened.has_value());
        session = std::move(opened.value());
        // Mutate both borrowed vector storage and Dataset metadata before Next.
        mutable_value = 127.25F;
        query->Dim(0)->NumElements(0);
        auto actual = session->Next(kBatch);
        auto expected = control.value()->Next(kBatch);
        REQUIRE(actual.has_value());
        REQUIRE(expected.has_value());
        REQUIRE(actual.value()->GetDim() == kBatch);
        RequireSamePage(actual.value(), expected.value());
    }
    // The caller's Dataset and vector storage no longer exist for later pages.
    for (int batch = 1; batch < 3; ++batch) {
        auto actual = session->Next(kBatch);
        auto expected = control.value()->Next(kBatch);
        REQUIRE(actual.has_value());
        REQUIRE(expected.has_value());
        REQUIRE(actual.value()->GetDim() == kBatch);
        RequireSamePage(actual.value(), expected.value());
    }
}

TEST_CASE("HGraph search session retains index and destructor releases ownership",
          "[ut][hgraph][search_session]") {
    // Cover destruction of both an unopened search and a partially consumed one.
    for (bool consume : {false, true}) {
        INFO("consume=" << consume);
        auto index = MakeSessionIndex();
        auto filter = std::make_shared<SessionFilter>(true);
        std::weak_ptr<SessionFilter> weak_filter = filter;
        float value = -0.25F;
        auto opened = index->OpenSearchSession(MakeSessionQuery(value), kBatch, kSearch, filter);
        REQUIRE(opened.has_value());
        auto session = std::move(opened.value());
        std::vector<std::vector<int64_t>> expected_ids;
        std::vector<std::vector<float>> expected_distances;
        if (consume) {
            // Collect the baseline before dropping the public index and destroy
            // its iterator so it cannot accidentally keep session state alive.
            LegacyContext legacy;
            for (int batch = 0; batch < 3; ++batch) {
                auto expected = index->KnnSearch(
                    MakeSessionQuery(value), kBatch, kSearch, filter, legacy.value, false);
                REQUIRE(expected.has_value());
                REQUIRE(expected.value()->GetDim() == kBatch);
                // Copy into standard-library storage: legacy Dataset buffers
                // may use the index allocator and must die before index.reset().
                expected_ids.emplace_back(expected.value()->GetIds(),
                                          expected.value()->GetIds() + kBatch);
                expected_distances.emplace_back(expected.value()->GetDistances(),
                                                expected.value()->GetDistances() + kBatch);
            }
        }
        index.reset();
        filter.reset();
        // The public wrapper may die; the underlying index must remain usable.
        REQUIRE_FALSE(weak_filter.expired());
        vsag::DatasetPtr retained_page;
        if (consume) {
            for (int batch = 0; batch < 3; ++batch) {
                auto page = session->Next(kBatch);
                REQUIRE(page.has_value());
                REQUIRE(page.value()->GetDim() == kBatch);
                for (int64_t i = 0; i < kBatch; ++i) {
                    REQUIRE(page.value()->GetIds()[i] == expected_ids[batch][i]);
                    REQUIRE(page.value()->GetDistances()[i] == expected_distances[batch][i]);
                }
                if (batch == 0) {
                    retained_page = page.value();
                }
            }
        }
        // Deliberately omit Close: the destructor must own all cleanup.
        session.reset();
        REQUIRE(weak_filter.expired());
        if (retained_page) {
            REQUIRE(retained_page->GetDim() == kBatch);
            for (int64_t i = 0; i < kBatch; ++i) {
                REQUIRE(retained_page->GetIds()[i] == expected_ids.front()[i]);
                REQUIRE(retained_page->GetDistances()[i] == expected_distances.front()[i]);
            }
        }
    }
}

namespace {
uint64_t
SessionCounter(const vsag::DatasetPtr& page, const std::string& name) {
    const auto text = page->GetStatistics();
    const auto key = "\"session_" + name + "\":";
    const auto pos = text.find(key);
    REQUIRE(pos != std::string::npos);
    return std::stoull(text.substr(pos + key.size()));
}
}  // namespace

TEST_CASE("HGraph dynamic session rechecks mutable filters and thresholds without traversal",
          "[ut][ft][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value));
    REQUIRE(opened.has_value());
    auto& session = opened.value();
    auto filter = std::make_shared<SessionFilter>(false);
    vsag::SearchSessionNextOptions options{8, kSearch, filter};
    auto rejected = session->Next(options);
    REQUIRE(rejected.has_value());
    REQUIRE(rejected.value()->GetDim() == 0);
    REQUIRE(session->HasMore());
    REQUIRE(SessionCounter(rejected.value(), "frontier_nodes") == 0);
    const auto discovered = SessionCounter(rejected.value(), "undelivered_nodes");
    REQUIRE(discovered > 16);
    REQUIRE(SessionCounter(rejected.value(), "routing_runs") == 1);
    // Same pointer, changed contents: no identity-based eligibility cache.
    filter->SetAccept(true);
    options.search_parameters = R"({"hgraph":{"ef_search":3},"threshold":2.0})";
    auto tight = session->Next(options);
    REQUIRE(tight.has_value());
    REQUIRE(tight.value()->GetDim() == 2);
    std::set<int64_t> seen;
    for (int64_t i = 0; i < tight.value()->GetDim(); ++i) {
        REQUIRE(seen.insert(tight.value()->GetIds()[i]).second);
        REQUIRE(tight.value()->GetDistances()[i] <= 2.0F);
    }
    options.search_parameters = "not-json";
    REQUIRE_FALSE(session->Next(options).has_value());
    options.search_parameters = R"({"hgraph":{"ef_search":3,"parallelism":2}})";
    REQUIRE_FALSE(session->Next(options).has_value());
    REQUIRE(session->HasMore());
    options.search_parameters = kSearch;
    auto wide = session->Next(options);
    REQUIRE(wide.has_value());
    REQUIRE(wide.value()->GetDim() == 8);
    for (int64_t i = 0; i < wide.value()->GetDim(); ++i) {
        REQUIRE(seen.insert(wide.value()->GetIds()[i]).second);
        REQUIRE(wide.value()->GetDistances()[i] > 2.0F);
    }
    // Tightening must discard ready eligibility, not the candidate registry.
    options.search_parameters = R"({"hgraph":{"ef_search":1},"threshold":2.0})";
    auto stricter = session->Next(options);
    REQUIRE(stricter.has_value());
    REQUIRE(stricter.value()->GetDim() == 0);
    REQUIRE(session->HasMore());
    options.search_parameters = kSearch;
    options.filter = std::make_shared<SessionFilter>(false);
    auto changed = session->Next(options);
    REQUIRE(changed.has_value());
    REQUIRE(changed.value()->GetDim() == 0);
    options.filter = nullptr;
    options.max_candidates = kCount;
    auto rest = session->Next(options);
    REQUIRE(rest.has_value());
    for (int64_t i = 0; i < rest.value()->GetDim(); ++i) {
        REQUIRE(seen.insert(rest.value()->GetIds()[i]).second);
    }
    REQUIRE(seen.size() == discovered);
    REQUIRE_FALSE(session->HasMore());
    for (auto page :
         {tight.value(), wide.value(), stricter.value(), changed.value(), rest.value()}) {
        REQUIRE(SessionCounter(page, "round_routing_runs") == 0);
        REQUIRE(SessionCounter(page, "round_computer_creations") == 0);
        REQUIRE(SessionCounter(page, "round_scored") == 0);
        REQUIRE(SessionCounter(page, "round_expanded") == 0);
        REQUIRE(SessionCounter(page, "scored") == SessionCounter(rejected.value(), "scored"));
    }
}

TEST_CASE("HGraph session retires permanent NaN candidates but traverses bridges",
          "[ut][ft][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = std::numeric_limits<float>::quiet_NaN();
    auto opened = index->OpenSearchSession(MakeSessionQuery(value));
    REQUIRE(opened.has_value());
    auto page = opened.value()->Next(vsag::SearchSessionNextOptions{});
    REQUIRE(page.has_value());
    REQUIRE(page.value()->GetDim() == 0);
    REQUIRE(SessionCounter(page.value(), "expanded") > 0);
    REQUIRE(SessionCounter(page.value(), "undelivered_nodes") == 0);
    REQUIRE_FALSE(opened.value()->HasMore());
}

TEST_CASE("HGraph dynamic session changes ef and demand without resetting work",
          "[ut][ft][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value));
    REQUIRE(opened.has_value());
    std::set<int64_t> seen;
    uint64_t total = 0;
    for (uint64_t ef : {4, 9, 3}) {
        vsag::SearchSessionNextOptions options;
        options.max_candidates = ef == 3 ? 6 : 2;
        options.search_parameters = "{\"hgraph\":{\"ef_search\":" + std::to_string(ef) + "}}";
        auto page = opened.value()->Next(options);
        REQUIRE(page.has_value());
        const auto effort = std::max(ef, options.max_candidates);
        total += effort;
        REQUIRE(SessionCounter(page.value(), "round_expanded") == effort);
        REQUIRE(SessionCounter(page.value(), "expanded") == total);
        REQUIRE(SessionCounter(page.value(), "routing_runs") == 1);
        REQUIRE(SessionCounter(page.value(), "round_routing_runs") == (ef == 4 ? 1 : 0));
        REQUIRE(SessionCounter(page.value(), "round_computer_creations") == (ef == 4 ? 1 : 0));
        REQUIRE(SessionCounter(page.value(), "scored") <= kCount);
        for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
            REQUIRE(seen.insert(page.value()->GetIds()[i]).second);
        }
    }
}

TEST_CASE("HGraph native session resumes expansion and reuses scores and computers",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    const auto params = R"({"hgraph":{"ef_search":8,"parallelism":1}})";
    auto opened = index->OpenSearchSession(MakeSessionQuery(value), 4, params);
    REQUIRE(opened.has_value());
    auto session = std::move(opened.value());
    auto first = session->Next(4);
    REQUIRE(first.has_value());
    const auto first_expanded = SessionCounter(first.value(), "expanded");
    REQUIRE(first_expanded == 8);
    REQUIRE(first_expanded < kCount);
    REQUIRE(SessionCounter(first.value(), "routing_runs") == 1);
    REQUIRE(SessionCounter(first.value(), "computers") == 1);
    auto second = session->Next(4);
    REQUIRE(second.has_value());
    REQUIRE(SessionCounter(second.value(), "expanded") == first_expanded + 8);
    REQUIRE(SessionCounter(second.value(), "routing_runs") == 1);
    REQUIRE(SessionCounter(second.value(), "computers") == 1);
    REQUIRE(SessionCounter(second.value(), "scored") >= SessionCounter(first.value(), "scored"));
    std::set<int64_t> ids;
    for (auto page : {first.value(), second.value()}) {
        for (int64_t i = 0; i < page->GetDim(); ++i) {
            REQUIRE(ids.insert(page->GetIds()[i]).second);
        }
    }
    uint64_t calls = 2;
    uint64_t scored = 0;
    uint64_t expanded_total = 0;
    while (session->HasMore()) {
        REQUIRE(++calls <= kCount);
        auto page = session->Next(4);
        REQUIRE(page.has_value());
        scored = SessionCounter(page.value(), "scored");
        expanded_total = SessionCounter(page.value(), "expanded");
        REQUIRE(scored <= kCount);  // Each internal id is scored at most once, including routing.
        REQUIRE(SessionCounter(page.value(), "routing_runs") == 1);
        REQUIRE(SessionCounter(page.value(), "computers") == 1);
        for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
            REQUIRE(ids.insert(page.value()->GetIds()[i]).second);
        }
    }
    // HGraph can contain vertices outside the routed entry's directed component.
    // Every discovered bottom vertex must be expanded and emitted exactly once;
    // routing may additionally score vertices outside that component.
    REQUIRE(ids.size() > 16);
    REQUIRE(expanded_total == ids.size());
    REQUIRE(scored >= ids.size());
    auto terminal = session->Next(4);
    REQUIRE(terminal.has_value());
    REQUIRE(terminal.value()->GetDim() == 0);
}

TEST_CASE("HGraph native session threshold and invalid options", "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto query = MakeSessionQuery(value);
    for (const auto* params : {R"({"hgraph":{"ef_search":8,"parallelism":2}})",
                               R"({"hgraph":{"ef_search":8,"brute_force_threshold":0.5}})",
                               R"({"hgraph":{"ef_search":8,"rabitq_one_bit_search":true}})"}) {
        REQUIRE_FALSE(index->OpenSearchSession(query, 4, params).has_value());
    }
    auto opened =
        index->OpenSearchSession(query, 4, R"({"hgraph":{"ef_search":8},"threshold":2.0})");
    REQUIRE(opened.has_value());
    auto first = opened.value()->Next(4);
    REQUIRE(first.has_value());
    REQUIRE(first.value()->GetDim() == 2);
    for (int64_t i = 0; i < first.value()->GetDim(); ++i) {
        REQUIRE(first.value()->GetDistances()[i] <= 2.0F);
    }
    REQUIRE(opened.value()->HasMore());
}

TEST_CASE("HGraph native session retains precise reorder computers and result storage",
          "[ut][hgraph][search_session]") {
    auto made = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32","metric_type":"l2","dim":8,
        "index_param":{"base_quantization_type":"sq8","use_reorder":true,
        "precise_quantization_type":"fp32","max_degree":16,"ef_construction":128}
    })");
    REQUIRE(made.has_value());
    auto index = made.value();
    made.value().reset();
    std::vector<float> data(kCount * 8);
    std::vector<int64_t> ids(kCount);
    for (int64_t i = 0; i < kCount; ++i) {
        ids[i] = 2000 + i;
        for (int64_t j = 0; j < 8; ++j) {
            data[i * 8 + j] = static_cast<float>(i) * 0.1F + static_cast<float>(j) * 0.03F;
        }
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(kCount)
                    ->Dim(8)
                    ->Float32Vectors(data.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    std::vector<float> query_data(8, -0.1F);
    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(8)
                     ->Float32Vectors(query_data.data())
                     ->Owner(false);
    auto dynamic = index->OpenSearchSession(query);
    REQUIRE(dynamic.has_value());
    vsag::SearchSessionNextOptions options{
        4, R"({"hgraph":{"ef_search":128,"enable_reorder":false},"threshold":-1})", nullptr};
    auto coarse = dynamic.value()->Next(options);
    REQUIRE(coarse.has_value());
    REQUIRE(coarse.value()->GetDim() == 0);
    REQUIRE(SessionCounter(coarse.value(), "computers") == 1);
    options.search_parameters =
        R"({"hgraph":{"ef_search":2,"enable_reorder":true},"threshold":-1})";
    auto precise = dynamic.value()->Next(options);
    REQUIRE(precise.has_value());
    REQUIRE(precise.value()->GetDim() == 0);
    REQUIRE(SessionCounter(precise.value(), "round_computer_creations") == 1);
    REQUIRE(SessionCounter(precise.value(), "round_reordered") > 0);
    // Independent ordinary KNN oracle uses the base cell, not session caches.
    auto coarse_oracle =
        index->KnnSearch(query, kCount, R"({"hgraph":{"ef_search":128,"enable_reorder":false}})");
    REQUIRE(coarse_oracle.has_value());
    bool witnessed_coarse_difference = false;
    for (bool reorder : {false, true}) {
        options.search_parameters = reorder
                                        ? R"({"hgraph":{"ef_search":1,"enable_reorder":true}})"
                                        : R"({"hgraph":{"ef_search":1,"enable_reorder":false}})";
        auto page = dynamic.value()->Next(options);
        REQUIRE(page.has_value());
        REQUIRE(page.value()->GetDim() == 4);
        REQUIRE(SessionCounter(page.value(), "round_reordered") == 0);
        REQUIRE(SessionCounter(page.value(), "round_scored") == 0);
        REQUIRE(SessionCounter(page.value(), "round_expanded") == 0);
        REQUIRE(SessionCounter(page.value(), "round_routing_runs") == 0);
        REQUIRE(SessionCounter(page.value(), "round_computer_creations") == 0);
        for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
            const auto label = page.value()->GetIds()[i];
            const auto id = label - 2000;
            REQUIRE(id >= 0);
            REQUIRE(id < kCount);
            float exact = 0;
            for (int64_t j = 0; j < 8; ++j) {
                const auto delta = data[id * 8 + j] - query_data[j];
                exact += delta * delta;
            }
            if (reorder) {
                REQUIRE(std::abs(page.value()->GetDistances()[i] - exact) < 0.001F);
            } else {
                const auto* begin = coarse_oracle.value()->GetIds();
                const auto* end = begin + coarse_oracle.value()->GetDim();
                const auto* found = std::find(begin, end, label);
                REQUIRE(found != end);
                const auto expected = coarse_oracle.value()->GetDistances()[found - begin];
                REQUIRE(page.value()->GetDistances()[i] == expected);
                witnessed_coarse_difference |= std::abs(expected - exact) > 0.001F;
            }
        }
    }
    REQUIRE(witnessed_coarse_difference);  // An always-precise implementation must fail.
    coarse_oracle.value().reset();         // Ordinary KNN buffers use the index allocator.
    auto opened = index->OpenSearchSession(query, 4, R"({"hgraph":{"ef_search":8}})");
    REQUIRE(opened.has_value());
    auto session = std::move(opened.value());
    dynamic.value().reset();
    index.reset();
    vsag::DatasetPtr retained;
    uint64_t previous = 0;
    for (int call = 0; call < 3; ++call) {
        auto page = session->Next(4);
        REQUIRE(page.has_value());
        REQUIRE(page.value()->GetDim() == 4);
        REQUIRE(SessionCounter(page.value(), "computers") == 2);
        REQUIRE(SessionCounter(page.value(), "routing_runs") == 1);
        REQUIRE(SessionCounter(page.value(), "expanded") > previous);
        previous = SessionCounter(page.value(), "expanded");
        REQUIRE(SessionCounter(page.value(), "reordered") <= kCount);
        for (int64_t i = 0; i < 4; ++i) {
            const auto id = page.value()->GetIds()[i] - 2000;
            REQUIRE(id >= 0);
            REQUIRE(id < kCount);
            float exact = 0;
            for (int64_t j = 0; j < 8; ++j) {
                const float delta = data[id * 8 + j] - query_data[j];
                exact += delta * delta;
            }
            REQUIRE(std::abs(page.value()->GetDistances()[i] - exact) < 0.001F);
        }
        retained = page.value();
    }
    const auto saved = retained->GetDistances()[0];
    session->Close();
    session.reset();
    REQUIRE(retained->GetDistances()[0] == saved);
}

TEST_CASE("HGraph native session measured multidimensional recall versus legacy",
          "[ft][hgraph][search_session]") {
    constexpr int64_t count = 1024;
    constexpr int64_t dim = 16;
    constexpr int64_t page_size = 10;
    auto made = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32","metric_type":"l2","dim":16,
        "index_param":{"base_quantization_type":"fp32","use_reorder":false,
        "max_degree":24,"ef_construction":160}})");
    REQUIRE(made.has_value());
    auto index = made.value();
    std::mt19937 random(424242);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::vector<float> data(count * dim);
    std::vector<int64_t> labels(count);
    for (auto& component : data) {
        component = uniform(random);
    }
    for (int64_t i = 0; i < count; ++i) {
        labels[i] = i;
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(count)
                    ->Dim(dim)
                    ->Ids(labels.data())
                    ->Float32Vectors(data.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    uint64_t native_hits = 0;
    uint64_t legacy_hits = 0;
    constexpr uint64_t queries = 8;
    for (uint64_t q = 0; q < queries; ++q) {
        std::vector<float> vector(dim);
        for (auto& component : vector) {
            component = uniform(random);
        }
        std::vector<std::pair<float, int64_t>> exact;
        for (int64_t id = 0; id < count; ++id) {
            float distance = 0;
            for (int64_t j = 0; j < dim; ++j) {
                const float delta = data[id * dim + j] - vector[j];
                distance += delta * delta;
            }
            exact.emplace_back(distance, id);
        }
        std::sort(exact.begin(), exact.end());
        std::set<int64_t> truth;
        for (int64_t i = 0; i < 3 * page_size; ++i) {
            truth.insert(exact[i].second);
        }
        auto query = vsag::Dataset::Make()
                         ->NumElements(1)
                         ->Dim(dim)
                         ->Float32Vectors(vector.data())
                         ->Owner(false);
        const auto params = R"({"hgraph":{"ef_search":64,"parallelism":1}})";
        using Clock = std::chrono::steady_clock;
        const auto open_start = Clock::now();
        auto opened = index->OpenSearchSession(query, page_size, params);
        const double open_us =
            std::chrono::duration<double, std::micro>(Clock::now() - open_start).count();
        std::cout << "PERF_OPEN q=" << q << " us=" << open_us << '\n';
        REQUIRE(opened.has_value());
        LegacyContext legacy;
        std::set<int64_t> native_seen;
        for (uint64_t call = 0; call < 3; ++call) {
            tl::expected<vsag::DatasetPtr, vsag::Error> page;
            tl::expected<vsag::DatasetPtr, vsag::Error> old;
            double native_us = 0;
            double legacy_us = 0;
            auto run_native = [&]() {
                const auto start = Clock::now();
                page = opened.value()->Next(page_size);
                native_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            };
            auto run_legacy = [&]() {
                const auto start = Clock::now();
                old = index->KnnSearch(
                    query, page_size, params, vsag::FilterPtr(nullptr), legacy.value, false);
                legacy_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            };
            if (q % 2 == 0) {
                run_native();
                run_legacy();
            } else {
                run_legacy();
                run_native();
            }
            REQUIRE(page.has_value());
            REQUIRE(old.has_value());
            REQUIRE(page.value()->GetDim() == page_size);
            REQUIRE(old.value()->GetDim() == page_size);
            std::cout << "PERF_ROUND q=" << q << " round=" << call << " native_us=" << native_us
                      << " legacy_us=" << legacy_us
                      << " native_stats=" << page.value()->GetStatistics()
                      << " legacy_stats=" << old.value()->GetStatistics() << '\n';
            REQUIRE(SessionCounter(page.value(), "expanded") == (call + 1) * 64);
            REQUIRE(SessionCounter(page.value(), "round_expanded") == 64);
            REQUIRE(SessionCounter(page.value(), "round_routing_runs") == (call == 0 ? 1 : 0));
            REQUIRE(SessionCounter(page.value(), "round_computer_creations") ==
                    (call == 0 ? 1 : 0));

            REQUIRE(SessionCounter(page.value(), "routing_runs") == 1);
            REQUIRE(SessionCounter(page.value(), "computers") == 1);
            for (int64_t i = 0; i < page_size; ++i) {
                REQUIRE(native_seen.insert(page.value()->GetIds()[i]).second);
                native_hits += truth.count(page.value()->GetIds()[i]);
                legacy_hits += truth.count(old.value()->GetIds()[i]);
            }
        }
    }
    const double denominator = queries * 3 * page_size;
    std::cout << "Native session recall@30=" << native_hits / denominator
              << " legacy=" << legacy_hits / denominator << '\n';
    REQUIRE(native_hits / denominator >= 0.90);
    REQUIRE(native_hits + 5 >= legacy_hits);
}

namespace {
class NativeEvenFilter : public vsag::Filter {
public:
    bool
    CheckValid(int64_t label) const override {
        return label % 2 == 0;
    }
    bool
    CheckValid(const char* extra) const override {
        return extra[0] % 2 == 0;
    }
    float
    ValidRatio() const override {
        return 0.5F;
    }
};
}  // namespace

TEST_CASE("HGraph native session preserves label and extra-info filtering and bytes",
          "[ut][hgraph][search_session]") {
    auto made = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32","metric_type":"l2","dim":1,"extra_info_size":1,
        "index_param":{"base_quantization_type":"fp32","use_reorder":false,
        "max_degree":16,"ef_construction":128}})");
    REQUIRE(made.has_value());
    auto index = made.value();
    made.value().reset();
    std::vector<float> data(kCount);
    std::vector<int64_t> labels(kCount);
    std::vector<char> extra(kCount);
    for (int64_t i = 0; i < kCount; ++i) {
        data[i] = static_cast<float>(i);
        labels[i] = 1000 + i;
        extra[i] = static_cast<char>(i % 100);
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(kCount)
                    ->Dim(1)
                    ->Ids(labels.data())
                    ->Float32Vectors(data.data())
                    ->ExtraInfos(extra.data())
                    ->ExtraInfoSize(1)
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    float value = -0.25F;
    auto filter = std::make_shared<NativeEvenFilter>();
    auto label_session = index->OpenSearchSession(
        MakeSessionQuery(value), 4, R"({"hgraph":{"ef_search":8}})", filter);
    auto extra_session =
        index->OpenSearchSession(MakeSessionQuery(value),
                                 4,
                                 R"({"hgraph":{"ef_search":8,"use_extra_info_filter":true}})",
                                 filter);
    REQUIRE(label_session.has_value());
    REQUIRE(extra_session.has_value());
    index.reset();
    vsag::DatasetPtr retained;
    for (uint64_t call = 0; call < 3; ++call) {
        auto a = label_session.value()->Next(4);
        auto b = extra_session.value()->Next(4);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        REQUIRE(a.value()->GetDim() == 4);
        RequireSamePage(a.value(), b.value());
        REQUIRE(b.value()->GetExtraInfoSize() == 1);
        for (int64_t i = 0; i < 4; ++i) {
            REQUIRE(b.value()->GetIds()[i] % 2 == 0);
            REQUIRE(b.value()->GetExtraInfos()[i] == extra[b.value()->GetIds()[i] - 1000]);
        }
        retained = b.value();
    }
    const char saved = retained->GetExtraInfos()[0];
    label_session.value().reset();
    extra_session.value().reset();
    REQUIRE(retained->GetExtraInfos()[0] == saved);
}

TEST_CASE("HGraph native session ordinary duplicate labels preserve extra info",
          "[ut][hgraph][search_session]") {
    auto made = vsag::Factory::CreateIndex("hgraph", R"({
        "dtype":"float32","metric_type":"l2","dim":1,"extra_info_size":1,
        "index_param":{"base_quantization_type":"fp32","use_reorder":false,
        "support_duplicate":true,"max_degree":16,"ef_construction":128}})");
    REQUIRE(made.has_value());
    auto index = made.value();
    std::vector<float> data(32, 1.0F);
    std::vector<int64_t> labels(32);
    std::vector<char> extra(32);
    for (int64_t i = 0; i < 32; ++i) {
        labels[i] = 1000 + i;
        extra[i] = static_cast<char>(i);
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(32)
                    ->Dim(1)
                    ->Ids(labels.data())
                    ->Float32Vectors(data.data())
                    ->ExtraInfos(extra.data())
                    ->ExtraInfoSize(1)
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    float value = 0;
    auto opened =
        index->OpenSearchSession(MakeSessionQuery(value),
                                 3,
                                 R"({"hgraph":{"ef_search":8,"use_extra_info_filter":true}})",
                                 std::make_shared<NativeEvenFilter>());
    REQUIRE(opened.has_value());
    std::set<int64_t> seen;
    while (opened.value()->HasMore()) {
        auto page = opened.value()->Next(3);
        REQUIRE(page.has_value());
        if (page.value()->GetDim() == 0) {
            break;  // Current filter is exhausted, but rejected aliases remain available.
        }
        for (int64_t i = 0; i < page.value()->GetDim(); ++i) {
            const auto label = page.value()->GetIds()[i];
            REQUIRE(seen.insert(label).second);
            REQUIRE(label % 2 == 0);
            REQUIRE(page.value()->GetExtraInfos()[i] == label - 1000);
            REQUIRE(page.value()->GetDistances()[i] == 1.0F);
        }
    }
    REQUIRE(seen.size() == 16);
    REQUIRE(opened.value()->HasMore());
    auto remaining = opened.value()->Next(vsag::SearchSessionNextOptions{32, kSearch, nullptr});
    REQUIRE(remaining.has_value());
    REQUIRE(remaining.value()->GetDim() == 16);
    REQUIRE(SessionCounter(remaining.value(), "round_scored") == 0);
    REQUIRE(SessionCounter(remaining.value(), "round_expanded") == 0);
    for (int64_t i = 0; i < remaining.value()->GetDim(); ++i) {
        const auto label = remaining.value()->GetIds()[i];
        REQUIRE(seen.insert(label).second);
        REQUIRE(label % 2 == 1);
        REQUIRE(remaining.value()->GetExtraInfos()[i] == label - 1000);
    }
    REQUIRE_FALSE(opened.value()->HasMore());
}

TEST_CASE("HGraph native session invalid demand is retryable and skip keys are parsed",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = -0.25F;
    auto query = MakeSessionQuery(value);
    REQUIRE_FALSE(
        index->OpenSearchSession(query, 4, R"({"hgraph":{"ef_search":8,"skip_\u0072atio":0.2}})")
            .has_value());
    auto opened =
        index->OpenSearchSession(query, 4, R"({"hgraph":{"ef_search":8},"note":"skip_ratio"})");
    REQUIRE(opened.has_value());
    auto invalid = opened.value()->Next(0);
    REQUIRE_FALSE(invalid.has_value());
    REQUIRE(opened.value()->HasMore());
    auto first = opened.value()->Next(4);
    REQUIRE(first.has_value());
    REQUIRE(first.value()->GetDim() == 4);
    REQUIRE(SessionCounter(first.value(), "round_computer_creations") == 1);
    REQUIRE(SessionCounter(first.value(), "round_routing_runs") == 1);
    REQUIRE_FALSE(opened.value()->Next(0).has_value());
    auto second = opened.value()->Next(4);
    REQUIRE(second.has_value());
    REQUIRE(SessionCounter(second.value(), "round_computer_creations") == 0);
    REQUIRE(SessionCounter(second.value(), "round_routing_runs") == 0);
}

TEST_CASE("HGraph native routing crosses cyclic unrankable bridges",
          "[ut][hgraph][search_session]") {
    std::vector<std::vector<vsag::InnerIdType>> graph{{1}, {0, 2}, {1, 3}, {2}};
    std::vector<float> scores{std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::quiet_NaN(),
                              0.25F};
    std::vector<uint64_t> calls(4);
    auto entry = vsag::FindFiniteSessionRoute(
        0,
        [&](vsag::InnerIdType id) {
            ++calls[id];
            return scores[id];
        },
        [&](vsag::InnerIdType id) { return graph[id]; });
    REQUIRE(entry == 3);
    for (auto count : calls) {
        REQUIRE(count == 1);
    }
    scores[3] = std::numeric_limits<float>::infinity();
    REQUIRE(vsag::FindFiniteSessionRoute(
                0,
                [&](vsag::InnerIdType id) { return scores[id]; },
                [&](vsag::InnerIdType id) { return graph[id]; }) == 0);
}

namespace {
class SessionCountingAllocator : public vsag::Allocator {
public:
    uint64_t live{0};
    uint64_t allocations{0};
    std::string
    Name() override {
        return "session-test-counting";
    }
    void*
    Allocate(uint64_t size) override {
        void* p = std::malloc(size == 0 ? 1 : size);
        if (p == nullptr) {
            throw std::bad_alloc();
        }
        ++live;
        ++allocations;
        return p;
    }
    void
    Deallocate(void* p) override {
        if (p != nullptr) {
            --live;
            std::free(p);
        }
    }
    void*
    Reallocate(void* p, uint64_t size) override {
        if (p == nullptr) {
            return Allocate(size);
        }
        void* result = std::realloc(p, size == 0 ? 1 : size);
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        return result;
    }
};
}  // namespace

TEST_CASE("HGraph native session external allocator releases before results die",
          "[ut][hgraph][search_session]") {
    vsag::DatasetPtr retained;
    int64_t saved_id = 0;
    float saved_distance = 0;
    {
        SessionCountingAllocator allocator;
        auto index = MakeSessionIndex();
        float value = -0.25F;
        auto opened = index->OpenSearchSession(
            MakeSessionQuery(value), 4, R"({"hgraph":{"ef_search":8}})", nullptr, &allocator);
        REQUIRE(opened.has_value());
        REQUIRE(allocator.allocations > 0);
        REQUIRE(allocator.live > 0);  // Deep query copy stays alive until Close.
        index.reset();
        auto page = opened.value()->Next(4);
        REQUIRE(page.has_value());
        REQUIRE(page.value()->GetDim() == 4);
        retained = page.value();
        saved_id = retained->GetIds()[0];
        saved_distance = retained->GetDistances()[0];
        opened.value()->Close();
        REQUIRE(allocator.live == 0);
        REQUIRE_FALSE(opened.value()->HasMore());
        auto terminal = opened.value()->Next(0);
        REQUIRE(terminal.has_value());
        REQUIRE(terminal.value()->GetDim() == 0);
        opened.value().reset();
        REQUIRE(allocator.live == 0);
    }
    // The caller allocator object itself no longer exists.
    REQUIRE(retained->GetIds()[0] == saved_id);
    REQUIRE(retained->GetDistances()[0] == saved_distance);
    retained.reset();
}

TEST_CASE("HGraph native reject-all empty page retains traversal statistics",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = 0;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value),
                                           4,
                                           R"({"hgraph":{"ef_search":8}})",
                                           std::make_shared<SessionFilter>(false));
    REQUIRE(opened.has_value());
    auto page = opened.value()->Next(4);
    REQUIRE(page.has_value());
    REQUIRE(page.value()->GetDim() == 0);
    REQUIRE(SessionCounter(page.value(), "expanded") > 0);
    REQUIRE(SessionCounter(page.value(), "scored") > 0);
    REQUIRE(SessionCounter(page.value(), "round_computer_creations") == 1);
    REQUIRE(opened.value()->HasMore());
    auto terminal = opened.value()->Next(4);
    REQUIRE(terminal.has_value());
    REQUIRE(terminal.value()->GetDim() == 0);
}

TEST_CASE("HGraph native nonfinite priorities match legacy bridge and result eligibility",
          "[ut][hgraph][search_session]") {
    vsag::InnerSearchParam params;
    for (float distance :
         {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        REQUIRE(vsag::is_result_distance_eligible<vsag::KNN_SEARCH>(distance, params));
        REQUIRE(-vsag::traversal_priority(distance) < -vsag::traversal_priority(0.25F));
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(vsag::is_result_distance_eligible<vsag::KNN_SEARCH>(nan, params));
    REQUIRE(-vsag::traversal_priority(nan) < -vsag::traversal_priority(0.25F));
    params.distance_threshold = 1.0F;
    REQUIRE_FALSE(vsag::is_result_distance_eligible<vsag::KNN_SEARCH>(
        std::numeric_limits<float>::infinity(), params));
    REQUIRE_FALSE(vsag::is_result_distance_eligible<vsag::KNN_SEARCH>(
        -std::numeric_limits<float>::infinity(), params));
}

static_assert(not std::is_constructible_v<vsag::HGraphContinueSession,
                                          const vsag::HGraph&,
                                          std::shared_ptr<const vsag::InnerIndexInterface>,
                                          vsag::DatasetPtr,
                                          std::string,
                                          vsag::FilterPtr,
                                          vsag::Allocator*>);

TEST_CASE("HGraph session safety guards reject unrepresentable bytes and invalid ids",
          "[ut][hgraph][search_session]") {
    const uint64_t limit = std::numeric_limits<std::size_t>::max();
    REQUIRE(vsag::CheckedSessionBytes(0, limit) == 0);
    REQUIRE(vsag::CheckedSessionBytes(limit, 0) == 0);
    REQUIRE(vsag::CheckedSessionBytes(limit / 8, 8) == (limit / 8) * 8);
    REQUIRE_THROWS_AS(vsag::CheckedSessionBytes(limit / 8 + 1, 8), vsag::VsagException);
    REQUIRE_THROWS_AS(vsag::CheckedSessionBytes(2, limit), vsag::VsagException);
    REQUIRE_NOTHROW(vsag::ValidateSessionId(3, 4));
    REQUIRE_THROWS_AS(vsag::ValidateSessionId(4, 4), vsag::VsagException);
    REQUIRE_THROWS_AS(vsag::ValidateSessionId(0, 0), vsag::VsagException);
    REQUIRE_THROWS_AS(vsag::ValidateSessionId(std::numeric_limits<vsag::InnerIdType>::max(), 4),
                      vsag::VsagException);
}

namespace {
void
RequireIdleStatistics(const vsag::DatasetPtr& idle, const vsag::DatasetPtr& previous) {
    REQUIRE(idle->GetDim() == 0);
    for (const auto* key : {"routing_runs", "computers", "scored", "reordered", "expanded"}) {
        REQUIRE(SessionCounter(idle, key) == SessionCounter(previous, key));
    }
    for (const auto* key : {"round_routing_runs",
                            "round_computer_creations",
                            "round_scored",
                            "round_reordered",
                            "round_expanded"}) {
        REQUIRE(SessionCounter(idle, key) == 0);
    }
}

class NonstandardThrowFilter : public SessionFilter {
public:
    NonstandardThrowFilter() : SessionFilter(true) {
    }
    bool
    CheckValid(int64_t) const override {
        throw 7;
    }
};
}  // namespace

TEST_CASE("HGraph session terminal and closed pages preserve cumulative statistics",
          "[ut][hgraph][search_session]") {
    for (bool populate : {false, true}) {
        auto index = MakeSessionIndex(populate);
        float value = 0;
        auto opened = index->OpenSearchSession(MakeSessionQuery(value), 4, kSearch);
        REQUIRE(opened.has_value());
        auto& session = opened.value();
        auto page = session->Next(kCount);
        REQUIRE(page.has_value());
        REQUIRE_FALSE(session->HasMore());
        for (int repeat = 0; repeat < 2; ++repeat) {
            auto terminal = session->Next(0);
            REQUIRE(terminal.has_value());
            RequireIdleStatistics(terminal.value(), page.value());
        }
        session->Close();
        session->Close();
        auto closed = session->Next(0);
        REQUIRE(closed.has_value());
        RequireIdleStatistics(closed.value(), page.value());
        REQUIRE(SessionCounter(closed.value(), "state_payload_bytes") == 0);
        REQUIRE(SessionCounter(closed.value(), "frontier_nodes") == 0);
        REQUIRE(SessionCounter(closed.value(), "pending_nodes") == 0);
    }
    auto index = MakeSessionIndex();
    float value = 0;
    auto opened = index->OpenSearchSession(MakeSessionQuery(value), 4, kSearch);
    REQUIRE(opened.has_value());
    opened.value()->Close();
    auto closed = opened.value()->Next(0);
    REQUIRE(closed.has_value());
    REQUIRE(SessionCounter(closed.value(), "scored") == 0);
    REQUIRE(SessionCounter(closed.value(), "computers") == 0);
    RequireIdleStatistics(closed.value(), closed.value());
}

TEST_CASE("HGraph session nonstandard traversal exception closes and retains work counts",
          "[ut][hgraph][search_session]") {
    auto index = MakeSessionIndex();
    float value = 0;
    auto opened = index->OpenSearchSession(
        MakeSessionQuery(value), 4, kSearch, std::make_shared<NonstandardThrowFilter>());
    REQUIRE(opened.has_value());
    auto failed = opened.value()->Next(4);
    REQUIRE_FALSE(failed.has_value());
    REQUIRE(failed.error().type == vsag::ErrorType::INTERNAL_ERROR);
    REQUIRE_FALSE(opened.value()->HasMore());
    auto closed = opened.value()->Next(0);
    REQUIRE(closed.has_value());
    REQUIRE(SessionCounter(closed.value(), "scored") > 0);
    REQUIRE(SessionCounter(closed.value(), "computers") == 1);
    REQUIRE(SessionCounter(closed.value(), "state_payload_bytes") == 0);
    RequireIdleStatistics(closed.value(), closed.value());
}
