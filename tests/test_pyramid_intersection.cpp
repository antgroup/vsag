// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <future>
#include <nlohmann/json.hpp>
#include <sstream>

#include "vsag/vsag.h"

namespace {

struct IntersectionFixture {
    static constexpr int64_t DIM = 16;
    std::vector<int64_t> ids{100, 101, 102, 103, 104, 105};
    std::vector<float> vectors = std::vector<float>(6 * DIM, 0.0F);
    std::vector<std::string> site{"a/one", "b/one", "a/one", "b/two", "a/two", "c"};
    std::vector<std::string> category{"x/one", "y/one", "y/two", "x/two", "y/two", "z"};
    std::vector<std::string> region{"south", "south", "south", "north", "north", "north"};
    std::vector<float> query_vector = std::vector<float>(DIM, 0.0F);
    std::string site_query{"a"};
    std::string category_query{"y"};
    std::string region_query{"north"};

    IntersectionFixture() {
        for (uint64_t i = 0; i < ids.size(); ++i) {
            vectors[i * DIM] = static_cast<float>(i + 1);
        }
    }

    vsag::DatasetPtr
    base(uint64_t begin = 0, uint64_t count = 6) {
        return vsag::Dataset::Make()
            ->NumElements(static_cast<int64_t>(count))
            ->Dim(DIM)
            ->Ids(ids.data() + begin)
            ->Float32Vectors(vectors.data() + begin * DIM)
            ->Paths("site", site.data() + begin)
            ->Paths("category", category.data() + begin)
            ->Paths("region", region.data() + begin)
            ->Owner(false);
    }

    vsag::DatasetPtr
    query() {
        return vsag::Dataset::Make()
            ->NumElements(1)
            ->Dim(DIM)
            ->Float32Vectors(query_vector.data())
            ->Paths("site", &site_query)
            ->Paths("category", &category_query)
            ->Paths("region", &region_query)
            ->Owner(false);
    }

    static nlohmann::json
    params(const std::string& graph = "nsw", int min_size = 100, bool reorder = false) {
        return {{"dtype", "float32"},
                {"metric_type", "l2"},
                {"dim", DIM},
                {"index_param",
                 {{"graph_type", graph},
                  {"max_degree", 16},
                  {"ef_construction", 100},
                  {"index_min_size", min_size},
                  {"use_reorder", reorder},
                  {"base_quantization_type", reorder ? "sq8" : "fp32"},
                  {"precise_quantization_type", "fp32"},
                  {"hierarchies",
                   {{{"name", "site"}, {"no_build_levels", {0}}},
                    {{"name", "category"}, {"no_build_levels", {0, 1}}},
                    {{"name", "region"}, {"no_build_levels", {0}}}}}}}};
    }

    static std::string
    search_params(int ef = 10) {
        return nlohmann::json{{"pyramid",
                               {{"ef_search", ef},
                                {"factor", 1.5},
                                {"hierarchies", {"site", "category"}},
                                {"hierarchy_op", "intersection"}}}}
            .dump();
    }

    static void
    check(const tl::expected<vsag::DatasetPtr, vsag::Error>& result,
          const std::vector<int64_t>& expected) {
        if (not result.has_value()) {
            FAIL(result.error().message);
        }
        REQUIRE(result.value()->GetDim() == static_cast<int64_t>(expected.size()));
        for (uint64_t i = 0; i < expected.size(); ++i) {
            REQUIRE(result.value()->GetIds()[i] == expected[i]);
        }
    }

    void
    check_entries(const vsag::IndexPtr& index,
                  const std::vector<int64_t>& expected,
                  const std::string& parameters = search_params(),
                  const vsag::FilterPtr& filter = nullptr,
                  int64_t limit = 6) {
        CAPTURE(site_query, category_query, region_query, parameters, limit, expected);
        auto q = query();
        check(index->KnnSearch(q, limit, parameters, filter), expected);
        check(index->RangeSearch(q, 100.0F, parameters, filter, limit), expected);
        vsag::SearchRequest request;
        request.query_ = q;
        request.params_str_ = parameters;
        request.topk_ = limit;
        request.filter_ = filter;
        request.enable_filter_ = filter != nullptr;
        check(index->SearchWithRequest(request), expected);
        request.mode_ = vsag::SearchMode::RANGE_SEARCH;
        request.radius_ = 100.0F;
        request.limited_size_ = limit;
        check(index->SearchWithRequest(request), expected);
    }
};

class ExcludeLabel : public vsag::Filter {
public:
    bool
    CheckValid(int64_t id) const override {
        return id != 102;
    }
};

}  // namespace

TEST_CASE("Pyramid intersection applies scopes before candidate limits",
          "[ft][pyramid][intersection]") {
    const auto graph = GENERATE("nsw", "odescent");
    const auto min_size = GENERATE(0, 100);
    const auto reorder = GENERATE(false, true);
    CAPTURE(graph, min_size, reorder);
    IntersectionFixture f;
    auto created = vsag::Factory::CreateIndex("pyramid", f.params(graph, min_size, reorder).dump());
    REQUIRE(created.has_value());
    auto index = created.value();
    REQUIRE(index->Build(f.base()).has_value());

    // Independent Top-1 results are 100 and 101; intersecting them loses the valid best match 102.
    f.check(index->KnnSearch(f.query(), 1, R"({"pyramid":{"ef_search":1,"hierarchies":["site"]}})"),
            {100});
    f.check(
        index->KnnSearch(f.query(), 1, R"({"pyramid":{"ef_search":1,"hierarchies":["category"]}})"),
        {101});
    f.check_entries(index, {102}, f.search_params(1), nullptr, 1);
    f.check_entries(index, {102, 104});
    f.check(index->RangeSearch(f.query(), 10.0F, f.search_params()), {102});
    vsag::SearchRequest threshold_request;
    threshold_request.query_ = f.query();
    threshold_request.topk_ = 6;
    threshold_request.params_str_ = f.search_params();
    threshold_request.threshold_ = 10.0F;
    f.check(index->SearchWithRequest(threshold_request), {102});
    f.check_entries(index, {104}, f.search_params(), std::make_shared<ExcludeLabel>());

    // OR on both sides, including duplicate/overlapping prefixes and nonexistent alternatives.
    f.site_query = "a/one|b/two|a/one|missing";
    f.category_query = "y|x/two|y/two";
    f.check_entries(index, {102, 103});
    f.site_query = "c";
    f.category_query = "y";
    f.check_entries(index, {});
    f.site_query = "missing";
    f.check_entries(index, {});
    f.site_query = "";
    f.check_entries(index, {});
    f.site_query = "a";
    f.category_query = "";
    f.check_entries(index, {});
    f.category_query = "y";

    auto three = nlohmann::json::parse(f.search_params());
    three["pyramid"]["hierarchies"] = {"site", "category", "region"};
    f.check_entries(index, {104}, three.dump());
    three["pyramid"]["hierarchies"] = {"region", "category", "site"};
    f.check_entries(index, {104}, three.dump());

    // Scope membership is restored independently of store_paths.
    std::stringstream stream;
    REQUIRE(index->Serialize(stream).has_value());
    auto restored =
        vsag::Factory::CreateIndex("pyramid", f.params(graph, min_size, reorder).dump());
    REQUIRE(restored.has_value());
    REQUIRE(restored.value()->Deserialize(stream).has_value());
    f.check_entries(restored.value(), {102, 104});
    std::stringstream streaming;
    REQUIRE(index->SerializeStreaming(streaming).has_value());
    auto streamed =
        vsag::Factory::CreateIndex("pyramid", f.params(graph, min_size, reorder).dump());
    REQUIRE(streamed.has_value());
    REQUIRE(streamed.value()->DeserializeStreaming(streaming).has_value());
    f.check_entries(streamed.value(), {102, 104});
    // MarkRemove does not change physical graph membership; its filter must still apply.
    for (const auto& target : {index, restored.value()}) {
        auto removed = target->Remove(std::vector<int64_t>{102}, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(removed.has_value());
        REQUIRE(removed.value() == 1);
        f.check_entries(target, {104});
    }
}

TEST_CASE("Pyramid intersection validates every named path", "[ft][pyramid][intersection]") {
    IntersectionFixture f;
    auto index = vsag::Factory::CreateIndex("pyramid", f.params().dump()).value();
    REQUIRE(index->Build(f.base()).has_value());
    const auto invalid = GENERATE("missing_path", "unknown", "union", "missing_first");
    auto q = f.query();
    auto params = nlohmann::json::parse(f.search_params());
    if (std::string(invalid) == "missing_path" || std::string(invalid) == "missing_first") {
        q = vsag::Dataset::Make()
                ->NumElements(1)
                ->Dim(f.DIM)
                ->Float32Vectors(f.query_vector.data())
                ->Paths(&f.category_query)
                ->Owner(false);
        if (std::string(invalid) == "missing_path") {
            q->Paths("site", &f.site_query);
        } else {
            q->Paths("category", &f.category_query);
        }
    } else if (std::string(invalid) == "unknown") {
        // An empty first scope must not hide an invalid later selector.
        f.site_query = "missing";
        params["pyramid"]["hierarchies"] = {"site", "unknown"};
    } else {
        params["pyramid"]["hierarchy_op"] = "union";
    }
    const auto sp = params.dump();
    const auto require_invalid = [](const auto& result) {
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    };
    require_invalid(index->KnnSearch(q, 1, sp));
    require_invalid(index->RangeSearch(q, 100.0F, sp));
    vsag::SearchRequest request;
    request.query_ = q;
    request.topk_ = 1;
    request.params_str_ = sp;
    require_invalid(index->SearchWithRequest(request));
    request.mode_ = vsag::SearchMode::RANGE_SEARCH;
    request.radius_ = 100.0F;
    require_invalid(index->SearchWithRequest(request));
}

TEST_CASE("Pyramid intersection follows Add and FLAT promotion", "[ft][pyramid][intersection]") {
    IntersectionFixture f;
    auto param = f.params("nsw", 3);
    auto index = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(index->Build(f.base(0, 2)).has_value());
    f.check_entries(index, {});
    REQUIRE(index->Add(f.base(2, 3)).has_value());
    f.check_entries(index, {102, 104});
    // A label inserted into only one hierarchy must not satisfy the other scope.
    auto partial = vsag::Dataset::Make()
                       ->NumElements(1)
                       ->Dim(f.DIM)
                       ->Ids(f.ids.data() + 5)
                       ->Float32Vectors(f.vectors.data() + 5 * f.DIM)
                       ->Paths("site", &f.site_query)
                       ->Owner(false);
    REQUIRE(index->Add(partial).has_value());
    f.check_entries(index, {102, 104});
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    f.check_entries(restored, {102, 104});
}

TEST_CASE("Pyramid intersection supports root scopes and partial hierarchy membership",
          "[ft][pyramid][intersection]") {
    const auto storage = GENERATE("flat", "compressed");
    CAPTURE(storage);
    IntersectionFixture f;
    auto param = f.params("nsw", 0);
    param["index_param"]["hierarchies"][1] = {{"name", "category"},
                                              {"root_graph_type", "multi_layer"},
                                              {"no_build_levels", nlohmann::json::array()}};
    param["index_param"]["graph_storage_type"] = storage;
    auto made = vsag::Factory::CreateIndex("pyramid", param.dump());
    REQUIRE(made.has_value());
    auto index = made.value();
    auto partial = vsag::Dataset::Make()
                       ->NumElements(2)
                       ->Dim(f.DIM)
                       ->Ids(f.ids.data())
                       ->Float32Vectors(f.vectors.data())
                       ->Paths("site", f.site.data())
                       ->Owner(false);
    REQUIRE(index->Build(partial).has_value());
    REQUIRE(index->Add(f.base(2, 1)).has_value());
    f.category_query = "/";
    // The singleton category root is a real zero-degree member, despite earlier ID holes.
    f.check_entries(index, {102});
    REQUIRE(index->Remove(std::vector<int64_t>{102}).value() == 1);
    f.check_entries(index, {});
    REQUIRE(index->Add(f.base(3, 3)).has_value());
    f.check_entries(index, {104});
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    // Existing serialization does not retain MarkRemove flags; reapply the deletion explicitly.
    REQUIRE(restored->Remove(std::vector<int64_t>{102}).value() == 1);
    f.check_entries(restored, {104});
}

TEST_CASE("Pyramid intersection includes graph duplicates", "[ft][pyramid][intersection]") {
    IntersectionFixture f;
    f.vectors[2 * f.DIM] = f.vectors[f.DIM];
    f.site_query = "a/one";
    f.category_query = "y/one";
    f.category[2] = "y/one";
    auto param = f.params("nsw", 0);
    param["index_param"]["support_duplicate"] = true;
    auto index = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(index->Build(f.base()).has_value());
    // 102 has no physical row in category/y/one: it is a duplicate of 101.
    f.check_entries(index, {102});
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    f.check_entries(restored, {102});
}

TEST_CASE("Pyramid intersection filters RaBitQ reorder and request bitsets",
          "[ft][pyramid][intersection]") {
    IntersectionFixture f;
    auto param = f.params("nsw", 0, true);
    param["index_param"]["base_quantization_type"] = "rabitq";
    param["index_param"]["precise_quantization_type"] = "rabitq";
    param["index_param"]["rabitq_bits_per_dim_base"] = 3;
    param["index_param"]["rabitq_bits_per_dim_precise"] = 5;
    auto index = vsag::Factory::CreateIndex("pyramid", param.dump()).value();
    REQUIRE(index->Build(f.base()).has_value());
    f.check_entries(index, {102, 104});
    vsag::SearchRequest request;
    request.query_ = f.query();
    request.params_str_ = f.search_params();
    request.topk_ = 1;
    request.enable_bitset_filter_ = true;
    request.bitset_filter_ = vsag::Bitset::Make();
    request.bitset_filter_->Set(102);
    f.check(index->SearchWithRequest(request), {104});
    request.mode_ = vsag::SearchMode::RANGE_SEARCH;
    request.radius_ = 100;
    f.check(index->SearchWithRequest(request), {104});
}

TEST_CASE("Pyramid intersection supports concurrent opposite selectors and parallel OR",
          "[ft][pyramid][intersection]") {
    IntersectionFixture f;
    auto pool = vsag::Engine::CreateThreadPool(4).value();
    auto allocator = vsag::Engine::CreateDefaultAllocator();
    vsag::Resource resource(allocator, pool);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("pyramid", f.params("nsw", 0).dump()).value();
    REQUIRE(index->Build(f.base()).has_value());
    f.site_query = "a|a/one|a/two";
    auto forward = nlohmann::json::parse(f.search_params());
    forward["pyramid"]["parallel_search_thread_count"] = 2;
    auto reverse = forward;
    reverse["pyramid"]["hierarchies"] = {"category", "site"};
    f.check_entries(index, {102, 104}, forward.dump());
    f.check_entries(index, {102, 104}, reverse.dump());
    auto q = f.query();
    auto search = [&](const std::string& parameters) {
        for (int i = 0; i < 30; ++i) {
            const auto result = index->KnnSearch(q, 2, parameters);
            if (not result.has_value() || result.value()->GetDim() != 2 ||
                result.value()->GetIds()[0] != 102 || result.value()->GetIds()[1] != 104) {
                return false;
            }
        }
        return true;
    };
    auto a = std::async(std::launch::async, search, forward.dump());
    auto b = std::async(std::launch::async, search, reverse.dump());
    REQUIRE(a.get());
    REQUIRE(b.get());
}
