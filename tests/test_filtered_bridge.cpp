
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "filtered_bridge_fixture.h"
#include "vsag/vsag.h"

namespace {
class EligibleLabels : public vsag::Filter {
public:
    EligibleLabels(std::vector<int64_t> labels, float ratio)
        : labels_(std::move(labels)), ratio_(ratio) {
    }

    bool
    CheckValid(int64_t label) const override {
        return std::find(labels_.begin(), labels_.end(), label) != labels_.end();
    }

    float
    ValidRatio() const override {
        return ratio_;
    }

private:
    std::vector<int64_t> labels_;
    float ratio_;
};
}  // namespace

TEST_CASE("Public filtered KNN agrees with exact search on small connected indexes",
          "[filtered_bridge_api]") {
    const auto route =
        GENERATE(std::string("hnsw"), std::string("hgraph"), std::string("hgraph_parallel"));
    const std::string backend = route == "hnsw" ? "hnsw" : "hgraph";
    const int parallelism = route == "hgraph_parallel" ? 2 : 1;
    const auto seed = GENERATE(0, 17, 3290);
    const auto degree = GENERATE(4, 16);
    const auto topk = GENERATE(3, 9);
    const std::string parameters =
        std::string("{\"dtype\":\"float32\",\"metric_type\":\"l2\",\"dim\":4,\"") +
        (backend == "hnsw" ? "hnsw" : "index_param") +
        "\":{\"max_degree\":" + std::to_string(degree) + ",\"ef_construction\":200" +
        (backend == "hgraph" ? ",\"base_quantization_type\":\"fp32\",\"build_thread_count\":1"
                             : "") +
        "}}";
    const std::string search_parameters =
        "{\"" + backend + "\":{\"ef_search\":512" +
        (backend == "hgraph" ? ",\"parallelism\":" + std::to_string(parallelism) : "") + "}}";
    std::vector<int64_t> ids(32);
    std::iota(ids.begin(), ids.end(), 1);
    if (seed != 0) {
        std::shuffle(ids.begin(), ids.end(), std::mt19937(seed));
    }
    std::vector<float> vectors(128);
    for (int i = 0; i < 32; ++i) {
        std::copy(fixtures::FILTERED_BRIDGE_VECTORS[ids[i] - 1],
                  fixtures::FILTERED_BRIDGE_VECTORS[ids[i] - 1] + 4,
                  vectors.data() + i * 4);
    }
    // Factory uses a resource without a thread pool. Provision one explicitly
    // through Engine for the public parallel search route.
    auto pool = vsag::Engine::CreateThreadPool(2);
    REQUIRE(pool.has_value());
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), pool.value());
    vsag::Engine engine(&resource);
    auto create_index = [&]() {
        return route == "hgraph_parallel" ? engine.CreateIndex(backend, parameters)
                                          : vsag::Factory::CreateIndex(backend, parameters);
    };
    auto index_result = create_index();
    REQUIRE(index_result.has_value());
    auto index = index_result.value();
    auto base = vsag::Dataset::Make();
    base->NumElements(32)->Dim(4)->Ids(ids.data())->Float32Vectors(vectors.data())->Owner(false);
    auto built = index->Build(base);
    REQUIRE(built.has_value());
    REQUIRE(built->empty());

    // Repeat on the same graph after the ordinary serialized round trip.
    for (bool reload : {false, true}) {
        if (reload) {
            auto binary = index->Serialize();
            REQUIRE(binary.has_value());
            auto restored = create_index();
            REQUIRE(restored.has_value());
            index = restored.value();
            REQUIRE(index->Deserialize(binary.value()).has_value());
        }
        for (float q : {0.0F, 1.5F, 3.0F}) {
            float query_data[] = {q, q, q, q};
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(4)->Float32Vectors(query_data)->Owner(false);
            for (const auto& eligible : std::vector<std::vector<int64_t>>{
                     {21, 23, 24, 25, 26, 28, 29, 30, 31}, {}, {32}, {1, 8, 20, 32}}) {
                std::vector<std::pair<float, int64_t>> exact;
                for (int i = 0; i < 32; ++i) {
                    if (std::find(eligible.begin(), eligible.end(), ids[i]) == eligible.end()) {
                        continue;
                    }
                    float distance = 0.0F;
                    for (int d = 0; d < 4; ++d) {
                        float delta = vectors[i * 4 + d] - q;
                        distance += delta * delta;
                    }
                    exact.emplace_back(distance, ids[i]);
                }
                std::sort(exact.begin(), exact.end());
                exact.resize(std::min<size_t>(topk, exact.size()));
                for (float ratio : {static_cast<float>(eligible.size()) / 32.0F, 1.0F}) {
                    vsag::FilterPtr filter = std::make_shared<EligibleLabels>(eligible, ratio);
                    auto result = index->KnnSearch(query, topk, search_parameters, filter);
                    INFO(route << " seed=" << seed << " degree=" << degree << " q=" << q
                               << " ratio=" << ratio << " reload=" << reload);
                    REQUIRE(result.has_value());
                    REQUIRE(result.value()->GetDim() == exact.size());
                    REQUIRE(result.value()->GetNumElements() == 1);
                    if (exact.empty()) {
                        REQUIRE(result.value()->GetIds() == nullptr);
                        REQUIRE(result.value()->GetDistances() == nullptr);
                    }
                    std::set<int64_t> unique;
                    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
                        auto id = result.value()->GetIds()[i];
                        REQUIRE(unique.insert(id).second);
                        REQUIRE(filter->CheckValid(id));
                        REQUIRE(id == exact[i].second);
                        REQUIRE(result.value()->GetDistances()[i] ==
                                Catch::Approx(exact[i].first).margin(1e-5));
                    }
                }
            }
        }
    }
}
