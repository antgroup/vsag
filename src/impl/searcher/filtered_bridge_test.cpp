
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

#include <catch2/catch_approx.hpp>

#include "basic_searcher.h"
#include "filtered_bridge_fixture.h"
#include "parallel_searcher.h"
#include "searcher_test.h"

using namespace vsag;

namespace {
class BridgeFilter : public Filter {
public:
    BridgeFilter(std::vector<int64_t> labels, float ratio)
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

TEST_CASE("Filtered search navigates nonmatching bridges", "[ut][filtered_bridge]") {
    // Both neighbors of entry 16 are ineligible. Neighbor 17 is the sole bridge
    // to the matching component. Stamping it visited before rejecting it loses
    // that entire component, even when ef exceeds the connected graph size.
    const auto backend = GENERATE(0, 1, 2);
    const auto topology = GENERATE(0, 1, 2);
    const auto query_value = GENERATE(0.0F, 1.5F, 3.0F);
    const float query[] = {query_value, query_value, query_value, query_value};
    const auto strategy = GENERATE(FilterSearchSkipStrategyType::DETERMINISTIC_ACCUMULATIVE,
                                   FilterSearchSkipStrategyType::RANDOM);
    const auto skip_ratio = GENERATE(0.0F, 0.7F, 1.0F);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IndexCommonParam common;
    common.dim_ = 4;
    common.allocator_ = allocator;
    common.metric_ = MetricType::METRIC_TYPE_L2SQR;

    std::vector<float> vectors;
    for (const auto& vector : fixtures::FILTERED_BRIDGE_VECTORS) {
        vectors.insert(vectors.end(), vector, vector + 4);
    }
    std::vector<InnerIdType> ids(32);
    std::iota(ids.begin(), ids.end(), 0);
    std::vector<std::vector<InnerIdType>> edges(32);
    for (InnerIdType i = 0; i < 32; ++i) {
        if (i > 0) {
            edges[i].push_back(i - 1);
        }
        if (i + 1 < 32) {
            edges[i].push_back(i + 1);
        }
        if (topology == 1) {
            std::reverse(edges[i].begin(), edges[i].end());
        } else if (topology == 2 && i + 2 < 32 && i != 16) {
            edges[i].push_back(i + 2);
        }
    }

    hnswlib::L2Space space(4);
    auto native = std::make_shared<hnswlib::HierarchicalNSW>(&space, 32, allocator.get(), 4, 32);
    REQUIRE(native->init_memory_space());
    for (InnerIdType i = 0; i < 32; ++i) {
        native->addPoint(vectors.data() + i * 4, i, 0);
    }
    for (InnerIdType i = 0; i < 32; ++i) {
        auto* links = native->get_linklist0(i);
        native->setListCount(links, edges[i].size());
        std::copy(edges[i].begin(), edges[i].end(), links + 1);
    }

    auto quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(JsonType::Parse(R"({"type":"fp32"})"));
    auto io_param = IOParameter::GetIOParameterByJson(JsonType::Parse(R"({"type":"memory_io"})"));
    auto flatten =
        std::make_shared<FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            quantizer_param, io_param, common);
    flatten->Train(vectors.data(), ids.size());
    flatten->BatchInsertVector(vectors.data(), ids.size(), ids.data());
    auto graph = std::make_shared<MockGraphDataCell>(edges);
    VisitedListPool pool(1, allocator.get(), 32, allocator.get());
    auto thread_pool = std::make_shared<SafeThreadPool>(new DefaultThreadPool(2), true);

    for (const auto& labels : std::vector<std::vector<int64_t>>{
             {20, 22, 23, 24, 25, 27, 28, 29, 30}, {}, {31}, {0, 7, 19, 31}}) {
        for (float ratio : {static_cast<float>(labels.size()) / 32.0F, 1.0F}) {
            auto filter = std::make_shared<BridgeFilter>(labels, ratio);
            std::vector<std::pair<float, InnerIdType>> expected;
            for (auto label : labels) {
                float distance = 0.0F;
                for (int d = 0; d < 4; ++d) {
                    float delta = vectors[label * 4 + d] - query[d];
                    distance += delta * delta;
                }
                expected.emplace_back(distance, label);
            }
            std::sort(expected.begin(), expected.end());
            std::vector<std::pair<float, InnerIdType>> actual;
            if (backend == 0) {
                auto result = native->searchBaseLayerST<false, false>(
                    16, query, size_t{512}, filter, skip_ratio, strategy);
                while (not result.empty()) {
                    actual.push_back(result.top());
                    result.pop();
                }
            } else {
                InnerSearchParam param;
                param.ep = 16;
                param.ef = 512;
                param.topk = 32;
                param.is_inner_id_allowed = filter;
                param.skip_ratio = skip_ratio;
                param.skip_strategy_type = strategy;
                param.parallel_search_thread_count = 2;
                auto visited = pool.TakeOne();
                auto result =
                    backend == 1
                        ? BasicSearcher(common).Search(
                              graph, flatten, visited, query, param, LabelTablePtr{}, nullptr)
                        : ParallelSearcher(common, thread_pool)
                              .Search(
                                  graph, flatten, visited, query, param, LabelTablePtr{}, nullptr);
                pool.ReturnOne(visited);
                while (not result->Empty()) {
                    actual.push_back(result->Top());
                    result->Pop();
                }
            }
            std::sort(actual.begin(), actual.end());
            INFO("backend=" << backend << " topology=" << topology << " ratio=" << ratio
                            << " query=" << query_value << " skip_ratio=" << skip_ratio);
            REQUIRE(actual.size() == expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                REQUIRE(actual[i].second == expected[i].second);
                REQUIRE(actual[i].first == Catch::Approx(expected[i].first).margin(1e-5));
            }
        }
    }
}

TEST_CASE("Filtered native HNSW across construction seeds", "[ut][filtered_bridge]") {
    const auto seed = GENERATE(1, 17, 3290);
    const auto degree = GENERATE(4, 16);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    hnswlib::L2Space space(4);
    hnswlib::HierarchicalNSW native(
        &space, 32, allocator.get(), degree, 200, false, false, 128 * 1024 * 1024, seed);
    REQUIRE(native.init_memory_space());
    for (int i = 0; i < 32; ++i) {
        REQUIRE(native.addPoint(fixtures::FILTERED_BRIDGE_VECTORS[i], i + 1));
    }
    for (float value : {0.0F, 1.5F, 3.0F}) {
        const float query[] = {value, value, value, value};
        for (const auto& labels : std::vector<std::vector<int64_t>>{
                 {21, 23, 24, 25, 26, 28, 29, 30, 31}, {}, {32}, {1, 8, 20, 32}}) {
            std::vector<std::pair<float, int64_t>> expected;
            for (auto label : labels) {
                float distance = 0.0F;
                for (int d = 0; d < 4; ++d) {
                    float delta = fixtures::FILTERED_BRIDGE_VECTORS[label - 1][d] - query[d];
                    distance += delta * delta;
                }
                expected.emplace_back(distance, label);
            }
            std::sort(expected.begin(), expected.end());
            for (float ratio : {static_cast<float>(labels.size()) / 32.0F, 1.0F}) {
                auto filter = std::make_shared<BridgeFilter>(labels, ratio);
                auto result = native.searchKnn(query, 9, 512, filter);
                std::vector<std::pair<float, int64_t>> actual;
                while (not result.empty()) {
                    actual.push_back(result.top());
                    result.pop();
                }
                std::sort(actual.begin(), actual.end());
                INFO("seed=" << seed << " degree=" << degree << " query=" << value
                             << " ratio=" << ratio);
                REQUIRE(actual.size() == expected.size());
                for (size_t i = 0; i < expected.size(); ++i) {
                    REQUIRE(actual[i].second == expected[i].second);
                    REQUIRE(actual[i].first == Catch::Approx(expected[i].first).margin(1e-5));
                }
            }
        }
    }
}
