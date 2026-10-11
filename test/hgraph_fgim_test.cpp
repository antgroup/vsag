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

#include "algorithm/hgraph/hgraph_fgim.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>

#include "algorithm/hgraph/hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"
#include "unittest.h"
#include "vsag/dataset.h"

namespace vsag {

// Test-only access remains independent of the fixture name.
class HGraphFGIMTestAccessor {
public:
    void
    CheckDuplicateSource(const HGraph& source) {
        REQUIRE(source.support_duplicate_);
        bool found = false;
        for (InnerIdType u = 0; u < source.GetNumElements(); ++u) {
            found = found || !source.bottom_graph_->GetDuplicateIds(u).empty();
        }
        REQUIRE(found);
    }

    struct Snapshot {
        std::vector<LabelType> labels;
        std::vector<float> vectors;
        std::vector<std::vector<InnerIdType>> neighbors;
        int64_t removed;
    };

    Snapshot
    Capture(const HGraph& source) {
        Snapshot snapshot;
        snapshot.removed = source.GetNumberRemoved();
        const auto count = source.GetNumElements();
        snapshot.vectors.resize(count * source.dim_);
        for (InnerIdType u = 0; u < count; ++u) {
            snapshot.labels.push_back(source.label_table_->GetLabelById(u));
            source.GetVectorByInnerId(u, snapshot.vectors.data() + u * source.dim_);
            Vector<InnerIdType> neighbors(source.allocator_);
            source.bottom_graph_->GetNeighbors(u, neighbors);
            snapshot.neighbors.emplace_back(neighbors.begin(), neighbors.end());
        }
        return snapshot;
    }

    FGIMNeighborList
    Expected(const Vector<const HGraph*>& sources,
             const std::vector<InnerIdType>& offsets,
             uint64_t source_index,
             InnerIdType u,
             uint64_t k) {
        const auto& source = *sources[source_index];
        Vector<InnerIdType> original(source.allocator_);
        source.bottom_graph_->GetNeighbors(u, original);
        FGIMNeighborList expected;
        for (const auto v : original) {
            expected.push_back(
                {offsets[source_index] + v, source.basic_flatten_codes_->ComputePairVectors(u, v)});
        }
        std::vector<float> vector(source.dim_);
        source.GetVectorByInnerId(u, vector.data());
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(source.dim_)->Float32Vectors(vector.data())->Owner(false);
        // Small test values allow the direct paper formula, independent of production arithmetic.
        const int64_t l = (k + sources.size() - 2) / (sources.size() - 1);
        const auto parameters = fmt::format(R"({{"hgraph":{{"ef_search":{}}}}})", l);
        for (uint64_t j = 0; j < sources.size(); ++j) {
            if (j == source_index) {
                continue;
            }
            const auto result = sources[j]->KnnSearch(query, l, parameters, nullptr);
            REQUIRE(result->GetDim() <= l);
            for (int64_t n = 0; n < result->GetDim(); ++n) {
                const auto label = result->GetIds()[n];
                REQUIRE(label >= 1000);
                const auto local_id = sources[j]->label_table_->GetIdByLabel(label);
                REQUIRE(local_id < sources[j]->GetNumElements());
                expected.push_back({offsets[j] + local_id, result->GetDistances()[n]});
            }
        }
        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return std::tie(a.distance, a.id) < std::tie(b.distance, b.id);
        });
        if (expected.size() > k) {
            expected.resize(k);
        }
        return expected;
    }

    void
    SetUnsupported(HGraph& graph, const std::string& condition) {
        if (condition == "conjugate") {
            graph.use_conjugate_graph_ = true;
        } else if (condition == "non-float32") {
            graph.data_type_ = DataTypes::DATA_TYPE_INT8;
        } else if (condition == "non-L2") {
            graph.metric_ = MetricType::METRIC_TYPE_IP;
        } else if (condition == "dimensions") {
            ++graph.dim_;
        } else if (condition == "deduplicate") {
            graph.deduplicate_storage_ = true;
        } else if (condition == "base storage") {
            graph.basic_flatten_codes_ = nullptr;
        } else if (condition == "fully built") {
            graph.bottom_graph_->SetTotalCount(0);
        } else if (condition == "missing label table") {
            graph.label_table_ = nullptr;
        }
    }

    void
    CompareCross(const HGraph& target, const std::vector<float>& vector, int64_t l) {
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(target.dim_)->Float32Vectors(vector.data())->Owner(false);
        const auto parameters = fmt::format(R"({{"hgraph":{{"ef_search":{}}}}})", l);
        const auto public_result = target.KnnSearch(query, l, parameters, nullptr);
        FGIMNeighborList expected;
        for (int64_t n = 0; n < public_result->GetDim(); ++n) {
            expected.push_back({target.label_table_->GetIdByLabel(public_result->GetIds()[n]),
                                public_result->GetDistances()[n]});
        }
        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
            return std::tie(a.distance, a.id) < std::tie(b.distance, b.id);
        });
        SearchStatistics stats;
        const auto counted = HGraphFGIM::CrossQuery(target, vector.data(), l, &stats);
        const auto fast = HGraphFGIM::CrossQuery(target, vector.data(), l);
        REQUIRE(fast.size() == expected.size());
        REQUIRE(counted.size() == expected.size());
        std::set<InnerIdType> seen;
        for (uint64_t i = 0; i < fast.size(); ++i) {
            REQUIRE(fast[i].id < target.GetNumElements());
            REQUIRE(seen.insert(fast[i].id).second);
            REQUIRE(fast[i].id == expected[i].id);
            REQUIRE(fast[i].distance == expected[i].distance);
            REQUIRE(counted[i].id == fast[i].id);
            REQUIRE(counted[i].distance == fast[i].distance);
        }
        const auto public_stats = JsonType::Parse(public_result->GetStatistics());
        REQUIRE(stats.distance_evaluations.load() ==
                public_stats["distance_evaluations"].GetUint64());
    }
    std::vector<std::map<InnerIdType, std::vector<InnerIdType>>>
    CaptureUpper(const HGraph& source) {
        std::vector<std::map<InnerIdType, std::vector<InnerIdType>>> result;
        for (const auto& layer : source.route_graphs_) {
            result.emplace_back();
            auto ids = layer->GetIds();
            REQUIRE(ids.size() == layer->TotalCount());
            for (const auto u : ids) {
                REQUIRE(u < source.GetNumElements());
                Vector<InnerIdType> neighbors(source.allocator_);
                layer->GetNeighbors(u, neighbors);
                REQUIRE(neighbors.size() <= layer->MaximumDegree());
                std::set<InnerIdType> seen;
                for (const auto v : neighbors) {
                    REQUIRE(v != u);
                    REQUIRE(layer->CheckIdExists(v));
                    REQUIRE(seen.insert(v).second);
                }
                result.back()[u] = {neighbors.begin(), neighbors.end()};
                if (result.size() > 1) {
                    REQUIRE(result[result.size() - 2].count(u) == 1);
                }
            }
        }
        REQUIRE(source.entry_point_id_ < source.GetNumElements());
        if (!result.empty()) {
            REQUIRE(result.back().count(source.entry_point_id_) == 1);
        }
        return result;
    }
};

class HGraphFGIMTest : public HGraphFGIMTestAccessor {
public:
    HGraphFGIMTest() {
        common.dim_ = 4;
        common.metric_ = MetricType::METRIC_TYPE_L2SQR;
        common.data_type_ = DataTypes::DATA_TYPE_FLOAT;
        common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    }

    std::unique_ptr<HGraph>
    MakeSource(uint64_t count,
               uint64_t source_index,
               bool tied = false,
               const std::string& base_quantization = "fp32",
               bool random = false,
               bool support_duplicate = false,
               bool fgim = false) {
        auto parameters = JsonType::Parse(R"({
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "build_thread_count": 1,
            "support_duplicate": false,
            "deduplicate_storage": false
        })");
        parameters["base_quantization_type"].SetString(base_quantization);
        parameters["support_duplicate"].SetBool(support_duplicate);
        if (fgim) {
            parameters["merge_strategy"].SetString("fgim");
        }
        auto graph = std::make_unique<HGraph>(
            HGraph::CheckAndMappingExternalParam(parameters, common), common);
        std::vector<float> vectors(count * common.dim_);
        std::vector<int64_t> labels(count);
        std::mt19937 rng(47 + source_index);
        std::uniform_real_distribution<float> distribution(-10.0F, 1400.0F);
        for (uint64_t u = 0; u < count; ++u) {
            // Reverse label order makes label arithmetic an invalid substitute for lookup.
            labels[u] = 1000 * (source_index + 1) + count - 1 - u;
            for (int64_t d = 0; d < common.dim_; ++d) {
                // Interleave sources so cross candidates survive top-k selection.
                vectors[u * common.dim_ + d] =
                    tied ? 0.0F
                         : (random ? distribution(rng)
                                   : static_cast<float>(10 * (support_duplicate ? u / 2 : u) +
                                                        source_index + d));
            }
        }
        auto data = Dataset::Make();
        data->NumElements(count)
            ->Dim(common.dim_)
            ->Ids(labels.data())
            ->Float32Vectors(vectors.data())
            ->Owner(false);
        if (count > 0) {
            REQUIRE(graph->Build(data).empty());
        }
        REQUIRE(graph->GetNumElements() == count);
        return graph;
    }

    void
    ExpectInvalid(const Vector<const HGraph*>& sources, uint64_t k, const std::string& reason) {
        try {
            (void)HGraphFGIM::BuildInitialKnnGraph(sources, k);
            FAIL("unsupported FGIM input was accepted");
        } catch (const VsagException& e) {
            REQUIRE(e.error_.type == ErrorType::INVALID_ARGUMENT);
            REQUIRE(std::string(e.what()).find(reason) != std::string::npos);
        }
    }

    IndexCommonParam common;
};

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM initial k-NNG matches independent candidates",
                 "[ut][hgraph][fgim]") {
    const auto m = GENERATE(2, 3, 4);
    const uint64_t k = GENERATE(1, 3, 8, 100);
    CAPTURE(m, k);
    std::vector<std::unique_ptr<HGraph>> owned;
    Vector<const HGraph*> sources(common.allocator_.get());
    std::vector<InnerIdType> offsets{0};
    std::vector<Snapshot> before;
    for (int i = 0; i < m; ++i) {
        owned.push_back(MakeSource(6 - i, i));
        sources.push_back(owned.back().get());
        offsets.push_back(offsets.back() + sources.back()->GetNumElements());
        before.push_back(Capture(*sources.back()));
    }

    const auto graph = HGraphFGIM::BuildInitialKnnGraph(sources, k);
    REQUIRE(graph.size() == offsets.back());
    uint64_t original_count = 0;
    uint64_t cross_count = 0;
    for (int i = 0; i < m; ++i) {
        for (InnerIdType u = 0; u < sources[i]->GetNumElements(); ++u) {
            const auto merged_u = offsets[i] + u;
            const auto& actual = graph[merged_u];
            const auto expected = Expected(sources, offsets, i, u, k);
            REQUIRE(actual.size() <= k);
            REQUIRE(actual.size() == expected.size());
            std::set<InnerIdType> seen;
            for (uint64_t n = 0; n < actual.size(); ++n) {
                const auto& candidate = actual[n];
                REQUIRE(candidate.id < offsets.back());
                REQUIRE(candidate.id != merged_u);
                REQUIRE(seen.insert(candidate.id).second);
                REQUIRE(candidate.id == expected[n].id);
                REQUIRE(candidate.distance == expected[n].distance);
                if (n != 0) {
                    REQUIRE(std::tie(actual[n - 1].distance, actual[n - 1].id) <=
                            std::tie(candidate.distance, candidate.id));
                }
                if (candidate.id >= offsets[i] && candidate.id < offsets[i + 1]) {
                    ++original_count;
                } else {
                    ++cross_count;
                }
            }
        }
        const auto after = Capture(*sources[i]);
        REQUIRE(after.labels == before[i].labels);
        REQUIRE(after.vectors == before[i].vectors);
        REQUIRE(after.neighbors == before[i].neighbors);
        REQUIRE(after.removed == before[i].removed);
    }
    REQUIRE(cross_count > 0);
    if (k == 100) {
        REQUIRE(original_count > 0);
        REQUIRE(graph.front().size() < k);
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM singleton sources use ceiling and deterministic ties",
                 "[ut][hgraph][fgim]") {
    std::vector<std::unique_ptr<HGraph>> owned;
    Vector<const HGraph*> sources(common.allocator_.get());
    for (uint64_t i = 0; i < 4; ++i) {
        owned.push_back(MakeSource(1, i, true));
        sources.push_back(owned.back().get());
    }
    const auto graph = HGraphFGIM::BuildInitialKnnGraph(sources, 2);
    REQUIRE(graph.size() == 4);
    for (InnerIdType u = 0; u < 4; ++u) {
        REQUIRE(graph[u].size() == 2);
        InnerIdType n = 0;
        for (InnerIdType v = 0; v < 4 && n < 2; ++v) {
            if (v != u) {
                REQUIRE(graph[u][n].id == v);
                REQUIRE(graph[u][n].distance == 0.0F);
                ++n;
            }
        }
    }
    const auto short_graph = HGraphFGIM::BuildInitialKnnGraph(sources, 20);
    for (const auto& neighbors : short_graph) {
        REQUIRE(neighbors.size() == 3);
    }
}

TEST_CASE_METHOD(HGraphFGIMTest, "FGIM rejects unsupported inputs", "[ut][hgraph][fgim]") {
    auto a = MakeSource(3, 0);
    auto b = MakeSource(2, 1);
    Vector<const HGraph*> sources({a.get(), b.get()}, common.allocator_.get());
    SECTION("too few sources") {
        sources.clear();
        ExpectInvalid(sources, 3, "at least two");
        sources.push_back(a.get());
        ExpectInvalid(sources, 3, "at least two");
    }
    SECTION("null source") {
        sources[1] = nullptr;
        ExpectInvalid(sources, 3, "null");
    }
    SECTION("same source twice") {
        sources[1] = a.get();
        ExpectInvalid(sources, 3, "distinct");
    }
    SECTION("invalid k") {
        ExpectInvalid(sources, 0, "k must");
        ExpectInvalid(sources, std::numeric_limits<uint64_t>::max(), "k must");
    }
    SECTION("empty source") {
        auto empty = MakeSource(0, 2);
        sources[1] = empty.get();
        ExpectInvalid(sources, 3, "non-empty");
    }
    SECTION("quantized source") {
        auto quantized = MakeSource(3, 2, false, "sq8");
        sources[1] = quantized.get();
        ExpectInvalid(sources, 3, "fp32 base");
    }
    SECTION("removed source node") {
        REQUIRE(b->Remove({2000}) == 1);
        ExpectInvalid(sources, 3, "deleted");
    }
    SECTION("missing label table") {
        SetUnsupported(*b, "missing label table");
        ExpectInvalid(sources, 3, "fully built");
    }
    SECTION("unsupported source properties") {
        const std::string condition = GENERATE(
            "non-float32", "non-L2", "dimensions", "deduplicate", "base storage", "fully built");
        const std::string expected_diagnostic = condition == "non-float32" ? "float32"
                                                : condition == "non-L2"    ? "L2"
                                                                           : condition;
        CAPTURE(condition);
        // Isolate input validation without requiring an unsupported source build.
        SetUnsupported(*b, condition);
        ExpectInvalid(sources, 3, expected_diagnostic);
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM internal CrossQuery matches public search",
                 "[ut][hgraph][fgim]") {
    const bool tied = GENERATE(false, true);
    common.dim_ = 32;
    auto target = MakeSource(129, 1, tied, "fp32", true);
    const auto before = Capture(*target);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distribution(-10.0F, 1400.0F);
    for (int64_t l : {1, 8, 32, 200}) {
        for (uint64_t q = 0; q < 12; ++q) {
            std::vector<float> vector(common.dim_);
            for (auto& value : vector) {
                value = tied ? 0.0F : distribution(rng);
            }
            CompareCross(*target, vector, l);
        }
    }
    const auto after = Capture(*target);
    REQUIRE(before.labels == after.labels);
    REQUIRE(before.vectors == after.vectors);
    REQUIRE(before.neighbors == after.neighbors);
    REQUIRE(before.removed == after.removed);
    Vector<const HGraph*> sources(common.allocator_.get());
    auto other = MakeSource(2, 2);
    sources.push_back(target.get());
    sources.push_back(other.get());
    SetUnsupported(*target, "conjugate");
    ExpectInvalid(sources, 8, "conjugate");
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM CrossQuery preserves distinct IDs with duplicate vector support",
                 "[ut][hgraph][fgim]") {
    const int64_t l = GENERATE(1, 3, 20);
    CAPTURE(l);
    auto target = MakeSource(6, 0, false, "fp32", false, true);
    CheckDuplicateSource(*target);
    const auto before = Capture(*target);
    REQUIRE(std::set<LabelType>(before.labels.begin(), before.labels.end()).size() ==
            target->GetNumElements());
    for (InnerIdType u = 0; u < target->GetNumElements(); ++u) {
        std::vector<float> query(common.dim_);
        target->GetVectorByInnerId(u, query.data());
        CompareCross(*target, query, l);
    }
    const auto after = Capture(*target);
    REQUIRE(after.labels == before.labels);
    REQUIRE(after.vectors == before.vectors);
    REQUIRE(after.neighbors == before.neighbors);
    REQUIRE(after.removed == before.removed);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM rejects duplicate-supporting sources",
                 "[ut][hgraph][fgim]") {
    const uint64_t k = GENERATE(1, 3, 20);
    const bool tied = GENERATE(false, true);
    CAPTURE(k, tied);
    auto a = MakeSource(6, 0);
    // Reject the configuration even when no duplicate group was formed.
    auto b = MakeSource(6, 1, tied, "fp32", !tied, true);
    if (tied) {
        CheckDuplicateSource(*b);
    }
    Vector<const HGraph*> sources({a.get(), b.get()}, common.allocator_.get());
    ExpectInvalid(sources, k, "duplicate grouping");
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement preserves bounded graph and sources",
                 "[ut][hgraph][fgim]") {
    const uint64_t k = GENERATE(1, 4, 12);
    std::vector<std::unique_ptr<HGraph>> owned;
    Vector<const HGraph*> sources(common.allocator_.get());
    std::vector<Snapshot> snapshots;
    std::vector<float> vectors;
    for (uint64_t i = 0; i < 3; ++i) {
        owned.push_back(MakeSource(12, i, false, "fp32", true));
        sources.push_back(owned.back().get());
        snapshots.push_back(Capture(*owned.back()));
        vectors.insert(
            vectors.end(), snapshots.back().vectors.begin(), snapshots.back().vectors.end());
    }
    auto graph = HGraphFGIM::BuildInitialKnnGraph(sources, k);
    auto repeated = graph;
    const auto stats = HGraphFGIM::RefineKnnGraph(sources, graph, k, 4);
    const auto again = HGraphFGIM::RefineKnnGraph(sources, repeated, k, 4);
    REQUIRE(stats.iterations > 0);
    REQUIRE(stats.iterations <= 4);
    REQUIRE(stats.distance_evaluations == stats.new_new_pairs + stats.new_old_pairs);
    REQUIRE(stats.distance_evaluations == again.distance_evaluations);
    REQUIRE(stats.same_source_pairs_skipped > 0);
    REQUIRE(graph.size() == 36);
    for (uint64_t u = 0; u < graph.size(); ++u) {
        REQUIRE(graph[u].size() <= k);
        REQUIRE(graph[u].size() == repeated[u].size());
        std::set<InnerIdType> seen;
        for (uint64_t i = 0; i < graph[u].size(); ++i) {
            const auto& edge = graph[u][i];
            REQUIRE(edge.id < graph.size());
            REQUIRE(edge.id != u);
            REQUIRE(seen.insert(edge.id).second);
            REQUIRE(edge.id == repeated[u][i].id);
            REQUIRE(edge.distance == repeated[u][i].distance);
            float exact = 0;
            for (int64_t d = 0; d < common.dim_; ++d) {
                const float delta =
                    vectors[u * common.dim_ + d] - vectors[edge.id * common.dim_ + d];
                exact += delta * delta;
            }
            REQUIRE(std::abs(edge.distance - exact) <= 0.00001F * std::max(1.0F, exact));
            if (i > 0) {
                REQUIRE(std::tie(graph[u][i - 1].distance, graph[u][i - 1].id) <=
                        std::tie(edge.distance, edge.id));
            }
        }
    }
    for (uint64_t i = 0; i < sources.size(); ++i) {
        const auto after = Capture(*sources[i]);
        REQUIRE(after.labels == snapshots[i].labels);
        REQUIRE(after.vectors == snapshots[i].vectors);
        REQUIRE(after.neighbors == snapshots[i].neighbors);
        REQUIRE(after.removed == snapshots[i].removed);
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement repairs zero indegree without exceeding k",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0, true);
    auto rhs = MakeSource(2, 1, true);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    FGIMKnnGraph graph{{{1, 0}, {2, 0}}, {{0, 0}, {2, 0}}, {{0, 0}, {1, 0}}, {{0, 0}, {1, 0}}};
    auto unchanged = graph;
    const auto zero = HGraphFGIM::RefineKnnGraph(sources, unchanged, 2, 0);
    REQUIRE(zero.iterations == 0);
    for (uint64_t u = 0; u < graph.size(); ++u) {
        for (uint64_t i = 0; i < graph[u].size(); ++i) {
            REQUIRE(unchanged[u][i].id == graph[u][i].id);
            REQUIRE(unchanged[u][i].distance == graph[u][i].distance);
        }
    }
    const auto stats = HGraphFGIM::RefineKnnGraph(sources, graph, 2, 1);
    REQUIRE(stats.neighbor_updates == 0);  // Equal distances do not replace a full row.
    REQUIRE(stats.indegree_repairs == 1);
    REQUIRE(graph[0][0].id == 1);
    REQUIRE(graph[0][1].id == 3);
    std::vector<uint64_t> indegree(4);
    for (const auto& row : graph) {
        REQUIRE(row.size() == 2);
        for (const auto& edge : row) {
            ++indegree[edge.id];
        }
    }
    REQUIRE(*std::min_element(indegree.begin(), indegree.end()) > 0);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement pairs cross-source neighbors and replaces farthest",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    // Coordinates are [0,10] in lhs and [1,11] in rhs, repeated over four dimensions.
    // Visiting 0 proposes pair (1,3); distance 4 must replace 1->2 (distance 324).
    FGIMKnnGraph graph{{{1, 400}, {3, 484}}, {{2, 324}, {0, 400}}, {{0, 4}, {1, 324}}, {{0, 484}}};
    const auto stats = HGraphFGIM::RefineKnnGraph(sources, graph, 2, 3);
    REQUIRE(stats.new_new_pairs > 0);
    REQUIRE(stats.new_old_pairs > 0);
    REQUIRE(stats.neighbor_updates > 0);
    REQUIRE(graph[1].front().id == 3);
    REQUIRE(graph[1].front().distance == 4);
    REQUIRE(graph[3].front().id == 1);
    REQUIRE(graph[3].front().distance == 4);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement validates intermediate graphs before mutation",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    auto graph = HGraphFGIM::BuildInitialKnnGraph(sources, 2);
    SECTION("node count") {
        graph.pop_back();
    }
    SECTION("degree") {
        graph[0].push_back({3, 484});
    }
    SECTION("self loop") {
        graph[0] = {{0, 0}};
    }
    SECTION("invalid id") {
        graph[0] = {{4, 0}};
    }
    SECTION("duplicate") {
        graph[0] = {{1, 400}, {1, 400}};
    }
    SECTION("unsorted") {
        graph[0] = {{1, 400}, {2, 4}};
    }
    SECTION("nonfinite") {
        graph[0] = {{1, std::numeric_limits<float>::infinity()}};
    }
    SECTION("negative distance") {
        graph[0] = {{1, -1}};
    }
    const auto before = graph;
    REQUIRE_THROWS_AS(HGraphFGIM::RefineKnnGraph(sources, graph, 2, 1), VsagException);
    REQUIRE(graph.size() == before.size());
    for (uint64_t u = 0; u < graph.size(); ++u) {
        REQUIRE(graph[u].size() == before[u].size());
        for (uint64_t i = 0; i < graph[u].size(); ++i) {
            REQUIRE(graph[u][i].id == before[u][i].id);
            REQUIRE(graph[u][i].distance == before[u][i].distance);
        }
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement skips all same-source pairs",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(3, 0);
    auto rhs = MakeSource(3, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    FGIMKnnGraph graph(6);
    for (InnerIdType offset : {0U, 3U}) {
        graph[offset] = {{offset + 1, 400}, {offset + 2, 1600}};
        graph[offset + 1] = {{offset, 400}, {offset + 2, 400}};
        graph[offset + 2] = {{offset + 1, 400}, {offset, 1600}};
    }
    const auto stats = HGraphFGIM::RefineKnnGraph(sources, graph, 2, 8);
    REQUIRE(stats.same_source_pairs_skipped > 0);
    REQUIRE(stats.distance_evaluations == 0);
    REQUIRE(stats.neighbor_updates == 0);
    REQUIRE(stats.indegree_repairs == 0);
    REQUIRE(stats.iterations < 8);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement visits pending reverse-new before convergence",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    // Node 0 has no outgoing edges. Only its incoming edges expose pair (1,2).
    FGIMKnnGraph graph{{}, {{0, 400}}, {{0, 4}}, {}};
    auto one_iteration = graph;
    const auto first = HGraphFGIM::RefineKnnGraph(sources, one_iteration, 1, 1);
    REQUIRE(first.distance_evaluations == 0);
    REQUIRE(first.neighbor_updates == 0);
    REQUIRE(first.indegree_repairs == 0);
    const auto second = HGraphFGIM::RefineKnnGraph(sources, graph, 1, 2);
    REQUIRE(second.iterations == 2);
    REQUIRE(second.new_new_pairs > 0);
    REQUIRE(second.neighbor_updates > 0);
    REQUIRE(graph[1].front().id == 2);
    REQUIRE(graph[1].front().distance == 324);
    REQUIRE(graph[3].empty());  // Repair cannot invent a neighbor for an empty row.
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement discovers an edge exclusively through new-old pairs",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources({lhs.get(), rhs.get()}, common.allocator_.get());
    // Merged coordinates are [0,10,1,11], repeated in four dimensions.
    // The first traversal only compares new-new pair (0,3), which cannot improve a row.
    // On the second traversal, pending reverse-new meets old neighbors: (3,1), then (2,0).
    const FGIMKnnGraph input{{{1, 400}}, {{3, 4}}, {{1, 324}}, {{0, 484}}};
    REQUIRE(input[0].front().id != 2);
    REQUIRE(input[2].front().id != 0);

    auto first = input;
    const auto first_stats = HGraphFGIM::RefineKnnGraph(sources, first, 1, 1);
    REQUIRE(first_stats.new_new_pairs == 1);
    REQUIRE(first_stats.new_old_pairs == 0);
    REQUIRE(first_stats.neighbor_updates == 0);
    REQUIRE(first_stats.indegree_repairs == 0);
    for (uint64_t u = 0; u < input.size(); ++u) {
        REQUIRE(first[u].size() == input[u].size());
        REQUIRE(first[u].front().id == input[u].front().id);
        REQUIRE(first[u].front().distance == input[u].front().distance);
    }

    // Run both iterations in one call so edge flags and reverse state survive.
    auto refined = input;
    const auto stats = HGraphFGIM::RefineKnnGraph(sources, refined, 1, 2);
    REQUIRE(stats.iterations == 2);
    REQUIRE(stats.new_new_pairs == first_stats.new_new_pairs);
    REQUIRE(stats.new_old_pairs == 2);
    REQUIRE(stats.distance_evaluations == 3);
    REQUIRE(stats.indegree_repairs == 0);
    // No additional new-new comparison or repair can explain these newly inserted edges.
    REQUIRE(refined[0].front().id == 2);
    REQUIRE(refined[2].front().id == 0);
    REQUIRE(refined[0].front().distance == 4);
    REQUIRE(refined[2].front().distance == 4);
    for (InnerIdType u = 0; u < refined.size(); ++u) {
        REQUIRE(refined[u].size() == 1);
        const auto v = refined[u].front().id;
        REQUIRE(v < refined.size());
        REQUIRE(v != u);
        REQUIRE((u < 2) != (v < 2));
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM refinement repair skips a farthest sole incoming edge",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(3, 0, true);
    auto rhs = MakeSource(2, 1, true);
    Vector<const HGraph*> sources({lhs.get(), rhs.get()}, common.allocator_.get());
    // All distances are zero and all rows are full, so pair updates cannot alter the fixture.
    // Node 4 has no incoming edge. Its first donor is 1, whose last (farthest-tied) edge
    // 1->3 is the sole incoming edge of 3. Edge 1->2 is a safe alternative.
    FGIMKnnGraph graph{
        {{1, 0}, {2, 0}}, {{2, 0}, {3, 0}}, {{0, 0}, {1, 0}}, {{0, 0}, {2, 0}}, {{1, 0}, {2, 0}}};
    const auto before = graph;
    std::vector<uint64_t> before_indegree(graph.size(), 0);
    for (const auto& row : graph) {
        for (const auto& edge : row) {
            ++before_indegree[edge.id];
        }
    }
    REQUIRE(before_indegree[4] == 0);
    REQUIRE(before_indegree[3] == 1);
    REQUIRE(before_indegree[2] > 1);
    REQUIRE(graph[1].back().id == 3);

    const auto stats = HGraphFGIM::RefineKnnGraph(sources, graph, 2, 1);
    REQUIRE(stats.neighbor_updates == 0);
    REQUIRE(stats.indegree_repairs == 1);
    REQUIRE(graph[1][0].id == 3);
    REQUIRE(graph[1][1].id == 4);
    std::vector<uint64_t> after_indegree(graph.size(), 0);
    for (InnerIdType u = 0; u < graph.size(); ++u) {
        REQUIRE(graph[u].size() == 2);
        REQUIRE(graph[u][0].id < graph[u][1].id);
        for (uint64_t i = 0; i < graph[u].size(); ++i) {
            const auto& edge = graph[u][i];
            REQUIRE(edge.id < graph.size());
            REQUIRE(edge.id != u);
            REQUIRE(edge.distance == 0);
            ++after_indegree[edge.id];
            if (u != 1) {
                REQUIRE(edge.id == before[u][i].id);
            }
        }
    }
    REQUIRE(after_indegree[3] == before_indegree[3]);
    REQUIRE(after_indegree[2] == before_indegree[2] - 1);
    REQUIRE(after_indegree[4] == 1);
    REQUIRE(*std::min_element(after_indegree.begin(), after_indegree.end()) > 0);
}
TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM reconstruction prunes redundant edges and preserves a connected chain",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    const FGIMKnnGraph input{{{2, 4}, {1, 400}, {3, 484}},
                             {{3, 4}, {2, 324}, {0, 400}},
                             {{0, 4}, {1, 324}, {3, 400}},
                             {{1, 4}, {2, 400}, {0, 484}}};
    uint64_t distances = 0;
    const auto result = HGraphFGIM::KnnGraphToPG(sources, input, 3, &distances);
    REQUIRE(distances > 0);
    const std::vector<std::vector<InnerIdType>> expected{{2}, {3, 2}, {0, 1}, {1}};
    for (uint64_t u = 0; u < result.size(); ++u) {
        REQUIRE(result[u].size() == expected[u].size());
        REQUIRE(input[u].size() == 3);
        for (uint64_t i = 0; i < result[u].size(); ++i) {
            REQUIRE(result[u][i].id == expected[u][i]);
        }
    }
    std::set<InnerIdType> visited{0};
    std::vector<InnerIdType> stack{0};
    while (!stack.empty()) {
        const auto u = stack.back();
        stack.pop_back();
        for (const auto& edge : result[u]) {
            if (visited.insert(edge.id).second) {
                stack.push_back(edge.id);
            }
        }
    }
    REQUIRE(visited.size() == result.size());
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM reconstruction protects sole incoming edges and bounds reverse edges",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(2, 0);
    auto rhs = MakeSource(2, 1);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    SECTION("sole incoming edge survives geometric pruning") {
        const FGIMKnnGraph input{{{2, 4}, {1, 400}}, {}, {{0, 4}}, {}};
        const auto result = HGraphFGIM::KnnGraphToPG(sources, input, 2);
        REQUIRE(result[0].size() == 2);
        REQUIRE(result[0][1].id == 1);
        REQUIRE(result[1].size() == 1);
        REQUIRE(result[1][0].id == 0);
    }
    SECTION("redundant incoming edge may be pruned") {
        const FGIMKnnGraph input{{{2, 4}, {1, 400}}, {}, {{0, 4}}, {{1, 4}}};
        const auto result = HGraphFGIM::KnnGraphToPG(sources, input, 2);
        REQUIRE(result[0].size() == 1);
        REQUIRE(result[0][0].id == 2);
    }
    SECTION("reverse additions are truncated by distance") {
        const FGIMKnnGraph input{{}, {{0, 400}}, {{0, 4}}, {{0, 484}}};
        const auto result = HGraphFGIM::KnnGraphToPG(sources, input, 1);
        REQUIRE(result[0].size() == 1);
        REQUIRE(result[0][0].id == 2);
        REQUIRE(result[0][0].distance == 4);
        for (const auto& row : result) {
            REQUIRE(row.size() <= 1);
        }
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM stages write valid bottom graph without changing vector or label storage",
                 "[ut][hgraph][fgim]") {
    auto lhs = MakeSource(6, 0);
    auto rhs = MakeSource(5, 1);
    auto third = MakeSource(4, 2);
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(lhs.get());
    sources.push_back(rhs.get());
    sources.push_back(third.get());
    auto initial = HGraphFGIM::BuildInitialKnnGraph(sources, 4);
    HGraphFGIM::RefineKnnGraph(sources, initial, 4, 3);
    const auto pg = HGraphFGIM::KnnGraphToPG(sources, initial, 4);
    // Only tests the topology writer. Public Merge supplies merged vectors/labels in PR4.
    auto destination = MakeSource(15, 3);
    const auto before = Capture(*destination);
    HGraphFGIM::WriteBottomGraph(*destination, pg, 4);
    const auto after = Capture(*destination);
    REQUIRE(after.labels == before.labels);
    REQUIRE(after.vectors == before.vectors);
    REQUIRE(after.removed == before.removed);
    REQUIRE(after.neighbors.size() == pg.size());
    for (uint64_t u = 0; u < pg.size(); ++u) {
        REQUIRE(after.neighbors[u].size() == pg[u].size());
        REQUIRE(pg[u].size() <= 4);
        std::set<InnerIdType> seen;
        for (uint64_t i = 0; i < pg[u].size(); ++i) {
            REQUIRE(after.neighbors[u][i] == pg[u][i].id);
            REQUIRE(pg[u][i].id != u);
            REQUIRE(pg[u][i].id < pg.size());
            REQUIRE(seen.insert(pg[u][i].id).second);
        }
    }
    auto invalid = pg;
    invalid[0] = {{0, 0}};
    REQUIRE_THROWS_AS(HGraphFGIM::WriteBottomGraph(*destination, invalid, 4), VsagException);
    REQUIRE(Capture(*destination).neighbors == after.neighbors);
    REQUIRE_THROWS_AS(HGraphFGIM::WriteBottomGraph(*destination, pg, 9), VsagException);
    REQUIRE(Capture(*destination).neighbors == after.neighbors);
    invalid.pop_back();
    REQUIRE_THROWS_AS(HGraphFGIM::KnnGraphToPG(sources, invalid, 4), VsagException);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM upper rebuild preserves storage and yields searchable nested layers",
                 "[ut][hgraph][fgim]") {
    const uint64_t count = GENERATE(1, 256);
    auto target = MakeSource(count, 0, false, "fp32", true);
    const auto before = Capture(*target);
    HGraphFGIM::RebuildUpperLayers(*target);
    const auto first = CaptureUpper(*target);
    const auto after = Capture(*target);
    REQUIRE(after.neighbors == before.neighbors);
    REQUIRE(after.labels == before.labels);
    REQUIRE(after.vectors == before.vectors);
    REQUIRE(after.removed == before.removed);
    if (count == 256) {
        REQUIRE_FALSE(first.empty());
    }
    for (uint64_t u = 0; u < count; u += 17) {
        auto query = Dataset::Make();
        query->NumElements(1)
            ->Dim(common.dim_)
            ->Float32Vectors(before.vectors.data() + u * common.dim_)
            ->Owner(false);
        const auto found = target->KnnSearch(query, 1, R"({"hgraph":{"ef_search":128}})", nullptr);
        REQUIRE(found->GetDim() == 1);
        REQUIRE(found->GetIds()[0] == before.labels[u]);
        REQUIRE(found->GetDistances()[0] == 0);
    }
    HGraphFGIM::RebuildUpperLayers(*target);
    REQUIRE(CaptureUpper(*target) == first);
    REQUIRE(Capture(*target).neighbors == before.neighbors);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM public Merge preserves mapped labels vectors and source indexes",
                 "[ut][hgraph][fgim]") {
    const bool nonempty = GENERATE(false, true);
    auto destination =
        std::shared_ptr<HGraph>(MakeSource(nonempty ? 5 : 0, 3, false, "fp32", true, false, true));
    auto public_destination = std::make_shared<IndexImpl<HGraph>>(destination, common);
    auto expected = Capture(*destination);
    std::vector<std::shared_ptr<HGraph>> sources;
    std::vector<Snapshot> before;
    std::vector<MergeUnit> units;
    std::vector<uint64_t> callback_counts(3, 0);
    for (uint64_t i = 0; i < 3; ++i) {
        sources.emplace_back(MakeSource(24, i, false, "fp32", true));
        before.push_back(Capture(*sources.back()));
        expected.vectors.insert(
            expected.vectors.end(), before.back().vectors.begin(), before.back().vectors.end());
        for (const auto label : before.back().labels) {
            expected.labels.push_back(label + 10000 * (i + 1));
        }
        units.push_back(
            {std::make_shared<IndexImpl<HGraph>>(sources.back(), common), [&, i](LabelType label) {
                 ++callback_counts[i];
                 return std::make_tuple(true, label + 10000 * (i + 1));
             }});
    }
    const auto merged = public_destination->Merge(units);
    REQUIRE(merged.has_value());
    REQUIRE(destination->GetNumElements() == expected.labels.size());
    const auto after = Capture(*destination);
    REQUIRE(after.labels == expected.labels);
    REQUIRE(after.vectors == expected.vectors);
    for (uint64_t u = 0; u < after.neighbors.size(); ++u) {
        std::set<InnerIdType> seen;
        REQUIRE(after.neighbors[u].size() <= 8);
        for (const auto v : after.neighbors[u]) {
            REQUIRE(v < after.neighbors.size());
            REQUIRE(v != u);
            REQUIRE(seen.insert(v).second);
        }
    }
    CaptureUpper(*destination);
    uint64_t hits = 0, queries = 0;
    for (uint64_t u = 0; u < after.labels.size(); u += 7) {
        std::vector<float> query_vector(common.dim_);
        for (int64_t d = 0; d < common.dim_; ++d) {
            query_vector[d] = after.vectors[u * common.dim_ + d] + 0.125F;
        }
        std::vector<std::pair<float, LabelType>> exact;
        for (uint64_t v = 0; v < after.labels.size(); ++v) {
            float distance = 0;
            for (int64_t d = 0; d < common.dim_; ++d) {
                const float delta = query_vector[d] - after.vectors[v * common.dim_ + d];
                distance += delta * delta;
            }
            exact.emplace_back(distance, after.labels[v]);
        }
        std::sort(exact.begin(), exact.end());
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(common.dim_)->Float32Vectors(query_vector.data())->Owner(false);
        const auto found =
            public_destination->KnnSearch(query, 10, R"({"hgraph":{"ef_search":256}})");
        REQUIRE(found.has_value());
        REQUIRE(found.value()->GetDim() == 10);
        for (uint64_t i = 0; i < 10; ++i) {
            const auto label = found.value()->GetIds()[i];
            hits += std::any_of(exact.begin(), exact.begin() + 10, [label](const auto& item) {
                return item.second == label;
            });
        }
        ++queries;
    }
    REQUIRE(static_cast<double>(hits) / (queries * 10) >= 0.95);
    for (uint64_t i = 0; i < sources.size(); ++i) {
        REQUIRE(callback_counts[i] == 24);
        const auto unchanged = Capture(*sources[i]);
        REQUIRE(unchanged.labels == before[i].labels);
        REQUIRE(unchanged.vectors == before[i].vectors);
        REQUIRE(unchanged.neighbors == before[i].neighbors);
    }
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM public Merge rejects invalid inputs before changing destination",
                 "[ut][hgraph][fgim]") {
    auto destination = std::shared_ptr<HGraph>(MakeSource(5, 0, false, "fp32", false, false, true));
    auto public_destination = std::make_shared<IndexImpl<HGraph>>(destination, common);
    auto source = std::shared_ptr<HGraph>(MakeSource(5, 1));
    std::vector<MergeUnit> units{{std::make_shared<IndexImpl<HGraph>>(source, common), nullptr}};
    SECTION("label collision") {
        units[0].id_map_func = [](LabelType) { return std::make_tuple(true, 1000); };
    }
    SECTION("filtered node") {
        units[0].id_map_func = [](LabelType label) { return std::make_tuple(false, label); };
    }
    SECTION("self merge") {
        units[0].index = public_destination;
    }
    SECTION("null index") {
        units[0].index.reset();
    }
    SECTION("repeated source") {
        units.push_back(units[0]);
    }
    SECTION("unsupported metric") {
        SetUnsupported(*source, "non-L2");
    }
    const auto before = Capture(*destination);
    const auto merged = public_destination->Merge(units);
    REQUIRE_FALSE(merged.has_value());
    const auto after = Capture(*destination);
    REQUIRE(after.labels == before.labels);
    REQUIRE(after.vectors == before.vectors);
    REQUIRE(after.neighbors == before.neighbors);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM merge parameters roundtrip and default strategy is unchanged",
                 "[ut][hgraph][fgim]") {
    auto external = JsonType::Parse(R"({"max_degree":8,"ef_construction":32})");
    auto parameters = std::dynamic_pointer_cast<HGraphParameter>(
        HGraph::CheckAndMappingExternalParam(external, common));
    REQUIRE(parameters->merge_strategy == "odescent");
    external["merge_strategy"].SetString("fgim");
    external["fgim_refinement_iterations"].SetUint64(0);
    parameters = std::dynamic_pointer_cast<HGraphParameter>(
        HGraph::CheckAndMappingExternalParam(external, common));
    REQUIRE(parameters->merge_strategy == "fgim");
    REQUIRE(parameters->fgim_refinement_iterations == 0);
    const HGraphParameter restored(parameters->ToJson());
    REQUIRE(restored.merge_strategy == "fgim");
    REQUIRE(restored.fgim_refinement_iterations == 0);
    external["merge_strategy"].SetString("unknown");
    REQUIRE_THROWS_AS(HGraph::CheckAndMappingExternalParam(external, common), VsagException);
    external["merge_strategy"].SetString("fgim");
    external["fgim_refinement_iterations"].SetInt(-1);
    REQUIRE_THROWS_AS(HGraph::CheckAndMappingExternalParam(external, common), VsagException);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM Merge supports identity mapping and empty-unit no-op",
                 "[ut][hgraph][fgim]") {
    auto destination = std::shared_ptr<HGraph>(MakeSource(5, 0, false, "fp32", false, false, true));
    auto wrapper = std::make_shared<IndexImpl<HGraph>>(destination, common);
    const auto before = Capture(*destination);
    REQUIRE(wrapper->Merge({}).has_value());
    REQUIRE(Capture(*destination).neighbors == before.neighbors);
    auto source = std::shared_ptr<HGraph>(MakeSource(5, 1));
    const auto source_before = Capture(*source);
    REQUIRE(wrapper->Merge({{std::make_shared<IndexImpl<HGraph>>(source, common), nullptr}})
                .has_value());
    auto expected_labels = before.labels;
    expected_labels.insert(
        expected_labels.end(), source_before.labels.begin(), source_before.labels.end());
    REQUIRE(Capture(*destination).labels == expected_labels);
    REQUIRE(destination->GetNumElements() == 10);
}

TEST_CASE_METHOD(HGraphFGIMTest,
                 "FGIM diagnostics preserve initial and completed graph results",
                 "[ut][hgraph][fgim]") {
    auto a = std::shared_ptr<HGraph>(MakeSource(24, 0, false, "fp32", true));
    auto b = std::shared_ptr<HGraph>(MakeSource(24, 1, false, "fp32", true));
    Vector<const HGraph*> sources(common.allocator_.get());
    sources.push_back(a.get());
    sources.push_back(b.get());
    const auto initial = HGraphFGIM::BuildInitialKnnGraph(sources, 8);
    uint64_t distances = 0;
    const auto counted = HGraphFGIM::BuildInitialKnnGraph(sources, 8, &distances);
    REQUIRE(distances > 0);
    REQUIRE(counted.size() == initial.size());
    for (uint64_t u = 0; u < initial.size(); ++u) {
        REQUIRE(counted[u].size() == initial[u].size());
        for (uint64_t i = 0; i < initial[u].size(); ++i) {
            REQUIRE(counted[u][i].id == initial[u][i].id);
            REQUIRE(counted[u][i].distance == initial[u][i].distance);
        }
    }
    std::vector<MergeUnit> units{{std::make_shared<IndexImpl<HGraph>>(a, common), nullptr},
                                 {std::make_shared<IndexImpl<HGraph>>(b, common), nullptr}};
    auto plain = MakeSource(0, 2, false, "fp32", true, false, true);
    auto diagnosed = MakeSource(0, 2, false, "fp32", true, false, true);
    HGraphFGIM::Merge(*plain, units, 4);
    FGIMMergeStatistics stats;
    HGraphFGIM::Merge(*diagnosed, units, 4, &stats);
    REQUIRE(stats.initial_distance_evaluations == distances);
    REQUIRE(stats.refinement.iterations > 0);
    REQUIRE(stats.refinement.distance_evaluations > 0);
    REQUIRE(stats.reconstruction_distance_evaluations > 0);
    for (const double seconds : {stats.initial_seconds,
                                 stats.refinement_seconds,
                                 stats.reconstruction_seconds,
                                 stats.storage_seconds,
                                 stats.bottom_write_seconds,
                                 stats.upper_seconds}) {
        REQUIRE(std::isfinite(seconds));
        REQUIRE(seconds >= 0);
    }
    const auto expected = Capture(*plain);
    const auto actual = Capture(*diagnosed);
    REQUIRE(actual.labels == expected.labels);
    REQUIRE(actual.vectors == expected.vectors);
    REQUIRE(actual.neighbors == expected.neighbors);
    REQUIRE(CaptureUpper(*diagnosed) == CaptureUpper(*plain));
    HGraphFGIM::Merge(*diagnosed, {}, 4, &stats);
    REQUIRE(stats.initial_distance_evaluations == 0);
    REQUIRE(stats.initial_seconds == 0);
}

}  // namespace vsag
