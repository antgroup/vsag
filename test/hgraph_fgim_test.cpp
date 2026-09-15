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
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>

#include "algorithm/hgraph/hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"
#include "vsag/dataset.h"

namespace vsag {

// Test-only fixture: no test entry points are added to HGraphFGIM.
class HGraphFGIMTest {
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
               bool random = false) {
        auto parameters = JsonType::Parse(R"({
            "base_quantization_type": "fp32",
            "max_degree": 8,
            "ef_construction": 32,
            "build_thread_count": 1,
            "support_duplicate": false,
            "deduplicate_storage": false
        })");
        parameters["base_quantization_type"].SetString(base_quantization);
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
                                   : static_cast<float>(10 * u + source_index + d));
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
    ExpectInvalid(const Vector<const HGraph*>& sources, uint64_t k, const std::string& reason) {
        try {
            (void)HGraphFGIM::BuildInitialKnnGraph(sources, k);
            FAIL("unsupported FGIM input was accepted");
        } catch (const VsagException& e) {
            REQUIRE(e.error_.type == ErrorType::INVALID_ARGUMENT);
            REQUIRE(std::string(e.what()).find(reason) != std::string::npos);
        }
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
        }
    }

    void
    CompareCross(const HGraph& target, const std::vector<float>& vector, int64_t l) {
        auto query = Dataset::Make();
        query->NumElements(1)->Dim(common.dim_)->Float32Vectors(vector.data())->Owner(false);
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
        for (uint64_t i = 0; i < fast.size(); ++i) {
            REQUIRE(fast[i].id == expected[i].id);
            REQUIRE(fast[i].distance == expected[i].distance);
            REQUIRE(counted[i].id == fast[i].id);
            REQUIRE(counted[i].distance == fast[i].distance);
        }
        const auto public_stats = JsonType::Parse(public_result->GetStatistics());
        REQUIRE(stats.distance_evaluations.load() ==
                public_stats["distance_evaluations"].GetUint64());
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
        ExpectInvalid(sources, std::numeric_limits<std::size_t>::max(), "k must");
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

}  // namespace vsag
