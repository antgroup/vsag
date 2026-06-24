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
#include <fstream>
#include <memory>
#include <vector>

#include "typing.h"
#include "unittest.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

class MinLabelFilter : public vsag::Filter {
public:
    explicit MinLabelFilter(int64_t min_label) : min_label_(min_label) {
    }

    bool
    CheckValid(int64_t id) const override {
        return id >= min_label_;
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = valid_ids_.data();
        count = static_cast<int64_t>(valid_ids_.size());
    }

    void
    Init(int64_t start_label, int64_t count) {
        valid_ids_.clear();
        valid_ids_.reserve(static_cast<size_t>(count));
        for (int64_t id = start_label; id < start_label + count; ++id) {
            if (id >= min_label_) {
                valid_ids_.push_back(id);
            }
        }
        valid_ratio_ =
            count > 0 ? static_cast<float>(valid_ids_.size()) / static_cast<float>(count) : 1.0F;
    }

private:
    int64_t min_label_{0};
    float valid_ratio_{1.0F};
    std::vector<int64_t> valid_ids_{};
};

vsag::JsonType
generate_mci_index_params(int64_t dim) {
    auto parameters = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "max_degree": 6,
                "mcs": 16,
                "clique_max": 6
            }
        }
    )");
    parameters["dim"].SetInt(dim);
    return parameters;
}

vsag::JsonType
generate_mci_rabitq_index_params(int64_t dim) {
    auto parameters = generate_mci_index_params(dim);
    parameters["index_param"]["use_reorder"].SetBool(true);
    parameters["index_param"]["reorder_source"].SetString("base");
    parameters["index_param"]["base_quantization_type"].SetString("rabitq");
    parameters["index_param"]["base_codes_type"].SetString("rabitq_split");
    parameters["index_param"]["rabitq_version"].SetString("split_1bit_7bit");
    parameters["index_param"]["rabitq_bits_per_dim_query"].SetInt(32);
    parameters["index_param"]["rabitq_bits_per_dim_base"].SetInt(8);
    return parameters;
}

vsag::JsonType
generate_mci_search_params() {
    auto parameters = vsag::JsonType::Parse(R"(
        {
            "mci": {
                "ef_search": 10,
                "seed_count": 8
            }
        }
    )");
    return parameters;
}

vsag::JsonType
generate_hgraph_index_params(int64_t dim) {
    auto parameters = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 10,
                "neighbor_sample_rate": 0.2
            }
        }
    )");
    parameters["dim"].SetInt(dim);
    return parameters;
}

vsag::JsonType
generate_mci_hybrid_overlay_params(int64_t dim, const std::string& hgraph_index_path) {
    auto parameters = generate_mci_index_params(dim);
    parameters["index_param"]["use_hgraph_hybrid"].SetBool(true);
    parameters["index_param"]["hgraph_valid_ratio_threshold"].SetFloat(0.5F);
    parameters["index_param"]["hgraph_index_path"].SetString(hgraph_index_path);
    parameters["index_param"]["hgraph_ef_search"].SetInt(32);
    parameters["index_param"]["hgraph_index_param"].SetJson(
        generate_hgraph_index_params(dim)["index_param"]);
    return parameters;
}

vsag::JsonType
generate_mci_rabitq_search_params() {
    auto parameters = generate_mci_search_params();
    parameters["mci"]["rabitq_one_bit_search"].SetBool(true);
    return parameters;
}

void
write_line_knng(const std::string& path, int64_t count, uint64_t degree) {
    std::ofstream output(path, std::ios::binary);
    REQUIRE(output.good());
    for (int64_t i = 0; i < count; ++i) {
        std::vector<vsag::InnerIdType> neighbors;
        neighbors.reserve(static_cast<size_t>(count - 1));
        for (int64_t j = 0; j < count; ++j) {
            if (i != j) {
                neighbors.push_back(static_cast<vsag::InnerIdType>(j));
            }
        }
        std::sort(neighbors.begin(), neighbors.end(), [i](auto lhs, auto rhs) {
            auto lhs_gap = std::abs(static_cast<int64_t>(lhs) - i);
            auto rhs_gap = std::abs(static_cast<int64_t>(rhs) - i);
            if (lhs_gap != rhs_gap) {
                return lhs_gap < rhs_gap;
            }
            return lhs < rhs;
        });
        output.write(reinterpret_cast<const char*>(neighbors.data()),
                     static_cast<std::streamsize>(degree * sizeof(vsag::InnerIdType)));
    }
}

}  // namespace

TEST_CASE("MCI build and filtered knn search", "[ut][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 32;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 100 + i;
        vectors[i * dim] = static_cast<float>(i);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("mci", generate_mci_index_params(dim).Dump());
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    auto search_params = generate_mci_search_params();

    SECTION("nearest neighbor uses clique candidates") {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data())->Owner(false);

        auto result = index.value()->KnnSearch(query, 3, search_params.Dump());
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() == 3);
        REQUIRE(result.value()->GetIds()[0] == ids[0]);
        auto stats = result.value()->GetStatistics({"total_clique_count"});
        REQUIRE(std::stoll(stats[0]) > 0);
    }

    SECTION("public filter is applied on labels") {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 20 * dim)->Owner(false);

        auto filter = std::make_shared<MinLabelFilter>(116);
        auto result = index.value()->KnnSearch(query, 5, search_params.Dump(), filter);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->GetDim() > 0);
        REQUIRE(result.value()->GetIds()[0] == ids[20]);
        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
            REQUIRE(result.value()->GetIds()[i] >= 116);
        }
    }
}

TEST_CASE("MCI rabitq split one bit search", "[ut][mci]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 32;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 1000 + i;
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 7);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("mci", generate_mci_rabitq_index_params(dim).Dump());
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 10 * dim)->Owner(false);
    auto result = index.value()->KnnSearch(query, 3, generate_mci_rabitq_search_params().Dump());
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 3);
    auto stats = result.value()->GetStatistics({"rabitq_one_bit_search", "total_clique_count"});
    REQUIRE(stats[0] == "true");
    REQUIRE(std::stoll(stats[1]) > 0);
}

TEST_CASE("MCI build from external knng", "[ut][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 24;
    constexpr uint64_t degree = 8;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 200 + i;
        vectors[i * dim] = static_cast<float>(i);
        vectors[i * dim + 1] = static_cast<float>(i % 3);
    }

    fixtures::TempDir dir("mci_knng");
    auto knng_path = dir.path + "line.knng";
    write_line_knng(knng_path, count, degree);

    auto params = generate_mci_index_params(dim);
    params["index_param"]["build_thread_count"].SetInt(2);
    params["index_param"]["mcs"].SetInt(static_cast<int64_t>(degree));
    params["index_param"]["clique_max"].SetInt(3);
    params["index_param"]["knng_path"].SetString(knng_path);

    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    auto index = vsag::Factory::CreateIndex("mci", params.Dump());
    REQUIRE(index.has_value());
    auto build_result = index.value()->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 7 * dim)->Owner(false);
    auto result = index.value()->KnnSearch(query, 3, generate_mci_search_params().Dump());
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() == 3);
    auto stats = result.value()->GetStatistics({"total_clique_count"});
    REQUIRE(std::stoll(stats[0]) > 0);
}

TEST_CASE("MCI hybrid search can reuse decoupled MCI and HGraph indexes", "[ut][mci][hybrid]") {
    constexpr int64_t dim = 4;
    constexpr int64_t count = 32;
    std::vector<int64_t> ids(count);
    std::vector<float> vectors(count * dim, 0.0F);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = 100 + i;
        vectors[i * dim] = static_cast<float>(i);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);

    fixtures::TempDir dir("mci_hybrid_decoupled");
    const auto hgraph_path = dir.path + "hgraph.index";
    const auto mci_path = dir.path + "mci.index";

    auto hgraph = vsag::Factory::CreateIndex("hgraph", generate_hgraph_index_params(dim).Dump());
    REQUIRE(hgraph.has_value());
    auto hgraph_build_result = hgraph.value()->Build(base);
    REQUIRE(hgraph_build_result.has_value());
    REQUIRE(hgraph_build_result.value().empty());
    {
        std::ofstream output(hgraph_path, std::ios::binary);
        REQUIRE(output.good());
        auto serialize_result = hgraph.value()->Serialize(output);
        REQUIRE(serialize_result.has_value());
    }

    auto mci = vsag::Factory::CreateIndex("mci", generate_mci_index_params(dim).Dump());
    REQUIRE(mci.has_value());
    auto mci_build_result = mci.value()->Build(base);
    REQUIRE(mci_build_result.has_value());
    REQUIRE(mci_build_result.value().empty());
    {
        std::ofstream output(mci_path, std::ios::binary);
        REQUIRE(output.good());
        auto serialize_result = mci.value()->Serialize(output);
        REQUIRE(serialize_result.has_value());
    }

    auto hybrid = vsag::Factory::CreateIndex(
        "mci", generate_mci_hybrid_overlay_params(dim, hgraph_path).Dump());
    REQUIRE(hybrid.has_value());
    {
        std::ifstream input(mci_path, std::ios::binary);
        REQUIRE(input.good());
        auto deserialize_result = hybrid.value()->Deserialize(input);
        REQUIRE(deserialize_result.has_value());
    }

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + 20 * dim)->Owner(false);

    auto filter = std::make_shared<MinLabelFilter>(110);
    filter->Init(ids.front(), count);
    auto result = hybrid.value()->KnnSearch(query, 5, generate_mci_search_params().Dump(), filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        REQUIRE(result.value()->GetIds()[i] >= 110);
    }
    auto stats = result.value()->GetStatistics({"mci_hybrid_route", "mci_hybrid_valid_ratio"});
    REQUIRE(stats[0].find("hgraph") != std::string::npos);
    REQUIRE(std::stof(stats[1]) > 0.5F);
}
