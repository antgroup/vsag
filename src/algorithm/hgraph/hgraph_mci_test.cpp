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

#include <numeric>
#include <string>
#include <vector>

#include "unittest.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/filter.h"

namespace {

class HalfRatioAllValidFilter : public vsag::Filter {
public:
    explicit HalfRatioAllValidFilter(const std::vector<int64_t>& ids) : ids_(ids) {
    }

    bool
    CheckValid(int64_t id) const override {
        return std::find(ids_.begin(), ids_.end(), id) != ids_.end();
    }

    float
    ValidRatio() const override {
        return 0.5F;
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        *valid_ids = ids_.data();
        count = static_cast<int64_t>(ids_.size());
    }

private:
    std::vector<int64_t> ids_;
};

std::string
generate_hgraph_mci_params(int64_t dim) {
    auto params = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "index_param": {
                "base_quantization_type": "fp32",
                "graph_type": "odescent",
                "max_degree": 6,
                "alpha": 1.2,
                "graph_iter_turn": 6,
                "neighbor_sample_rate": 0.3,
                "mci": {
                    "use_mci": true,
                    "mcs": 8,
                    "clique_max": 4,
                    "alpha": 1.2,
                    "seed_count": 8,
                    "hgraph_valid_ratio_threshold": 1.0
                }
            }
        }
    )");
    params["dim"].SetInt(dim);
    return params.Dump();
}

vsag::DatasetPtr
make_dataset(std::vector<int64_t>& ids,
             std::vector<float>& vectors,
             int64_t offset,
             int64_t count,
             int64_t dim) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data() + offset)
        ->Float32Vectors(vectors.data() + offset * dim)
        ->Owner(false);
    return dataset;
}

}  // namespace

TEST_CASE("HGraph companion MCI incrementally updates cliques after Add", "[ut][hgraph][mci]") {
    constexpr int64_t dim = 4;
    constexpr int64_t base_count = 24;
    constexpr int64_t add_count = 4;
    constexpr int64_t total = base_count + add_count;

    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 1000);

    std::vector<float> vectors(total * dim, 0.0F);
    for (int64_t i = 0; i < total; ++i) {
        vectors[i * dim] = static_cast<float>(i / 4);
        vectors[i * dim + 1] = static_cast<float>(i % 4);
        vectors[i * dim + 2] = static_cast<float>((i * 3) % 7);
        vectors[i * dim + 3] = static_cast<float>((i * 5) % 11);
    }

    auto index = vsag::Factory::CreateIndex("hgraph", generate_hgraph_mci_params(dim));
    REQUIRE(index.has_value());

    auto build_result = index.value()->Build(make_dataset(ids, vectors, 0, base_count, dim));
    REQUIRE(build_result.has_value());
    REQUIRE(build_result.value().empty());

    auto add_result = index.value()->Add(make_dataset(ids, vectors, base_count, add_count, dim));
    REQUIRE(add_result.has_value());
    REQUIRE(add_result.value().empty());

    auto query = vsag::Dataset::Make();
    query->NumElements(1)
        ->Dim(dim)
        ->Float32Vectors(vectors.data() + (base_count + 1) * dim)
        ->Owner(false);

    auto filter = std::make_shared<HalfRatioAllValidFilter>(ids);
    auto result = index.value()->KnnSearch(
        query,
        3,
        R"({"hgraph":{"ef_search":16,"use_mci":true,"seed_count":8,"hgraph_valid_ratio_threshold":1.0}})",
        filter);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->GetDim() > 0);
    REQUIRE(result.value()->GetStatistics({"mci_hybrid_route"})[0] == R"("mci")");
}
