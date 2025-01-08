
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

#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <limits>

#include "fixtures/test_dataset_pool.h"
#include "fixtures/test_logger.h"
#include "test_index.h"

namespace fixtures {
class MergeTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateHnswBuildParametersString(const std::string& metric_type, int64_t dim);

    static std::vector<vsag::DatasetPtr>
    SplitDataset(vsag::DatasetPtr raw_data, int64_t split_n);

    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 2000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "hnsw": {{
                "ef_search": {}
            }}
        }})";
};

TestDatasetPool MergeTestIndex::pool{};
std::vector<int> MergeTestIndex::dims{128, 256};
std::string
MergeTestIndex::GenerateHnswBuildParametersString(const std::string& metric_type, int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 32,
            "ef_construction": 100
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters_str;
}

std::vector<vsag::DatasetPtr>
MergeTestIndex::SplitDataset(vsag::DatasetPtr raw_data, int64_t split_n) {
    std::vector<vsag::DatasetPtr> sub_datasets;
    auto dataset = vsag::Dataset::Make();
    int64_t all_data_num = raw_data->GetNumElements();
    int64_t data_dim = raw_data->GetDim();
    const float* vectors = raw_data->GetFloat32Vectors();  // shape = (all_data_num, data_dim)
    const int64_t* ids = raw_data->GetIds();               // shape = (all_data_num)

    int64_t subset_size = all_data_num / split_n;
    int64_t remaining = all_data_num % split_n;

    int64_t start_index = 0;

    for (int64_t i = 0; i < split_n; ++i) {
        int64_t current_subset_size = subset_size + (i < remaining ? 1 : 0);
        auto subset = vsag::Dataset::Make();
        subset->Float32Vectors(vectors + start_index * data_dim);
        subset->Ids(ids + start_index);
        subset->NumElements(current_subset_size);
        subset->Dim(data_dim);
        subset->Owner(false);
        sub_datasets.push_back(subset);
        start_index += current_subset_size;
    }
    return sub_datasets;
}

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::MergeTestIndex, "Merge HNSW", "[ft][merge]") {
    auto metric_type = GENERATE("l2");
    std::string base_quantization_str = GENERATE("fp32");
    const std::string name = "hnsw";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        std::vector<std::shared_ptr<vsag::Index>> sub_indexes;
        auto param = GenerateHnswBuildParametersString(metric_type, dim);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        std::vector<DatasetPtr> sub_datas = SplitDataset(dataset->base_, 1);
        for (auto sub_dataset : sub_datas) {
            auto sub_index = TestFactory(name, param, true);
            sub_index->Build(sub_dataset);
            sub_indexes.push_back(sub_index);
        }
        vsag::Engine e;
        auto result = e.MergeGraphIndex(name, param, sub_indexes);
        REQUIRE(result.has_value());
        std::shared_ptr<vsag::Index> index = result.value();
        TestKnnSearch(index, dataset, search_param, 0.98, true);
        TestRangeSearch(index, dataset, search_param, 0.98, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.48, 5, true);
        TestFilterSearch(index, dataset, search_param, 0.98, true);
    }
}