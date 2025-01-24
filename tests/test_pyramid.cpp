
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

#include "fixtures/test_dataset_pool.h"
#include "test_index.h"
#include "vsag/vsag.h"

namespace fixtures {
class PyramidTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GeneratePyramidBuildParametersString(const std::string& metric_type, int64_t dim);

    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 3000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "pyramid": {{
                "ef_search": 100
            }}
        }})";
};

TestDatasetPool PyramidTestIndex::pool{};
std::vector<int> PyramidTestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));

std::string
PyramidTestIndex::GeneratePyramidBuildParametersString(const std::string& metric_type,
                                                       int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "odescent": {{
                "io_params": {{
                    "type": "memory"
                }},
                "max_degree": 32,
                "alpha": 1.2,
                "graph_iter_turn": 15,
                "neighbor_sample_rate": 0.2
            }}
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters_str;
}
}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Build & ContinueAdd Test",
                             "[ft][pyramid]") {
    auto metric_type = GENERATE("l2", "ip", "cosine");
    std::string base_quantization_str = GENERATE("fp32");
    const std::string name = "pyramid";
    auto search_param = fmt::format(search_param_tmp, 100);
    for (auto& dim : dims) {
        auto param = GeneratePyramidBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        //        TestFilterSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
        TestRangeSearch(index, dataset, search_param, 0.49, 5, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::PyramidTestIndex,
                             "Pyramid Serialize File",
                             "[ft][pyramid]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("l2");
    const std::string name = "pyramid";
    auto search_param = fmt::format(search_param_tmp, 200);

    for (auto& dim : dims) {
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GeneratePyramidBuildParametersString(metric_type, dim);
        auto index = TestFactory(name, param, true);
        SECTION("serialize empty index") {
            auto index2 = TestFactory(name, param, true);
            auto serialize_binary = index->Serialize();
            REQUIRE(serialize_binary.has_value());
            auto deserialize_index = index2->Deserialize(serialize_binary.value());
            REQUIRE(deserialize_index.has_value());
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type, /*with_path=*/true);
        TestBuildIndex(index, dataset, true);
        SECTION("serialize/deserialize by binary") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeBinarySet(index, index2, dataset, search_param, true);
        }
        SECTION("serialize/deserialize by readerset") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
        }
        SECTION("serialize/deserialize by file") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeFile(index, index2, dataset, search_param, true);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}