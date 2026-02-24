
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "algorithm/inner_index_interface.h"
#include "algorithm/warp.h"
#include "data_type.h"
#include "datacell/flatten_datacell_parameter.h"
#include "fixtures/test_dataset.h"
#include "fixtures/test_dataset_pool.h"
#include "fixtures/test_logger.h"
#include "impl/allocator/default_allocator.h"
#include "index_common_param.h"
#include "test_index.h"
#include "typing.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/vsag.h"

using namespace vsag;

struct WarpParam {
    std::string base_quantization_type = "fp32";
    std::string base_io_type = "memory_io";
};

namespace fixtures {
class WarpTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateWarpBuildParametersString(const std::string& metric_type,
                                      int64_t dim,
                                      const WarpParam& param);

    static std::string
    GenerateWarpSearchParametersString();

    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 1000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "warp": {{
            }}
        }})";
};

TestDatasetPool WarpTestIndex::pool{};
std::vector<int> WarpTestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));

std::string
WarpTestIndex::GenerateWarpBuildParametersString(const std::string& metric_type,
                                                 int64_t dim,
                                                 const WarpParam& param) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "{}",
            "base_io_type": "{}"
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(
        parameter_temp, metric_type, dim, param.base_quantization_type, param.base_io_type);
    return build_parameters_str;
}

std::string
WarpTestIndex::GenerateWarpSearchParametersString() {
    return fmt::format(search_param_tmp);
}

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex, "Warp Add Test", "[ft][warp]") {
    auto metric_type = GENERATE("ip");
    std::string base_quantization_str = GENERATE("fp32");
    WarpParam warp_param;
    warp_param.base_quantization_type = base_quantization_str;
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateWarpBuildParametersString(metric_type, dim, warp_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::WARP);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
        TestAddIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex,
                             "Warp Serialize File",
                             "[ft][warp][serialization]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("ip");
    std::string base_quantization_str = GENERATE("fp32");
    WarpParam warp_param;
    warp_param.base_quantization_type = base_quantization_str;
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateWarpBuildParametersString(metric_type, dim, warp_param);
        auto index = TestFactory(name, param, true);
        SECTION("serialize empty index") {
            auto index2 = TestFactory(name, param, true);
            auto serialize_binary = index->Serialize();
            REQUIRE(serialize_binary.has_value());
            auto deserialize_index = index2->Deserialize(serialize_binary.value());
            REQUIRE(deserialize_index.has_value());
        }
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, metric_type);
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
