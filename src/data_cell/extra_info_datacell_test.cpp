
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

#include "extra_info_datacell.h"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <utility>

#include "extra_info_interface_test.h"
#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "parameter_test.h"

using namespace vsag;

void
TestExtraInfoDataCell(ExtraInfoDataCellParamPtr& param,
                      IndexCommonParam& common_param,
                      uint64_t spec_count) {
    auto count = spec_count > 0 ? spec_count : GENERATE(100, 1000);
    auto extra_info = ExtraInfoInterface::MakeInstance(param, common_param);

    ExtraInfoInterfaceTest test(extra_info);
    test.TestForceInMemory(count);
    test.BasicTest(count);

    auto other = ExtraInfoInterface::MakeInstance(param, common_param);
    test.TestSerializeAndDeserialize(other);
}

TEST_CASE("ExtraInfoDataCell Basic Test", "[ut][ExtraInfoDataCell] ") {
    logger::set_level(logger::level::debug);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    uint64_t extra_info_sizes[4] = {32, 128, 512, 3 * 1024};
    uint64_t counts[4] = {0, 0, 0, 50};
    int dim = 512;
    MetricType metric = MetricType::METRIC_TYPE_L2SQR;
    constexpr const char* param_str =
        R"(
        {
            "io_params": {
                "type": "block_memory_io"
            }
        }
        )";
    int i = 0;
    for (auto& extra_info_size : extra_info_sizes) {
        auto param_json = JsonType::parse(param_str);
        logger::debug("param_json: {}", param_json.dump());
        auto param = std::make_shared<ExtraInfoDataCellParameter>();
        param->FromJson(param_json);
        vsag::ParameterTest::TestToJson(param);
        logger::debug("param->ToJson(): {}", param->ToJson().dump());

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = metric;
        common_param.extra_info_size_ = extra_info_size;

        TestExtraInfoDataCell(param, common_param, counts[i]);
        i++;
    }
}
