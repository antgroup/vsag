
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
#include <utility>

#include "extra_info_interface_test.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "io/io_headers.h"
#include "parameter_test.h"
#include "unittest.h"

using namespace vsag;

namespace {
class ShrinkFailureLayout : public FixedLayout<MemoryBlockIO> {
public:
    using FixedLayout<MemoryBlockIO>::FixedLayout;

    void
    Shrink(uint64_t capacity) {
        if (fail) {
            throw std::bad_alloc();
        }
        FixedLayout<MemoryBlockIO>::Shrink(capacity);
    }

    bool fail{false};
};
}  // namespace

TEST_CASE("Extra info shrink failure preserves logical truncation", "[ut][ExtraInfoDataCell]") {
    IndexCommonParam common;
    common.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common.extra_info_size_ = 4;
    auto params = std::make_shared<ExtraInfoDataCellParameter>();
    params->FromJson(JsonType::Parse(R"({"io_params":{"type":"block_memory_io"}})"));
    auto cell =
        std::make_shared<ExtraInfoDataCell<ShrinkFailureLayout>>(params->io_parameter, common);
    cell->Resize(1024);
    cell->InsertExtraInfo("abcd", 0);
    cell->InsertExtraInfo("efgh", 1);
    cell->layout_->fail = true;
    REQUIRE_THROWS_AS(cell->ShrinkToFit(1), std::bad_alloc);
    cell->layout_->fail = false;
    REQUIRE(cell->TotalCount() == 1);
    cell->InsertExtraInfo("ijkl", std::numeric_limits<InnerIdType>::max());
    REQUIRE(cell->TotalCount() == 2);
    char bytes[4];
    REQUIRE(cell->GetExtraInfoById(1, bytes));
    REQUIRE(std::string(bytes, 4) == "ijkl");
}

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
        auto param_json = JsonType::Parse(param_str);
        logger::debug("param_json: {}", param_json.Dump());
        auto param = std::make_shared<ExtraInfoDataCellParameter>();
        param->FromJson(param_json);
        vsag::ParameterTest::TestToJson(param);
        logger::debug("param->ToJson(): {}", param->ToJson().Dump());

        IndexCommonParam common_param;
        common_param.allocator_ = allocator;
        common_param.dim_ = dim;
        common_param.metric_ = metric;
        common_param.extra_info_size_ = extra_info_size;

        TestExtraInfoDataCell(param, common_param, counts[i]);
        i++;
    }
}
