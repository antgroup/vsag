
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

#include "index_detail_data.h"

#include <catch2/catch_test_macros.hpp>

using namespace vsag;

TEST_CASE("DetailDataImpl Test", "[DetailDataImpl][ut]") {
    DetailDataImpl data;
    DetailData* detail_data = &data;
    data.SetData1DArrayInt64({1, 2, 3});
    REQUIRE(detail_data->GetData1DArrayInt64() == std::vector<int64_t>({1, 2, 3}));

    data.SetData2DArrayInt64({{1, 2}, {3, 4}});
    REQUIRE(detail_data->GetData2DArrayInt64() ==
            std::vector<std::vector<int64_t>>({{1, 2}, {3, 4}}));

    data.SetDataScalarInt64(100);
    REQUIRE(detail_data->GetDataScalarInt64() == 100);

    data.SetDataScalarDouble(3.14);
    REQUIRE(detail_data->GetDataScalarDouble() == 3.14);

    data.SetDataScalarString("hello");
    REQUIRE(detail_data->GetDataScalarString() == "hello");

    data.SetDataScalarBool(true);
    REQUIRE(detail_data->GetDataScalarBool() == true);
}
