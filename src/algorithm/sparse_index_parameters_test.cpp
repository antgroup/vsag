

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

#include "algorithm/sparse_index_parameters.h"

#include <catch2/catch_test_macros.hpp>

#include "index_common_param.h"
#include "vsag/logger.h"

TEST_CASE("SparseIndexParameters FromJson and ToJson", "[ut][sparse_index_parameters]") {
    vsag::SparseIndexParameters params;

    SECTION("ToJson returns correct value") {
        auto json = params.ToJson();
        REQUIRE(json.Contains("need_sort"));
    }

    SECTION("FromJson with need_sort true") {
        vsag::JsonWrapper json;
        json.SetBool(true);
        vsag::JsonWrapper root;
        root.SetJson(json);
        params.FromJson(root);
    }

    SECTION("FromJson with need_sort false") {
        vsag::JsonWrapper json;
        json.SetBool(false);
        vsag::JsonWrapper root;
        root.SetJson(json);
        params.FromJson(root);
    }
}
