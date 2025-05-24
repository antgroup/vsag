
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

#include "compressed_graph_datacell.h"

#include <fmt/format-inl.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "graph_datacell_parameter.h"
#include "graph_interface_test.h"
#include "safe_allocator.h"
using namespace vsag;

void
TestCompressedGraphDataCell(const GraphInterfaceParamPtr& param,
                            const IndexCommonParam& common_param) {
    auto count = GENERATE(1000, 2000);
    auto max_id = 10000;

    auto graph = GraphInterface::MakeInstance(param, common_param);
    GraphInterfaceTest test(graph, true);
    auto other = GraphInterface::MakeInstance(param, common_param);
    test.BasicTest(max_id, count, other, false);
}

TEST_CASE("CompressedGraphDataCell Basic Test", "[ut][CompressedGraphDataCell]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dim = GENERATE(32, 64);
    auto max_degree = GENERATE(5, 12, 32, 64, 128);
    auto max_capacity = 10000;

    IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.allocator_ = allocator;
    auto graph_param = std::make_shared<CompressedGraphDatacellParameter>();
    graph_param->max_degree_ = max_degree;
    graph_param->graph_storage_type_ = GraphStorageTypes::GRAPH_STORAGE_TYPE_COMPRESSED;
    TestCompressedGraphDataCell(graph_param, common_param);
}
