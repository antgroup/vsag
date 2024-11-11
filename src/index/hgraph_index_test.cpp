
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

#include "hgraph_index.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "fixtures.h"
#include "vsag/errors.h"

TEST_CASE("build with allocator", "[ut][hgraphindex]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    std::string json_str = R"(
        {
          "index_type": "hgraph",
          "metric_type": "l2",
          "dim": 128,
          "data_type": "fp32",
          "index_param": {
            "use_reorder": false,
            "graph": {
              "io_type": "block_memory",
              "io_params": {
                "block_size": 134217728
              },
              "type": "NSW",
              "graph_params": {
                "max_degree": 64,
                "init_capacity": 1000000
              }
            },
            "base_codes": {
              "io_type": "block_memory",
              "io_params": {
                "block_size": 134217728
              },
              "codes_type": "flatten_codes",
              "codes_param": {
              },
              "quantization_type": "sq8",
              "quantization_params": {
                "subspace": 64,
                "nbits": 8
              }
            }
          }
        })";
    auto json_param = nlohmann::json::parse(json_str);
    vsag::IndexCommonParam param;
    auto allocator = std::make_shared<vsag::DefaultAllocator>();
    param.dim_ = 128;
    param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    param.allocator_ = allocator.get();

    auto index = std::make_shared<vsag::HGraphIndex>(json_param["index_param"], param);
    index->Init();

    const int64_t num_elements = 10000;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, param.dim_);

    auto dataset = vsag::Dataset::Make();
    dataset->Dim(param.dim_)
        ->NumElements(num_elements)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto func = [&]() {
        for (auto i = 0; i < 100; ++i) {
            auto new_id = random() % num_elements;
            auto query = vsag::Dataset::Make();
            query->NumElements(1)
                ->Dim(param.dim_)
                ->Float32Vectors(vectors.data() + new_id * param.dim_)
                ->Owner(false);
            nlohmann::json params{
                {"hnsw", {{"ef_search", 100}}},
            };
            auto result2 = index->KnnSearch(query, 1, params.dump());
            REQUIRE(result2.has_value());
            auto vec = result2.value();
            REQUIRE(vec->GetIds()[0] == new_id);
        }
    };
    func();
    std::string dirname = "/tmp/hgraph_TestSerializeAndDeserialize_" + std::to_string(random());
    std::filesystem::create_directory(dirname);
    auto filename = dirname + "/file_" + std::to_string(random());
    std::ofstream outfile(filename.c_str(), std::ios::binary);
    IOStreamWriter writer(outfile);
    index->serialize(writer);
    outfile.close();

    index = std::make_shared<vsag::HGraphIndex>(json_param["index_param"], param);
    index->Init();
    std::ifstream infile(filename.c_str(), std::ios::binary);
    IOStreamReader reader(infile);
    index->deserialize(reader);
    func();

    infile.close();
}
