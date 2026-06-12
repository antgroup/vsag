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

#include "mci_parameter.h"

#include "datacell/flatten_datacell_parameter.h"
#include "parameter_test.h"
#include "unittest.h"

namespace {

std::string
generate_mci_param() {
    return R"({
        "type": "mci",
        "use_reorder": false,
        "base_codes": {
            "codes_type": "flatten",
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        },
        "max_degree": 32,
        "mcs": 200,
        "clique_max": 50,
        "alpha": 1.2,
        "knng_path": "/tmp/knng.bin"
    })";
}

}  // namespace

TEST_CASE("MCI Parameters Test", "[ut][MCIParameter]") {
    auto param = std::make_shared<vsag::MCIParameter>();
    param->FromString(generate_mci_param());

    REQUIRE(param->max_degree == 32);
    REQUIRE(param->mcs == 200);
    REQUIRE(param->clique_max == 50);
    REQUIRE(param->alpha == 1.2F);
    REQUIRE(param->join_ratio_threshold == 0.6F);
    REQUIRE(param->added_mct == 3);
    REQUIRE(param->knng_path == "/tmp/knng.bin");
    REQUIRE(param->base_codes_param != nullptr);
    REQUIRE(param->base_codes_param->quantizer_parameter->GetTypeName() == "fp32");

    vsag::ParameterTest::TestToJson(param);

    auto custom_json = vsag::JsonType::Parse(generate_mci_param());
    custom_json[vsag::MCI_PARAMETER_JOIN_RATIO_THRESHOLD].SetFloat(0.75F);
    custom_json[vsag::MCI_PARAMETER_ADDED_MCT].SetInt(5);
    auto custom_param = std::make_shared<vsag::MCIParameter>();
    custom_param->FromJson(custom_json);
    REQUIRE(custom_param->join_ratio_threshold == 0.75F);
    REQUIRE(custom_param->added_mct == 5);

    auto search_param = vsag::MCISearchParameters::FromJson(R"({
        "mci": {
            "ef_search": 120,
            "seed_count": 64,
            "hops_limit": 1000,
            "rabitq_one_bit_search": true,
            "parallelism": 2
        }
    })");
    REQUIRE(search_param.ef_search == 120);
    REQUIRE(search_param.seed_count == 64);
    REQUIRE(search_param.hops_limit == 1000);
    REQUIRE(search_param.rabitq_one_bit_search);
    REQUIRE(search_param.parallel_search_thread_count == 2);
}

TEST_CASE("MCI Parameters CheckCompatibility", "[ut][MCIParameter][CheckCompatibility]") {
    auto param = std::make_shared<vsag::MCIParameter>();
    param->FromString(generate_mci_param());
    REQUIRE(param->CheckCompatibility(param));
    REQUIRE_FALSE(param->CheckCompatibility(std::make_shared<vsag::EmptyParameter>()));

    auto changed = std::make_shared<vsag::MCIParameter>();
    auto json = vsag::JsonType::Parse(generate_mci_param());
    json["mcs"].SetInt(100);
    changed->FromJson(json);
    REQUIRE_FALSE(param->CheckCompatibility(changed));

    auto hybrid = std::make_shared<vsag::MCIParameter>();
    auto hybrid_json = vsag::JsonType::Parse(generate_mci_param());
    hybrid_json[vsag::MCI_PARAMETER_USE_HGRAPH_HYBRID].SetBool(true);
    hybrid_json[vsag::MCI_PARAMETER_HGRAPH_INDEX_PARAM].SetJson(vsag::JsonType::Parse(R"({
        "type": "hgraph",
        "use_reorder": false,
        "base_codes": {
            "codes_type": "flatten_codes",
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        },
        "precise_codes": {
            "codes_type": "flatten_codes",
            "io_params": {
                "type": "memory_io"
            },
            "quantization_params": {
                "type": "fp32"
            }
        },
        "graph": {
            "io_params": {
                "type": "memory_io"
            },
            "max_degree": 32,
            "graph_storage_type": "flat",
            "init_capacity": 100,
            "support_remove": false,
            "remove_flag_bit": 0
        }
    })"));
    hybrid->FromJson(hybrid_json);
    REQUIRE(param->CheckCompatibility(hybrid));
    REQUIRE(hybrid->CheckCompatibility(param));
}
