
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

#include <vsag/vsag.h>

#include <iostream>

int
main(int argc, char** argv) {
    vsag::init();
    /******************* Create HGraph Index *****************/
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Check Feature *****************/
    if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
        auto estimate_memory = index->EstimateMemory(100'000);
        std::cout << "Index Support EstimateMemory, when given 100000 vectors to build, the "
                     "estimate memory is "
                  << estimate_memory << " byte" << std::endl;
    }

    if (not index->CheckFeature(vsag::SUPPORT_DELETE_BY_ID)) {
        std::cout << "Index doesn't support DeleteById" << std::endl;
        auto result = index->UpdateId(18, 37);
        std::cout << "When UpdateId, you will get error:";
        if (not result.has_value()) {
            std::cout << result.error().message << std::endl;
        }
    }

    engine.Shutdown();
    return 0;
}
