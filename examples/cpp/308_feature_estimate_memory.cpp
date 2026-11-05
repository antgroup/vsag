
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

    const uint64_t element_count = 100000;

    /******************* Estimate Final Index Memory *****************/
    if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
        uint64_t final_index_bytes = index->EstimateMemory(element_count);
        std::cout << "Index supports EstimateMemory. For " << element_count
                  << " elements, the final index memory estimate is " << final_index_bytes
                  << " bytes" << std::endl;
    }

    /******************* Estimate Peak Build Memory *****************/
    if (index->CheckFeature(vsag::SUPPORT_ESTIMATE_BUILD_MEMORY)) {
        uint64_t peak_bytes = index->EstimateBuildMemory(element_count);
        std::cout << "Index supports EstimateBuildMemory. For " << element_count
                  << " elements, the peak build memory estimate is " << peak_bytes << " bytes"
                  << std::endl;
    } else {
        std::cout << "EstimateBuildMemory is unsupported. This index does not provide a peak "
                     "build-memory estimate."
                  << std::endl;
    }

    engine.Shutdown();
    return 0;
}
