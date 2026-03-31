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

#include <chrono>
#include <iostream>

int
main() {
    std::cout << "=== Pyramid Duplicate Detection Test ===" << std::endl;

    // Use smaller dataset first
    const int64_t NUM = 100;
    const int64_t DIM = 128;

    auto ids = new int64_t[NUM];
    auto vectors = new float[DIM * NUM];
    auto paths = new std::string[NUM];

    std::cout << "Generating " << NUM << " vectors..." << std::endl;

    // Generate base data
    for (int i = 0; i < NUM; ++i) {
        ids[i] = i;
        paths[i] = "a/b/c";
        for (int j = 0; j < DIM; ++j) {
            vectors[i * DIM + j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }

    // Create 5 duplicates (ID 10-14 copy ID 0-4)
    for (int i = 10; i < 15; ++i) {
        for (int j = 0; j < DIM; ++j) {
            vectors[i * DIM + j] = vectors[(i - 10) * DIM + j];
        }
    }

    std::cout << "Duplicates created: ID 10-14 are copies of ID 0-4" << std::endl;

    auto base = vsag::Dataset::Make();
    base->NumElements(NUM)->Dim(DIM)->Ids(ids)->Paths(paths)->Float32Vectors(vectors);

    // Simple configuration - NSW mode for duplicate detection
    auto params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "index_min_size": 10,
            "support_duplicate": true,
            "graph_type": "nsw",
            "ef_construction": 50
        }
    })";

    std::cout << "Creating index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto index = vsag::Factory::CreateIndex("pyramid", params);
    if (!index.has_value()) {
        std::cerr << "Failed to create index: " << index.error().message << std::endl;
        return 1;
    }

    std::cout << "Building index..." << std::endl;
    auto build_result = index.value()->Build(base);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    if (build_result.has_value()) {
        std::cout << "Build succeeded in " << duration.count() << " seconds" << std::endl;
        std::cout << "Elements in index: " << index.value()->GetNumElements() << std::endl;
        std::cout << "Expected: " << NUM << " total, less if duplicates detected" << std::endl;
    } else {
        std::cerr << "Build failed: " << build_result.error().message << std::endl;
    }

    // Note: ids, vectors, and paths are owned by the Dataset object
    // and will be automatically freed when the Dataset is destroyed.
    // Do not manually delete them here to avoid double-free.

    return 0;
}
