
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

#include <algorithm>
#include <iostream>
#include <random>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }

    /******************* Partition With A Random Projection Tree *****************/
    vsag::RPTPartitionParams params;
    params.bucket_size = 1000;  // every partition holds at most 1000 vectors
    params.seed = 42;           // same input and seed => same partitions

    auto result = vsag::rpt_partition(dim, num_vectors, datas.data(), params);
    if (not result.has_value()) {
        std::cerr << "rpt_partition failed: " << result.error().message << std::endl;
        return -1;
    }
    const auto& partitions = result.value();

    /******************* Inspect The Partitions *****************/
    uint64_t max_size = 0;
    uint64_t min_size = num_vectors;
    for (const auto& partition : partitions) {
        max_size = std::max<uint64_t>(max_size, partition.size());
        min_size = std::min<uint64_t>(min_size, partition.size());
    }
    std::cout << "num partitions: " << partitions.size() << std::endl;
    std::cout << "largest partition: " << max_size << std::endl;
    std::cout << "smallest partition: " << min_size << std::endl;
    std::cout << "first ids of partition 0:";
    for (uint64_t i = 0; i < 5 && i < partitions[0].size(); ++i) {
        std::cout << " " << partitions[0][i];
    }
    std::cout << std::endl;

    return 0;
}
