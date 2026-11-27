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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// Example: Hamming distance over bit-packed binary vectors with the brute_force index.
//
// For binary vectors, "dim" is the number of logical bits. Each vector is stored as a
// contiguous byte array of ceil(dim / 8) bytes. Bits are packed least-significant-bit
// first inside each byte, and the unused high bits of the final byte are ignored by the
// Hamming distance computation:
//
//     hamming(a, b) = popcount(a XOR b)

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 256;  // number of logical bits
    int64_t bytes_per_vector = (dim + 7) / 8;

    std::vector<int64_t> ids(num_vectors);
    std::vector<uint8_t> datas(num_vectors * bytes_per_vector);
    std::mt19937 rng(47);
    std::uniform_int_distribution<int> distrib_byte(0, 255);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (auto& byte : datas) {
        byte = static_cast<uint8_t>(distrib_byte(rng));
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->BinaryVectors(datas.data())
        ->Owner(false);

    /******************* Create BruteForce Index *****************/
    std::string index_parameters = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 256
    }
    )";
    auto index = vsag::Factory::CreateIndex("brute_force", index_parameters).value();

    /******************* Build BruteForce Index *****************/
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index BruteForce contains: " << index->GetNumElements()
                  << std::endl;
    } else {
        std::cerr << "Failed to build index: " << build_result.error().message << std::endl;
        return -1;
    }

    /******************* Prepare Query Dataset *****************/
    // Query with a copy of the last base vector flipped in two bits, so the exact
    // Hamming distance to that vector must be 2.
    std::vector<uint8_t> query_vector(bytes_per_vector);
    std::memcpy(
        query_vector.data(), datas.data() + (num_vectors - 1) * bytes_per_vector, bytes_per_vector);
    query_vector[0] ^= 0x05;  // flip two bits

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->BinaryVectors(query_vector.data())->Owner(false);

    /******************* KnnSearch For BruteForce Index *****************/
    int64_t topk = 10;
    auto result = index->KnnSearch(query, topk, R"({})").value();

    /******************* Verify and Print Search Result *****************/
    auto popcount = [](const uint8_t* x, const uint8_t* y, int64_t num_bytes) {
        uint64_t distance = 0;
        for (int64_t i = 0; i < num_bytes; ++i) {
            uint8_t diff = static_cast<uint8_t>(x[i] ^ y[i]);
            while (diff != 0) {
                diff = static_cast<uint8_t>(diff & (diff - 1));
                ++distance;
            }
        }
        return distance;
    };

    std::cout << "results (id: distance):" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        int64_t id = result->GetIds()[i];
        auto expected =
            popcount(query_vector.data(), datas.data() + id * bytes_per_vector, bytes_per_vector);
        if (static_cast<uint64_t>(result->GetDistances()[i]) != expected) {
            std::cerr << "distance mismatch for id " << id << std::endl;
            return -1;
        }
        std::cout << id << ": " << result->GetDistances()[i] << std::endl;
    }

    if (result->GetIds()[0] != num_vectors - 1 || result->GetDistances()[0] != 2.0F) {
        std::cerr << "nearest neighbor verification failed" << std::endl;
        return -1;
    }
    std::cout << "verified: nearest id " << result->GetIds()[0] << " has Hamming distance "
              << result->GetDistances()[0] << std::endl;

    return 0;
}
