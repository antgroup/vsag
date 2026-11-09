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

// Binary vectors with the Hamming distance.
//
// A binary vector of `dim` bits is stored in ceil(dim / 8) bytes, packed MSB-first: logical bit i
// lives in byte i / 8 under the mask 0x80 >> (i % 8) (the layout of numpy.packbits). `dim` counts
// bits and does not have to be a multiple of eight; the unused low bits of the last byte are
// ignored. The distance is the number of differing bits, popcount(a XOR b).

#include <vsag/vsag.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t DIM = 100;  // 100 bits -> 13 bytes per vector, 4 padding bits
constexpr int64_t BYTES_PER_VECTOR = (DIM + 7) / 8;

// Packs a string of '0'/'1' characters (bit 0 first) into `out`.
void
pack_bit_string(const std::string& bits, uint8_t* out) {
    for (int64_t i = 0; i < BYTES_PER_VECTOR; ++i) {
        out[i] = 0;
    }
    for (int64_t i = 0; i < static_cast<int64_t>(bits.size()) and i < DIM; ++i) {
        if (bits[i] == '1') {
            out[i / 8] = static_cast<uint8_t>(out[i / 8] | (0x80U >> (i % 8)));
        }
    }
}

// Reference distance computed bit by bit, independently of VSAG.
int64_t
hamming(const uint8_t* a, const uint8_t* b) {
    int64_t distance = 0;
    for (int64_t i = 0; i < DIM; ++i) {
        const auto mask = static_cast<uint8_t>(0x80U >> (i % 8));
        distance += ((a[i / 8] & mask) != 0) != ((b[i / 8] & mask) != 0) ? 1 : 0;
    }
    return distance;
}

}  // namespace

int
main() {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    const int64_t num_vectors = 5000;
    std::vector<int64_t> ids(num_vectors);
    std::vector<uint8_t> vectors(num_vectors * BYTES_PER_VECTOR);
    std::mt19937 rng(47);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (auto& byte : vectors) {
        byte = static_cast<uint8_t>(rng());
    }
    // Vector 0 is written as a bit string to show the bit order explicitly.
    const std::string first_bits =
        "1011001110001111000011111000001111110000001111111000000011111111"
        "000000001111111110000000001111";
    pack_bit_string(first_bits, vectors.data());

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(DIM)  // number of bits, not bytes
        ->Ids(ids.data())
        ->BinaryVectors(vectors.data())
        ->Owner(false);

    /******************* Prepare Query Dataset *****************/
    // The query is vector 0 with three flipped bits, so vector 0 is at distance 3.
    std::vector<uint8_t> query_vector(vectors.begin(), vectors.begin() + BYTES_PER_VECTOR);
    for (int64_t bit : {2, 50, 99}) {
        query_vector[bit / 8] = static_cast<uint8_t>(query_vector[bit / 8] ^ (0x80U >> (bit % 8)));
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(DIM)->BinaryVectors(query_vector.data())->Owner(false);

    bool all_verified = true;
    auto verify = [&](const std::string& name, const vsag::DatasetPtr& result) {
        std::cout << name << " results:" << std::endl;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            const auto id = result->GetIds()[i];
            const auto distance = static_cast<int64_t>(result->GetDistances()[i]);
            const auto expected =
                hamming(query_vector.data(), vectors.data() + id * BYTES_PER_VECTOR);
            const bool ok = distance == expected;
            all_verified = all_verified and ok;
            std::cout << "  id=" << id << " hamming=" << distance
                      << (ok ? " (verified)" : " (MISMATCH)") << std::endl;
        }
    };

    /******************* BruteForce: exact search *****************/
    // "dtype": "binary" requires "metric_type": "hamming"; the index stores the bits exactly.
    const std::string build_parameters = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 100
    }
    )";
    auto brute_force = vsag::Factory::CreateIndex("brute_force", build_parameters).value();
    if (not brute_force->Build(base).has_value()) {
        std::cerr << "failed to build the brute_force index" << std::endl;
        return 1;
    }
    auto exact = brute_force->KnnSearch(query, 5, "{}").value();
    verify("brute_force knn", exact);
    all_verified = all_verified and exact->GetIds()[0] == 0 and exact->GetDistances()[0] == 3.0F;

    // Range search returns every vector whose Hamming distance is <= the radius.
    auto range = brute_force->RangeSearch(query, 3.0F, "{}").value();
    verify("brute_force range (radius 3)", range);

    /******************* HGraph: graph search *****************/
    // Binary codes are selected automatically; "base_quantization_type": "binary" is optional.
    const std::string hgraph_parameters = R"(
    {
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": 100,
        "index_param": {
            "max_degree": 32,
            "ef_construction": 200
        }
    }
    )";
    auto hgraph = vsag::Factory::CreateIndex("hgraph", hgraph_parameters).value();
    if (not hgraph->Build(base).has_value()) {
        std::cerr << "failed to build the hgraph index" << std::endl;
        return 1;
    }
    auto approximate = hgraph->KnnSearch(query, 5, R"({"hgraph": {"ef_search": 100}})").value();
    verify("hgraph knn", approximate);

    /******************* Distance by id *****************/
    auto distance = hgraph->CalcDistanceById(query, 0).value();
    std::cout << "distance to id 0 = " << distance << std::endl;
    all_verified = all_verified and distance == 3.0F;

    std::cout << (all_verified ? "all Hamming distances verified" : "verification FAILED")
              << std::endl;
    return all_verified ? 0 : 1;
}
