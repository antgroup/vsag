// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
std::vector<float>
read(const char* path, uint64_t count) {
    std::ifstream input(path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("input open failed");
    }
    std::vector<float> data(count * 960);
    for (uint64_t row = 0; row < count; ++row) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (dim != 960) {
            throw std::runtime_error("dimension differs");
        }
        input.read(reinterpret_cast<char*>(data.data() + row * 960), 960 * sizeof(float));
        if (not input) {
            throw std::runtime_error("truncated data");
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("extra data");
    }
    if (not std::all_of(data.begin(), data.end(), [](float v) { return std::isfinite(v); })) {
        throw std::runtime_error("nonfinite vector");
    }
    return data;
}
struct Neighbor {
    uint64_t id;
    double distance;
};
bool
closer(const Neighbor& left, const Neighbor& right) {
    return left.distance < right.distance or
           (left.distance == right.distance and left.id < right.id);
}
}  // namespace
int
main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("BASE_FVECS QUERY_FVECS required");
        }
        const auto base = read(argv[1], 10000);
        const auto queries = read(argv[2], 1000);
        std::vector<Neighbor> neighbors(10000);
        std::cout << "query,rank,id,distance\n" << std::hexfloat;
        for (uint64_t query = 0; query < 1000; ++query) {
            for (uint64_t id = 0; id < 10000; ++id) {
                double distance = 0.0;
                for (uint64_t col = 0; col < 960; ++col) {
                    const double delta = static_cast<double>(queries[query * 960 + col]) -
                                         static_cast<double>(base[id * 960 + col]);
                    distance += delta * delta;
                }
                neighbors[id] = {id, distance};
            }
            std::partial_sort(neighbors.begin(), neighbors.begin() + 10, neighbors.end(), closer);
            for (uint64_t rank = 0; rank < 10; ++rank) {
                std::cout << query << ',' << rank << ',' << neighbors[rank].id << ','
                          << neighbors[rank].distance << '\n';
            }
        }
        if (not std::cout) {
            throw std::runtime_error("oracle output failed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
