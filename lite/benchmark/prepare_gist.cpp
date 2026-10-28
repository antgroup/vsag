// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "lite/fp32_distance.h"

namespace {
void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}

uint32_t
read_u32(std::istream& input) {
    uint8_t bytes[4]{};
    input.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    require(static_cast<bool>(input), "truncated fbin header");
    return static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U) |
           (static_cast<uint32_t>(bytes[2]) << 16U) | (static_cast<uint32_t>(bytes[3]) << 24U);
}

void
write_u32(std::ostream& output, uint32_t value) {
    const uint8_t bytes[4]{static_cast<uint8_t>(value),
                           static_cast<uint8_t>(value >> 8U),
                           static_cast<uint8_t>(value >> 16U),
                           static_cast<uint8_t>(value >> 24U)};
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    require(static_cast<bool>(output), "dataset write failed");
}

struct Matrix {
    uint64_t count;
    uint64_t dim;
    std::vector<float> values;
};

Matrix
read_fbin_prefix(const std::filesystem::path& path, uint64_t requested) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open fbin input");
    const uint64_t declared_count = read_u32(input);
    const uint64_t dim = read_u32(input);
    require(requested > 0 and requested <= declared_count and dim > 0, "invalid fbin shape");
    require(requested <= std::numeric_limits<uint64_t>::max() / dim, "fbin size overflow");
    Matrix result{requested, dim, std::vector<float>(requested * dim)};
    input.read(reinterpret_cast<char*>(result.values.data()),
               static_cast<std::streamsize>(result.values.size() * sizeof(float)));
    require(static_cast<bool>(input), "truncated fbin payload");
    require(std::all_of(result.values.begin(),
                        result.values.end(),
                        [](float value) { return std::isfinite(value); }),
            "non-finite fbin value");
    return result;
}

void
write_fvecs(const std::filesystem::path& path, const Matrix& matrix, uint64_t count) {
    require(count <= matrix.count and matrix.dim <= std::numeric_limits<int32_t>::max(),
            "invalid fvecs output shape");
    std::ofstream output(path, std::ios::binary);
    require(static_cast<bool>(output), "cannot create fvecs output");
    for (uint64_t row = 0; row < count; ++row) {
        write_u32(output, static_cast<uint32_t>(matrix.dim));
        output.write(reinterpret_cast<const char*>(matrix.values.data() + row * matrix.dim),
                     static_cast<std::streamsize>(matrix.dim * sizeof(float)));
        require(static_cast<bool>(output), "fvecs write failed");
    }
}

struct Neighbor {
    uint64_t id;
    float distance;
};

bool
closer(const Neighbor& left, const Neighbor& right) {
    return left.distance < right.distance or
           (left.distance == right.distance and left.id < right.id);
}

void
write_ground_truth(const std::filesystem::path& path,
                   const Matrix& base,
                   uint64_t base_count,
                   const Matrix& queries,
                   uint64_t k) {
    require(base.dim == queries.dim and k > 0 and k <= base_count, "invalid ground-truth shape");
    const auto distance = vsag::lite::detail::select_fp32_distance();
    std::ofstream output(path, std::ios::binary);
    require(static_cast<bool>(output), "cannot create ivecs output");
    for (uint64_t query = 0; query < queries.count; ++query) {
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(&closer)> heap(&closer);
        for (uint64_t id = 0; id < base_count; ++id) {
            const Neighbor candidate{id,
                                     distance(queries.values.data() + query * queries.dim,
                                              base.values.data() + id * base.dim,
                                              base.dim)};
            if (heap.size() < k) {
                heap.push(candidate);
            } else if (closer(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
        std::vector<Neighbor> result(heap.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = heap.top();
            heap.pop();
        }
        write_u32(output, static_cast<uint32_t>(k));
        for (const auto& neighbor : result) {
            require(neighbor.id <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
                    "ground-truth ID overflow");
            write_u32(output, static_cast<uint32_t>(neighbor.id));
        }
    }
}

uint64_t
parse_positive(const char* value) {
    const std::string text(value);
    uint64_t parsed = 0;
    try {
        size_t consumed = 0;
        parsed = std::stoull(text, &consumed);
        require(consumed == text.size() and parsed > 0, "invalid positive integer");
    } catch (const std::exception&) {
        throw std::runtime_error("invalid positive integer");
    }
    return parsed;
}

void
prepare(const std::filesystem::path& base_path,
        const std::filesystem::path& query_path,
        const std::filesystem::path& output_root,
        uint64_t maximum_count,
        uint64_t query_count) {
    require(not std::filesystem::exists(output_root), "output directory already exists");
    const auto base = read_fbin_prefix(base_path, maximum_count);
    const auto queries = read_fbin_prefix(query_path, query_count);
    require(base.dim == 960 and queries.dim == 960, "expected 960-dimensional GIST data");
    constexpr uint64_t k = 10;
    const std::vector<uint64_t> counts = maximum_count >= 100000
                                             ? std::vector<uint64_t>{10000, 100000}
                                             : std::vector<uint64_t>{maximum_count};
    std::filesystem::create_directories(output_root);
    for (uint64_t count : counts) {
        require(count <= maximum_count, "requested subset exceeds base prefix");
        const auto directory = output_root / ("scale-" + std::to_string(count));
        std::filesystem::create_directory(directory);
        write_fvecs(directory / "base.fvecs", base, count);
        write_fvecs(directory / "queries.fvecs", queries, queries.count);
        write_ground_truth(directory / "groundtruth.ivecs", base, count, queries, k);
        std::ofstream manifest(directory / "manifest.json");
        require(static_cast<bool>(manifest), "cannot create manifest");
        manifest << "{\n"
                 << "  \"source\": \"maknee/gist1m fbin mirror of ANN_GIST1M\",\n"
                 << "  \"source_base_prefix\": " << count << ",\n"
                 << "  \"queries_prefix\": " << queries.count << ",\n"
                 << "  \"dimension\": " << base.dim << ",\n"
                 << "  \"k\": " << k << ",\n"
                 << "  \"groundtruth\": \"recomputed squared-L2, ties by ascending ID\"\n"
                 << "}\n";
        std::cout << directory << '\n';
    }
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: lite_prepare_gist BASE_FBIN QUERY_FBIN OUTPUT MAX_BASE QUERIES\n";
            return 2;
        }
        prepare(argv[1], argv[2], argv[3], parse_positive(argv[4]), parse_positive(argv[5]));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
