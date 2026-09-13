// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/lite/index.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void
require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

int32_t
read_dim(std::istream& input) {
    int32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    require(static_cast<bool>(input) and dim > 0 and dim <= 4096, "invalid record dimension");
    return dim;
}

template <typename T>
std::vector<std::vector<T>>
read_records(const std::string& path, int32_t expected_dim) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "input open failed");
    const auto size = input.tellg();
    require(size > 0 and size % (sizeof(int32_t) + sizeof(T) * expected_dim) == 0,
            "invalid input file length");
    const auto count = static_cast<uint64_t>(size / (sizeof(int32_t) + sizeof(T) * expected_dim));
    input.seekg(0);
    std::vector<std::vector<T>> records;
    records.reserve(count);
    for (uint64_t i = 0; i < count; ++i) {
        require(read_dim(input) == expected_dim, "inconsistent record dimension");
        std::vector<T> row(expected_dim);
        input.read(reinterpret_cast<char*>(row.data()), sizeof(T) * expected_dim);
        require(static_cast<bool>(input), "truncated record");
        records.push_back(std::move(row));
    }
    return records;
}

double
percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto rank =
        static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(values.size()))) - 1;
    return values[rank];
}

int
run(const std::string& directory, const std::string& snapshot) {
    const auto base_path = directory + "/base.fvecs";
    const auto query_path = directory + "/queries.fvecs";
    const auto truth_path = directory + "/groundtruth.ivecs";
    const int32_t dim = [&] {
        std::ifstream file(base_path, std::ios::binary);
        require(static_cast<bool>(file), "base open failed");
        return read_dim(file);
    }();
    const int32_t k = [&] {
        std::ifstream file(truth_path, std::ios::binary);
        require(static_cast<bool>(file), "groundtruth open failed");
        return read_dim(file);
    }();
    require(not std::filesystem::exists(snapshot), "snapshot path already exists");
    auto base = read_records<float>(base_path, dim);
    auto queries = read_records<float>(query_path, dim);
    auto truth = read_records<int32_t>(truth_path, k);
    require(queries.size() == truth.size(), "query and groundtruth count differ");
    require(not base.empty() and not queries.empty() and static_cast<uint64_t>(k) <= base.size(),
            "invalid dataset shape");
    auto created = vsag::lite::Index::Create(dim);
    require(static_cast<bool>(created), "index create failed");
    auto index = std::move(*created);
    auto start = Clock::now();
    for (uint64_t i = 0; i < base.size(); ++i) {
        auto added = index->Add(static_cast<int64_t>(i), base[i].data(), dim);
        require(static_cast<bool>(added), "index add failed");
    }
    const auto build_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    uint64_t hits = 0;
    std::vector<double> latencies;
    std::vector<std::vector<vsag::lite::Neighbor>> before;
    for (uint64_t i = 0; i < queries.size(); ++i) {
        start = Clock::now();
        auto result = index->Search(queries[i].data(), dim, k);
        latencies.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        require(static_cast<bool>(result) and result->size() == static_cast<uint64_t>(k),
                "search failed");
        std::unordered_set<int64_t> expected(truth[i].begin(), truth[i].end());
        for (const auto& neighbor : *result) {
            hits += expected.count(neighbor.id);
        }
        before.push_back(std::move(*result));
    }
    start = Clock::now();
    std::ofstream output(snapshot, std::ios::binary);
    require(static_cast<bool>(output) and static_cast<bool>(index->Save(output)), "save failed");
    output.close();
    require(static_cast<bool>(output), "snapshot close failed");
    const auto save_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    start = Clock::now();
    std::ifstream input(snapshot, std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    require(static_cast<bool>(loaded) and (*loaded)->Size() == base.size(), "load failed");
    const auto load_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (uint64_t i = 0; i < queries.size(); ++i) {
        auto after = (*loaded)->Search(queries[i].data(), dim, k);
        require(static_cast<bool>(after) and after->size() == before[i].size(),
                "post-load search failed");
        for (uint64_t j = 0; j < before[i].size(); ++j) {
            require(
                (*after)[j].id == before[i][j].id and (*after)[j].distance == before[i][j].distance,
                "post-load result changed");
        }
    }
    std::cout << "base_count,query_count,dim,k,recall_at_k,build_ms,search_p50_us,"
                 "search_p99_us,save_ms,load_ms,snapshot_bytes\n";
    std::cout << base.size() << ',' << queries.size() << ',' << dim << ',' << k << ',' << std::fixed
              << std::setprecision(6)
              << static_cast<double>(hits) / static_cast<double>(queries.size() * k) << ','
              << build_ms << ',' << percentile(latencies, 0.50) << ','
              << percentile(latencies, 0.99) << ',' << save_ms << ',' << load_ms << ','
              << std::filesystem::file_size(snapshot) << '\n';
    return 0;
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: lite_dataset_benchmark DATASET_DIRECTORY SNAPSHOT_PATH");
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
