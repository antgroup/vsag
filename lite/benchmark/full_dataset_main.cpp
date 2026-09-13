// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <vsag/dataset.h>
#include <vsag/factory.h>
#include <vsag/index.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Neighbor {
    int64_t id;
    float distance;
};

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

std::shared_ptr<vsag::Index>
create_index(int32_t dim) {
    const auto parameters =
        std::string(R"({"dtype":"float32","metric_type":"l2","dim":)") + std::to_string(dim) +
        R"(,"index_param":{"base_quantization_type":"fp32","store_raw_vector":true}})";
    auto created = vsag::Factory::CreateIndex("brute_force", parameters);
    require(static_cast<bool>(created), "index create failed");
    return *created;
}

std::vector<Neighbor>
search(const std::shared_ptr<vsag::Index>& index, const vsag::DatasetPtr& query, int32_t k) {
    auto result = index->KnnSearch(query, k, "");
    require(static_cast<bool>(result) and (*result)->GetDim() == k, "search failed");
    std::vector<Neighbor> neighbors;
    neighbors.reserve(k);
    for (int32_t i = 0; i < k; ++i) {
        neighbors.push_back({(*result)->GetIds()[i], (*result)->GetDistances()[i]});
    }
    return neighbors;
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
    auto index = create_index(dim);
    int64_t id = 0;
    std::vector<float> vector(dim);
    auto one = vsag::Dataset::Make()
                   ->NumElements(1)
                   ->Dim(dim)
                   ->Ids(&id)
                   ->Float32Vectors(vector.data())
                   ->Owner(false);
    auto start = Clock::now();
    for (uint64_t i = 0; i < base.size(); ++i) {
        id = static_cast<int64_t>(i);
        std::copy(base[i].begin(), base[i].end(), vector.begin());
        auto added = index->Add(one);
        require(static_cast<bool>(added) and added->empty(), "index add failed");
    }
    const auto build_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto query_data = vsag::Dataset::Make()
                          ->NumElements(1)
                          ->Dim(dim)
                          ->Float32Vectors(vector.data())
                          ->Owner(false);
    uint64_t hits = 0;
    std::vector<double> latencies;
    std::vector<std::vector<Neighbor>> before;
    for (uint64_t i = 0; i < queries.size(); ++i) {
        std::copy(queries[i].begin(), queries[i].end(), vector.begin());
        start = Clock::now();
        auto result = search(index, query_data, k);
        latencies.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        require(result.size() == static_cast<uint64_t>(k), "search failed");
        std::unordered_set<int64_t> expected(truth[i].begin(), truth[i].end());
        for (const auto& neighbor : result) {
            hits += expected.count(neighbor.id);
        }
        before.push_back(std::move(result));
    }
    start = Clock::now();
    std::ofstream output(snapshot, std::ios::binary);
    require(static_cast<bool>(output) and static_cast<bool>(index->Serialize(output)),
            "save failed");
    output.close();
    require(static_cast<bool>(output), "snapshot close failed");
    const auto save_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    index.reset();
    auto loaded = create_index(dim);
    start = Clock::now();
    std::ifstream input(snapshot, std::ios::binary);
    require(static_cast<bool>(loaded->Deserialize(input)) and
                loaded->GetNumElements() == static_cast<int64_t>(base.size()),
            "load failed");
    const auto load_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (uint64_t i = 0; i < queries.size(); ++i) {
        std::copy(queries[i].begin(), queries[i].end(), vector.begin());
        auto after = search(loaded, query_data, k);
        require(after.size() == before[i].size(), "post-load search failed");
        for (uint64_t j = 0; j < before[i].size(); ++j) {
            require(after[j].id == before[i][j].id and
                        std::abs(after[j].distance - before[i][j].distance) <= 1e-5F,
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
        require(argc == 3, "usage: full_dataset_benchmark DATASET_DIRECTORY SNAPSHOT_PATH");
        return run(argv[1], argv[2]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
