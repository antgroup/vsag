// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <malloc.h>
#include <sys/resource.h>
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

template <typename T>
struct Matrix {
    int32_t dim{0};
    uint64_t count{0};
    std::vector<T> values;
};

void
require(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
Matrix<T>
read_matrix(const std::string& path, int32_t expected_dim = 0) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "input open failed: " + path);
    Matrix<T> matrix;
    while (true) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (input.eof()) {
            break;
        }
        require(static_cast<bool>(input) and dim > 0 and dim <= 4096, "invalid record dimension");
        if (matrix.dim == 0) {
            matrix.dim = dim;
        }
        require(dim == matrix.dim and (expected_dim == 0 or dim == expected_dim),
                "inconsistent record dimension");
        const auto offset = matrix.values.size();
        matrix.values.resize(offset + static_cast<uint64_t>(dim));
        input.read(reinterpret_cast<char*>(matrix.values.data() + offset), sizeof(T) * dim);
        require(static_cast<bool>(input), "truncated record");
        ++matrix.count;
    }
    require(matrix.count > 0, "empty input");
    return matrix;
}

double
percentile(std::vector<double> values, double fraction) {
    require(not values.empty() and std::isfinite(fraction) and fraction >= 0.0 and fraction <= 1.0,
            "invalid percentile input");
    std::sort(values.begin(), values.end());
    const auto rank =
        std::max<uint64_t>(
            1, static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(values.size())))) -
        1;
    return values[rank];
}

uint64_t
current_rss_kib() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            uint64_t value = 0;
            std::string unit;
            status >> value >> unit;
            require(unit == "kB", "unexpected VmRSS unit");
            return value;
        }
        std::string remainder;
        std::getline(status, remainder);
    }
    throw std::runtime_error("VmRSS is unavailable");
}

std::string
create_parameters(int32_t dim, const std::string& mode) {
    const std::string prefix = std::string(R"({"dtype":"float32","metric_type":"l2","dim":)") +
                               std::to_string(dim) +
                               R"(,"index_param":{"max_degree":16,"ef_construction":128,)"
                               R"("graph_storage_type":"compressed",)";
    if (mode == "fp32") {
        return prefix + R"("base_quantization_type":"fp32","store_raw_vector":true}})";
    }
    if (mode == "rabitq1") {
        return prefix + R"("base_quantization_type":"rabitq","precise_quantization_type":"fp32",)"
                        R"("use_reorder":true,"store_raw_vector":true,"rabitq_use_fht":true,)"
                        R"("rabitq_bits_per_dim_query":32,"rabitq_bits_per_dim_base":1}})";
    }
    if (mode == "rabitq3x5") {
        return prefix +
               R"("base_quantization_type":"rabitq","precise_quantization_type":"rabitq",)"
               R"("use_reorder":true,"rabitq_use_fht":true,"rabitq_bits_per_dim_query":32,)"
               R"("rabitq_bits_per_dim_base":3,"rabitq_bits_per_dim_precise":5,)"
               R"("rabitq_error_rate":1.9}})";
    }
    throw std::invalid_argument("mode must be fp32, rabitq1, or rabitq3x5");
}

std::string
search_parameters(const std::string& mode) {
    return std::string(R"({"hgraph":{"ef_search":128,"rabitq_one_bit_search":)") +
           (mode == "fp32" ? "false" : "true") + "}}";
}

std::shared_ptr<vsag::Index>
create_index(int32_t dim, const std::string& mode) {
    auto created = vsag::Factory::CreateIndex("hgraph", create_parameters(dim, mode));
    require(static_cast<bool>(created), "index create failed");
    return *created;
}

std::vector<Neighbor>
search(const std::shared_ptr<vsag::Index>& index,
       const vsag::DatasetPtr& query,
       int32_t k,
       const std::string& parameters) {
    auto result = index->KnnSearch(query, k, parameters);
    require(static_cast<bool>(result) and (*result)->GetDim() == k, "search failed");
    std::vector<Neighbor> neighbors;
    neighbors.reserve(k);
    for (int32_t i = 0; i < k; ++i) {
        neighbors.push_back({(*result)->GetIds()[i], (*result)->GetDistances()[i]});
    }
    return neighbors;
}

int
run(const std::string& directory, const std::string& snapshot, const std::string& mode) {
    require(not std::filesystem::exists(snapshot), "snapshot path already exists");
    auto base = read_matrix<float>(directory + "/base.fvecs");
    const auto queries = read_matrix<float>(directory + "/queries.fvecs", base.dim);
    const auto truth = read_matrix<int32_t>(directory + "/groundtruth.ivecs");
    require(queries.count == truth.count and truth.dim > 0, "query and truth shape differ");
    require(static_cast<uint64_t>(truth.dim) <= base.count, "invalid Top-K");

    std::vector<int64_t> ids(base.count);
    for (uint64_t i = 0; i < base.count; ++i) {
        ids[i] = static_cast<int64_t>(i);
    }
    auto base_dataset = vsag::Dataset::Make()
                            ->NumElements(static_cast<int64_t>(base.count))
                            ->Dim(base.dim)
                            ->Ids(ids.data())
                            ->Float32Vectors(base.values.data())
                            ->Owner(false);
    auto index = create_index(base.dim, mode);
    auto start = Clock::now();
    auto built = index->Build(base_dataset);
    require(static_cast<bool>(built) and built->empty(), "index build failed");
    const auto build_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    base_dataset.reset();
    std::vector<float>().swap(base.values);
    std::vector<int64_t>().swap(ids);
    malloc_trim(0);
    const auto build_steady_rss_kib = current_rss_kib();

    std::vector<float> query_vector(base.dim);
    auto query_dataset = vsag::Dataset::Make()
                             ->NumElements(1)
                             ->Dim(base.dim)
                             ->Float32Vectors(query_vector.data())
                             ->Owner(false);
    const auto parameters = search_parameters(mode);
    uint64_t hits = 0;
    std::vector<double> latencies;
    std::vector<std::vector<Neighbor>> before;
    before.reserve(queries.count);
    for (uint64_t i = 0; i < queries.count; ++i) {
        std::copy_n(queries.values.data() + i * static_cast<uint64_t>(base.dim),
                    base.dim,
                    query_vector.data());
        start = Clock::now();
        auto result = search(index, query_dataset, truth.dim, parameters);
        latencies.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        const auto* expected_begin = truth.values.data() + i * static_cast<uint64_t>(truth.dim);
        const std::unordered_set<int64_t> expected(expected_begin, expected_begin + truth.dim);
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
    malloc_trim(0);
    auto loaded = create_index(base.dim, mode);
    start = Clock::now();
    std::ifstream input(snapshot, std::ios::binary);
    require(static_cast<bool>(input) and static_cast<bool>(loaded->Deserialize(input)),
            "load failed");
    const auto load_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    require(loaded->GetNumElements() == static_cast<int64_t>(base.count),
            "loaded index size differs");

    for (uint64_t i = 0; i < queries.count; ++i) {
        std::copy_n(queries.values.data() + i * static_cast<uint64_t>(base.dim),
                    base.dim,
                    query_vector.data());
        const auto after = search(loaded, query_dataset, truth.dim, parameters);
        require(after.size() == before[i].size(), "post-load result size changed");
        for (uint64_t j = 0; j < after.size(); ++j) {
            require(after[j].id == before[i][j].id, "post-load result ID changed");
            const auto scale =
                std::max({1.0F, std::abs(after[j].distance), std::abs(before[i][j].distance)});
            require(std::abs(after[j].distance - before[i][j].distance) <= 1e-5F * scale,
                    "post-load distance changed");
        }
    }

    malloc_trim(0);
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    std::cout << "mode,base_count,query_count,dim,k,recall_at_k,build_ms,search_p50_us,"
                 "search_p99_us,save_ms,load_ms,snapshot_bytes,build_steady_rss_kib,"
                 "final_steady_rss_kib,process_peak_rss_kib\n";
    std::cout << mode << ',' << base.count << ',' << queries.count << ',' << base.dim << ','
              << truth.dim << ',' << std::fixed << std::setprecision(6)
              << static_cast<double>(hits) / static_cast<double>(queries.count * truth.dim) << ','
              << build_ms << ',' << percentile(latencies, 0.50) << ','
              << percentile(latencies, 0.99) << ',' << save_ms << ',' << load_ms << ','
              << std::filesystem::file_size(snapshot) << ',' << build_steady_rss_kib << ','
              << current_rss_kib() << ',' << usage.ru_maxrss << '\n';
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        require(argc == 4,
                "usage: full_rabitq_dataset_benchmark DATASET_DIRECTORY SNAPSHOT_PATH "
                "MODE(fp32|rabitq1|rabitq3x5)");
        return run(argv[1], argv[2], argv[3]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
