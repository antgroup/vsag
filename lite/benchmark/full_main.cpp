// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>
#include <vsag/dataset.h>
#include <vsag/factory.h>
#include <vsag/index.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
    uint64_t count;
    uint64_t dim;
    uint64_t queries;
    uint64_t k;
    uint64_t crud_ops;
    uint64_t seed;
    std::string snapshot;
};

struct Neighbor {
    int64_t id;
    float distance;
};

uint64_t
parse_uint(const char* text, const char* name) {
    std::string value(text);
    std::string::size_type consumed = 0;
    uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return parsed;
}

Config
parse_config(int argc, char** argv) {
    if (argc != 8) {
        throw std::invalid_argument(
            "usage: full_benchmark COUNT DIM QUERIES K CRUD_OPS SEED SNAPSHOT_PATH");
    }
    Config config{parse_uint(argv[1], "count"),
                  parse_uint(argv[2], "dimension"),
                  parse_uint(argv[3], "queries"),
                  parse_uint(argv[4], "k"),
                  parse_uint(argv[5], "CRUD operation count"),
                  parse_uint(argv[6], "seed"),
                  argv[7]};
    if (config.count == 0 or config.dim == 0 or config.queries == 0 or config.k == 0 or
        config.crud_ops == 0 or config.k > config.count or config.crud_ops > config.count or
        config.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) or
        config.dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) or
        config.k > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("configuration values are out of range");
    }
    if (std::filesystem::exists(config.snapshot)) {
        throw std::invalid_argument("snapshot path already exists");
    }
    return config;
}

uint64_t
mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

void
make_vector(
    uint64_t id, uint64_t variant, uint64_t dim, uint64_t seed, std::vector<float>& vector) {
    for (uint64_t d = 0; d < dim; ++d) {
        uint64_t bits = mix(seed);
        bits = mix(bits ^ id);
        bits = mix(bits ^ variant);
        bits = mix(bits ^ d);
        vector[d] = static_cast<float>(bits >> 40U) / 16777216.0F;
    }
}

double
milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double
microseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double
percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto rank =
        static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(values.size()))) - 1;
    return values[std::min<uint64_t>(rank, values.size() - 1)];
}

void
require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

uint64_t
result_checksum(const std::vector<Neighbor>& result, uint64_t checksum) {
    for (const auto& neighbor : result) {
        uint32_t distance_bits = 0;
        std::memcpy(&distance_bits, &neighbor.distance, sizeof(distance_bits));
        checksum ^= mix(static_cast<uint64_t>(neighbor.id));
        checksum *= 1099511628211ULL;
        checksum ^= distance_bits;
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

std::vector<Neighbor>
search(const std::shared_ptr<vsag::Index>& index, const vsag::DatasetPtr& query, uint64_t k) {
    auto result = index->KnnSearch(query, static_cast<int64_t>(k), "");
    require(static_cast<bool>(result), "search failed");
    require((*result)->GetDim() == static_cast<int64_t>(k), "unexpected search result size");
    std::vector<Neighbor> neighbors;
    neighbors.reserve(k);
    for (uint64_t i = 0; i < k; ++i) {
        neighbors.push_back({(*result)->GetIds()[i], (*result)->GetDistances()[i]});
    }
    return neighbors;
}

bool
same_results(const std::vector<Neighbor>& left, const std::vector<Neighbor>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (uint64_t i = 0; i < left.size(); ++i) {
        if (left[i].id != right[i].id or std::abs(left[i].distance - right[i].distance) > 1e-5F) {
            return false;
        }
    }
    return true;
}

uint64_t
peak_rss_kib() {
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    return static_cast<uint64_t>(usage.ru_maxrss);
}

std::shared_ptr<vsag::Index>
create_index(uint64_t dim) {
    const auto parameters =
        std::string(R"({"dtype":"float32","metric_type":"l2","dim":)") + std::to_string(dim) +
        R"(,"index_param":{"base_quantization_type":"fp32","store_raw_vector":true}})";
    auto created = vsag::Factory::CreateIndex("brute_force", parameters);
    require(static_cast<bool>(created), "index creation failed");
    return *created;
}

int
run(const Config& config) {
    auto index = create_index(config.dim);
    std::vector<float> vector(config.dim);
    std::vector<uint64_t> variants(config.count, 0);
    int64_t current_id = 0;
    auto one = vsag::Dataset::Make()
                   ->NumElements(1)
                   ->Dim(static_cast<int64_t>(config.dim))
                   ->Ids(&current_id)
                   ->Float32Vectors(vector.data())
                   ->Owner(false);

    const auto build_start = Clock::now();
    for (uint64_t id = 0; id < config.count; ++id) {
        current_id = static_cast<int64_t>(id);
        make_vector(id, 0, config.dim, config.seed, vector);
        auto added = index->Add(one);
        require(static_cast<bool>(added) and added->empty(), "add failed");
    }
    const double build_ms = milliseconds(build_start, Clock::now());

    std::vector<double> update_us;
    std::vector<double> remove_us;
    std::vector<double> readd_us;
    update_us.reserve(config.crud_ops);
    remove_us.reserve(config.crud_ops);
    readd_us.reserve(config.crud_ops);
    for (uint64_t operation = 0; operation < config.crud_ops; ++operation) {
        const uint64_t id = (operation * 8191ULL) % config.count;
        current_id = static_cast<int64_t>(id);
        variants[id] = operation + 1;
        make_vector(id, variants[id], config.dim, config.seed, vector);

        auto start = Clock::now();
        auto updated = index->UpdateVector(current_id, one, true);
        require(static_cast<bool>(updated) and *updated, "update failed");
        update_us.push_back(microseconds(start, Clock::now()));

        start = Clock::now();
        auto removed = index->Remove(current_id, vsag::RemoveMode::FORCE_REMOVE);
        require(static_cast<bool>(removed) and *removed == 1, "remove failed");
        remove_us.push_back(microseconds(start, Clock::now()));

        start = Clock::now();
        auto added = index->Add(one);
        require(static_cast<bool>(added) and added->empty(), "re-add failed");
        readd_us.push_back(microseconds(start, Clock::now()));
    }

    auto query_data = vsag::Dataset::Make()
                          ->NumElements(1)
                          ->Dim(static_cast<int64_t>(config.dim))
                          ->Float32Vectors(vector.data())
                          ->Owner(false);
    std::vector<double> search_us;
    std::vector<std::vector<Neighbor>> expected;
    search_us.reserve(config.queries);
    expected.reserve(config.queries);
    uint64_t top1_hits = 0;
    uint64_t checksum = 1469598103934665603ULL;
    for (uint64_t query_number = 0; query_number < config.queries; ++query_number) {
        const uint64_t id = (query_number * 104729ULL) % config.count;
        make_vector(id, variants[id], config.dim, config.seed, vector);
        const auto start = Clock::now();
        auto result = search(index, query_data, config.k);
        search_us.push_back(microseconds(start, Clock::now()));
        top1_hits += result.front().id == static_cast<int64_t>(id) ? 1 : 0;
        checksum = result_checksum(result, checksum);
        expected.push_back(std::move(result));
    }

    const auto save_start = Clock::now();
    std::ofstream output(config.snapshot, std::ios::binary);
    require(static_cast<bool>(output), "snapshot open for write failed");
    require(static_cast<bool>(index->Serialize(output)), "snapshot save failed");
    output.close();
    require(static_cast<bool>(output), "snapshot close failed");
    const double save_ms = milliseconds(save_start, Clock::now());
    const uint64_t snapshot_bytes = std::filesystem::file_size(config.snapshot);
    index.reset();

    auto loaded_index = create_index(config.dim);
    const auto load_start = Clock::now();
    std::ifstream input(config.snapshot, std::ios::binary);
    require(static_cast<bool>(loaded_index->Deserialize(input)), "snapshot load failed");
    const double load_ms = milliseconds(load_start, Clock::now());

    std::vector<double> loaded_search_us;
    loaded_search_us.reserve(config.queries);
    uint64_t loaded_checksum = 1469598103934665603ULL;
    for (uint64_t query_number = 0; query_number < config.queries; ++query_number) {
        const uint64_t id = (query_number * 104729ULL) % config.count;
        make_vector(id, variants[id], config.dim, config.seed, vector);
        const auto start = Clock::now();
        auto result = search(loaded_index, query_data, config.k);
        loaded_search_us.push_back(microseconds(start, Clock::now()));
        require(same_results(result, expected[query_number]), "loaded search result changed");
        loaded_checksum = result_checksum(result, loaded_checksum);
    }
    // Full VSAG may recompute equivalent FP32 distances with a small rounding
    // difference after serialization. IDs and ordering are checked exactly;
    // same_results applies the distance tolerance above.

    std::cout
        << "implementation,count,dim,queries,k,crud_ops,seed,build_ms,update_p50_us,"
           "update_p99_us,remove_p50_us,remove_p99_us,readd_p50_us,readd_p99_us,"
           "search_p50_us,search_p99_us,load_search_p50_us,load_search_p99_us,save_ms,"
           "warm_load_ms,snapshot_bytes,peak_rss_kib,top1_recall,result_checksum,final_size\n";
    std::cout << std::fixed << std::setprecision(3) << "full," << config.count << ',' << config.dim
              << ',' << config.queries << ',' << config.k << ',' << config.crud_ops << ','
              << config.seed << ',' << build_ms << ',' << percentile(update_us, 0.50) << ','
              << percentile(update_us, 0.99) << ',' << percentile(remove_us, 0.50) << ','
              << percentile(remove_us, 0.99) << ',' << percentile(readd_us, 0.50) << ','
              << percentile(readd_us, 0.99) << ',' << percentile(search_us, 0.50) << ','
              << percentile(search_us, 0.99) << ',' << percentile(loaded_search_us, 0.50) << ','
              << percentile(loaded_search_us, 0.99) << ',' << save_ms << ',' << load_ms << ','
              << snapshot_bytes << ',' << peak_rss_kib() << ','
              << static_cast<double>(top1_hits) / static_cast<double>(config.queries) << ','
              << checksum << ',' << loaded_index->GetNumElements() << '\n';
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        return run(parse_config(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
