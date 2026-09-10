// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>
#include <vsag/lite/index.h>

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
#include <sstream>
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

struct StabilityConfig {
    uint64_t count;
    uint64_t dim;
    uint64_t rounds;
    uint64_t crud_ops;
    uint64_t queries;
    uint64_t k;
    uint64_t seed;
    std::string snapshot_directory;
};

struct LoadConfig {
    std::string snapshot;
    uint64_t dim;
    uint64_t query_id;
    uint64_t query_variant;
    uint64_t queries;
    uint64_t k;
    uint64_t seed;
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
            "usage: lite_benchmark COUNT DIM QUERIES K CRUD_OPS SEED SNAPSHOT_PATH");
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
        config.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("configuration values are out of range");
    }
    if (std::filesystem::exists(config.snapshot)) {
        throw std::invalid_argument("snapshot path already exists");
    }
    return config;
}

StabilityConfig
parse_stability_config(int argc, char** argv) {
    if (argc != 10) {
        throw std::invalid_argument(
            "usage: lite_benchmark stability COUNT DIM ROUNDS CRUD_OPS QUERIES K SEED "
            "SNAPSHOT_DIRECTORY");
    }
    StabilityConfig config{parse_uint(argv[2], "count"),
                           parse_uint(argv[3], "dimension"),
                           parse_uint(argv[4], "round count"),
                           parse_uint(argv[5], "CRUD operation count"),
                           parse_uint(argv[6], "query count"),
                           parse_uint(argv[7], "k"),
                           parse_uint(argv[8], "seed"),
                           argv[9]};
    if (config.count == 0 or config.dim == 0 or config.rounds == 0 or config.crud_ops == 0 or
        config.queries == 0 or config.k == 0 or config.k > config.count or
        config.crud_ops > config.count or
        config.count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("configuration values are out of range");
    }
    if (std::filesystem::exists(config.snapshot_directory)) {
        throw std::invalid_argument("snapshot directory already exists");
    }
    return config;
}

LoadConfig
parse_load_config(int argc, char** argv) {
    if (argc != 9) {
        throw std::invalid_argument(
            "usage: lite_benchmark load SNAPSHOT DIM QUERY_ID QUERY_VARIANT QUERIES K SEED");
    }
    LoadConfig config{argv[2],
                      parse_uint(argv[3], "dimension"),
                      parse_uint(argv[4], "query ID"),
                      parse_uint(argv[5], "query variant"),
                      parse_uint(argv[6], "query count"),
                      parse_uint(argv[7], "k"),
                      parse_uint(argv[8], "seed")};
    if (not std::filesystem::is_regular_file(config.snapshot) or config.dim == 0 or
        config.query_id > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) or
        config.queries == 0 or config.k == 0) {
        throw std::invalid_argument("load configuration values are out of range");
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
        const uint64_t bits = mix(seed ^ mix(id) ^ mix(variant) ^ mix(d));
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

uint64_t
result_checksum(const std::vector<vsag::lite::Neighbor>& result, uint64_t checksum) {
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

bool
same_results(const std::vector<vsag::lite::Neighbor>& left,
             const std::vector<vsag::lite::Neighbor>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (uint64_t i = 0; i < left.size(); ++i) {
        if (left[i].id != right[i].id or left[i].distance != right[i].distance) {
            return false;
        }
    }
    return true;
}

uint64_t
peak_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return static_cast<uint64_t>(usage.ru_maxrss);
}

uint64_t
current_rss_kib() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream fields(line);
            std::string key;
            uint64_t value = 0;
            std::string unit;
            fields >> key >> value >> unit;
            if (unit != "kB") {
                throw std::runtime_error("unexpected VmRSS unit");
            }
            return value;
        }
    }
    throw std::runtime_error("VmRSS is unavailable");
}

void
require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

int
run(const Config& config) {
    auto created = vsag::lite::Index::Create(config.dim);
    require(static_cast<bool>(created), "index creation failed");
    auto index = std::move(*created);
    std::vector<float> vector(config.dim);
    std::vector<uint64_t> variants(config.count, 0);

    const auto build_start = Clock::now();
    for (uint64_t id = 0; id < config.count; ++id) {
        make_vector(id, 0, config.dim, config.seed, vector);
        require(static_cast<bool>(index->Add(static_cast<int64_t>(id), vector.data(), config.dim)),
                "add failed");
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
        variants[id] = operation + 1;
        make_vector(id, variants[id], config.dim, config.seed, vector);

        auto start = Clock::now();
        require(
            static_cast<bool>(index->Update(static_cast<int64_t>(id), vector.data(), config.dim)),
            "update failed");
        update_us.push_back(microseconds(start, Clock::now()));

        start = Clock::now();
        require(index->Remove(static_cast<int64_t>(id)), "remove failed");
        remove_us.push_back(microseconds(start, Clock::now()));

        start = Clock::now();
        require(static_cast<bool>(index->Add(static_cast<int64_t>(id), vector.data(), config.dim)),
                "re-add failed");
        readd_us.push_back(microseconds(start, Clock::now()));
    }

    std::vector<double> search_us;
    std::vector<std::vector<vsag::lite::Neighbor>> expected;
    search_us.reserve(config.queries);
    expected.reserve(config.queries);
    uint64_t top1_hits = 0;
    uint64_t checksum = 1469598103934665603ULL;
    for (uint64_t query = 0; query < config.queries; ++query) {
        const uint64_t id = (query * 104729ULL) % config.count;
        make_vector(id, variants[id], config.dim, config.seed, vector);
        const auto start = Clock::now();
        auto result = index->Search(vector.data(), config.dim, config.k);
        search_us.push_back(microseconds(start, Clock::now()));
        require(static_cast<bool>(result), "search failed");
        require(result->size() == config.k, "unexpected search result size");
        top1_hits += result->front().id == static_cast<int64_t>(id) ? 1 : 0;
        checksum = result_checksum(*result, checksum);
        expected.push_back(std::move(*result));
    }

    const auto save_start = Clock::now();
    std::ofstream output(config.snapshot, std::ios::binary);
    require(static_cast<bool>(output), "snapshot open for write failed");
    require(static_cast<bool>(index->Save(output)), "snapshot save failed");
    output.close();
    require(static_cast<bool>(output), "snapshot close failed");
    const double save_ms = milliseconds(save_start, Clock::now());
    const uint64_t snapshot_bytes = std::filesystem::file_size(config.snapshot);
    index.reset();

    const auto load_start = Clock::now();
    std::ifstream input(config.snapshot, std::ios::binary);
    auto loaded = vsag::lite::Index::Load(input);
    require(static_cast<bool>(loaded), "snapshot load failed");
    const double load_ms = milliseconds(load_start, Clock::now());
    auto loaded_index = std::move(*loaded);

    std::vector<double> loaded_search_us;
    loaded_search_us.reserve(config.queries);
    uint64_t loaded_checksum = 1469598103934665603ULL;
    for (uint64_t query = 0; query < config.queries; ++query) {
        const uint64_t id = (query * 104729ULL) % config.count;
        make_vector(id, variants[id], config.dim, config.seed, vector);
        const auto start = Clock::now();
        auto result = loaded_index->Search(vector.data(), config.dim, config.k);
        loaded_search_us.push_back(microseconds(start, Clock::now()));
        require(static_cast<bool>(result), "loaded search failed");
        require(same_results(*result, expected[query]), "loaded search result changed");
        loaded_checksum = result_checksum(*result, loaded_checksum);
    }
    require(checksum == loaded_checksum, "loaded search checksum changed");

    std::cout << "count,dim,queries,k,crud_ops,seed,build_ms,update_p50_us,update_p99_us,"
                 "remove_p50_us,remove_p99_us,readd_p50_us,readd_p99_us,search_p50_us,"
                 "search_p99_us,load_search_p50_us,load_search_p99_us,save_ms,warm_load_ms,"
                 "snapshot_bytes,peak_rss_kib,top1_recall,result_checksum,final_size\n";
    std::cout << std::fixed << std::setprecision(3) << config.count << ',' << config.dim << ','
              << config.queries << ',' << config.k << ',' << config.crud_ops << ',' << config.seed
              << ',' << build_ms << ',' << percentile(update_us, 0.50) << ','
              << percentile(update_us, 0.99) << ',' << percentile(remove_us, 0.50) << ','
              << percentile(remove_us, 0.99) << ',' << percentile(readd_us, 0.50) << ','
              << percentile(readd_us, 0.99) << ',' << percentile(search_us, 0.50) << ','
              << percentile(search_us, 0.99) << ',' << percentile(loaded_search_us, 0.50) << ','
              << percentile(loaded_search_us, 0.99) << ',' << save_ms << ',' << load_ms << ','
              << snapshot_bytes << ',' << peak_rss_kib() << ','
              << static_cast<double>(top1_hits) / static_cast<double>(config.queries) << ','
              << checksum << ',' << loaded_index->Size() << '\n';
    return 0;
}

int
run_load(const LoadConfig& config) {
    const uint64_t snapshot_bytes = std::filesystem::file_size(config.snapshot);
    const auto load_start = Clock::now();
    std::ifstream input(config.snapshot, std::ios::binary);
    require(static_cast<bool>(input), "snapshot open for read failed");
    auto loaded = vsag::lite::Index::Load(input);
    require(static_cast<bool>(loaded), "snapshot load failed");
    const double load_ms = milliseconds(load_start, Clock::now());
    auto index = std::move(*loaded);
    require(index->Dim() == config.dim, "loaded dimension does not match query configuration");
    require(index->Size() > 0, "loaded index is empty");

    std::vector<float> query(config.dim);
    make_vector(config.query_id, config.query_variant, config.dim, config.seed, query);
    const auto first_start = Clock::now();
    auto first = index->Search(query.data(), config.dim, config.k);
    const double first_query_us = microseconds(first_start, Clock::now());
    require(static_cast<bool>(first), "first query failed");
    require(not first->empty(), "first query returned no result");
    require(first->front().id == static_cast<int64_t>(config.query_id),
            "first query did not recover the configured ID");

    uint64_t checksum = result_checksum(*first, 1469598103934665603ULL);
    std::vector<double> search_us;
    search_us.reserve(config.queries);
    for (uint64_t query_number = 0; query_number < config.queries; ++query_number) {
        const auto start = Clock::now();
        auto result = index->Search(query.data(), config.dim, config.k);
        search_us.push_back(microseconds(start, Clock::now()));
        require(static_cast<bool>(result), "follow-up query failed");
        require(same_results(*result, *first), "follow-up query result changed");
        checksum = result_checksum(*result, checksum);
    }

    std::cout << "dim,query_id,query_variant,queries,k,seed,snapshot_bytes,load_ms,"
                 "first_query_us,search_p50_us,search_p99_us,current_rss_kib,peak_rss_kib,"
                 "result_checksum,final_size,page_cache_control\n";
    std::cout << std::fixed << std::setprecision(3) << config.dim << ',' << config.query_id << ','
              << config.query_variant << ',' << config.queries << ',' << config.k << ','
              << config.seed << ',' << snapshot_bytes << ',' << load_ms << ',' << first_query_us
              << ',' << percentile(search_us, 0.50) << ',' << percentile(search_us, 0.99) << ','
              << current_rss_kib() << ',' << peak_rss_kib() << ',' << checksum << ','
              << index->Size() << ",uncontrolled\n";
    return 0;
}

int
run_stability(const StabilityConfig& config) {
    require(std::filesystem::create_directories(config.snapshot_directory),
            "snapshot directory creation failed");
    const auto snapshot =
        (std::filesystem::path(config.snapshot_directory) / "latest.snapshot").string();
    auto created = vsag::lite::Index::Create(config.dim);
    require(static_cast<bool>(created), "index creation failed");
    auto index = std::move(*created);
    std::vector<float> vector(config.dim);
    std::vector<uint64_t> variants(config.count, 0);
    for (uint64_t id = 0; id < config.count; ++id) {
        make_vector(id, 0, config.dim, config.seed, vector);
        require(static_cast<bool>(index->Add(static_cast<int64_t>(id), vector.data(), config.dim)),
                "add failed");
    }

    std::cout << "round,count,crud_ops,update_p50_us,update_p99_us,remove_p50_us,"
                 "remove_p99_us,readd_p50_us,readd_p99_us,search_p50_us,search_p99_us,"
                 "save_ms,load_ms,snapshot_bytes,current_rss_kib,peak_rss_kib,top1_recall,"
                 "result_checksum\n";
    for (uint64_t round = 0; round < config.rounds; ++round) {
        std::vector<double> update_us;
        std::vector<double> remove_us;
        std::vector<double> readd_us;
        update_us.reserve(config.crud_ops);
        remove_us.reserve(config.crud_ops);
        readd_us.reserve(config.crud_ops);
        for (uint64_t operation = 0; operation < config.crud_ops; ++operation) {
            const uint64_t id = (round * 1000003ULL + operation * 8191ULL) % config.count;
            variants[id] = mix(round ^ mix(operation + 1));
            make_vector(id, variants[id], config.dim, config.seed, vector);

            auto start = Clock::now();
            require(static_cast<bool>(
                        index->Update(static_cast<int64_t>(id), vector.data(), config.dim)),
                    "update failed");
            update_us.push_back(microseconds(start, Clock::now()));
            start = Clock::now();
            require(index->Remove(static_cast<int64_t>(id)), "remove failed");
            remove_us.push_back(microseconds(start, Clock::now()));
            start = Clock::now();
            require(
                static_cast<bool>(index->Add(static_cast<int64_t>(id), vector.data(), config.dim)),
                "re-add failed");
            readd_us.push_back(microseconds(start, Clock::now()));
        }
        require(index->Size() == config.count, "index size changed after CRUD round");

        std::vector<double> search_us;
        std::vector<std::vector<vsag::lite::Neighbor>> expected;
        search_us.reserve(config.queries);
        expected.reserve(config.queries);
        uint64_t top1_hits = 0;
        uint64_t checksum = 1469598103934665603ULL;
        for (uint64_t query = 0; query < config.queries; ++query) {
            const uint64_t id = (round * 65537ULL + query * 104729ULL) % config.count;
            make_vector(id, variants[id], config.dim, config.seed, vector);
            const auto start = Clock::now();
            auto result = index->Search(vector.data(), config.dim, config.k);
            search_us.push_back(microseconds(start, Clock::now()));
            require(static_cast<bool>(result), "search failed");
            require(result->size() == config.k, "unexpected search result size");
            top1_hits += result->front().id == static_cast<int64_t>(id) ? 1 : 0;
            checksum = result_checksum(*result, checksum);
            expected.push_back(std::move(*result));
        }
        const uint64_t resident_rss = current_rss_kib();

        const auto save_start = Clock::now();
        std::ofstream output(snapshot, std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(output), "snapshot open for write failed");
        require(static_cast<bool>(index->Save(output)), "snapshot save failed");
        output.close();
        require(static_cast<bool>(output), "snapshot close failed");
        const double save_ms = milliseconds(save_start, Clock::now());
        const uint64_t snapshot_bytes = std::filesystem::file_size(snapshot);

        const auto load_start = Clock::now();
        std::ifstream input(snapshot, std::ios::binary);
        auto loaded = vsag::lite::Index::Load(input);
        require(static_cast<bool>(loaded), "snapshot load failed");
        const double load_ms = milliseconds(load_start, Clock::now());
        auto loaded_index = std::move(*loaded);
        require(loaded_index->Size() == config.count, "loaded index size changed");
        uint64_t loaded_checksum = 1469598103934665603ULL;
        for (uint64_t query = 0; query < config.queries; ++query) {
            const uint64_t id = (round * 65537ULL + query * 104729ULL) % config.count;
            make_vector(id, variants[id], config.dim, config.seed, vector);
            auto result = loaded_index->Search(vector.data(), config.dim, config.k);
            require(static_cast<bool>(result), "loaded search failed");
            require(same_results(*result, expected[query]), "loaded search result changed");
            loaded_checksum = result_checksum(*result, loaded_checksum);
        }
        require(checksum == loaded_checksum, "loaded search checksum changed");

        std::cout << std::fixed << std::setprecision(3) << round + 1 << ',' << index->Size() << ','
                  << config.crud_ops << ',' << percentile(update_us, 0.50) << ','
                  << percentile(update_us, 0.99) << ',' << percentile(remove_us, 0.50) << ','
                  << percentile(remove_us, 0.99) << ',' << percentile(readd_us, 0.50) << ','
                  << percentile(readd_us, 0.99) << ',' << percentile(search_us, 0.50) << ','
                  << percentile(search_us, 0.99) << ',' << save_ms << ',' << load_ms << ','
                  << snapshot_bytes << ',' << resident_rss << ',' << peak_rss_kib() << ','
                  << static_cast<double>(top1_hits) / static_cast<double>(config.queries) << ','
                  << checksum << '\n';
    }
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc > 1 and std::string(argv[1]) == "load") {
            return run_load(parse_load_config(argc, argv));
        }
        if (argc > 1 and std::string(argv[1]) == "stability") {
            return run_stability(parse_stability_config(argc, argv));
        }
        return run(parse_config(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "lite_benchmark: " << error.what() << '\n';
        return 1;
    }
}
