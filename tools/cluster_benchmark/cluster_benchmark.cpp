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

// Compares KMeansCluster and BalancedKMeansCluster on the same vectors: cluster-size
// balance, loss, training time and the peak memory the trainer allocates.

#include <H5Cpp.h>
#include <fmt/format.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "impl/cluster/balanced_kmeans_cluster.h"
#include "impl/cluster/kmeans_cluster.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "simd/fp32_simd.h"

namespace {

/// Records the bytes the trainer holds at once, which is what the issue's peak
/// memory asks for: the input vectors belong to the caller and are excluded.
class PeakAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "peak_allocator";
    }

    void*
    Allocate(uint64_t size) override {
        auto* block = static_cast<uint64_t*>(std::malloc(size + HEADER));
        if (block == nullptr) {
            return nullptr;
        }
        block[0] = size;
        add(static_cast<int64_t>(size));
        return reinterpret_cast<char*>(block) + HEADER;
    }

    void
    Deallocate(void* p) override {
        if (p == nullptr) {
            return;
        }
        auto* block = reinterpret_cast<uint64_t*>(static_cast<char*>(p) - HEADER);
        add(-static_cast<int64_t>(block[0]));
        std::free(block);
    }

    void*
    Reallocate(void* p, uint64_t size) override {
        if (p == nullptr) {
            return Allocate(size);
        }
        auto* block = reinterpret_cast<uint64_t*>(static_cast<char*>(p) - HEADER);
        const uint64_t old_size = block[0];
        auto* grown = static_cast<uint64_t*>(std::realloc(block, size + HEADER));
        if (grown == nullptr) {
            return nullptr;
        }
        grown[0] = size;
        add(static_cast<int64_t>(size) - static_cast<int64_t>(old_size));
        return reinterpret_cast<char*>(grown) + HEADER;
    }

    [[nodiscard]] uint64_t
    Peak() const {
        return peak_;
    }

private:
    void
    add(int64_t delta) {
        std::lock_guard lock(mutex_);
        current_ += delta;
        peak_ = std::max(peak_, static_cast<uint64_t>(current_));
    }

    static constexpr uint64_t HEADER = 16;
    std::mutex mutex_;
    int64_t current_{0};
    uint64_t peak_{0};
};

struct Vectors {
    std::string name;
    uint64_t count{0};
    uint64_t dim{0};
    std::vector<float> data;
};

Vectors
uniform_vectors(uint64_t count, uint64_t dim, uint32_t seed) {
    Vectors vectors{"uniform", count, dim, std::vector<float>(count * dim)};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> unit(0.0F, 1.0F);
    for (auto& value : vectors.data) {
        value = unit(gen);
    }
    return vectors;
}

/// A Zipf-weighted Gaussian mixture with exact duplicates: a few dense regions hold
/// most of the vectors, and `duplicates` of all vectors repeat one of eight hot vectors.
Vectors
skewed_vectors(uint64_t count, uint64_t dim, uint32_t seed, double duplicates, float spread) {
    constexpr uint64_t components = 64;
    constexpr uint64_t hot_vectors = 8;
    Vectors vectors{"skewed", count, dim, std::vector<float>(count * dim)};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> box(-spread, spread);
    std::vector<float> centers(components * dim);
    for (auto& value : centers) {
        value = box(gen);
    }
    std::vector<double> weights(components);
    for (uint64_t c = 0; c < components; ++c) {
        weights[c] = 1.0 / std::pow(static_cast<double>(c + 1), 1.2);
    }
    std::discrete_distribution<uint64_t> component(weights.begin(), weights.end());
    std::normal_distribution<float> noise(0.0F, 0.25F);
    std::bernoulli_distribution duplicate(duplicates);
    std::uniform_int_distribution<uint64_t> hot(0, hot_vectors - 1);
    for (uint64_t row = 0; row < count; ++row) {
        float* x = vectors.data.data() + row * dim;
        if (row >= hot_vectors and duplicate(gen)) {
            std::memcpy(x, vectors.data.data() + hot(gen) * dim, dim * sizeof(float));
            continue;
        }
        const float* center = centers.data() + component(gen) * dim;
        for (uint64_t d = 0; d < dim; ++d) {
            x[d] = center[d] + noise(gen);
        }
    }
    return vectors;
}

Vectors
read_fvecs(const std::string& path, uint64_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (not in) {
        throw std::runtime_error("cannot open " + path);
    }
    Vectors vectors{path, 0, 0, {}};
    int32_t dim = 0;
    while (vectors.count < limit and in.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (vectors.dim == 0) {
            vectors.dim = static_cast<uint64_t>(dim);
        }
        const uint64_t offset = vectors.data.size();
        vectors.data.resize(offset + vectors.dim);
        in.read(reinterpret_cast<char*>(vectors.data.data() + offset),
                static_cast<std::streamsize>(vectors.dim * sizeof(float)));
        ++vectors.count;
    }
    return vectors;
}

/// The "train" split of an ann-benchmarks HDF5 file.
Vectors
read_hdf5(const std::string& path, uint64_t limit) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("/train");
    H5::DataSpace space = dataset.getSpace();
    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims);
    Vectors vectors{path, std::min<uint64_t>(dims[0], limit), dims[1], {}};
    vectors.data.resize(vectors.count * vectors.dim);
    hsize_t count[2] = {vectors.count, vectors.dim};
    hsize_t offset[2] = {0, 0};
    space.selectHyperslab(H5S_SELECT_SET, count, offset);
    H5::DataSpace memory(2, count);
    dataset.read(vectors.data.data(), H5::PredType::NATIVE_FLOAT, memory, space);
    return vectors;
}

struct Report {
    std::string dataset;
    std::string algorithm;
    uint64_t count{0};
    uint64_t dim{0};
    uint64_t k{0};
    uint64_t min_size{0};
    uint64_t max_size{0};
    double mean_size{0.0};
    double stddev{0.0};
    double loss{0.0};
    double seconds{0.0};
    uint64_t peak_bytes{0};
    int32_t iterations{0};
    uint64_t bids{0};
    uint64_t rescored{0};
    bool two_level{false};
    bool stopped{false};
    uint64_t fallback{0};
};

/// Sizes and the mean squared distance of every vector to its centroid.
void
measure(const Vectors& vectors,
        uint64_t k,
        const vsag::Vector<int32_t>& labels,
        const float* centroids,
        Report& report) {
    std::vector<uint64_t> sizes(k, 0);
    double loss = 0.0;
    for (uint64_t row = 0; row < vectors.count; ++row) {
        const auto label = static_cast<uint64_t>(labels[row]);
        ++sizes[label];
        loss += vsag::FP32ComputeL2Sqr(
            vectors.data.data() + row * vectors.dim, centroids + label * vectors.dim, vectors.dim);
    }
    const auto [low, high] = std::minmax_element(sizes.begin(), sizes.end());
    report.min_size = *low;
    report.max_size = *high;
    report.mean_size = static_cast<double>(vectors.count) / static_cast<double>(k);
    double variance = 0.0;
    for (const auto size : sizes) {
        const double diff = static_cast<double>(size) - report.mean_size;
        variance += diff * diff;
    }
    report.stddev = std::sqrt(variance / static_cast<double>(k));
    report.loss = loss / static_cast<double>(vectors.count);
}

Report
run(const std::string& algorithm,
    const Vectors& vectors,
    uint64_t k,
    int32_t iterations,
    uint32_t seed,
    uint32_t candidates,
    const std::string& init,
    double epsilon_ratio,
    double work_budget,
    const vsag::SafeThreadPoolPtr& pool) {
    Report report{vectors.name, algorithm, vectors.count, vectors.dim, k};
    PeakAllocator allocator;
    const auto start = std::chrono::steady_clock::now();
    if (algorithm == "kmeans") {
        vsag::KMeansCluster cluster(static_cast<int32_t>(vectors.dim), &allocator, pool);
        auto labels = cluster.Run(static_cast<uint32_t>(k),
                                  vectors.data.data(),
                                  vectors.count,
                                  iterations,
                                  nullptr,
                                  false,
                                  1e-6F,
                                  vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                  seed);
        report.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        report.peak_bytes = allocator.Peak();
        report.iterations = iterations;
        measure(vectors, k, labels, cluster.k_centroids_, report);
    } else {
        vsag::BalancedKMeansCluster cluster(static_cast<int32_t>(vectors.dim), &allocator, pool);
        vsag::BalancedKMeansOptions options;
        options.iterations = iterations;
        options.random_seed = seed;
        options.candidate_count = candidates;
        options.epsilon_ratio = epsilon_ratio;
        options.work_budget = work_budget;
        options.init_method = init == "kmeans++" ? vsag::KMeansInitMethod::KMEANS_PLUS_PLUS
                                                 : vsag::KMeansInitMethod::RANDOM;
        auto labels =
            cluster.Run(static_cast<uint32_t>(k), vectors.data.data(), vectors.count, options);
        report.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        report.peak_bytes = allocator.Peak();
        report.iterations = cluster.stats_.iterations;
        report.bids = cluster.stats_.bids;
        report.rescored = cluster.stats_.rescored_vectors;
        report.two_level = cluster.stats_.two_level;
        report.stopped = cluster.stats_.stopped_on_budget;
        report.fallback = cluster.stats_.fallback_assignments;
        measure(vectors, k, labels, cluster.k_centroids_, report);
    }
    return report;
}

void
print_table(const std::vector<Report>& reports) {
    std::cout << "| dataset | algorithm | n | dim | k | min | max | mean | std | CV | loss (MSE) "
                 "| time (s) | peak MiB | iters | bids/n | rescored | path |\n"
              << "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& r : reports) {
        std::cout << std::fixed << "| " << r.dataset << " | " << r.algorithm << " | " << r.count
                  << " | " << r.dim << " | " << r.k << " | " << r.min_size << " | " << r.max_size
                  << " | " << std::setprecision(2) << r.mean_size << " | " << r.stddev << " | "
                  << std::setprecision(4) << r.stddev / r.mean_size << " | " << std::setprecision(6)
                  << r.loss << " | " << std::setprecision(3) << r.seconds << " | "
                  << std::setprecision(1) << static_cast<double>(r.peak_bytes) / (1024.0 * 1024.0)
                  << " | " << r.iterations << " | " << std::setprecision(2)
                  << static_cast<double>(r.bids) / static_cast<double>(r.count) << " | "
                  << r.rescored << " | "
                  << (r.algorithm == "kmeans" ? "-"
                      : r.two_level           ? "two-level"
                      : r.stopped             ? "flat (stopped)"
                                              : "flat")
                  << " |\n";
    }
}

void
write_json(const std::vector<Report>& reports, const std::string& path) {
    std::ofstream out(path);
    out << "[\n";
    for (uint64_t i = 0; i < reports.size(); ++i) {
        const auto& r = reports[i];
        out << std::setprecision(10) << "  {\"dataset\": \"" << r.dataset << "\", \"algorithm\": \""
            << r.algorithm << "\", \"count\": " << r.count << ", \"dim\": " << r.dim
            << ", \"k\": " << r.k << ", \"min_size\": " << r.min_size
            << ", \"max_size\": " << r.max_size << ", \"mean_size\": " << r.mean_size
            << ", \"stddev\": " << r.stddev << ", \"cv\": " << r.stddev / r.mean_size
            << ", \"loss\": " << r.loss << ", \"seconds\": " << r.seconds
            << ", \"peak_bytes\": " << r.peak_bytes << ", \"iterations\": " << r.iterations
            << ", \"bids\": " << r.bids << ", \"rescored\": " << r.rescored
            << ", \"two_level\": " << (r.two_level ? "true" : "false")
            << ", \"stopped_on_budget\": " << (r.stopped ? "true" : "false")
            << ", \"fallback\": " << r.fallback << "}" << (i + 1 < reports.size() ? ",\n" : "\n");
    }
    out << "]\n";
}

/// The queries and the exact top 100 of an ann-benchmarks HDF5 file.
struct Queries {
    uint64_t count{0};
    uint64_t dim{0};
    std::vector<float> data;
    uint64_t truth_width{0};
    std::vector<int64_t> truth;
};

Queries
read_queries(const std::string& path, uint64_t limit) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    Queries queries;
    {
        H5::DataSet dataset = file.openDataSet("/test");
        hsize_t dims[2] = {0, 0};
        dataset.getSpace().getSimpleExtentDims(dims);
        queries.dim = dims[1];
        std::vector<float> all(dims[0] * dims[1]);
        dataset.read(all.data(), H5::PredType::NATIVE_FLOAT);
        queries.count = std::min<uint64_t>(dims[0], limit);
        queries.data.assign(all.begin(), all.begin() + queries.count * queries.dim);
    }
    {
        H5::DataSet dataset = file.openDataSet("/neighbors");
        hsize_t dims[2] = {0, 0};
        dataset.getSpace().getSimpleExtentDims(dims);
        queries.truth_width = dims[1];
        std::vector<int64_t> all(dims[0] * dims[1]);
        dataset.read(all.data(), H5::PredType::NATIVE_INT64);
        queries.truth.assign(all.begin(), all.begin() + queries.count * queries.truth_width);
    }
    return queries;
}

/// Builds an IVF index with every trainer and searches it at several
/// scan_buckets_count values: build time, bucket sizes (from GetStats), recall@10,
/// throughput and distance evaluations per query.
int
run_ivf(const std::string& path,
        const std::vector<std::string>& algorithms,
        uint64_t buckets,
        const std::string& metric,
        const std::vector<int>& scans,
        uint64_t threads,
        uint64_t query_limit) {
    auto base = read_hdf5(path, std::numeric_limits<uint64_t>::max());
    auto queries = read_queries(path, query_limit);
    constexpr int64_t topk = 10;
    std::vector<int64_t> ids(base.count);
    std::iota(ids.begin(), ids.end(), 0);
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(static_cast<int64_t>(base.count))
        ->Dim(static_cast<int64_t>(base.dim))
        ->Ids(ids.data())
        ->Float32Vectors(base.data.data())
        ->Owner(false);
    std::cout << "| dataset | trainer | buckets | build (s) | bucket min | bucket max | bucket std "
                 "| scan | recall@10 | QPS | distances/query |\n"
              << "|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& algorithm : algorithms) {
        const auto params = fmt::format(
            R"({{"dtype": "float32", "metric_type": "{}", "dim": {}, "index_param": {{
                "buckets_count": {}, "base_quantization_type": "fp32",
                "partition_strategy_type": "ivf", "ivf_train_type": "{}",
                "thread_count": {}}}}})",
            metric,
            base.dim,
            buckets,
            algorithm,
            threads);
        auto created = vsag::Factory::CreateIndex("ivf", params);
        if (not created.has_value()) {
            std::cerr << created.error().message << "\n";
            return 1;
        }
        auto index = created.value();
        const auto start = std::chrono::steady_clock::now();
        auto built = index->Build(dataset);
        const double build_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (not built.has_value()) {
            std::cerr << built.error().message << "\n";
            return 1;
        }
        const auto stats = index->GetStats();
        auto stat = [&](const std::string& key) {
            const auto at = stats.find("\"" + key + "\"", stats.find("\"bucket_num\""));
            return std::stod(stats.substr(stats.find(':', at) + 1));
        };
        for (const int scan : scans) {
            const auto search_params =
                fmt::format(R"({{"ivf": {{"scan_buckets_count": {}}}}})", scan);
            std::vector<uint64_t> hits(queries.count, 0);
            std::vector<uint64_t> distances(queries.count, 0);
            std::atomic<uint64_t> next{0};
            const auto search_start = std::chrono::steady_clock::now();
            std::vector<std::thread> workers;
            for (uint64_t t = 0; t < threads; ++t) {
                workers.emplace_back([&]() {
                    for (uint64_t q = next++; q < queries.count; q = next++) {
                        auto query = vsag::Dataset::Make();
                        query->NumElements(1)
                            ->Dim(static_cast<int64_t>(queries.dim))
                            ->Float32Vectors(queries.data.data() + q * queries.dim)
                            ->Owner(false);
                        auto result = index->KnnSearch(query, topk, search_params);
                        if (not result.has_value()) {
                            continue;
                        }
                        const int64_t* found = result.value()->GetIds();
                        const int64_t* truth = queries.truth.data() + q * queries.truth_width;
                        for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
                            hits[q] += std::find(truth, truth + topk, found[i]) != truth + topk;
                        }
                        const auto counted =
                            result.value()->GetStatistics({"distance_evaluations"});
                        distances[q] =
                            counted.empty() or counted[0].empty() ? 0 : std::stoull(counted[0]);
                    }
                });
            }
            for (auto& worker : workers) {
                worker.join();
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - search_start)
                    .count();
            const double recall =
                static_cast<double>(std::accumulate(hits.begin(), hits.end(), 0ULL)) /
                static_cast<double>(queries.count * topk);
            const double mean_distances =
                static_cast<double>(std::accumulate(distances.begin(), distances.end(), 0ULL)) /
                static_cast<double>(queries.count);
            std::cout << std::fixed << "| " << path.substr(path.find_last_of('/') + 1) << " | "
                      << algorithm << " | " << buckets << " | " << std::setprecision(1)
                      << build_seconds << " | " << std::setprecision(0) << stat("min") << " | "
                      << stat("max") << " | " << std::setprecision(1) << stat("std") << " | "
                      << scan << " | " << std::setprecision(4) << recall << " | "
                      << std::setprecision(0) << static_cast<double>(queries.count) / seconds
                      << " | " << mean_distances << " |\n";
        }
    }
    return 0;
}

}  // namespace

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("cluster_benchmark");
    parser.add_description(
        "Compare k-means and balanced k-means: cluster sizes, loss, training time and peak "
        "memory.");
    parser.add_argument("--dataset")
        .help("uniform, skewed, indivisible, suite (all three), or a .fvecs/.hdf5 file")
        .default_value(std::string("suite"));
    parser.add_argument("--count").help("vectors").default_value(100000).scan<'i', int>();
    parser.add_argument("--dim")
        .help("dimension of synthetic data")
        .default_value(64)
        .scan<'i', int>();
    parser.add_argument("-k", "--clusters").help("clusters").default_value(256).scan<'i', int>();
    parser.add_argument("--iterations").default_value(25).scan<'i', int>();
    parser.add_argument("--seed").default_value(2870).scan<'i', int>();
    parser.add_argument("--threads").default_value(16).scan<'i', int>();
    parser.add_argument("--candidates")
        .help("balanced k-means candidate_count")
        .default_value(16)
        .scan<'i', int>();
    parser.add_argument("--init")
        .help("balanced k-means seeding: random or kmeans++")
        .default_value(std::string("random"));
    parser.add_argument("--work-budget")
        .help("balanced k-means work_budget (inf never switches to the two-level path)")
        .default_value(1.0)
        .scan<'g', double>();
    parser.add_argument("--epsilon")
        .help("balanced k-means epsilon_ratio")
        .default_value(1e-4)
        .scan<'g', double>();
    parser.add_argument("--duplicates")
        .help("fraction of skewed vectors that repeat one of eight hot vectors")
        .default_value(0.05)
        .scan<'g', double>();
    parser.add_argument("--spread")
        .help("half-width of the box holding the skewed mixture's centers (noise 0.25)")
        .default_value(4.0)
        .scan<'g', double>();
    parser.add_argument("--algorithms")
        .help("comma-separated: kmeans,balanced_kmeans")
        .default_value(std::string("kmeans,balanced_kmeans"));
    parser.add_argument("--json")
        .help("also write the results to this file")
        .default_value(std::string(""));
    parser.add_argument("--ivf")
        .help(
            "build IVF indexes from an ann-benchmarks .hdf5 file with each trainer and search them")
        .default_value(false)
        .implicit_value(true);
    parser.add_argument("--metric")
        .help("IVF metric: l2, ip or cosine")
        .default_value(std::string("l2"));
    parser.add_argument("--scan")
        .help("IVF scan_buckets_count values, comma-separated")
        .default_value(std::string("8,16,32,64,128"));
    parser.add_argument("--queries")
        .help("IVF queries to run")
        .default_value(10000)
        .scan<'i', int>();
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n" << parser;
        return 2;
    }

    // Checked before any of them sizes a dataset or divides.
    for (const char* name : {"--count",
                             "--dim",
                             "--clusters",
                             "--iterations",
                             "--threads",
                             "--candidates",
                             "--queries"}) {
        if (parser.get<int>(name) <= 0) {
            std::cerr << name << " must be positive\n";
            return 2;
        }
    }
    const auto count = static_cast<uint64_t>(parser.get<int>("--count"));
    const auto dim = static_cast<uint64_t>(parser.get<int>("--dim"));
    const auto k = static_cast<uint64_t>(parser.get<int>("--clusters"));
    const auto iterations = parser.get<int>("--iterations");
    const auto seed = static_cast<uint32_t>(parser.get<int>("--seed"));
    const auto threads = static_cast<uint64_t>(parser.get<int>("--threads"));
    const auto candidates = static_cast<uint32_t>(parser.get<int>("--candidates"));
    const auto init = parser.get<std::string>("--init");
    const auto dataset = parser.get<std::string>("--dataset");

    std::vector<std::string> algorithms;
    std::stringstream list(parser.get<std::string>("--algorithms"));
    for (std::string item; std::getline(list, item, ',');) {
        if (item != "kmeans" and item != "balanced_kmeans") {
            std::cerr << "unknown algorithm: " << item << "\n";
            return 2;
        }
        algorithms.push_back(item);
    }

    if (parser.get<bool>("--ivf")) {
        std::vector<int> scans;
        std::stringstream scan_list(parser.get<std::string>("--scan"));
        for (std::string item; std::getline(scan_list, item, ',');) {
            char* end = nullptr;
            const int64_t scan = std::strtoll(item.c_str(), &end, 10);
            if (item.empty() or *end != '\0' or scan <= 0 or
                scan > std::numeric_limits<int>::max()) {
                std::cerr << "--scan takes positive integers, not '" << item << "'\n";
                return 2;
            }
            scans.push_back(static_cast<int>(scan));
        }
        try {
            return run_ivf(dataset,
                           algorithms,
                           k,
                           parser.get<std::string>("--metric"),
                           scans,
                           threads,
                           static_cast<uint64_t>(parser.get<int>("--queries")));
        } catch (const std::exception& e) {
            std::cerr << "ivf evaluation failed: " << e.what() << "\n";
            return 2;
        }
    }

    std::vector<Vectors> inputs;
    if (dataset == "suite" or dataset == "uniform") {
        inputs.push_back(uniform_vectors(count, dim, seed));
    }
    if (dataset == "suite" or dataset == "skewed") {
        inputs.push_back(skewed_vectors(count,
                                        dim,
                                        seed,
                                        parser.get<double>("--duplicates"),
                                        static_cast<float>(parser.get<double>("--spread"))));
    }
    if (dataset == "suite" or dataset == "indivisible") {
        // k / 2 clusters hold one vector more than the others.
        const uint64_t indivisible = count / k * k + k / 2;
        auto vectors = uniform_vectors(indivisible, dim, seed + 1);
        vectors.name = "indivisible";
        inputs.push_back(std::move(vectors));
    }
    if (inputs.empty()) {
        try {
            const bool hdf5 = dataset.size() > 5 and dataset.substr(dataset.size() - 5) == ".hdf5";
            inputs.push_back(hdf5 ? read_hdf5(dataset, count) : read_fvecs(dataset, count));
        } catch (const std::exception& e) {
            std::cerr << "cannot read " << dataset << ": " << e.what() << "\n";
            return 2;
        }
    }

    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    pool->SetPoolSize(threads);

    std::vector<Report> reports;
    for (const auto& vectors : inputs) {
        if (k > vectors.count) {
            std::cerr << "k must be in [1, " << vectors.count << "]\n";
            return 2;
        }
        for (const auto& algorithm : algorithms) {
            reports.push_back(run(algorithm,
                                  vectors,
                                  k,
                                  iterations,
                                  seed,
                                  candidates,
                                  init,
                                  parser.get<double>("--epsilon"),
                                  parser.get<double>("--work-budget"),
                                  pool));
            std::cerr << "done: " << vectors.name << " / " << algorithm << " in "
                      << reports.back().seconds << " s\n";
        }
    }
    print_table(reports);
    const auto json = parser.get<std::string>("--json");
    if (not json.empty()) {
        write_json(reports, json);
    }
    return 0;
}
