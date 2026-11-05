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

// cluster_balance_bench: reproducible comparison of the plain k-means and the
// balanced k-means IVF trainers (issue #2870).
//
// It trains both on synthetic datasets - uniform, deliberately skewed, a
// skewed one whose size is not divisible by k, and a duplicated-vector set -
// and reports, per trainer and dataset:
//   * minimum, maximum and average cluster size, and the size standard
//     deviation and coefficient of variation;
//   * the clustering loss (mean squared distance from each vector to its
//     assigned centroid, recomputed from the returned labels and the final
//     centroids so the two trainers are measured with the same yardstick);
//   * the training time;
//   * the peak memory the trainer allocated beyond the dataset itself,
//     measured through a recording allocator, alongside the size of a dense
//     n x k float distance matrix for reference.
//
// Usage:
//   cluster_balance_bench [--dataset all|uniform|skewed|indivisible|duplicates]
//                         [--n 50000] [--dim 64] [--k 256] [--iter 25]
//                         [--seed 42] [--threads N]
//
// Every run is fully seeded; repeated invocations with the same arguments
// reproduce the reported numbers up to timing noise.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "impl/cluster/balanced_kmeans_cluster.h"
#include "impl/cluster/kmeans_cluster.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace {

/// Allocator that tracks currently-live bytes and records the peak, so the
/// benchmark can report trainer memory without process-level instrumentation.
class RecordingAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "RecordingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        void* ptr = std::malloc(size);
        std::lock_guard<std::mutex> lock(mutex_);
        live_[ptr] = size;
        current_ += size;
        peak_ = std::max(peak_, current_);
        return ptr;
    }

    void
    Deallocate(void* p) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = live_.find(p);
            if (it != live_.end()) {
                current_ -= it->second;
                live_.erase(it);
            }
        }
        std::free(p);
    }

    void*
    Reallocate(void* p, uint64_t size) override {
        Deallocate(p);
        return Allocate(size);
    }

    uint64_t
    Peak() const {
        return peak_;
    }

    void
    Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        peak_ = current_;
    }

private:
    std::mutex mutex_;
    std::unordered_map<void*, uint64_t> live_;
    uint64_t current_{0};
    uint64_t peak_{0};
};

/// Geometric-mixture dataset: cluster sizes drop by ~3x and well-separated
/// centers make plain k-means pile vectors into few buckets. The duplicated
/// mode collapses every point of the largest component onto one vector.
std::vector<float>
make_dataset(const std::string& kind,
             uint32_t k,
             uint64_t n,
             int32_t dim,
             uint32_t seed,
             uint64_t& actual_n) {
    std::mt19937 gen(seed);
    std::vector<float> data;
    if (kind == "uniform") {
        actual_n = n;
        data.resize(n * static_cast<uint64_t>(dim));
        std::uniform_real_distribution<float> coord(0.0F, 1.0F);
        for (auto& v : data) {
            v = coord(gen);
        }
        return data;
    }
    if (kind == "duplicates") {
        actual_n = n;
        data.assign(n * static_cast<uint64_t>(dim), 0.0F);
        // The whole set coincides except for a few outliers: an unconstrained
        // trainer has nowhere sensible to put 90% of the mass.
        for (uint64_t i = 0; i < n / 20; ++i) {
            for (int32_t d = 0; d < dim; ++d) {
                data[(n - 1 - i) * static_cast<uint64_t>(dim) + d] = 100.0F + static_cast<float>(i);
            }
        }
        return data;
    }
    // skewed (and indivisible = skewed trimmed to n % k != 0)
    std::vector<uint64_t> sizes(k);
    uint64_t total = 0;
    // Geometric tail: with cur as the head component, the sizes sum to roughly
    // 1.5 * cur + k, so pick cur to land near the requested n.
    uint64_t cur = (n > 2 * static_cast<uint64_t>(k)) ? (n - k) * 2 / 3 : n;
    for (uint32_t c = 0; c < k; ++c) {
        sizes[c] = std::max<uint64_t>(1, cur);
        total += sizes[c];
        cur = cur / 3 + 1;
    }
    actual_n = total;
    // With one cluster every size is divisible; in particular, do not turn the
    // valid n=1, k=1 case into an empty dataset.
    if (kind == "indivisible" and k > 1 and total % k == 0) {
        actual_n = total - 1;
    }
    std::vector<float> centroids(static_cast<uint64_t>(k) * dim);
    std::uniform_real_distribution<float> coord(-100.0F, 100.0F);
    for (auto& v : centroids) {
        v = coord(gen);
    }
    std::normal_distribution<float> noise(0.0F, 0.05F);
    data.assign(actual_n * static_cast<uint64_t>(dim), 0.0F);
    uint64_t i = 0;
    for (uint32_t c = 0; c < k and i < actual_n; ++c) {
        for (uint64_t j = 0; j < sizes[c] and i < actual_n; ++j, ++i) {
            for (int32_t d = 0; d < dim; ++d) {
                data[i * static_cast<uint64_t>(dim) + d] =
                    centroids[static_cast<uint64_t>(c) * static_cast<uint64_t>(dim) + d] +
                    static_cast<float>(noise(gen));
            }
        }
    }
    return data;
}

struct Result {
    std::string trainer;
    std::string dataset;
    uint64_t n{0};
    uint32_t k{0};
    uint64_t min_size{0};
    uint64_t max_size{0};
    double mean_size{0.0};
    double std_size{0.0};
    double cv_size{0.0};
    double loss{0.0};
    double seconds{0.0};
    uint64_t peak_bytes{0};
};

Result
measure(const std::string& trainer,
        const std::string& dataset,
        const vsag::Vector<int>& labels,
        const std::vector<float>& data,
        const float* centroids,
        uint32_t k,
        int32_t dim,
        uint64_t n,
        double seconds,
        uint64_t peak_bytes) {
    Result r;
    r.trainer = trainer;
    r.dataset = dataset;
    r.n = n;
    r.k = k;
    std::vector<uint64_t> sizes(k, 0);
    for (uint64_t i = 0; i < n; ++i) {
        ++sizes[static_cast<uint32_t>(labels[i])];
    }
    r.min_size = *std::min_element(sizes.begin(), sizes.end());
    r.max_size = *std::max_element(sizes.begin(), sizes.end());
    r.mean_size = static_cast<double>(n) / static_cast<double>(k);
    double var = 0.0;
    for (uint64_t s : sizes) {
        var += (static_cast<double>(s) - r.mean_size) * (static_cast<double>(s) - r.mean_size);
    }
    var /= static_cast<double>(k);
    r.std_size = std::sqrt(var);
    r.cv_size = r.std_size / r.mean_size;
    // Loss measured identically for both trainers: mean squared distance from
    // each vector to the centroid it was assigned to.
    double loss = 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const float* x = data.data() + i * static_cast<uint64_t>(dim);
        const float* c = centroids + static_cast<uint64_t>(labels[i]) * static_cast<uint64_t>(dim);
        double sum = 0.0;
        for (int32_t d = 0; d < dim; ++d) {
            const double diff = static_cast<double>(x[d]) - static_cast<double>(c[d]);
            sum += diff * diff;
        }
        loss += sum;
    }
    r.loss = loss / static_cast<double>(n);
    r.seconds = seconds;
    r.peak_bytes = peak_bytes;
    return r;
}

void
run_bench(const std::string& dataset,
          uint64_t n,
          int32_t dim,
          uint32_t k,
          int iter,
          uint32_t seed,
          int threads,
          std::vector<Result>& results) {
    uint64_t actual_n = 0;
    const auto data = make_dataset(dataset, k, n, dim, seed, actual_n);

    auto pool = vsag::SafeThreadPool::FactoryDefaultThreadPool();
    if (threads > 0) {
        pool->SetPoolSize(static_cast<uint64_t>(threads));
    }

    // Plain k-means, seeded identically to the balanced trainer.
    {
        RecordingAllocator allocator;
        vsag::KMeansCluster cluster(dim, &allocator, pool);
        const auto start = std::chrono::steady_clock::now();
        const auto labels = cluster.Run(k,
                                        data.data(),
                                        actual_n,
                                        iter,
                                        nullptr,
                                        false,
                                        1e-6F,
                                        vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                        seed,
                                        /*deterministic_reduction=*/true);
        const auto end = std::chrono::steady_clock::now();
        results.push_back(measure("kmeans",
                                  dataset,
                                  labels,
                                  data,
                                  cluster.k_centroids_,
                                  k,
                                  dim,
                                  actual_n,
                                  std::chrono::duration<double>(end - start).count(),
                                  allocator.Peak()));
    }

    // Balanced k-means.
    {
        RecordingAllocator allocator;
        vsag::BalancedKMeansCluster cluster(dim, &allocator, pool);
        const auto start = std::chrono::steady_clock::now();
        const auto labels = cluster.Run(k,
                                        data.data(),
                                        actual_n,
                                        iter,
                                        nullptr,
                                        1e-6F,
                                        vsag::KMeansInitMethod::KMEANS_PLUS_PLUS,
                                        seed);
        const auto end = std::chrono::steady_clock::now();
        results.push_back(measure("balanced_kmeans",
                                  dataset,
                                  labels,
                                  data,
                                  cluster.k_centroids_,
                                  k,
                                  dim,
                                  actual_n,
                                  std::chrono::duration<double>(end - start).count(),
                                  allocator.Peak()));
    }
}

void
print_usage(const char* argv0) {
    std::printf(
        "usage: %s [--dataset all|uniform|skewed|indivisible|duplicates] [--n N] [--dim D]\n"
        "          [--k K] [--iter I] [--seed S] [--threads T]\n",
        argv0);
}

bool
parse(const int argc,
      char** argv,
      std::string& dataset,
      uint64_t& n,
      int32_t& dim,
      uint32_t& k,
      int& iter,
      uint32_t& seed,
      int& threads) {
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::printf("missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (key == "--dataset") {
            dataset = next_value("--dataset");
        } else if (key == "--n") {
            n = std::strtoull(next_value("--n").c_str(), nullptr, 10);
        } else if (key == "--dim") {
            dim = static_cast<int32_t>(std::strtol(next_value("--dim").c_str(), nullptr, 10));
        } else if (key == "--k") {
            k = static_cast<uint32_t>(std::strtoul(next_value("--k").c_str(), nullptr, 10));
        } else if (key == "--iter") {
            iter = static_cast<int>(std::strtol(next_value("--iter").c_str(), nullptr, 10));
        } else if (key == "--seed") {
            seed = static_cast<uint32_t>(std::strtoul(next_value("--seed").c_str(), nullptr, 10));
        } else if (key == "--threads") {
            threads = static_cast<int>(std::strtol(next_value("--threads").c_str(), nullptr, 10));
        } else if (key == "--help" or key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::printf("unknown argument: %s\n", key.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (k >= 10000) {
        // KMeansCluster switches to non-deterministic parallel HGraph assignment here.
        std::printf("--k must be less than 10000 for reproducible HGraph-free assignment\n");
        return false;
    }
    return k > 0 and n > 0 and dim > 0 and iter > 0 and (k <= n);
}

}  // namespace

int
main(int argc, char** argv) {
    std::string dataset = "all";
    uint64_t n = 50000;
    int32_t dim = 64;
    uint32_t k = 256;
    int iter = 25;
    uint32_t seed = 42;
    int threads = 0;
    if (not parse(argc, argv, dataset, n, dim, k, iter, seed, threads)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::vector<std::string> all = {"uniform", "skewed", "indivisible", "duplicates"};
    std::vector<std::string> datasets;
    if (dataset == "all") {
        datasets = all;
    } else {
        datasets.push_back(dataset);
    }

    std::vector<Result> results;
    for (const auto& name : datasets) {
        run_bench(name, n, dim, k, iter, seed, threads, results);
    }

    std::printf("cluster_balance_bench: n=%llu dim=%d k=%u iter=%d seed=%u threads=%d\n",
                static_cast<unsigned long long>(n),
                dim,
                k,
                iter,
                seed,
                threads);
    std::printf("dense n*k*4 reference: %.1f MiB (balanced_kmeans must stay far below)\n",
                static_cast<double>(n) * static_cast<double>(k) * 4.0 / 1048576.0);
    std::printf("\n%-16s %-12s %-12s %-13s %-9s %-9s %-8s %-9s %-8s %-11s\n",
                "trainer",
                "dataset",
                "n",
                "sizes",
                "mean",
                "std",
                "CV",
                "loss(MSE)",
                "time",
                "peak-mem");
    for (const auto& r : results) {
        char sizes[64];
        std::snprintf(sizes,
                      sizeof(sizes),
                      "%llu-%llu",
                      static_cast<unsigned long long>(r.min_size),
                      static_cast<unsigned long long>(r.max_size));
        char mem[32];
        std::snprintf(mem, sizeof(mem), "%.1f MiB", static_cast<double>(r.peak_bytes) / 1048576.0);
        std::printf("%-16s %-12s %-12llu %-13s %-9.1f %-9.1f %-8.3f %-9.4f %-7.2fs %s\n",
                    r.trainer.c_str(),
                    r.dataset.c_str(),
                    static_cast<unsigned long long>(r.n),
                    sizes,
                    r.mean_size,
                    r.std_size,
                    r.cv_size,
                    r.loss,
                    r.seconds,
                    mem);
        if (r.trainer == "balanced_kmeans") {
            const bool exact =
                r.max_size - r.min_size <= 1 and r.min_size >= 1;  // valid input has k <= n
            std::printf("  balance check: %s\n", exact ? "PASS" : "FAIL");
            if (not exact) {
                return 1;
            }
        }
    }
    return 0;
}
