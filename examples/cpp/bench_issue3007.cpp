// Benchmark harness for vsag issue #3007
//   [Performance] Low and fluctuating CPU utilization during batched Add with Fused RaBitQ 1+7
//
// Reproduces the reported ingestion pattern: sequential batched Add() calls
// (20,000 vectors/batch) building an HGRAPH index with Fused RaBitQ 1+7
// (base 1 bit + precise 7 bit) and graph_type=odescent.
//
// Reports per-batch wall time, achieved vectors/second and process CPU
// utilization, so serial (single-core) phases inside Add() are visible as
// util ~= 100% of one core with long wall time.

#include <vsag/vsag.h>

#include "algorithm/hgraph/hgraph.h"
#include "index/index_impl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

double
cpu_seconds_now() {
    struct rusage usage {};
    getrusage(RUSAGE_SELF, &usage);
    return static_cast<double>(usage.ru_utime.tv_sec) +
           static_cast<double>(usage.ru_utime.tv_usec) / 1e6 +
           static_cast<double>(usage.ru_stime.tv_sec) +
           static_cast<double>(usage.ru_stime.tv_usec) / 1e6;
}

double
wall_seconds_now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int64_t
arg_int(int argc, char** argv, const char* name, int64_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoll(argv[i + 1]);
        }
    }
    return fallback;
}

}  // namespace

int
main(int argc, char** argv) {
    const int64_t dim = arg_int(argc, argv, "--dim", 1024);
    const int64_t batch = arg_int(argc, argv, "--batch", 20000);
    const int64_t batches = arg_int(argc, argv, "--batches", 5);
    const int64_t threads = arg_int(argc, argv, "--threads", 32);
    const int64_t total = batch * batches;
    const bool use_build = arg_int(argc, argv, "--use-build", 0) != 0;

    vsag::init();

    std::cout << "== issue #3007 harness ==\n"
              << "dim=" << dim << " batch=" << batch << " batches=" << batches
              << " total=" << total << " build_threads=" << threads
              << " mode=" << (use_build ? "single Build()" : "batched Add()") << "\n";

    // Synthetic dataset. Log-normal magnitude gives a skewed radius distribution,
    // closer to real embedding data than a uniform hypercube.
    std::mt19937 rng(47);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::lognormal_distribution<float> magnitude(0.0F, 0.6F);
    std::vector<float> data(static_cast<size_t>(total) * dim);
    {
        auto t0 = wall_seconds_now();
        for (int64_t i = 0; i < total; ++i) {
            const float scale = magnitude(rng);
            float* row = data.data() + static_cast<size_t>(i) * dim;
            for (int64_t d = 0; d < dim; ++d) {
                row[d] = normal(rng) * scale;
            }
        }
        std::cout << "generated data in " << (wall_seconds_now() - t0) << " s\n";
    }
    std::vector<int64_t> ids(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        ids[i] = i;
    }

    const int64_t max_degree = arg_int(argc, argv, "--max-degree", 48);
    const bool digest = arg_int(argc, argv, "--digest", 0) != 0;
    const int64_t ef_construction = arg_int(argc, argv, "--ef-construction", 600);
    std::cout << "max_degree=" << max_degree << " ef_construction=" << ef_construction << "\n";

    std::string params = R"({
        "dtype": "float32",
        "metric_type": "ip",
        "dim": )" + std::to_string(dim) +
                         R"(,
        "index_param": {
            "base_quantization_type": "rabitq",
            "precise_quantization_type": "rabitq",
            "rabitq_bits_per_dim_base": 1,
            "rabitq_bits_per_dim_precise": 7,
            "rabitq_bits_per_dim_query": 32,
            "rabitq_error_rate": 1.9,
            "rabitq_fused_datacell": true,
            "rabitq_use_fht": true,
            "rabitq_pca_dim": 0,
            "fast_encode_rabitq": true,
            "max_degree": )" + std::to_string(max_degree) + R"(,
            "ef_construction": )" + std::to_string(ef_construction) + R"(,
            "build_thread_count": )" +
                         std::to_string(threads) +
                         R"(,
            "graph_type": "odescent",
            "graph_storage_type": "flat",
            "base_io_type": "memory_io",
            "base_supplement_io_type": "memory_io",
            "graph_io_type": "memory_io",
            "precise_io_type": "block_memory_io",
            "use_reorder": true,
            "reorder_source": "base",
            "build_by_base": true,
            "store_raw_vector": true,
            "use_mci": false
        }
    })";

    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", params).value();

    const double cpu_0 = cpu_seconds_now();
    const double wall_0 = wall_seconds_now();

    if (use_build) {
        auto base = vsag::Dataset::Make();
        base->NumElements(total)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float32Vectors(data.data())
            ->Owner(false);
        auto t0 = wall_seconds_now();
        auto result = index->Build(base);
        if (not result.has_value()) {
            std::cerr << "Build failed: " << result.error().message << "\n";
            return 1;
        }
        const double dt = wall_seconds_now() - t0;
        std::printf("Build: %.3f s  (%.1f vec/s)\n", dt, static_cast<double>(total) / dt);
    } else {
        for (int64_t b = 0; b < batches; ++b) {
            auto base = vsag::Dataset::Make();
            base->NumElements(batch)
                ->Dim(dim)
                ->Ids(ids.data() + b * batch)
                ->Float32Vectors(data.data() + static_cast<size_t>(b) * batch * dim)
                ->Owner(false);

            const double cpu_before = cpu_seconds_now();
            const double wall_before = wall_seconds_now();
            auto result = index->Add(base);
            const double wall_dt = wall_seconds_now() - wall_before;
            const double cpu_dt = cpu_seconds_now() - cpu_before;
            if (not result.has_value()) {
                std::cerr << "Add failed: " << result.error().message << "\n";
                return 1;
            }
            std::printf(
                "batch %3lld  rows=%lld  add=%7.3f s  %8.1f vec/s  cpu=%7.3f s  "
                "cpu/wall=%6.2fx  total=%lld\n",
                static_cast<long long>(b),
                static_cast<long long>(batch),
                wall_dt,
                static_cast<double>(batch) / wall_dt,
                cpu_dt,
                cpu_dt / wall_dt,
                static_cast<long long>(index->GetNumElements()));
            std::fflush(stdout);
        }
    }

    if (digest) {
        // Deterministic post-build digest: fixed queries, fixed topk, ordered ids.
        // Identical graphs (same neighbour lists) produce identical digests.
        int64_t same = 0;
        int64_t diff = 0;
        int64_t repeat_mismatch = 0;
        std::string search_params = R"({"hgraph": {"ef_search": 100}})";
        for (int64_t qi = 0; qi < 50; ++qi) {
            auto q = vsag::Dataset::Make();
            q->NumElements(1)->Dim(dim)->Float32Vectors(data.data() + static_cast<size_t>(qi) * dim)->Owner(false);
            auto res = index->KnnSearch(q, 10, search_params);
            auto res2 = index->KnnSearch(q, 10, search_params);
            if (not res.has_value() or not res2.has_value()) { std::printf("DIGEST search-failed\n"); break; }
            const auto* rid = res.value()->GetIds();
            const auto* rid2 = res2.value()->GetIds();
            for (int64_t k = 0; k < 10; ++k) {
                if (rid[k] != rid2[k]) { ++repeat_mismatch; }
                if (rid[k] >= 0) { same += rid[k] + 1; } else { ++diff; }
            }
        }
        // Direct graph-content fingerprint: tells "the graph differs" apart from "search
        // amplifies a difference".
        {
            // Same downcast path the HGraph tests use.
            uint64_t graph_hash = 0;
            auto index_impl = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(index);
            if (index_impl != nullptr) {
                auto hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index_impl->GetInnerIndex());
                if (hgraph != nullptr) {
                    graph_hash = hgraph->GraphChecksum();
                }
            }
            std::printf("GRAPHCHECK hash=%llu\n", static_cast<unsigned long long>(graph_hash));
        }
        std::printf("DIGEST sum=%lld neg=%lld elements=%lld repeat_mismatch=%lld\n",
                    static_cast<long long>(same), static_cast<long long>(diff),
                    static_cast<long long>(index->GetNumElements()),
                    static_cast<long long>(repeat_mismatch));
    }

    const double cpu_dt = cpu_seconds_now() - cpu_0;
    const double wall_dt = wall_seconds_now() - wall_0;
    std::printf("TOTAL wall=%.3f s  cpu=%.3f s  cpu/wall=%.2fx  avg=%.1f vec/s  (max %.0f cores)\n",
                wall_dt,
                cpu_dt,
                cpu_dt / wall_dt,
                static_cast<double>(total) / wall_dt,
                cpu_dt / wall_dt);

    engine.Shutdown();
    return 0;
}
