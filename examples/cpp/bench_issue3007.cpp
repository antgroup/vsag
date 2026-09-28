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

// ---------------------------------------------------------------------------
// Minimal SIGPROF sampling profiler.
//
// This container has no perf and gdb cannot attach, so sampling is done in-process:
// one POSIX timer per thread delivers SIGPROF at a fixed interval, and the signal
// handler records the interrupted instruction pointer. Addresses are resolved to
// symbols offline (addr2line against the dumped /proc/self/maps), so no external
// profiler is needed. Async-signal-safe: only a bounded atomic array is written.
// ---------------------------------------------------------------------------
#include <dirent.h>
#include <sys/syscall.h>

#include <csignal>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>

namespace sampling {

constexpr int kMaxSamples = 200000;
constexpr int kFrames = 6;
std::atomic<uint64_t> g_ips[kMaxSamples];
// Return addresses leading to the interrupted instruction, so a sample that lands in
// futex_wait can be attributed to the exact call site / lock that blocked.
std::atomic<uint64_t> g_frames[kMaxSamples][kFrames];
std::atomic<int> g_count{0};

// Signal-safe frame-pointer walk. The build keeps frame pointers enabled
// (ENABLE_FRAME_POINTER defaults on), so rbp-chasing is valid.
extern "C" void
handler(int, siginfo_t*, void* ucontext) {
    auto* uc = static_cast<ucontext_t*>(ucontext);
    const int slot = g_count.fetch_add(1, std::memory_order_relaxed);
    if (slot >= kMaxSamples) {
        return;
    }
    g_ips[slot].store(static_cast<uint64_t>(uc->uc_mcontext.gregs[REG_RIP]),
                      std::memory_order_relaxed);
    uint64_t* rbp = reinterpret_cast<uint64_t*>(uc->uc_mcontext.gregs[REG_RBP]);
    for (int i = 0; i < kFrames; ++i) {
        if (rbp == nullptr) {
            break;
        }
        const uint64_t ret = rbp[1];
        g_frames[slot][i].store(ret, std::memory_order_relaxed);
        auto* next = reinterpret_cast<uint64_t*>(rbp[0]);
        if (next <= rbp) {
            break;
        }
        rbp = next;
    }
}

void
start_thread_timer(int interval_us) {
    struct sigaction sa {};
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, nullptr);

    struct sigevent sev {};
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGPROF;
    sev._sigev_un._tid = static_cast<pid_t>(::syscall(SYS_gettid));
    timer_t timer{};
    if (timer_create(CLOCK_MONOTONIC, &sev, &timer) != 0) {
        return;
    }
    struct itimerspec its {};
    its.it_interval.tv_nsec = interval_us * 1000L;
    its.it_value.tv_nsec = interval_us * 1000L;
    timer_settime(timer, 0, &its, nullptr);
}

void
start_all_threads(int interval_us) {
    static const bool installed = []() {
        struct sigaction sa {};
        sa.sa_sigaction = handler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGPROF, &sa, nullptr);
        return true;
    }();
    (void)installed;
    DIR* dir = opendir("/proc/self/task");
    if (dir == nullptr) {
        return;
    }
    while (struct dirent* ent = readdir(dir)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        const auto tid = static_cast<pid_t>(std::strtol(ent->d_name, nullptr, 10));
        struct sigevent sev {};
        sev.sigev_notify = SIGEV_THREAD_ID;
        sev.sigev_signo = SIGPROF;
        sev._sigev_un._tid = tid;
        timer_t timer{};
        if (timer_create(CLOCK_MONOTONIC, &sev, &timer) != 0) {
            continue;
        }
        struct itimerspec its {};
        its.it_interval.tv_nsec = interval_us * 1000L;
        its.it_value.tv_nsec = interval_us * 1000L;
        timer_settime(timer, 0, &its, nullptr);
    }
    closedir(dir);
}

void
dump(const char* path) {
    const int n = std::min(g_count.load(), kMaxSamples);
    if (n <= 0) {
        return;
    }
    FILE* f = std::fopen(path, "a");
    if (f == nullptr) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        std::fprintf(f, "%llx", static_cast<unsigned long long>(g_ips[i].load()));
        for (int k = 0; k < kFrames; ++k) {
            std::fprintf(f, " %llx", static_cast<unsigned long long>(g_frames[i][k].load()));
        }
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

}  // namespace sampling



#include "algorithm/hgraph/hgraph.h"
#include "index/index_impl.h"
#include "utils/lock_strategy.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>
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
    const bool profile = arg_int(argc, argv, "--profile", 0) != 0;
    const int64_t stress_readers = arg_int(argc, argv, "--stress-readers", 0);
    const int64_t profile_us = arg_int(argc, argv, "--profile-us", 500);
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

    std::atomic<bool> stress_stop{false};
    std::atomic<int64_t> stress_queries{0};
    std::atomic<int64_t> stress_bad_ids{0};
    std::atomic<int64_t> stress_failed{0};
    std::vector<std::thread> stress_threads;
    if (stress_readers > 0) {
        for (int64_t r = 0; r < stress_readers; ++r) {
            stress_threads.emplace_back([&, r]() {
                std::string sp = R"({"hgraph": {"ef_search": 100}})";
                int64_t q = r * 7;
                while (not stress_stop.load(std::memory_order_relaxed)) {
                    const int64_t row = (q++) % total;
                    const float* qv = data.data() + static_cast<size_t>(row) * dim;
                    auto qds = vsag::Dataset::Make();
                    qds->NumElements(1)->Dim(dim)->Float32Vectors(const_cast<float*>(qv))->Owner(false);
                    auto res = index->KnnSearch(qds, 10, sp);
                    stress_queries.fetch_add(1, std::memory_order_relaxed);
                    if (not res.has_value()) {
                        stress_failed.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    const auto* got = res.value()->GetIds();
                    const int64_t elements = index->GetNumElements();
                    if (elements == 0) {
                        continue;  // nothing built yet; no id can be validated
                    }
                    for (int64_t k = 0; k < 10; ++k) {
                        // Every non-negative id must name a node that exists.
                        if (got[k] >= 0 and got[k] >= elements) {
                            stress_bad_ids.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
            });
        }
        std::printf("STRESS readers=%lld (concurrent with Add)\n",
                    static_cast<long long>(stress_readers));
    }

    if (profile) {
        std::remove("/tmp/vsag_samples.txt");
        std::printf("PROFILING interval=%lldus\n", static_cast<long long>(profile_us));
        sampling::start_all_threads(static_cast<int>(profile_us));
    }
    // Dump samples as soon as the workload finishes (also on the error paths below).
    struct SampleDumper {
        bool enabled;
        ~SampleDumper() {
            if (enabled) {
                sampling::dump("/tmp/vsag_samples.txt");
                if (FILE* m = std::fopen("/tmp/vsag_maps.txt", "w")) {
                    if (FILE* self = std::fopen("/proc/self/maps", "r")) {
                        char buf[4096];
                        while (std::fgets(buf, sizeof(buf), self) != nullptr) {
                            std::fputs(buf, m);
                        }
                        std::fclose(self);
                    }
                    std::fclose(m);
                }
            }
        }
    } sample_dumper{profile};

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

    // Held-out query set: drawn after the base data, never inserted, so recall measures
    // search quality rather than "did the index find the vector we just inserted".
    std::vector<float> queries;
    if (digest) {
        std::normal_distribution<float> qnormal(0.0F, 1.0F);
        std::lognormal_distribution<float> qmag(0.0F, 0.6F);
        queries.resize(static_cast<size_t>(50) * dim);
        for (int64_t qi = 0; qi < 50; ++qi) {
            const float scale = qmag(rng);
            for (int64_t d2 = 0; d2 < dim; ++d2) {
                queries[static_cast<size_t>(qi) * dim + d2] = qnormal(rng) * scale;
            }
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
            std::shared_ptr<vsag::HGraph> hgraph;
            auto index_impl = std::dynamic_pointer_cast<vsag::IndexImpl<vsag::HGraph>>(index);
            if (index_impl != nullptr) {
                hgraph = std::dynamic_pointer_cast<vsag::HGraph>(index_impl->GetInnerIndex());
                if (hgraph != nullptr) {
                    graph_hash = hgraph->GraphChecksum();
                }
            }
            std::printf("GRAPHCHECK hash=%llu\n", static_cast<unsigned long long>(graph_hash));
            if (hgraph != nullptr) {
                const auto amc = hgraph->add_mutex_stats.exclusive_calls.load();
                const auto amn = hgraph->add_mutex_stats.exclusive_hold_ns.load();
                std::printf("ADDMUTEX calls=%llu hold=%.3f s\n",
                            static_cast<unsigned long long>(amc),
                            static_cast<double>(amn) / 1e9);
                auto nbr = std::dynamic_pointer_cast<vsag::PointsMutex>(
                    hgraph->GetNeighborsMutexArray());
                if (nbr != nullptr) {
                    std::printf("NEIGHMUTEX excl_calls=%llu excl_wait=%.3f s "
                                "shared_calls=%llu shared_wait=%.3f s\n",
                                static_cast<unsigned long long>(
                                    nbr->stats.exclusive_calls.load()),
                                static_cast<double>(nbr->stats.exclusive_wait_ns.load()) / 1e9,
                                static_cast<unsigned long long>(nbr->stats.shared_calls.load()),
                                static_cast<double>(nbr->stats.shared_wait_ns.load()) / 1e9);
                }
            }
        }
        // recall@10 against an exact brute-force ground truth computed on the raw vectors
        {
            int64_t hit = 0;
            int64_t recall_total = 0;
            for (int64_t qi = 0; qi < 50; ++qi) {
                const float* q = queries.data() + static_cast<size_t>(qi) * dim;
                std::vector<std::pair<float, int64_t>> exact;
                exact.reserve(static_cast<size_t>(total));  // `total` = dataset size
                for (int64_t j = 0; j < total; ++j) {
                    const float* v = data.data() + static_cast<size_t>(j) * dim;
                    double ip = 0.0;
                    for (int64_t d2 = 0; d2 < dim; ++d2) {
                        ip += static_cast<double>(q[d2]) * static_cast<double>(v[d2]);
                    }
                    exact.emplace_back(static_cast<float>(ip), ids[j]);
                }
                std::partial_sort(exact.begin(), exact.begin() + 10, exact.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                auto qds = vsag::Dataset::Make();
                qds->NumElements(1)->Dim(dim)->Float32Vectors(const_cast<float*>(q))->Owner(false);
                auto res = index->KnnSearch(qds, 10, search_params);
                if (not res.has_value()) { continue; }
                const auto* got = res.value()->GetIds();
                for (int64_t k = 0; k < 10; ++k) {
                    ++recall_total;
                    for (int64_t e = 0; e < 10; ++e) {
                        if (exact[e].second == got[k]) { ++hit; break; }
                    }
                }
            }
            std::printf("RECALL10 %.4f (%lld/%lld)\n",
                        recall_total > 0 ? static_cast<double>(hit) /
                                               static_cast<double>(recall_total)
                                         : 0.0,
                        static_cast<long long>(hit), static_cast<long long>(recall_total));
        }
        std::printf("DIGEST sum=%lld neg=%lld elements=%lld repeat_mismatch=%lld\n",
                    static_cast<long long>(same), static_cast<long long>(diff),
                    static_cast<long long>(index->GetNumElements()),
                    static_cast<long long>(repeat_mismatch));
    }

    if (stress_readers > 0) {
        stress_stop.store(true, std::memory_order_relaxed);
        for (auto& t : stress_threads) {
            t.join();
        }
        std::printf("STRESS queries=%lld failed=%lld bad_ids=%lld\n",
                    static_cast<long long>(stress_queries.load()),
                    static_cast<long long>(stress_failed.load()),
                    static_cast<long long>(stress_bad_ids.load()));
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
