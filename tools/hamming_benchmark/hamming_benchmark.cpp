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

// Benchmarks the Hamming distance kernels of VSAG: the generic scalar implementation and every
// SIMD implementation the running CPU supports, on common binary dimensions. Before timing, each
// implementation is checked against the generic one on every vector of the benchmark set.
//
// Usage:
//   hamming_benchmark [--dims 64,128,256,512,1024,2048,4096] [--vectors 2048]
//                     [--min-time 0.3] [--json report.json]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "simd/hamming_simd.h"
#include "simd/simd_status.h"

namespace {

struct Kernel {
    std::string name;
    vsag::HammingDistanceType single;
    vsag::HammingDistanceBatch4Type batch4;
    vsag::HammingDistanceScanType scan;
};

#define HAMMING_KERNEL(label, ns)                                          \
    Kernel {                                                               \
        label, vsag::ns::HammingDistance, vsag::ns::HammingDistanceBatch4, \
            vsag::ns::HammingDistanceScan                                  \
    }

std::vector<Kernel>
available_kernels() {
    using vsag::SimdStatus;
    std::vector<Kernel> kernels{HAMMING_KERNEL("generic", generic)};
    if (SimdStatus::SupportPOPCNT()) {
        kernels.push_back(HAMMING_KERNEL("popcnt", sse));
    }
    if (SimdStatus::SupportAVX2()) {
        kernels.push_back(HAMMING_KERNEL("avx2", avx2));
    }
    if (SimdStatus::SupportAVX512()) {
        kernels.push_back(HAMMING_KERNEL("avx512bw", avx512));
    }
    if (SimdStatus::SupportAVX512() and SimdStatus::SupportAVX512VPOPCNTDQ()) {
        kernels.push_back(HAMMING_KERNEL("avx512vpopcntdq", avx512vpopcntdq));
    }
    if (SimdStatus::SupportNEON()) {
        kernels.push_back(HAMMING_KERNEL("neon", neon));
    }
    if (SimdStatus::SupportSVE()) {
        kernels.push_back(HAMMING_KERNEL("sve", sve));
    }
    return kernels;
}

std::string
dispatched_name(const std::vector<Kernel>& kernels) {
    for (const auto& kernel : kernels) {
        if (kernel.single == vsag::HammingDistance) {
            return kernel.name;
        }
    }
    return "unknown";
}

struct Options {
    std::vector<uint64_t> dims{64, 128, 256, 512, 1024, 2048, 4096};
    uint64_t vectors{2048};
    double min_time{0.3};
    std::string json_path;
};

Options
parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << arg << std::endl;
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--dims") {
            options.dims.clear();
            std::stringstream list(value());
            std::string item;
            while (std::getline(list, item, ',')) {
                options.dims.push_back(std::stoull(item));
            }
        } else if (arg == "--vectors") {
            options.vectors = std::max<uint64_t>(4, std::stoull(value()) / 4 * 4);
        } else if (arg == "--min-time") {
            options.min_time = std::stod(value());
        } else if (arg == "--json") {
            options.json_path = value();
        } else {
            std::cerr << "usage: hamming_benchmark [--dims 64,128,...] [--vectors N] "
                         "[--min-time seconds] [--json path]"
                      << std::endl;
            std::exit(arg == "--help" ? 0 : 2);
        }
    }
    return options;
}

// One pass compares the query with every vector; returns the sum of all distances.
uint64_t
pass_single(const Kernel& kernel,
            const uint8_t* query,
            const std::vector<uint8_t>& codes,
            uint64_t count,
            uint64_t bytes,
            uint64_t dim) {
    uint64_t sum = 0;
    for (uint64_t i = 0; i < count; ++i) {
        sum += kernel.single(query, codes.data() + i * bytes, dim);
    }
    return sum;
}

uint64_t
pass_batch4(const Kernel& kernel,
            const uint8_t* query,
            const std::vector<uint8_t>& codes,
            uint64_t count,
            uint64_t bytes,
            uint64_t dim) {
    uint64_t sum = 0;
    uint64_t results[4];
    for (uint64_t i = 0; i < count; i += 4) {
        const uint8_t* block = codes.data() + i * bytes;
        kernel.batch4(
            query, block, block + bytes, block + 2 * bytes, block + 3 * bytes, dim, results);
        sum += results[0] + results[1] + results[2] + results[3];
    }
    return sum;
}

uint64_t
pass_scan(const Kernel& kernel,
          const uint8_t* query,
          const std::vector<uint8_t>& codes,
          uint64_t count,
          uint64_t dim,
          std::vector<float>& dists) {
    kernel.scan(query, codes.data(), dim, count, dists.data());
    return static_cast<uint64_t>(dists.front() + dists.back());
}

template <typename Pass>
double
nanoseconds_per_distance(Pass pass, uint64_t count, double min_time, volatile uint64_t& sink) {
    using clock = std::chrono::steady_clock;
    sink += pass();  // warm up caches and branch predictors
    uint64_t passes = 0;
    const auto start = clock::now();
    double elapsed = 0.0;
    do {
        sink += pass();
        ++passes;
        elapsed = std::chrono::duration<double>(clock::now() - start).count();
    } while (elapsed < min_time);
    return elapsed * 1e9 / static_cast<double>(passes * count);
}

}  // namespace

int
main(int argc, char** argv) {
    const auto options = parse_options(argc, argv);
    const auto kernels = available_kernels();
    std::cout << "# Hamming kernel benchmark" << std::endl;
    std::cout << "kernels:";
    for (const auto& kernel : kernels) {
        std::cout << " " << kernel.name;
    }
    std::cout << "\ndispatched: " << dispatched_name(kernels)
              << "\nvectors per pass: " << options.vectors
              << ", minimum time per measurement: " << options.min_time << " s\n\n";

    std::ostringstream json;
    json << "{\"dispatched\":\"" << dispatched_name(kernels) << "\",\"vectors\":" << options.vectors
         << ",\"results\":[";
    bool first_json = true;
    volatile uint64_t sink = 0;
    bool consistent = true;

    std::cout << "| bits | bytes | kernel | single ns/dist | batch4 ns/dist | scan ns/dist | "
                 "scan GB/s | speedup vs generic |\n";
    std::cout << "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |\n";
    std::mt19937_64 rng(2872);
    for (const auto dim : options.dims) {
        const uint64_t bytes = (dim + 7) / 8;
        std::vector<uint8_t> codes(options.vectors * bytes);
        std::vector<uint8_t> query(bytes);
        for (auto& value : codes) {
            value = static_cast<uint8_t>(rng());
        }
        for (auto& value : query) {
            value = static_cast<uint8_t>(rng());
        }
        // Correctness first: every kernel must reproduce the generic distance of every vector,
        // one code at a time and in a scan over all of them.
        std::vector<float> dists(options.vectors);
        for (const auto& kernel : kernels) {
            kernel.scan(query.data(), codes.data(), dim, options.vectors, dists.data());
            for (uint64_t i = 0; i < options.vectors; ++i) {
                const auto expected =
                    vsag::generic::HammingDistance(query.data(), codes.data() + i * bytes, dim);
                if (kernel.single(query.data(), codes.data() + i * bytes, dim) != expected or
                    dists[i] != static_cast<float>(expected)) {
                    std::cerr << "MISMATCH: " << kernel.name << " dim=" << dim << " row=" << i
                              << std::endl;
                    consistent = false;
                }
            }
        }

        double generic_ns = 0.0;
        for (const auto& kernel : kernels) {
            const double single_ns = nanoseconds_per_distance(
                [&]() {
                    return pass_single(kernel, query.data(), codes, options.vectors, bytes, dim);
                },
                options.vectors,
                options.min_time,
                sink);
            const double batch_ns = nanoseconds_per_distance(
                [&]() {
                    return pass_batch4(kernel, query.data(), codes, options.vectors, bytes, dim);
                },
                options.vectors,
                options.min_time,
                sink);
            const double scan_ns = nanoseconds_per_distance(
                [&]() {
                    return pass_scan(kernel, query.data(), codes, options.vectors, dim, dists);
                },
                options.vectors,
                options.min_time,
                sink);
            if (kernel.name == "generic") {
                generic_ns = single_ns;
            }
            const double gbps = static_cast<double>(bytes) / scan_ns;
            const double speedup = generic_ns / std::min({single_ns, batch_ns, scan_ns});
            std::cout << "| " << dim << " | " << bytes << " | " << kernel.name << " | "
                      << std::fixed << std::setprecision(2) << single_ns << " | " << batch_ns
                      << " | " << scan_ns << " | " << gbps << " | " << speedup << "x |\n";
            json << (first_json ? "" : ",") << "{\"dim\":" << dim << ",\"kernel\":\"" << kernel.name
                 << "\",\"single_ns\":" << single_ns << ",\"batch4_ns\":" << batch_ns
                 << ",\"scan_ns\":" << scan_ns << ",\"speedup\":" << speedup << "}";
            first_json = false;
        }
    }
    json << "],\"consistent\":" << (consistent ? "true" : "false") << "}";
    std::cout << "\nall kernels agree with the generic implementation: "
              << (consistent ? "yes" : "NO") << std::endl;
    if (not options.json_path.empty()) {
        std::ofstream(options.json_path) << json.str() << std::endl;
    }
    return consistent ? 0 : 1;
}
