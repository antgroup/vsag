// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sched.h>
#include <sys/resource.h>
#ifdef VSAG_BENCH_FULL
#include <vsag/dataset.h>
#include <vsag/factory.h>
#include <vsag/index.h>
#else
#include <vsag/lite/index.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}
uint64_t
number(const char* argument) {
    const std::string text(argument);
    require(
        not text.empty() and
            std::all_of(
                text.begin(), text.end(), [](char value) { return value >= '0' and value <= '9'; }),
        "invalid decimal integer");
    return std::stoull(text);
}
struct Neighbor {
    int64_t id;
    float distance;
};
double
cpu_us(const rusage& usage) {
    return 1e6 * static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
           static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec);
}
int
run(int argc, char** argv) {
    require(argc == 12, "snapshot queries dim count ef begin end output full-config storage cpu");
    const uint64_t dim = number(argv[3]);
    const uint64_t count = number(argv[4]);
    const uint64_t ef = number(argv[5]);
    require(sched_getcpu() == static_cast<int>(number(argv[11])), "CPU affinity differs");
    const uint64_t begin = number(argv[6]);
    const uint64_t end = number(argv[7]);
    const std::string output(argv[8]);
    require(dim > 0 and dim <= 4096 and count > 0 and ef >= 10 and begin < end and end <= 1000,
            "invalid query bounds");
    require(not std::filesystem::exists(output + ".queries.csv") and
                not std::filesystem::exists(output + ".neighbors.csv"),
            "output exists");
    std::ifstream queries(argv[2], std::ios::binary);
    require(static_cast<bool>(queries), "query open failed");
    std::vector<float> vectors((end - begin) * dim);
    for (uint64_t row = 0; row < end; ++row) {
        int32_t width = 0;
        queries.read(reinterpret_cast<char*>(&width), sizeof(width));
        require(static_cast<bool>(queries) and width == static_cast<int32_t>(dim),
                "query shape differs");
        if (row < begin) {
            queries.seekg(static_cast<std::streamoff>(dim * sizeof(float)), std::ios::cur);
        } else {
            queries.read(reinterpret_cast<char*>(vectors.data() + (row - begin) * dim),
                         static_cast<std::streamsize>(dim * sizeof(float)));
        }
        require(static_cast<bool>(queries), "truncated query payload");
    }
    require(std::all_of(
                vectors.begin(), vectors.end(), [](float value) { return std::isfinite(value); }),
            "nonfinite query");
    std::ifstream snapshot(argv[1], std::ios::binary);
    require(static_cast<bool>(snapshot), "snapshot open failed");
#ifdef VSAG_BENCH_FULL
    std::ifstream configuration(argv[9]);
    require(static_cast<bool>(configuration), "Full configuration open failed");
    const std::string config((std::istreambuf_iterator<char>(configuration)), {});
    auto created = vsag::Factory::CreateIndex("hgraph", config);
    require(static_cast<bool>(created), "Full create failed");
    auto index = *created;
    require(static_cast<bool>(index->Deserialize(snapshot)) and
                index->GetNumElements() == static_cast<int64_t>(count),
            "Full load/count failed");
    const std::string parameters = std::string(R"({"hgraph":{"ef_search":)") + std::to_string(ef) +
                                   R"(,"rabitq_one_bit_search":)" +
                                   (std::string(argv[10]) == "rabitq8" ? "true}}" : "false}}");
    std::vector<vsag::DatasetPtr> inputs;
    for (uint64_t row = 0; row < end - begin; ++row) {
        inputs.push_back(vsag::Dataset::Make()
                             ->NumElements(1)
                             ->Dim(static_cast<int64_t>(dim))
                             ->Float32Vectors(vectors.data() + row * dim)
                             ->Owner(false));
    }
#else
    auto loaded = vsag::lite::Index::Load(snapshot);
    require(static_cast<bool>(loaded), "Lite load failed");
    auto index = std::move(*loaded);
    require(index->Size() == count and index->Dim() == dim, "Lite load/count failed");
#endif
    using Clock = std::chrono::steady_clock;
    const uint64_t n = end - begin;
    std::vector<double> latencies(n);
    std::vector<std::vector<Neighbor>> neighbors(n);
    rusage before{}, after{};
    for (uint64_t pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            require(getrusage(RUSAGE_SELF, &before) == 0, "resource capture failed");
        }
        for (uint64_t row = 0; row < n; ++row) {
            const auto start = Clock::now();
#ifdef VSAG_BENCH_FULL
            auto found = index->KnnSearch(inputs[row], 10, parameters);
#else
            auto found = index->SearchWithOptions(vectors.data() + row * dim, dim, 10, {ef});
#endif
            const double elapsed =
                std::chrono::duration<double, std::micro>(Clock::now() - start).count();
            require(static_cast<bool>(found), "query failed");
            if (pass == 1) {
                latencies[row] = elapsed;
                neighbors[row].reserve(10);
#ifdef VSAG_BENCH_FULL
                require((*found)->GetDim() == 10, "Full result count differs");
                for (uint64_t rank = 0; rank < 10; ++rank) {
                    neighbors[row].push_back(
                        {(*found)->GetIds()[rank], (*found)->GetDistances()[rank]});
                }
#else
                require(found->size() == 10, "Lite result count differs");
                for (const auto& neighbor : *found) {
                    neighbors[row].push_back({neighbor.id, neighbor.distance});
                }
#endif
            }
        }
    }
    require(getrusage(RUSAGE_SELF, &after) == 0, "resource capture failed");
    std::ofstream samples(output + ".queries.csv"), results(output + ".neighbors.csv");
    require(static_cast<bool>(samples) and static_cast<bool>(results), "output open failed");
    samples << "query,latency_us\n" << std::setprecision(17);
    results << "query,rank,id,distance\n" << std::hexfloat;
    for (uint64_t row = 0; row < n; ++row) {
        samples << begin + row << ',' << latencies[row] << '\n';
        for (uint64_t rank = 0; rank < 10; ++rank) {
            const auto& neighbor = neighbors[row][rank];
            results << begin + row << ',' << rank << ',' << neighbor.id << ',' << neighbor.distance
                    << '\n';
        }
    }
    samples.close();
    results.close();
    require(static_cast<bool>(samples) and static_cast<bool>(results), "output write failed");
    std::cout << "queries,ef,phase_cpu_us,minor_faults,major_faults,voluntary_switches,involuntary_"
                 "switches,process_peak_rss_kib\n"
              << n << ',' << ef << ',' << std::setprecision(17) << cpu_us(after) - cpu_us(before)
              << ',' << after.ru_minflt - before.ru_minflt << ','
              << after.ru_majflt - before.ru_majflt << ',' << after.ru_nvcsw - before.ru_nvcsw
              << ',' << after.ru_nivcsw - before.ru_nivcsw << ',' << after.ru_maxrss << '\n';
    return 0;
}
}  // namespace
int
main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
