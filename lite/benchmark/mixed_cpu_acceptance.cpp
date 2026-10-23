// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vsag/lite/index.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Rows = std::vector<std::vector<float>>;

void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}

Rows
read_rows(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "open fvecs failed");
    Rows rows;
    int32_t first_dim = 0;
    while (input.peek() != std::char_traits<char>::eof()) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        require(static_cast<bool>(input) and dim > 0 and dim <= 4096, "invalid dimension");
        if (rows.empty()) {
            first_dim = dim;
        }
        require(dim == first_dim, "dimension mismatch");
        std::vector<float> row(dim);
        input.read(reinterpret_cast<char*>(row.data()),
                   static_cast<std::streamsize>(dim * sizeof(float)));
        require(static_cast<bool>(input), "truncated row");
        for (float value : row) {
            require(std::isfinite(value), "nonfinite input");
        }
        rows.push_back(std::move(row));
    }
    require(not rows.empty(), "empty input");
    return rows;
}

rusage
usage() {
    rusage result{};
    require(getrusage(RUSAGE_SELF, &result) == 0, "getrusage failed");
    return result;
}

double
milliseconds(const timeval& value) {
    return static_cast<double>(value.tv_sec) * 1000.0 + static_cast<double>(value.tv_usec) / 1000.0;
}

struct Sample {
    uint64_t round;
    int64_t id;
    std::string operation;
    double wall_us;
    double user_us;
    double system_us;
    int64_t voluntary;
    int64_t involuntary;
    int64_t peak_kib;
};

template <typename Action>
void
measure(std::vector<Sample>& samples,
        uint64_t round,
        int64_t id,
        const std::string& operation,
        const Action& action) {
    const auto before = usage();
    const auto begin = Clock::now();
    action();
    const auto end = Clock::now();
    const auto after = usage();
    samples.push_back({round,
                       id,
                       operation,
                       std::chrono::duration<double, std::micro>(end - begin).count(),
                       (milliseconds(after.ru_utime) - milliseconds(before.ru_utime)) * 1000,
                       (milliseconds(after.ru_stime) - milliseconds(before.ru_stime)) * 1000,
                       after.ru_nvcsw - before.ru_nvcsw,
                       after.ru_nivcsw - before.ru_nivcsw,
                       after.ru_maxrss});
}

uint64_t
status_number(const std::string& field) {
    std::ifstream input("/proc/self/status");
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(field, 0) == 0) {
            return std::stoull(line.substr(field.size()));
        }
    }
    throw std::runtime_error("process status field unavailable");
}

uint64_t
rss_kib() {
    return status_number("VmRSS:");
}

std::vector<int64_t>
truth(const Rows& base, const std::vector<float>& query, uint64_t k) {
    std::vector<std::pair<double, int64_t>> distances;
    distances.reserve(base.size());
    for (uint64_t id = 0; id < base.size(); ++id) {
        double distance = 0;
        for (uint64_t d = 0; d < query.size(); ++d) {
            const double delta = static_cast<double>(base[id][d]) - query[d];
            distance += delta * delta;
        }
        distances.emplace_back(distance, static_cast<int64_t>(id));
    }
    std::partial_sort(
        distances.begin(), distances.begin() + static_cast<int64_t>(k), distances.end());
    std::vector<int64_t> result;
    for (uint64_t rank = 0; rank < k; ++rank) {
        result.push_back(distances[rank].second);
    }
    return result;
}

void
validate(const std::vector<vsag::lite::Neighbor>& neighbors,
         const std::vector<bool>& live,
         uint64_t k) {
    require(neighbors.size() == k, "result size changed");
    std::vector<int64_t> ids;
    for (const auto& neighbor : neighbors) {
        require(neighbor.id >= 0 and static_cast<uint64_t>(neighbor.id) < live.size() and
                    live[neighbor.id] and std::isfinite(neighbor.distance),
                "dead, invalid or nonfinite result");
        ids.push_back(neighbor.id);
    }
    std::sort(ids.begin(), ids.end());
    require(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "duplicate result");
    require(std::is_sorted(neighbors.begin(),
                           neighbors.end(),
                           [](const auto& a, const auto& b) {
                               return a.distance < b.distance or
                                      (a.distance == b.distance and a.id < b.id);
                           }),
            "unordered results");
}

// Isolate fresh-build/oracle allocations from the resident parent's allocator and RSS.
// This is a Linux single-threaded diagnostic, not a crash-durability or concurrency test.
void
checkpoint(const std::filesystem::path& output,
           uint64_t round,
           uint64_t cycles,
           const Rows& base,
           const Rows& queries,
           const vsag::lite::Index& maintained,
           vsag::lite::VectorStorage format,
           uint64_t calls_per_cycle,
           bool ceiling = false) {
    require(status_number("Threads:") == 1, "checkpoint requires one thread");
    const auto child = fork();
    require(child >= 0, "checkpoint fork failed");
    if (child == 0) {
        int status = 0;
        try {
            const auto begin = Clock::now();
            auto created = vsag::lite::Index::Create(base.front().size());
            require(static_cast<bool>(created), "checkpoint create failed");
            auto fresh = std::move(*created);
            for (uint64_t id = 0; id < base.size(); ++id) {
                require(static_cast<bool>(
                            fresh->Add(static_cast<int64_t>(id), base[id].data(), base[id].size())),
                        "checkpoint add failed");
            }
            require(static_cast<bool>(fresh->BuildGraph(format, 16, 128)),
                    "checkpoint rebuild failed");
            const auto rebuilt = Clock::now();
            const auto prefix = output / ("checkpoint-" + std::to_string(round));
            std::ofstream neighbors(prefix.string() + "-neighbors.csv");
            std::ofstream timings(prefix.string() + "-queries.csv");
            std::ofstream state(prefix.string() + "-state.csv");
            neighbors << "round,query,rank,truth_id,maintained_id,maintained_distance,fresh_id,"
                         "fresh_distance\n"
                      << std::setprecision(17);
            timings << "round,query,maintained_hits,fresh_hits,k,maintained_us,fresh_us\n"
                    << std::setprecision(17);
            state << "round,mutation_calls,query_calls,count,rebuild_ms\n"
                  << std::setprecision(17) << round << ',' << round * cycles * calls_per_cycle
                  << ',' << round * cycles * calls_per_cycle << ',' << base.size() << ','
                  << std::chrono::duration<double, std::milli>(rebuilt - begin).count() << '\n';
            std::ofstream exhaustive;
            if (ceiling) {
                exhaustive.open(output / "ceiling.csv");
                exhaustive
                    << "round,query,rank,ef_search,truth_id,maintained_id,maintained_distance,"
                       "fresh_id,fresh_distance\n"
                    << std::setprecision(17);
            }
            const std::vector<bool> live(base.size(), true);
            const vsag::lite::SearchOptions options{512};
            for (uint64_t q = 0; q < queries.size(); ++q) {
                const auto expected = truth(base, queries[q], 10);
                std::vector<vsag::lite::Neighbor> results[2];
                double elapsed[2]{};
                // Rotate the first search to avoid always warming the same side first.
                for (uint64_t step = 0; step < 2; ++step) {
                    const uint64_t side = (q + step) % 2;
                    const auto& selected = side == 0 ? maintained : *fresh;
                    const auto start = Clock::now();
                    auto found = selected.SearchWithOptions(
                        queries[q].data(), queries[q].size(), 10, options);
                    elapsed[side] =
                        std::chrono::duration<double, std::micro>(Clock::now() - start).count();
                    require(static_cast<bool>(found), "checkpoint query failed");
                    results[side] = std::move(*found);
                    validate(results[side], live, 10);
                }
                uint64_t hits[2]{};
                for (uint64_t rank = 0; rank < 10; ++rank) {
                    for (uint64_t side = 0; side < 2; ++side) {
                        hits[side] += static_cast<uint64_t>(
                            std::find(expected.begin(), expected.end(), results[side][rank].id) !=
                            expected.end());
                    }
                    neighbors << round << ',' << q << ',' << rank << ',' << expected[rank] << ','
                              << results[0][rank].id << ',' << results[0][rank].distance << ','
                              << results[1][rank].id << ',' << results[1][rank].distance << '\n';
                }
                if (ceiling) {
                    std::vector<vsag::lite::Neighbor> full[2];
                    for (uint64_t step = 0; step < 2; ++step) {
                        const uint64_t side = (q + step) % 2;
                        const auto& selected = side == 0 ? maintained : *fresh;
                        auto found = selected.SearchWithOptions(
                            queries[q].data(), queries[q].size(), 10, {base.size()});
                        require(static_cast<bool>(found), "ceiling query failed");
                        full[side] = std::move(*found);
                        validate(full[side], live, 10);
                    }
                    for (uint64_t rank = 0; rank < 10; ++rank) {
                        exhaustive << round << ',' << q << ',' << rank << ',' << base.size() << ','
                                   << expected[rank] << ',' << full[0][rank].id << ','
                                   << full[0][rank].distance << ',' << full[1][rank].id << ','
                                   << full[1][rank].distance << '\n';
                    }
                }
                timings << round << ',' << q << ',' << hits[0] << ',' << hits[1] << ",10,"
                        << elapsed[0] << ',' << elapsed[1] << '\n';
            }
            if (ceiling) {
                exhaustive.close();
                require(static_cast<bool>(exhaustive), "ceiling evidence write failed");
            }
            neighbors.close();
            timings.close();
            state.close();
            require(static_cast<bool>(neighbors) and static_cast<bool>(timings) and
                        static_cast<bool>(state),
                    "checkpoint evidence write failed");
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            status = 1;
        }
        // Do not run inherited parent stream destructors or flush their copied buffers.
        _exit(status);
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 and errno == EINTR);
    require(waited == child and WIFEXITED(status) and WEXITSTATUS(status) == 0,
            "checkpoint child failed");
}

void
run(int argc, char** argv) {
    require(argc >= 7 and argc <= 10,
            "usage: mixed_cpu_acceptance DATASET OUTPUT STORAGE "
            "ROUNDS CYCLES SNAPDIR [resident [checkpoint_every [mutation_mode]]]");
    const bool resident = argc >= 8;
    require(not resident or std::string(argv[7]) == "resident", "invalid lifecycle mode");
    const uint64_t checkpoint_every = argc >= 9 ? std::stoull(argv[8]) : 0;
    require(argc < 9 or (checkpoint_every > 0 and checkpoint_every <= 1000000),
            "invalid checkpoint interval");
    const std::string mutation_mode = argc == 10 ? argv[9] : "all";
    require(mutation_mode == "all" or mutation_mode == "update" or mutation_mode == "replace" or
                mutation_mode == "reinsert",
            "invalid mutation mode");
    const uint64_t calls_per_cycle = mutation_mode == "all" ? 3 : mutation_mode == "update" ? 1 : 2;
    const std::filesystem::path output(argv[2]);
    require(not std::filesystem::exists(output), "output already exists");
    std::filesystem::create_directory(output);
    auto base = read_rows(std::string(argv[1]) + "/base.fvecs");
    const auto original = base;
    const auto queries = read_rows(std::string(argv[1]) + "/queries.fvecs");
    const uint64_t dim = base.front().size();
    const uint64_t rounds = std::stoull(argv[4]);
    const uint64_t cycles = std::stoull(argv[5]);
    require(rounds <= 1000000 and cycles <= 1000000 and rounds * cycles <= 1000000,
            "measurement sample bound exceeded");
    constexpr uint64_t k = 10;
    require(base.size() > k and rounds > 0 and cycles > 0 and queries.front().size() == dim,
            "invalid configuration");
    const std::string storage(argv[3]);
    require(storage == "fp32" or storage == "fp16" or storage == "rabitq8", "invalid storage");
    const auto format = storage == "fp32"   ? vsag::lite::VectorStorage::FP32
                        : storage == "fp16" ? vsag::lite::VectorStorage::FP16
                                            : vsag::lite::VectorStorage::RABITQ8;
    auto created = vsag::lite::Index::Create(dim);
    require(static_cast<bool>(created), "create failed");
    auto index = std::move(*created);
    std::vector<Sample> samples;
    // Commit the measurement buffer before the initial RSS sample, not progressively during CRUD.
    samples.resize(rounds * cycles * 6 + rounds * 2 + 1);
    samples.clear();
    std::vector<bool> live(base.size(), true);
    const vsag::lite::SearchOptions options{512};
    measure(samples, 0, -1, "build", [&] {
        for (uint64_t id = 0; id < base.size(); ++id) {
            require(static_cast<bool>(index->Add(static_cast<int64_t>(id), base[id].data(), dim)),
                    "initial add failed");
        }
        require(static_cast<bool>(index->BuildGraph(format, 16, 128)), "build failed");
    });
    std::ofstream quality(output / "quality.csv");
    quality << "round,query,hits,k\n";
    const auto quality_check = [&](uint64_t round) {
        for (uint64_t q = 0; q < queries.size(); ++q) {
            const auto expected = truth(base, queries[q], k);
            auto result = index->SearchWithOptions(queries[q].data(), dim, k, options);
            require(static_cast<bool>(result), "quality query failed");
            validate(*result, live, k);
            uint64_t hits = 0;
            for (const auto& neighbor : *result) {
                hits += static_cast<uint64_t>(
                    std::find(expected.begin(), expected.end(), neighbor.id) != expected.end());
            }
            quality << round << ',' << q << ',' << hits << ',' << k << '\n';
        }
    };
    quality_check(0);
    if (checkpoint_every > 0) {
        checkpoint(output, 0, cycles, base, queries, *index, format, calls_per_cycle);
    }
    std::ofstream states(output / "states.csv");
    states << "round,count,rss_kib,peak_kib,snapshot_bytes,rss_before_persistence_kib,threads\n";
    states << "0," << index->Size() << ',' << rss_kib() << ',' << usage().ru_maxrss << ",0,"
           << rss_kib() << ',' << status_number("Threads:") << '\n';
    uint64_t query_number = 0;
    const auto mixed_query = [&](uint64_t round) {
        const auto q = query_number++ % queries.size();
        std::vector<vsag::lite::Neighbor> neighbors;
        measure(samples, round, static_cast<int64_t>(q), "query", [&] {
            auto result = index->SearchWithOptions(queries[q].data(), dim, k, options);
            require(static_cast<bool>(result), "mixed query failed");
            neighbors = std::move(*result);
        });
        validate(neighbors, live, k);
    };
    for (uint64_t round = 1; round <= rounds; ++round) {
        for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
            const uint64_t id = ((round - 1) * 65537ULL + cycle * 8191ULL) % base.size();
            const uint64_t donor = (id + round * 97 + cycle + 1) % base.size();
            std::vector<float> replacement = base[id];
            if (mutation_mode != "reinsert") {
                for (uint64_t d = 0; d < dim; ++d) {
                    replacement[d] = 0.875F * base[id][d] + 0.125F * original[donor][d];
                }
                require(replacement != base[id], "replacement did not change vector");
            }
            if (mutation_mode == "all" or mutation_mode == "update") {
                measure(samples, round, static_cast<int64_t>(id), "update", [&] {
                    require(static_cast<bool>(
                                index->Update(static_cast<int64_t>(id), replacement.data(), dim)),
                            "update failed");
                });
                base[id] = replacement;
                mixed_query(round);
            }
            if (mutation_mode != "update") {
                measure(samples, round, static_cast<int64_t>(id), "remove", [&] {
                    require(index->Remove(static_cast<int64_t>(id)), "remove failed");
                });
                live[id] = false;
                require(index->Size() + 1 == base.size(), "remove count mismatch");
                mixed_query(round);
                measure(samples, round, static_cast<int64_t>(id), "add", [&] {
                    require(static_cast<bool>(
                                index->Add(static_cast<int64_t>(id), replacement.data(), dim)),
                            "readd failed");
                });
                base[id] = replacement;
                live[id] = true;
                require(index->Size() == base.size(), "readd count mismatch");
                mixed_query(round);
            }
        }
        if (checkpoint_every > 0 and (round % checkpoint_every == 0 or round == rounds)) {
            checkpoint(output,
                       round,
                       cycles,
                       base,
                       queries,
                       *index,
                       format,
                       calls_per_cycle,
                       round == rounds and mutation_mode != "all");
        }
        const auto before_persistence_rss = rss_kib();
        const auto actual_threads = status_number("Threads:");
        const auto snapshot = std::filesystem::path(argv[6]) / ("round-" + std::to_string(round));
        require(not std::filesystem::exists(snapshot), "snapshot already exists");
        measure(samples, round, -1, "save", [&] {
            std::ofstream stream(snapshot, std::ios::binary);
            require(static_cast<bool>(stream) and static_cast<bool>(index->Save(stream)),
                    "save failed");
            stream.close();
            require(static_cast<bool>(stream), "save close failed");
        });
        auto loaded = vsag::lite::Index::Create(dim);
        measure(samples, round, -1, "load", [&] {
            std::ifstream stream(snapshot, std::ios::binary);
            loaded = vsag::lite::Index::Load(stream);
            require(static_cast<bool>(loaded) and (*loaded)->Size() == base.size(), "load failed");
        });
        for (const auto& query : queries) {
            auto before = index->SearchWithOptions(query.data(), dim, k, options);
            auto after = (*loaded)->SearchWithOptions(query.data(), dim, k, options);
            require(static_cast<bool>(before) and static_cast<bool>(after),
                    "roundtrip query failed");
            require(before->size() == after->size(), "roundtrip count mismatch");
            for (uint64_t rank = 0; rank < before->size(); ++rank) {
                require((*before)[rank].id == (*after)[rank].id and
                            (*before)[rank].distance == (*after)[rank].distance,
                        "roundtrip result changed");
            }
        }
        if (resident) {
            // Only the comparison copy is destroyed: the original graph/allocator remains resident.
            loaded->reset();
        } else {
            index = std::move(*loaded);
        }
        states << round << ',' << index->Size() << ',' << rss_kib() << ',' << usage().ru_maxrss
               << ',' << std::filesystem::file_size(snapshot) << ',' << before_persistence_rss
               << ',' << actual_threads << '\n';
    }
    quality_check(rounds);
    std::ofstream timings(output / "samples.csv");
    timings << "round,id,operation,wall_us,user_us,system_us,voluntary,"
               "involuntary,peak_kib\n"
            << std::setprecision(12);
    for (const auto& sample : samples) {
        timings << sample.round << ',' << sample.id << ',' << sample.operation << ','
                << sample.wall_us << ',' << sample.user_us << ',' << sample.system_us << ','
                << sample.voluntary << ',' << sample.involuntary << ',' << sample.peak_kib << '\n';
    }
    require(static_cast<bool>(quality) and static_cast<bool>(states) and static_cast<bool>(timings),
            "evidence write failed");
    std::cout << "PASS storage=" << storage << " count=" << base.size() << " dim=" << dim
              << " rounds=" << rounds << " cycles=" << cycles << " queries=" << queries.size()
              << " mutation_calls=" << rounds * cycles * calls_per_cycle
              << " mixed_queries=" << query_number << " resident=" << resident << '\n';
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        run(argc, argv);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
