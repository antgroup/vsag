// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "vsag/lite/persistent_index.h"

using vsag::lite::Index;
using vsag::lite::PersistentIndex;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr uint64_t QUERIES = 100;
using Row = std::vector<float>;

void
require(bool ok, const char* operation) {
    if (!ok) {
        throw std::runtime_error(operation);
    }
}

Row
read_row(std::ifstream& input) {
    uint32_t dimension{};
    require(bool(input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension))),
            "read dimension");
    require(dimension != 0 and dimension <= 65536, "invalid fvecs dimension");
    Row row(dimension);
    require(bool(input.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float))),
            "read vector");
    return row;
}

std::unique_ptr<Index>
load_index(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    auto index = Index::Load(input);
    require(bool(index), "load snapshot");
    return std::move(*index);
}

void
sync_path(const fs::path& path, int flags) {
    const int fd = ::open(path.c_str(), flags | O_CLOEXEC);
    require(fd >= 0, "open path for sync");
    int result;
    do {
        result = ::fsync(fd);
    } while (result < 0 and errno == EINTR);
    ::close(fd);
    require(result == 0, "fsync");
}

template <typename IndexType>
void
save(const IndexType& index, const fs::path& path) {
    std::ofstream output(path, std::ios::binary);
    require(bool(index.Save(output)), "save snapshot");
    output.close();
    require(bool(output), "close snapshot");
}

void
replace_snapshot(const Index& index, const fs::path& path) {
    const fs::path temporary = path.string() + ".tmp";
    save(index, temporary);
    sync_path(temporary, O_RDWR);
    fs::rename(temporary, path);
    sync_path(path.parent_path(), O_RDONLY | O_DIRECTORY);
}

double
elapsed_us(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

void
seed(const fs::path& input_path,
     uint64_t count,
     const fs::path& snapshot_path,
     const std::string& mode) {
    require(count != 0, "empty benchmark seed");
    std::ifstream input(input_path, std::ios::binary);
    auto vector = read_row(input);
    auto index = Index::Create(vector.size()).value();
    for (uint64_t id = 0; id < count; ++id) {
        require(bool(index->Add(static_cast<int64_t>(id), vector.data(), vector.size())),
                "seed add");
        if (id + 1 < count) {
            vector = read_row(input);
        }
    }
    if (mode != "bruteforce") {
        using vsag::lite::VectorStorage;
        require(mode == "fp32" or mode == "fp16" or mode == "rabitq8", "unknown storage mode");
        const auto storage = mode == "fp32"   ? VectorStorage::FP32
                             : mode == "fp16" ? VectorStorage::FP16
                                              : VectorStorage::RABITQ8;
        require(bool(index->BuildGraph(storage, 16, 128)), "seed graph");
    }
    save(*index, snapshot_path);
}

void
write_adds(const std::string& variant,
           const fs::path& seed_path,
           const fs::path& output,
           const fs::path& input_path,
           uint64_t adds) {
    const bool journal = variant == "journal";
    const bool rewrite = variant == "snapshot";
    require(journal or rewrite or variant == "memory", "unknown write variant");
    auto plain = load_index(seed_path);
    const auto initial = plain->Size();
    std::unique_ptr<PersistentIndex> persistent;
    const auto snapshot_path = output / "state.vsag";
    if (journal) {
        persistent = PersistentIndex::Create((output / "store").string(), std::move(plain)).value();
    } else if (rewrite) {
        replace_snapshot(*plain, snapshot_path);
    }

    std::ifstream input(input_path, std::ios::binary);
    std::ofstream samples(output / "adds.csv");
    samples << "id,wall_us,bytes\n" << std::setprecision(12);
    uint64_t previous = journal ? fs::file_size(output / "store/add.log") : 0;
    for (uint64_t row = 0; row < adds; ++row) {
        auto vector = read_row(input);
        vector[0] += 0.25F;
        const auto id = static_cast<int64_t>(initial + row);
        const auto start = Clock::now();
        if (journal) {
            require(bool(persistent->Add(id, vector.data(), vector.size())), "persistent add");
        } else {
            require(bool(plain->Add(id, vector.data(), vector.size())), "ordinary add");
            if (rewrite) {
                replace_snapshot(*plain, snapshot_path);
            }
        }
        const double latency = elapsed_us(start);
        uint64_t bytes = 0;
        if (journal) {
            const auto size = fs::file_size(output / "store/add.log");
            bytes = size - previous;
            previous = size;
        } else if (rewrite) {
            bytes = fs::file_size(snapshot_path);
        }
        samples << id << ',' << latency << ',' << bytes << '\n';
    }
    if (variant == "memory") {
        save(*plain, snapshot_path);
    }
}

void
read_back(bool journal, const fs::path& output, const fs::path& query_path) {
    std::unique_ptr<Index> plain;
    std::unique_ptr<PersistentIndex> persistent;
    const auto start = Clock::now();
    if (journal) {
        persistent = PersistentIndex::Open((output / "store").string()).value();
    } else {
        plain = load_index(output / "state.vsag");
    }
    const double load_us = elapsed_us(start);
    const uint64_t count = journal ? persistent->Size().value() : plain->Size();

    std::ifstream queries(query_path, std::ios::binary);
    std::ofstream results(output / "neighbors.bin", std::ios::binary);
    std::ofstream samples(output / "queries.csv");
    samples << "row,wall_us\n" << std::setprecision(12);
    for (uint64_t row = 0; row < QUERIES; ++row) {
        const auto query = read_row(queries);
        const auto query_start = Clock::now();
        const auto result = journal
                                ? persistent->Search(query.data(), query.size(), 10, {128})
                                : plain->SearchWithOptions(query.data(), query.size(), 10, {128});
        const double latency = elapsed_us(query_start);
        require(bool(result) && result->size() == 10, "search recovered index");
        for (const auto& neighbor : *result) {
            results.write(reinterpret_cast<const char*>(&neighbor.id), sizeof(neighbor.id));
            results.write(reinterpret_cast<const char*>(&neighbor.distance),
                          sizeof(neighbor.distance));
        }
        samples << row << ',' << latency << '\n';
    }
    struct rusage usage{};
    require(::getrusage(RUSAGE_SELF, &usage) == 0, "getrusage");
    std::ofstream metrics(output / "recovery.csv");
    metrics << "count,load_us,peak_rss_kib\n"
            << std::setprecision(12) << count << ',' << load_us << ',' << usage.ru_maxrss << '\n';
    if (journal) {
        save(*persistent, output / "state.vsag");
    }
}

int
main(int argc, char** argv) {
    try {
        const std::string action = argc > 1 ? argv[1] : "";
        if (action == "seed" && argc == 6) {
            seed(argv[2], std::stoull(argv[3]), argv[4], argv[5]);
        } else if (action == "write" && argc == 7) {
            write_adds(argv[2], argv[3], argv[4], argv[5], std::stoull(argv[6]));
        } else if (action == "read" && argc == 5) {
            read_back(std::string(argv[2]) == "journal", argv[3], argv[4]);
        } else {
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
