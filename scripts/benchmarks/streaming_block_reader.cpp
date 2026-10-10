// SPDX-License-Identifier: Apache-2.0
// Copyright 2026-present the vsag project
// Linux-only adapter microbenchmark; see README.md for scope and invocation.
#include <sys/resource.h>

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>

#include "storage/serialization.h"
#include "storage/tlv_section.h"
#include "vsag/options.h"

int
main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    constexpr uint64_t size = 129ULL * 1024 * 1024;
    vsag::Options::Instance().set_block_size_limit(128ULL * 1024 * 1024);
    std::array<char, 8192> chunk{};
    uint32_t crc = vsag::StreamHeader::InitialChecksum();
    for (uint64_t offset = 0; offset < size; offset += chunk.size()) {
        crc = vsag::StreamHeader::UpdateChecksum(crc, {chunk.data(), chunk.size()});
    }
    vsag::StreamBlockHeader header;
    header.value_len = size;
    header.payload_checksum = vsag::StreamHeader::FinalizeChecksum(crc);
    uint64_t source_bytes = 0;
    vsag::ReadFuncStreamReader source(
        [&](uint64_t, uint64_t count, void* destination) {
            std::memset(destination, 0, count);
            source_bytes += count;
        },
        0,
        size);
    auto consume = [&](vsag::StreamReader& block) {
        for (uint64_t offset = 0; offset < size; offset += chunk.size()) {
            block.Read(chunk.data(), chunk.size());
        }
    };
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    const auto start = std::chrono::steady_clock::now();
    if (std::string(argv[1]) == "legacy") {
        vsag::ReadSeekableBlockPayload(source, header, consume);
    } else {
        vsag::ReadForwardBlockPayload(source, header, consume);
    }
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
    getrusage(RUSAGE_SELF, &after);
    std::cout << argv[1] << " seconds=" << seconds.count()
              << " rss_peak_delta_kib=" << after.ru_maxrss - before.ru_maxrss
              << " block_output_bytes=" << (after.ru_oublock - before.ru_oublock) * 512
              << " source_bytes=" << source_bytes << '\n';
}
