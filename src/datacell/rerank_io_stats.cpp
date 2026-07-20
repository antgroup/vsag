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

#include "rerank_io_stats.h"

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>

#include "vsag/options.h"

namespace vsag {

namespace {

class RerankIOStats {
public:
    RerankIOStats() {
        const char* path = std::getenv("VSAG_RERANK_IO_STATS_PATH");
        if (path != nullptr) {
            path_ = path;
        }
    }

    ~RerankIOStats() {
        if (path_.empty()) {
            return;
        }
        std::ofstream output(path_, std::ios::trunc);
        output << "calls=" << calls_.load() << '\n';
        output << "requests=" << requests_.load() << '\n';
        output << "payload_bytes=" << payload_bytes_.load() << '\n';
        output << "read_bytes=" << read_bytes_.load() << '\n';
    }

    void
    Record(uint64_t payload_bytes, const uint64_t* sizes, const uint64_t* offsets, uint64_t count) {
        if (path_.empty()) {
            return;
        }
        const uint64_t align_bit = Options::Instance().direct_IO_object_align_bit();
        const uint64_t align_size = 1ULL << align_bit;
        const uint64_t align_mask = align_size - 1;
        uint64_t read_bytes = 0;
        for (uint64_t i = 0; i < count; ++i) {
            const uint64_t inner_offset = offsets[i] & align_mask;
            read_bytes += (sizes[i] + inner_offset + align_mask) & ~align_mask;
        }
        calls_.fetch_add(1, std::memory_order_relaxed);
        requests_.fetch_add(count, std::memory_order_relaxed);
        payload_bytes_.fetch_add(payload_bytes, std::memory_order_relaxed);
        read_bytes_.fetch_add(read_bytes, std::memory_order_relaxed);
    }

private:
    std::string path_{};
    std::atomic<uint64_t> calls_{0};
    std::atomic<uint64_t> requests_{0};
    std::atomic<uint64_t> payload_bytes_{0};
    std::atomic<uint64_t> read_bytes_{0};
};

RerankIOStats&
GetRerankIOStats() {
    static RerankIOStats stats;
    return stats;
}

}  // namespace

void
RecordRerankIOStats(uint64_t payload_bytes,
                    const uint64_t* sizes,
                    const uint64_t* offsets,
                    uint64_t count) {
    GetRerankIOStats().Record(payload_bytes, sizes, offsets, count);
}

}  // namespace vsag
