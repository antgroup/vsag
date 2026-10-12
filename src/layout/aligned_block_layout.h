// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include "io/memory_block_io/memory_block_io.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/options.h"

namespace vsag {

/**
 * Cache-line-aligned records backed by MemoryBlockIO. Each record stays within one
 * physical block; growing the directory never moves existing records. Callers must
 * exclude reads/writes while Reserve changes the directory. Disjoint records may
 * be written concurrently after Reserve completes.
 *
 * Wire records are packed in ID order, without physical block alignment padding.
 */
class AlignedBlockLayout {
public:
    explicit AlignedBlockLayout(uint64_t record_size,
                                Allocator* allocator,
                                uint64_t block_size_limit = std::min<uint64_t>(
                                    2 * 1024 * 1024, Options::Instance().block_size_limit()))
        : record_size_(record_size),
          block_size_(ChooseBlockSize(record_size, block_size_limit)),
          records_per_block_((block_size_ - (K_ALIGNMENT - 1)) / record_size),
          io_(block_size_, allocator),
          blocks_(allocator) {
    }

    void
    Reserve(uint64_t count) {
        const uint64_t required = count == 0 ? 0 : (count - 1) / records_per_block_ + 1;
        CHECK_ARGUMENT(required <= std::numeric_limits<uint64_t>::max() / block_size_,
                       "aligned block layout capacity overflow");
        if (required <= blocks_.size()) {
            return;
        }
        // Reserve directory space first so publishing pointers below cannot allocate.
        blocks_.reserve(required);
        io_.Resize(required * block_size_);
        while (blocks_.size() < required) {
            auto lease = io_.Acquire(blocks_.size() * block_size_, block_size_);
            // A whole, block-aligned acquisition from this privately owned MemoryBlockIO
            // always borrows its mutable allocation (NoCache), never a temporary copy.
            auto* raw = const_cast<uint8_t*>(lease.Data());
            const auto address = reinterpret_cast<uintptr_t>(raw);
            const auto padding = (K_ALIGNMENT - address % K_ALIGNMENT) % K_ALIGNMENT;
            blocks_.push_back(raw + padding);
        }
    }

    [[nodiscard]] const uint8_t*
    GetRecord(uint64_t id) const {
        return blocks_[id / records_per_block_] + (id % records_per_block_) * record_size_;
    }

    [[nodiscard]] uint8_t*
    GetMutableRecord(uint64_t id) {
        return blocks_[id / records_per_block_] + (id % records_per_block_) * record_size_;
    }

    [[nodiscard]] uint64_t
    RecordsPerBlock() const {
        return records_per_block_;
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        return sizeof(*this) + io_.GetMemoryUsage() + blocks_.capacity() * sizeof(uint8_t*);
    }

    void
    WriteRecords(StreamWriter& writer, uint64_t count) const {
        for (uint64_t id = 0; id < count;) {
            const auto records = std::min(records_per_block_, count - id);
            writer.Write(reinterpret_cast<const char*>(GetRecord(id)), records * record_size_);
            id += records;
        }
    }

    void
    ReadRecords(StreamReader& reader, uint64_t count) {
        for (uint64_t id = 0; id < count;) {
            const auto records = std::min(records_per_block_, count - id);
            reader.Read(reinterpret_cast<char*>(GetMutableRecord(id)), records * record_size_);
            id += records;
        }
    }

private:
    static constexpr uint64_t K_ALIGNMENT = 64;

    static uint64_t
    ChooseBlockSize(uint64_t record_size, uint64_t limit) {
        CHECK_ARGUMENT(record_size > 0 and record_size % K_ALIGNMENT == 0,
                       "aligned block layout requires a cache-line-sized record stride");
        CHECK_ARGUMENT(record_size <= (uint64_t{1} << 63) - (K_ALIGNMENT - 1),
                       "aligned block layout record size overflow");
        CHECK_ARGUMENT(limit > 0, "aligned block layout block size must be positive");
        auto size = MemoryBlockIOParameter::NearestPowerOfTwo(limit);
        // An unusually large record must still fit contiguously with alignment slack.
        while (size < record_size + K_ALIGNMENT - 1) {
            size *= 2;
        }
        return size;
    }

    const uint64_t record_size_;
    const uint64_t block_size_;
    const uint64_t records_per_block_;
    MemoryBlockIO io_;
    Vector<uint8_t*> blocks_;
};

}  // namespace vsag
