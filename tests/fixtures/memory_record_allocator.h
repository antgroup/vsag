
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

#pragma once

#include <memory>
#include <random>
#include <unordered_map>

#include "vsag/allocator.h"

namespace fixtures {

class MemoryRecordAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "memory_record_allocator";
    }
    MemoryRecordAllocator() : memory_bytes_(0) {
    }

    void*
    Allocate(size_t size) override {
        auto ptr = malloc(size);
        {
            std::lock_guard lock(mutex_);
            records_[ptr] = size;
            memory_bytes_ += size;
            memory_peak_ = std::max(memory_peak_, memory_bytes_);
        }
        return ptr;
    }

    void
    Deallocate(void* p) override {
        {
            std::lock_guard lock(mutex_);
            memory_bytes_ -= records_[p];
        }
        return free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        {
            std::lock_guard lock(mutex_);
            memory_bytes_ -= records_[p];
            records_[p] = size;
            memory_bytes_ += size;
            memory_peak_ = std::max(memory_peak_, memory_bytes_);
        }
        return realloc(p, size);
    }

    uint64_t
    GetMemoryPeak() const {
        return this->memory_peak_;
    }

    uint64_t
    GetCurrentMemory() const {
        return this->memory_bytes_;
    }

private:
    uint64_t memory_bytes_{0};

    uint64_t memory_peak_{0};

    std::unordered_map<void*, uint64_t> records_{};

    std::mutex mutex_{};
};
}  // namespace fixtures
