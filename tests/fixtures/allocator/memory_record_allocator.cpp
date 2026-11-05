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

#include "memory_record_allocator.h"

#include <cstdlib>

namespace fixtures {

std::string
MemoryRecordAllocator::Name() {
    return "memory_record_allocator";
}

MemoryRecordAllocator::MemoryRecordAllocator() : memory_bytes_(0) {
}

void*
MemoryRecordAllocator::Allocate(uint64_t size) {
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
MemoryRecordAllocator::Deallocate(void* p) {
    if (p == nullptr) {
        return;
    }
    {
        std::lock_guard lock(mutex_);
        auto it = records_.find(p);
        if (it != records_.end()) {
            memory_bytes_ -= it->second;
            records_.erase(it);
        }
    }
    free(p);
}

void*
MemoryRecordAllocator::Reallocate(void* p, uint64_t size) {
    void* new_ptr = realloc(p, size);
    if (new_ptr == nullptr && size != 0) {
        return nullptr;
    }
    {
        std::lock_guard lock(mutex_);
        if (p != nullptr) {
            auto it = records_.find(p);
            if (it != records_.end()) {
                memory_bytes_ -= it->second;
                records_.erase(it);
            }
        }
        if (new_ptr != nullptr) {
            records_[new_ptr] = size;
            memory_bytes_ += size;
            memory_peak_ = std::max(memory_peak_, memory_bytes_);
        }
    }
    return new_ptr;
}

uint64_t
MemoryRecordAllocator::GetMemoryPeak() const {
    return memory_peak_;
}

uint64_t
MemoryRecordAllocator::GetCurrentMemory() const {
    return memory_bytes_;
}

}  // namespace fixtures
