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

#include "random_allocator.h"

#include <cstdlib>

namespace fixtures {

std::string
RandomAllocator::Name() {
    return "random_allocator";
}

RandomAllocator::RandomAllocator() : RandomAllocator(std::random_device{}()) {
}

RandomAllocator::RandomAllocator(uint32_t seed) : gen_(seed) {
}

bool
RandomAllocator::ShouldFail() {
    std::lock_guard lock(mutex_);
    return dis_(gen_) < error_ratio_;
}

void*
RandomAllocator::Allocate(uint64_t size) {
    if (ShouldFail()) {
        return nullptr;
    }
    return malloc(size);
}

void
RandomAllocator::Deallocate(void* p) {
    free(p);
}

void*
RandomAllocator::Reallocate(void* p, uint64_t size) {
    if (ShouldFail()) {
        return nullptr;
    }
    return realloc(p, size);
}

}  // namespace fixtures
