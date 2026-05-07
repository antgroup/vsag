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

RandomAllocator::RandomAllocator() {
    rd_ = std::make_shared<std::random_device>();
    gen_ = std::make_shared<std::mt19937>((*rd_)());
    std::uniform_int_distribution<int> seed_random;
    int seed = seed_random(*gen_);
    gen_->seed(seed);
    error_ratio_ = 0.025f;
}

void*
RandomAllocator::Allocate(uint64_t size) {
    auto number = dis_(*gen_);
    if (number < error_ratio_) {
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
    auto number = dis_(*gen_);
    if (number < error_ratio_) {
        return nullptr;
    }
    return realloc(p, size);
}

}  // namespace fixtures
