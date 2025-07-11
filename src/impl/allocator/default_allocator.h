
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

#include <mutex>
#include <unordered_set>

#include "vsag/allocator.h"

namespace vsag {

class DefaultAllocator : public Allocator {
public:
    DefaultAllocator() = default;
    ~DefaultAllocator() override;

    DefaultAllocator(const DefaultAllocator&) = delete;
    DefaultAllocator(DefaultAllocator&&) = delete;

public:
    std::string
    Name() override;

    void*
    Allocate(size_t size) override;

    void
    Deallocate(void* p) override;

    void*
    Reallocate(void* p, size_t size) override;

private:
#ifndef NDEBUG
    std::unordered_set<void*> allocated_ptrs_;
    std::mutex set_mutex_;
#endif
};

}  // namespace vsag
