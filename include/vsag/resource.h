
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

#include "allocator.h"
#include "thread_pool.h"

namespace vsag {
class Resource {
public:
    explicit Resource();

    explicit Resource(Allocator* allocator, ThreadPool* thread_pool);

    virtual ~Resource() = default;

    virtual std::shared_ptr<Allocator>
    GetAllocator() {
        return this->allocator;
    }

    virtual std::shared_ptr<ThreadPool>
    GetThreadPool() {
        return this->thread_pool;
    }

public:
    std::shared_ptr<Allocator> allocator;
    std::shared_ptr<ThreadPool> thread_pool;
};
}  // namespace vsag
