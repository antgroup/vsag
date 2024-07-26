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

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include "vsag/allocator.h"
#include "vsag/logger.h"

namespace vsag {

class Options {
public:
    static Options&
    Instance();

public:
    // Gets the sector size with memory order acquire for thread safety
    inline size_t
    sector_size() const {
        return sector_size_.load(std::memory_order_acquire);
    }

    inline void
    set_sector_size(size_t size) {
        sector_size_.store(size, std::memory_order_release);
    }

    // Gets the limit of block size with memory order acquire for thread safety
    inline size_t
    block_size_limit() const {
        return block_size_limit_.load(std::memory_order_acquire);
    }

    inline void
    set_block_size_limit(size_t size) {
        block_size_limit_.store(size, std::memory_order_release);
    }

    Logger*
    logger();

    inline bool
    set_logger(Logger* logger) {
        logger_ = logger;
        return true;
    }

private:
    Options() = default;
    ~Options() = default;

    // Deleted copy constructor and assignment operator to prevent copies
    Options(const Options&) = delete;
    Options(const Options&&) = delete;
    Options&
    operator=(const Options&) = delete;

private:
    // In a single query, the space size used to store disk vectors.
    std::atomic<size_t> sector_size_{512};

    // The size of the maximum memory allocated each time (default is 128MB)
    std::atomic<size_t> block_size_limit_{128 * 1024 * 1024};

    Logger* logger_ = nullptr;
};
using Option = Options;  // for compatibility

}  // namespace vsag
