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

/**
 * @file random_allocator.h
 * @brief Allocator that randomly fails allocations for testing error handling.
 */

#pragma once

#include <memory>
#include <random>

#include "vsag/allocator.h"

namespace fixtures {

/**
 * @class RandomAllocator
 * @brief Allocator that randomly returns nullptr on allocate/reallocate operations.
 * Used for testing error handling and resilience to allocation failures.
 * Default error ratio is 2.5%.
 */
class RandomAllocator : public vsag::Allocator {
public:
    /**
     * @brief Returns the name of this allocator.
     * @return String identifier "RandomAllocator".
     */
    std::string
    Name() override;

    /**
     * @brief Constructs a new RandomAllocator with default error ratio.
     */
    RandomAllocator();

    /**
     * @brief Attempts allocation, may randomly return nullptr.
     * @param size The number of bytes to allocate.
     * @return Pointer to allocated memory or nullptr (randomly).
     */
    void*
    Allocate(uint64_t size) override;

    /**
     * @brief Deallocates memory (always succeeds).
     * @param p Pointer to the memory to deallocate.
     */
    void
    Deallocate(void* p) override;

    /**
     * @brief Attempts reallocation, may randomly return nullptr.
     * @param p Pointer to existing memory.
     * @param size The new size in bytes.
     * @return Pointer to reallocated memory or nullptr (randomly).
     */
    void*
    Reallocate(void* p, uint64_t size) override;

private:
    std::shared_ptr<std::random_device> rd_;  // Random device for seeding.
    std::shared_ptr<std::mt19937> gen_;       // Mersenne twister generator.
    std::uniform_real_distribution<> dis_;    // Uniform distribution for randomness.
    float error_ratio_ = 0.0f;                // Probability of allocation failure.
};

}  // namespace fixtures
