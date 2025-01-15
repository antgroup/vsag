
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

#include "vsag/allocator.h"

namespace fixtures {

class RandomAllocator : public vsag::Allocator {
public:
    std::string
    Name() override {
        return "random_allocator";
    }
    RandomAllocator() {
        rd_ = std::make_shared<std::random_device>();
        gen_ = std::make_shared<std::mt19937>((*rd_)());
        std::uniform_int_distribution<int> seed_random;
        int seed = seed_random(*gen_);
        gen_->seed(seed);
        error_ratio_ = 0.025f;
    }

    void*
    Allocate(size_t size) override {
        auto number = dis_(*gen_);
        if (number < error_ratio_) {
            return nullptr;
        }
        return malloc(size);
    }

    void
    Deallocate(void* p) override {
        return free(p);
    }

    void*
    Reallocate(void* p, size_t size) override {
        auto number = dis_(*gen_);
        if (number < error_ratio_) {
            return nullptr;
        }
        return realloc(p, size);
    }

private:
    std::shared_ptr<std::random_device> rd_;
    std::shared_ptr<std::mt19937> gen_;
    std::uniform_real_distribution<> dis_;
    float error_ratio_ = 0.0f;
};
}  // namespace fixtures
