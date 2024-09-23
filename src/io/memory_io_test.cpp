
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

#include "memory_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "default_allocator.h"
#include "fixtures.h"
using namespace vsag;

template <typename T>
void
TestReadWrite(BasicIO<T>* basicIo) {
    int dim = 32;
    auto vector = fixtures::generate_vectors(100, dim);
    auto code_size = dim * sizeof(float);
    std::unordered_map<uint64_t, float*> maps;
    for (int i = 0; i < 100; ++i) {
        auto offset = random() % 10000000 * code_size;
        basicIo->Write((uint8_t*)(vector.data() + i * dim), code_size, offset);
        maps[offset] = vector.data() + i * dim;
    }

    for (auto& iter : maps) {
        const auto* result = (const float*)(basicIo->Read(code_size, iter.first));
        auto* gt = iter.second;
        for (int i = 0; i < dim; ++i) {
            REQUIRE(result[i] == gt[i]);
        }
    }
}

TEST_CASE("read&write[ut][memory_io]") {
    auto allocator = std::make_unique<DefaultAllocator>();
    auto io = std::make_unique<MemoryIO>(allocator.get());
    TestReadWrite(io.get());
}
