
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

#include "io_array.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "async_io.h"
#include "basic_io_test.h"
#include "buffer_io.h"
#include "impl/allocator/safe_allocator.h"
#include "memory_block_io.h"
#include "memory_io.h"
#include "mmap_io.h"
#include "noncontinuous_io.h"

namespace vsag {
template <typename IOType>
class IOArrayTest {
public:
    template <typename... Args>
    IOArrayTest(Allocator* allocator, Args&&... args)
        : array_(std::make_unique<IOArray<IOType>>(allocator, std::forward<Args>(args)...)),
          array2_(std::make_unique<IOArray<IOType>>(allocator, std::forward<Args>(args)...)) {
    }

    void
    TestBasic() {
        this->array_->Resize(5);
        for (size_t i = 0; i < 5; ++i) {
            TestBasicReadWrite((*this->array_)[i]);
        }
        for (size_t i = 0; i < 5; ++i) {
            TestBasicReadWrite(this->array_->At(i));
        }
        this->array_->Resize(10);
        for (size_t i = 5; i < 10; ++i) {
            TestBasicReadWrite(this->array_->At(i));
        }
        this->array2_->Resize(10);
        for (size_t i = 0; i < 10; ++i) {
            TestSerializeAndDeserialize((*this->array_)[i], (*this->array2_)[i]);
        }
    }

    std::unique_ptr<IOArray<IOType>> array_{nullptr};
    std::unique_ptr<IOArray<IOType>> array2_{nullptr};
};
}  // namespace vsag

using namespace vsag;

TEST_CASE("IOArrayTest MemoryIO Basic Test", "[IOArray][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IOArrayTest<MemoryIO> test(allocator.get(), allocator.get());
    test.TestBasic();
}

TEST_CASE("IOArrayTest MemoryBlockIO Basic Test", "[IOArray][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    IOArrayTest<MemoryBlockIO> test(allocator.get(), static_cast<uint64_t>(4096), allocator.get());
    test.TestBasic();
}

TEST_CASE("IOArrayTest BufferIO Basic Test", "[IOArray][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dir = fixtures::TempDir("test_buffer_io");
    auto path = dir.GenerateRandomFile();
    IOArrayTest<NonContinuousIO<BufferIO>> test(allocator.get(), path, allocator.get());
    test.TestBasic();
}

TEST_CASE("IOArrayTest AsyncIO Basic Test", "[IOArray][ut]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto dir = fixtures::TempDir("test_async_io");
    auto path = dir.GenerateRandomFile();
    IOArrayTest<NonContinuousIO<AsyncIO>> test(allocator.get(), path, allocator.get());
    test.TestBasic();
}
