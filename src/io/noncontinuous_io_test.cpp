
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

#include "noncontinuous_io.h"

#include <catch2/catch_template_test_macros.hpp>

#include "async_io.h"
#include "basic_io_test.h"
#include "buffer_io.h"
#include "impl/allocator/safe_allocator.h"
#include "mmap_io.h"
namespace vsag {
template <typename IOTmpl>
class NonContinuousIOTest {
public:
    NonContinuousIOTest() = default;
    ~NonContinuousIOTest() = default;

    template <typename... Args>
    NonContinuousIO<IOTmpl>*
    CreateNonContinuousIO(NonContinuousAllocator* non_continuous_allocator,
                          Allocator* allocator,
                          Args&&... args) {
        return new NonContinuousIO<IOTmpl>(
            non_continuous_allocator, allocator, std::forward<Args>(args)...);
    }
};
}  // namespace vsag

using namespace vsag;
template <typename T>
void
NonContinuousIOTestBasic() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        NonContinuousIOTest<T> test;
        auto non_continuous_allocator = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io = test.CreateNonContinuousIO(non_continuous_allocator.get(),
                                             allocator.get(),
                                             "/tmp/test_noncontinuous_io",
                                             allocator.get());
        TestBasicReadWrite(*io);
        delete io;
    }
}

TEST_CASE("NonContinuousIO Basic Test", "[NonContinuousIO][ut]") {
    NonContinuousIOTestBasic<MMapIO>();
    NonContinuousIOTestBasic<BufferIO>();
    NonContinuousIOTestBasic<AsyncIO>();
}

template <typename T>
void
NonContinuousIOTestSerialize() {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    {
        NonContinuousIOTest<T> test;
        auto non_continuous_allocator1 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io1 = test.CreateNonContinuousIO(non_continuous_allocator1.get(),
                                              allocator.get(),
                                              "/tmp/test_noncontinuous_io1",
                                              allocator.get());
        auto non_continuous_allocator2 = std::make_unique<NonContinuousAllocator>(allocator.get());
        auto io2 = test.CreateNonContinuousIO(non_continuous_allocator2.get(),
                                              allocator.get(),
                                              "/tmp/test_noncontinuous_io2",
                                              allocator.get());
        TestSerializeAndDeserialize(*io1, *io2);
        delete io1;
        delete io2;
    }
}

TEST_CASE("NonContinuousIO Serialize Test", "[NonContinuousIO][ut]") {
    NonContinuousIOTestSerialize<MMapIO>();
    NonContinuousIOTestSerialize<BufferIO>();
    NonContinuousIOTestSerialize<AsyncIO>();
}
