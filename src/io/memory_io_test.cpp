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

#include <cstdlib>
#include <memory>

#include "basic_io_test.h"
#include "impl/allocator/safe_allocator.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("MemoryIO Read and Write", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto io = std::make_unique<MemoryIO>(allocator.get());
    TestBasicReadWrite(*io);
}

TEST_CASE("MemoryIO Serialize and Deserialize", "[ut][MemoryIO]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto wio = std::make_unique<MemoryIO>(allocator.get());
    auto rio = std::make_unique<MemoryIO>(allocator.get());
    TestSerializeAndDeserialize(*wio, *rio);
}

class FailAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "FailAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        if (fail_on_allocate_) {
            return nullptr;
        }
        return std::malloc(size);
    }

    void
    Deallocate(void* p) override {
        std::free(p);
    }

    void*
    Reallocate(void* p, uint64_t size) override {
        if (fail_on_reallocate_) {
            return nullptr;
        }
        return std::realloc(p, size);
    }

    void
    SetFailOnReallocate(bool fail) {
        fail_on_reallocate_ = fail;
    }
    void
    SetFailOnAllocate(bool fail) {
        fail_on_allocate_ = fail;
    }

private:
    bool fail_on_allocate_ = false;
    bool fail_on_reallocate_ = false;
};

TEST_CASE("MemoryIO Allocation Failure", "[ut][MemoryIO]") {
    FailAllocator fail_allocator;
    fail_allocator.SetFailOnAllocate(true);

    auto io = std::make_unique<MemoryIO>(&fail_allocator);

    REQUIRE_THROWS_AS(io->Write(reinterpret_cast<const uint8_t*>("test"), 4, 100), VsagException);
}

TEST_CASE("MemoryIO Allocation Failure Error Type", "[ut][MemoryIO]") {
    FailAllocator fail_allocator;
    fail_allocator.SetFailOnAllocate(true);

    auto io = std::make_unique<MemoryIO>(&fail_allocator);

    try {
        io->Write(reinterpret_cast<const uint8_t*>("test"), 4, 100);
        FAIL("Expected VsagException to be thrown");
    } catch (const VsagException& e) {
        REQUIRE(e.error_.type == ErrorType::NO_ENOUGH_MEMORY);
    }
}

TEST_CASE("MemoryIO Reallocate Failure", "[ut][MemoryIO]") {
    FailAllocator fail_allocator;
    // Do not fail on initial allocate
    fail_allocator.SetFailOnAllocate(false);
    fail_allocator.SetFailOnReallocate(true);

    auto io = std::make_unique<MemoryIO>(&fail_allocator);

    // Write data to populate the buffer (triggers initial allocation)
    const char* test_data = "initial data";
    io->Write(reinterpret_cast<const uint8_t*>(test_data), strlen(test_data), 0);

    // Trigger a growth that would require reallocation
    // The reallocate will fail, but original buffer should remain valid
    REQUIRE_THROWS_AS(
        io->Write(reinterpret_cast<const uint8_t*>("more data"), strlen("more data"), 1000),
        VsagException);

    // Verify the original buffer contents are still readable after the failure
    std::vector<uint8_t> read_buffer(strlen(test_data));
    REQUIRE(io->Read(strlen(test_data), 0, read_buffer.data()));
    REQUIRE(std::string(reinterpret_cast<char*>(read_buffer.data()), strlen(test_data)) ==
            test_data);
}

TEST_CASE("MemoryIO Reallocate Failure Error Type", "[ut][MemoryIO]") {
    FailAllocator fail_allocator;
    fail_allocator.SetFailOnAllocate(false);
    fail_allocator.SetFailOnReallocate(true);

    auto io = std::make_unique<MemoryIO>(&fail_allocator);

    io->Write(reinterpret_cast<const uint8_t*>("initial"), 7, 0);

    try {
        io->Write(reinterpret_cast<const uint8_t*>("large"), 5, 1000);
        FAIL("Expected VsagException to be thrown");
    } catch (const VsagException& e) {
        REQUIRE(e.error_.type == ErrorType::NO_ENOUGH_MEMORY);
    }
}
