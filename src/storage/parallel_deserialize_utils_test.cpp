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

#include "parallel_deserialize_utils.h"

#include <atomic>
#include <functional>
#include <istream>
#include <memory>
#include <stdexcept>

#include "impl/thread_pool/default_thread_pool.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "unittest.h"
#include "vsag_exception.h"

namespace {

std::shared_ptr<vsag::SafeThreadPool>
MakePool(uint64_t threads) {
    return std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(threads), true);
}

// reader whose ReadDecompressed throws a plain std::runtime_error, mirroring
// the public DeserializeReader default implementation
class RuntimeThrowingReader : public vsag::DeserializeReader {
public:
    [[nodiscard]] uint64_t
    Size() const override {
        return 0;
    }

    void
    Read(uint64_t /*offset*/, uint64_t /*len*/, void* /*dest*/) override {
    }

    void
    ReadDecompressed(uint64_t /*offset*/,
                     uint64_t /*compressed_size*/,
                     const std::function<void(std::istream&)>& /*consume*/) override {
        throw std::runtime_error("reader does not support compressed frames");
    }
};

// reader whose ReadDecompressed throws a typed VsagException, which must
// propagate untouched
class VsagThrowingReader : public vsag::DeserializeReader {
public:
    [[nodiscard]] uint64_t
    Size() const override {
        return 0;
    }

    void
    Read(uint64_t /*offset*/, uint64_t /*len*/, void* /*dest*/) override {
    }

    void
    ReadDecompressed(uint64_t /*offset*/,
                     uint64_t /*compressed_size*/,
                     const std::function<void(std::istream&)>& /*consume*/) override {
        throw vsag::VsagException(vsag::ErrorType::INVALID_BINARY, "frame checksum mismatch");
    }
};

}  // namespace

TEST_CASE("read_decompressed_checked Normalizes Foreign Exceptions",
          "[ut][parallel_deserialize_utils]") {
    auto consume = [](std::istream&) {};

    SECTION("std::runtime_error becomes a VsagException") {
        RuntimeThrowingReader reader;
        REQUIRE_THROWS_AS(vsag::read_decompressed_checked(reader, 0, 16, consume),
                          vsag::VsagException);
    }

    SECTION("VsagException passes through with its error type intact") {
        VsagThrowingReader reader;
        try {
            vsag::read_decompressed_checked(reader, 0, 16, consume);
            FAIL("expected the reader exception to propagate");
        } catch (const vsag::VsagException& e) {
            REQUIRE(e.error_.type == vsag::ErrorType::INVALID_BINARY);
        }
    }
}

TEST_CASE("TaskBatch Run Joins And Rethrows First Failure", "[ut][parallel_deserialize_utils]") {
    auto pool = MakePool(2);

    SECTION("all tasks succeed") {
        std::atomic<int> counter{0};
        {
            vsag::TaskBatch batch(*pool, 4);
            for (int i = 0; i < 4; ++i) {
                batch.Submit([&counter]() { counter.fetch_add(1); });
            }
            batch.Run();
        }
        REQUIRE(counter.load() == 4);
    }

    SECTION("a failing task surfaces through Run") {
        vsag::TaskBatch batch(*pool, 2);
        batch.Submit(
            []() { throw vsag::VsagException(vsag::ErrorType::INTERNAL_ERROR, "task failed"); });
        batch.Submit([]() {
            throw vsag::VsagException(vsag::ErrorType::INVALID_ARGUMENT, "second failure");
        });
        // the first recorded failure wins; SafeThreadPool swallows task
        // exceptions in its futures, so the batch recording is what surfaces
        REQUIRE_THROWS_AS(batch.Run(), vsag::VsagException);
    }
}

TEST_CASE("TaskBatch Destructor Surfaces Failures When Run Is Skipped",
          "[ut][parallel_deserialize_utils]") {
    auto pool = MakePool(2);

    SECTION("skipped Run with a failing task rethrows from the destructor") {
        bool threw = false;
        try {
            vsag::TaskBatch batch(*pool, 1);
            batch.Submit([]() {
                throw vsag::VsagException(vsag::ErrorType::INTERNAL_ERROR, "task failed");
            });
            // no Run(): the destructor must join and rethrow instead of
            // silently dropping the recorded failure
        } catch (const vsag::VsagException& e) {
            threw = e.error_.type == vsag::ErrorType::INTERNAL_ERROR;
        }
        REQUIRE(threw);
    }

    SECTION("skipped Run with healthy tasks stays silent") {
        std::atomic<int> counter{0};
        {
            vsag::TaskBatch batch(*pool, 2);
            for (int i = 0; i < 2; ++i) {
                batch.Submit([&counter]() { counter.fetch_add(1); });
            }
            // destructor joins; nothing to rethrow
        }
        REQUIRE(counter.load() == 2);
    }

    SECTION("Run before destruction is not rethrown twice") {
        bool caught_once = false;
        try {
            vsag::TaskBatch batch(*pool, 1);
            batch.Submit([]() {
                throw vsag::VsagException(vsag::ErrorType::INTERNAL_ERROR, "task failed");
            });
            try {
                batch.Run();
            } catch (const vsag::VsagException&) {
                caught_once = true;
            }
            // batch destructs here with ran_ set: the destructor must not
            // rethrow the already surfaced failure a second time
        } catch (...) {
            FAIL("destructor rethrew after Run had already surfaced the failure");
        }
        REQUIRE(caught_once);
    }
}
