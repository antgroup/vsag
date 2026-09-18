// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "io/external_storage_io/external_storage_io.h"

#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "index_common_param.h"
#include "unittest.h"
#include "vsag/factory.h"

namespace {

struct ByteStore {
    std::shared_ptr<vsag::Allocator> allocator = vsag::Engine::CreateDefaultAllocator();
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    uint64_t reads{0};
    uint64_t writes{0};
    uint64_t resizes{0};
    bool fail_writes{false};
    bool fail_resizes{false};
};

vsag::ExternalStorage
make_storage(const std::shared_ptr<ByteStore>& store) {
    return vsag::Factory::CreateExternalStorage(
        [store](uint64_t offset, uint64_t len, void* destination) {
            std::lock_guard<std::mutex> lock(store->mutex);
            if (offset > store->bytes.size() or len > store->bytes.size() - offset) {
                throw std::runtime_error("read out of bounds");
            }
            if (len > 0) {
                std::memcpy(destination, store->bytes.data() + offset, len);
            }
            ++store->reads;
        },
        [store](uint64_t offset, uint64_t len, const void* source) {
            std::lock_guard<std::mutex> lock(store->mutex);
            if (store->fail_writes) {
                throw std::runtime_error("write failed");
            }
            if (offset > UINT64_MAX - len) {
                throw std::runtime_error("write overflow");
            }
            if (offset + len > store->bytes.size()) {
                store->bytes.resize(offset + len);
            }
            if (len > 0) {
                std::memcpy(store->bytes.data() + offset, source, len);
            }
            ++store->writes;
        },
        [store](uint64_t size) {
            std::lock_guard<std::mutex> lock(store->mutex);
            if (store->fail_resizes) {
                throw std::runtime_error("resize failed");
            }
            store->bytes.resize(size);
            ++store->resizes;
        },
        0);
}

std::unique_ptr<vsag::ExternalStorageIO>
make_io(const std::shared_ptr<ByteStore>& store, bool cache = false) {
    auto storage = make_storage(store);
    auto storages = std::make_shared<vsag::ExternalStorageSet>();
    storages->Set("test", storage.reader, storage.writer);
    auto param = std::make_shared<vsag::ExternalStorageIOParameter>();
    param->external_storage = "test";
    param->enable_read_cache_ = cache;
    param->read_cache_total_size_ = 1 << 20;
    vsag::IndexCommonParam common;
    common.allocator_ = store->allocator;
    common.external_storages_ = storages;
    return std::make_unique<vsag::ExternalStorageIO>(param, common);
}

}  // namespace

TEST_CASE("ExternalStorageIO grow overwrite shrink and failures", "[ut][ExternalStorageIO]") {
    auto store = std::make_shared<ByteStore>();
    auto io = make_io(store, true);
    const std::vector<uint8_t> initial{1, 2, 3, 4};
    io->WriteAt(0, initial.data(), initial.size());
    REQUIRE(io->Size() == initial.size());

    std::vector<uint8_t> result(initial.size());
    REQUIRE(io->ReadAt(0, result.size(), result.data()));
    REQUIRE(result == initial);
    const auto cached_reads = store->reads;
    REQUIRE(cached_reads > 0);
    REQUIRE(io->ReadAt(0, result.size(), result.data()));
    REQUIRE(store->reads == cached_reads);

    const std::vector<uint8_t> overwrite{9, 8};
    io->WriteAt(1, overwrite.data(), overwrite.size());
    REQUIRE(io->ReadAt(0, result.size(), result.data()));
    REQUIRE(store->reads > cached_reads);
    REQUIRE(result == std::vector<uint8_t>{1, 9, 8, 4});

    io->ResizeForOverwrite(8);
    REQUIRE(io->Size() == 8);
    REQUIRE(store->bytes.size() == 8);
    io->Shrink(2);
    REQUIRE(io->Size() == 2);
    REQUIRE_FALSE(io->ReadAt(0, result.size(), result.data()));
    REQUIRE(store->writes == 2);
    REQUIRE(store->resizes == 2);

    store->fail_writes = true;
    REQUIRE_THROWS(io->WriteAt(0, initial.data(), initial.size()));
    REQUIRE(io->Size() == 2);
}

TEST_CASE("CallbackStorage validates reads, completes once, and grows empty writes",
          "[ut][ExternalStorageIO]") {
    auto store = std::make_shared<ByteStore>();
    auto storage = make_storage(store);
    REQUIRE_THROWS(storage.reader->Read(0, 1, nullptr));

    uint64_t completions = 0;
    REQUIRE_THROWS(storage.reader->AsyncRead(
        0, 1, nullptr, [&completions](vsag::IOErrorCode code, const std::string&) {
            ++completions;
            REQUIRE(code == vsag::IOErrorCode::IO_ERROR);
            throw std::runtime_error("completion failed");
        }));
    REQUIRE(completions == 1);

    storage.writer->Write(7, 0, nullptr);
    REQUIRE(storage.reader->Size() == 7);
    REQUIRE(store->bytes.size() == 7);
    REQUIRE(store->writes == 0);
    REQUIRE(store->resizes == 1);

    store->fail_resizes = true;
    REQUIRE_THROWS(storage.writer->Write(9, 0, nullptr));
    REQUIRE(storage.reader->Size() == 7);
    REQUIRE(store->bytes.size() == 7);
    REQUIRE_THROWS(storage.writer->Resize(2));
    REQUIRE(storage.reader->Size() == 7);
    REQUIRE(store->bytes.size() == 7);
    store->fail_resizes = false;
    storage.writer->Resize(2);
    REQUIRE(storage.reader->Size() == 2);
    REQUIRE(store->bytes.size() == 2);
}

TEST_CASE("ExternalStorageIO parameter reuse does not leak bound handles",
          "[ut][ExternalStorageIO]") {
    auto first = std::make_shared<ByteStore>();
    auto second = std::make_shared<ByteStore>();
    auto first_storage = make_storage(first);
    auto second_storage = make_storage(second);
    auto parameter = std::make_shared<vsag::ExternalStorageIOParameter>();
    parameter->external_storage = "shared_name";

    vsag::IndexCommonParam first_common;
    first_common.allocator_ = first->allocator;
    auto first_set = std::make_shared<vsag::ExternalStorageSet>();
    first_set->Set("shared_name", first_storage.reader, first_storage.writer);
    first_common.external_storages_ = first_set;

    vsag::IndexCommonParam second_common;
    second_common.allocator_ = second->allocator;
    auto second_set = std::make_shared<vsag::ExternalStorageSet>();
    second_set->Set("shared_name", second_storage.reader, second_storage.writer);
    second_common.external_storages_ = second_set;

    vsag::ExternalStorageIO first_io(parameter, first_common);
    vsag::ExternalStorageIO second_io(parameter, second_common);
    REQUIRE(parameter->reader == nullptr);
    REQUIRE(parameter->writer == nullptr);

    const std::vector<uint8_t> first_bytes{1, 2};
    const std::vector<uint8_t> second_bytes{8, 9, 10};
    first_io.WriteAt(0, first_bytes.data(), first_bytes.size());
    second_io.WriteAt(0, second_bytes.data(), second_bytes.size());
    REQUIRE(first->bytes == first_bytes);
    REQUIRE(second->bytes == second_bytes);
}

TEST_CASE("ExternalStorageSet rejects invalid registrations", "[ut][ExternalStorageIO]") {
    auto store = std::make_shared<ByteStore>();
    auto storage = make_storage(store);
    vsag::ExternalStorageSet storages;

    REQUIRE_THROWS_AS(storages.Set("", storage.reader, storage.writer), std::invalid_argument);
    REQUIRE_THROWS_AS(storages.Set("null_reader", nullptr, storage.writer), std::invalid_argument);
    REQUIRE_THROWS_AS(storages.Set("null_writer", storage.reader, nullptr), std::invalid_argument);
    REQUIRE_NOTHROW(storages.Set("valid", storage.reader, storage.writer));
    REQUIRE_THROWS_AS(storages.Set("valid", storage.reader, storage.writer), std::invalid_argument);
    REQUIRE(storages.Contains("valid"));
    REQUIRE(storage.reader != nullptr);
    REQUIRE(storage.writer != nullptr);
    const auto* retained = storages.Get("valid");
    REQUIRE(retained != nullptr);
    for (uint64_t i = 0; i < 1024; ++i) {
        storages.Set("other_" + std::to_string(i), storage.reader, storage.writer);
    }
    REQUIRE(storages.Get("valid") == retained);
    REQUIRE(retained->reader == storage.reader);
    REQUIRE(retained->writer == storage.writer);
}

TEST_CASE("CallbackStorage completion can reenter after success and failure",
          "[ut][ExternalStorageIO]") {
    auto store = std::make_shared<ByteStore>();
    auto storage = make_storage(store);
    storage.writer->Resize(1);
    uint8_t byte = 0;
    uint64_t completions = 0;
    for (const auto offset : {uint64_t{0}, uint64_t{2}}) {
        storage.reader->AsyncRead(
            offset, 1, &byte, [&](vsag::IOErrorCode code, const std::string&) {
                ++completions;
                REQUIRE(code == (offset == 0 ? vsag::IOErrorCode::IO_SUCCESS
                                             : vsag::IOErrorCode::IO_ERROR));
                REQUIRE(storage.reader->Size() == 1);
                storage.writer->Resize(1);
            });
    }
    REQUIRE(completions == 2);
}

TEST_CASE("ExternalStorageBackend preserves read error channels and skips empty reads",
          "[ut][ExternalStorageIO]") {
    auto allocator = vsag::Engine::CreateDefaultAllocator();
    uint64_t reads = 0;
    auto storage = vsag::Factory::CreateExternalStorage(
        [&](uint64_t, uint64_t, void*) {
            ++reads;
            throw std::runtime_error("read failed");
        },
        [](uint64_t, uint64_t, const void*) {},
        [](uint64_t) {},
        1);
    auto param = std::make_shared<vsag::ExternalStorageIOParameter>();
    param->reader = storage.reader;
    param->writer = storage.writer;
    vsag::ExternalStorageBackend backend(allocator.get());
    REQUIRE(backend.Initialize(param, false, 0, 0) == 1);
    auto strict_param = std::make_shared<vsag::ExternalStorageIOParameter>(*param);
    strict_param->reader = vsag::Factory::CreateReadFuncReader(
        [](uint64_t, uint64_t, void*) { throw std::runtime_error("unexpected read"); }, 1);
    vsag::ExternalStorageBackend strict_backend(allocator.get());
    REQUIRE(strict_backend.Initialize(strict_param, false, 0, 0) == 1);
    REQUIRE(strict_backend.ReadAt(0, 0, nullptr));
    REQUIRE(reads == 0);
    uint8_t byte = 0;
    REQUIRE_THROWS(backend.ReadAt(0, 1, &byte));
    const vsag::ReadRequest request{&byte, 0, 1};
    REQUIRE_THROWS(backend.ReadMany(&request, 1));
    const uint64_t size = 1;
    const uint64_t offset = 0;
    REQUIRE_FALSE(backend.ReadManyContiguous(&byte, &size, &offset, 1));
    REQUIRE(reads == 3);
}

TEST_CASE("ExternalStorageIO factory parsing preserves common configuration",
          "[ut][ExternalStorageIO]") {
    auto original = std::make_shared<vsag::ExternalStorageIOParameter>();
    original->external_storage = "test";
    original->enable_read_cache_ = true;
    original->read_cache_total_size_ = 4096;
    original->enable_prefetch_hint_ = true;
    auto parsed = vsag::IOParameter::GetIOParameterByJson(original->ToJson());
    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->enable_read_cache_);
    REQUIRE(parsed->read_cache_total_size_ == 4096);
    REQUIRE(parsed->enable_prefetch_hint_);
}

TEST_CASE("CallbackStorage rejects empty callbacks as invalid arguments",
          "[ut][ExternalStorageIO]") {
    const auto read = [](uint64_t, uint64_t, void*) {};
    const auto write = [](uint64_t, uint64_t, const void*) {};
    const auto resize = [](uint64_t) {};
    REQUIRE_THROWS_AS(vsag::Factory::CreateExternalStorage({}, write, resize, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::Factory::CreateExternalStorage(read, {}, resize, 0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(vsag::Factory::CreateExternalStorage(read, write, {}, 0),
                      std::invalid_argument);
    auto store = std::make_shared<ByteStore>();
    auto storage = make_storage(store);
    storage.writer->Resize(1);
    uint8_t byte = 0;
    REQUIRE_THROWS_AS(storage.reader->AsyncRead(0, 1, &byte, {}), std::invalid_argument);
    REQUIRE(store->reads == 0);
}

TEST_CASE("ExternalStorageBackend submission completes synchronously", "[ut][ExternalStorageIO]") {
    REQUIRE_FALSE(vsag::ExternalStorageBackendCapabilities::AsyncReadable);
    auto store = std::make_shared<ByteStore>();
    auto storage = make_storage(store);
    storage.writer->Resize(1);
    auto param = std::make_shared<vsag::ExternalStorageIOParameter>();
    param->reader = storage.reader;
    param->writer = storage.writer;
    vsag::ExternalStorageBackend backend(store->allocator.get());
    REQUIRE(backend.Initialize(param, false, 0, 0) == 1);
    uint8_t byte = 1;
    const vsag::ReadRequest request{&byte, 0, 1};
    auto operation = backend.SubmitReads(&request, 1);
    REQUIRE(store->reads == 1);
    REQUIRE(byte == 0);
    REQUIRE(operation.Poll());
    REQUIRE(operation.Wait());
}

TEST_CASE("ExternalStorageIO rejects missing pairs", "[ut][ExternalStorageIO]") {
    auto param = std::make_shared<vsag::ExternalStorageIOParameter>();
    param->external_storage = "missing";
    vsag::IndexCommonParam common;
    common.allocator_ = vsag::Engine::CreateDefaultAllocator();
    common.external_storages_ = std::make_shared<vsag::ExternalStorageSet>();
    REQUIRE_THROWS(vsag::ExternalStorageIO(param, common));
}
