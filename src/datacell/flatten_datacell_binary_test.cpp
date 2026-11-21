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

#include <atomic>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "data_type.h"
#include "flatten_datacell.h"
#include "flatten_datacell_parameter.h"
#include "flatten_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/binary_quantizer.h"
#include "query_context.h"
#include "storage/serialization.h"
#include "unittest.h"
#include "utils/lock_strategy.h"
#include "vsag/factory.h"
#include "vsag/options.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

float
reference_hamming(const uint8_t* x, const uint8_t* y, uint64_t dim) {
    uint64_t distance = 0;
    for (uint64_t i = 0; i < dim; ++i) {
        const auto shift = 7 - (i % 8);
        distance += ((x[i / 8] >> shift) & 1U) != ((y[i / 8] >> shift) & 1U) ? 1 : 0;
    }
    return static_cast<float>(distance);
}

IndexCommonParam
binary_common_param(int64_t dim, const std::shared_ptr<Allocator>& allocator) {
    IndexCommonParam common;
    common.allocator_ = allocator;
    common.dim_ = dim;
    common.metric_ = MetricType::METRIC_TYPE_HAMMING;
    common.data_type_ = DataTypes::DATA_TYPE_BINARY;
    return common;
}

FlattenInterfaceParamPtr
binary_flatten_param(const std::string& io_type,
                     const std::string& quantization,
                     const fixtures::TempDir& dir) {
    JsonType json;
    json[IO_PARAMS_KEY][TYPE_KEY].SetString(io_type);
    if (io_type == IO_TYPE_VALUE_BUFFER_IO or io_type == IO_TYPE_VALUE_ASYNC_IO or
        io_type == IO_TYPE_VALUE_MMAP_IO or io_type == IO_TYPE_VALUE_URING_IO or
        io_type == IO_TYPE_VALUE_READER_IO) {
        json[IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetString(dir.GenerateRandomFile(true));
    }
    json[QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetString(quantization);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(json);
    return param;
}

template <typename IOTmpl>
void
check_binary_io_dispatch(const std::string& io_type, const fixtures::TempDir& dir) {
    INFO("io=" << io_type);
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto cell = FlattenInterface::MakeInstance(
        binary_flatten_param(io_type, QUANTIZATION_TYPE_VALUE_BINARY, dir),
        binary_common_param(64, allocator));
    using Cell = FlattenDataCell<BinaryQuantizer, FixedLayout<IOTmpl>>;
    REQUIRE(std::dynamic_pointer_cast<Cell>(cell) != nullptr);
}

}  // namespace

TEST_CASE("Binary FlattenDataCell factory dispatches every IO type",
          "[ut][FlattenDataCell][binary]") {
    fixtures::TempDir dir("binary_flatten_factory");
    check_binary_io_dispatch<MemoryBlockIO>(IO_TYPE_VALUE_BLOCK_MEMORY_IO, dir);
    check_binary_io_dispatch<MemoryIO>(IO_TYPE_VALUE_MEMORY_IO, dir);
    check_binary_io_dispatch<BufferIO>(IO_TYPE_VALUE_BUFFER_IO, dir);
    check_binary_io_dispatch<AsyncIO>(IO_TYPE_VALUE_ASYNC_IO, dir);
    check_binary_io_dispatch<UringIO>(IO_TYPE_VALUE_URING_IO, dir);
    check_binary_io_dispatch<MMapIO>(IO_TYPE_VALUE_MMAP_IO, dir);
    check_binary_io_dispatch<ReaderIO>(IO_TYPE_VALUE_READER_IO, dir);
}

TEST_CASE("Binary FlattenDataCell stores codes through user-defined IO",
          "[ut][FlattenDataCell][binary]") {
    struct Store {
        std::mutex mutex;
        std::vector<uint8_t> bytes;
        uint64_t reads{0};
        uint64_t writes{0};
    };
    auto store = std::make_shared<Store>();
    auto storage = Factory::CreateUserDefinedIO(
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
            store->bytes.resize(size);
        },
        0);
    auto storages = std::make_shared<UserDefinedIOSet>();
    storages->Set("binary_codes", storage.reader, storage.writer);

    constexpr int64_t dim = 100;
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto common = binary_common_param(dim, allocator);
    common.user_defined_ios_ = storages;
    JsonType json;
    json[IO_PARAMS_KEY][TYPE_KEY].SetString(IO_TYPE_VALUE_USER_DEFINED_IO);
    json[IO_PARAMS_KEY][IO_USER_DEFINED_IO_KEY].SetString("binary_codes");
    json[QUANTIZATION_PARAMS_KEY][TYPE_KEY].SetString(QUANTIZATION_TYPE_VALUE_BINARY);
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->FromJson(json);
    auto flatten = FlattenInterface::MakeInstance(param, common);
    using Cell = FlattenDataCell<BinaryQuantizer, FixedLayout<UserDefinedIO>>;
    REQUIRE(std::dynamic_pointer_cast<Cell>(flatten) != nullptr);

    const uint64_t row_bytes = BinaryVectorBytes(dim);
    constexpr InnerIdType count = 50;
    std::mt19937_64 rng(31);
    std::vector<uint8_t> vectors(count * row_bytes);
    for (auto& value : vectors) {
        value = static_cast<uint8_t>(rng());
    }
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);
    REQUIRE(flatten->TotalCount() == count);
    REQUIRE(store->writes > 0);
    std::vector<uint8_t> query(row_bytes);
    for (auto& value : query) {
        value = static_cast<uint8_t>(rng());
    }
    std::vector<InnerIdType> ids(count);
    for (InnerIdType i = 0; i < count; ++i) {
        ids[i] = i;
    }
    auto computer = flatten->FactoryComputer(query.data());
    std::vector<float> dists(count, -1.0F);
    flatten->Query(dists.data(), computer, ids.data(), count);
    for (InnerIdType i = 0; i < count; ++i) {
        REQUIRE(dists[i] == reference_hamming(query.data(), vectors.data() + i * row_bytes, dim));
    }
    REQUIRE(store->reads > 0);

    // Without the storage set the cell cannot bind its IO, which is an error rather than a null
    // datacell that an index would dereference.
    REQUIRE_THROWS_AS(FlattenInterface::MakeInstance(param, binary_common_param(dim, allocator)),
                      VsagException);
}

TEST_CASE("Binary FlattenDataCell range scans take external node locks",
          "[ut][FlattenDataCell][binary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("binary_flatten_locked_range");
    constexpr int64_t dim = 100;
    auto flatten = FlattenInterface::MakeInstance(
        binary_flatten_param(IO_TYPE_VALUE_BLOCK_MEMORY_IO, QUANTIZATION_TYPE_VALUE_BINARY, dir),
        binary_common_param(dim, allocator));
    const uint64_t row_bytes = BinaryVectorBytes(dim);
    constexpr InnerIdType count = 64;
    std::mt19937_64 rng(47);
    std::vector<uint8_t> vectors(count * row_bytes);
    for (auto& value : vectors) {
        value = static_cast<uint8_t>(rng());
    }
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);
    std::vector<uint8_t> query(row_bytes);
    for (auto& value : query) {
        value = static_cast<uint8_t>(rng());
    }
    auto computer = flatten->FactoryComputer(query.data());

    // An owner that hands the cell per-node read locks (HGraph with cluster vector node locks)
    // relies on every read taking them, range scans included.
    auto mutexes = std::make_shared<PointsMutex>(count, allocator.get());
    flatten->SetExternalReadMutexArray(mutexes);
    std::vector<float> dists(count, -1.0F);
    std::atomic<bool> done{false};
    mutexes->Lock(5);
    std::thread scan([&]() {
        flatten->QueryRange(dists.data(), computer, 0, count);
        done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const bool finished_while_locked = done.load();
    mutexes->Unlock(5);
    scan.join();
    REQUIRE_FALSE(finished_while_locked);
    for (InnerIdType i = 0; i < count; ++i) {
        REQUIRE(dists[i] == reference_hamming(query.data(), vectors.data() + i * row_bytes, dim));
    }
}

TEST_CASE("Binary FlattenDataCell scattered inserts and id ranges",
          "[ut][FlattenDataCell][binary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("binary_flatten_range");
    const int64_t dim = GENERATE(13, 256);
    INFO("dim=" << dim);
    auto flatten = FlattenInterface::MakeInstance(
        binary_flatten_param(IO_TYPE_VALUE_BLOCK_MEMORY_IO, QUANTIZATION_TYPE_VALUE_BINARY, dir),
        binary_common_param(dim, allocator));
    const uint64_t row_bytes = BinaryVectorBytes(dim);
    constexpr InnerIdType count = 40;
    std::mt19937_64 rng(static_cast<uint64_t>(dim) + 11);
    std::vector<uint8_t> vectors(count * row_bytes);
    for (auto& value : vectors) {
        value = static_cast<uint8_t>(rng());
    }
    flatten->Train(vectors.data(), count);
    // Ids that are not consecutive make the datacell insert row by row, stepping through the
    // packed input with the binary row size.
    std::vector<InnerIdType> ids(count);
    for (InnerIdType i = 0; i < count; ++i) {
        ids[i] = count - 1 - i;
    }
    flatten->BatchInsertVector(vectors.data(), count, ids.data());
    REQUIRE(flatten->TotalCount() == count);

    std::vector<uint8_t> query(row_bytes);
    for (auto& value : query) {
        value = static_cast<uint8_t>(rng());
    }
    auto computer = flatten->FactoryComputer(query.data());
    auto expected = [&](InnerIdType id) {
        return reference_hamming(query.data(), vectors.data() + (count - 1 - id) * row_bytes, dim);
    };

    std::vector<float> dists(count, -1.0F);
    flatten->QueryRange(dists.data(), computer, 7, 0);
    REQUIRE(dists[0] == -1.0F);  // an empty range writes nothing

    SearchStatistics stats;
    QueryContext ctx{nullptr, &stats};
    flatten->QueryRange(dists.data(), computer, 3, count - 3, &ctx);
    for (InnerIdType i = 0; i + 3 < count; ++i) {
        REQUIRE(dists[i] == expected(3 + i));
    }
    REQUIRE(stats.distance_evaluations.load() == count - 3);

    // Ranges must lie inside the stored ids.
    REQUIRE_THROWS_AS(flatten->QueryRange(dists.data(), computer, count - 2, 3), VsagException);
    REQUIRE_THROWS_AS(flatten->QueryRange(dists.data(), computer, count, 1), VsagException);
}

namespace {

// Records the allocations made through it, so that a test can tell whether a read copied codes.
class RecordingAllocator : public Allocator {
public:
    std::string
    Name() override {
        return "RecordingAllocator";
    }

    void*
    Allocate(uint64_t size) override {
        allocations_.fetch_add(1, std::memory_order_relaxed);
        uint64_t largest = largest_.load(std::memory_order_relaxed);
        while (size > largest and
               not largest_.compare_exchange_weak(largest, size, std::memory_order_relaxed)) {
        }
        return std::malloc(size);
    }

    void
    Deallocate(void* data) override {
        std::free(data);
    }

    void*
    Reallocate(void* data, uint64_t size) override {
        return std::realloc(data, size);
    }

    void
    Reset() {
        allocations_.store(0, std::memory_order_relaxed);
        largest_.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Allocations() const {
        return allocations_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    Largest() const {
        return largest_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> largest_{0};
};

class BlockSizeLimitGuard {
public:
    explicit BlockSizeLimitGuard(uint64_t block_size_limit)
        : old_block_size_limit_(Options::Instance().block_size_limit()) {
        Options::Instance().set_block_size_limit(block_size_limit);
    }

    ~BlockSizeLimitGuard() {
        Options::Instance().set_block_size_limit(old_block_size_limit_);
    }

    BlockSizeLimitGuard(const BlockSizeLimitGuard&) = delete;
    BlockSizeLimitGuard&
    operator=(const BlockSizeLimitGuard&) = delete;

private:
    uint64_t old_block_size_limit_;
};

}  // namespace

TEST_CASE("Binary FlattenDataCell scans block memory ranges in place",
          "[ut][FlattenDataCell][binary]") {
    // The smallest block size Options accepts, so that a short range crosses blocks. A lease over
    // a range that crosses a block copies the whole range.
    constexpr uint64_t block_size = 256 * 1024;
    BlockSizeLimitGuard guard(block_size);
    auto allocator = std::make_shared<RecordingAllocator>();
    fixtures::TempDir dir("binary_flatten_block_scan");
    // 2 KiB rows fill the blocks exactly; 96-byte rows also cross piece and block boundaries.
    const int64_t dim = GENERATE(16384, 768);
    INFO("dim=" << dim);
    auto flatten = FlattenInterface::MakeInstance(
        binary_flatten_param(IO_TYPE_VALUE_BLOCK_MEMORY_IO, QUANTIZATION_TYPE_VALUE_BINARY, dir),
        binary_common_param(dim, allocator));
    const uint64_t row_bytes = BinaryVectorBytes(dim);
    const auto count = static_cast<InnerIdType>(3 * block_size / row_bytes + 5);
    std::mt19937_64 rng(static_cast<uint64_t>(dim) + 29);
    std::vector<uint8_t> vectors(count * row_bytes);
    for (auto& value : vectors) {
        value = static_cast<uint8_t>(rng());
    }
    flatten->Train(vectors.data(), count);
    flatten->BatchInsertVector(vectors.data(), count);
    REQUIRE(flatten->TotalCount() == count);

    std::vector<uint8_t> query(row_bytes);
    for (auto& value : query) {
        value = static_cast<uint8_t>(rng());
    }
    auto computer = flatten->FactoryComputer(query.data());
    for (const InnerIdType begin : {InnerIdType{0}, InnerIdType{7}}) {
        INFO("begin=" << begin);
        std::vector<float> dists(count - begin, -1.0F);
        allocator->Reset();
        flatten->QueryRange(dists.data(), computer, begin, count - begin);
        const auto allocations = allocator->Allocations();
        const auto largest = allocator->Largest();
        for (InnerIdType i = 0; i < count - begin; ++i) {
            REQUIRE(dists[i] ==
                    reference_hamming(query.data(), vectors.data() + (begin + i) * row_bytes, dim));
        }
        // At most a row that crosses a block is copied, never the range.
        REQUIRE(largest <= row_bytes);
        if (block_size % row_bytes == 0) {
            REQUIRE(allocations == 0);
        }
    }
}

TEST_CASE("Binary FlattenDataCell stores packed bits exactly", "[ut][FlattenDataCell][binary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("binary_flatten");
    const int64_t dim = GENERATE(9, 64, 300);
    const std::string io_type = GENERATE(std::string(IO_TYPE_VALUE_MEMORY_IO),
                                         std::string(IO_TYPE_VALUE_BLOCK_MEMORY_IO),
                                         std::string(IO_TYPE_VALUE_BUFFER_IO),
                                         std::string(IO_TYPE_VALUE_MMAP_IO),
                                         std::string(IO_TYPE_VALUE_ASYNC_IO));
    INFO("dim=" << dim << " io=" << io_type);
    const auto common = binary_common_param(dim, allocator);
    auto flatten = FlattenInterface::MakeInstance(
        binary_flatten_param(io_type, QUANTIZATION_TYPE_VALUE_BINARY, dir), common);
    REQUIRE(flatten != nullptr);
    REQUIRE(flatten->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_BINARY);
    REQUIRE(flatten->GetMetricType() == MetricType::METRIC_TYPE_HAMMING);
    REQUIRE(flatten->code_size_ == BinaryVectorBytes(dim));

    const uint64_t row_bytes = BinaryVectorBytes(dim);
    constexpr InnerIdType batch_count = 60;
    constexpr InnerIdType single_count = 7;
    constexpr InnerIdType count = batch_count + single_count;
    std::mt19937_64 rng(static_cast<uint64_t>(dim));
    std::vector<uint8_t> vectors(count * row_bytes);
    for (auto& value : vectors) {
        value = static_cast<uint8_t>(rng());
    }
    flatten->Train(vectors.data(), count);
    // A contiguous batch and single inserts exercise both binary input strides.
    flatten->BatchInsertVector(vectors.data(), batch_count);
    for (InnerIdType i = batch_count; i < count; ++i) {
        flatten->InsertVector(vectors.data() + i * row_bytes, i);
    }
    REQUIRE(flatten->TotalCount() == count);

    std::vector<uint8_t> query(row_bytes);
    for (auto& value : query) {
        value = static_cast<uint8_t>(rng());
    }
    std::vector<InnerIdType> ids(count);
    for (InnerIdType i = 0; i < count; ++i) {
        ids[i] = i;
    }
    auto check = [&](const FlattenInterfacePtr& cell) {
        auto computer = cell->FactoryComputer(query.data());
        std::vector<float> dists(count, -1.0F);
        cell->Query(dists.data(), computer, ids.data(), count);
        for (InnerIdType i = 0; i < count; ++i) {
            REQUIRE(dists[i] ==
                    reference_hamming(query.data(), vectors.data() + i * row_bytes, dim));
        }
        for (InnerIdType i = 0; i + 1 < count; i += 5) {
            REQUIRE(cell->ComputePairVectors(i, i + 1) ==
                    reference_hamming(
                        vectors.data() + i * row_bytes, vectors.data() + (i + 1) * row_bytes, dim));
        }
        std::vector<uint8_t> code(row_bytes);
        cell->GetCodesById(count - 1, code.data());
        REQUIRE(reference_hamming(code.data(), vectors.data() + (count - 1) * row_bytes, dim) ==
                0.0F);
    };
    check(flatten);

    std::stringstream stream;
    IOStreamWriter writer(stream);
    flatten->Serialize(writer);
    auto restored = FlattenInterface::MakeInstance(
        binary_flatten_param(io_type, QUANTIZATION_TYPE_VALUE_BINARY, dir), common);
    IOStreamReader reader(stream);
    restored->Deserialize(reader);
    REQUIRE(restored->TotalCount() == count);
    check(restored);
}

TEST_CASE("Binary FlattenDataCell rejects other quantizers", "[ut][FlattenDataCell][binary]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    fixtures::TempDir dir("binary_flatten_reject");
    const auto common = binary_common_param(128, allocator);
    for (const auto* quantization : {QUANTIZATION_TYPE_VALUE_FP32,
                                     QUANTIZATION_TYPE_VALUE_SQ8,
                                     QUANTIZATION_TYPE_VALUE_PQ,
                                     QUANTIZATION_TYPE_VALUE_RABITQ}) {
        REQUIRE_THROWS_WITH(
            FlattenInterface::MakeInstance(
                binary_flatten_param(IO_TYPE_VALUE_MEMORY_IO, quantization, dir), common),
            Catch::Matchers::ContainsSubstring("only supports binary quantization"));
    }

    // The binary quantizer, in turn, needs binary input and the Hamming metric.
    IndexCommonParam float_common;
    float_common.allocator_ = allocator;
    float_common.dim_ = 128;
    float_common.metric_ = MetricType::METRIC_TYPE_L2SQR;
    REQUIRE_THROWS_WITH(
        FlattenInterface::MakeInstance(
            binary_flatten_param(IO_TYPE_VALUE_MEMORY_IO, QUANTIZATION_TYPE_VALUE_BINARY, dir),
            float_common),
        Catch::Matchers::ContainsSubstring("binary quantization requires dtype binary"));

    auto mismatched = binary_common_param(128, allocator);
    mismatched.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    REQUIRE_THROWS_WITH(
        FlattenInterface::MakeInstance(
            binary_flatten_param(IO_TYPE_VALUE_MEMORY_IO, QUANTIZATION_TYPE_VALUE_BINARY, dir),
            mismatched),
        Catch::Matchers::ContainsSubstring("metric hamming requires dtype binary"));
}

}  // namespace vsag
