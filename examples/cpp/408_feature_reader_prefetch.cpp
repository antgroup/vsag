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

#include <vsag/vsag.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t K_DIM = 64;
constexpr int64_t K_NUM_VECTORS = 1000;
constexpr uint64_t K_FILE_HEADER_SIZE = 4096;
constexpr const char* K_FILE_PATH = "/tmp/vsag-reader-prefetch.index";

const char* K_BUILD_PARAMETERS = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 64,
    "index_param": {
        "buckets_count": 16,
        "base_quantization_type": "sq8",
        "partition_strategy_type": "ivf",
        "ivf_train_type": "random",
        "train_sample_count": 800,
        "use_reorder": true,
        "precise_quantization_type": "fp32",
        "base_io_type": "memory_io",
        "precise_io_type": "block_memory_io"
    }
}
)";

class PrefetchingFileReader : public vsag::Reader, public vsag::ReaderPrefetcher {
public:
    PrefetchingFileReader(std::string path, uint64_t base_offset, uint64_t size)
        : file_(std::move(path), std::ios::binary), base_offset_(base_offset), size_(size) {
        if (!file_) {
            throw std::runtime_error("failed to open serialized index");
        }
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        if (offset > size_ or len > size_ - offset) {
            throw std::out_of_range("reader range is out of bounds");
        }
        std::lock_guard<std::mutex> lock(mutex_);
        file_.clear();
        file_.seekg(static_cast<std::streamoff>(base_offset_ + offset), std::ios::beg);
        file_.read(static_cast<char*>(dest), static_cast<std::streamsize>(len));
        if (file_.fail()) {
            throw std::runtime_error("failed to read serialized index");
        }
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
        try {
            Read(offset, len, dest);
            callback(vsag::IOErrorCode::IO_SUCCESS, "success");
        } catch (const std::exception& error) {
            callback(vsag::IOErrorCode::IO_ERROR, error.what());
        }
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return size_;
    }

    void
    Prefetch(uint64_t offset, uint64_t len) override {
        // A remote/object-store implementation can enqueue an asynchronous range fetch here.
        // Return promptly and keep failures non-fatal. ReaderIO already clamps this range.
        last_offset_.store(offset, std::memory_order_relaxed);
        last_len_.store(len, std::memory_order_relaxed);
        count_.fetch_add(1, std::memory_order_relaxed);
    }

    [[nodiscard]] uint64_t
    PrefetchCount() const {
        return count_.load(std::memory_order_relaxed);
    }

private:
    std::ifstream file_;
    std::mutex mutex_;
    uint64_t base_offset_;
    uint64_t size_;
    std::atomic<uint64_t> count_{0};
    std::atomic<uint64_t> last_offset_{0};
    std::atomic<uint64_t> last_len_{0};
};

template <typename T>
T
Require(tl::expected<T, vsag::Error> result, const char* action) {
    if (!result.has_value()) {
        std::cerr << action << " failed: " << result.error().message << std::endl;
        std::abort();
    }
    return std::move(result.value());
}

}  // namespace

int
main() {
    std::remove(K_FILE_PATH);

    std::vector<int64_t> ids(K_NUM_VECTORS);
    std::vector<float> vectors(K_NUM_VECTORS * K_DIM);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distribution;
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = distribution(rng);
    }
    auto base = vsag::Dataset::Make()
                    ->NumElements(K_NUM_VECTORS)
                    ->Dim(K_DIM)
                    ->Ids(ids.data())
                    ->Float32Vectors(vectors.data())
                    ->Owner(false);

    auto index = Require(vsag::Factory::CreateIndex("ivf", K_BUILD_PARAMETERS), "CreateIndex");
    if (auto result = index->Build(base); !result.has_value()) {
        std::cerr << "Build failed: " << result.error().message << std::endl;
        return EXIT_FAILURE;
    }

    {
        std::ofstream output(K_FILE_PATH, std::ios::binary);
        std::vector<char> header(K_FILE_HEADER_SIZE, 'H');
        output.write(header.data(), static_cast<std::streamsize>(header.size()));
        if (auto result = index->SerializeStreaming(output); !result.has_value()) {
            std::cerr << "SerializeStreaming failed: " << result.error().message << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::ifstream metadata_stream(K_FILE_PATH, std::ios::binary);
    metadata_stream.seekg(static_cast<std::streamoff>(K_FILE_HEADER_SIZE), std::ios::beg);
    auto metadata = Require(vsag::Index::GetStreamingMetadata(metadata_stream), "metadata");
    const vsag::StreamingBlockLayout* precise_codes = nullptr;
    for (const auto& block : metadata.blocks) {
        if (block.name == "high_precision_codes") {
            precise_codes = &block;
            break;
        }
    }
    if (precise_codes == nullptr) {
        std::cerr << "high_precision_codes block is missing" << std::endl;
        return EXIT_FAILURE;
    }

    auto reader =
        std::make_shared<PrefetchingFileReader>(K_FILE_PATH,
                                                K_FILE_HEADER_SIZE + precise_codes->payload_offset,
                                                precise_codes->payload_size);
    vsag::LoadParameters load_parameters;
    load_parameters.Set("precise_io_type", "reader_io")
        .Set("precise_enable_prefetch_hint", true)
        .SetReader("precise_reader", reader);

    std::ifstream load_stream(K_FILE_PATH, std::ios::binary);
    load_stream.seekg(static_cast<std::streamoff>(K_FILE_HEADER_SIZE), std::ios::beg);
    auto loaded = Require(vsag::Index::Load(load_stream, load_parameters), "Load");

    auto query = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(K_DIM)
                     ->Float32Vectors(vectors.data())
                     ->Owner(false);
    auto result =
        Require(loaded->KnnSearch(query, 10, R"({"ivf":{"scan_buckets_count":16}})"), "KnnSearch");

    std::cout << "top result: " << result->GetIds()[0] << std::endl;
    std::cout << "ReaderPrefetcher hints observed: " << reader->PrefetchCount() << std::endl;
    if (reader->PrefetchCount() == 0) {
        std::cerr << "expected reader prefetch hints" << std::endl;
        return EXIT_FAILURE;
    }
    std::remove(K_FILE_PATH);
    return EXIT_SUCCESS;
}
