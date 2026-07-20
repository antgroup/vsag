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

#include <array>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <iostream>
#include <mutex>
#include <random>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#include "vsag/options.h"

namespace {

constexpr int64_t K_DIM = 16;
constexpr int64_t K_NUM_VECTORS = 200;
constexpr int64_t K_TOP_K = 5;

const char* hgraph_build_parameters = R"(
{
    "dtype": "float32",
    "metric_type": "l2",
    "dim": 16,
    "index_param": {
        "base_quantization_type": "sq8",
        "max_degree": 16,
        "ef_construction": 100,
        "alpha": 1.2
    }
}
)";

const char* hgraph_search_parameters = R"(
{
    "hgraph": {
        "ef_search": 100
    }
}
)";

class BlockingStreamBuffer : public std::streambuf {
public:
    void
    Close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

protected:
    int_type
    overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        char value = traits_type::to_char_type(ch);
        return xsputn(&value, 1) == 1 ? ch : traits_type::eof();
    }

    std::streamsize
    xsputn(const char* data, std::streamsize count) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::streamsize i = 0; i < count; ++i) {
                buffer_.push_back(data[i]);
            }
        }
        ready_.notify_all();
        return count;
    }

    int
    sync() override {
        ready_.notify_all();
        return 0;
    }

    int_type
    underflow() override {
        if (gptr() != nullptr && gptr() < egptr()) {
            return traits_type::to_int_type(*gptr());
        }

        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return closed_ || !buffer_.empty(); });
        if (buffer_.empty()) {
            return traits_type::eof();
        }

        uint64_t count = 0;
        while (count < read_buffer_.size() && !buffer_.empty()) {
            read_buffer_[count] = buffer_.front();
            buffer_.pop_front();
            ++count;
        }
        setg(read_buffer_.data(),
             read_buffer_.data(),
             read_buffer_.data() + static_cast<std::ptrdiff_t>(count));
        return traits_type::to_int_type(*gptr());
    }

    std::streamsize
    showmanyc() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<std::streamsize>(buffer_.size());
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<char> buffer_;
    std::array<char, 4096> read_buffer_{};
    bool closed_{false};
};

vsag::DatasetPtr
make_base_dataset(std::vector<int64_t>& ids, std::vector<float>& vectors) {
    ids.resize(K_NUM_VECTORS);
    vectors.resize(K_NUM_VECTORS * K_DIM);

    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist;
    for (int64_t i = 0; i < K_NUM_VECTORS; ++i) {
        ids[i] = i;
    }
    for (auto& value : vectors) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(K_NUM_VECTORS)
        ->Dim(K_DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
make_query_dataset(std::vector<float>& query_vector) {
    query_vector.resize(K_DIM);
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist;
    for (auto& value : query_vector) {
        value = dist(rng);
    }

    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(K_DIM)
        ->Float32Vectors(query_vector.data())
        ->Owner(false);
}

template <typename T>
void
check_result(const tl::expected<T, vsag::Error>& result, const std::string& action) {
    if (!result.has_value()) {
        std::cerr << action << " failed: " << result.error().message << std::endl;
        std::abort();
    }
}

void
rethrow_if_set(const std::exception_ptr& error) {
    if (error != nullptr) {
        std::rethrow_exception(error);
    }
}

void
print_results(const vsag::DatasetPtr& result) {
    std::cout << "results:" << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  " << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}

}  // namespace

int
main() {
    vsag::Options::Instance().set_block_size_limit(2UL * 1024 * 1024);

    std::vector<int64_t> ids;
    std::vector<float> vectors;
    auto base = make_base_dataset(ids, vectors);

    auto index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();
    check_result(index->Build(base), "Build");

    BlockingStreamBuffer shared_buffer;
    std::iostream shared_stream(&shared_buffer);
    vsag::IndexPtr received_index;
    std::exception_ptr writer_error;
    std::exception_ptr reader_error;

    std::thread reader([&] {
        try {
            auto create_index = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters);
            check_result(create_index, "Create receiver index");
            received_index = create_index.value();
            check_result(received_index->DeserializeStreaming(shared_stream),
                         "DeserializeStreaming");
        } catch (...) {
            reader_error = std::current_exception();
        }
    });

    std::thread writer([&] {
        try {
            check_result(index->SerializeStreaming(shared_stream), "SerializeStreaming");
            shared_stream.flush();
        } catch (...) {
            writer_error = std::current_exception();
        }
        shared_buffer.Close();
    });

    writer.join();
    reader.join();
    rethrow_if_set(writer_error);
    rethrow_if_set(reader_error);

    std::vector<float> query_vector;
    auto query = make_query_dataset(query_vector);
    auto result = received_index->KnnSearch(query, K_TOP_K, hgraph_search_parameters);
    check_result(result, "KnnSearch");

    std::cout << "Transferred HGraph through one std::iostream with streaming serialization"
              << std::endl;
    std::cout << "Received index contains: " << received_index->GetNumElements() << std::endl;
    print_results(result.value());
    return 0;
}
