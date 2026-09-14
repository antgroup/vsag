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

#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <vector>

/// Trivial mutexed byte store shared by the three callbacks.
struct ByteStore {
    std::mutex mutex;
    std::vector<uint8_t> bytes;
    uint64_t reads{0};
    uint64_t writes{0};
    uint64_t resizes{0};
};

int
main(int argc, char** argv) {
    vsag::init();

    int64_t dim = 128;
    int64_t count = 1000;
    int64_t num_query = 10;
    int64_t topk = 10;

    /******************* Prepare Dataset *****************/
    std::vector<int64_t> ids(count);
    std::vector<float> base_vectors(count * dim);
    std::vector<float> query_vectors(num_query * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * count; ++i) {
        base_vectors[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < dim * num_query; ++i) {
        query_vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    auto query = vsag::Dataset::Make();
    query->NumElements(num_query)->Dim(dim)->Float32Vectors(query_vectors.data())->Owner(false);

    /******************* Create External Storage *****************/
    // Three callbacks share a mutexed byte vector.
    // read_func must throw on out-of-bounds, copy bytes, and be immediately consistent
    // with completed writes.
    // write_func must grow the store on over-size writes and consume `source` before
    // returning.
    // resize_func must truncate or extend the backing store and preserve the retained prefix.
    // initial_size (last argument) must match existing backing storage, or 0 for a fresh
    // store.
    auto store = std::make_shared<ByteStore>();
    auto read_func = [store](uint64_t offset, uint64_t len, void* dest) {
        std::lock_guard<std::mutex> lock(store->mutex);
        if (offset > store->bytes.size() or len > store->bytes.size() - offset) {
            throw std::runtime_error("read out of bounds");
        }
        if (len > 0) {
            std::memcpy(dest, store->bytes.data() + offset, len);
        }
        ++store->reads;
    };
    auto write_func = [store](uint64_t offset, uint64_t len, const void* source) {
        std::lock_guard<std::mutex> lock(store->mutex);
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
    };
    auto resize_func = [store](uint64_t size) {
        std::lock_guard<std::mutex> lock(store->mutex);
        store->bytes.resize(size);
        ++store->resizes;
    };
    auto storage = vsag::Factory::CreateUserDefinedIO(read_func, write_func, resize_func, 0);

    // Register the pair under a name that matches the index parameter.
    vsag::UserDefinedIOSet storages;
    storages.Set("hgraph_precise", storage.reader, storage.writer);

    /******************* Create HGraph Index *****************/
    std::string build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100,
            "use_reorder": true,
            "precise_quantization_type": "fp32",
            "precise_io_type": "user_defined_io",
            "precise_user_defined_io": "hgraph_precise"
        }
    }
    )";
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", build_parameters, storages).value();

    /******************* Build *****************/
    // The write_func and resize_func callbacks are invoked during Build.
    auto build_result = index->Build(base);
    if (not build_result.has_value()) {
        std::cerr << "build failed: " << build_result.error().message << std::endl;
        return 1;
    }

    /******************* Search *****************/
    auto search_result = index->KnnSearch(query, topk, R"(
    { "hgraph": { "ef_search": 50 } }
    )");
    if (not search_result.has_value()) {
        std::cerr << "search failed: " << search_result.error().message << std::endl;
        return 1;
    }

    /******************* Print Results *****************/
    auto result = search_result.value();
    for (int64_t i = 0; i < num_query; ++i) {
        std::cout << "query " << i << " top-" << topk << ": ";
        for (int64_t j = 0; j < topk; ++j) {
            std::cout << result->GetIds()[i * topk + j] << "("
                      << result->GetDistances()[i * topk + j] << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "reads: " << store->reads << ", writes: " << store->writes
              << ", resizes: " << store->resizes << std::endl;

    return 0;
}
