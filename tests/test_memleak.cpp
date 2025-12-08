
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

#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <unordered_map>

#include "fixtures.h"
#include "fmt/format.h"
#include "vsag/allocator.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

using namespace vsag;

class MyAllocator : public vsag::Allocator {
public:
    MyAllocator() = default;

    uint64_t
    UsedMemory() const {
        uint64_t used = 0;
        for (const auto& [_, size] : addrs_) {
            used += size;
        }
        return used;
    }

public:
    std::string
    Name() override {
        return "my_allocator";
    }

    void*
    Allocate(size_t size) override {
        std::scoped_lock lock(mutex_);
        auto* ptr = malloc(size);
        if (ptr != nullptr) {
            addrs_[ptr] = size;
        }
        return ptr;
    }

    void
    Deallocate(void* p) override {
        std::scoped_lock lock(mutex_);
        if (addrs_.erase(p) != 0) {
            free(p);
        }
    }

    void*
    Reallocate(void* p, size_t size) override {
        std::scoped_lock lock(mutex_);
        if (p == nullptr) {
            return Allocate(size);
        }
        if (addrs_.find(p) == addrs_.end()) {
            return nullptr;
        }
        auto* new_ptr = realloc(p, size);
        if (new_ptr != nullptr) {
            addrs_.erase(p);
            addrs_[new_ptr] = size;
        } else if (size == 0) {
            // realloc(p, 0) may have freed p. We must untrack it.
            addrs_.erase(p);
        }
        return new_ptr;
    }

private:
    std::mutex mutex_;
    std::unordered_map<void*, uint64_t> addrs_;
};

class ScopedIndex {
public:
    explicit ScopedIndex(const std::string& name,
                         const std::string& build_parameters,
                         int64_t num_vectors,
                         int64_t dim,
                         Allocator* allocator) {
        resource_ = std::make_unique<vsag::Resource>(allocator, nullptr);
        engine_ = std::make_unique<vsag::Engine>(resource_.get());

        index_ = engine_->CreateIndex(name, build_parameters).value();

        auto [ids, vecs] = fixtures::generate_ids_and_vectors(num_vectors, dim);
        auto base = vsag::Dataset::Make();
        base->NumElements(num_vectors)
            ->Dim(dim)
            ->Ids(ids.data())
            ->Float32Vectors(vecs.data())
            ->Owner(false);

        index_->Build(base).value();
    }

    ~ScopedIndex() {
        index_ = nullptr;
        engine_->Shutdown();
        engine_ = nullptr;
        resource_ = nullptr;
    }

public:
    std::shared_ptr<vsag::Index>&
    GetIndex() {
        return index_;
    }

private:
    std::unique_ptr<vsag::Resource> resource_;
    std::unique_ptr<vsag::Engine> engine_;
    std::shared_ptr<vsag::Index> index_;
};

std::string
generate_param(const std::string& index_name, const std::string& base_quantization_type, int dim) {
    if (index_name == "hgraph") {
        return fmt::format(
            R"({{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "index_param": {{
                "base_quantization_type": "{}",
                "max_degree": 48,
                "ef_construction": 100
            }}
            }})",
            dim,
            base_quantization_type);
    } else if (index_name == "ivf") {
        return fmt::format(
            R"({{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": {},
            "index_param": {{
                "buckets_count": 50,
                "base_quantization_type": "{}",
                "partition_strategy_type": "ivf",
                "ivf_train_type": "kmeans",
                "ivf_train_sample_count": 1000
            }}
            }})",
            dim,
            base_quantization_type);
    }
    throw std::runtime_error("Unsupported index name: " + index_name);
}

const std::tuple<std::string, std::string, std::string, int64_t, int64_t> INDEX_PARAMS[] = {
    {"hgraph", "fp32", R"({"hgraph": {"ef_search": 100}})", 10000, 128},

    {"hgraph", "sq8_uniform", R"({"hgraph": {"ef_search": 100}})", 10000, 128},

    {"ivf", "fp32", R"({"ivf": {"scan_buckets_count": 10}})", 10000, 128},
};

TEST_CASE("Test Classic Index Memory Leak", "[ft][memleak]") {
    Options::Instance().set_block_size_limit(4ULL * 1024 * 1024);
    auto* allocator = new MyAllocator();
    REQUIRE(allocator->UsedMemory() == 0);
    for (const auto& [index_name, base_quantization_type, search_parameter, num_vectors, dim] :
         INDEX_PARAMS) {
        auto index_parameter = generate_param(index_name, base_quantization_type, dim);
        ScopedIndex scoped_index(index_name, index_parameter, num_vectors, dim, allocator);

        const int64_t num_queries = 1'000;
        auto query_vector = fixtures::generate_vectors(num_queries, dim);
        auto query = vsag::Dataset::Make();

        for (uint64_t i = 0; i < num_queries; ++i) {
            query->NumElements(1)
                ->Dim(dim)
                ->Float32Vectors(query_vector.data() + i * dim)
                ->Owner(false);

            scoped_index.GetIndex()->KnnSearch(query, 10, search_parameter).value();
        }
    }
    REQUIRE(allocator->UsedMemory() == 0);
    delete allocator;
}
