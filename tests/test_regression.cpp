
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

#include <vsag/options.h>
#include <vsag/vsag.h>
#include <vsag/vsag_ext.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <limits>
#include <new>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <unordered_map>

#include "algorithm/hnswlib/visited_list_pool.h"

using namespace vsag;

// https://github.com/antgroup/vsag/issues/369
TEST_CASE("gh#369", "[ft][github]") {
    using namespace nlohmann;

    class MyAllocator : public Allocator {
    public:
        std::string
        Name() override {
            return "MyAllocator";
        }

        void*
        Allocate(uint64_t size) override {
            if (size == 0) {
                throw std::bad_alloc();
            }
            return malloc(size);
        }

        void
        Deallocate(void* p) override {
            free(p);
        }

        void*
        Reallocate(void* p, uint64_t size) override {
            return realloc(p, size);
        }
    };

    auto dim = 32;
    MyAllocator vsag_allocator;
    int64_t ids[11000];
    float vector_list[11000 * dim];
    float query_vector[dim];
    auto search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )"_json;

    // filter out 100%, empty result
    std::function<bool(int64_t)> filter = [](int64_t) -> bool { return true; };

    json hnswsq_parameters{{"base_quantization_type", "sq8"},
                           {"max_degree", 8},
                           {"ef_construction", 100},
                           {"build_thread_count", 1}};
    json index_parameters{{"dtype", "float32"},
                          {"metric_type", "l2"},
                          {"dim", dim},
                          {"index_param", hnswsq_parameters}};
    auto create_result =
        vsag::Factory::CreateIndex("hgraph", index_parameters.dump(), &vsag_allocator);
    REQUIRE(create_result.has_value());
    auto hgraph = create_result.value();

    auto dataset = Dataset::Make();
    dataset->Dim(dim)->NumElements(11000)->Ids(ids)->Float32Vectors(vector_list)->Owner(false);

    auto build_result = hgraph->Build(dataset);
    REQUIRE(build_result.has_value());

    auto query = Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);

    auto search_result = hgraph->KnnSearch(query, 10, search_parameters.dump(), filter);
    REQUIRE(search_result.has_value());
}

// https://github.com/antgroup/vsag/issues/1974
TEST_CASE("gh#1974", "[ft][github]") {
    using namespace nlohmann;

    class BlockSizeLimitGuard {
    public:
        explicit BlockSizeLimitGuard(uint64_t block_size_limit)
            : origin_size_(vsag::Options::Instance().block_size_limit()) {
            vsag::Options::Instance().set_block_size_limit(block_size_limit);
        }

        ~BlockSizeLimitGuard() {
            vsag::Options::Instance().set_block_size_limit(origin_size_);
        }

    private:
        uint64_t origin_size_;
    };

    class LimitedAllocator : public Allocator {
    public:
        std::string
        Name() override {
            return "LimitedAllocator";
        }

        void*
        Allocate(uint64_t size) override {
            if (used_ + size > limit_) {
                return nullptr;
            }
            auto* ptr = std::malloc(size);
            if (ptr != nullptr) {
                sizes_[ptr] = size;
                used_ += size;
            }
            return ptr;
        }

        void
        Deallocate(void* p) override {
            auto iter = sizes_.find(p);
            if (iter != sizes_.end()) {
                used_ -= iter->second;
                sizes_.erase(iter);
            }
            std::free(p);
        }

        void*
        Reallocate(void* p, uint64_t size) override {
            auto old_size = uint64_t{0};
            auto iter = sizes_.find(p);
            if (iter != sizes_.end()) {
                old_size = iter->second;
            }
            if (used_ - old_size + size > limit_) {
                return nullptr;
            }
            auto* new_ptr = std::realloc(p, size);
            if (new_ptr != nullptr) {
                if (iter != sizes_.end()) {
                    sizes_.erase(iter);
                }
                sizes_[new_ptr] = size;
                used_ = used_ - old_size + size;
            }
            return new_ptr;
        }

        void
        SetLimit(uint64_t limit) {
            limit_ = limit;
        }

        uint64_t
        Used() const {
            return used_;
        }

    private:
        uint64_t limit_{std::numeric_limits<uint64_t>::max()};
        uint64_t used_{0};
        std::unordered_map<void*, uint64_t> sizes_;
    };

    BlockSizeLimitGuard block_size_limit_guard(512ULL * 1024ULL);

    constexpr int64_t dim = 32;
    constexpr int64_t base_count = 4096;
    constexpr int64_t add_count = 1;
    std::vector<int64_t> base_ids(base_count);
    std::vector<float> base_vectors(base_count * dim, 0.0F);
    std::vector<int64_t> add_ids(add_count);
    std::vector<float> add_vectors(add_count * dim, 1.0F);
    std::iota(base_ids.begin(), base_ids.end(), 0);
    std::iota(add_ids.begin(), add_ids.end(), base_count);

    json base_param{{"base_quantization_type", "sq8"},
                    {"max_degree", 8},
                    {"ef_construction", 100},
                    {"build_thread_count", 1},
                    {"store_raw_vector", true}};
    json tuned_param = base_param;
    tuned_param["base_quantization_type"] = "bf16";
    json index_param{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"index_param", base_param}};
    json tune_param{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"index_param", tuned_param}};

    LimitedAllocator allocator;
    auto create_result = vsag::Factory::CreateIndex("hgraph", index_param.dump(), &allocator);
    REQUIRE(create_result.has_value());
    auto hgraph = create_result.value();

    auto base = Dataset::Make();
    base->Dim(dim)
        ->NumElements(base_count)
        ->Ids(base_ids.data())
        ->Float32Vectors(base_vectors.data())
        ->Owner(false);
    auto build_result = hgraph->Build(base);
    REQUIRE(build_result.has_value());
    REQUIRE(hgraph->GetNumElements() == base_count);

    allocator.SetLimit(allocator.Used());
    auto add = Dataset::Make();
    add->Dim(dim)
        ->NumElements(add_count)
        ->Ids(add_ids.data())
        ->Float32Vectors(add_vectors.data())
        ->Owner(false);
    auto add_result = hgraph->Add(add);
    REQUIRE_FALSE(add_result.has_value());
    REQUIRE(add_result.error().type == ErrorType::NO_ENOUGH_MEMORY);
    REQUIRE(hgraph->GetNumElements() == base_count);

    allocator.SetLimit(std::numeric_limits<uint64_t>::max());
    auto tune_result = hgraph->Tune(tune_param.dump());
    REQUIRE(tune_result.has_value());
    REQUIRE(tune_result.value());
}
