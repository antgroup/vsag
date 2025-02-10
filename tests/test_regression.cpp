
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
#include <vsag/vsag_ext.h>

#include <catch2/catch_test_macros.hpp>
#include <new>
#include <nlohmann/json.hpp>
#include <string>

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
        Allocate(size_t size) override {
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
        Reallocate(void* p, size_t size) override {
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
