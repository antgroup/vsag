
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

#include "dataset_impl.h"

#include <catch2/catch_test_macros.hpp>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "vsag/dataset.h"

TEST_CASE("Dataset Implement Test", "[ut][dataset]") {
    vsag::DefaultAllocator allocator;
    SECTION("allocator") {
        auto dataset = vsag::Dataset::Make();
        auto* data = (float*)allocator.Allocate(sizeof(float) * 1);
        dataset->Float32Vectors(data)->Owner(true, &allocator);
    }

    SECTION("delete") {
        auto dataset = vsag::Dataset::Make();
        auto* data = new float[1];
        dataset->Float32Vectors(data);
    }

    SECTION("default") {
        auto dataset = vsag::Dataset::Make();
        auto* data = new float[1];
        dataset->Float32Vectors(data)->Owner(false);
        delete[] data;
    }

    SECTION("extra_info") {
        auto dataset = vsag::Dataset::Make();
        std::string extra_info = "0123456789";
        int64_t extra_info_size = 2;
        dataset->ExtraInfoSize(extra_info_size)->ExtraInfos(extra_info.c_str())->Owner(false);

        REQUIRE(dataset->GetExtraInfoSize() == extra_info_size);
        auto* get_result = dataset->GetExtraInfos();
        REQUIRE(get_result[6] == '6');
    }

    SECTION("sparse vector") {
        uint32_t size = 100;
        uint32_t max_dim = 256;
        uint32_t max_id = 1000000;
        float min_val = -100;
        float max_val = 100;
        int seed = 114514;

        // generate data
        std::vector<vsag::SparseVector> sparse_vectors =
            fixtures::GenerateSparseVectors(size, max_dim, max_id, min_val, max_val, seed);
        auto dataset = vsag::Dataset::Make();
        dataset->SparseVectors(fixtures::CopyVector(sparse_vectors))
            ->NumElements(size)
            ->Owner(true);

        // validate data
        auto sparse_vectors_ptr = dataset->GetSparseVectors();
        for (int i = 0; i < dataset->GetNumElements(); i++) {
            uint32_t dim = sparse_vectors_ptr[i].len_;
            REQUIRE(dim <= max_dim);
            for (int d = 0; d < dim; d++) {
                REQUIRE(sparse_vectors_ptr[i].ids_[d] < max_id);
                REQUIRE(min_val < sparse_vectors_ptr[i].vals_[d]);
                REQUIRE(sparse_vectors_ptr[i].vals_[d] < max_val);
            }
        }
    }

    SECTION("sparse vector with allocator") {
        uint32_t size = 100;
        uint32_t max_dim = 256;
        uint32_t max_id = 1000000;
        float min_val = -100;
        float max_val = 100;
        int seed = 114514;

        // generate data
        vsag::Vector<vsag::SparseVector> sparse_vectors = fixtures::GenerateSparseVectors(
            &allocator, size, max_dim, max_id, min_val, max_val, seed);
        auto dataset = vsag::Dataset::Make();
        dataset->SparseVectors(fixtures::CopyVector(sparse_vectors, &allocator))
            ->NumElements(size)
            ->Owner(true, &allocator);

        // validate data
        auto sparse_vectors_ptr = dataset->GetSparseVectors();
        for (int i = 0; i < dataset->GetNumElements(); i++) {
            uint32_t dim = sparse_vectors_ptr[i].len_;
            REQUIRE(dim < max_dim);
            for (int d = 0; d < dim; d++) {
                REQUIRE(sparse_vectors_ptr[i].ids_[d] < max_id);
                REQUIRE(min_val < sparse_vectors_ptr[i].vals_[d]);
                REQUIRE(sparse_vectors_ptr[i].vals_[d] < max_val);
            }
        }
    }
}
