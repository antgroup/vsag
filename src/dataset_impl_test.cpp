
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
#include <catch2/generators/catch_generators.hpp>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "vsag/dataset.h"
#include "vsag/engine.h"

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

vsag::DatasetPtr
CreateTestDataset(int num_elements = 777,
                  int dim = 38,
                  int64_t extra_info_size = 13,
                  vsag::Allocator* allocator = nullptr) {
    auto base = vsag::Dataset::Make();
    auto vecs = fixtures::generate_vectors(num_elements, dim, false, fixtures::RandomValue(0, 564));
    auto distances =
        fixtures::generate_vectors(num_elements, dim, false, fixtures::RandomValue(0, 564));
    auto vecs_int8 =
        fixtures::generate_int8_codes(num_elements, dim, fixtures::RandomValue(0, 564));
    auto attr_sets = fixtures::generate_attributes(num_elements);
    std::string* paths = new std::string[num_elements];
    for (int i = 0; i < num_elements; ++i) {
        paths[i] = fixtures::create_random_string(false);
    }
    std::vector<int64_t> ids(num_elements);
    std::iota(ids.begin(), ids.end(), 0);
    base->Dim(dim)
        ->Ids(fixtures::CopyVector(ids, allocator))
        ->Paths(paths)
        ->AttributeSets(attr_sets)
        ->NumElements(num_elements)
        ->Distances(fixtures::CopyVector(distances, allocator))
        ->Owner(true, allocator);
    base->Float32Vectors(fixtures::CopyVector(vecs, allocator))
        ->Int8Vectors(fixtures::CopyVector(vecs_int8, allocator));
    if (allocator != nullptr) {
        base->SparseVectors(fixtures::CopyVector(
            fixtures::GenerateSparseVectors(allocator, num_elements, dim), allocator));
    } else {
        base->SparseVectors(
            fixtures::CopyVector(fixtures::GenerateSparseVectors(num_elements, dim), allocator));
    }
    auto extro_infos = fixtures::generate_extra_infos(num_elements, extra_info_size);
    base->ExtraInfoSize(extra_info_size)->ExtraInfos(fixtures::CopyVector(extro_infos, allocator));
    return base;
}

bool
EqualDataset(const vsag::DatasetPtr& data1, const vsag::DatasetPtr& data2) {
    if (data1->GetNumElements() != data2->GetNumElements()) {
        return false;
    }
    if (data1->GetDim() != data2->GetDim()) {
        return false;
    }
    auto num_element = data1->GetNumElements();
    auto dim = data1->GetDim();
    if (memcmp(data1->GetIds(), data2->GetIds(), sizeof(int64_t) * num_element) != 0) {
        return false;
    }
    if (memcmp(data1->GetFloat32Vectors(),
               data2->GetFloat32Vectors(),
               sizeof(float) * num_element * dim) != 0) {
        return false;
    }
    if (memcmp(data1->GetInt8Vectors(),
               data2->GetInt8Vectors(),
               sizeof(int8_t) * num_element * dim) != 0) {
        return false;
    }
    if (memcmp(data1->GetDistances(), data2->GetDistances(), sizeof(float) * num_element * dim) !=
        0) {
        return false;
    }

    auto path1 = data1->GetPaths();
    auto path2 = data2->GetPaths();
    if (path1 != nullptr && path2 != nullptr) {
        for (int i = 0; i < num_element; ++i) {
            if (path1[i] != path2[i]) {
                return false;
            }
        }
    } else if (path1 != nullptr || path2 != nullptr) {
        return false;
    }

    auto attr_sets1 = data1->GetAttributeSets();
    auto attr_sets2 = data2->GetAttributeSets();
    if (attr_sets1 != nullptr && attr_sets2 != nullptr) {
        if (attr_sets1->attrs_.size() != attr_sets2->attrs_.size()) {
            return false;
        }
        for (int i = 0; i < attr_sets1->attrs_.size(); ++i) {
            const auto& attrs1 = attr_sets1->attrs_[i];
            const auto& attrs2 = attr_sets2->attrs_[i];
            if (not attrs1->Equal(attrs2)) {
                return false;
            }
        }
    } else if (attr_sets1 != nullptr || attr_sets2 != nullptr) {
        return false;
    }

    auto sparse_vectors1 = data1->GetSparseVectors();
    auto sparse_vectors2 = data2->GetSparseVectors();
    if (sparse_vectors1 != nullptr && sparse_vectors2 != nullptr) {
        for (int i = 0; i < num_element; ++i) {
            if (sparse_vectors1[i].len_ != sparse_vectors2[i].len_) {
                return false;
            }
            if (memcmp(sparse_vectors1[i].vals_,
                       sparse_vectors2[i].vals_,
                       sizeof(float) * sparse_vectors1[i].len_) != 0) {
                return false;
            }
            if (memcmp(sparse_vectors1[i].ids_,
                       sparse_vectors2[i].ids_,
                       sizeof(uint32_t) * sparse_vectors1[i].len_) != 0) {
                return false;
            }
        }
    } else if (sparse_vectors1 != nullptr || sparse_vectors2 != nullptr) {
        return false;
    }
    if (data1->GetExtraInfoSize() != data2->GetExtraInfoSize()) {
        return false;
    }
    if (data1->GetExtraInfoSize() > 0 &&
        memcmp(data1->GetExtraInfos(),
               data2->GetExtraInfos(),
               sizeof(char) * data1->GetExtraInfoSize() * num_element) != 0) {
        return false;
    }
    return true;
}

template <typename T>
bool
AreAllPointersDifferent(T* original, T* copy, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        if (std::memcmp(original + i, copy + i, sizeof(T)) == 0) {
            return false;
        }
    }
    return true;
}

TEST_CASE("Dataset Copy and Append Test", "[ut][Dataset]") {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> great_num(1000, 2000);
    std::uniform_int_distribution<> append_num(1000, 2000);
    std::uniform_int_distribution<> dim_random(100, 200);
    int num_elements = great_num(gen);
    int append_num_elements = append_num(gen);
    int dim = dim_random(gen);

    auto use_allocator = GENERATE(true, false);
    std::shared_ptr<vsag::Allocator> allocator =
        use_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;
    int64_t extra_info_size = 13;
    auto original = CreateTestDataset(num_elements, dim, extra_info_size, allocator.get());
    SECTION("Deep Copy") {
        auto use_copy_allocator = GENERATE(true, false);
        std::shared_ptr<vsag::Allocator> copy_allocator =
            use_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;
        auto copy = original->DeepCopy(copy_allocator.get());
        REQUIRE(EqualDataset(original, copy));
        REQUIRE(AreAllPointersDifferent(
            original->GetSparseVectors(), copy->GetSparseVectors(), num_elements));

        REQUIRE(AreAllPointersDifferent(
            original->GetAttributeSets(), copy->GetAttributeSets(), num_elements));

        REQUIRE(AreAllPointersDifferent(original->GetPaths(), copy->GetPaths(), num_elements));
    }
    SECTION("Append") {
        auto copy = original->DeepCopy();
        auto append_dataset = CreateTestDataset(append_num_elements, dim);
        original->Append(append_dataset);
        REQUIRE(original->GetNumElements() == num_elements + append_num_elements);
        original->NumElements(num_elements);
        REQUIRE(EqualDataset(original, copy));
        original->NumElements(num_elements + append_num_elements);
        auto sub_original = vsag::Dataset::Make();
        sub_original->Dim(dim)
            ->Ids(original->GetIds() + num_elements)
            ->SparseVectors(original->GetSparseVectors() + num_elements)
            ->Float32Vectors(original->GetFloat32Vectors() + num_elements * dim)
            ->Int8Vectors(original->GetInt8Vectors() + num_elements * dim)
            ->Distances(original->GetDistances() + num_elements * dim)
            ->Paths(original->GetPaths() + num_elements)
            ->AttributeSets(original->GetAttributeSets() + num_elements)
            ->NumElements(append_num_elements)
            ->ExtraInfoSize(extra_info_size)
            ->ExtraInfos(original->GetExtraInfos() + num_elements * extra_info_size)
            ->Owner(false);
        REQUIRE(EqualDataset(sub_original, append_dataset));
    }
}