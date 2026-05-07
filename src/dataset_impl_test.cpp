
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

#include "impl/allocator/default_allocator.h"
#include "unittest.h"
#include "vsag/dataset.h"
#include "vsag/engine.h"
TEST_CASE("Dataset Implement Test", "[ut][dataset]") {
    vsag::DefaultAllocator allocator;
    SECTION("allocator") {
        auto dataset = vsag::Dataset::Make();
        auto* data = static_cast<float*>(allocator.Allocate(sizeof(float) * 1));
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

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(1, 5);

    std::vector<uint32_t> vector_counts(num_elements);
    uint64_t total_vectors = 0;
    for (uint64_t i = 0; i < num_elements; ++i) {
        vector_counts[i] = dist(gen);
        total_vectors += vector_counts[i];
    }

    std::iota(ids.begin(), ids.end(), 0);
    base->Dim(dim)
        ->Ids(fixtures::CopyVector(ids, allocator))
        ->Paths(paths)
        ->AttributeSets(attr_sets)
        ->NumElements(num_elements)
        ->Distances(fixtures::CopyVector(distances, allocator))
        ->VectorCounts(fixtures::CopyVector(vector_counts, allocator))
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

    if (memcmp(data1->GetVectorCounts(),
               data2->GetVectorCounts(),
               sizeof(uint32_t) * num_element) != 0) {
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
AreAllPointersDifferent(T* original, T* copy, uint64_t num_elements) {
    for (uint64_t i = 0; i < num_elements; ++i) {
        if (std::memcmp(original + i, copy + i, sizeof(T)) == 0) {
            return false;
        }
    }
    return true;
}

bool
ArePathArraysDeepCopied(const std::string* original,
                        const std::string* copy,
                        uint64_t num_elements) {
    if (num_elements == 0) {
        return true;
    }
    if (original == nullptr || copy == nullptr || original == copy) {
        return false;
    }
    for (uint64_t i = 0; i < num_elements; ++i) {
        if (original[i] != copy[i]) {
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
            use_copy_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;
        auto copy = original->DeepCopy(copy_allocator.get());
        REQUIRE(EqualDataset(original, copy));
        REQUIRE(AreAllPointersDifferent(
            original->GetSparseVectors(), copy->GetSparseVectors(), num_elements));

        REQUIRE(AreAllPointersDifferent(
            original->GetAttributeSets(), copy->GetAttributeSets(), num_elements));

        REQUIRE(ArePathArraysDeepCopied(original->GetPaths(), copy->GetPaths(), num_elements));
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
            ->VectorCounts(original->GetVectorCounts() + num_elements)
            ->NumElements(append_num_elements)
            ->ExtraInfoSize(extra_info_size)
            ->ExtraInfos(original->GetExtraInfos() + num_elements * extra_info_size)
            ->Owner(false);
        REQUIRE(EqualDataset(sub_original, append_dataset));
    }
}

TEST_CASE("Dataset MultiVector Basic Test", "[ut][dataset]") {
    SECTION("MultiVectorDim default is 0") {
        auto dataset = vsag::Dataset::Make();
        REQUIRE(dataset->GetMultiVectorDim() == 0);
    }

    SECTION("MultiVectorDim setter/getter round-trip") {
        auto dataset = vsag::Dataset::Make();
        dataset->MultiVectorDim(128);
        REQUIRE(dataset->GetMultiVectorDim() == 128);
    }

    SECTION("MultiVectors getter returns nullptr by default") {
        auto dataset = vsag::Dataset::Make();
        REQUIRE(dataset->GetMultiVectors() == nullptr);
    }

    SECTION("MultiVectors setter/getter round-trip") {
        constexpr int64_t num_elements = 10;
        constexpr int64_t mv_dim = 16;

        auto* multi_vectors = new vsag::MultiVector[num_elements];
        for (int64_t i = 0; i < num_elements; ++i) {
            multi_vectors[i].len_ = 3;
            multi_vectors[i].vectors_ = new float[3 * mv_dim];
            for (int j = 0; j < 3 * mv_dim; ++j) {
                multi_vectors[i].vectors_[j] = static_cast<float>(i * 100 + j);
            }
        }

        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(num_elements)->MultiVectorDim(mv_dim)->MultiVectors(multi_vectors);

        REQUIRE(dataset->GetMultiVectors() == multi_vectors);
        REQUIRE(dataset->GetMultiVectors()[0].len_ == 3);
        REQUIRE(dataset->GetMultiVectors()[0].vectors_[0] == 0.0f);
        REQUIRE(dataset->GetMultiVectors()[1].vectors_[0] == 100.0f);

        dataset->Owner(true, nullptr);
    }

    SECTION("Owner(true) destructor releases MultiVector without crash") {
        constexpr int64_t num_elements = 5;
        constexpr int64_t mv_dim = 8;

        auto* multi_vectors = new vsag::MultiVector[num_elements];
        for (int64_t i = 0; i < num_elements; ++i) {
            multi_vectors[i].len_ = 2;
            multi_vectors[i].vectors_ = new float[2 * mv_dim];
        }
        multi_vectors[2].len_ = 0;
        delete[] multi_vectors[2].vectors_;
        multi_vectors[2].vectors_ = nullptr;

        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(num_elements)
            ->MultiVectorDim(mv_dim)
            ->MultiVectors(multi_vectors)
            ->Owner(true, nullptr);
    }

    SECTION("Owner(true) with allocator releases MultiVector without crash") {
        vsag::DefaultAllocator allocator;
        constexpr int64_t num_elements = 5;
        constexpr int64_t mv_dim = 8;

        auto* multi_vectors = static_cast<vsag::MultiVector*>(
            allocator.Allocate(num_elements * sizeof(vsag::MultiVector)));
        for (int64_t i = 0; i < num_elements; ++i) {
            multi_vectors[i].len_ = 2;
            multi_vectors[i].vectors_ =
                static_cast<float*>(allocator.Allocate(2 * mv_dim * sizeof(float)));
        }

        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(num_elements)
            ->MultiVectorDim(mv_dim)
            ->MultiVectors(multi_vectors)
            ->Owner(true, &allocator);
    }
}

TEST_CASE("Dataset MultiVector Append Test", "[ut][dataset]") {
    constexpr int64_t mv_dim = 16;

    auto use_allocator = GENERATE(true, false);
    std::shared_ptr<vsag::Allocator> allocator =
        use_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;

    auto make_multi_vector_dataset = [&](int64_t num_elements, uint32_t len_per_doc) {
        vsag::MultiVector* multi_vectors = nullptr;
        if (allocator) {
            multi_vectors = static_cast<vsag::MultiVector*>(
                allocator->Allocate(num_elements * sizeof(vsag::MultiVector)));
        } else {
            multi_vectors = new vsag::MultiVector[num_elements];
        }
        for (int64_t i = 0; i < num_elements; ++i) {
            multi_vectors[i].len_ = len_per_doc;
            uint64_t num_floats = static_cast<uint64_t>(len_per_doc) * mv_dim;
            if (allocator) {
                multi_vectors[i].vectors_ =
                    static_cast<float*>(allocator->Allocate(num_floats * sizeof(float)));
            } else {
                multi_vectors[i].vectors_ = new float[num_floats];
            }
            for (uint64_t j = 0; j < num_floats; ++j) {
                multi_vectors[i].vectors_[j] = static_cast<float>(i * 1000 + j);
            }
        }
        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(num_elements)
            ->Dim(mv_dim)
            ->MultiVectorDim(mv_dim)
            ->MultiVectors(multi_vectors)
            ->Owner(true, allocator.get());
        return dataset;
    };

    SECTION("Append preserves data") {
        constexpr int64_t n1 = 10;
        constexpr int64_t n2 = 7;
        constexpr uint32_t len = 3;

        auto ds1 = make_multi_vector_dataset(n1, len);
        auto ds2 = make_multi_vector_dataset(n2, len);

        const vsag::MultiVector* mv2_before = ds2->GetMultiVectors();
        std::vector<float> mv2_first_vec(mv_dim * len);
        std::memcpy(mv2_first_vec.data(), mv2_before[0].vectors_, mv_dim * len * sizeof(float));

        ds1->Append(ds2);

        REQUIRE(ds1->GetNumElements() == n1 + n2);
        REQUIRE(ds1->GetMultiVectors() != nullptr);

        const vsag::MultiVector* result_mv = ds1->GetMultiVectors();
        for (int64_t i = 0; i < n1; ++i) {
            REQUIRE(result_mv[i].len_ == len);
            REQUIRE(result_mv[i].vectors_[0] == static_cast<float>(i * 1000));
        }
        for (int64_t i = 0; i < n2; ++i) {
            REQUIRE(result_mv[n1 + i].len_ == len);
            REQUIRE(std::memcmp(result_mv[n1 + i].vectors_,
                                mv2_before[i].vectors_,
                                len * mv_dim * sizeof(float)) == 0);
        }
    }

    SECTION("Append with variable-length MultiVectors") {
        constexpr int64_t n1 = 5;
        constexpr int64_t n2 = 3;

        auto make_varlen_dataset = [&](int64_t num_elements, int seed) {
            std::mt19937 gen(seed);
            std::uniform_int_distribution<uint32_t> len_dist(1, 6);

            vsag::MultiVector* multi_vectors = nullptr;
            if (allocator) {
                multi_vectors = static_cast<vsag::MultiVector*>(
                    allocator->Allocate(num_elements * sizeof(vsag::MultiVector)));
            } else {
                multi_vectors = new vsag::MultiVector[num_elements];
            }
            for (int64_t i = 0; i < num_elements; ++i) {
                uint32_t len = len_dist(gen);
                multi_vectors[i].len_ = len;
                uint64_t num_floats = static_cast<uint64_t>(len) * mv_dim;
                if (allocator) {
                    multi_vectors[i].vectors_ =
                        static_cast<float*>(allocator->Allocate(num_floats * sizeof(float)));
                } else {
                    multi_vectors[i].vectors_ = new float[num_floats];
                }
                for (uint64_t j = 0; j < num_floats; ++j) {
                    multi_vectors[i].vectors_[j] = static_cast<float>(seed * 10000 + i * 100 + j);
                }
            }
            auto dataset = vsag::Dataset::Make();
            dataset->NumElements(num_elements)
                ->Dim(mv_dim)
                ->MultiVectorDim(mv_dim)
                ->MultiVectors(multi_vectors)
                ->Owner(true, allocator.get());
            return dataset;
        };

        auto ds1 = make_varlen_dataset(n1, 42);
        auto ds2 = make_varlen_dataset(n2, 99);

        std::vector<uint32_t> expected_lens(n1 + n2);
        for (int64_t i = 0; i < n1; ++i) {
            expected_lens[i] = ds1->GetMultiVectors()[i].len_;
        }
        for (int64_t i = 0; i < n2; ++i) {
            expected_lens[n1 + i] = ds2->GetMultiVectors()[i].len_;
        }

        ds1->Append(ds2);

        REQUIRE(ds1->GetNumElements() == n1 + n2);
        const vsag::MultiVector* result_mv = ds1->GetMultiVectors();
        for (int64_t i = 0; i < n1 + n2; ++i) {
            REQUIRE(result_mv[i].len_ == expected_lens[i]);
            REQUIRE(result_mv[i].vectors_ != nullptr);
        }
    }

    SECTION("Append mismatch MultiVectorDim throws") {
        auto ds1 = make_multi_vector_dataset(5, 2);

        auto* mv2 = new vsag::MultiVector[3];
        for (int i = 0; i < 3; ++i) {
            mv2[i].len_ = 2;
            mv2[i].vectors_ = new float[2 * 32];
        }
        auto ds2 = vsag::Dataset::Make();
        ds2->NumElements(3)->Dim(mv_dim)->MultiVectorDim(32)->MultiVectors(mv2)->Owner(true,
                                                                                       nullptr);

        REQUIRE_THROWS(ds1->Append(ds2));
    }

    SECTION("Append this has MultiVectors but other does not throws") {
        auto ds1 = make_multi_vector_dataset(5, 2);
        auto ds2 = vsag::Dataset::Make();
        ds2->NumElements(3)->Dim(mv_dim)->Owner(false);

        REQUIRE_THROWS(ds1->Append(ds2));
    }
}

TEST_CASE("Dataset MultiVector DeepCopy Test", "[ut][dataset]") {
    constexpr int64_t num_elements = 50;
    constexpr int64_t mv_dim = 32;

    bool use_allocator = GENERATE(true, false);
    std::shared_ptr<vsag::Allocator> allocator =
        use_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;

    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> len_dist(1, 8);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    vsag::MultiVector* multi_vectors = nullptr;
    if (allocator) {
        multi_vectors = static_cast<vsag::MultiVector*>(
            allocator->Allocate(num_elements * sizeof(vsag::MultiVector)));
    } else {
        multi_vectors = new vsag::MultiVector[num_elements];
    }

    for (int64_t i = 0; i < num_elements; ++i) {
        uint32_t len = len_dist(gen);
        multi_vectors[i].len_ = len;
        uint64_t num_floats = static_cast<uint64_t>(len) * mv_dim;
        if (allocator) {
            multi_vectors[i].vectors_ =
                static_cast<float*>(allocator->Allocate(num_floats * sizeof(float)));
        } else {
            multi_vectors[i].vectors_ = new float[num_floats];
        }
        for (uint64_t j = 0; j < num_floats; ++j) {
            multi_vectors[i].vectors_[j] = val_dist(gen);
        }
    }

    vsag::DatasetPtr dataset = vsag::Dataset::Make();
    dataset->NumElements(num_elements)
        ->Dim(mv_dim)
        ->MultiVectorDim(mv_dim)
        ->MultiVectors(multi_vectors)
        ->Owner(true, allocator.get());

    SECTION("DeepCopy preserves data") {
        bool use_copy_allocator = GENERATE(true, false);
        std::shared_ptr<vsag::Allocator> copy_allocator =
            use_copy_allocator ? vsag::Engine::CreateDefaultAllocator() : nullptr;

        vsag::DatasetPtr copy = dataset->DeepCopy(copy_allocator.get());

        REQUIRE(copy->GetNumElements() == num_elements);
        REQUIRE(copy->GetMultiVectorDim() == mv_dim);
        REQUIRE(copy->GetMultiVectors() != nullptr);
        REQUIRE(copy->GetMultiVectors() != dataset->GetMultiVectors());

        const vsag::MultiVector* src_mv = dataset->GetMultiVectors();
        const vsag::MultiVector* dst_mv = copy->GetMultiVectors();
        for (int64_t i = 0; i < num_elements; ++i) {
            REQUIRE(dst_mv[i].len_ == src_mv[i].len_);
            REQUIRE(dst_mv[i].vectors_ != src_mv[i].vectors_);
            uint64_t num_floats = static_cast<uint64_t>(src_mv[i].len_) * mv_dim;
            REQUIRE(std::memcmp(
                        dst_mv[i].vectors_, src_mv[i].vectors_, num_floats * sizeof(float)) == 0);
        }
    }

    SECTION("DeepCopy with zero-length MultiVector element") {
        multi_vectors[0].len_ = 0;
        if (allocator) {
            allocator->Deallocate(multi_vectors[0].vectors_);
        } else {
            delete[] multi_vectors[0].vectors_;
        }
        multi_vectors[0].vectors_ = nullptr;

        vsag::DatasetPtr copy = dataset->DeepCopy(nullptr);
        const vsag::MultiVector* dst_mv = copy->GetMultiVectors();
        REQUIRE(dst_mv[0].len_ == 0);
        REQUIRE(dst_mv[0].vectors_ == nullptr);
        REQUIRE(dst_mv[1].len_ == dataset->GetMultiVectors()[1].len_);
    }
}
