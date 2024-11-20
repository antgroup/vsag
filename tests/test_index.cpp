
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

#include "test_index.h"

#include "simd/fp32_simd.h"

namespace fixtures {
static int64_t
Intersection(const int64_t* x, int64_t x_count, const int64_t* y, int64_t y_count) {
    std::unordered_set<int64_t> set_x(x, x + x_count);
    int result = 0;

    for (int i = 0; i < y_count; ++i) {
        if (set_x.count(y[i])) {
            ++result;
        }
    }

    return result;
}

TestIndex::IndexPtr
TestIndex::FastGeneralTest(const std::string& name,
                           const std::string& build_param,
                           const std::string& search_parameters,
                           const std::string& metric_type,
                           int64_t dim,
                           IndexStatus end_status) const {
    auto dataset = TestIndex::GenerateAndSetDataset<float>(dim, 1000);

    auto top_k = 10;
    auto range = 0.01f;

    vsag::IndexPtr index = nullptr;

    // Test Factory
    { index = TestFactory(name, build_param, true); }
    if (end_status == IndexStatus::Factory) {
        return index;
    }

    // Test Build
    { TestBuildIndex(index, dim); }
    if (end_status == IndexStatus::Build) {
        return index;
    }

    //    // Test CalcDistanceById;
    //    {
    //        TestCalcDistanceById(index, dataset, metric_type);
    //    }

    // Test KnnSearch and RangeSearch
    {
        TestKnnSearch(index, dataset, search_parameters, top_k, 0.99);
        TestRangeSearch(index, dataset, search_parameters, range, 0.99);
    }

    // Serialize & Deserialize
    {
        fixtures::TempDir dir("serialize");
        auto filename = dir.GenerateRandomFile();
        TestSerializeFile(index, filename, true);

        auto another_index = TestDeserializeFile(filename, name, build_param, true);

        TestKnnSearch(another_index, dataset, search_parameters, top_k, 0.99);
        TestRangeSearch(another_index, dataset, search_parameters, range, 0.99);
        TestCalcDistanceById(another_index, dataset, metric_type);
        if (end_status == IndexStatus::DeSerialize) {
            return another_index;
        }
    }
    return index;
}

TestIndex::IndexPtr
TestIndex::TestFactory(const std::string& name,
                       const std::string& build_param,
                       bool expect_success) {
    auto created_index = vsag::Factory::CreateIndex(name, build_param);
    REQUIRE(created_index.has_value() == expect_success);
    return created_index.value();
}

void
TestIndex::TestBuildIndex(IndexPtr index, int64_t dim, bool expected_success) const {
    auto dataset = GenerateAndSetDataset<float>(dim, dataset_base_count)->base_;
    TestBuildIndex(index, dataset, expected_success);
}

void
TestIndex::TestBuildIndex(IndexPtr index, DatasetPtr dataset, bool expected_success) {
    auto build_index = index->Build(dataset);
    if (expected_success) {
        REQUIRE(build_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->GetNumElements());
    } else {
        REQUIRE(build_index.has_value() == expected_success);
    }
}

void
TestIndex::TestAddIndex(IndexPtr index, int64_t dim, bool expected_success) const {
    auto dataset = GenerateAndSetDataset<float>(dim, dataset_base_count)->base_;
    TestAddIndex(index, dataset, expected_success);
}

void
TestIndex::TestAddIndex(IndexPtr index, DatasetPtr dataset, bool expected_success) {
    auto add_index = index->Add(dataset);
    if (expected_success) {
        REQUIRE(add_index.has_value());
        // check the number of vectors in index
        REQUIRE(index->GetNumElements() == dataset->GetNumElements());
    } else {
        REQUIRE(not add_index.has_value());
    }
}

void
TestIndex::TestContinueAdd(IndexPtr index, int64_t dim, int64_t count, bool expected_success) {
    auto cur_count = index->GetNumElements();
    for (auto i = 0; i < count; ++i) {
        auto one_vector = fixtures::generate_one_dataset(dim, 1);
        delete[] one_vector->GetIds();
        auto ids = new int64_t[1];
        ids[0] = i + cur_count;
        one_vector->Ids(ids);
        if (expected_success) {
            REQUIRE(index->Add(one_vector).has_value());
        }
    }
    if (expected_success) {
        REQUIRE(index->GetNumElements() == cur_count + count);
    }
}

void
TestIndex::TestKnnSearch(IndexPtr index,
                         std::shared_ptr<fixtures::TestDataset> dataset,
                         const std::string& search_param,
                         int topk,
                         float recall,
                         bool expected_success) {
    auto querys = dataset->query_;
    auto query_count = querys->GetNumElements();
    auto dim = querys->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = gts->GetDim();
    float cur_recall = 0.0f;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(querys->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->KnnSearch(query, topk, search_param);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        REQUIRE(res.value()->GetDim() == topk);
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, topk);
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        auto has_func_cal_dis_by_id =
            index->CalcDistanceById(query->GetFloat32Vectors(), result[0]);
        if (has_func_cal_dis_by_id.has_value()) {
            for (int j = 0; j < topk; ++j) {
                REQUIRE(index->CalcDistanceById(query->GetFloat32Vectors(), result[j]) ==
                        res.value()->GetDistances()[j]);
            }
        }
    }
    REQUIRE(cur_recall > recall * query_count);
}

void
TestIndex::TestRangeSearch(IndexPtr index,
                           std::shared_ptr<fixtures::TestDataset> dataset,
                           const std::string& search_param,
                           float radius,
                           float recall,
                           int64_t limited_size,
                           bool expected_success) {
    auto querys = dataset->query_;
    auto query_count = querys->GetNumElements();
    auto dim = querys->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = gts->GetDim();
    float cur_recall = 0.0f;
    for (auto i = 0; i < query_count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(dim)
            ->Float32Vectors(querys->GetFloat32Vectors() + i * dim)
            ->Owner(false);
        auto res = index->RangeSearch(query, radius, search_param, limited_size);
        REQUIRE(res.has_value() == expected_success);
        if (!expected_success) {
            return;
        }
        if (limited_size > 0) {
            REQUIRE(res.value()->GetDim() <= limited_size);
        }
        auto result = res.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result, res.value()->GetDim());
        cur_recall += static_cast<float>(val) / static_cast<float>(gt_topK);
        auto has_func_cal_dis_by_id =
            index->CalcDistanceById(query->GetFloat32Vectors(), result[0]);
        if (has_func_cal_dis_by_id.has_value()) {
            for (int j = 0; j < res.value()->GetDim(); ++j) {
                REQUIRE(index->CalcDistanceById(query->GetFloat32Vectors(), result[j]) ==
                        res.value()->GetDistances()[j]);
            }
        }
    }
    REQUIRE(cur_recall > recall * query_count);
}

void
TestIndex::TestCalcDistanceById(IndexPtr index,
                                std::shared_ptr<fixtures::TestDataset> dataset,
                                const std::string& metric_str,
                                float error) {
    int64_t dim = dataset->base_->GetDim();
    auto query = fixtures::generate_one_dataset(dim, 1);
    for (int j = 0; j < 10; ++j) {
        auto id = fixtures::RandomValue<int>(0, 999);

        auto result = index->CalcDistanceById(query->GetFloat32Vectors(), id);
        const auto data = dataset->base_->GetFloat32Vectors() + dim * id;
        float score = 0;
        if (metric_str == std::string("l2")) {
            score = vsag::FP32ComputeL2Sqr(query->GetFloat32Vectors(), data, dim);
        } else if (metric_str == std::string("ip")) {
            score = 1 - vsag::FP32ComputeIP(query->GetFloat32Vectors(), data, dim);
        } else if (metric_str == std::string("cosine")) {
            float mold_query =
                vsag::FP32ComputeIP(query->GetFloat32Vectors(), query->GetFloat32Vectors(), dim);
            float mold_base = vsag::FP32ComputeIP(data, data, dim);
            score = 1 - vsag::FP32ComputeIP(query->GetFloat32Vectors(), data, dim) /
                            std::sqrt(mold_query * mold_base);
        }
        float return_score = result.value();
        REQUIRE(std::abs(return_score - score) < error);
    }
}
void
TestIndex::TestSerializeFile(TestIndex::IndexPtr index,
                             const std::string& path,
                             bool expected_success) {
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    auto serialize_index = index->Serialize(outfile);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();
}

TestIndex::IndexPtr
TestIndex::TestDeserializeFile(const std::string& path,
                               const std::string& name,
                               const std::string& param,
                               bool expected_success) {
    auto another_index = TestFactory(name, param, true);
    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = another_index->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();
    return another_index;
}

template <typename T>
static T*
CopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

TestDatasetPtr
TestIndex::GenerateDatasetFloat(int64_t dim, uint64_t count) {
    auto base = vsag::Dataset::Make();
    auto [ids, vectors] = generate_ids_and_vectors(count, dim, true, time(nullptr));
    base->Dim(dim)
        ->NumElements(count)
        ->Float32Vectors(CopyVector(vectors))
        ->Ids(CopyVector(ids))
        ->Owner(true);
    int64_t query_count = count / 10;
    auto start = random() % (count - query_count);
    auto query = vsag::Dataset::Make();
    query->Dim(dim)
        ->NumElements(query_count)
        ->Float32Vectors(base->GetFloat32Vectors() + start * dim)
        ->Ids(base->GetIds() + start)
        ->Owner(false);
    auto gt = vsag::Dataset::Make();  // TODO(LHT) brute_force
    gt->Dim(1)->NumElements(query_count)->Ids(base->GetIds() + start)->Owner(false);
    auto dataset = std::make_shared<TestDataset>(base, query, gt);
    return dataset;
}

}  // namespace fixtures