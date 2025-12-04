
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

#include "test_dataset.h"

#include <algorithm>
#include <cstring>
#include <functional>

#include "algorithm/pyramid.h"
#include "fixtures.h"
#include "simd/fp32_simd.h"

namespace fixtures {

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, int64_t> const& a,
               std::pair<float, int64_t> const& b) const noexcept {
        return a.first > b.first;
    }
};

using MaxHeap = std::priority_queue<std::pair<float, int64_t>,
                                    std::vector<std::pair<float, int64_t>>,
                                    CompareByFirst>;

bool
is_path_belong_to(const std::string& a, const std::string& b) {
    auto paths = vsag::split(a, '|');
    for (const auto& path : paths) {
        if (b.compare(0, path.size(), path) == 0) {
            return true;
        }
    }
    return false;
}

static TestDataset::DatasetPtr
GenerateRandomDataset(uint64_t dim,
                      uint64_t count,
                      std::string metric_str = "l2",
                      bool is_query = false,
                      uint64_t extra_info_size = 0,
                      std::string vector_type = "dense",
                      bool has_duplicate = false,
                      int64_t id_shift = 16) {
    auto base = vsag::Dataset::Make();
    bool need_normalize = (metric_str != "cosine");
    auto vecs =
        fixtures::generate_vectors(count, dim, need_normalize, fixtures::RandomValue(0, 564));
    auto vecs_int8 = fixtures::generate_int8_codes(count, dim, fixtures::RandomValue(0, 564));
    auto attr_sets = fixtures::generate_attributes(count);
    auto paths = new std::string[count];
    for (int i = 0; i < count; ++i) {
        paths[i] = create_random_string(!is_query);
    }
    std::vector<int64_t> ids(count);
    for (int64_t i = 0; i < count; ++i) {
        ids[i] = (i << id_shift);
    }
    base->Dim(dim)
        ->Ids(CopyVector(ids))
        ->Paths(paths)
        ->AttributeSets(attr_sets)
        ->NumElements(count)
        ->Owner(true);
    if (not has_duplicate) {
        base->Float32Vectors(CopyVector(vecs))->Int8Vectors(CopyVector(vecs_int8));
    } else {
        base->Float32Vectors(DuplicateCopyVector(vecs))
            ->Int8Vectors(DuplicateCopyVector(vecs_int8));
    }
    if (vector_type == "sparse") {
        if (not has_duplicate) {
            base->SparseVectors(CopyVector(GenerateSparseVectors(count, dim)));
        } else {
            base->SparseVectors(DuplicateCopyVector(GenerateSparseVectors(count, dim)));
        }
    }
    if (extra_info_size != 0) {
        auto extra_infos = fixtures::generate_extra_infos(count, extra_info_size);
        base->ExtraInfos(CopyVector(extra_infos));
        base->ExtraInfoSize(extra_info_size);
    }
    return base;
}

static TestDataset::DatasetPtr
GenerateNanRandomDataset(uint64_t dim, uint64_t count, std::string metric_str = "l2") {
    auto base = vsag::Dataset::Make();
    bool need_normalize = (metric_str != "cosine");

    std::vector<float> vecs =
        fixtures::generate_vectors(count, dim, need_normalize, fixtures::RandomValue(0, 564));
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_real_distribution real;
    for (int i = 0; i < count; ++i) {
        float r = real(g);
        if (r < 0.001) {
            vecs[i * dim] = std::numeric_limits<float>::quiet_NaN();
        } else if (r < 0.002) {
            for (int j = 0; j < dim; ++j) {
                vecs[i * dim + j] = 0.0F;
            }
        }
    }

    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 16);
    base->Dim(dim)
        ->Ids(CopyVector(ids))
        ->Float32Vectors(CopyVector(vecs))
        ->NumElements(count)
        ->Owner(true);
    return base;
}

static std::pair<float*, int64_t*>
CalDistanceFloatMetrix(const vsag::DatasetPtr query,
                       const vsag::DatasetPtr base,
                       const std::string& metric_str,
                       const std::string& vector_type = "dense") {
    uint64_t query_count = query->GetNumElements();
    uint64_t base_count = base->GetNumElements();

    auto* result = new float[query_count * base_count];
    auto* ids = new int64_t[query_count * base_count];
    auto dist_func = vsag::FP32ComputeL2Sqr;
    if (metric_str == "ip") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            return 1 - vsag::FP32ComputeIP(query, codes, dim);
        };
    } else if (metric_str == "cosine") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            auto norm_query = std::unique_ptr<float[]>(new float[dim]);
            auto norm_codes = std::unique_ptr<float[]>(new float[dim]);
            vsag::Normalize(query, norm_query.get(), dim);
            vsag::Normalize(codes, norm_codes.get(), dim);
            return 1 - vsag::FP32ComputeIP(norm_query.get(), norm_codes.get(), dim);
        };
    }
    auto dim = base->GetDim();
#pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < query_count; ++i) {
        MaxHeap heap;
        for (uint64_t j = 0; j < base_count; ++j) {
            float dist;
            if (vector_type == "dense") {
                dist = dist_func(
                    query->GetFloat32Vectors() + dim * i, base->GetFloat32Vectors() + dim * j, dim);
            } else if (vector_type == "sparse") {
                dist = GetSparseDistance(query->GetSparseVectors()[i], base->GetSparseVectors()[j]);
            } else {
                throw std::runtime_error("no such vector type");
            }
            heap.emplace(dist, base->GetIds()[j]);
        }
        auto idx = 0;
        while (not heap.empty()) {
            auto [dist, id] = heap.top();
            result[i * base_count + idx] = dist;
            ids[i * base_count + idx] = id;
            ++idx;
            heap.pop();
        }
    }
    return {result, ids};
}

static std::pair<float*, int64_t*>
CalDistanceFloatMetrixWithExFilter(const vsag::DatasetPtr query,
                                   const vsag::DatasetPtr base,
                                   const std::string& metric_str,
                                   std::function<bool(const char*)> filter,
                                   const std::string& vector_type = "dense") {
    uint64_t query_count = query->GetNumElements();
    uint64_t base_count = base->GetNumElements();
    auto extra_info_size = base->GetExtraInfoSize();

    auto* result = new float[query_count * base_count];
    auto* ids = new int64_t[query_count * base_count];
    auto dist_func = vsag::FP32ComputeL2Sqr;
    if (metric_str == "ip") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            return 1 - vsag::FP32ComputeIP(query, codes, dim);
        };
    } else if (metric_str == "cosine") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            auto norm_query = std::unique_ptr<float[]>(new float[dim]);
            auto norm_codes = std::unique_ptr<float[]>(new float[dim]);
            vsag::Normalize(query, norm_query.get(), dim);
            vsag::Normalize(codes, norm_codes.get(), dim);
            return 1 - vsag::FP32ComputeIP(norm_query.get(), norm_codes.get(), dim);
        };
    }
    auto dim = base->GetDim();
#pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < query_count; ++i) {
        MaxHeap heap;
        for (uint64_t j = 0; j < base_count; ++j) {
            float dist;
            const char* extra_info =
                extra_info_size > 0 ? (base->GetExtraInfos() + extra_info_size * j) : nullptr;
            if (vector_type == "dense") {
                dist = dist_func(
                    query->GetFloat32Vectors() + dim * i, base->GetFloat32Vectors() + dim * j, dim);
            } else if (vector_type == "sparse") {
                dist = GetSparseDistance(query->GetSparseVectors()[i], base->GetSparseVectors()[j]);
            } else {
                throw std::runtime_error("no such vector type");
            }
            if (extra_info_size == 0 || not filter(extra_info)) {
                heap.emplace(dist, base->GetIds()[j]);
            }
        }
        auto idx = 0;
        while (not heap.empty()) {
            auto [dist, id] = heap.top();
            result[i * base_count + idx] = dist;
            ids[i * base_count + idx] = id;
            ++idx;
            heap.pop();
        }
    }
    return {result, ids};
}

static vsag::DatasetPtr
CalTopKGroundTruth(const std::pair<float*, int64_t*>& result,
                   uint64_t top_k,
                   uint64_t base_count,
                   uint64_t query_count) {
    auto gt = vsag::Dataset::Make();
    auto* ids = new int64_t[query_count * top_k];
    auto* dists = new float[query_count * top_k];
    for (uint64_t i = 0; i < query_count; ++i) {
        for (int j = 0; j < top_k; ++j) {
            ids[i * top_k + j] = result.second[i * base_count + j];
            dists[i * top_k + j] = result.first[i * base_count + j];
        }
    }
    gt->Dim(top_k)->Ids(ids)->Distances(dists)->Owner(true)->NumElements(query_count);
    return gt;
}

static vsag::DatasetPtr
CalFilterGroundTruth(const std::pair<float*, int64_t*>& result,
                     uint64_t top_k,
                     std::function<bool(int64_t)> filter,
                     uint64_t base_count,
                     uint64_t query_count) {
    auto gt = vsag::Dataset::Make();
    auto* ids = new int64_t[query_count * top_k];
    auto* dists = new float[query_count * top_k];
    for (uint64_t i = 0; i < query_count; ++i) {
        auto start = 0;
        for (int j = 0; j < top_k; ++j) {
            while (start < base_count) {
                if (not filter(result.second[i * base_count + start])) {
                    ids[i * top_k + j] = result.second[i * base_count + start];
                    dists[i * top_k + j] = result.first[i * base_count + start];
                    ++start;
                    break;
                }
                ++start;
            }
        }
    }
    gt->Dim(top_k)->Ids(ids)->Distances(dists)->Owner(true)->NumElements(query_count);
    return gt;
}

static vsag::DatasetPtr
CalGroundTruthWithPath(const std::pair<float*, int64_t*>& result,
                       uint64_t top_k,
                       const vsag::DatasetPtr base,
                       const vsag::DatasetPtr query,
                       std::function<bool(int64_t)> filter = nullptr,
                       int64_t id_shift = 16) {
    auto base_count = base->GetNumElements();
    auto query_count = query->GetNumElements();
    auto base_paths = base->GetPaths();
    auto query_paths = query->GetPaths();
    auto gt = vsag::Dataset::Make();
    auto* ids = new int64_t[query_count * top_k];
    auto* dists = new float[query_count * top_k];
    for (uint64_t i = 0; i < query_count; ++i) {
        auto start = 0;
        for (int j = 0; j < top_k; ++j) {
            while (start < base_count) {
                auto base_id = result.second[i * base_count + start];
                if (is_path_belong_to(query_paths[i], base_paths[base_id >> id_shift]) &&
                    (not filter || not filter(base_id))) {
                    ids[i * top_k + j] = base_id;
                    dists[i * top_k + j] = result.first[i * base_count + start];
                    ++start;
                    break;
                }
                ++start;
            }
        }
    }
    gt->Dim(top_k)->Ids(ids)->Distances(dists)->Owner(true)->NumElements(query_count);
    return gt;
}

TestDatasetPtr
TestDataset::CreateTestDataset(uint64_t dim,
                               uint64_t count,
                               std::string metric_str,
                               bool with_path,
                               float valid_ratio,
                               std::string vector_type,
                               uint64_t extra_info_size,
                               bool has_duplicate,
                               int64_t id_shift) {
    TestDatasetPtr dataset = std::shared_ptr<TestDataset>(new TestDataset);
    dataset->dim_ = dim;
    dataset->id_shift = id_shift;
    dataset->count_ = count;
    dataset->base_ = GenerateRandomDataset(dim,
                                           count,
                                           metric_str,
                                           false /*is_query*/,
                                           extra_info_size,
                                           vector_type,
                                           has_duplicate,
                                           dataset->id_shift);
    constexpr uint64_t query_count = 100;
    dataset->query_ =
        GenerateRandomDataset(dim, query_count, metric_str, true, extra_info_size, vector_type);
    dataset->filter_query_ = dataset->query_;
    dataset->range_query_ = dataset->query_;
    dataset->valid_ratio_ = valid_ratio;
    {
        auto result =
            CalDistanceFloatMetrix(dataset->query_, dataset->base_, metric_str, vector_type);
        dataset->top_k = 10;

        dataset->filter_function_ =
            [valid_ratio, count, shift = dataset->id_shift](int64_t id) -> bool {
            return (id >> shift) > valid_ratio * count;
        };

        dataset->ex_filter_function_ = [valid_ratio, count](const char* data) -> bool {
            uint8_t abs = *data - INT8_MIN;
            return abs > UINT8_MAX * valid_ratio;
        };
        auto ex_result = CalDistanceFloatMetrixWithExFilter(
            dataset->query_, dataset->base_, metric_str, dataset->ex_filter_function_, vector_type);

        if (with_path) {
            dataset->ground_truth_ = CalGroundTruthWithPath(result,
                                                            dataset->top_k,
                                                            dataset->base_,
                                                            dataset->query_,
                                                            nullptr,
                                                            dataset->id_shift);
            dataset->filter_ground_truth_ = CalGroundTruthWithPath(result,
                                                                   dataset->top_k,
                                                                   dataset->base_,
                                                                   dataset->query_,
                                                                   dataset->filter_function_,
                                                                   dataset->id_shift);
            dataset->ex_filter_ground_truth_ = CalGroundTruthWithPath(ex_result,
                                                                      dataset->top_k,
                                                                      dataset->base_,
                                                                      dataset->query_,
                                                                      nullptr,
                                                                      dataset->id_shift);
        } else {
            dataset->ground_truth_ = CalTopKGroundTruth(result, dataset->top_k, count, query_count);
            dataset->filter_ground_truth_ = CalFilterGroundTruth(
                result, dataset->top_k, dataset->filter_function_, count, query_count);
            dataset->ex_filter_ground_truth_ =
                CalTopKGroundTruth(ex_result, dataset->top_k, count, query_count);
        }
        dataset->range_ground_truth_ = dataset->ground_truth_;
        dataset->range_radius_.resize(query_count);
        for (uint64_t i = 0; i < query_count; ++i) {
            dataset->range_radius_[i] =
                0.5f * (dataset->range_ground_truth_
                            ->GetDistances()[i * dataset->top_k + dataset->top_k - 1] +
                        dataset->range_ground_truth_
                            ->GetDistances()[i * dataset->top_k + dataset->top_k - 2]);
        }
        delete[] result.first;
        delete[] result.second;
        delete[] ex_result.first;
        delete[] ex_result.second;
    }
    return dataset;
}

TestDatasetPtr
TestDataset::CreateNanDataset(const std::string& metric_str) {
    TestDatasetPtr dataset = std::shared_ptr<TestDataset>(new TestDataset);
    dataset->dim_ = 64;
    dataset->count_ = 1000;
    constexpr uint64_t query_count = 100;
    dataset->base_ = GenerateNanRandomDataset(dataset->dim_, dataset->count_, metric_str);
    dataset->query_ = GenerateNanRandomDataset(dataset->dim_, query_count, metric_str);
    {
        auto result = CalDistanceFloatMetrix(dataset->query_, dataset->base_, metric_str);
        dataset->top_k = 10;
        dataset->ground_truth_ =
            CalTopKGroundTruth(result, dataset->top_k, dataset->count_, query_count);
        dataset->range_ground_truth_ = dataset->ground_truth_;
        dataset->range_radius_.resize(query_count);
        for (uint64_t i = 0; i < query_count; ++i) {
            dataset->range_radius_[i] =
                dataset->ground_truth_->GetDistances()[i * dataset->top_k + dataset->top_k - 1];
        }
        delete[] result.first;
        delete[] result.second;
    }
    return dataset;
}

}  // namespace fixtures
