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

#include "ground_truth.h"

#include <cassert>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>

#include "simd/simd.h"
#include "vsag/dataset.h"

namespace fixtures {

vsag::DatasetPtr
brute_force(const vsag::DatasetPtr& query,
            const vsag::DatasetPtr& base,
            int64_t k,
            const std::string& metric_type) {
    assert(metric_type == "l2");
    assert(query->GetDim() == base->GetDim());
    assert(query->GetNumElements() == 1);

    auto result = vsag::Dataset::Make();
    auto* ids = new int64_t[k];
    auto* dists = new float[k];
    result->Ids(ids)->Distances(dists)->NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    uint64_t dim = query->GetDim();
    for (uint32_t i = 0; i < base->GetNumElements(); i++) {
        float dist = vsag::L2Sqr(
            query->GetFloat32Vectors(), base->GetFloat32Vectors() + i * base->GetDim(), &dim);
        if (bf_result.size() < static_cast<uint64_t>(k)) {
            bf_result.emplace(dist, base->GetIds()[i]);
        } else {
            if (dist < bf_result.top().first) {
                bf_result.pop();
                bf_result.emplace(dist, base->GetIds()[i]);
            }
        }
    }

    for (int i = k - 1; i >= 0; i--) {
        ids[i] = bf_result.top().second;
        dists[i] = bf_result.top().first;
        bf_result.pop();
    }

    return std::move(result);
}

vsag::DatasetPtr
brute_force(const vsag::DatasetPtr& query,
            const vsag::DatasetPtr& base,
            int64_t k,
            const std::string& metric_type,
            const std::string& data_type) {
    assert(query->GetDim() == base->GetDim());
    assert(query->GetNumElements() == 1);

    auto result = vsag::Dataset::Make();
    auto* ids = new int64_t[k];
    auto* dists = new float[k];
    result->Ids(ids)->Distances(dists)->NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    uint64_t dim = query->GetDim();
    const void* query_vec = nullptr;
    const void* base_vec = nullptr;
    float dist = 0;
    for (uint32_t i = 0; i < base->GetNumElements(); i++) {
        if (data_type == "float32") {
            query_vec = query->GetFloat32Vectors();
            base_vec = base->GetFloat32Vectors() + i * base->GetDim();
        } else if (data_type == "int8") {
            query_vec = query->GetInt8Vectors();
            base_vec = base->GetInt8Vectors() + i * base->GetDim();
        } else {
            throw std::runtime_error("un-support data type");
        }

        if (metric_type == "l2") {
            dist = vsag::L2Sqr(query_vec, base_vec, &dim);
        } else if (metric_type == "ip") {
            if (data_type == "float32") {
                dist = vsag::InnerProductDistance(query_vec, base_vec, &dim);
            } else {
                dist = vsag::INT8InnerProductDistance(query_vec, base_vec, &dim);
            }
        }

        if (bf_result.size() < static_cast<uint64_t>(k)) {
            bf_result.emplace(dist, base->GetIds()[i]);
        } else {
            if (dist < bf_result.top().first) {
                bf_result.pop();
                bf_result.emplace(dist, base->GetIds()[i]);
            }
        }
    }

    for (int i = k - 1; i >= 0; i--) {
        ids[i] = bf_result.top().second;
        dists[i] = bf_result.top().first;
        bf_result.pop();
    }

    return std::move(result);
}

}  // namespace fixtures
