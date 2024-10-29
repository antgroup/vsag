
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

#include "fixtures.h"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>

#include "fmt/format.h"
#include "test_dataset.h"
#include "vsag/dataset.h"

namespace vsag {

extern float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);

extern float
INT8InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);
}  // namespace vsag

namespace fixtures {

void
normalize(float* input_vector, int64_t dim) {
    float magnitude = std::numeric_limits<float>::min();
    for (int64_t i = 0; i < dim; ++i) {
        magnitude += input_vector[i] * input_vector[i];
    }
    magnitude = std::sqrt(magnitude);

    for (int64_t i = 0; i < dim; ++i) {
        input_vector[i] = input_vector[i] / magnitude;
    }
}

std::vector<int>
get_common_used_dims(uint64_t count, int seed) {
    const std::vector<int> dims = {
        1,    8,    9,      // generic (dim < 32)
        32,   33,   48,     // sse(32) + generic(dim < 16)
        64,   65,   70,     // avx(64) + generic(dim < 16)
        96,   97,   109,    // avx(64) + sse(32) + generic(dim < 16)
        128,  129,          // avx512(128) + generic(dim < 16)
        160,  161,          // avx512(128) + sse(32) + generic(dim < 16)
        192,  193,          // avx512(128) + avx(64) + generic(dim < 16)
        224,  225,          // avx512(128) + avx(64) + sse(32) + generic(dim < 16)
        256,  512,          // common used dims
        784,  960,          // common used dims
        1024, 1536, 2048};  // common used dims
    if (count == -1 || count >= dims.size()) {
        return dims;
    }
    std::vector<int> result(dims.begin(), dims.end());
    std::shuffle(result.begin(), result.end(), std::mt19937(seed));
    result.resize(count);
    return result;
}

std::vector<float>
generate_vectors(int64_t num_vectors, int64_t dim, bool need_normalize, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<> distrib_real;
    std::vector<float> vectors(dim * num_vectors);
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    if (need_normalize) {
        for (int64_t i = 0; i < num_vectors; ++i) {
            normalize(vectors.data() + i * dim, dim);
        }
    }

    return vectors;
}

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed) {
    auto code_size = dim;
    std::vector<int8_t> codes(count * code_size, 0);
    auto vec = fixtures::generate_vectors(count, dim, true, seed);

    for (int i = 0; i < count; i++) {
        for (int d = 0; d < dim; d++) {
            float delta = vec[d + i * dim];
            if (delta < 0) {
                delta = 0;
            } else if (delta > 0.999) {
                delta = 1;
            }
            // Note that here we use overflowed uint8 value to obtain int8 value
            codes[code_size * i + d] = 255.0 * delta;
        }
    }
    return codes;
}

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed) {
    auto code_size = (dim + 1) / 2;
    std::vector<uint8_t> codes(count * code_size, 0);
    auto vec = fixtures::generate_vectors(count, dim, true, seed);

    for (int i = 0; i < count; i++) {
        auto pos = code_size * i;

        for (int d = 0; d < dim; d++) {
            float delta = vec[d + i * dim];
            if (delta < 0) {
                delta = 0;
            } else if (delta > 0.999) {
                delta = 1;
            }
            uint8_t scaled = 15.0 * delta;

            if (d & 1) {
                codes[pos + (d >> 1)] |= scaled << 4;
            } else {
                codes[pos + (d >> 1)] = 0;
                codes[pos + (d >> 1)] |= scaled;
            }
        }
    }
    return codes;
}

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_vectors, int64_t dim, bool need_normalize, int seed) {
    std::vector<int64_t> ids(num_vectors);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }

    return {ids, generate_vectors(num_vectors, dim, need_normalize, seed)};
}

vsag::IndexPtr
generate_index(const std::string& name,
               const std::string& metric_type,
               int64_t num_vectors,
               int64_t dim,
               std::vector<int64_t>& ids,
               std::vector<float>& vectors,
               bool use_conjugate_graph) {
    auto index = vsag::Factory::CreateIndex(name,
                                            vsag::generate_build_parameters(
                                                metric_type, num_vectors, dim, use_conjugate_graph)
                                                .value())
                     .value();

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    if (not index->Build(base).has_value()) {
        return nullptr;
    }

    return index;
}

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors) {
    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
        auto result = index->KnnSearch(query, 10, search_parameters).value();
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            if (ids[i] == result->GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    return recall;
}

float
test_range_recall(const vsag::IndexPtr& index,
                  const std::string& search_parameters,
                  int64_t num_vectors,
                  int64_t dim,
                  std::vector<int64_t>& ids,
                  std::vector<float>& vectors) {
    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(vectors.data() + i * dim)->Owner(false);
        auto result = index->RangeSearch(query, 0, search_parameters).value();
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            if (ids[i] == result->GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    return recall;
}

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 64,
            "ef_construction": 500
        }}
    }}
    )";
    auto build_parameters = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters;
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
    int64_t* ids = new int64_t[k];
    float* dists = new float[k];
    result->Ids(ids)->Distances(dists)->NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    size_t dim = query->GetDim();
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

        if (bf_result.size() < k) {
            bf_result.push({dist, base->GetIds()[i]});
        } else {
            if (dist < bf_result.top().first) {
                bf_result.pop();
                bf_result.push({dist, base->GetIds()[i]});
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
            const std::string& metric_type) {
    assert(metric_type == "l2");
    assert(query->GetDim() == base->GetDim());
    assert(query->GetNumElements() == 1);

    auto result = vsag::Dataset::Make();
    int64_t* ids = new int64_t[k];
    float* dists = new float[k];
    result->Ids(ids)->Distances(dists)->NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    size_t dim = query->GetDim();
    for (uint32_t i = 0; i < base->GetNumElements(); i++) {
        float dist = vsag::L2Sqr(
            query->GetFloat32Vectors(), base->GetFloat32Vectors() + i * base->GetDim(), &dim);
        if (bf_result.size() < k) {
            bf_result.push({dist, base->GetIds()[i]});
        } else {
            if (dist < bf_result.top().first) {
                bf_result.pop();
                bf_result.push({dist, base->GetIds()[i]});
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

std::vector<IOItem>
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index) {
    std::vector<IOItem> result(count);
    std::unordered_set<uint64_t> maps;
    for (auto& item : result) {
        while (true) {
            item.start_ = (random() % max_index) * max_length;
            if (not maps.count(item.start_)) {
                maps.insert(item.start_);
                break;
            }
        };
        item.length_ = random() % max_length;
        item.data_ = new uint8_t[item.length_];
        auto vec = fixtures::generate_vectors(1, max_length, false, random());
        memcpy(item.data_, vec.data(), item.length_);
    }
    return result;
}

template <typename T>
static T*
CopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

vsag::DatasetPtr
generate_one_dataset(int64_t dim, uint64_t count) {
    auto result = vsag::Dataset::Make();
    auto [ids, vectors] = generate_ids_and_vectors(count, dim, true, time(nullptr));
    result->Dim(dim)
        ->NumElements(count)
        ->Float32Vectors(CopyVector(vectors))
        ->Ids(CopyVector(ids))
        ->Owner(true);
    return result;
}

uint64_t
GetFileSize(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    return static_cast<uint64_t>(file.tellg());
}

}  // namespace fixtures
