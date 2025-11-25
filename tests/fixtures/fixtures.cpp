
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

#include <unistd.h>

#include <cstdint>
#include <random>
#include <string>
#include <unordered_set>

#include "fmt/format.h"
#include "simd/simd.h"
#include "vsag/dataset.h"

namespace fixtures {

const int RABITQ_MIN_RACALL_DIM = 960;

static std::vector<int>
select_dims(const std::vector<int>& dims, uint64_t count, int seed, int limited_dim) {
    if (count == -1 || count >= dims.size()) {
        return dims;
    }
    if (limited_dim > 0) {
        // find dim less than or equal to limited_dim (binary search)
        auto it = std::upper_bound(dims.begin(), dims.end(), limited_dim);
        if (it != dims.begin()) {
            std::vector<int> result(dims.begin(), it);
            if (result.size() < count) {
                return result;
            } else {
                std::shuffle(result.begin(), result.end(), std::mt19937(seed));
                result.resize(count);
                return result;
            }
        }
    }
    std::vector<int> result(dims.begin(), dims.end());
    std::shuffle(result.begin(), result.end(), std::mt19937(seed));
    result.resize(count);
    return result;
}
std::vector<int>
get_common_used_dims(uint64_t count, int seed, int limited_dim) {
    const std::vector<int> dims = {
        7,    8,   9,    // generic (dim < 32)
        32,   33,  48,   // sse(32) + generic(dim < 16)
        64,   65,  70,   // avx(64) + generic(dim < 16)
        96,   97,  109,  // avx(64) + sse(32) + generic(dim < 16)
        128,  129,       // avx512(128) + generic(dim < 16)
        160,  161,       // avx512(128) + sse(32) + generic(dim < 16)
        192,  193,       // avx512(128) + avx(64) + generic(dim < 16)
        224,  225,       // avx512(128) + avx(64) + sse(32) + generic(dim < 16)
        256,  512,       // common used dims
        784,  960,       // common used dims
        1024, 1536};     // common used dims
    return select_dims(dims, count, seed, limited_dim);
}

std::vector<int>
get_index_test_dims(uint64_t count, int seed, int limited_dim) {
    const std::vector<int> dims = {32, 57, 128, 256, 768, 1536};
    return select_dims(dims, count, seed, limited_dim);
}

std::vector<vsag::SparseVector>
GenerateSparseVectors(
    uint32_t count, uint32_t max_dim, uint32_t max_id, float min_val, float max_val, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(min_val, max_val);
    std::uniform_int_distribution<int> distrib_dim(max_dim / 2, max_dim);
    std::uniform_int_distribution<int> distrib_id(0, max_id);

    std::vector<vsag::SparseVector> sparse_vectors(count);
    if (max_dim > max_id) {
        throw std::runtime_error("generate sparse vectors failed, max_dim > max_id");
    }

    for (int i = 0; i < count; i++) {
        sparse_vectors[i].len_ = distrib_dim(rng);
        sparse_vectors[i].ids_ = new uint32_t[sparse_vectors[i].len_];
        sparse_vectors[i].vals_ = new float[sparse_vectors[i].len_];
        std::unordered_set<uint32_t> unique_ids;
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            auto u_id = distrib_id(rng);
            while (unique_ids.count(u_id) > 0) {
                u_id = distrib_id(rng);
            }
            unique_ids.insert(u_id);
            sparse_vectors[i].ids_[d] = u_id;
            sparse_vectors[i].vals_[d] = distrib_real(rng);
        }
    }

    return sparse_vectors;
}

vsag::Vector<vsag::SparseVector>
GenerateSparseVectors(vsag::Allocator* allocator,
                      uint32_t count,
                      uint32_t max_dim,
                      uint32_t max_id,
                      float min_val,
                      float max_val,
                      int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(min_val, max_val);
    std::uniform_int_distribution<int> distrib_dim(0, max_dim);
    std::uniform_int_distribution<int> distrib_id(0, max_id);

    vsag::Vector<vsag::SparseVector> sparse_vectors(count, allocator);

    for (int i = 0; i < count; i++) {
        sparse_vectors[i].len_ = distrib_dim(rng);
        sparse_vectors[i].ids_ =
            (uint32_t*)allocator->Allocate(sizeof(uint32_t) * sparse_vectors[i].len_);
        sparse_vectors[i].vals_ =
            (float*)allocator->Allocate(sizeof(float) * sparse_vectors[i].len_);
        for (int d = 0; d < sparse_vectors[i].len_; d++) {
            sparse_vectors[i].ids_[d] = distrib_id(rng);
            sparse_vectors[i].vals_[d] = distrib_real(rng);
        }
    }

    return sparse_vectors;
}

std::pair<std::vector<float>, std::vector<uint8_t>>
GenerateBinaryVectorsAndCodes(uint32_t count, uint32_t dim, int seed) {
    assert(count % 2 == 0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib_real(-1, 1);
    float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(dim));

    uint32_t code_size = (dim + 7) / 8;
    std::vector<uint8_t> codes(count * code_size);
    std::vector<float> vectors(count * dim);

    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            if (distrib_real(rng) >= 0.0f) {
                codes[i * code_size + d / 8] |= (1 << (d % 8));
                vectors[i * dim + d] = inv_sqrt_d;
            } else {
                vectors[i * dim + d] = -inv_sqrt_d;
            }
        }
    }

    return {vectors, codes};
}

std::string
create_random_string(bool is_full) {
    const std::vector<std::string> level1 = {"a", "b", "c"};
    const std::vector<std::string> level2 = {"d", "e"};
    const std::vector<std::string> level3 = {"f", "g", "h"};

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<> distr;

    std::vector<std::string> selected_levels;

    if (is_full) {
        selected_levels.push_back(level1[distr(mt) % level1.size()]);
        selected_levels.push_back(level2[distr(mt) % level2.size()]);
        selected_levels.push_back(level3[distr(mt) % level3.size()]);
        return fmt::to_string(fmt::join(selected_levels, "/"));
    } else {
        std::uniform_int_distribution<> dist(1, 3);
        int num_path = dist(mt);
        std::vector<std::string> paths;
        for (int i = 0; i < num_path; ++i) {
            selected_levels.clear();
            int num_levels = dist(mt);
            if (num_levels >= 1) {
                selected_levels.push_back(level1[distr(mt) % level1.size()]);
            }
            if (num_levels >= 2) {
                selected_levels.push_back(level2[distr(mt) % level2.size()]);
            }
            if (num_levels == 3) {
                selected_levels.push_back(level3[distr(mt) % level3.size()]);
            }
            paths.push_back(fmt::to_string(fmt::join(selected_levels, "/")));
        }
        return fmt::to_string(fmt::join(paths, "|"));
    }
}

std::vector<float>
generate_vectors(uint64_t count, uint32_t dim, bool need_normalize, int seed) {
    return std::move(GenerateVectors<float>(count, dim, seed, need_normalize));
}

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed) {
    return GenerateVectors<int8_t>(count, dim, seed);
}

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed) {
    return generate_uint8_codes(count, dim, seed);
}

std::vector<uint8_t>
generate_uint8_codes(uint64_t count, uint32_t dim, int seed) {
    return GenerateVectors<uint8_t>(count, dim, seed);
}

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_vectors, int64_t dim, bool need_normalize, int seed) {
    std::vector<int64_t> ids(num_vectors);
    std::iota(ids.begin(), ids.end(), 0);
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

std::vector<char>
generate_extra_infos(uint64_t count, uint32_t size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int8_t> distrib_real(std::numeric_limits<int8_t>::lowest(),
                                                       std::numeric_limits<int8_t>::max());
    std::vector<char> vectors(size * count);
    for (int64_t i = 0; i < size * count; ++i) {
        vectors[i] = static_cast<char>(distrib_real(rng));
    }
    return vectors;
}

template <typename T, typename Gen>
void
FillIntegerValues(vsag::AttributeValue<T>* attr, uint32_t count, Gen& gen) {
    using Limits = std::numeric_limits<T>;
    std::uniform_int_distribution<int64_t> dist(Limits::min(), Limits::max());
    for (uint32_t i = 0; i < count; ++i) {
        attr->GetValue().emplace_back(static_cast<T>(dist(gen)));
    }
}

template <typename Gen>
void
FillStringValues(vsag::AttributeValue<std::string>* attr,
                 uint32_t count,
                 uint32_t max_len,
                 Gen& gen) {
    std::uniform_int_distribution<uint32_t> len_dist(1, max_len);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = len_dist(gen);
        std::string str;
        str.reserve(len);
        for (uint32_t c = 0; c < len; ++c) {
            str += char_dist(gen);
        }
        attr->GetValue().emplace_back(str);
    }
}

static std::vector<int>
select_k_numbers(int64_t n, int k) {
    std::vector<int> numbers(n);
    std::iota(numbers.begin(), numbers.end(), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<> dist(i, static_cast<int>(n - 1));
        std::swap(numbers[i], numbers[dist(gen)]);
    }
    numbers.resize(k);
    return numbers;
}

template <typename Gen>
vsag::Attribute*
CreateAttribute(std::string term_name,
                vsag::AttrValueType type,
                uint32_t term_count,
                uint32_t max_str_len,
                Gen& gen) {
    switch (type) {
        case vsag::AttrValueType::INT32: {
            auto attr = new vsag::AttributeValue<int32_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT32: {
            auto attr = new vsag::AttributeValue<uint32_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT64: {
            auto attr = new vsag::AttributeValue<int64_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT64: {
            auto attr = new vsag::AttributeValue<uint64_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT8: {
            auto attr = new vsag::AttributeValue<int8_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT8: {
            auto attr = new vsag::AttributeValue<uint8_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::INT16: {
            auto attr = new vsag::AttributeValue<int16_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::UINT16: {
            auto attr = new vsag::AttributeValue<uint16_t>();
            attr->name_ = term_name;
            FillIntegerValues(attr, term_count, gen);
            return attr;
        }
        case vsag::AttrValueType::STRING: {
            auto attr = new vsag::AttributeValue<std::string>();
            attr->name_ = term_name;
            FillStringValues(attr, term_count, max_str_len, gen);
            return attr;
        }
        default:
            return nullptr;
    }
}

vsag::AttributeSet*
generate_attributes(uint64_t count, uint32_t max_term_count, uint32_t max_value_count, int seed) {
    auto* results = new vsag::AttributeSet[count];
    thread_local std::mt19937 gen(seed);
    std::uniform_int_distribution<int> term_count_dist(1, max_term_count);
    auto term_count = term_count_dist(gen);
    std::vector<std::pair<std::string, vsag::AttrValueType>> terms(term_count);

    std::uniform_int_distribution<int> type_dist(1, 9);
    std::uniform_int_distribution<int> value_count_dist(1, max_value_count);

    for (uint64_t i = 0; i < term_count; ++i) {
        std::string term_name = fmt::format("term_{}", i);
        auto term_type = static_cast<vsag::AttrValueType>(type_dist(gen));
        if (term_type == vsag::AttrValueType::UINT64) {
            term_type = vsag::AttrValueType::INT64;
        }
        terms[i] = {term_name, term_type};
    }

    for (uint32_t i = 0; i < count; ++i) {
        auto cur_term_count = RandomValue(1, term_count);
        results[i].attrs_.reserve(cur_term_count);
        auto idxes = select_k_numbers(term_count, cur_term_count);

        for (uint32_t j = 0; j < cur_term_count; ++j) {
            auto term_id = idxes[j];
            auto& term_name = terms[term_id].first;
            auto& term_type = terms[term_id].second;
            auto attr = CreateAttribute(term_name, term_type, value_count_dist(gen), 10, gen);
            results[i].attrs_.emplace_back(attr);
        }
    }
    return results;
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
    auto build_parameters_str = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters_str;
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
            const std::string& metric_type) {
    assert(metric_type == "l2");
    assert(query->GetDim() == base->GetDim());
    assert(query->GetNumElements() == 1);

    auto result = vsag::Dataset::Make();
    auto* ids = new int64_t[k];
    auto* dists = new float[k];
    result->Ids(ids)->Distances(dists)->NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    size_t dim = query->GetDim();
    for (uint32_t i = 0; i < base->GetNumElements(); i++) {
        float dist = vsag::L2Sqr(
            query->GetFloat32Vectors(), base->GetFloat32Vectors() + i * base->GetDim(), &dim);
        if (bf_result.size() < k) {
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
        item.length_ = random() % max_length + 1;
        item.data_ = new uint8_t[item.length_];
        auto vec = fixtures::generate_vectors(1, max_length, false, random());
        memcpy(item.data_, vec.data(), item.length_);
    }
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

std::vector<std::string>
SplitString(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(s);

    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }

    return tokens;
}

float
GetSparseDistance(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2) {
    if (vec1.len_ == 0 || vec2.len_ == 0) {
        return 0.0f;
    }

    std::unordered_map<uint32_t, float> id_to_val;
    for (uint32_t i = 0; i < vec1.len_; ++i) {
        id_to_val[vec1.ids_[i]] = vec1.vals_[i];
    }

    float distance = 0.0f;
    for (uint32_t i = 0; i < vec2.len_; ++i) {
        const auto& id = vec2.ids_[i];
        auto it = id_to_val.find(id);
        if (it != id_to_val.end()) {
            distance += it->second * vec2.vals_[i];
        }
    }
    return 1 - distance;
}

}  // namespace fixtures
