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

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "common.h"
#include "simd/normalize.h"
#include "typing.h"
#include "vsag/vsag.h"

namespace fixtures {

extern const int RABITQ_MIN_RACALL_DIM;

std::vector<int>
get_common_used_dims(uint64_t count = -1, int seed = 369, int limited_dim = -1);

std::vector<int>
get_index_test_dims(uint64_t count = -1, int seed = 369, int limited_dim = -1);

template <typename T>
T*
CopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T>
T*
DuplicateCopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    if (vec.size() % 2 != 0) {
        throw std::runtime_error("Vector size must be even for duplication.");
    }
    memcpy(result, vec.data(), (vec.size() / 2) * sizeof(T));
    memcpy(result + vec.size() / 2, vec.data(), (vec.size() / 2) * sizeof(T));
    return result;
}

template <typename T>
T*
CopyVector(const vsag::Vector<T>& vec, vsag::Allocator* allocator) {
    T* result;
    if (allocator) {
        result = (T*)allocator->Allocate(sizeof(T) * vec.size());
    } else {
        result = new T[vec.size()];
    }
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T>
T*
CopyVector(const std::vector<T>& vec, vsag::Allocator* allocator) {
    T* result;
    if (allocator) {
        result = (T*)allocator->Allocate(sizeof(T) * vec.size());
    } else {
        result = new T[vec.size()];
    }
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

template <typename T, typename RT = typename std::enable_if<std::is_integral_v<T>, T>::type>
std::vector<RT>
GenerateVectors(uint64_t count,
                uint32_t dim,
                int seed = 47,
                T min = std::numeric_limits<T>::lowest(),
                T max = std::numeric_limits<T>::max()) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> distrib_real(min, max);
    std::vector<T> vectors(dim * count);
    for (int64_t i = 0; i < dim * count; ++i) {
        vectors[i] = distrib_real(rng);
    }
    return vectors;
}

template <typename T, typename RT = typename std::enable_if<std::is_floating_point_v<T>, T>::type>
std::vector<RT>
GenerateVectors(uint64_t count, uint32_t dim, int seed = 47, bool need_normalize = true) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> distrib_real(0.1, 0.9);
    std::vector<T> vectors(dim * count);
    for (int64_t i = 0; i < dim * count; ++i) {
        vectors[i] = distrib_real(rng);
    }
    if (need_normalize) {
        for (int64_t i = 0; i < count; ++i) {
            vsag::Normalize(vectors.data() + i * dim, vectors.data() + i * dim, dim);
        }
    }
    return vectors;
}

std::vector<vsag::SparseVector>
GenerateSparseVectors(uint32_t count,
                      uint32_t max_dim = 100,
                      uint32_t max_id = 1000,
                      float min_val = -1,
                      float max_val = 1,
                      int seed = 47);

vsag::Vector<vsag::SparseVector>
GenerateSparseVectors(vsag::Allocator* allocator,
                      uint32_t count,
                      uint32_t max_dim = 100,
                      uint32_t max_id = 1000,
                      float min_val = -1,
                      float max_val = 1,
                      int seed = 47);

std::pair<std::vector<float>, std::vector<uint8_t>>
GenerateBinaryVectorsAndCodes(uint32_t count, uint32_t dim, int seed = 47);

std::string
create_random_string(bool is_full);

std::vector<float>
generate_vectors(uint64_t count, uint32_t dim, bool need_normalize = true, int seed = 47);

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<uint8_t>
generate_uint8_codes(uint64_t count, uint32_t dim, int seed = 47);

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_elements,
                         int64_t dim,
                         bool need_normalize = true,
                         int seed = 47);

vsag::IndexPtr
generate_index(const std::string& name,
               const std::string& metric_type,
               int64_t num_vectors,
               int64_t dim,
               std::vector<int64_t>& ids,
               std::vector<float>& vectors,
               bool use_conjugate_graph = false);

std::vector<char>
generate_extra_infos(uint64_t count, uint32_t size, int seed = 47);

vsag::AttributeSet*
generate_attributes(uint64_t count,
                    uint32_t max_term_count = 10,
                    uint32_t max_value_count = 10,
                    int seed = 97);

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors);

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim);

vsag::DatasetPtr
brute_force(const vsag::DatasetPtr& query,
            const vsag::DatasetPtr& base,
            int64_t k,
            const std::string& metric_type);

vsag::DatasetPtr
brute_force(const vsag::DatasetPtr& query,
            const vsag::DatasetPtr& base,
            int64_t k,
            const std::string& metric_type,
            const std::string& data_type);

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
RandomValue(const T& min, const T& max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    return dis(gen);
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
RandomValue(const T& min, const T& max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(min, max);
    return dis(gen);
}

class TempDir {
public:
    explicit TempDir(const std::string& prefix) {
        namespace fs = std::filesystem;
        std::stringstream dirname;
        do {
            auto epoch_time = std::chrono::system_clock::now().time_since_epoch();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch_time).count();

            int random_number = RandomValue<int>(1000, 9999);

            dirname << "vsagtest_" << prefix << "_" << std::setfill('0') << std::setw(14) << seconds
                    << "_" << std::to_string(random_number);
            path = "/tmp/" + dirname.str() + "/";
            dirname.clear();
        } while (fs::exists(path));

        std::filesystem::create_directory(path);
    }

    ~TempDir() {
        std::filesystem::remove_all(path);
    }

    [[nodiscard]] std::string
    GenerateRandomFile(bool create_file = true) const {
        namespace fs = std::filesystem;
        const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string fileName;
        do {
            fileName = "";
            for (int i = 0; i < 10; i++) {
                fileName += chars[RandomValue<uint64_t>(0, chars.length() - 1)];
            }
        } while (fs::exists(path + fileName));

        if (create_file) {
            std::ofstream file(path + fileName);
            if (file.is_open()) {
                file.close();
            }
        }
        return path + fileName;
    }

    std::string path;
};

struct comparable_float_t {
    comparable_float_t(float val) {
        this->value = val;
    }

    bool
    operator==(const comparable_float_t& d) const {
        double a = this->value;
        double b = d.value;
        double max_value = std::max(std::abs(a), std::abs(b));
        int power = std::max(0, int(log10(max_value) + 1));
        return std::abs(a - b) <= epsilon * pow(10.0, power);
    }

    friend std::ostream&
    operator<<(std::ostream& os, const comparable_float_t& obj) {
        os << obj.value;
        return os;
    }

    float value;
    const double epsilon = 2e-6;
};
using dist_t = comparable_float_t;
// The error epsilon between time_t and recall_t should be 1e-6; however, the error does not fall
// between 1e-6 and 2e-6 in actual situations. Therefore, to ensure compatibility with dist_t,
// we will limit the error to within 2e-6.
using time_t = comparable_float_t;
using recall_t = comparable_float_t;

struct IOItem {
    uint64_t start_;
    uint64_t length_;
    uint8_t* data_;

    ~IOItem() {
        delete[] data_;
    }
};

std::vector<IOItem>
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index = 10000);

vsag::DatasetPtr
generate_one_dataset(int64_t dim, uint64_t count);

uint64_t
GetFileSize(const std::string& filename);

std::vector<std::string>
SplitString(const std::string& s, char delimiter);

float
GetSparseDistance(const vsag::SparseVector& vec1, const vsag::SparseVector& vec2);

template <typename T>
std::vector<T>
RandomSelect(const std::vector<T>& vec, int64_t count = 1) {
    std::vector<T> selected;
    count = std::min(count, static_cast<int64_t>(vec.size()));
    std::sample(vec.begin(),
                vec.end(),
                std::back_inserter(selected),
                count,
                std::mt19937(RandomValue(0, 10000)));
    return selected;
}

template <typename T>
void
test_serializion_file(T& old_instance, T& new_instance, const std::string name) {
    auto temp_dir = TempDir(name);
    auto file = temp_dir.GenerateRandomFile();
    std::ofstream ofs(file);
    old_instance.Serialize(ofs);
    ofs.close();

    std::ifstream ifs(file);
    auto value = new_instance.Deserialize(ifs);
    ifs.close();
    if (not value.has_value()) {
        throw std::runtime_error("deserialize failed: " + value.error().message);
    }
}

}  // namespace fixtures
