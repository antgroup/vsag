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
#include <filesystem>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>

#include "vsag/vsag.h"

namespace fixtures {

std::vector<int>
get_common_used_dims(uint64_t count = -1, int seed = 369);

std::vector<float>
generate_vectors(int64_t num_vectors, int64_t dim, bool need_normalize = true, int seed = 47);

std::vector<uint8_t>
generate_int4_codes(uint64_t count, uint32_t dim, int seed = 47);

std::vector<int8_t>
generate_int8_codes(uint64_t count, uint32_t dim, int seed = 47);

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

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors);

float
test_range_recall(const vsag::IndexPtr& index,
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

    std::string
    GenerateRandomFile() const {
        namespace fs = std::filesystem;
        const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        std::string fileName;
        do {
            fileName = "";
            for (int i = 0; i < 10; i++) {
                fileName += chars[RandomValue<int>(0, chars.length() - 1)];
            }
        } while (fs::exists(path + fileName));

        std::ofstream file(path + fileName);
        if (file.is_open()) {
            file.close();
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
GenTestItems(uint64_t count, uint64_t max_length, uint64_t max_index = 100000);

vsag::DatasetPtr
generate_one_dataset(int64_t dim, uint64_t count);

uint64_t
GetFileSize(const std::string& filename);
}  // Namespace fixtures
