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

// Deterministic HDF5 fixtures for the profile_dataset tests. Files are generated in a
// temporary directory from fixed seeds, so the repository carries no binary test data.

#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "deterministic_random.h"

namespace vsag::dataset_profile::testing {

/// Row 60 of the dense fixture holds these mid-range values (7.654321 + 0.001 * d). They are
/// not extremes, so no aggregate should reproduce them; tests check that none of them reaches
/// an AI request, which would mean per-row data leaked.
constexpr uint64_t CANARY_ROW = 60;

inline float
canary_value(uint64_t dimension) {
    return 7.654321F + 0.001F * static_cast<float>(dimension);
}

class TempDir {
public:
    TempDir() {
        static uint64_t counter = 0;
        const auto base = std::filesystem::temp_directory_path();
        DeterministicRng rng(
            static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) +
            counter++);
        path_ = base / ("vsag_profile_test_" + std::to_string(rng.Next()));
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TempDir(const TempDir&) = delete;
    TempDir&
    operator=(const TempDir&) = delete;

    [[nodiscard]] std::string
    File(const std::string& name) const {
        return (path_ / name).string();
    }

private:
    std::filesystem::path path_;
};

inline void
write_string_attribute(H5::H5File& file, const std::string& name, const std::string& value) {
    H5::StrType type(H5::PredType::C_S1, value.size());
    H5::DataSpace space(H5S_SCALAR);
    file.createAttribute(name, type, space).write(type, value);
}

template <typename T>
void
write_matrix(H5::H5File& file,
             const std::string& name,
             const std::vector<T>& data,
             uint64_t rows,
             uint64_t cols,
             const H5::PredType& type) {
    const hsize_t dims[2] = {rows, cols};
    H5::DataSpace space(2, dims);
    H5::DataSet dataset = file.createDataSet(name, type, space);
    if (!data.empty()) {
        dataset.write(data.data(), type);
    }
}

template <typename T>
void
write_vector(H5::H5File& file,
             const std::string& name,
             const std::vector<T>& data,
             const H5::PredType& type) {
    const hsize_t dims[1] = {data.size()};
    H5::DataSpace space(1, dims);
    H5::DataSet dataset = file.createDataSet(name, type, space);
    if (!data.empty()) {
        dataset.write(data.data(), type);
    }
}

struct SparseRecord {
    std::vector<uint32_t> ids;
    std::vector<float> values;
};

/// Serializes records in the VSAG sparse byte-stream layout and returns the record offsets.
inline std::vector<int8_t>
encode_sparse(const std::vector<SparseRecord>& records, std::vector<uint64_t>* offsets) {
    std::vector<int8_t> bytes;
    if (offsets != nullptr) {
        offsets->assign(1, 0);
    }
    for (const auto& record : records) {
        const auto length = static_cast<uint32_t>(record.ids.size());
        const uint64_t start = bytes.size();
        bytes.resize(start + 4 + 8ULL * length);
        std::memcpy(bytes.data() + start, &length, 4);
        // An empty record has no id or value storage; memcpy must not see a null pointer.
        if (length > 0) {
            std::memcpy(bytes.data() + start + 4, record.ids.data(), 4ULL * length);
            std::memcpy(
                bytes.data() + start + 4 + 4ULL * length, record.values.data(), 4ULL * length);
        }
        if (offsets != nullptr) {
            offsets->push_back(bytes.size());
        }
    }
    return bytes;
}

/// Exact ground truth for small dense fixtures (sqrt(L2Sqr) for euclidean).
inline void
dense_ground_truth(const std::vector<float>& train,
                   uint64_t rows,
                   const std::vector<float>& test,
                   uint64_t queries,
                   uint64_t dim,
                   uint64_t k,
                   std::vector<int64_t>& neighbors,
                   std::vector<float>& distances) {
    neighbors.assign(queries * k, 0);
    distances.assign(queries * k, 0.0F);
    for (uint64_t q = 0; q < queries; ++q) {
        std::vector<std::pair<double, int64_t>> all;
        for (uint64_t r = 0; r < rows; ++r) {
            double sum = 0.0;
            bool finite = true;
            for (uint64_t d = 0; d < dim; ++d) {
                const double value = train[r * dim + d];
                finite = finite && std::isfinite(value);
                const double diff = static_cast<double>(test[q * dim + d]) - value;
                sum += diff * diff;
            }
            if (finite) {
                all.emplace_back(sum, static_cast<int64_t>(r));
            }
        }
        std::sort(all.begin(), all.end());
        for (uint64_t j = 0; j < k; ++j) {
            neighbors[q * k + j] = all[j].second;
            distances[q * k + j] = static_cast<float>(std::sqrt(all[j].first));
        }
    }
}

struct DenseFixture {
    uint64_t rows{3000};
    uint64_t queries{100};
    uint64_t dim{16};
    std::vector<float> train;
    std::vector<float> test;
};

/// Four Gaussian clusters with planted defects: rows 10-12 contain NaN/+Inf/-Inf, rows 20-29
/// duplicate rows 30-39, row 40 is all zeros, rows 50-52 are norm outliers and row 60 is the
/// canary row.
inline DenseFixture
make_dense_fixture(const std::string& path,
                   uint64_t rows = 3000,
                   const std::string& distance = "euclidean") {
    DenseFixture fixture;
    fixture.rows = rows;
    const uint64_t dim = fixture.dim;
    DeterministicRng rng(2024);
    auto gaussian = [&rng]() {
        // Box-Muller from the deterministic generator.
        const double u = std::max(rng.UniformDouble(), 1e-12);
        const double v = rng.UniformDouble();
        return std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v);
    };
    std::vector<float> centers(4 * dim);
    for (auto& value : centers) {
        value = static_cast<float>(gaussian() * 5.0);
    }
    fixture.train.resize(rows * dim);
    for (uint64_t r = 0; r < rows; ++r) {
        const uint64_t cluster = r % 4;
        for (uint64_t d = 0; d < dim; ++d) {
            fixture.train[r * dim + d] =
                centers[cluster * dim + d] + static_cast<float>(gaussian());
        }
    }
    for (uint64_t r = 20; r < 30; ++r) {
        std::copy_n(fixture.train.begin() + static_cast<int64_t>((r + 10) * dim),
                    dim,
                    fixture.train.begin() + static_cast<int64_t>(r * dim));
    }
    std::fill_n(fixture.train.begin() + static_cast<int64_t>(40 * dim), dim, 0.0F);
    for (uint64_t r = 50; r < 53; ++r) {
        for (uint64_t d = 0; d < dim; ++d) {
            fixture.train[r * dim + d] *= 40.0F;
        }
    }
    fixture.train[10 * dim + 3] = std::numeric_limits<float>::quiet_NaN();
    fixture.train[11 * dim + 4] = std::numeric_limits<float>::infinity();
    fixture.train[12 * dim + 5] = -std::numeric_limits<float>::infinity();
    for (uint64_t d = 0; d < dim; ++d) {
        fixture.train[CANARY_ROW * dim + d] = canary_value(d);
    }
    fixture.test.resize(fixture.queries * dim);
    for (uint64_t q = 0; q < fixture.queries; ++q) {
        for (uint64_t d = 0; d < dim; ++d) {
            fixture.test[q * dim + d] =
                centers[(q % 4) * dim + d] + static_cast<float>(gaussian() * 1.1);
        }
    }
    std::vector<int64_t> neighbors;
    std::vector<float> distances;
    dense_ground_truth(
        fixture.train, rows, fixture.test, fixture.queries, dim, 10, neighbors, distances);
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_string_attribute(file, "type", "dense");
    write_string_attribute(file, "distance", distance);
    write_matrix(file, "train", fixture.train, rows, dim, H5::PredType::NATIVE_FLOAT);
    write_matrix(file, "test", fixture.test, fixture.queries, dim, H5::PredType::NATIVE_FLOAT);
    write_matrix(file, "neighbors", neighbors, fixture.queries, 10, H5::PredType::NATIVE_INT64);
    write_matrix(file, "distances", distances, fixture.queries, 10, H5::PredType::NATIVE_FLOAT);
    std::vector<int64_t> train_labels(rows);
    std::vector<int64_t> test_labels(fixture.queries);
    for (uint64_t r = 0; r < rows; ++r) {
        train_labels[r] = static_cast<int64_t>(r % 5);
    }
    for (uint64_t q = 0; q < fixture.queries; ++q) {
        test_labels[q] = static_cast<int64_t>(q % 6);
    }
    write_vector(file, "train_labels", train_labels, H5::PredType::NATIVE_INT64);
    write_vector(file, "test_labels", test_labels, H5::PredType::NATIVE_INT64);
    return fixture;
}

struct SparseFixture {
    std::vector<SparseRecord> train;
    std::vector<SparseRecord> test;
};

inline std::vector<SparseRecord>
random_sparse_records(uint64_t count,
                      uint32_t vocabulary,
                      uint32_t mean_terms,
                      DeterministicRng& rng) {
    std::vector<SparseRecord> records(count);
    for (auto& record : records) {
        const uint32_t terms = 1 + static_cast<uint32_t>(rng.Uniform(2ULL * mean_terms));
        std::vector<uint32_t> ids;
        for (uint32_t t = 0; t < terms; ++t) {
            // Skewed term distribution: low ids are frequent.
            const double u = rng.UniformDouble();
            ids.push_back(static_cast<uint32_t>(std::pow(u, 3.0) * vocabulary));
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        record.ids = ids;
        for (uint64_t i = 0; i < ids.size(); ++i) {
            record.values.push_back(static_cast<float>(0.05 + rng.UniformDouble()));
        }
    }
    return records;
}

/// Sparse fixture with ground truth under ip (1 - inner product). Record 7 duplicates
/// record 8 and record 15 is empty.
inline SparseFixture
make_sparse_fixture(const std::string& path, bool with_offsets = true) {
    SparseFixture fixture;
    DeterministicRng rng(77);
    fixture.train = random_sparse_records(1500, 2000, 12, rng);
    fixture.test = random_sparse_records(60, 2000, 6, rng);
    fixture.train[7] = fixture.train[8];
    fixture.train[15] = SparseRecord{};
    const uint64_t k = 10;
    std::vector<int64_t> neighbors(fixture.test.size() * k);
    std::vector<float> distances(fixture.test.size() * k);
    for (uint64_t q = 0; q < fixture.test.size(); ++q) {
        std::vector<std::pair<double, int64_t>> all;
        for (uint64_t r = 0; r < fixture.train.size(); ++r) {
            double dot = 0.0;
            const auto& a = fixture.test[q];
            const auto& b = fixture.train[r];
            uint64_t i = 0;
            uint64_t j = 0;
            while (i < a.ids.size() && j < b.ids.size()) {
                if (a.ids[i] == b.ids[j]) {
                    dot += static_cast<double>(a.values[i]) * b.values[j];
                    ++i;
                    ++j;
                } else if (a.ids[i] < b.ids[j]) {
                    ++i;
                } else {
                    ++j;
                }
            }
            all.emplace_back(1.0 - dot, static_cast<int64_t>(r));
        }
        std::sort(all.begin(), all.end());
        for (uint64_t j = 0; j < k; ++j) {
            neighbors[q * k + j] = all[j].second;
            distances[q * k + j] = static_cast<float>(all[j].first);
        }
    }
    std::vector<uint64_t> train_offsets;
    std::vector<uint64_t> test_offsets;
    const auto train_bytes = encode_sparse(fixture.train, &train_offsets);
    const auto test_bytes = encode_sparse(fixture.test, &test_offsets);
    H5::H5File file(path, H5F_ACC_TRUNC);
    write_string_attribute(file, "type", "sparse");
    write_string_attribute(file, "distance", "ip");
    write_vector(file, "train", train_bytes, H5::PredType::NATIVE_INT8);
    write_vector(file, "test", test_bytes, H5::PredType::NATIVE_INT8);
    if (with_offsets) {
        write_vector(file, "train_offsets", train_offsets, H5::PredType::NATIVE_UINT64);
        write_vector(file, "test_offsets", test_offsets, H5::PredType::NATIVE_UINT64);
    }
    write_matrix(file, "neighbors", neighbors, fixture.test.size(), k, H5::PredType::NATIVE_INT64);
    write_matrix(file, "distances", distances, fixture.test.size(), k, H5::PredType::NATIVE_FLOAT);
    return fixture;
}

}  // namespace vsag::dataset_profile::testing
