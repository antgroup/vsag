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

// HDF5 inspection and streaming readers, including malformed input.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <fstream>

#include "dataset_profiler.h"
#include "hdf5_source.h"
#include "profile_dataset_test_utils.h"

using namespace vsag::dataset_profile;
using namespace vsag::dataset_profile::testing;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string
inspect_error(const std::string& path) {
    try {
        (void)inspect_dataset(path);
    } catch (const ProfileError& error) {
        return error.what();
    }
    return "";
}

}  // namespace

TEST_CASE("Dense files are inspected without loading vectors", "[profile][reader]") {
    TempDir dir;
    const auto path = dir.File("dense.hdf5");
    const DenseFixture fixture = make_dense_fixture(path);
    const DatasetInfo info = inspect_dataset(path);
    CHECK(info.kind == VectorKind::DENSE);
    CHECK(info.distance_attribute == "euclidean");
    CHECK(info.train.rows == fixture.rows);
    CHECK(info.train.dim == fixture.dim);
    CHECK(info.test.rows == fixture.queries);
    CHECK(info.train.storage_type == "float32");
    CHECK(info.has_neighbors);
    CHECK(info.ground_truth_k == 10);
    CHECK(info.has_train_labels);
    CHECK(info.eval_compatibility_issues.empty());
    DenseSplitReader reader(path, "train");
    std::vector<float> rows(3 * fixture.dim);
    reader.ReadRows(100, 3, rows.data());
    for (uint64_t i = 0; i < rows.size(); ++i) {
        CHECK(rows[i] == fixture.train[100 * fixture.dim + i]);
    }
    REQUIRE_THROWS_AS(reader.ReadRows(fixture.rows - 1, 2, rows.data()), ProfileError);
}

TEST_CASE("Element types beyond float32/int8 are read and flagged for eval_performance",
          "[profile][reader]") {
    TempDir dir;
    for (const auto& [name, type] : std::vector<std::pair<std::string, H5::PredType>>{
             {"uint8", H5::PredType::NATIVE_UINT8}, {"int16", H5::PredType::NATIVE_INT16}}) {
        const auto path = dir.File(name + ".hdf5");
        {
            H5::H5File file(path, H5F_ACC_TRUNC);
            write_string_attribute(file, "distance", "euclidean");
            std::vector<uint8_t> bytes(40);
            std::vector<int16_t> shorts(40);
            for (uint64_t i = 0; i < 40; ++i) {
                bytes[i] = static_cast<uint8_t>(200 + i % 50);
                shorts[i] = static_cast<int16_t>(1000 - static_cast<int16_t>(i));
            }
            if (name == "uint8") {
                write_matrix(file, "train", bytes, 10, 4, type);
            } else {
                write_matrix(file, "train", shorts, 10, 4, type);
            }
        }
        const DatasetInfo info = inspect_dataset(path);
        CHECK(info.train.storage_type == name);
        bool flagged = false;
        for (const auto& issue : info.eval_compatibility_issues) {
            flagged = flagged || issue.find(name) != std::string::npos;
        }
        CHECK(flagged);
        DenseSplitReader reader(path, "train");
        std::vector<float> values(4);
        reader.ReadRows(0, 1, values.data());
        CHECK(values[0] == (name == "uint8" ? 200.0F : 1000.0F));
    }
}

TEST_CASE("Sparse records are parsed identically for any chunk size", "[profile][reader]") {
    TempDir dir;
    const auto path = dir.File("sparse.hdf5");
    const SparseFixture fixture = make_sparse_fixture(path);
    const DatasetInfo info = inspect_dataset(path);
    CHECK(info.kind == VectorKind::SPARSE);
    CHECK(info.train.has_offsets);
    CHECK(info.train.rows == fixture.train.size());
    std::vector<std::vector<uint32_t>> reference;
    for (const uint64_t chunk : {1024ULL, 4096ULL, 64ULL * 1024 * 1024}) {
        SparseSplitReader reader(path, "train", chunk);
        SparseBatch batch;
        std::vector<std::vector<uint32_t>> parsed;
        uint64_t next_row = 0;
        while (reader.NextBatch(97, batch)) {
            CHECK(batch.first_row == next_row);
            next_row += batch.Rows();
            for (uint64_t r = 0; r < batch.Rows(); ++r) {
                parsed.emplace_back(batch.ids.begin() + static_cast<int64_t>(batch.offsets[r]),
                                    batch.ids.begin() + static_cast<int64_t>(batch.offsets[r + 1]));
            }
        }
        REQUIRE(parsed.size() == fixture.train.size());
        for (uint64_t r = 0; r < parsed.size(); ++r) {
            CHECK(parsed[r] == fixture.train[r].ids);
        }
        if (reference.empty()) {
            reference = parsed;
        }
        CHECK(parsed == reference);
    }
    CHECK(count_sparse_rows(path, "train", 2048) == fixture.train.size());
}

TEST_CASE("Unsorted and repeated sparse ids are sorted and counted", "[profile][reader]") {
    TempDir dir;
    const auto path = dir.File("unsorted.hdf5");
    std::vector<SparseRecord> records = {
        {{5, 1, 3}, {0.5F, 0.1F, 0.3F}}, {{2, 2}, {1.0F, 2.0F}}, {{}, {}}};
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_vector(file, "train", encode_sparse(records, nullptr), H5::PredType::NATIVE_INT8);
    }
    SparseSplitReader reader(path, "train", 1024);
    SparseBatch batch;
    REQUIRE(reader.NextBatch(10, batch));
    REQUIRE(batch.Rows() == 3);
    CHECK(std::vector<uint32_t>(batch.ids.begin(), batch.ids.begin() + 3) ==
          std::vector<uint32_t>({1, 3, 5}));
    CHECK(std::vector<float>(batch.values.begin(), batch.values.begin() + 3) ==
          std::vector<float>({0.1F, 0.3F, 0.5F}));
    CHECK(batch.unsorted_rows == 2);
    CHECK(batch.duplicate_id_rows == 1);
    CHECK(batch.offsets[3] == batch.offsets[2]);
    CHECK_FALSE(reader.NextBatch(10, batch));
}

TEST_CASE("Truncated and oversized sparse records are rejected with their offset",
          "[profile][reader]") {
    TempDir dir;
    std::vector<SparseRecord> records = {{{1, 2}, {0.5F, 0.25F}}, {{3}, {1.0F}}};
    auto bytes = encode_sparse(records, nullptr);
    const auto truncated = dir.File("truncated.hdf5");
    {
        H5::H5File file(truncated, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_vector(file,
                     "train",
                     std::vector<int8_t>(bytes.begin(), bytes.end() - 3),
                     H5::PredType::NATIVE_INT8);
    }
    SparseSplitReader reader(truncated, "train", 1024);
    SparseBatch batch;
    REQUIRE_THROWS_WITH(reader.NextBatch(10, batch),
                        ContainsSubstring("truncated sparse record at byte 20"));
    const auto oversized = dir.File("oversized.hdf5");
    {
        std::vector<int8_t> corrupt(bytes.begin(), bytes.end());
        const uint32_t huge = 0x7FFFFFFF;
        std::memcpy(corrupt.data(), &huge, 4);
        H5::H5File file(oversized, H5F_ACC_TRUNC);
        write_string_attribute(file, "type", "sparse");
        write_vector(file, "train", corrupt, H5::PredType::NATIVE_INT8);
    }
    SparseSplitReader corrupt_reader(oversized, "train", 1024);
    REQUIRE_THROWS_WITH(corrupt_reader.NextBatch(10, batch), ContainsSubstring("byte 0"));
}

TEST_CASE("Invalid inputs produce clear errors", "[profile][reader]") {
    TempDir dir;
    CHECK_THAT(inspect_error(dir.File("missing.hdf5")), ContainsSubstring("does not exist"));
    const auto text = dir.File("text.hdf5");
    std::ofstream(text) << "not an hdf5 file";
    CHECK_THAT(inspect_error(text), ContainsSubstring("not an HDF5 file"));

    const auto empty = dir.File("empty.hdf5");
    {
        H5::H5File file(empty, H5F_ACC_TRUNC);
    }
    CHECK_THAT(inspect_error(empty), ContainsSubstring("neither /train nor /test"));

    const auto rank = dir.File("rank.hdf5");
    {
        H5::H5File file(rank, H5F_ACC_TRUNC);
        write_vector(file, "train", std::vector<float>(8, 1.0F), H5::PredType::NATIVE_FLOAT);
    }
    CHECK_THAT(inspect_error(rank), ContainsSubstring("two-dimensional"));

    const auto mismatch = dir.File("mismatch.hdf5");
    {
        H5::H5File file(mismatch, H5F_ACC_TRUNC);
        write_matrix(file, "train", std::vector<float>(8, 1.0F), 2, 4, H5::PredType::NATIVE_FLOAT);
        write_matrix(file, "test", std::vector<float>(6, 1.0F), 2, 3, H5::PredType::NATIVE_FLOAT);
    }
    CHECK_THAT(inspect_error(mismatch), ContainsSubstring("dimensions must match"));

    const auto zero_rows = dir.File("zero.hdf5");
    {
        H5::H5File file(zero_rows, H5F_ACC_TRUNC);
        write_matrix(file, "train", std::vector<float>(), 0, 4, H5::PredType::NATIVE_FLOAT);
    }
    CHECK_THAT(inspect_error(zero_rows), ContainsSubstring("must not be empty"));

    for (const auto& [type, message] : std::vector<std::pair<std::string, std::string>>{
             {"multi_vector", "multi_vector datasets are not supported"},
             {"graph", "unsupported HDF5 dataset type"}}) {
        const auto path = dir.File(type + ".hdf5");
        {
            H5::H5File file(path, H5F_ACC_TRUNC);
            write_string_attribute(file, "type", type);
            write_matrix(
                file, "train", std::vector<float>(8, 1.0F), 2, 4, H5::PredType::NATIVE_FLOAT);
        }
        CHECK_THAT(inspect_error(path), ContainsSubstring(message));
    }

    const auto no_truth = dir.File("no_truth.hdf5");
    {
        H5::H5File file(no_truth, H5F_ACC_TRUNC);
        write_matrix(file, "train", std::vector<float>(8, 1.0F), 2, 4, H5::PredType::NATIVE_FLOAT);
    }
    const DatasetInfo info = inspect_dataset(no_truth);
    bool missing_truth = false;
    bool missing_distance = false;
    for (const auto& issue : info.eval_compatibility_issues) {
        missing_truth =
            missing_truth || issue.find("/neighbors and /distances") != std::string::npos;
        missing_distance =
            missing_distance || issue.find("'distance' attribute") != std::string::npos;
    }
    CHECK(missing_truth);
    CHECK(missing_distance);

    ProfileOptions options;
    options.data_path = no_truth;
    options.metric = "hamming";
    REQUIRE_THROWS_WITH(profile_dataset(options, nullptr), ContainsSubstring("unsupported metric"));
    options.metric = "auto";
    options.sample_size = 1;
    REQUIRE_THROWS_WITH(profile_dataset(options, nullptr), ContainsSubstring("--sample-size"));
    options.sample_size = 100;
    options.disabled_sections = {"everything"};
    REQUIRE_THROWS_WITH(profile_dataset(options, nullptr), ContainsSubstring("unknown section"));
    options.disabled_sections.clear();
    options.splits = {"test"};
    REQUIRE_THROWS_WITH(profile_dataset(options, nullptr), ContainsSubstring("not present"));
}
