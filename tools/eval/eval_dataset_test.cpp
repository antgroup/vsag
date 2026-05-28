
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

#include "eval_dataset.h"

#include <H5Cpp.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using vsag::SparseVector;
using vsag::eval::EvalDataset;
using vsag::eval::EvalDatasetPtr;

EvalDatasetPtr
BuildSparseDataset(bool with_token_sequences) {
    // Build a tiny sparse dataset (3 train, 2 test) and optionally attach
    // the original tokenized term-id sequences.
    auto ds = std::make_shared<EvalDataset>();

    // Sparse train vectors.
    std::vector<SparseVector> train(3);
    train[0].len_ = 2;
    train[0].ids_ = new uint32_t[2]{1, 5};
    train[0].vals_ = new float[2]{0.5f, 1.0f};
    train[1].len_ = 1;
    train[1].ids_ = new uint32_t[1]{3};
    train[1].vals_ = new float[1]{0.25f};
    train[2].len_ = 0;  // empty sparse vector

    std::vector<SparseVector> test(2);
    test[0].len_ = 2;
    test[0].ids_ = new uint32_t[2]{2, 7};
    test[0].vals_ = new float[2]{0.4f, 0.6f};
    test[1].len_ = 1;
    test[1].ids_ = new uint32_t[1]{9};
    test[1].vals_ = new float[1]{1.0f};

    if (with_token_sequences) {
        train[0].token_seq_len_ = 4;
        train[0].token_sequence_ = new uint32_t[4]{10, 20, 10, 30};
        train[1].token_seq_len_ = 2;
        train[1].token_sequence_ = new uint32_t[2]{42, 42};
        // train[2] intentionally has no original document.
        train[2].token_seq_len_ = 0;
        train[2].token_sequence_ = nullptr;

        test[0].token_seq_len_ = 1;
        test[0].token_sequence_ = new uint32_t[1]{99};
        test[1].token_seq_len_ = 3;
        test[1].token_sequence_ = new uint32_t[3]{1, 2, 3};
    }

    // Inject the sparse vectors via friend access through pointer.
    // EvalDataset's members are private; expose a tiny helper through
    // direct memory write would be invasive. Instead we build via Save's
    // public API by reaching through a shim subclass.
    struct ShimDataset : public EvalDataset {
        std::vector<SparseVector>&
        train() {
            return this->sparse_train_;
        }
        std::vector<SparseVector>&
        test() {
            return this->sparse_test_;
        }
        void
        set_metric(const std::string& m) {
            this->metric_ = m;
        }
        void
        set_type(const std::string& t) {
            this->vector_type_ = t;
        }
        void
        set_counts(int64_t base, int64_t query) {
            this->number_of_base_ = base;
            this->number_of_query_ = query;
        }
        void
        set_neighbors_shape(int64_t q, int64_t k) {
            this->neighbors_shape_ = {q, k};
        }
        void
        set_neighbors(int64_t* ptr) {
            this->neighbors_.reset(ptr);
        }
        void
        set_distances(float* ptr) {
            this->distances_.reset(ptr);
        }
    };

    auto shim = std::make_shared<ShimDataset>();
    shim->set_type(vsag::SPARSE_VECTORS);
    shim->set_metric("ip");
    shim->train() = std::move(train);
    shim->test() = std::move(test);
    shim->set_counts(static_cast<int64_t>(shim->train().size()),
                     static_cast<int64_t>(shim->test().size()));

    // Minimal ground truth (k=1 per query; values do not matter for the
    // load/save round-trip we are testing).
    constexpr int64_t kK = 1;
    int64_t* nb = new int64_t[shim->test().size() * kK];
    float* dist = new float[shim->test().size() * kK];
    for (int64_t i = 0; i < static_cast<int64_t>(shim->test().size()); ++i) {
        nb[i] = 0;
        dist[i] = 0.0f;
    }
    shim->set_neighbors_shape(static_cast<int64_t>(shim->test().size()), kK);
    shim->set_neighbors(nb);
    shim->set_distances(dist);

    return shim;
}

std::string
TempPath(const std::string& tag) {
    std::string path = "/tmp/eval_dataset_test_" + tag + ".hdf5";
    std::remove(path.c_str());
    return path;
}

}  // namespace

TEST_CASE("EvalDataset sparse round-trip without token sequences", "[ut][eval_dataset]") {
    auto path = TempPath("nosqry");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/false);
        EvalDataset::Save(ds, path);
    }

    // Verify the optional datasets are absent on disk.
    {
        H5::H5File file(path, H5F_ACC_RDONLY);
        H5::Group root = file.openGroup("/");
        bool has_train_token = false;
        bool has_test_token = false;
        hsize_t numObj = root.getNumObjs();
        for (unsigned i = 0; i < numObj; ++i) {
            std::string n = root.getObjnameByIdx(i);
            if (n == "train_token_sequences")
                has_train_token = true;
            if (n == "test_token_sequences")
                has_test_token = true;
        }
        REQUIRE_FALSE(has_train_token);
        REQUIRE_FALSE(has_test_token);
    }

    // Loading must succeed and token_sequence_ fields stay empty.
    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 3);
    REQUIRE(loaded->GetNumberOfQuery() == 2);
    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    for (int i = 0; i < 3; ++i) {
        REQUIRE(train[i].token_seq_len_ == 0);
        REQUIRE(train[i].token_sequence_ == nullptr);
    }
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip with token sequences", "[ut][eval_dataset]") {
    auto path = TempPath("withseq");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }

    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 3);
    REQUIRE(loaded->GetNumberOfQuery() == 2);

    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    REQUIRE(train[0].token_seq_len_ == 4);
    REQUIRE(train[0].token_sequence_[0] == 10u);
    REQUIRE(train[0].token_sequence_[2] == 10u);  // duplicates preserved
    REQUIRE(train[1].token_seq_len_ == 2);
    REQUIRE(train[1].token_sequence_[0] == 42u);
    REQUIRE(train[1].token_sequence_[1] == 42u);
    REQUIRE(train[2].token_seq_len_ == 0);
    REQUIRE(train[2].token_sequence_ == nullptr);

    const auto* test = static_cast<const SparseVector*>(loaded->GetTest());
    REQUIRE(test[0].token_seq_len_ == 1);
    REQUIRE(test[0].token_sequence_[0] == 99u);
    REQUIRE(test[1].token_seq_len_ == 3);
    REQUIRE(test[1].token_sequence_[0] == 1u);
    REQUIRE(test[1].token_sequence_[2] == 3u);
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset legacy sparse file without token_sequences key still loads",
          "[ut][eval_dataset]") {
    // Synthesize a minimal legacy sparse HDF5 file by hand (no
    // train_token_sequences / test_token_sequences keys), and verify
    // EvalDataset::Load still succeeds.
    auto path = TempPath("legacy");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);

        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        {
            auto a = file.createAttribute("type", str_type, H5::DataSpace(H5S_SCALAR));
            std::string v = "sparse";
            a.write(str_type, v);
        }
        {
            auto a = file.createAttribute("distance", str_type, H5::DataSpace(H5S_SCALAR));
            std::string v = "ip";
            a.write(str_type, v);
        }

        // Encode 1 train and 1 test sparse vector each. Layout:
        //   uint32 len | uint32 ids[len] | float vals[len]
        auto encode =
            [](uint32_t len, const std::vector<uint32_t>& ids, const std::vector<float>& vals) {
                std::vector<char> buf(sizeof(uint32_t) + len * (sizeof(uint32_t) + sizeof(float)));
                char* p = buf.data();
                std::memcpy(p, &len, sizeof(uint32_t));
                p += sizeof(uint32_t);
                std::memcpy(p, ids.data(), len * sizeof(uint32_t));
                p += len * sizeof(uint32_t);
                std::memcpy(p, vals.data(), len * sizeof(float));
                return buf;
            };
        auto train_buf = encode(2, {1, 4}, {0.5f, 1.0f});
        auto test_buf = encode(1, {2}, {0.7f});
        {
            hsize_t dims[1] = {static_cast<hsize_t>(train_buf.size())};
            H5::DataSpace sp(1, dims);
            auto ds = file.createDataSet("/train", H5::PredType::ALPHA_I8, sp);
            ds.write(train_buf.data(), H5::PredType::NATIVE_CHAR);
        }
        {
            hsize_t dims[1] = {static_cast<hsize_t>(test_buf.size())};
            H5::DataSpace sp(1, dims);
            auto ds = file.createDataSet("/test", H5::PredType::ALPHA_I8, sp);
            ds.write(test_buf.data(), H5::PredType::NATIVE_CHAR);
        }
        // Minimal ground truth: shape (1, 1).
        {
            hsize_t dims[2] = {1, 1};
            H5::DataSpace sp(2, dims);
            int64_t nb[1] = {0};
            auto ds = file.createDataSet("/neighbors", H5::PredType::NATIVE_INT64, sp);
            ds.write(nb, H5::PredType::NATIVE_INT64);
        }
        {
            hsize_t dims[2] = {1, 1};
            H5::DataSpace sp(2, dims);
            float dv[1] = {0.0f};
            auto ds = file.createDataSet("/distances", H5::PredType::NATIVE_FLOAT, sp);
            ds.write(dv, H5::PredType::NATIVE_FLOAT);
        }
    }

    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetVectorType() == vsag::SPARSE_VECTORS);
    REQUIRE(loaded->GetNumberOfBase() == 1);
    REQUIRE(loaded->GetNumberOfQuery() == 1);
    const auto* train = static_cast<const SparseVector*>(loaded->GetTrain());
    REQUIRE(train[0].token_seq_len_ == 0);
    REQUIRE(train[0].token_sequence_ == nullptr);
    // The reader rebuilds offsets from the byte stream when the file does
    // not store them. For a single 2-nnz record the layout is
    //   [u32 len=2][u32 ids[2]][f32 vals[2]] = 4 + 8 + 8 = 20 bytes.
    const auto& train_off = loaded->GetSparseTrainOffsets();
    REQUIRE(train_off.size() == 2);
    REQUIRE(train_off[0] == 0u);
    REQUIRE(train_off[1] == 20u);
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset sparse round-trip writes record-offset indexes", "[ut][eval_dataset]") {
    auto path = TempPath("offsets");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }

    // Verify the four offsets datasets exist on disk and have the correct
    // contents. The /train byte layout is:
    //   record 0 (len=2): 4 + 8 + 8 = 20 bytes -> starts at 0
    //   record 1 (len=1): 4 + 4 + 4 = 12 bytes -> starts at 20
    //   record 2 (len=0): 4 bytes              -> starts at 32
    //   total                                  = 36 bytes
    {
        H5::H5File file(path, H5F_ACC_RDONLY);
        std::vector<uint64_t> train_off(4);
        H5::DataSet ds = file.openDataSet("/train_offsets");
        ds.read(train_off.data(), H5::PredType::NATIVE_UINT64);
        REQUIRE(train_off == std::vector<uint64_t>{0, 20, 32, 36});

        std::vector<uint64_t> test_off(3);
        H5::DataSet tds = file.openDataSet("/test_offsets");
        tds.read(test_off.data(), H5::PredType::NATIVE_UINT64);
        // record 0 (len=2): 20, record 1 (len=1): 12 -> offsets {0, 20, 32}
        REQUIRE(test_off == std::vector<uint64_t>{0, 20, 32});

        std::vector<uint64_t> train_token_off(4);
        H5::DataSet ttds = file.openDataSet("/train_token_sequences_offsets");
        ttds.read(train_token_off.data(), H5::PredType::NATIVE_UINT64);
        // train tokens: seq_lens 4, 2, 0
        //   record 0: 4 + 16 = 20 bytes  -> 0
        //   record 1: 4 + 8  = 12 bytes  -> 20
        //   record 2: 4      = 4  bytes  -> 32
        //   total                        = 36 bytes
        REQUIRE(train_token_off == std::vector<uint64_t>{0, 20, 32, 36});

        std::vector<uint64_t> test_token_off(3);
        H5::DataSet etds = file.openDataSet("/test_token_sequences_offsets");
        etds.read(test_token_off.data(), H5::PredType::NATIVE_UINT64);
        // test tokens: seq_lens 1, 3
        //   record 0: 4 + 4  = 8  bytes -> 0
        //   record 1: 4 + 12 = 16 bytes -> 8
        //   total                       = 24 bytes
        REQUIRE(test_token_off == std::vector<uint64_t>{0, 8, 24});
    }

    // Loading must succeed and surface the offsets through the public API.
    auto loaded = EvalDataset::Load(path);
    REQUIRE(loaded->GetSparseTrainOffsets() == std::vector<uint64_t>{0, 20, 32, 36});
    REQUIRE(loaded->GetSparseTestOffsets() == std::vector<uint64_t>{0, 20, 32});
    REQUIRE(loaded->GetTrainTokenSequenceOffsets() == std::vector<uint64_t>{0, 20, 32, 36});
    REQUIRE(loaded->GetTestTokenSequenceOffsets() == std::vector<uint64_t>{0, 8, 24});
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with token_sequences missing offsets",
          "[ut][eval_dataset]") {
    // Contract: whenever a *_token_sequences byte stream is present, the
    // companion *_token_sequences_offsets dataset MUST also exist. Files
    // that violate this invariant are considered malformed.
    auto path = TempPath("seq_no_off");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }
    // Delete only /train_token_sequences_offsets, keep the byte stream.
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        file.unlink("/train_token_sequences_offsets");
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with orphan token_sequences_offsets",
          "[ut][eval_dataset]") {
    // The reverse direction: a *_token_sequences_offsets dataset must not
    // appear without its byte-stream counterpart.
    auto path = TempPath("orphan_off");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/true);
        EvalDataset::Save(ds, path);
    }
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        file.unlink("/test_token_sequences");
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}

TEST_CASE("EvalDataset rejects sparse files with corrupted offsets", "[ut][eval_dataset]") {
    auto path = TempPath("badoff");
    {
        auto ds = BuildSparseDataset(/*with_token_sequences=*/false);
        EvalDataset::Save(ds, path);
    }
    // Corrupt /train_offsets: replace the sentinel with a wrong total size.
    {
        H5::H5File file(path, H5F_ACC_RDWR);
        H5::DataSet ds = file.openDataSet("/train_offsets");
        std::vector<uint64_t> tmp(4);
        ds.read(tmp.data(), H5::PredType::NATIVE_UINT64);
        tmp.back() += 7;  // sentinel no longer matches byte stream length
        ds.write(tmp.data(), H5::PredType::NATIVE_UINT64);
    }
    REQUIRE_THROWS(EvalDataset::Load(path));
    std::remove(path.c_str());
}
