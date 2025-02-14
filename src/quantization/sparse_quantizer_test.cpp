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

#include "sparse_quantizer.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <memory>

#include "default_allocator.h"
#include "fixtures.h"
#include "safe_allocator.h"
#include "sparse_computer.h"
#include <iostream>

using namespace vsag;

const auto dims = {200, 300};
const auto counts = {10, 100};

void PrintCodes(const std::vector<uint8_t>& codes) {
    uint32_t nnz;
    std::memcpy(&nnz, codes.data(), sizeof(uint32_t));
    std::cout << "Encoded nnz: " << nnz << std::endl;

    const uint8_t* ptr = codes.data() + sizeof(uint32_t);
    for (uint32_t i = 0; i < nnz; ++i) {
        uint32_t id;
        float val;
        std::memcpy(&id, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);
        std::memcpy(&val, ptr, sizeof(float));
        ptr += sizeof(float);
        
        std::cout << "Encoded id, val pair: " << id << ", " << val << std::endl;
    }
}

void PrintSparseVector(const vsag::SparseVectors& vec) {
    if (vec.num != 1 || !vec.offsets || !vec.ids || !vec.vals) {
        std::cout << "Invalid SparseVector" << std::endl;
        return;
    }

    uint32_t nnz = vec.offsets[1] - vec.offsets[0];
    std::cout << "SparseVectors nnz: " << nnz << std::endl;

    std::cout << "SparseVectors ids: ";
    for (uint32_t i = 0; i < nnz; ++i) {
        std::cout << vec.ids[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "SparseVectors vals: ";
    for (uint32_t i = 0; i < nnz; ++i) {
        std::cout << vec.vals[i] << " ";
    }
    std::cout << std::endl;
}

void SortSparseVector(SparseVectors& vec, uint32_t start, uint32_t end) {
    uint32_t nnz = end - start;
    std::vector<std::pair<uint32_t, float>> id_val_pairs(nnz);
    for (uint32_t j = 0; j < nnz; ++j) {
        id_val_pairs[j] = {vec.ids[start + j], vec.vals[start + j]};
    }

    std::sort(id_val_pairs.begin(), id_val_pairs.end(), 
              [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                  return a.first < b.first;
              });

    for (uint32_t j = 0; j < nnz; ++j) {
        vec.ids[start + j] = id_val_pairs[j].first;
        vec.vals[start + j] = id_val_pairs[j].second;
    }
}

void TestQuantizerEncodeDecode(
    SparseQuantizer& quant, int64_t dim, int count, float error = 1e-2f, bool retrain = false) {
    
    SparseVectors vecs = fixtures::GenerateSparseVectors(count, dim, 123);
    
    if (retrain) {
        quant.ReTrain(&vecs, count);
    }
    
    // Test EncodeOne & DecodeOne
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t nnz = vecs.offsets[i + 1] - vecs.offsets[i];
        std::vector<uint8_t> codes(quant.GetCodeSize(nnz, 1));

        SparseVectors single_vec;
        single_vec.num = 1;
        single_vec.offsets = new uint32_t[2]{0, nnz};
        single_vec.ids = new uint32_t[nnz];
        single_vec.vals = new float[nnz];
        
        std::memcpy(single_vec.ids, vecs.ids + vecs.offsets[i], nnz * sizeof(uint32_t));
        std::memcpy(single_vec.vals, vecs.vals + vecs.offsets[i], nnz * sizeof(float));

        // 将 ids 和 vals 组合成对并进行排序
        std::vector<std::pair<uint32_t, float>> id_val_pairs(nnz);
        for (uint32_t j = 0; j < nnz; ++j) {
            id_val_pairs[j] = {single_vec.ids[j], single_vec.vals[j]};
        }

        std::sort(id_val_pairs.begin(), id_val_pairs.end(), 
                  [](const std::pair<uint32_t, float>& a, const std::pair<uint32_t, float>& b) {
                      return a.first < b.first;
                  });

        // 排序完成后，将结果复制回 single_vec.ids 和 single_vec.vals
        for (uint32_t j = 0; j < nnz; ++j) {
            single_vec.ids[j] = id_val_pairs[j].first;
            single_vec.vals[j] = id_val_pairs[j].second;
        }

        // Encode one vector
        quant.EncodeOne(&single_vec, codes.data());
        PrintCodes(codes);
        PrintSparseVector(single_vec);

        SparseVectors out_vec;
        quant.DecodeOne(codes.data(), &out_vec);

        REQUIRE(out_vec.num == 1);
        REQUIRE(out_vec.offsets[0] == 0);
        REQUIRE(out_vec.offsets[1] == nnz);
        
        for (uint32_t j = 0; j < nnz; ++j) {
            REQUIRE(single_vec.ids[j] == out_vec.ids[j]);
            REQUIRE(std::abs(single_vec.vals[j] - out_vec.vals[j]) < error);
        }

        delete[] single_vec.offsets; 
        delete[] single_vec.ids;
        delete[] single_vec.vals;
        delete[] out_vec.offsets;
        delete[] out_vec.ids;
        delete[] out_vec.vals;
    }

    // Test EncodeBatch & DecodeBatch
    std::vector<uint8_t> codes(quant.GetCodeSize(vecs.offsets[count], count));
    quant.EncodeBatch(&vecs, codes.data(), count);

    SparseVectors out_vecs;
    quant.DecodeBatch(codes.data(), &out_vecs, count);

    // 比对 batch offsets, ids, vals
    REQUIRE(out_vecs.num == count);
    for (uint64_t i = 0; i <= count; ++i) {
        REQUIRE(vecs.offsets[i] == out_vecs.offsets[i]);
    }

    for (uint64_t i = 0; i < count; ++i) {
        uint32_t nnz = vecs.offsets[i + 1] - vecs.offsets[i];
        for (uint32_t j = 0; j < nnz; ++j) {
            REQUIRE(vecs.ids[vecs.offsets[i] + j] == out_vecs.ids[out_vecs.offsets[i] + j]);
            REQUIRE(std::abs(vecs.vals[vecs.offsets[i] + j] - out_vecs.vals[out_vecs.offsets[i] + j]) < error);
        }
    }
        delete[] out_vecs.offsets;
        delete[] out_vecs.ids;
        delete[] out_vecs.vals;
}


// 计算稀疏向量的内积
float
SparseInnerProduct(const uint32_t* ids1,
                   const float* vals1,
                   int32_t nnz1,
                   const uint32_t* ids2,
                   const float* vals2,
                   int32_t nnz2) {
    
    float result = 0.0f;

    for (int32_t i = 0; i < nnz1; ++i) {
        for (int32_t j = 0; j < nnz2; ++j) {
            if (ids1[i] == ids2[j]) {
                result += vals1[i] * vals2[j];
                break;
            }
        }
    }

    return result;
}

void
TestComputer(
    SparseQuantizer& quant, size_t dim, uint32_t count, float error = 1e-2f, bool retrain = false) {
    auto query_count = 100;

    auto vecs = fixtures::GenerateSparseVectors(query_count, dim, 123);
    auto querys = fixtures::GenerateSparseVectors(query_count, dim, 321);

    if (retrain) {
        quant.ReTrain(&vecs, count);
    }

    for (int i = 0; i < query_count; ++i) {
        std::shared_ptr<SparseComputer> computer;
        computer = quant.FactoryComputer();

        //generate sparse query vector
        uint32_t query_nnz = querys.offsets[i + 1] - querys.offsets[i];
        SparseVectors single_query;
        single_query.num = 1;
        single_query.offsets = new uint32_t[2]{0, query_nnz};
        single_query.ids = new uint32_t[query_nnz];
        single_query.vals = new float[query_nnz];
        
        std::memcpy(single_query.ids, querys.ids + querys.offsets[i], query_nnz * sizeof(uint32_t));
        std::memcpy(single_query.vals, querys.vals + querys.offsets[i], query_nnz * sizeof(float));

        computer->SetQuery(&single_query);
        for (int j = 0; j < 100; ++j) {
            auto idx1 = random() % count;

            uint32_t vec_nnz = vecs.offsets[idx1 + 1] - vecs.offsets[idx1];

            SparseVectors single_vec;
            single_vec.num = 1;
            single_vec.offsets = new uint32_t[2]{0, vec_nnz};
            single_vec.ids = new uint32_t[vec_nnz];
            single_vec.vals = new float[vec_nnz];
        
            std::memcpy(single_vec.ids, vecs.ids + vecs.offsets[idx1], vec_nnz * sizeof(uint32_t));
            std::memcpy(single_vec.vals, vecs.vals + vecs.offsets[idx1], vec_nnz * sizeof(float));

            auto* codes1 = new uint8_t[quant.GetCodeSize(vec_nnz, 1)];
            quant.EncodeOne(&single_vec, codes1);
            float gt = 0.0f;
            float value = 0.0f;
            quant.ComputeDist(*computer, codes1, &value);
            REQUIRE(quant.ComputeDist(*computer, codes1) == value);
            gt = SparseInnerProduct(single_query.ids, single_query.vals, query_nnz, single_vec.ids, single_vec.vals, vec_nnz);
            gt = 1.f - gt;

            REQUIRE(std::abs(gt - value) < error);
            delete[] codes1;
            delete[] single_vec.offsets; 
            delete[] single_vec.ids;
            delete[] single_vec.vals;
        }

        delete[] single_query.offsets; 
        delete[] single_query.ids;
        delete[] single_query.vals;
    }
}

TEST_CASE("sparse_compute", "[ut][sparse_quantizer]") {
    for (auto dim : dims) {
        for (auto count : counts) {
            auto allocator = SafeAllocator::FactoryDefaultAllocator();
            SparseQuantizer quantizer(allocator.get());
            TestQuantizerEncodeDecode(quantizer, dim, count);
            TestComputer(quantizer, dim, count);
        }
    }
}
