
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
#include <memory>
#include <vector>
#include <algorithm>

#include "../logger.h"
#include "sparse_computer.h"
#include "metric_type.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "dataset.h"
#include "inner_string_params.h"

namespace vsag {
/**
 * @class SparseQuantizer
 * @brief This class is used for distance computation and encoding/decoding of data.
 */
class SparseQuantizer {
public:
    explicit SparseQuantizer(Allocator* allocator)
        : allocator_(allocator){};

    ~SparseQuantizer(){
        this->allocator_->Deallocate(indptr_);
        this->allocator_->Deallocate(indices_);
        this->allocator_->Deallocate(data_);
    }

    bool
    InsertAll(const SparseVectors& sv);

    inline void
    Serialize(StreamWriter& writer) {};

    inline void
    Deserialize(StreamReader& reader) {};

    inline float
    Compute(int32_t nnz1, const uint32_t* ids1, const float* vals1, 
            int32_t nnz2, const uint32_t* ids2, const float* vals2) {};

    inline void
    ComputeDist(SparseComputer& computer, int32_t nnz, const uint32_t* ids, const float* vals, float* dists) const {};

    inline float
    ComputeDist(SparseComputer& computer, int32_t nnz, const uint32_t* ids, const float* vals) const {};

    std::shared_ptr<SparseComputer>
    FactoryComputer() {
        return std::make_shared<SparseComputer>(static_cast<SparseQuantizer*>(this));
    }

    inline void
    ReleaseComputer(SparseComputer& computer) const {};

    inline void
    ProcessQuery(int32_t nnz, const uint32_t* ids, const float* vals, SparseComputer& computer) const {};

    [[nodiscard]] std::string
    Name() const {
        return QUANTIZATION_TYPE_VALUE_SPARSE;
    }

    [[nodiscard]] MetricType
    Metric() const {
        return this->metric_;
    }

private:
    uint32_t *indptr_{nullptr};
    uint32_t *indices_{nullptr};
    float *data_{nullptr};
    uint32_t num_ = 0;
    Allocator* const allocator_{nullptr};
    MetricType metric_{MetricType::METRIC_TYPE_IP};

};

    bool
    SparseQuantizer::InsertAll(const SparseVectors& sv) {
    // 首先释放旧的内存
    this->allocator_->Deallocate(indptr_);
    this->allocator_->Deallocate(indices_);
    this->allocator_->Deallocate(data_);
    this.num_ = sv.num;

    // 定义所需要分配的内存大小
    size_t indptr_size = (sv.num + 1) * sizeof(uint32_t);
    size_t indices_size = sv.offsets[sv.num] * sizeof(uint32_t);
    size_t data_size = sv.offsets[sv.num] * sizeof(float);

    try{
    indptr_ = reinterpret_cast<uint32_t*>(allocator_->Allocate(indptr_size));
    indices_ = reinterpret_cast<uint32_t*>(allocator_->Allocate(indices_size));
    data_ = reinterpret_cast<float*>(allocator_->Allocate(data_size));
    } catch (const std::bad_alloc& e) {
        indptr_ = nullptr;
        indices_ = nullptr;
        data_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }

    // 使用 memcpy 复制数据
    std::memcpy(indptr_, sv.offsets, indptr_size);
    std::memcpy(indices_, sv.ids, indices_size);
    std::memcpy(data_, sv.vals, data_size);

    return true;
    }

    inline float Compute(int32_t nnz1, const uint32_t* ids1, const float* vals1,
                         int32_t nnz2, const uint32_t* ids2, const float* vals2) const {
        
        std::vector<std::pair<uint32_t, float>> vec1(nnz1), vec2(nnz2);

        // Fill the vectors with the input data
        for (int32_t i = 0; i < nnz1; ++i) {
            vec1[i] = {ids1[i], vals1[i]};
        }
        for (int32_t i = 0; i < nnz2; ++i) {
            vec2[i] = {ids2[i], vals2[i]};
        }

        // Sort both vectors by id
        std::sort(vec1.begin(), vec1.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::sort(vec2.begin(), vec2.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Use double pointer technique to calculate inner product
        float result = 0.0f;
        int32_t i = 0, j = 0;
        while (i < nnz1 && j < nnz2) {
            if (vec1[i].first == vec2[j].first) {
                result += vec1[i].second * vec2[j].second;
                ++i;
                ++j;
            } else if (vec1[i].first < vec2[j].first) {
                ++i;
            } else {
                ++j;
            }
        }

        return result;
    }

    inline void
    SparseQuantizer::Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->num_);
        StreamWriter::WriteObj(writer, this->metric_);
        StreamWriter::WriteObj(writer, this->indptr_);
        StreamWriter::WriteObj(writer, this->indices_);
        StreamWriter::WriteObj(writer, this->data_);
    }

    inline void
    SparseQuantizer::Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->num_);
        StreamReader::ReadObj(reader, this->metric_);
        StreamReader::ReadObj(reader, this->indptr_);
        StreamReader::ReadObj(reader, this->indices_);
        StreamReader::ReadObj(reader, this->data_);
    }

    inline void
    SparseQuantizer::ComputeDist(SparseComputer& computer, uint32_t u, float* dists) const {

        if (u >= num_) {
            // Out of bounds, handle error appropriately
            std::cerr << "Row index out of bounds!" << std::endl;
            return;
        }
        std::vector<std::pair<uint32_t, float>> vec1(nnz1), vec2(nnz2);

        uint32_t start = this->indptr[u];
        uint32_t end = this->indptr[u+1];
        int32_t nnz1 = end - start;
        // Fill the vectors with the input data
        for (int32_t i = start; i < end; ++i) {
            vec1[i] = {this->indices_[i], this->data_[i]};
        }

        int32_t nnz2 = computer.nnz_; 
        for (int32_t i = 0; i < computer.nnz_; ++i) {
            vec2[i] = {computer.ids_[i], computer.vals_[i]};
        }

        // Sort both vectors by id
        std::sort(vec1.begin(), vec1.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::sort(vec2.begin(), vec2.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Use double pointer technique to calculate inner product
        float result = 0.0f;
        int32_t i = 0, j = 0;
        while (i < nnz1 && j < nnz2) {
            if (vec1[i].first == vec2[j].first) {
                result += vec1[i].second * vec2[j].second;
                ++i;
                ++j;
            } else if (vec1[i].first < vec2[j].first) {
                ++i;
            } else {
                ++j;
            }
        }

        *dists = results;
    }

    inline float
    SparseQuantizer::ComputeDist(SparseComputer& computer, uint32_t u) const {

        if (u >= num_) {
            // Out of bounds, handle error appropriately
            std::cerr << "Row index out of bounds!" << std::endl;
            return;
        }
        std::vector<std::pair<uint32_t, float>> vec1(nnz1), vec2(nnz2);

        uint32_t start = this->indptr[u];
        uint32_t end = this->indptr[u+1];
        int32_t nnz1 = end - start;
        // Fill the vectors with the input data
        for (int32_t i = start; i < end; ++i) {
            vec1[i] = {this->indices_[i], this->data_[i]};
        }

        int32_t nnz2 = computer.nnz_; 
        for (int32_t i = 0; i < computer.nnz_; ++i) {
            vec2[i] = {computer.ids_[i], computer.vals_[i]};
        }

        // Sort both vectors by id
        std::sort(vec1.begin(), vec1.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::sort(vec2.begin(), vec2.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        // Use double pointer technique to calculate inner product
        float result = 0.0f;
        int32_t i = 0, j = 0;
        while (i < nnz1 && j < nnz2) {
            if (vec1[i].first == vec2[j].first) {
                result += vec1[i].second * vec2[j].second;
                ++i;
                ++j;
            } else if (vec1[i].first < vec2[j].first) {
                ++i;
            } else {
                ++j;
            }
        }

        return result;
    }

    inline void
    SparseQuantizer::ProcessQuery(int32_t nnz, const uint32_t* ids, const float* vals, SparseComputer& computer) const {
        computer.nnz_ = nnz;
    try {
        computer.ids_ = reinterpret_cast<uint32_t*>(this->allocator_->Allocate(nnz * sizeof(uint32_t)));
        computer.vals_ = reinterpret_cast<float*>(this->allocator_->Allocate(nnz * sizeof(float)));
    } catch (const std::bad_alloc& e) {
        computer.ids_ = nullptr;
        computer.vals_ = nullptr;
        logger::error("bad alloc when init computer buf");
        throw std::bad_alloc();
    }
        memcpy(computer.ids_, ids, nnz * sizeof(uint32_t));
        memcpy(computer.vals_, vals, nnz * sizeof(float));
    }

    inline void
    SparseQuantizer::ReleaseComputer(SparseComputer& computer) const {
        this->allocator_->Deallocate(computer.ids_);
        this->allocator_->Deallocate(computer.vals_);
    }

}  // namespace vsag
