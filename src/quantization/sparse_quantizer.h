
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

#include "../logger.h"
#include "inner_string_params.h"
#include "metric_type.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "vsag/dataset.h"

namespace vsag {
using SDataType = SparseVectors;

class SparseComputer;
/**
 * @class SparseQuantizer
 * @brief This class is used for distance computation and encoding/decoding of data.
 */
class SparseQuantizer {
public:
    explicit SparseQuantizer(Allocator* allocator) : allocator_(allocator){}

    ~SparseQuantizer() =default;

    bool
    Train(const SDataType* data, uint64_t count){
        return this->TrainImpl(data, count);
    }

    bool
    ReTrain(const SDataType* data, uint64_t count) {
        this->is_trained_ = false;
        return this->TrainImpl(data, count);
    }

    bool
    EncodeOne(const SDataType* data, uint8_t* codes) const;

    bool
    EncodeBatch(const SDataType* data, uint8_t* codes, uint64_t count) const;

    bool
    DecodeOne(const uint8_t* codes, SDataType* data) const;

    bool
    DecodeBatch(const uint8_t* codes, SDataType* data, uint64_t count);

    inline float
    Compute(const uint8_t* codes1, const uint8_t* codes2)const {
        return this->ComputeDistImpl(codes1, codes2);
    }

    inline void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->metric_);
        StreamWriter::WriteObj(writer, this->is_trained_);
    }

    inline void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->metric_);
        StreamReader::ReadObj(reader, this->is_trained_);
    }

    void
    ComputeDist(SparseComputer& computer, const uint8_t* codes, float* dists) const;

    float
    ComputeDist(SparseComputer& computer, const uint8_t* codes) const;

    std::shared_ptr<SparseComputer>
    FactoryComputer();

    void
    ReleaseComputer(SparseComputer& computer) const;

    void
    ProcessQuery(const SDataType* query, SparseComputer& computer) const;

    [[nodiscard]] std::string
    Name() const {
        return QUANTIZATION_TYPE_VALUE_SPARSE;
    }

    [[nodiscard]] MetricType
    Metric() const {
        return this->metric_;
    }

    inline uint64_t
    GetCodeSize(int dim = -1, int count= 1) const {//dim is the number off non zero entries, count is the number of sparse vectors
        if (dim == -1)
            return this->code_size_;
        else {
            return count * sizeof(uint32_t) + dim * (sizeof(uint32_t) + sizeof(float));
            }
    }

private:
    bool
    TrainImpl(const SDataType* data, uint64_t count);

    bool
    EncodeOneImpl(uint32_t nnz, const uint32_t* ids, const float* vals, uint8_t* codes) const;

    bool
    DecodeOneImpl(uint32_t nnz, uint32_t* ids, float* vals, const uint8_t* codes) const;

    float
    ComputeDistImpl(const uint8_t* codes1, const uint8_t* codes2) const;

private:
    uint64_t code_size_{0};
    bool is_trained_{false};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    Allocator* const allocator_{nullptr};
};

}  // namespace vsag
