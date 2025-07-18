
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

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "product_quantizer_parameter.h"
#include "quantization/quantizer.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class ProductQuantizer : public Quantizer<ProductQuantizer<metric>> {
public:
    explicit ProductQuantizer(int dim, int64_t pq_dim, Allocator* allocator);

    ProductQuantizer(const ProductQuantizerParamPtr& param, const IndexCommonParam& common_param);

    ProductQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~ProductQuantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    void
    ProcessQueryImpl(const DataType* query, Computer<ProductQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ScanBatchDistImpl(Computer<ProductQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    void
    ReleaseComputerImpl(Computer<ProductQuantizer<metric>>& computer) const;

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_PQ;
    }

private:
    [[nodiscard]] const float*
    get_codebook_data(int64_t subspace_idx, int64_t centroid_num) const {
        return this->codebooks_.data() + subspace_idx * subspace_dim_ * CENTROIDS_PER_SUBSPACE +
               centroid_num * subspace_dim_;
    }

    void
    transpose_codebooks();

public:
    constexpr static int64_t PQ_BITS = 8L;
    constexpr static int64_t CENTROIDS_PER_SUBSPACE = 256L;

public:
    int64_t pq_dim_{1};
    int64_t subspace_dim_{1};  // equal to dim/pq_dim_;

    Vector<float> codebooks_;

    Vector<float> reverse_codebooks_;
};

}  // namespace vsag
