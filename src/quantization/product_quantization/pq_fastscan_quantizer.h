
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
#include "pq_fastscan_quantizer_parameter.h"
#include "quantization/quantizer.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class PQFastScanQuantizer : public Quantizer<PQFastScanQuantizer<metric>> {
public:
    explicit PQFastScanQuantizer(int dim, int64_t pq_dim, Allocator* allocator);

    PQFastScanQuantizer(const PQFastScanQuantizerParamPtr& param,
                        const IndexCommonParam& common_param);

    PQFastScanQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~PQFastScanQuantizer() override = default;

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
    ProcessQueryImpl(const DataType* query, Computer<PQFastScanQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<PQFastScanQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<PQFastScanQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    void
    ReleaseComputerImpl(Computer<PQFastScanQuantizer<metric>>& computer) const;

    [[nodiscard]] inline std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_PQFS;
    }

    void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const override;

    void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const override;

private:
    [[nodiscard]] inline const float*
    get_codebook_data(int64_t subspace_idx, int64_t centroid_num) const {
        return this->codebooks_.data() + subspace_idx * subspace_dim_ * CENTROIDS_PER_SUBSPACE +
               centroid_num * subspace_dim_;
    }

public:
    constexpr static int64_t PQ_BITS = 4L;
    constexpr static int64_t CENTROIDS_PER_SUBSPACE = 16L;
    constexpr static int64_t BLOCK_SIZE_PACKAGE = 32L;

public:
    int64_t pq_dim_{1};
    int64_t subspace_dim_{1};  // equal to dim/pq_dim_;

    Vector<float> codebooks_;
};

}  // namespace vsag
