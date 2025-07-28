
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

#include "impl/transform/pca_transformer.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "rabitq_quantizer_parameter.h"

namespace vsag {

/** Implement of RaBitQ Quantization, Integrate MRQ (Minimized Residual Quantization)
 *
 *  RaBitQ: Supports bit-level quantization
 *  MRQ: Support use residual part of PCA to increase precision
 *
 *  Reference:
 *  [1] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970
 *  [2] Mingyu Yang, Wentao Li, Wei Wang. Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class RaBitQuantizer : public Quantizer<RaBitQuantizer<metric>> {
public:
    using norm_type = float;
    using error_type = float;
    using sum_type = float;

    explicit RaBitQuantizer(int dim,
                            uint64_t pca_dim,
                            uint64_t num_bits_per_dim_query,
                            bool use_fht,
                            bool use_mrq,
                            Allocator* allocator);

    explicit RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                            const IndexCommonParam& common_param);

    explicit RaBitQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    float
    ComputeQueryBaseImpl(const uint8_t* query_codes, const uint8_t* base_codes) const;

    void
    ProcessQueryImpl(const DataType* query, Computer<RaBitQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<RaBitQuantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ScanBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<RaBitQuantizer<metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_RABITQ;
    }

public:
    void
    ReOrderSQ4(const uint8_t* input, uint8_t* output) const;

    void
    RecoverOrderSQ4(const uint8_t* output, uint8_t* input) const;

    void
    EncodeSQ(const DataType* normed_data,
             uint8_t* quantized_data,
             float& upper_bound,
             float& lower_bound,
             float& delta,
             sum_type& query_sum) const;
    void
    DecodeSQ(const uint8_t* codes,
             DataType* data,
             const float upper_bound,
             const float lower_bound) const;

    void
    ReOrderSQ(const uint8_t* quantized_data, uint8_t* reorder_data) const;

    void
    RecoverOrderSQ(const uint8_t* output, uint8_t* input) const;

private:
    // compute related
    float inv_sqrt_d_{0.0F};

    // random projection related
    bool use_fht_{false};
    std::shared_ptr<VectorTransformer> rom_;
    std::vector<float> centroid_;  // TODO(ZXY): use centroids (e.g., IVF or Graph) outside

    // pca related
    std::shared_ptr<PCATransformer> pca_;
    std::uint64_t original_dim_{0};
    std::uint64_t pca_dim_{0};
    bool use_mrq_{false};

    /***
     * query layout: sq-code(required) + lower_bound(sq4) + delta(sq4) + sum(sq4) + norm(required) + mrq_norm(required)
     */
    uint64_t aligned_dim_{0};
    uint64_t num_bits_per_dim_query_{32};
    uint64_t query_offset_lb_{0};
    uint64_t query_offset_delta_{0};
    uint64_t query_offset_sum_{0};
    uint64_t query_offset_norm_{0};
    uint64_t query_offset_mrq_norm_{0};
    uint64_t query_offset_raw_norm_{0};

    /***
     * code layout: bq-code(required) + norm(required) + error(required) + sum(sq4) + mrq_norm(required)
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_error_{0};
    uint64_t offset_sum_{0};
    uint64_t offset_mrq_norm_{0};
    uint64_t offset_raw_norm_{0};
};

}  // namespace vsag
