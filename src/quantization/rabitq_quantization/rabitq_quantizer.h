
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

#include <limits>
#include <string>

#include "impl/transform/pca_transformer.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "rabitq_quantizer_parameter.h"

namespace vsag {

/** Implement of RaBitQ Quantization, Integrate MRQ (Minimized Residual Quantization) and Extend-RaBitQ
 *
 *  RaBitQ: Supports bit-level quantization
 *  MRQ: Support use residual part of PCA to increase precision
 *  Extend-RaBitQ: Supports multi-bit quantization
 *
 *  Reference:
 *  [1] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages. https://doi.org/10.1145/3654970
 *  [2] Mingyu Yang, Wentao Li, Wei Wang. Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space
 *  [3] Jianyang Gao, Yutong Gou, Yuexuan Xu, Yongyi Yang, Cheng Long, and Raymond Chi-Wing Wong. 2025. Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 3, 3, Article 202 (June 2025), 26 pages. https://doi.org/10.1145/3725413
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class RaBitQuantizer : public Quantizer<RaBitQuantizer<metric>> {
public:
    using norm_type = float;
    using error_type = float;
    using sum_type = float;

    explicit RaBitQuantizer(
        int dim,
        uint64_t pca_dim,
        uint64_t num_bits_per_dim_query,
        uint64_t num_bits_per_dim_base,
        bool use_fht,
        bool use_mrq,
        Allocator* allocator,
        std::string rabitq_version = RaBitQuantizerParameter::DEFAULT_RABITQ_VERSION,
        float rabitq_error_rate = RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE,
        uint64_t num_bits_per_dim_filter =
            RaBitQuantizerParameter::DEFAULT_RABITQ_BITS_PER_DIM_FILTER);

    explicit RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                            const IndexCommonParam& common_param);

    explicit RaBitQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    float
    ComputeQueryBaseImpl(const uint8_t* query_codes, const uint8_t* base_codes) const;

    void
    ProcessQueryImpl(const float* query, Computer<RaBitQuantizer>& computer) const;

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
    // query sq4 related
    void
    ReOrderSQ4(const uint8_t* input, uint8_t* output) const;

    void
    RecoverOrderSQ4(const uint8_t* output, uint8_t* input) const;

    void
    EncodeSQ(const float* normed_data,
             uint8_t* quantized_data,
             float& upper_bound,
             float& lower_bound,
             float& delta,
             sum_type& query_sum) const;
    void
    DecodeSQ(const uint8_t* codes,
             float* data,
             const float upper_bound,
             const float lower_bound) const;

    void
    ReOrderSQ(const uint8_t* quantized_data, uint8_t* reorder_data) const;

    void
    RecoverOrderSQ(const uint8_t* output, uint8_t* input) const;

public:
    // base multi-bit related
    void
    EncodeExtendRaBitQ(const float* o_prime, uint8_t* code, float& y_norm) const;

    void
    PackIntoPlanes(const uint8_t* src, uint8_t* dst) const;

    float
    RaBitQFloatSQIPByPlanes(const float* query, const uint8_t* planes) const;

    float
    RaBitQFloatSQIPBySplitCode(const float* query,
                               const uint8_t* one_bit_code,
                               const uint8_t* supplement_code) const;

    float
    RaBitQFloatSQIPBySplitCode(const float* query,
                               const uint8_t* filter_code,
                               const uint8_t* supplement_code,
                               uint32_t filter_bits,
                               uint32_t supplement_bits) const;

    [[nodiscard]] uint64_t
    StoredPlaneIndex(uint32_t logical_bit) const;

    [[nodiscard]] const uint8_t*
    GetStoredPlane(const uint8_t* planes, uint32_t logical_bit, uint64_t plane_bytes) const;

    [[nodiscard]] uint8_t*
    GetStoredPlane(uint8_t* planes, uint32_t logical_bit, uint64_t plane_bytes) const;

    [[nodiscard]] bool
    SupportSplitCodeStorage() const;

    [[nodiscard]] uint64_t
    GetOneBitCodeSize() const;

    [[nodiscard]] uint64_t
    GetSupplementCodeSize() const;

    void
    SplitCode(const uint8_t* full_code, uint8_t* one_bit_code, uint8_t* supplement_code) const;

    void
    MergeSplitCode(const uint8_t* one_bit_code,
                   const uint8_t* supplement_code,
                   uint8_t* full_code) const;

    bool
    ComputeDistWithOneBitLowerBound(
        Computer<RaBitQuantizer>& computer,
        const uint8_t* one_bit_code,
        float* dists,
        float* lower_bound,
        float runtime_rabitq_error_rate = std::numeric_limits<float>::quiet_NaN()) const;

    void
    ComputeDistsWithOneBitLowerBoundBatch4(
        Computer<RaBitQuantizer>& computer,
        const uint8_t* one_bit_code1,
        const uint8_t* one_bit_code2,
        const uint8_t* one_bit_code3,
        const uint8_t* one_bit_code4,
        float& dist1,
        float& dist2,
        float& dist3,
        float& dist4,
        float* lower_bound1,
        float* lower_bound2,
        float* lower_bound3,
        float* lower_bound4,
        bool& computed1,
        bool& computed2,
        bool& computed3,
        bool& computed4,
        float runtime_rabitq_error_rate = std::numeric_limits<float>::quiet_NaN()) const;

    bool
    ComputeDistWithSplitCode(Computer<RaBitQuantizer>& computer,
                             const uint8_t* one_bit_code,
                             const uint8_t* supplement_code,
                             float* dists) const;

    // Computes the full x+y split distance while reusing the filter-stage distance.
    // `filter_dist` is the x-bit distance already produced by
    // ComputeDistWithOneBitLowerBound(); it is not a lower bound and not the final
    // x+y distance. Passing it here lets the reorder path scan only the y-bit
    // supplement planes instead of rescanning the x-bit filter planes.
    bool
    ComputeDistWithSplitCodeAndFilterDist(Computer<RaBitQuantizer>& computer,
                                          const uint8_t* one_bit_code,
                                          const uint8_t* supplement_code,
                                          float filter_dist,
                                          float* dists) const;

    [[nodiscard]] uint64_t
    OneBitRecordNormOffset() const;

    [[nodiscard]] uint64_t
    OneBitRecordLowBoundErrorOffset() const;

    [[nodiscard]] uint64_t
    OneBitRecordOneBitErrorOffset() const;

    [[nodiscard]] uint64_t
    OneBitRecordMrqNormOffset() const;

    [[nodiscard]] uint64_t
    OneBitRecordRawNormOffset() const;

    [[nodiscard]] uint64_t
    OneBitRecordSize() const;

    [[nodiscard]] uint64_t
    PlaneBytes() const;

    [[nodiscard]] uint64_t
    CodePlanesSize() const;

    [[nodiscard]] uint64_t
    CodeMetaOffset() const;

    [[nodiscard]] uint64_t
    SupplementPlanesSize() const;

    [[nodiscard]] uint32_t
    FilterBits() const;

    [[nodiscard]] uint32_t
    ReorderBits() const;

    [[nodiscard]] uint64_t
    FilterPlanesSize() const;

    [[nodiscard]] bool
    HasMultiBitFilter() const;

    [[nodiscard]] uint64_t
    OneBitRecordNormCodeOffset() const;

    [[nodiscard]] uint64_t
    SupplementMetaOffset() const;

    [[nodiscard]] uint64_t
    AlignCodeField(uint64_t size) const;

private:
    [[nodiscard]] bool
    HasFilterQueryLookupTable() const;

    [[nodiscard]] uint64_t
    FilterQueryLookupTableSize() const;

    [[nodiscard]] norm_type
    ComputeScalarCodeNorm(const uint8_t* scalar_codes,
                          uint32_t code_bits,
                          uint32_t dropped_bits) const;

    [[nodiscard]] norm_type
    ComputeFilterCodeNorm(const uint8_t* filter_code, uint32_t filter_bits) const;

    // bit related
    uint64_t num_bits_per_dim_query_{32};
    uint32_t num_bits_per_dim_base_{1};
    uint32_t num_bits_per_dim_filter_{RaBitQuantizerParameter::DEFAULT_RABITQ_BITS_PER_DIM_FILTER};

    // compute related
    float inv_sqrt_d_{0.0F};
    std::string rabitq_version_{RaBitQuantizerParameter::DEFAULT_RABITQ_VERSION};
    float rabitq_error_rate_{RaBitQuantizerParameter::DEFAULT_RABITQ_ERROR_RATE};

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
     * query layout: sq-code(required) + lower_bound(sq4) + delta(sq4) + sum(sq4 or extend_rabitq) + norm(required) + mrq_norm(required)
     */
    uint64_t aligned_dim_{0};
    uint64_t query_offset_lb_{0};
    uint64_t query_offset_delta_{0};
    uint64_t query_offset_sum_{0};
    uint64_t query_offset_norm_{0};
    uint64_t query_offset_mrq_norm_{0};
    uint64_t query_offset_raw_norm_{0};
    uint64_t query_offset_filter_lut_{0};

    /***
     * code layout: bq-code(required) + norm(required) + error(required) + offset_norm_code(extend_rabitq) + sum(sq4) + mrq_norm(required)
     */
    uint64_t offset_code_{0};
    uint64_t offset_norm_{0};
    uint64_t offset_error_{0};
    uint64_t offset_norm_code_{0};
    uint64_t offset_sum_{0};
    uint64_t offset_mrq_norm_{0};
    uint64_t offset_raw_norm_{0};
    uint64_t offset_low_bound_error_{0};
    uint64_t offset_one_bit_error_{0};
};

}  // namespace vsag
