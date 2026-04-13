
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

/**
 * @file rabitq_quantizer.h
 * @brief RaBitQ quantizer implementation with MRQ and Extend-RaBitQ support.
 */

#pragma once

#include "impl/transform/pca_transformer.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "rabitq_quantizer_parameter.h"

namespace vsag {

/**
 * @brief RaBitQ Quantizer with bit-level quantization support.
 *
 * Integrates MRQ (Minimized Residual Quantization) and Extend-RaBitQ:
 * - RaBitQ: Supports bit-level quantization
 * - MRQ: Support use residual part of PCA to increase precision
 * - Extend-RaBitQ: Supports multi-bit quantization
 *
 * Reference:
 * [1] Jianyang Gao and Cheng Long. 2024. RaBitQ: Quantizing High-Dimensional Vectors
 *     with a Theoretical Error Bound for Approximate Nearest Neighbor Search.
 *     Proc. ACM Manag. Data 2, 3, Article 167 (June 2024), 27 pages.
 * [2] Mingyu Yang, Wentao Li, Wei Wang. Fast High-dimensional Approximate Nearest
 *     Neighbor Search with Efficient Index Time and Space.
 * [3] Jianyang Gao et al. 2025. Practical and Asymptotically Optimal Quantization of
 *     High-Dimensional Vectors in Euclidean Space. Proc. ACM Manag. Data 3, 3, Article 202.
 *
 * @tparam metric Distance metric type.
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class RaBitQuantizer : public Quantizer<RaBitQuantizer<metric>> {
public:
    using norm_type = float;   ///< Type for vector norm.
    using error_type = float;  ///< Type for quantization error.
    using sum_type = float;    ///< Type for sum values.

    /**
     * @brief Constructs a RaBitQ quantizer with detailed parameters.
     * @param dim Vector dimension.
     * @param pca_dim PCA dimension for MRQ.
     * @param num_bits_per_dim_query Bits per dimension for query quantization.
     * @param num_bits_per_dim_base Bits per dimension for base quantization.
     * @param use_fht Whether to use Fast Hadamard Transform.
     * @param use_mrq Whether to use MRQ (Minimized Residual Quantization).
     * @param allocator Memory allocator.
     */
    explicit RaBitQuantizer(int dim,
                            uint64_t pca_dim,
                            uint64_t num_bits_per_dim_query,
                            uint64_t num_bits_per_dim_base,
                            bool use_fht,
                            bool use_mrq,
                            Allocator* allocator);

    /**
     * @brief Constructs a RaBitQ quantizer from parameter pointer.
     * @param param RaBitQ quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit RaBitQuantizer(const RaBitQuantizerParamPtr& param,
                            const IndexCommonParam& common_param);

    /**
     * @brief Constructs a RaBitQ quantizer from base parameter pointer.
     * @param param Quantizer parameter pointer.
     * @param common_param Common index parameters.
     */
    explicit RaBitQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Trains the quantizer with given data.
     * @param data Training data.
     * @param count Number of vectors.
     * @return True if training succeeded.
     */
    bool
    TrainImpl(const DataType* data, uint64_t count);

    /**
     * @brief Encodes a single vector.
     * @param data Input vector data.
     * @param codes Output code buffer.
     * @return True if encoding succeeded.
     */
    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    /**
     * @brief Encodes a batch of vectors.
     * @param data Input vector data.
     * @param codes Output code buffer.
     * @param count Number of vectors.
     * @return True if encoding succeeded.
     */
    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    /**
     * @brief Decodes a single code to vector.
     * @param codes Input code buffer.
     * @param data Output vector data.
     * @return True if decoding succeeded.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    /**
     * @brief Decodes a batch of codes to vectors.
     * @param codes Input code buffer.
     * @param data Output vector data.
     * @param count Number of vectors.
     * @return True if decoding succeeded.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    /**
     * @brief Computes distance between two base codes.
     * @param codes1 First code buffer.
     * @param codes2 Second code buffer.
     * @return Computed distance.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    /**
     * @brief Computes distance between query code and base code.
     * @param query_codes Query code buffer.
     * @param base_codes Base code buffer.
     * @return Computed distance.
     */
    float
    ComputeQueryBaseImpl(const uint8_t* query_codes, const uint8_t* base_codes) const;

    /**
     * @brief Processes a query vector for distance computation.
     * @param query Query vector data.
     * @param computer Computer object to store query codes.
     */
    void
    ProcessQueryImpl(const DataType* query, Computer<RaBitQuantizer>& computer) const;

    /**
     * @brief Computes distance between query code and base code.
     * @param computer Computer object containing query codes.
     * @param codes Base code buffer.
     * @param dists Output distance array.
     */
    void
    ComputeDistImpl(Computer<RaBitQuantizer>& computer, const uint8_t* codes, float* dists) const;

    /**
     * @brief Computes distances for a batch of codes.
     * @param computer Computer object containing query codes.
     * @param count Number of codes.
     * @param codes Base code buffer.
     * @param dists Output distance array.
     */
    void
    ScanBatchDistImpl(Computer<RaBitQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    /**
     * @brief Releases resources held by computer.
     * @param computer Computer object to release.
     */
    void
    ReleaseComputerImpl(Computer<RaBitQuantizer<metric>>& computer) const;

    /**
     * @brief Serializes the quantizer to stream.
     * @param writer Stream writer.
     */
    void
    SerializeImpl(StreamWriter& writer);

    /**
     * @brief Deserializes the quantizer from stream.
     * @param reader Stream reader.
     */
    void
    DeserializeImpl(StreamReader& reader);

    /**
     * @brief Gets the quantizer name.
     * @return Quantizer type name string.
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_RABITQ;
    }

public:
    /**
     * @brief Reorders SQ4 codes for efficient computation.
     * @param input Input SQ4 codes.
     * @param output Output reordered codes.
     */
    void
    ReOrderSQ4(const uint8_t* input, uint8_t* output) const;

    /**
     * @brief Recovers original SQ4 codes from reordered format.
     * @param output Reordered codes.
     * @param input Output original format.
     */
    void
    RecoverOrderSQ4(const uint8_t* output, uint8_t* input) const;

    /**
     * @brief Encodes data using scalar quantization.
     * @param normed_data Normalized input data.
     * @param quantized_data Output quantized codes.
     * @param upper_bound Upper bound for quantization range.
     * @param lower_bound Lower bound for quantization range.
     * @param delta Quantization step size.
     * @param query_sum Sum of quantized values.
     */
    void
    EncodeSQ(const DataType* normed_data,
             uint8_t* quantized_data,
             float& upper_bound,
             float& lower_bound,
             float& delta,
             sum_type& query_sum) const;

    /**
     * @brief Decodes scalar quantized codes.
     * @param codes Input quantized codes.
     * @param data Output decoded data.
     * @param upper_bound Upper bound for quantization range.
     * @param lower_bound Lower bound for quantization range.
     */
    void
    DecodeSQ(const uint8_t* codes,
             DataType* data,
             const float upper_bound,
             const float lower_bound) const;

    /**
     * @brief Reorders scalar quantized codes.
     * @param quantized_data Input quantized codes.
     * @param reorder_data Output reordered codes.
     */
    void
    ReOrderSQ(const uint8_t* quantized_data, uint8_t* reorder_data) const;

    /**
     * @brief Recovers original SQ codes from reordered format.
     * @param output Reordered codes.
     * @param input Output original format.
     */
    void
    RecoverOrderSQ(const uint8_t* output, uint8_t* input) const;

public:
    /**
     * @brief Encodes using extended RaBitQ format.
     * @param o_prime Transformed input vector.
     * @param code Output code buffer.
     * @param y_norm Output norm of transformed vector.
     */
    void
    EncodeExtendRaBitQ(const float* o_prime, uint8_t* code, float& y_norm) const;

    /**
     * @brief Packs bits into planes for multi-bit quantization.
     * @param src Source bit data.
     * @param dst Destination plane data.
     */
    void
    PackIntoPlanes(const uint8_t* src, uint8_t* dst) const;

    /**
     * @brief Computes float SQIP distance using plane format.
     * @param query Query vector.
     * @param planes Packed plane data.
     * @return Computed distance.
     */
    float
    RaBitQFloatSQIPByPlanes(const float* query, const uint8_t* planes) const;

private:
    uint64_t num_bits_per_dim_query_{32};  ///< Bits per dimension for query.
    uint32_t num_bits_per_dim_base_{1};    ///< Bits per dimension for base.

    float inv_sqrt_d_{0.0F};  ///< Inverse square root of dimension.

    bool use_fht_{false};                     ///< Whether to use Fast Hadamard Transform.
    std::shared_ptr<VectorTransformer> rom_;  ///< Random orthogonal matrix transformer.
    std::vector<float> centroid_;             ///< Centroid vector for IVF/Graph usage.

    std::shared_ptr<PCATransformer> pca_;  ///< PCA transformer for MRQ.
    std::uint64_t original_dim_{0};        ///< Original dimension before PCA.
    std::uint64_t pca_dim_{0};             ///< PCA dimension.
    bool use_mrq_{false};                  ///< Whether to use MRQ.

    uint64_t aligned_dim_{0};            ///< Aligned dimension.
    uint64_t query_offset_lb_{0};        ///< Query lower bound offset.
    uint64_t query_offset_delta_{0};     ///< Query delta offset.
    uint64_t query_offset_sum_{0};       ///< Query sum offset.
    uint64_t query_offset_norm_{0};      ///< Query norm offset.
    uint64_t query_offset_mrq_norm_{0};  ///< Query MRQ norm offset.
    uint64_t query_offset_raw_norm_{0};  ///< Query raw norm offset.

    uint64_t offset_code_{0};       ///< Base code offset.
    uint64_t offset_norm_{0};       ///< Base norm offset.
    uint64_t offset_error_{0};      ///< Base error offset.
    uint64_t offset_norm_code_{0};  ///< Base norm code offset.
    uint64_t offset_sum_{0};        ///< Base sum offset.
    uint64_t offset_mrq_norm_{0};   ///< Base MRQ norm offset.
    uint64_t offset_raw_norm_{0};   ///< Base raw norm offset.
};

}  // namespace vsag
